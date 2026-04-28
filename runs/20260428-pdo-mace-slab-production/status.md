# PdO MACE Slab SSW CUDA Status

Date: 2026-04-28

## Runtime

- Environment: `mace_les`
- Model: `/root/.cache/mace/mace-omat-0-small.model`
- Calculator: MACE via ASE wrapper
- Device: CUDA, `NVIDIA GeForce RTX 3060`
- Precision: `float32`
- cuEq: disabled

## Input Handling

- Input file: `PdO.xyz`
- Input atom count: 115 (`O20Pd95`)
- Input cell: `13.75323 x 13.75323 x 23.89 A`
- Input PBC: `(True, True, True)`
- Run PBC: `(True, True, False)` because this is a slab with vacuum, not a bulk test.
- Fixed atoms: bottom 35% by z, 40 atoms fixed.

## Code Fix During Run

The first CUDA run completed but showed periodic archive minima with coordinates outside the wrapped cell after local relaxation. The cause was that `Relaxer` returned `state.with_active_positions(...)` directly, bypassing the `CartesianCoordinates.displace()` wrapping path.

Fix applied:

- `Relaxer.relax()` wraps final periodic coordinates before final energy/gradient evaluation.
- Added regression test `test_relaxer_wraps_final_periodic_coordinates`.
- Full tests after the fix: `91 passed in 4.07s`.

## Final Run

Output directory:

`runs/20260428-pdo-mace-slab-production/seed0_trials8_cuda_wrapfix`

Command:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
MPLCONFIGDIR=/tmp/mplconfig python runs/20260428-pdo-mace-slab-production/run_pdo_mace_ssw.py \
  --input PdO.xyz \
  --model /root/.cache/mace/mace-omat-0-small.model \
  --output-dir runs/20260428-pdo-mace-slab-production/seed0_trials8_cuda_wrapfix \
  --device cuda \
  --dtype float32 \
  --seed 0 \
  --trials 8 \
  --steps-per-walk 4 \
  --oracle-candidates 6 \
  --proposal-relax-steps 80 \
  --proposal-fmax 0.05 \
  --quench-fmax 0.03 \
  --target-uphill-energy 0.25 \
  --fix-bottom-fraction 0.35
```

## Results

- Best energy: `-568.3961181640625 eV`
- Unique minima: `6`
- Local relaxations: `9`
- Force/energy evaluations: `1027`
- Duplicate rate: `0.3333333333333333`
- Walk displacement clips: `0`
- Fragment rejections: `0`
- Direction choices: `32`
- Bond candidates requested/generated/valid: `64/64/64`
- Direction selected: soft `20`, random `8`, bond `4`
- True quench unconverged: `3/9`, max gradient `0.03923653261362602 eV/A`
- Proposal relax unconverged: `15/32`, max gradient `0.7056868999513327 eV/A`
- All final archive minima preserve `cell` and `pbc=(True, True, False)`.
- All final archive minima have periodic x/y coordinates wrapped inside the cell.

## Outputs

- Summary JSON: `seed0_trials8_cuda_wrapfix/ssw_summary.json`
- Energy trace JSON: `seed0_trials8_cuda_wrapfix/energy_trace.json`
- Energy trace plot: `seed0_trials8_cuda_wrapfix/energy_trace.png`
- Combined archive minima: `seed0_trials8_cuda_wrapfix/archive_minima.xyz`
- Best minimum: `seed0_trials8_cuda_wrapfix/best_minimum.xyz`
- Per-minimum XYZ files: `seed0_trials8_cuda_wrapfix/minima_xyz/`

## Interpretation

The fixed-cell periodic path now works for this PdO slab smoke-production run in the narrow sense: CUDA MACE evaluation succeeds, SSW completes, archive minima keep periodic metadata, MIC/wrapping diagnostics pass, and the code discovers multiple minima without walk clipping or slab fragmentation.

This is not yet a full production claim for arbitrary slabs. The remaining bottleneck is convergence quality: proposal relaxation still has `15/32` unconverged events, and one proposal relaxation reached `0.706 eV/A`. The next production step is a sweep over proposal relaxation steps, proposal trust radius, and `target_uphill_energy` on this same PdO case.

## Review-Driven Diagnosis And Larger Run

The smoke run produced six minima, but their energy span was only `0.09356689453125 eV` and the median MIC RMSD to the best minimum was only `0.02233831335032546 A`. That is not convincing global exploration; it is mostly same-basin local variation.

Two review items are directly relevant:

- Trust update used only the quadratic term `0.5 sigma^2 rho_true`. In a biased walk, the current point is generally not a true-PES stationary point, so the linear term `sigma * grad(U) dot d` is missing.
- The actual displacement scale used biased-PES curvature. That can make `sigma` too small or too large when accumulated Gaussian biases distort the local curvature.

Fix applied:

- `TrustRegionBiasController.predicted_delta()` now includes `g_parallel`.
- `_walk_candidate_from_seed()` computes `sigma` from true-PES curvature for the selected direction.
- Gaussian bias weight still uses inner/biased curvature, preserving the role of bias as local destabilization on the modified PES.
- Added unit tests for the trust linear term.
- Full tests after the fix: `93 passed in 6.11s`.

Larger CUDA run:

`runs/20260428-pdo-mace-slab-production/seed0_trials40_cuda_truecurv`

Settings:

- trials: `40`
- steps per walk: `8`
- oracle candidates: `8`
- proposal relax steps: `120`
- target uphill energy: `0.8 eV`
- max step scale: `1.2 A`
- proposal trust radius: `1.5 A`
- walk trust radius: `4.0 A`
- dedup RMSD tolerance: `0.25 A`

Results:

- Best energy improved to `-573.0081176757812 eV`.
- Unique minima: `29`
- Energy span: `4.61328125 eV`
- Median MIC RMSD to best minimum: `0.5535368711521987 A`
- Max MIC RMSD to best minimum: `0.8399441991696203 A`
- Final archive minima all preserve cell and `pbc=(True, True, False)`.
- Periodic x/y coordinates are wrapped inside the cell for all archive minima.

Remaining blocker:

- Proposal relaxation is not production-grade yet: `proposal_relax_unconverged=243/318`, max proposal gradient `1.3142722611198498 eV/A`.
- Trust model error is still high: `trust_model_error_mean=11.081653898024518`.

Conclusion: the slab path is no longer failing because of PBC/MIC metadata or pure tiny-budget repetition. It can reach substantially different basins when the step scale is allowed to grow. The current production blocker is now the inner proposal relaxation/trust policy, especially for MACE slab force landscapes.
