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

## Relaxation Diagnostics Before Sweeping

The next review split the remaining issue into three possible causes:

- coordinate box bound truncation,
- insufficient L-BFGS-B iterations,
- or proposal `fmax` being too strict for the biased PES.

Code changes made for diagnosis, not algorithm retuning:

- `RelaxResult` now records `active_bound_fraction`, `displacement_rms`, and `displacement_max`.
- `SurfaceWalker` reports these diagnostics separately for true quench and proposal relax.
- `SurfaceWalker` reports `bias_zero_weight_fraction`, `bias_weight_mean`, and `bias_weight_max`.
- `SSWConfig(proposal_trust_radius=None)` is allowed, so the coordinate box can be disabled in a controlled sweep.
- The PdO runner accepts `--proposal-trust-radius none`.

Stage 0 diagnostic run:

`runs/20260428-pdo-mace-slab-production/stage0_diag_trials12_truecurv`

Same exploration settings as the 40-trial true-curvature run, but shortened to 12 trials:

- Best energy: `-572.5740356445312 eV`
- Unique minima: `8`
- Proposal relax unconverged: `79/96`
- Proposal relax mean iterations: `16.239583333333332` of `120`
- Proposal relax max gradient: `0.9327116646689173 eV/A`
- Proposal active bound fraction mean/max: `0.0/0.0`
- Proposal displacement RMS mean: `0.11333869164470123 A`
- Proposal displacement max: `2.955319038389482 A`
- Bias zero-weight fraction: `0.21875`
- Periodic wrapping diagnostics: no bad final minima.

Interpretation:

- The high proposal-unconverged count is not caused by the coordinate box bound in this run: no finite coordinate bound was active at termination.
- It is also not simply a max-iteration limit: mean iterations are far below the 120-step cap.
- The likely immediate cause is the optimizer stopping before meeting `proposal_fmax`, on a rugged biased PES where L-BFGS-B termination criteria are not equivalent to a force convergence criterion.
- Zero-weight bias steps are non-negligible (`21/96`), so `weight_min` remains a candidate, but it is not yet identified as the primary blocker.

Stage 1 fmax-only probe:

`runs/20260428-pdo-mace-slab-production/stage1_fmax010_trials12_truecurv`

Only changed `proposal_fmax` from `0.05` to `0.10`:

- Best energy: `-569.7075805664062 eV`
- Unique minima: `8`
- Proposal relax unconverged: `58/96`
- Proposal relax mean iterations: `5.5`
- Proposal relax max gradient: `1.1574454173538347 eV/A`
- Proposal active bound fraction mean/max: `0.0/0.0`
- Proposal displacement RMS mean: `0.054404413676921136 A`
- Proposal displacement max: `2.0442236314204076 A`
- True quench unconverged: `5/13`
- Bias zero-weight fraction: `0.09375`
- Periodic wrapping diagnostics: no bad final minima.

Interpretation:

- Relaxing `proposal_fmax` lowers the counted proposal-unconverged rate, but the best energy becomes worse and true-quench unconverged events increase.
- Therefore `proposal_fmax=0.10` is not a safe default from this evidence. It is a useful ablation point, not a production setting.

Current conclusion:

The sweep should be split exactly as the review argues. First keep exploration fixed and sweep relaxation-quality controls:

- `proposal_fmax`: `0.02`, `0.04`, `0.05`, `0.08`
- `proposal_relax_steps`: `80`, `120`, `200`
- `proposal_trust_radius`: `1.5`, `3.0`, `none`

Only after the relax diagnostics are stable should the exploration controls be swept:

- `target_uphill_energy`
- `max_step_scale`
- direction-potential choice for oracle curvature
- optional `weight_min`
