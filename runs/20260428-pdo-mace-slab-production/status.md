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
