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

## ASE Optimizer Probe

Review hypothesis:

The high proposal-relax unconverged rate may come from SciPy `L-BFGS-B` line-search or termination behavior, because the optimizer often stops with mean iteration counts far below the configured cap while the force is still above `proposal_fmax`.

Implementation:

- `Relaxer` now supports three backends: `scipy-lbfgsb`, `ase-fire`, and `ase-lbfgs`.
- `SSWConfig` exposes `proposal_optimizer` and `quench_optimizer`.
- The proposal relax default is now `ase-fire`; true quench remains `scipy-lbfgsb` unless configured.
- The PdO runner exposes `--proposal-optimizer` and `--quench-optimizer`.

CUDA PdO comparison, all with the same 12-trial true-curvature settings:

| run | proposal optimizer | best energy (eV) | unique minima | force evals | proposal unconverged | max proposal gradient | mean proposal iterations | true quench unconverged | escape rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage0_diag_trials12_truecurv` | `scipy-lbfgsb` | `-572.5740356445312` | `8` | `4332` | `79/96` | `0.9327116646689173` | `16.239583333333332` | `2/13` | `0.5833333333333334` |
| `stage2_asefire_trials12_truecurv` | `ase-fire` | `-572.1031494140625` | `13` | `6035` | `70/96` | `1.1623452432089016` | `36.020833333333336` | `6/13` | `1.0` |
| `stage3_aselbfgs_trials12_truecurv` | `ase-lbfgs` | `-571.4243774414062` | `9` | `5120` | `57/96` | `2.095417583898853` | `26.927083333333332` | `5/13` | `0.6666666666666666` |

Interpretation:

- The line-search hypothesis is partially supported: ASE optimizers reduce the counted proposal-unconverged rate.
- ASE FIRE improves exploration diagnostics in this small test: unique minima rise from `8` to `13`, and escape rate rises from `0.5833` to `1.0`.
- ASE FIRE is more expensive and does not improve best energy in this short run.
- ASE LBFGS lowers the unconverged count most, but its maximum residual proposal gradient is worse and the final best energy is worse.
- Therefore the next production sweep should use ASE FIRE as the main proposal-relax backend, but it should not be declared solved. The remaining coupled controls are `proposal_fmax`, `proposal_relax_steps`, and step target policy.

## Stage 6 Conservative LS-SSW Trust-Floor Run

Constraint from review:

- Do not change `proposal_fmax`, `quench_fmax`, `proposal_optimizer`, or `quench_optimizer`.
- Fix the over-strong LS-SSW perturbation and the trust-model near-zero denominator issue first.

Code changes:

- `TrustRegionBiasController.update()` now uses `max(abs(predicted_delta), error_floor) + eps` in the model-error denominator.
- The SSW walk passes `error_floor = 0.1 * step_target` into the trust update.
- PdO runner LS-SSW defaults are now conservative: `active_neighbors`, `active_count=5`, `cutoff_scale=1.15`, `strength=0.15`.
- PdO runner `proposal_relax_steps` default is `300` so FIRE is not prematurely capped, while `proposal_fmax` and optimizer defaults are unchanged.

Validation command:

```bash
pytest -q tests/unit/test_walker_policy.py tests/unit/test_config.py tests/unit/test_softening.py tests/unit/test_relax.py tests/integration/test_ls_ssw.py
```

Result: `82 passed in 1.13s`.

CUDA production run:

```bash
MPLCONFIGDIR=/tmp/mplconfig /root/miniforge3/envs/mace_les/bin/python \
  runs/20260428-pdo-mace-slab-production/run_pdo_mace_ssw.py \
  --input PdO.xyz \
  --model /root/.cache/mace/mace-omat-0-small.model \
  --output-dir runs/20260428-pdo-mace-slab-production/stage6_lsssw_conservative_trustfix_trials40_prod \
  --search-kind ls-ssw \
  --device cuda \
  --dtype float32 \
  --seed 0 \
  --trials 40 \
  --steps-per-walk 8 \
  --oracle-candidates 8 \
  --target-uphill-energy 0.8 \
  --min-step-scale 0.05 \
  --max-step-scale 1.2 \
  --proposal-trust-radius 1.5 \
  --walk-trust-radius 4.0 \
  --dedup-rmsd-tol 0.4
```

Comparison:

| run | best energy (eV) | unique minima | energy span (eV) | median RMSD to best (A) | proposal unconverged | trust error mean | LS terms last |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary SSW `seed0_trials40_cuda_truecurv` | `-573.0081176757812` | `29` | `4.61328125` | `0.5532448716828537` | `243/318` | `11.081653898024518` | n/a |
| over-strong LS `stage5_lsssw_active20_asefire_trials40_prod` | `-571.263671875` | `41` | `9.48114013671875` | `2.4065009088081686` | `79/95` | `373.9362808586328` | `125` |
| conservative LS `stage6_lsssw_conservative_trustfix_trials40_prod` | `-573.54443359375` | `23` | `5.14959716796875` | `0.4994547400871758` | `184/320` | `2.1973554646585445` | `40` |

Interpretation:

- Conservative LS-SSW fixes the main stage5 failure mode: the archive no longer looks like over-dispersed high-energy fragmentation.
- Best energy improves by `0.53631591796875 eV` relative to the ordinary 40-trial SSW run and by `2.28076171875 eV` relative to the over-strong LS run.
- Trust-model error drops from `373.936` to `2.197`, consistent with the near-zero denominator fix plus reduced softening scale.
- The remaining issue is not gone: proposal relaxation still has `184/320` unconverged records. However, the median iterations are `20`, p90 is `52.3`, and max is `278`, so the 300-step cap is being used only in hard cases rather than clipping nearly every proposal.
- Duplicate rate increases to `0.439`, so the conservative setting improves quality but sacrifices some archive diversity. This is acceptable for this correction step because the target was to stop overdriving LS-SSW, not to maximize novelty.

Current conclusion:

The slab failure was primarily an LS-SSW intensity-control problem, not a `quench_fmax` problem. The production direction should keep conservative local softening as the default and next improve the acquisition/trust policy so duplicate rate comes down without returning to the stage5 high-energy, high-RMSD regime.

## Stage 7/8 Buckingham Dynamic Softening A/B

Review hypothesis:

- Gaussian well can behave like passive symmetric softening and, with a stale reference distance, can effectively constrain local geometry.
- Buckingham repulsive softening should actively stretch local neighbor pairs.
- Rebuilding local softening every micro-step should prevent stale `r0` from pulling the walk back toward the seed geometry.
- `direction_curvature_source=true` should test whether old Gaussian bias curvature is the main source of soft-direction collapse.

Code state for both runs:

- commit `384866a`
- LS-SSW, active-neighbor local softening
- Buckingham penalty: `local_softening_penalty=buckingham_repulsive`
- dynamic per-micro-step softening rebuild
- `active_count=5`, `strength=0.15`, `cutoff_scale=1.15`, `xi=0.3`, `cutoff=2.0`
- `proposal_relax_steps=300`, `proposal_optimizer=ase-fire`, `proposal_fmax=0.05`
- `quench_fmax=0.03`, `quench_optimizer=scipy-lbfgsb`
- CUDA, fp32, MACE `mace-omat-0-small.model`

Runs:

- A: `runs/20260428-pdo-mace-slab-production/stage7_lsssw_buckingham_dynamic_inner_trials40`
- B: `runs/20260428-pdo-mace-slab-production/stage8_lsssw_buckingham_dynamic_true_trials40`

Comparison:

| run | best energy (eV) | unique minima | energy span (eV) | median RMSD to best (A) | z-span range (A) | soft/random/bond direction fraction | duplicate rate | escape rate | trust error mean | proposal unconverged | force evals |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| stage6 Gaussian/static/inner | `-573.544434` | `23` | `5.150` | `0.499` | `7.452-7.940` | `0.791/0.156/0.053` | `0.439` | `0.550` | `2.197` | `184/320` | `16821` |
| A Buckingham/dynamic/inner | `-575.396667` | `37` | `7.002` | `1.406` | `7.351-8.233` | `0.781/0.176/0.044` | `0.098` | `0.900` | `2.718` | `219/319` | `23204` |
| B Buckingham/dynamic/true | `-573.871643` | `35` | `5.477` | `0.635` | `7.508-8.235` | `0.759/0.213/0.028` | `0.146` | `0.850` | `4.989` | `176/320` | `22937` |

Interpretation:

- Buckingham + dynamic `r0` is a clear improvement over stage6: A improves best energy by `1.852 eV`, raises unique minima from `23` to `37`, lowers duplicate rate from `0.439` to `0.098`, and broadens z-span from `0.488 A` to `0.882 A`.
- The direction-collapse hypothesis is not strongly supported by this A/B. Switching direction scoring from inner curvature to true-PES curvature only lowers soft-direction fraction from `0.781` to `0.759`, while best energy becomes worse by `1.525 eV` relative to A.
- B does reduce proposal unconverged count (`176/320` vs A's `219/319`) and narrows energy/RMSD spread, but this appears to make the search more conservative rather than better at finding low-energy minima.
- Current best PdO setting from this batch is A: Buckingham + per-step rebuilding + inner direction curvature.

Current conclusion:

For this PdO slab, the main production gain came from fixing LS-SSW activation semantics (`Buckingham`) and stale softening references (`per-step r0`), not from using true-PES curvature for direction selection. The next controlled experiment should keep `direction_curvature_source=inner` and tune Buckingham activation intensity, especially `xi` and `active_count`, while monitoring proposal relaxation cost.
