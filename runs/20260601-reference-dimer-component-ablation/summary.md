# Reference Dimer Component Ablation

Status: CUDA Phase 1 completed.

Scope:
- System: C60
- Device: CUDA
- Seeds: 0, 1, 2
- Trials per seed: 40
- Cases: 18

## Aggregate Results

| Variant | best min | best mean | mean minima | mean duplicate | mean force evals | mean wall s |
|---|---:|---:|---:|---:|---:|---:|
| `paw_current_bias_relax` | -504.182 | -499.847 | 35.3 | 0.138 | 21502 | 565 |
| `paw_metropolis_pool_bias_relax` | -501.495 | -496.722 | 32.3 | 0.211 | 20929 | 663 |
| `paw_ucb_pool_no_momentum_bias_relax` | -499.275 | -495.217 | 25.3 | 0.376 | 18377 | 386 |
| `paw_ucb_reference_dimer_bias_relax` | -499.262 | -493.888 | 18.0 | 0.561 | 14134 | 385 |
| `reference_original` | -488.715 | -486.519 | 38.0 | 0.000 | 130282 | 2501 |
| `paw_ucb_reference_dimer_direct_qp_adaptive50` | -474.670 | -474.670 | 2.0 | 0.951 | 14474 | 274 |

## Per-Seed Results

| Variant | Seed | Best energy | Minima | Duplicate | Force evals | Wall s |
|---|---:|---:|---:|---:|---:|---:|
| `reference_original` | 0 | -488.715 | 38 | 0.000 | 133317 | 2518 |
| `reference_original` | 1 | -488.684 | 36 | 0.000 | 120691 | 2222 |
| `reference_original` | 2 | -482.158 | 40 | 0.000 | 136838 | 2762 |
| `paw_current_bias_relax` | 0 | -504.182 | 32 | 0.220 | 22849 | 436 |
| `paw_current_bias_relax` | 1 | -498.504 | 37 | 0.098 | 23104 | 610 |
| `paw_current_bias_relax` | 2 | -496.854 | 37 | 0.098 | 18553 | 649 |
| `paw_metropolis_pool_bias_relax` | 0 | -496.582 | 28 | 0.317 | 18977 | 616 |
| `paw_metropolis_pool_bias_relax` | 1 | -501.495 | 35 | 0.146 | 22420 | 792 |
| `paw_metropolis_pool_bias_relax` | 2 | -492.089 | 34 | 0.171 | 21391 | 580 |
| `paw_ucb_reference_dimer_bias_relax` | 0 | -490.004 | 19 | 0.537 | 14242 | 391 |
| `paw_ucb_reference_dimer_bias_relax` | 1 | -499.262 | 17 | 0.585 | 13778 | 370 |
| `paw_ucb_reference_dimer_bias_relax` | 2 | -492.398 | 18 | 0.561 | 14382 | 394 |
| `paw_ucb_pool_no_momentum_bias_relax` | 0 | -499.275 | 28 | 0.300 | 18681 | 455 |
| `paw_ucb_pool_no_momentum_bias_relax` | 1 | -494.892 | 24 | 0.415 | 20104 | 386 |
| `paw_ucb_pool_no_momentum_bias_relax` | 2 | -491.485 | 24 | 0.415 | 16346 | 317 |
| `paw_ucb_reference_dimer_direct_qp_adaptive50` | 0 | -474.669 | 1 | 0.976 | 14483 | 265 |
| `paw_ucb_reference_dimer_direct_qp_adaptive50` | 1 | -474.670 | 3 | 0.927 | 14426 | 272 |
| `paw_ucb_reference_dimer_direct_qp_adaptive50` | 2 | -474.670 | 2 | 0.951 | 14512 | 284 |

## Findings

1. The current production-style PAM setting is the best C60 Phase 1 configuration.
   `paw_current_bias_relax` has the best 3-seed mean energy (-499.847 eV), the best single run (-504.182 eV), and good coverage (35.3 minima). It is also about 6x cheaper than `reference_original` in force evaluations.

2. Returning to the original-style dimer soft mode is not competitive in this implementation.
   `paw_ucb_reference_dimer_bias_relax` is cheaper, but duplicate rate rises to 0.561 and mean minima drops to 18.0. It occasionally finds a decent basin (-499.262 eV on seed 1), but the average result is far below `paw_current_bias_relax`.

3. The dimer rotator did not actually converge.
   For all three `paw_ucb_reference_dimer_bias_relax` seeds: `reference_dimer_mean_rotations=15.0`, `reference_dimer_converged_fraction=0.0`, and `reference_dimer_mean_abs_dot_initial≈0.999`. This means the current dimer direction is effectively staying close to the initial mode, not performing a useful soft-mode rotation.

4. UCB outperforms Metropolis under the current direction pool.
   `paw_metropolis_pool_bias_relax` has a respectable best run (-501.495 eV), but its mean best energy (-496.722 eV), duplicate rate (0.211), and coverage (32.3 minima) are all worse than `paw_current_bias_relax`.

5. Momentum candidates are useful in C60.
   Removing momentum worsens mean best energy from -499.847 to -495.217 eV, increases duplicate rate from 0.138 to 0.376, and reduces coverage from 35.3 to 25.3 minima.

6. `reference_dimer + Direct-QP adaptive50` collapsed completely.
   It produced only 1-3 minima per seed, duplicate rate averaged 0.951, and all best energies stayed near -474.67 eV. Direct-QP diagnostics show low progress (~0.120) and high model error, while the same non-converged dimer direction problem remains.

## Acceptance Check

- Approved CUDA Phase 1 scope was executed: yes.
- All expected 18 rows exist in `results.csv`: yes.
- Per-case summary paths in `results.csv` exist: verified separately.
- CUDA was visible only outside the sandbox; the first sandboxed attempt failed and is recorded in `event_log.jsonl`.

Primary conclusion:
Do not replace the current scored direction pool with the current reference-dimer implementation. For C60, the current PAM-SSW configuration remains the production baseline to beat.
