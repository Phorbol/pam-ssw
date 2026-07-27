# Proposal relaxation `fmax=0.05` versus `0.10`

## Fixed protocol

The experiment replays the same captured one-bias proposal task with
safe-L-BFGS (`history=10`) at `fmax=0.05` and `0.10 eV/Å`. Local softening is
disabled. Each proposal endpoint is then quenched on the true MACE PES with the
same SciPy L-BFGS-B configuration (`fmax=0.05 eV/Å`, `maxiter=400`).

Capture eligibility is decided before either arm is run. The matrix contains
eight paired tasks per system. PdO seed 46 is recorded as a shared pre-arm
geometry failure and is excluded from both arm denominators.

## Cost and certificate result

| System | Arm | Proposal FE | Landing FE | Proposal + landing FE | Wall time (s) | Proposal certified | Landing certified |
|---|---:|---:|---:|---:|---:|---:|---:|
| C60 | 0.05 | 612 | 429 | 1,041 | 17.43 | 2/8 | 8/8 |
| C60 | 0.10 | 423 | 401 | 824 | 13.62 | 8/8 | 8/8 |
| PdO | 0.05 | 811 | 420 | 1,231 | 26.43 | 8/8 | 8/8 |
| PdO | 0.10 | 686 | 400 | 1,086 | 23.37 | 8/8 | 8/8 |

Relative to `0.05`, `0.10` reduces proposal force evaluations by 30.9% for
C60 and 15.4% for PdO. Including the downstream landing quench, the reductions
are 20.8% and 11.8%, respectively. It is never more expensive on these paired
tasks.

Shared pre-arm costs are not charged to either arm: C60 uses 37 bootstrap plus
216 capture evaluations; PdO uses 34 bootstrap plus 171 capture evaluations.

## Semantic result

The looser stopping condition is not a numerically neutral optimizer
substitution. Under the current archive energy-plus-RMSD matcher, the paired
landing-equivalence counts are 4/8 for C60 and 1/8 for PdO. Four C60 pairs have
materially different landing energies and/or structures. Most nonmatching PdO
pairs differ by only about 0.001--0.010 eV and 0.004--0.024 Å, but still fail
the current `0.001 eV` energy threshold.

## Decision

`fmax=0.05 eV/Å` is not justified as the proposal-level default by this
experiment. On C60 it often spends extra line-search evaluations without
actually satisfying its stricter force certificate. Because proposal
relaxation is an escape action followed by a true-PES quench, `0.10 eV/Å` is
the better exploration-level candidate.

Do not describe this as an equivalent drop-in acceleration, and do not change
the global default solely from this local replay. The next end-to-end SSW
ablation should compare the two proposal stopping conditions at fixed total
force-evaluation budget and measure best-energy and unique-minimum discovery.

The machine-readable source of truth is `evidence.json`; raw GPU output remains
untracked under `output-v2/`.
