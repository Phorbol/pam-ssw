# Energy-bounded anchor versus detached Ritz on C60

## Question and frozen protocol

This experiment asks whether the zero-extra-HVP
`energy_bounded_anchor` mechanism should replace the stronger detached-Ritz
direction control.

Both arms use the same exact C60 starters, seeds, Gaussian-bias propagator,
safe-L-BFGS proposal relaxation, strict true-PES quench, archive settings, and
12-central-FD-HVP budget per direction selection:

- `detached_ritz`: a two-column detached intent block at depth 6;
- `energy_bounded_anchor`: a one-column anchor block at depth 12, followed by
  maximum-anchor-overlap reconstruction under the existing 0.8 eV local
  quadratic energy budget.

The two-starter, three-seed, two-arm matrix was executed twice at the same
commit because one detached-Ritz terminal basin proved path sensitive.  No
configuration or random seed changed between repeats.

No starter selector, UCB/TS policy, posterior update, reaction-network
objective, or production default is active or changed.

## Execution integrity

- Execution commit:
  `0e4da3de49dbc5db03f9c7c73fb90e66a500df50`.
- Completed cases: 24/24 across two exact repeats.
- Fresh strict terminal-quench certificates: 24/24.
- Quench fallbacks: 0/24.
- Independently checked escape and landing hashes: 48/48.
- Every completed selection consumed exactly 12 HVPs and 24 direction force
  evaluations.
- Bootstrap, starter-quench, and unattributed evaluations: 0.
- Both processes exited with code 0.

## Terminal outcomes

### Repeat 1

| Starter | Seed | Detached Ritz delta (eV) | Energy bounded delta (eV) |
|---|---:|---:|---:|
| intermediate | 42 | +0.000031 | +0.000275 |
| intermediate | 43 | +5.364380 | +0.000061 |
| intermediate | 44 | +0.000214 | -0.000092 |
| plateau | 42 | -0.676819 | +0.474182 |
| plateau | 43 | -7.078888 | -6.385712 |
| plateau | 44 | -4.997162 | -7.078949 |

### Repeat 2

| Starter | Seed | Detached Ritz delta (eV) | Energy bounded delta (eV) |
|---|---:|---:|---:|
| intermediate | 42 | +0.000000 | +0.000122 |
| intermediate | 43 | +5.673828 | +0.000061 |
| intermediate | 44 | +0.000031 | +0.000122 |
| plateau | 42 | -0.676758 | +0.474091 |
| plateau | 43 | -7.078888 | -6.385620 |
| plateau | 44 | +0.358521 | -7.078827 |

A meaningful outcome is a certified new basin at least 0.001 eV below its
starter.

| Arm | Repeat-1 meaningful | Repeat-2 meaningful | Combined | Total FE |
|---|---:|---:|---:|---:|
| detached Ritz | 3/6 | 2/6 | 5/12 | 5,160 |
| energy bounded | 2/6 | 2/6 | 4/12 | 4,952 |

Both arms fail every intermediate case in both repeats.  On the plateau:

- detached Ritz is reliably meaningful for seeds 42 and 43;
- energy bounded is reliably meaningful for seeds 43 and 44;
- detached seed 44 is path sensitive, landing at -4.997162 eV in repeat 1
  and +0.358521 eV in repeat 2.

Thus neither arm dominates by starter-seed condition.  Detached Ritz has one
additional successful outcome across the 12 repeated cases, while
energy-bounded uses 208 fewer force evaluations in total.  The cost difference
is only 4.0%, and the per-case medians are 426.5 and 439.0 evaluations,
respectively, so there is no robust cost dominance either.

## Direction mechanism

| Repeat | Arm | Selections | Direction FE | Median absolute anchor overlap | Median true curvature |
|---|---|---:|---:|---:|---:|
| 1 | detached Ritz | 40 | 960 | 0.0783 | 4.0690 |
| 1 | energy bounded | 33 | 792 | 0.5426 | 4.1667 |
| 2 | detached Ritz | 41 | 984 | 0.0753 | 3.8860 |
| 2 | energy bounded | 31 | 744 | 0.5393 | 4.1667 |

This is the clean positive result.  At nearly the same median true curvature,
the energy-bounded construction retains about seven times more absolute
anchor overlap.  It therefore solves the intended local
curvature-versus-intent problem without extra HVPs.

Across the repeats, 60/64 requested energy-bounded steps are feasible in the
paid subspace and four are analytically shortened.  The maximum executed local
quadratic energy remains within floating-point tolerance of 0.8 eV.

The lower direction total for energy bounded is caused by fewer walk steps
(64 versus 81), not a cheaper selection.  It saves 408 direction evaluations,
but spends 195 more proposal-relaxation evaluations and 21 more true-quench
evaluations:

| Purpose | Detached Ritz | Energy bounded |
|---|---:|---:|
| biased proposal relaxation | 2,697 | 2,892 |
| direction oracle | 1,944 | 1,536 |
| escape true-PES check | 113 | 97 |
| landing true quench | 394 | 415 |
| post-relax validation | 12 | 12 |
| total | 5,160 | 4,952 |

## Decision

1. Do not replace detached Ritz with `energy_bounded_anchor`.
2. Keep `energy_bounded_anchor` as an opt-in diagnostic arm.  Its local
   geometry is cleaner, but cleaner geometry does not improve the repeated
   terminal success count.
3. Keep detached Ritz as the current direction control, not as a proven
   universal default.  Its plateau-seed44 path sensitivity shows that one
   terminal outcome is not deterministic evidence.
4. Do not proceed to posterior UCB/TS direction learning.  Both action sets
   remain 0/6 on the intermediate starter, so the missing productive action is
   still the bottleneck.
5. Do not spend a PdO production run on energy-bounded anchor as a replacement
   candidate.  The next useful experiment should target the missing
   intermediate event family or the direction-propagator interaction, while
   retaining detached Ritz as the fixed control.

The machine-readable sources of truth are:

- `output/evidence.json`;
- `repeat-output/evidence.json`.
