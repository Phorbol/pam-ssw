# TYPE4 TiO₂@Au₂₄O₄ periodic-direction qualification

**Status: staged only; no search job submitted.** This is the next example-case
qualification after the frozen bulk TiO₂ run. It tests whether the periodic
direction mode can operate on the intact 514-atom source structure with fixed
substrate atoms and a smaller direction subspace. It does not rank structures
against old energies or validate the Au/Ti/O model against DFT.

The verified source bundle is
`/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/type4-source-direction-mace-v100/`.
It contains the original `input.arc` and `lasp.in`; the original source tree is
referenced in place. The ARC input has 514 atoms and full 3D PBC. The script
records the source path, cell, PBC, atom order, initial structure and effective
configuration in a new caller-selected output root.

## Controlled comparison

Both arms use the same 514 atoms, source cell/PBC, seed `26092631`, two outer
steps, three-Gaussian ceiling, MACE-OMAT-0-small `omat_pbe` head, float64 CUDA,
and the same direction exclusion `0..350` (0-based). Physical FixAtoms remain
`0..296` (0-based), so 217 atoms are physically mobile and 163 can move in the
direction subspace. The 54 atoms `297..350` remain mobile during quenches but
are excluded from direction generation and rotation in both arms. Atoms
`351..485` remain mobile; the source segmentation support is not treated as a
physical fixed mask.

The `global_rotation` arm uses the recovered rotation-only settings. The
`periodic_local` arm uses the periodic local-direction controller with the
same recovered rotation tolerances and 40-call budget. It sets
`c1_radius_policy="per_atom"` prospectively because the intact extended source
may exceed the existing restricted all-near domain. This keeps the recovered
12 Å per-atom radius rule; it adds no new radius or threshold. Startup order is
the already qualified randomized setting. The shared independent numerical
configuration uses width `0.6 Å`, inner/forward force `0.1 eV/Å`, temperature
`100 K`, true-force threshold `0.05 eV/Å`, finite-difference step `0.001 Å`,
quench limit 500, history 500 and at most three Gaussians.

The arms differ only in direction policy. Both receive the same additional
direction exclusion so any energy/cost difference is not caused by searching
different Cartesian rotation subspaces. Different random sequences after the
initialization draws remain part of the policy comparison; equal seeds are not
claimed to produce equal trajectories.

## Budgets and stopping

The script enforces at most 4,000 live search E/F requests per arm, at most
4,000 same-oracle replay requests for the local arm, at most six independent
fresh endpoint E/F checks total, and a 1,200-second combined wall cap. It does
not submit Slurm work. Any later submission must use one V100 and the declared
20-minute wall limit. A cap or solver error terminates that arm and is retained
with its partial ledger; there are no automatic retries or budget increases.

For the local arm, split `1+1` recovery is evaluated by replaying its own
recorded E/F stream. The driver compares every request's ordered atoms and
coordinates, then compares the final checkpoint against the continuous
two-step checkpoint. This tests state continuity without assuming repeated
GPU calls return bitwise-identical outputs. Replay is not charged as a live
MACE call, but it has its own bounded count and ledger.

## Qualification evidence

For each arm the script records actual E/F requests, failures and denials,
search status, landing count, per-Gaussian initial directions, direction-mask
support checks, and any reported local route. If route telemetry is absent, it
records `unreported`; c4/c6 coefficient presence is reported separately and
does not stand in for an executed geometry route. Multi-Gaussian memory use is
reported as observed or absent. The local checkpoint must retain the exact
source cell and PBC.

Each available initial/landing minimum gets an independent raw MACE E/F check.
Physical fixed positions, cell, PBC and atom order must remain exact; maximum
force norm over the 217 physically mobile atoms must be at most
`0.05 eV/Å`. This is constrained fixed-cell force qualification only: it does
The force maximum includes all 217 physically mobile atoms, including the
direction-excluded mobile layer. This remains constrained fixed-cell
qualification; it does not certify Hessian stability, novelty, intactness, a
phase transition, chemical accuracy or a global-search success rate. Fresh
and stored endpoint energies must also agree within `1e-6 eV`.

The old TYPE4 source run is a prior interface/cost feasibility record. Its
energies and the unrelated xTB/GPU values are not pooled or compared as a
ranking. This short pair can establish implementation use and numerical
qualification on this exact input, not an efficiency advantage or material
generalization.
