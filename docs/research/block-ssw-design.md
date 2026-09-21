# Cell/atomic block SSW: explicit experimental design

The goal is a useful independent ASE algorithm informed by physical derivatives,
native code and literature, not literal imitation of paper inconsistencies.
`block_ssw.py` implements cell displacements and atomic climbing as successive
operators; `vc_reference.py` retains the unified log-strain alternative. Neither
is declared universally superior or complete native release parity.

## Implemented physical and lifecycle contracts

`CellChart` uses nine row-lattice entries with fixed unwrapped fractional atom
coordinates. ASE enthalpy derivatives are `gL=V L^-T (sigma+pI)`. Three physical
rotation tangents `L A`, with antisymmetric A, are removed. A projector fixed at
the center is used for each finite-difference mode problem; full physical E/F/S
remain intact. Real triclinic EMT checks include pressure and all nine entries.

Cell directions reuse the independently tested plane-dimer numerical solver,
without atomic rank-one orientation bias. The default six-request budget and
0.005 Angstrom spacing come from examples in the2014 work; callers may specify
another numerical budget. Paper rotation-force units translate to the normalized
HVP residual using `tol_HVP=tol_Frot/(2*dL)`. Conventional positive Rayleigh
curvature is retained; the printed sign ambiguity is not copied. Budget-limited
modes are explicitly marked unconverged and usable, rather than falsely certified
as exact eigenvectors.

Each cycle moves L by `fraction*||L||F*Ncell`, remaps atoms affinely, and performs
at most the configured fixed-cell atom steps. Maxiter at this intermediate stage
is allowed; numerical line-search/evaluator failure is not treated as the same
condition. The Frobenius step is not a universal bound on volume or each cell
edge, nor is it lattice-basis invariant. Positive determinant is checked without
clamping or switching potentials.

The optional atomic operator is `atomic_climb`: extracted from the existing
fixed-cell SSW path with no initial/final true quench and no MC. It uses the outer
enthalpy threshold translated to a fixed-cell energy by subtracting pV. Its
Ritz/dimer events and call counts match the existing first-step Cu baseline.

A pure shared `cell_quench` handles final E+pV relaxation. Its physical
force/stress stopping rule uses the accepted optimizer state and a fresh final
certificate. The unified VC walker reuses the same numerical helper; existing
VC regression tests pass. Final selection acts only after all-DOF quenching,
and valid rejected landings remain in the returned archive candidates.

## Explicit assumptions, not attributed native facts

- `cell_cycles` is an actual count; no compatibility toggle for paper indexing.
- One-based steps divisible by `atomic_period` include atomic climbing. The
  schedule is explicit; paper figure/text disagreement does not become an
  unlabelled branch.
- Current prototype resamples a random cell anchor at each cycle. Native
  direction persistence is **unresolved**; see `native-cell-direction-lifecycle.md`.
  This is a provisional strategy, not an established native rule or universally
  optimal continuation mechanism. Any change must preserve this version's data.
- Final quench uses a log-strain optimizer chart with an explicit numerical
  length scale; that scale does not define the block's cell proposal geometry.
- Parameters0.15, five cycles,25 atom steps, atomic width0.6/ten Gaussians derive
  from reported examples, not mathematical optima for MACE oxides. Numerical
  Safe-total/plane-dimer choices are separate from native Broyden/LBFGS behavior.

## Evidence and current validation purpose

Native original-instruction tests now establish stored
`dedlatt=-V*(stored_stress+pI)@stored_celli` to1.42e-14 on four matrix inputs.
The stored stress sign, inverse-cell convention and complete consumers must be
closed before mapping that expression directly to ASE. Native scart contains
nine raw lattice entries rather than an explicit log-strain field; its complete
atomic remapping remains partly unresolved. These findings support independent
physical cell-vector operators, not a claim of complete native parity.

Cu7 vacancy wiring tests exercise partial maxiter continuation, both schedule
branches, physical certificates, complete costs and input immutability. They
are implementation checks only. Main scientific development uses uploaded
AlOH26 and official-SI brookite TiO2_48 with MACE-OMAT-small. The first two-step
runs have independently declared1500-request/300-second per-case CPU limits,
full source snapshots and identical final physical tolerances. Censored work
will not be counted as a successful full trajectory; no same-cost superiority
claim is possible from these development runs.
