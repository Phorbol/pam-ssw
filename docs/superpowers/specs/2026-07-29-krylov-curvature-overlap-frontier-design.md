# Krylov Curvature--Overlap Frontier Diagnostic Design

## Decision

Expose the complete Ritz spectrum already present in each block-Krylov solve,
then replay the 12 refined-direction C60 proposals from the completed
anchor-consistent ablation:

- `detached_ritz`;
- `anchor_lanczos`;
- two locked accepted starters;
- paired seeds 42, 43, and 44.

This phase is diagnostic only.  It does not change the selected lowest Ritz
vector, HVP depth, proposal path, Gaussian bias, local softening, optimizer,
starter, archive, or any production default.  It introduces no overlap
threshold, scalarized score, posterior, UCB-like rule, or Thompson sampler.

The preceding report explicitly selected this as the next experiment and the
user approved continuation.  The implementation therefore remains within
that approved single-mechanism scope.

## Scientific question

The completed 18-case experiment established three facts:

1. the raw random-plus-bond anchor is physically stiff and produces no useful
   lower basin;
2. anchor-seeded Lanczos is softer and more anchor-aligned than detached
   Ritz, but produces fewer useful terminal outcomes;
3. minimizing the Rayleigh quotient alone is therefore not sufficient.

The unresolved question is more precise:

> Does the already-paid Krylov subspace contain a direction that retains more
> of the random-plus-bond event intent at an observable curvature cost, or are
> softness and anchor continuity intrinsically separated in the present
> subspace?

This must be answered before inventing an overlap weight, constraint, CBD
rotation tolerance, or learned direction selector.

## Alternatives considered

### Inspect only the first direction selection

This would cost only one 12-HVP solve per case.  It is rejected because the
direction is recomputed along the biased walk.  A first-step spectrum cannot
explain whether the later accepted uphill path contains or loses a useful
curvature--overlap tradeoff.

### Add an overlap-constrained or scalarized Ritz selection now

Examples include:

\[
\min_u u^\mathsf{T} H u
\quad\text{subject to}\quad
|u^\mathsf{T}a|\ge\eta
\]

or

\[
u^\mathsf{T} H u+\lambda\left(1-|u^\mathsf{T}a|^2\right).
\]

This is rejected because neither \(\eta\) nor \(\lambda\) is identified by
current evidence.  Testing such a rule now would confound subspace content
with a new heuristic selection parameter.

### Replay the full proposal path and expose the complete spectrum

This is selected.  Every retained basis vector and its total/true HVP are
already available inside `solve_krylov_block`.  All Ritz-point diagnostics can
therefore be computed by linear algebra without a new calculator call.

The replay stops at the escape state.  It does not repeat terminal true
quenching.  Instead, it must reproduce the locked prior escape file hash and
selected-direction trace before it may reference the previously validated
terminal outcome.

## Mathematical diagnostic

Let \(Q\) be the orthonormal Krylov basis and let

\[
\widetilde H = \frac{1}{2}
\left(Q^\mathsf{T}H Q + Q^\mathsf{T}H^\mathsf{T}Q\right).
\]

For every projected eigenpair

\[
\widetilde H c_j = \lambda_j c_j,
\qquad
u_j = Qc_j,
\]

record:

- projected total curvature
  \(\kappa_j=u_j^\mathsf{T}Hu_j\);
- true-PES curvature
  \(\kappa_j^{\mathrm{true}}
  =u_j^\mathsf{T}H_{\mathrm{true}}u_j\);
- residual
  \(\lVert Hu_j-\kappa_j u_j\rVert\);
- overlap with the initial Krylov span
  \(\lVert Q_0^\mathsf{T}u_j\rVert\);
- absolute overlap with the exact physical anchor
  \(|a^\mathsf{T}u_j|\);
- atomic participation ratio.

The eigenpairs are ordered by increasing \(\kappa_j\).  The executed direction
remains point zero of the globally lowest-curvature block.

Define the parameter-free curvature--overlap Pareto frontier as the ordered
points for which no lower-or-equal-curvature point has greater-or-equal
absolute anchor overlap, with at least one strict inequality.  Operationally,
a sorted point joins the frontier only when its overlap exceeds the largest
overlap seen at lower curvature.

No area, thresholded success flag, curvature penalty, or scalar reward is
introduced.  The report retains the exact point set and summarizes only:

- frontier size;
- maximum available anchor overlap;
- the total- and true-curvature differences between the maximum-overlap point
  and the executed lowest-Ritz point;
- distributions separated by arm, starter, seed, and walk step.

## Minimal code seam

### Krylov algebra

Add an immutable `KrylovRitzPoint` record to `pamssw/krylov.py` containing the
direction and the five algebraic diagnostics.  Add
`ritz_points: tuple[KrylovRitzPoint, ...]` to `KrylovResult`.

`solve_krylov_block` accepts an optional normalized reference direction used
only for overlap diagnostics.  It constructs all `KrylovRitzPoint` records
from the existing `Q`, total `HQ`, true `HQ`, and projected eigendecomposition.
The existing top-level `KrylovResult` fields are populated from
`ritz_points[0]`, preserving the current API and selected result.

The solver must not invoke `hvp` after constructing the existing basis
products.

### Walker diagnostics

Pass the already-created common anchor to the block-Krylov solve.  Serialize a
flat `krylov_ritz_spectrum` list for every direction selection.  Each item
contains:

- block index and Ritz index;
- whether it is the executed point;
- total and true curvature;
- residual;
- initial-span overlap;
- absolute anchor overlap;
- participation ratio.

The existing scalar diagnostics and direction selection remain unchanged.
No new configuration field is added.

## Deterministic proposal replay

Create a dedicated runner under:

```text
runs/20260729-krylov-curvature-overlap-frontier/
```

The runner imports the locked state/model loader and configuration builder
from the completed anchor-consistent experiment.  It executes proposal
generation only for the 12 refined-arm cases.

For each case it must:

1. verify the prior evidence commit, model hash, starter hash, and case key;
2. execute the unchanged proposal walk with direction diagnostics enabled;
3. require the same direction-selection count and purpose-resolved direction
   cost as the prior case;
4. compare all pre-existing selected-direction scalar diagnostics to the
   prior trace;
5. write the replay escape structure and require its SHA-256 hash to equal the
   prior escape hash;
6. attach the prior certified terminal outcome by evidence reference, without
   running a new terminal quench.

If the escape hash or selected trace differs, the case is a failed replay and
must not inherit the old landing outcome.

## Accounting

Expected direction cost is unchanged:

- detached Ritz: 12 HVP = 24 force evaluations per selection;
- anchor Lanczos: 12 HVP = 24 force evaluations per selection.

The full-spectrum calculation must add:

- 0 HVP;
- 0 force evaluations;
- 0 proposal relaxations;
- 0 true-PES evaluations.

The replay purpose ledger permits direction-oracle, biased-proposal-relax,
and escape true-PES checks only.  Bootstrap, starter quench, terminal landing
quench, post-relax validation, and unattributed counts must all be zero.

The runner reports the diagnostic replay cost separately from the prior
terminal-outcome cost.  It must never combine inherited quench counts with
new replay counts as if they came from one execution.

## Interpretation boundary

This experiment can establish:

- whether alternative curvature--overlap choices already exist in the paid
  subspace;
- how much total and true curvature separates them from the selected softest
  point;
- whether the separation changes along the biased walk or between the
  intermediate and plateau starters;
- whether detached and anchor-seeded spaces differ in their available
  frontiers.

It cannot establish that selecting any unexecuted Ritz point would improve a
terminal outcome.  Those points are counterfactual directions and receive no
credit label in this phase.

A later selection experiment is justified only if the diagnostic reveals a
repeatable nontrivial frontier.  Its selection rule must then be preregistered
separately and tested with true terminal quenches.  If high anchor overlap is
available only at the same very large curvature seen for the raw anchor, the
next mechanism should change physical event construction rather than tune a
Ritz selector.

## Tests and acceptance

Tests must fail before implementation and then prove:

1. every projected Ritz point is returned in curvature order;
2. total curvature, true curvature, residual, initial-span overlap, and
   reference overlap match analytic small-matrix values;
3. the HVP call count is exactly unchanged when full-spectrum diagnostics are
   requested;
4. the selected direction and all existing `KrylovResult` scalar fields are
   unchanged;
5. returned direction arrays are read-only;
6. walker JSON includes finite per-point participation ratios and identifies
   exactly one executed point;
7. the replay cohort contains exactly 12 cases and excludes `exact_anchor`;
8. every completed replay closes its allowed purpose ledger, matches the
   locked escape hash and selected trace, and references rather than repeats
   the terminal quench;
9. production defaults and direction-selection behavior remain unchanged.

The GPU audit is complete only when all 12 replays match, every ledger closes,
the full point evidence is independently rebuilt, and the conclusion
separates observed subspace geometry from untested counterfactual outcomes.
