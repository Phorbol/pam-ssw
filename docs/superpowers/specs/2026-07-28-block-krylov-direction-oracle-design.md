# Block Krylov Direction Oracle Design

## Status

This specification defines the next direction-algorithm experiment after the
paired plain Rayleigh--Ritz validation.

It is intentionally not an implementation plan. No production default changes
in this milestone.

The implementation must start from the clean
`feature/posterior-terminal-outcome-validation` worktree at commit `6492356`.
The dirty main checkout and the historical CBD-like, constrained-Lanczos, and
reference-Dimer worktrees are evidence sources only.

## Decision

Test a native, budget-limited block Krylov--Ritz direction oracle before adding
direction-level UCB, Thompson sampling, learned scoring, or a new Dimer
optimizer.

The oracle keeps the two physical ingredients of the original SSW initial
intent separate:

- one projected global random displacement;
- one projected local pair displacement.

It does not combine them with a fixed mixing coefficient. Instead,
Rayleigh--Ritz determines their locally soft combination from Hessian-vector
products.

The first experiment answers:

> Under a fixed HVP budget, is PES exploration improved by spending direction
> evaluations on many shallow random/local intents or on deeper curvature-based
> refinement of fewer intents?

It does not answer whether the lowest Hessian mode is universally the best
escape direction.

## Evidence motivating the design

The original SSW biased-CBD direction is not required to converge to the
globally softest Hessian eigenvector. It is intended to remain related to an
initial random/local direction while becoming softer.

The existing PAM-SSW experiments do not isolate that mechanism:

- `rayleigh_ritz` only recombines an already evaluated candidate span and adds
  no new Hessian information;
- historical `constrained_lanczos` adds a cone parameter, an extra candidate,
  the existing weighted scorer, and unequal HVP cost;
- historical `cbd_like` uses a penalized tangent-gradient objective and costly
  backtracking rather than a Dimer/CBD rotation;
- historical reference-Dimer runs report no converged rotations and directions
  almost identical to their initial axes.

Consequently, none of those outcomes is evidence for or against native seeded
Krylov refinement.

## Scientific boundary

The milestone changes one algorithmic seam: construction of the direction
returned to the existing biased walk.

The following remain frozen:

- bootstrap true-PES quench;
- starter archive and starter selector, including the current fixed-prior
  UCB-like policy;
- Gaussian-bias equations and accumulation lifetime;
- local-softening configuration;
- proposal displacement, proposal optimizer, and its stopping criteria;
- true-PES quench optimizer and its stopping criteria;
- archive matching and duplicate logic;
- one-action terminal credit assignment;
- total campaign force-evaluation budget;
- calculator precision, model, structures, frozen atoms, and random seeds.

This milestone does not add:

- Thompson sampling or another posterior acquisition rule;
- contextual features or learned direction scores;
- reaction-network rewards;
- direction novelty, damage, anchor, continuity, or history weights;
- cone-overlap constraints;
- active-region localization penalties;
- DPP selection;
- a batch relaxer;
- variable-cell motion.

## Meaning of "unbiased"

Here, unbiased has the SSW exploration meaning:

- no predefined reaction coordinate;
- no target product, transition state, or pathway;
- globally random initial displacement retains support;
- the local pair displacement supplies a generic structural perturbation rather
  than a target reaction label.

It does not mean detailed balance, canonical sampling, an unbiased stationary
distribution, or an unbiased estimator of thermodynamic observables.

The sign of a local pair axis is not assigned a bond-formation or bond-breaking
meaning. Curvature and the axial Gaussian are invariant under reversal of the
direction axis.

## Mathematical model

Let \(P\) be the existing projector that removes frozen and rigid-body degrees
of freedom. For physical intent \(b\), generate

\[
r_b = \operatorname{normalize}(P\xi_b),
\qquad
\ell_b = \operatorname{normalize}(P d_{i_bj_b}),
\]

where \(\xi_b\) is a global Gaussian displacement and \(d_{i_bj_b}\) is the
equal-and-opposite displacement of one sampled movable atom pair.

The initial intent block is

\[
Q_{b,0} = \operatorname{orth}\{r_b,\ell_b\}.
\]

If the local pair direction is unavailable or linearly dependent after
projection, the block has rank one. The implementation does not resample merely
to force rank two.

For the current proposal-PES Hessian operator \(H\), build a truncated block
Krylov space

\[
\mathcal K_K(H,Q_{b,0}) =
\operatorname{span}\{Q_{b,0},HQ_{b,0},\ldots,H^{K-1}Q_{b,0}\}.
\]

All generated basis vectors are fully reorthogonalized because the intended
spaces are small. No restart algorithm, preconditioner, residual tolerance, or
eigenvalue-shift parameter is introduced.

For orthonormal basis \(Q_b\), form

\[
T_b = \frac{1}{2}
\left(Q_b^\top H Q_b + (Q_b^\top H Q_b)^\top\right)
\]

and solve

\[
T_b y_b = \theta_b y_b.
\]

The block proposal is the lowest Ritz direction

\[
u_b = Q_b y_{b,\min}.
\]

Across independent intent blocks, execute the direction with the lowest
Rayleigh quotient \(\theta_b\). No existing weighted `DirectionScorer` term is
applied to the block-Krylov arms.

The sign is aligned with the largest-magnitude component in the initial block
for deterministic logging. The sign has no physical selection meaning.

## Operator choice

The first end-to-end experiment uses the current total proposal-PES operator.
This isolates the direction representation/refinement seam from the existing
walk semantics.

Each finite-difference probe already exposes both:

- total proposal-PES gradient;
- true-PES gradient.

Therefore every basis HVP must retain both the total and true components. The
true-PES projected curvature is a diagnostic and costs no additional model
evaluation.

Selecting directions from the true-PES operator instead of the total proposal
operator is a later one-seam ablation. It must not be combined with the first
block-Krylov comparison.

## HVP semantics and accounting

The correctness baseline remains the existing central finite difference:

\[
Hq \approx
\frac{\nabla V(x+hq)-\nabla V(x-hq)}{2h}.
\]

One central HVP costs two calculator force evaluations. Both calls must be
attributed to `direction_oracle`; no probe, residual check, fallback, or final
curvature evaluation may bypass the budgeted calculator.

The existing native HVP bundle must be reused:

- a basis HVP contributes to Krylov expansion;
- the same HVP contributes to the projected Hessian;
- its true-PES component contributes to projected true curvature;
- selecting the resulting Ritz direction must not trigger another HVP.

The first implementation must not add an HVP cache outside one direction-oracle
call.

### One-sided reuse gate

A later cost-only arm may reuse the already known base gradient:

\[
Hq \approx
\frac{\nabla V(x+hq)-\nabla V(x)}{h},
\]

reducing the incremental cost to one force evaluation per HVP.

It may advance to production comparison only if fixed-state diagnostics show:

- stable direction-subspace principal angles relative to central differences;
- stable candidate ordering;
- acceptable bilinear symmetry defect;
- stable results over a preregistered finite-difference step sweep.

Central and one-sided results must never be pooled as if they used the same
operator accuracy.

## Breadth--depth allocation

An intent block normally has two initial columns. With an HVP budget of 12,
test the following allocations:

| Arm | Independent intent blocks \(B\) | Krylov depth \(K\) | Maximum HVPs \(2BK\) |
|---|---:|---:|---:|
| variational breadth | 6 | 1 | 12 |
| shallow refinement | 3 | 2 | 12 |
| balanced refinement | 2 | 3 | 12 |
| deep refinement | 1 | 6 | 12 |

Rank-deficient blocks may consume fewer HVPs. Unused direction budget returns
to the fixed total campaign budget; it is not filled with replacement
heuristics.

The existing production direction path is retained as a separate baseline.
Its actual HVP count can vary because momentum and valid pair candidates vary.
End-to-end fairness is therefore enforced by the identical total campaign
force budget, while the block-Krylov arms additionally have equal maximum
direction cost.

## Minimal software change

Do not refactor `walker.py` broadly.

Add one narrow direction-oracle implementation with the existing external
contract:

```text
State + ProposalPotential + walk-local intent seed
  -> DirectionChoice
```

The implementation needs only:

1. a small immutable walk-local intent record containing the random axis and
   sampled pair identity/axis;
2. a pure block-Krylov/Ritz routine consuming an HVP callable and HVP budget;
3. one opt-in direction mode routed through the existing oracle call;
4. structured diagnostics on the existing proposal/action record;
5. unit tests for the linear algebra, evaluation count, projection, and
   selected-direction no-extra-HVP invariant.

Do not create a new scheduler, archive, calculator wrapper, optimizer, or
generic eigensolver framework.

The production default remains the existing discrete path.

## Diagnostic record

Each direction selection must record:

- direction mode;
- number and rank of intent blocks;
- requested and consumed HVP budget;
- total and true direction force evaluations;
- Ritz dimension per block;
- selected Ritz value;
- selected true-PES projected curvature;
- explicit projected residual norm;
- overlap with the initial random/local span;
- participation ratio or equivalent displacement-locality statistic;
- projected-Hessian antisymmetry before symmetrization;
- termination reason;
- wall time.

Diagnostics are observations, not score terms.

Near-degenerate low modes should be interpreted using low-dimensional subspace
angles. A jump between individual eigenvectors inside a near-degenerate
subspace is not by itself a physical mechanism change.

## Validation stages

### Stage 0: exact or controlled operator tests

Use analytic quadratic and small analytic atomic systems where an explicit
Hessian or exact HVP is available.

Verify:

- lowest Ritz values are monotone non-increasing as the Krylov space expands;
- the reported residual agrees with the explicit residual;
- rigid/frozen projections are preserved;
- every HVP is charged exactly once;
- rank-one and near-degenerate initial blocks behave deterministically;
- no selected-direction HVP is evaluated after Ritz extraction.

This stage validates the algorithm and accounting, not chemical usefulness.

### Stage 1: fixed-state C60/PdO direction audit

Select preregistered early, intermediate, and plateau states from existing
trajectories. Use the same state, random intent seeds, pair identities,
calculator, precision, and HVP step for every arm.

Compare:

- current native candidate span;
- current zero-extra-HVP plain Ritz;
- the four block breadth--depth allocations.

Report curvature, residual, initial-span overlap, subspace angle, locality,
symmetry defect, force evaluations, and wall time.

No end-to-end energy claim follows from this stage.

### Stage 2: paired fixed-budget survivor gate

Run C60 with seeds 42, 43, and 44 and the same 6000 total force-evaluation
budget used by the latest paired direction study.

Compare:

- existing discrete production baseline;
- variational breadth;
- balanced refinement;
- at most one additional allocation selected by Stage 1 without looking at
  Stage 2 terminal energies.

Primary outcome:

- fixed-budget best true-PES energy.

Secondary outcomes:

- best-energy versus force-evaluation AUC;
- unique archived minima;
- duplicate fraction;
- proposal and true-quench failure counts;
- force evaluations by purpose;
- direction-oracle fraction of total cost;
- wall time;
- selected-direction diagnostic distributions.

Three seeds are a survivor gate, not a statistical significance claim.

An experimental arm survives only if it:

- improves paired best energy in at least two of three C60 seeds;
- does not materially collapse unique-minimum coverage;
- stays within the exact total force budget;
- has no unattributed evaluations.

### Stage 3: transfer gate

Only a surviving C60 arm runs on PdO with the same three paired seeds and fixed
total force budget.

The production default remains unchanged unless the arm survives both systems.
Even then, a 200-macro-step C60/PdO production run is a separate confirmation
milestone, not part of this initial implementation.

## Dimer comparison

Do not call the first implementation CBD or biased-CBD.

After the block-Krylov experiment, a standard rotation-only Dimer may be added
as a solver comparison using:

- the same initial intent;
- the same projected coordinates;
- the same finite-difference oracle;
- the same maximum force/HVP cost;
- no translation or altered proposal optimizer.

The comparison isolates minimum-mode refinement. A full Dimer
rotation-plus-translation path is a different uphill policy and must be tested
separately.

The exact historical biased-CBD anchoring potential and Broyden update are not
reconstructed from third-party code or incomplete accessible equations.

## Posterior phase boundary

No direction posterior is implemented in this milestone.

The diagnostic and terminal logs are designed so a later study can define a
direction action as:

```text
intent source x refinement allocation
```

Before online adaptation, use the accumulated data to test whether curvature,
residual, initial-span overlap, locality, and cost predict terminal improvement.

Only if they have predictive value should the same logged actions be used for
a paired comparison of:

- uniform allocation;
- the current UCB-like acquisition form;
- Thompson sampling.

The posterior model and acquisition rule remain separate. Thompson sampling is
not promoted on mathematical form alone.

## Reproducibility

Every run records:

- exact commit and dirty-state preflight;
- complete effective configuration;
- structure and model checksums;
- device, precision, dependency versions, and random seed;
- total and purpose-resolved force counts;
- HVP allocation and operator semantics;
- terminal action records;
- wall-clock timing.

The run must fail closed on a dirty worktree, budget mismatch, missing
direction diagnostics, or unattributed calculator call.

## Claim ceiling

Passing Stages 0 and 1 proves only that the native block-Krylov oracle:

- constructs the intended subspaces;
- reuses HVP information correctly;
- obeys the evaluation budget;
- returns measurable direction diagnostics.

Passing the three-seed C60/PdO gates would support the limited empirical claim
that one preregistered breadth--depth allocation improved the tested
fixed-budget PAM-SSW searches.

It would not prove:

- universal superiority over Dimer, CBD, Lanczos variants, or the existing
  direction pool;
- statistically unbiased PES sampling;
- canonical or kinetic correctness;
- cross-system generalization;
- optimality of the HVP budget or finite-difference step;
- that lower local curvature is causally sufficient for better basin
  discovery.
