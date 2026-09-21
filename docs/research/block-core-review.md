# Block SSW: correctness versus the scientific claim to test

2026-09-10. Read-only review of `pamssw/standalone/block_ssw.py` and `cbd_cell.py`, against [the native cell-force contract](native-cell-force-contract.md). No implementation or PES changes.

**The strongest falsifiable claim is that the complete cell block provides useful extra basin coverage per oracle cost, relative to atomic SSW with the same final cell quench.** The current code has a consistent physical-coordinate objective, but its cell proposal is a sequence of fresh soft-direction lattice displacements and partial atomic relaxations. Its usefulness as landscape-aware escape remains unvalidated. The proposed four arms can test that package-level claim; they cannot identify one internal design choice as its cause.

## 1. The physical derivative is not the central open problem

`CellChart.evaluate` (cbd_cell.py:84) evaluates H=E+pV at fixed fractional atom coordinates and returns the row-cell derivative `gL = V L^(-T)(sigma+pI)`. It keeps physical force/stress intact. Cell-mode rotation tangents are projected with a center-fixed projector during the finite-difference direction problem. This is an explicit derivative/coordinate contract, rather than treating stress as a force with arbitrary units.

The native original-instruction result `dedlatt=-V*(stored_stress+pI)@stored_celli` supports a volume/pressure/inverse-cell-aware conversion, but its stored conventions and full consumer are not yet equivalent to ASE's positive mathematical derivative. Our sign is justified by the objective chain rule, not by making the native field name agree with a printed equation. The independent coordinate derivative is a correctness question already addressable by finite differences; matching a native label is not a scientific launch prerequisite.

In `run_block_ssw`, partial atom-only optimization uses E at fixed cell; pV is constant at that stage. The optional atomic climb receives `current.objective-p*V_cell`, so its fixed-cell energy comparison corresponds to the same outer enthalpy reference. Final quench and MC selection use the true E+pV objective. The valid-candidate archive receives converged rejected landings as well as accepted ones. These are coherent aspects of the objective/selection implementation; they do not establish efficient global exploration or equilibrium sampling.

## 2. Highest-priority unvalidated design: what direction is actually useful?

Each cell cycle in block_ssw.py constructs a new CellChart at the current work geometry, calls `cell_direction` with a fresh nine-dimensional random anchor, moves by `cell_step_fraction*||L||F`, then performs at most `partial_atom_steps` fixed-cell atom steps. The default block repeats this five times. The declared direction source correctly says `fresh_random_each_cycle_unverified_native_policy`.

The mode problem evaluates affine cell perturbations with fractional atoms frozen, while the subsequent atomic stage allows internal motion. Therefore its curvature is **clamped-internal-coordinate cell curvature**, not the Hessian of a fully internally relaxed enthalpy surface. At an internally stationary, stable point that relaxed curvature would involve the familiar Schur-complement term `H_LL-H_Lx H_xx^(-1) H_xL`; away from such a point, that reduction is not justified. Partial relaxations may terminate at maxiter, and the next cell mode need not be computed at an atomic stationary point. This is an explicitly approximate proposal design, not an erroneous derivative of the stated fixed-fractional objective.

Consequently there is no established argument that the repeatedly selected affine soft directions remain useful directions for the *relaxed* basin landscape. Each random restart also breaks directional continuity between cycles. These limitations are directly relevant to complex internal modes in AlOH26 and brookite48. The measured expanded, higher-energy rejected endpoints and expensive incomplete combined proposals are consistent with the concern, but do not prove it: there is only limited development data and budget censoring.

The four-arm test falsifies the package's utility if, at matched total cost and shared starts, block VC produces no added physically credible coverage over posterior-cell-quench SSW, or its failures/pathological candidates consume the apparent benefit. It does not prove that fresh resampling, rather than finite step size or partial relaxation, caused the outcome. No new mechanism or targeted retuning is proposed in this review.

## 3. Finite step size: substantial and representation-dependent

The implemented displacement is `ΔL=0.15*||L||F*Ncell` for a unit direction. This fixes a **relative matrix norm**, not a bound on each principal strain, determinant, density, or the physically relevant barrier scale. Under a non-orthogonal integer change of cell basis, `||L||F` and the Euclidean metric on its entries change. A particular input representation is therefore part of the experiment. Center rotation projection removes infinitesimal rotational directions, but does not make a finite lattice displacement basis-invariant.

Neither a positive determinant check nor a final small stress guarantees that this finite move stays in a useful crystal basin region or the MLIP's credible domain. The existing +13–16% endpoint expansions must remain visible in the coverage analysis. These are search-design and validation limits, not evidence that the affine coordinate/force chain rule is wrong. Four arms with fixed representations evaluate this configuration's performance; they do not establish representation invariance or universal suitability of 0.15.

## 4. Cell cycles are not Gaussian-biased cell continuation

`cell_direction` uses `rotation_bias=0`; the cell cycles do not build a cell Gaussian history or quench a progressively deformed cell objective. They perform finite lattice displacements followed by partial atomic relaxation. Gaussian-biased continuation is present in the optional **atomic** `atomic_climb` stage. It is therefore inaccurate to describe the current cell block itself as a recovered Gaussian-biased lattice SSW continuation.

This absence is a structural description, not a demonstrated missing ingredient: neither the native lifecycle audit nor the current comparison proves that adding cell bias is necessary or beneficial. The block mechanism can be evaluated as the explicitly implemented proposal. The joint arm is a useful alternative, but a block-vs-joint difference combines coordinates, mode selection, finite step, relaxation and bias lifecycle; it is not an isolated test of cell Gaussian history.

## Decision and test interpretation

Preserve the current implementation/configuration as an experimental baseline. Freeze its fresh-direction policy without waiting for native parity closure. Before interpreting the four-arm outcomes, ensure the runner keeps the persistent outer-step schedule so `atomic_period=2` actually executes the combined branch; repeated one-step whole-walker calls would instead test a different algorithm.

Prioritize: (1) completed useful candidates and complete cost versus posterior-cell-quench baseline; (2) physical plausibility of expanded/coordination-changing endpoints; (3) clarity about which proposal was executed and censored. Fixed-cell SSW gives a constrained-space reference; joint VC gives a different full proposal mechanism. Report all valid rejected landings, all failures and all initialization/HVP/partial-relax costs. Coverage advantage, if observed, remains case/configuration-specific; a negative result limits this block package rather than identifying a single guilty parameter. Neither result establishes new phases or strict native reproduction.
