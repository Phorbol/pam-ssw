# PAM-SSW Direction Anchor Fix Report

## Rationale

The provided direction-selection plan is mostly correct. The important correction is that standard SSW should not select directions only by independent candidate curvature. It needs a stochastic initial intent, formed from a global random direction plus a local non-contact pair direction, and the soft-mode choice should stay regularized toward that intent.

This implementation follows that principle:

- Generate a per-walk mixed anchor direction.
- Add random non-neighbor bond-formation directions to standard SSW, not only LS-SSW.
- Score candidates with curvature, damage, continuity, novelty, and anchor deviation.
- Keep random candidates in the pool so baseline reachability is preserved.

One ambiguity in the input plan is that it states both "anchor to the step-0 mixed direction" and "update anchor to the previous dimer result". This implementation uses a fixed per-walk initial anchor, matching the engineering section and directly preventing drift into the same floppy mode.

## Code Changes

Changed files:

- `pamssw/config.py`
  - Added `anchor_weight`.
  - Added `n_bond_pairs`.
  - Added `bond_distance_threshold`.
  - Added `lambda_bond_start` and `lambda_bond_end`.
  - Changed default `proposal_fmax` from `5e-3` to `2e-2`.
- `pamssw/walker.py`
  - `DirectionScorer` now accepts `anchor_direction` and penalizes deviation from it.
  - `CandidateDirectionGenerator` can sample random non-neighbor pairs for standard SSW.
  - Added adaptive non-neighbor threshold based on median nearest-neighbor distance when no threshold is configured.
  - Added `generate_initial_direction()` for random + bond mixed anchors.
  - `SoftModeOracle.choose_direction()` now passes anchor direction through scoring.
  - `_walk_candidate_from_seed()` creates one mixed anchor per walk and reuses it through the walk.
  - Dynamic bond candidates are counted inside the existing candidate budget, so HVP cost is not increased.
- `pamssw/relax.py`
  - Bound-constrained proposal relaxation now reports projected gradient norm, matching L-BFGS-B/KKT convergence semantics.
- Tests:
  - Added unit tests for dynamic non-neighbor bond candidates.
  - Added unit tests for mixed initial direction diversity.
  - Added unit tests for anchor penalty scoring.
  - Added unit test for projected-gradient reporting under active bounds.
  - Updated LS-SSW integration test to disable standard SSW bond candidates for that specific LS-vs-plain comparison.

## Verification

Full test suite:

```text
pytest -q
73 passed in 4.21 s
```

SSW-only LJ13/LJ38 smoke after fixed candidate budget and `proposal_fmax=0.02`:

```text
LJ13 seed0 budget30:
  gap = 0.4273960382832911
  proposal_relax_unconverged = 134 / 236
  direction_mean_candidate_pool_size = 12.872881355932204
  direction_selected_bond = 38
  force_evaluations = 21723

LJ38 seed0 budget30:
  gap = 5.334589853144962
  proposal_relax_unconverged = 211 / 238
  direction_mean_candidate_pool_size = 12.873949579831933
  direction_selected_bond = 23
  force_evaluations = 28689
```

ASE quick gate before the candidate-budget correction, using the same anchor/bond idea, gave:

```text
LJ13 seeds 0/1 budget60:
  SSW mean gap = 0.4273960369243923
  ASE-GA mean gap = 0.42739604026920475
  ASE-BH mean gap = 1.8548068526518584

LJ38 seeds 0/1 budget60:
  SSW mean gap = 4.0731958747153385
  ASE-GA mean gap = 6.944124653889929
  ASE-BH mean gap = 11.217608681924034
```

That run had a larger candidate pool than intended, so it is useful as a direction signal but not the final fair benchmark for the corrected candidate-budget implementation.

## Proposal Relaxation Status

The direction fix improves exploration and makes bond directions active, but it does not fully solve proposal relaxation convergence.

Findings from LJ38 seed0 budget30 sweeps:

- Increasing `proposal_relax_steps` reduces `proposal_relax_unconverged`, but the energy gap is not monotonic and force cost rises substantially.
- `proposal_fmax=0.02` is a better default than `5e-3` for biased intermediate proposal relaxation: lower force cost and acceptable gap, while true quench remains strict at `1e-3`.
- The old unconverged diagnostic was partly inflated because bound-constrained L-BFGS-B was judged by raw gradient instead of projected gradient. This is now fixed.

Remaining issue:

- Even after these fixes, LJ38 still has a high proposal-relax unconverged fraction under budget30. The next production step should tune the trust controller and step target, not just increase max iterations.

## Generality

This change is not LJ-cluster overfitting:

- The non-neighbor threshold is adaptive by default.
- No LJ motif, magic size, literature structure, or cluster-specific coordinate template is introduced.
- PBC systems skip random non-neighbor pair generation until MIC-aware pair selection is implemented.
- The added mechanism is a direction-selection regularizer, not a GA/BH reseed or recombination operation.

