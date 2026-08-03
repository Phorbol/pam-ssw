# Direction Feature Learnability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `executing-plans` to implement this plan task-by-task.

**Goal:** Determine, with zero new PES evaluations, whether already-paid
direction features contain cross-state and cross-system information that can
rank terminally useful candidates.

**Architecture:** Load and repeat-average the completed shared-K4
counterfactual campaigns. Compare unfitted physical baselines with three fixed
ridge rankers under leave-group, leave-context, and leave-system-out
validation. Keep the entire experiment offline and outside `pamssw`.

**Tech Stack:** Python standard library, NumPy, pytest, existing JSON evidence.

---

### Task 1: Lock the dataset and feature semantics

**Files:**

- Create: `runs/20260731-direction-feature-learnability-gate/protocol.py`
- Create: `tests/unit/test_direction_feature_learnability_gate.py`

- [ ] Write a failing test that constructs two identical repeats of 12 groups
  with four candidates each and verifies that repeat averaging produces 48
  unique candidate rows.
- [ ] Run:
  `pytest -q tests/unit/test_direction_feature_learnability_gate.py`
  and confirm failure because `protocol.py` does not exist.
- [ ] Implement strict dataset validation:
  exact systems, states, seeds, candidate indices, certificates, geometry,
  direction SHA identity, and static-rank identity.
- [ ] Extract only:
  `true_curvature`, `abs(anchor_cosine)`, `kind`, `static_score`, and
  repeat-averaged `landing_delta_eV`.
- [ ] Exclude `score_sigma` because adaptive scaling makes its curvature energy
  term algebraically redundant in the observed positive-curvature regime.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Implement fixed-capacity held-out ranking

**Files:**

- Modify: `runs/20260731-direction-feature-learnability-gate/protocol.py`
- Modify: `tests/unit/test_direction_feature_learnability_gate.py`

- [ ] Write failing tests for disjoint leave-group, leave-context
  `(system, state_id)`, and leave-system folds.
- [ ] Write a failing synthetic-data test where complementary softness and
  intent features are required, and confirm the combined fixed ridge ranker
  beats either block alone on held-out groups.
- [ ] Implement deterministic within-group feature standardization.
- [ ] Implement closed-form ridge with a fixed intercept-free unit penalty:
  `beta = solve(X.T @ X + I, X.T @ y)`.
- [ ] Implement four rankers:
  `static_score`, `softness`, `intent`, and `combined`.
- [ ] Rank test candidates only within their four-member group and record top-1
  correctness, terminal regret, and per-system results.
- [ ] Re-run the focused test and confirm it passes.

### Task 3: Apply the preregistered promotion gate

**Files:**

- Modify: `runs/20260731-direction-feature-learnability-gate/protocol.py`
- Modify: `tests/unit/test_direction_feature_learnability_gate.py`
- Create: `runs/20260731-direction-feature-learnability-gate/README.md`
- Create: `runs/20260731-direction-feature-learnability-gate/analyze.py`

- [ ] Write a failing test that rejects the posterior stage when the combined
  ranker fails either held-out system.
- [ ] Implement the promotion rule without weighted acquisition:
  under leave-system-out, the same combined model must achieve at least 4/6
  correct top-1 selections in each system, zero median regret in each system,
  lower global mean regret than both softness and intent, and no worse global
  mean regret than the existing static baseline.
- [ ] Treat leave-group and leave-context results as diagnostics; they may not
  override a failed leave-system gate.
- [ ] Implement `analyze.py` to read the two immutable counterfactual evidence
  files, write structured evidence, and render a concise conclusion.
- [ ] Document that passing permits a later posterior design but does not
  promote TS/UCB or modify production.

### Task 4: Execute, verify, and publish

**Files:**

- Create:
  `runs/20260731-direction-feature-learnability-gate/evidence.json`
- Create:
  `runs/20260731-direction-feature-learnability-gate/conclusion.md`

- [ ] Run the analysis against both completed 48-case campaigns.
- [ ] Assert zero new calculator calls and 96/96 certified,
  geometry-valid terminal labels.
- [ ] Run focused tests and `git diff --check`.
- [ ] Run the reproducible repository suite excluding only the four known
  historical-output fixture modules.
- [ ] Commit the result, push
  `feature/direction-continuation-ablation`, and update PR #14.

## Self-review

- Scope is one offline analysis module; `pamssw` and production profiles remain
  untouched.
- There is no model family or regularization sweep.
- Fold identities are explicit and prevent same-system evidence from entering
  leave-system test predictions.
- Promotion depends only on the preregistered combined ranker and
  leave-system-out results.
- No placeholder, optional component, or hidden fitted acquisition weight is
  present.
