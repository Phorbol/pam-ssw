# Paired Continuation/Restart Selector Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a fixed full-support best-continuation/uniform-restart pair is a cleaner and more effective starter policy than node-level UCB-like selection or a Metropolis chain.

**Architecture:** First audit existing selector traces without new PES calls. Then add one opt-in two-slot selector implemented inside the existing walker seed-selection seam. A dedicated shared-bootstrap runner freezes all inner-kernel mechanics and performs preregistered C60/PdO/CuO fixed-FE comparisons.

**Tech Stack:** Python, NumPy, pytest, ASE, MACE CUDA, existing `BudgetedCalculator` purpose ledger.

---

### Task 1: Zero-FE growing-arm audit

**Files:**
- Create: `runs/20260801-selector-support-audit/PLAN.md`
- Create: `runs/20260801-selector-support-audit/analyze.py`
- Create: `runs/20260801-selector-support-audit/evidence.json`
- Create: `runs/20260801-selector-support-audit/conclusion.md`
- Test: `tests/unit/test_selector_support_audit.py`

- [ ] Write pure tests for Shannon effective support, repeat fraction and exact trace closure.
- [ ] Run the focused test and confirm the missing analyzer fails.
- [ ] Implement the pure analyzer with no `pamssw` or MACE runtime dependency.
- [ ] Analyze the nine seed-42 selector traces and record source hashes.
- [ ] Conclude only whether node-UCB-like is empirically diffuse in these archives; do not infer search superiority.
- [ ] Run JSON, hash, and focused pytest checks.
- [ ] Commit the audit independently.

### Task 2: Opt-in snapshot-paired selector

**Files:**
- Modify: `pamssw/config.py`
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_config.py`
- Modify: `tests/unit/test_walker_policy.py`

- [ ] Add failing config tests accepting only the new exact mode name while retaining `archive_ucb` as default.
- [ ] Add failing walker tests proving both starters are selected from the same pre-pair archive and the cached uniform entry survives archive growth.
- [ ] Add a failing test proving selection RNG changes do not shift physical-action RNG draws.
- [ ] Implement the smallest mode branch and cached starter state.
- [ ] Run focused config/walker tests, then the broader exploration tests.
- [ ] Commit implementation independently.

### Task 3: Shared-bootstrap four-arm gate runner

**Files:**
- Create: `runs/20260801-paired-continuation-restart-gate/.gitignore`
- Create: `runs/20260801-paired-continuation-restart-gate/PLAN.md`
- Create: `runs/20260801-paired-continuation-restart-gate/protocol.py`
- Create: `runs/20260801-paired-continuation-restart-gate/run_gate.py`
- Test: `tests/unit/test_paired_continuation_restart_gate.py`

- [ ] Add pure tests for matrix closure, shared-bootstrap equality, purpose accounting, gain-AUC integration and the exact S-CR1 admission rule.
- [ ] Implement the pure protocol.
- [ ] Reuse the existing C60/PdO/CuO resource loaders and frozen production config; change only starter mode and seed.
- [ ] Persist bootstrap minimum, every arm summary, energy-vs-FE trace, model/input hashes, effective config and purpose ledger.
- [ ] Add mechanical `--check-evidence` validation.
- [ ] Run focused tests and a low-budget CUDA smoke excluded from scientific evidence.
- [ ] Commit the runner before the full experiment.

### Task 4: Execute S-CR1 and apply the preregistered gate

**Files:**
- Create after execution: `runs/20260801-paired-continuation-restart-gate/evidence.json`
- Create after execution: `runs/20260801-paired-continuation-restart-gate/conclusion.md`

- [ ] Execute all 12 system/arm cases at seed 45 and 20,000 total FE per arm.
- [ ] Verify raw-evidence hash, exact purpose closure, `unattributed=0`, shared bootstrap and complete matrix.
- [ ] Compute final energy, gain AUC, actions, archive coverage and component costs without scalarization.
- [ ] Apply the strict two-of-three S-CR1 rule.
- [ ] If the gate fails, record the mechanism and close Tasks 5-6 without code expansion.
- [ ] Commit S-CR1 evidence independently.

### Task 5: Conditional S-CR2 repeat gate

**Files:**
- Extend: `runs/20260801-paired-continuation-restart-gate/run_gate.py`
- Extend after execution: `runs/20260801-paired-continuation-restart-gate/evidence.json`
- Extend after execution: `runs/20260801-paired-continuation-restart-gate/conclusion.md`

- [ ] Execute only if S-CR1 admits it.
- [ ] Run seeds 46-47 for UCB-like, Metropolis and paired best/uniform with exact shared bootstrap.
- [ ] Verify all 18 new cases and aggregate nine system-seed blocks.
- [ ] Apply the preregistered six-of-nine and positive-median rule against both comparators.
- [ ] Record system reversals as evidence against a universal selector, not as a prompt for per-system tuning.
- [ ] Commit repeated evidence independently.

### Task 6: Conditional family-posterior admission audit

**Files:**
- Create only if S-CR2 passes: `runs/20260801-paired-continuation-restart-gate/posterior_admission.md`

- [ ] Separate continuation-lane and restart-lane outcome/cost distributions from the recorded paired policy.
- [ ] Test whether zero-new-FE pre-action context predicts lane productivity under leave-system-out validation.
- [ ] Admit only a later two-family posterior experiment if both outcome separability and held-out prediction pass.
- [ ] Do not implement TS, UCB, PCA, MACE-feature learning or node posteriors in this plan.

### Task 7: Verify, reconcile and publish

**Files:**
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] Reconcile the selector result with the direction and uphill mechanism closures.
- [ ] Run all focused selector, walker, exploration, accounting and evidence checks.
- [ ] Run `git diff --check` and confirm only intended tracked changes.
- [ ] Push `feature/direction-continuation-ablation` and update PR #14 with the claim ceiling.
