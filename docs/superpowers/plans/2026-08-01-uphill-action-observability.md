# Uphill Action Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add zero-extra-force-evaluation, typed action histories that connect already-computed true-PES micro-step heights to landing-quench cost and basin outcomes.

**Architecture:** Immutable records live in `pamssw.result`; `_walk_candidate_from_seed()` fills a caller-provided trace sink while preserving its `State` return contract; `CandidateProposal` carries the trace through the unchanged true-quench loop; `SearchResult.action_history` exposes completed and rejected actions. `landing_converged` means that the returned true-quench force norm satisfies the configured force certificate, independently of optimizer status. A run-local analyzer serializes and diagnoses one frozen three-system observational cohort.

**Tech Stack:** Python dataclasses, NumPy, pytest, existing `EvalCounter` purpose snapshots, ASE/MACE CUDA, JSON.

---

### Task 1: Immutable physical records

**Files:**
- Modify: `pamssw/result.py`
- Create: `tests/unit/test_action_history.py`

- [ ] **Step 1: Write failing validation and derived-quantity tests**

Define tests that import missing `UphillStepRecord`, `UphillWalkTrace` and
`ActionRecord`. A two-step trace with target 0.8 eV and endpoint energies
`[-10.0, -9.6, -9.2]` must report observed maximum height 0.8 eV, terminal
height 0.8 eV and delivery ratio 1.0. Reject a non-positive target, unordered
step indices, negative purpose counts, inconsistent escape energy, and a
landing outcome that claims both new and duplicate.

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_action_history.py
```

Expected: import failure for the three missing types.

- [ ] **Step 2: Implement minimal frozen dataclasses**

Add:

```python
@dataclass(frozen=True)
class UphillStepRecord:
    step_index: int
    direction_kind: str
    target_eV: float
    true_energy_before_eV: float
    true_energy_after_eV: float
    requested_sigma: float
    executed_sigma: float
    base_bias_weight: float
    final_bias_weight: float
    true_curvature: float
    inner_curvature: float
    proposal_relax_iterations: int
    proposal_relax_outcome: str
    proposal_relax_termination: str
    direction_oracle_force_evaluations: int
    biased_relax_force_evaluations: int
    true_pes_check_force_evaluations: int
    displacement_clipped: bool
    step_termination_reason: str

@dataclass(frozen=True)
class UphillWalkTrace:
    target_eV: float
    termination_reason: str
    steps: tuple[UphillStepRecord, ...]
    observed_max_height_eV: float | None = field(init=False)
    observed_terminal_height_eV: float | None = field(init=False)
    target_delivery_ratio: float | None = field(init=False)

@dataclass(frozen=True)
class ActionRecord:
    trial_index: int
    proposal_index: int
    seed_entry_id: int
    seed_energy_eV: float
    walk: UphillWalkTrace
    escape_energy_eV: float | None
    landing_energy_eV: float | None
    landing_gradient_norm: float | None
    landing_iterations: int | None
    landing_converged: bool | None
    landing_force_evaluations: int
    accepted_new_basin: bool | None
    is_duplicate: bool | None
    global_improved: bool | None
    status: str
```

Extend `SearchResult` with `action_history: list[ActionRecord]` using a default
factory so all existing constructors remain source compatible.

- [ ] **Step 3: Run focused tests and commit**

Expected: the new tests pass and existing result consumers remain green.

```bash
git add pamssw/result.py tests/unit/test_action_history.py
git commit -m "Define uphill action records"
```

### Task 2: Capture completed micro-step endpoints without new evaluations

**Files:**
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_action_history.py`
- Test: `tests/integration/test_epam_accounting.py`

- [ ] **Step 1: Write a failing analytic walk-trace test**

Reuse a deterministic one-dimensional calculator and the existing monkeypatch
style in `test_walker_policy.py`. Pass `trace_sink=[]` to
`_walk_candidate_from_seed()`. Assert one `UphillWalkTrace`, ordered steps,
exact `true_before/true_after`, exact purpose deltas and unchanged returned
state. Snapshot total and purpose counts and assert the known pre-trace values.

- [ ] **Step 2: Implement the trace sink**

Add optional keyword-only
`trace_sink: list[UphillWalkTrace] | None = None` to
`_walk_candidate_from_seed()`. Snapshot counts at the start and end of every
completed micro step, build `UphillStepRecord` only from existing values, and
append exactly one `UphillWalkTrace` at walk termination. Keep every existing
calculator call, branch and RNG draw in its original order.

Determine `step_termination_reason` as one of `continued`,
`reached_step_cap`, `walk_displacement_clipped`, or the existing early-stop
reason. Walks that fail before `true_after` retain an empty or partial step
tuple and still expose the existing terminal reason.

- [ ] **Step 3: Verify exact accounting invariance**

Run:

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_action_history.py \
  tests/unit/test_walker_policy.py \
  tests/integration/test_epam_accounting.py
```

Expected: all pass, with no change to existing FE assertions.

- [ ] **Step 4: Commit**

```bash
git add pamssw/walker.py tests/unit/test_action_history.py
git commit -m "Capture uphill walk traces"
```

### Task 3: Join walk traces to landing outcomes

**Files:**
- Modify: `pamssw/walker.py`
- Modify: `tests/unit/test_action_history.py`

- [ ] **Step 1: Write failing full-search action-history tests**

Cover four outcomes with existing fake relaxers and archives:

1. new minimum and global improvement;
2. duplicate landing;
3. fragment or energy-sanity rejection;
4. landing-quench budget exhaustion.

Assert proposal ordering, seed/landing energies, exact landing-purpose delta,
walk termination, and that every completed proposal yields exactly one action
record. The budget-exhausted action has `landing_energy_eV=None` and preserves
the existing search stop.

- [ ] **Step 2: Carry traces through `CandidateProposal`**

Add `walk_trace: UphillWalkTrace | None = None` to `CandidateProposal`.
`_proposal_pool()` provides a fresh sink for each proposal and attaches the
single returned trace. Duplicate rescue uses the same path.

- [ ] **Step 3: Append `ActionRecord` in the true-quench loop**

Initialize `action_history=[]` beside `walk_history`. Snapshot
`LANDING_TRUE_QUENCH` immediately before and after each unchanged
`relax_true_minimum()` call. Append one record in the normal, duplicate,
fragment-rejected, energy-sanity-rejected and budget-exhausted branches.
Return the list through `SearchResult`.

- [ ] **Step 4: Verify and commit**

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_action_history.py \
  tests/unit/test_walker_policy.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw.py
git diff --check
git add pamssw/walker.py pamssw/result.py tests/unit/test_action_history.py
git commit -m "Record complete SSW action outcomes"
```

### Task 4: Frozen observational runner and analyzer

**Files:**
- Create: `runs/20260801-uphill-action-observability/.gitignore`
- Create: `runs/20260801-uphill-action-observability/PLAN.md`
- Create: `runs/20260801-uphill-action-observability/analyze.py`
- Create: `runs/20260801-uphill-action-observability/run_gate.py`
- Create: `tests/unit/test_uphill_action_observability.py`

- [ ] **Step 1: Write failing pure-analysis tests**

Test serialization of `ActionRecord` without coordinates, continuous delivery
statistics, the exact `ratio >= 1` split, conditional new/duplicate/global
improvement rates, landing-quench FE and quench energy drop. Reject missing
actions, duplicate `(system, seed, trial, proposal)` keys, non-closed ledgers
and unattributed work.

- [ ] **Step 2: Implement the pure analyzer**

The analyzer consumes three case summaries plus action JSONL files and emits:

- per-system action count and target distribution;
- observed maximum/terminal height and delivery-ratio quantiles;
- attained versus unattained basin outcomes;
- landing-quench FE and energy-drop distributions;
- walk termination, proposal-relax outcome and zero-step failure taxonomy;
- component FE totals and raw evidence hashes.

Do not fit a classifier or choose a new target.

- [ ] **Step 3: Implement the runner by reusing frozen resources**

Load the fixed-target gate resource builder. Use archive-scaled target,
Metropolis starter, seed 49, 20,000 FE, and unchanged production profiles for
C60, PdO and CuO. Serialize `result.action_history` after each case. Record
execution commit, structure/model hashes, CUDA runtime, effective config,
purpose ledger and `unattributed=0`.

- [ ] **Step 4: Run unit tests and a 1,000-FE C60 CUDA smoke**

The smoke must preserve exact budget/accounting and produce at least one valid
action record when a proposal completes. Output remains ignored.

- [ ] **Step 5: Commit the runner before production execution**

```bash
git add runs/20260801-uphill-action-observability \
  tests/unit/test_uphill_action_observability.py
git commit -m "Add uphill observability GPU gate"
```

### Task 5: Execute U-O1 and close the physical diagnosis

**Files:**
- Create: `runs/20260801-uphill-action-observability/evidence.json`
- Create: `runs/20260801-uphill-action-observability/conclusion.md`
- Modify: `docs/research/2026-07-31-review-reconciled-roadmap.md`

- [ ] **Step 1: Execute the exact 60,000-FE cohort**

Run C60, PdO and CuO seed 49 at 20,000 FE per case on CUDA. Require exact
purpose closure or an atomic-batch residual smaller than one submitted HVP
batch, and `unattributed=0`.

- [ ] **Step 2: Apply the diagnosis boundary**

Report all three systems separately. Diagnosis 1 requires the majority of
actions in at least two systems to have `target_delivery_ratio < 1` at observed
endpoints. Diagnosis 2 requires delivered actions to be the majority in at
least two systems while most delivered actions are duplicates/rejections or
landing quench is the largest non-proposal action cost. Otherwise record
diagnosis 3, mixed scalar-target evidence. These are diagnostic labels, not
promotion of a controller.

- [ ] **Step 3: Write the evidence-bounded conclusion**

State explicitly that observed endpoint height is not a barrier, one seed is
not significance evidence, and the result cannot yet train a posterior model.
Choose only the next mechanism gate implied by the diagnosis; do not implement
it in this stage.

- [ ] **Step 4: Verify, commit, push and update PR #14**

```bash
/root/miniforge3/envs/mace_les/bin/python -m pytest -q \
  tests/unit/test_action_history.py \
  tests/unit/test_uphill_action_observability.py \
  tests/unit/test_walker_policy.py \
  tests/unit/test_accounting.py \
  tests/integration/test_epam_accounting.py \
  tests/integration/test_ssw.py
git diff --check
git add runs/20260801-uphill-action-observability \
  docs/research/2026-07-31-review-reconciled-roadmap.md
git commit -m "Close uphill action observability gate"
git push pam feature/direction-continuation-ablation
```
