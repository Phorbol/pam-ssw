# Fixed Proposal Energy Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record zero-extra-call energy and force traces for the frozen C60/PdO proposal tasks and explain FIRE versus safe L-BFGS cost.

**Architecture:** Keep all instrumentation in one run-local module. A
`RecordingProposalPotential` observes the existing component evaluation and a
trajectory callback labels evaluated coordinate hashes that became optimizer
states. The runner verifies every trace against the prior cap-400 ledger before
writing scientific summaries.

**Tech Stack:** Python, NumPy, ASE/MACE through existing calculator factories,
pytest, JSON.

---

### Task 1: Test the zero-extra-call recorder

**Files:**
- Create: `runs/20260727-proposal-energy-traces/trace_recorder.py`
- Create: `tests/unit/test_proposal_energy_trace.py`

- [ ] **Step 1: Write failing tests**

Test an analytic `ProposalPotential` with one Gaussian bias. Require that one
recorder call produces one underlying calculator call, correct component
energies, correct active maximum force, and a deterministic position hash.
Also require trajectory-state labelling to mark only matching evaluated
positions.

- [ ] **Step 2: Verify red**

Run:

```bash
pytest -q tests/unit/test_proposal_energy_trace.py
```

Expected: failure because `trace_recorder.py` does not exist.

- [ ] **Step 3: Implement the recorder**

Implement `RecordingProposalPotential`, `position_hash`, and
`mark_accepted_state_evaluations`. The override must call the parent method
once and must not access the calculator otherwise.

- [ ] **Step 4: Verify green**

Run:

```bash
pytest -q tests/unit/test_proposal_energy_trace.py
```

Expected: all recorder tests pass.

- [ ] **Step 5: Commit**

```bash
git add runs/20260727-proposal-energy-traces/trace_recorder.py tests/unit/test_proposal_energy_trace.py
git commit -m "experiment: record proposal energies without extra calls"
```

### Task 2: Build and validate the frozen GPU replay

**Files:**
- Create: `runs/20260727-proposal-energy-traces/run_gpu_traces.py`
- Create: `runs/20260727-proposal-energy-traces/plan.md`

- [ ] **Step 1: Add a failing analytic integration test**

Add a test that runs FIRE and safe L-BFGS through the recorder and requires
`recorded_evaluations == EvalCounter.total == telemetry.evaluator_calls`.

- [ ] **Step 2: Verify red**

Run the new integration test and confirm it fails because the replay helper is
missing.

- [ ] **Step 3: Implement the run-local replay helper and GPU driver**

Load frozen task payloads from a required `--source-summary` argument, apply the
shared cap of 400, warm independent calculators, replay FIRE and safe L-BFGS,
and fail closed on any mismatch with the prior cap-400 summary.

- [ ] **Step 4: Run unit and full tests**

```bash
pytest -q tests/unit/test_proposal_energy_trace.py
pytest -q
```

Expected: recorder tests and the full suite pass.

- [ ] **Step 5: Commit the frozen experiment**

```bash
git add runs/20260727-proposal-energy-traces tests/unit/test_proposal_energy_trace.py
git commit -m "experiment: define fixed GPU proposal energy traces"
```

### Task 3: Execute and analyze the real GPU matrix

**Files:**
- Create: `runs/20260727-proposal-energy-traces/analyze_traces.py`
- Create: `runs/20260727-proposal-energy-traces/conclusion.md`
- Create: `runs/20260727-proposal-energy-traces/evidence.json`

- [ ] **Step 1: Commit the analysis contract**

The analyzer must report per task/backend calls, accepted-state evaluations,
non-accepted evaluations, energy/force curve points, certificate status,
endpoint displacement, and source hashes. Commit it before running GPU work.

- [ ] **Step 2: Run the frozen GPU matrix**

Use the `mace_les` environment, CUDA float32, MACE-OMAT-0-small, and the raw
fixed-task/cap-400 summaries from the preceding worktree.

- [ ] **Step 3: Validate integrity**

Require all 32 task/backend traces to match the existing cap-400 calls and
endpoints, contain only finite values, and close exact accounting.

- [ ] **Step 4: Write the conclusion**

Separate verified trace facts, endpoint-equivalence limitations, physical
interpretation, and any next ablation. Do not tune an optimizer or promote a
default.

- [ ] **Step 5: Re-run full verification and commit evidence**

Run the full suite and `git diff --check`, then commit only compact evidence and
the conclusion. Keep large raw trace files local unless their size remains
reviewable.
