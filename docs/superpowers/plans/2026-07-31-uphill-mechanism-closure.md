# Uphill Mechanism Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining causal gaps in the serial Gaussian uphiller before changing selectors or adding a new propagator.

**Architecture:** Preserve the production walk and add only observer telemetry first. Reuse frozen C60/PdO proposal prefixes for one-factor replays, then run a small fixed-starter horizon gate only if the earlier mechanisms remain unresolved.

**Tech Stack:** Python, pytest, ASE, MACE, CUDA, purpose-resolved force accounting.

## Global Constraints

- Do not change production defaults during mechanism experiments.
- Freeze starter, direction, optimizer, quench protocol, and random seed inside every paired replay.
- Require force-convergence certificates and exact purpose-accounting closure.
- Do not promote an arm from escape energy alone.
- Stop before PdO expansion when C60 has no stable paired mechanism signal.
- Do not add selector, TS/UCB, OPES, CCQN, or continuous tuning parameters.

---

### Task 1: Lossless Uphill Termination and Control Telemetry

**Files:**
- Modify: `pamssw/walker.py`
- Test: `tests/unit/test_walker_policy.py`

**Interfaces:**
- Produces diagnostic fields in `SurfaceWalker.stats()` and direction records.
- Does not change selected directions, coordinates, calculator calls, or optimizer calls.

- [x] Write failing tests for walk-radius clipping, normal step-cap termination, and requested-versus-executed controls.
- [x] Run the focused tests and confirm they fail because termination reasons and control telemetry are absent.
- [x] Add a single walk-termination recorder and fields for requested/configured/executed sigma, base/final weight, cap status, and true/inner curvature.
- [x] Run the focused tests and the complete unit baseline.
- [ ] Commit the observer-only change.

### Task 2: U4-0 Proposal-Relaxation Censor Audit

**Files:**
- Create: `runs/20260731-uphill-mechanism-closure/run_u4_0.py`
- Create: `runs/20260731-uphill-mechanism-closure/protocol.py`
- Create after execution: `runs/20260731-uphill-mechanism-closure/u4_0_evidence.json`
- Create after execution: `runs/20260731-uphill-mechanism-closure/u4_0_conclusion.md`
- Test: `tests/unit/test_uphill_mechanism_closure_protocol.py`

**Interfaces:**
- Consumes frozen C60 prefixes that stopped at proposal `maxiter=80`.
- Produces paired 80-step endpoint versus continuation-to-300 evidence on the identical modified PES.

- [ ] Write failing protocol tests for frozen-prefix identity, cost closure, and certificate classification.
- [ ] Implement the minimal replay protocol and verify tests.
- [ ] Run at most six C60 prefixes on CUDA.
- [ ] Stop if continuation changes no certificate or physical endpoint beyond replay noise.
- [ ] Write and commit the evidence-backed conclusion.

### Task 3: U4-A Cumulative Gaussian History

**Files:**
- Extend: `runs/20260731-uphill-mechanism-closure/protocol.py`
- Extend: `runs/20260731-uphill-mechanism-closure/run_u4_0.py`
- Create after execution: `runs/20260731-uphill-mechanism-closure/u4_a_evidence.json`
- Create after execution: `runs/20260731-uphill-mechanism-closure/u4_a_conclusion.md`
- Test: `tests/unit/test_uphill_mechanism_closure_protocol.py`

**Interfaces:**
- Compares the current cumulative Gaussian prefix with newest-Gaussian-only.
- Reuses the identical frozen state, newest center/direction/sigma/weight, LS term, optimizer, and budget.

- [ ] Write a failing test proving the two arms differ only in historical bias retention.
- [ ] Implement the two-arm task constructor and verify tests.
- [ ] Run the preregistered late C60 prefix cohort with a hard aggregate cap.
- [ ] Promote cumulative history only if it is certificate-neutral and Pareto-nondominated on landing outcome and force cost.
- [ ] Write and commit the evidence-backed conclusion.

### Task 4: U4-B Proposal Local-Softening Scope

**Files:**
- Extend: `runs/20260731-uphill-mechanism-closure/protocol.py`
- Extend: `runs/20260731-uphill-mechanism-closure/run_u4_0.py`
- Create after execution: `runs/20260731-uphill-mechanism-closure/u4_b_evidence.json`
- Create after execution: `runs/20260731-uphill-mechanism-closure/u4_b_conclusion.md`
- Test: `tests/unit/test_uphill_mechanism_closure_protocol.py`

**Interfaces:**
- Compares current proposal LS with oracle-only LS for an already selected frozen direction.
- Keeps all Gaussian history and optimizer inputs identical.

- [ ] Write a failing test proving only the proposal softening component changes.
- [ ] Implement the two-arm task constructor and verify tests.
- [ ] Run the same bounded C60 cohort.
- [ ] Retain proposal LS only if it is certificate-neutral and improves paired landing outcomes without higher force cost.
- [ ] Write and commit the evidence-backed conclusion.

### Task 5: U5 Horizon Capacity Gate

**Files:**
- Create: `runs/20260731-uphill-horizon-gate/run_gate.py`
- Create after execution: `runs/20260731-uphill-horizon-gate/evidence.json`
- Create after execution: `runs/20260731-uphill-horizon-gate/conclusion.md`
- Test: `tests/unit/test_uphill_horizon_gate_protocol.py`

**Interfaces:**
- Compares `H=8` and `H=14` under the surviving U4 mechanism.
- Strictly quenches checkpoints 8 and 14; it does not shoot every checkpoint.

- [ ] Write failing tests for paired configuration diffs, endpoint certificates, termination-reason completeness, and purpose closure.
- [ ] Implement and verify the fixed-starter C60 gate.
- [ ] Run two locked starters by three seeds with a 12,000-FE aggregate ceiling.
- [ ] Add a no-walk-ball replay only for radius-censored pairs; do not scan radius values.
- [ ] Write the final mechanism conclusion and production-default decision.
