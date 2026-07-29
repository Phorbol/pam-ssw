# Posterior Starter-Policy GPU Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed real-C60/PdO GPU harness that compares only the three existing outer starter policies under a serial, exact-force-accounted, production-derived unsoftened SSW action.

**Architecture:** The runner projects the existing 200-step LS-SSW production configuration onto `SSWConfig`, disables all shared filesystem output and then calls `run_posterior_ssw`.  A harness-local calculator factory uses one short-lived main-thread calculator for bootstrap and one worker-thread-local calculator for every serial action, while the existing runner retains its independent `EvalCounter` for bootstrap and each action.

**Tech Stack:** Python, pytest, ASE, MACE CUDA calculator, `pamssw.exploration.run_posterior_ssw`.

---

### Task 1: Test configuration projection and strict experiment identity

**Files:**

- Create: `tests/unit/test_posterior_starter_policy_gpu_ablation.py`
- Create: `runs/20260728-posterior-starter-policy-gpu-ablation/run_ablation.py`

- [x] **Step 1: Write failing tests**

```python
def test_production_config_projects_to_unsoftened_ssw_without_worker_side_effects(tmp_path):
    config, provenance = runner.build_ssw_config("c60", tmp_path)
    assert type(config) is SSWConfig
    assert provenance["softening_enabled"] is False
    assert config.proposal_pool_size == 1
    assert config.accepted_structures_log is None
    assert config.direction_diagnostics_enabled is False

def test_preflight_rejects_nonserial_or_unknown_policy(tmp_path):
    with pytest.raises(ValueError):
        runner.build_exploration_config("uniform", tmp_path, batch_size=2, max_workers=1)
```

- [x] **Step 2: Run tests and observe missing-module failure**

Run: `pytest -q tests/unit/test_posterior_starter_policy_gpu_ablation.py`

Expected: fail because the harness module does not yet exist.

- [x] **Step 3: Implement minimal projection and immutable preflight manifest**

```python
source = production.build_config(system, scratch_directory)
values = {item.name: getattr(source, item.name) for item in fields(SSWConfig)}
config = SSWConfig(**(values | disabled_output_fields))
```

The preflight must reject a dirty worktree, commit mismatch, unavailable CUDA, wrong policy set, non-serial dispatch, missing model/input files, and an invalid projected config before it creates output or constructs a calculator.

- [x] **Step 4: Re-run the focused projection tests**

Run: `pytest -q tests/unit/test_posterior_starter_policy_gpu_ablation.py`

Expected: pass.

### Task 2: Test thread-owned evaluator reuse and action telemetry wrapper

**Files:**

- Modify: `tests/unit/test_posterior_starter_policy_gpu_ablation.py`
- Modify: `runs/20260728-posterior-starter-policy-gpu-ablation/run_ablation.py`

- [x] **Step 1: Write failing tests**

```python
def test_thread_owned_factory_creates_one_bootstrap_and_one_worker_calculator():
    factory = runner.ThreadOwnedCalculatorFactory(build_calculator)
    bootstrap = factory()
    with ThreadPoolExecutor(max_workers=1) as executor:
        action_one = executor.submit(factory).result()
        action_two = executor.submit(factory).result()
    assert bootstrap is not action_one
    assert action_one is action_two
    assert factory.snapshot()["bootstrap_instances"] == 1
    assert factory.snapshot()["action_instances"] == 1
```

- [x] **Step 2: Run the new test and observe missing-class failure**

Run: `pytest -q tests/unit/test_posterior_starter_policy_gpu_ablation.py::test_thread_owned_factory_creates_one_bootstrap_and_one_worker_calculator`

Expected: fail because `ThreadOwnedCalculatorFactory` is not defined.

- [x] **Step 3: Implement a serial-only factory and non-invasive timing worker**

The bootstrap calculator is never cached across threads.  The sole executor thread caches exactly one action calculator.  The timing wrapper delegates unchanged `StarterAction` and `State` values to `SSWAttemptWorker`, records duration in a thread-safe map, and cannot change the returned `AttemptResult`.

- [x] **Step 4: Re-run focused tests**

Run: `pytest -q tests/unit/test_posterior_starter_policy_gpu_ablation.py`

Expected: pass.

### Task 3: Run campaigns and fail closed on accounting

**Files:**

- Modify: `tests/unit/test_posterior_starter_policy_gpu_ablation.py`
- Modify: `runs/20260728-posterior-starter-policy-gpu-ablation/run_ablation.py`

- [x] **Step 1: Write failing tests for artifact closure with an analytic runner double**

```python
def test_campaign_summary_closes_exact_ledgers_and_emits_one_action_metric_per_event(tmp_path):
    summary = runner.run_campaign(..., run_posterior=fake_run_posterior_ssw)
    assert summary["total_evaluations"] + summary["unused_force_budget"] == 200
    assert summary["purpose_counts"]["unattributed"] == 0
    assert len(read_jsonl(summary["action_metrics_path"])) == summary["completed_attempts"] + summary["failed_attempts"]
```

- [x] **Step 2: Run closure test and observe missing API failure**

Run: `pytest -q tests/unit/test_posterior_starter_policy_gpu_ablation.py::test_campaign_summary_closes_exact_ledgers_and_emits_one_action_metric_per_event`

Expected: fail because campaign execution and artifact analysis are not defined.

- [x] **Step 3: Implement smoke-parameterized campaign execution**

For any caller-supplied total/action budgets and seeds, run only `uniform`, `posterior_proportional`, and `minimal_ucb`; write a manifest, raw event log, optimizer diagnostics, action metrics, campaign summary, and top-level index.  Validate exact cost, zero unattributed FE, event-log replay, serial snapshots, and one timing record per action before publishing final JSON.

- [x] **Step 4: Run focused test suite and static checks**

Run: `pytest -q tests/unit/test_posterior_starter_policy_gpu_ablation.py && python -m compileall -q runs/20260728-posterior-starter-policy-gpu-ablation && git diff --check`

Expected: all pass.

### Task 4: Commit preflight-only harness

**Files:**

- Create: `runs/20260728-posterior-starter-policy-gpu-ablation/run_ablation.py`
- Create: `tests/unit/test_posterior_starter_policy_gpu_ablation.py`
- Create: `docs/superpowers/plans/2026-07-28-posterior-starter-policy-gpu-ablation.md`

- [x] **Step 1: Inspect tracked diff and retain unrelated direction-oracle files untouched**

Run: `git status --short && git diff --check`

Expected: only harness, test, and plan are staged; direction-oracle untracked files remain untouched.

- [x] **Step 2: Commit only the harness implementation and its tests**

```bash
git add runs/20260728-posterior-starter-policy-gpu-ablation/run_ablation.py \
        tests/unit/test_posterior_starter_policy_gpu_ablation.py \
        docs/superpowers/plans/2026-07-28-posterior-starter-policy-gpu-ablation.md
git commit -m "experiment: add posterior starter-policy GPU harness"
```
