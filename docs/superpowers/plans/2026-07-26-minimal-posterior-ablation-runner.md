# Minimal Posterior Ablation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the existing exact-accounting exploration components into one minimal analytic `ThreadPoolExecutor` SSW/LS-SSW campaign runner and one raw paired-seed three-policy ablation harness.

**Architecture:** Keep `SurfaceWalker`, the three Phase-1 policies, and all numerical defaults unchanged. Add one thin orchestration module that performs a budgeted bootstrap true quench, constructs the existing controller/worker/ledger, runs complete synchronous batches, and returns the existing campaign result. Retain the compact event log for audit facts, but add no manifest, resume, journal replay, or recovery layer.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, `concurrent.futures.ThreadPoolExecutor`, existing analytic calculators, JSON, pytest.

---

## Supersession and scope

This plan implements
`docs/superpowers/specs/2026-07-26-minimal-posterior-ablation-runner-design.md`.
It replaces Tasks 6–11 of
`docs/superpowers/plans/2026-07-25-recoverable-budgeted-posterior-runner.md`.
The earlier plan's Tasks 1–5 remain complete and are reused unchanged except
for removal of three now-unused recovery configuration fields.

Do not add:

- `run_store.py`, manifests, sessions, resume, replay, or atomic batch files;
- a new policy, posterior, reward, direction source, or numerical parameter;
- an async/process/GPU executor;
- a benchmark winner or performance claim.

## File map

- Modify `pamssw/exploration/campaign.py`
  - Remove configuration fields used only by the cancelled recovery design.
- Create `pamssw/exploration/runner.py`
  - Implement bootstrap and the complete minimal campaign loop.
- Modify `pamssw/exploration/__init__.py`
  - Export the two runner functions.
- Modify `pamssw/__init__.py`
  - Export the opt-in public functions without changing legacy runners.
- Modify `tests/unit/test_exploration_campaign.py`
  - Lock the reduced configuration contract.
- Create `tests/integration/test_posterior_exploration_runner.py`
  - Validate real analytic bootstrap, ThreadPool execution, exact accounting,
    terminal summaries, and reproducibility.
- Create `benchmarks/posterior_policy_compare.py`
  - Run the three fixed policies with paired seeds and emit raw JSON.
- Create `tests/unit/test_posterior_policy_compare.py`
  - Validate harness pairing and output schema without ranking policies.
- Modify `README.md`
  - Replace obsolete Phase-1 runner boundaries with the exact minimal claim.
- Modify `docs/superpowers/plans/2026-07-25-recoverable-budgeted-posterior-runner.md`
  - Add a visible supersession notice before old Task 6.

---

### Task 1: Remove recovery-only campaign configuration

**Files:**
- Modify: `pamssw/exploration/campaign.py`
- Modify: `tests/unit/test_exploration_campaign.py`

- [ ] **Step 1: Write the failing reduced-contract tests**

Change the shared configuration fixture to:

```python
def _config_values(tmp_path):
    return {
        "policy_name": "uniform",
        "batch_size": 3,
        "max_workers": 2,
        "action_force_budget": 10,
        "total_force_budget": 101,
        "master_seed": 7,
        "run_directory": tmp_path / "run",
    }
```

Require the exact dataclass field set:

```python
def test_posterior_exploration_config_contains_only_executed_campaign_fields(tmp_path):
    config = PosteriorExplorationConfig(**_config_values(tmp_path))

    assert tuple(field.name for field in fields(config)) == (
        "policy_name",
        "batch_size",
        "max_workers",
        "action_force_budget",
        "total_force_budget",
        "master_seed",
        "run_directory",
    )
```

Delete tests for `mode`, `calculator_label`, and `calculator_fingerprint`.
Retain all validation tests for policy, strict positive integers, worker/batch
relation, seed, and path-like run directory.

- [ ] **Step 2: Run the configuration tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_exploration_campaign.py -k "config"
```

Expected: failure because the dataclass still contains the three cancelled
recovery fields.

- [ ] **Step 3: Remove only the unused fields**

Make the public configuration:

```python
@dataclass(frozen=True)
class PosteriorExplorationConfig:
    policy_name: str
    batch_size: int
    max_workers: int
    action_force_budget: int
    total_force_budget: int
    master_seed: int
    run_directory: Path
```

Keep the existing validation and `Path` normalization for these seven fields.
Remove `_stripped_nonempty` only if no remaining code uses it.

- [ ] **Step 4: Run the complete campaign tests**

Run:

```bash
pytest -q tests/unit/test_exploration_campaign.py tests/unit/test_accounting.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add pamssw/exploration/campaign.py tests/unit/test_exploration_campaign.py
git commit -m "Reduce exploration campaign configuration"
```

---

### Task 2: Implement exact bootstrap as a private runner boundary

**Files:**
- Create: `pamssw/exploration/runner.py`
- Create: `tests/integration/test_posterior_exploration_runner.py`

- [ ] **Step 1: Write failing bootstrap success and accounting tests**

Use the real analytic backend:

```python
def _state() -> State:
    return State(
        numbers=np.array([1]),
        positions=np.array([[-0.8, 0.0, 0.0]]),
    )


def _calculator_factory():
    return AnalyticCalculator(DoubleWell2D())
```

Test the private bootstrap boundary directly:

```python
def test_bootstrap_relaxes_raw_state_and_records_exact_purposes():
    state, energy, counts = _bootstrap_minimum(
        _state(),
        _calculator_factory,
        SSWConfig(quench_maxiter=40),
        total_force_budget=100,
    )

    assert isinstance(state, State)
    assert np.isfinite(energy)
    assert counts.total > 0
    assert counts.count(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH) > 0
    assert counts.count(EvaluationPurpose.POST_RELAX_VALIDATION) == 1
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert sum(counts.values) == counts.total
```

- [ ] **Step 2: Write failing bootstrap failure tests**

Cover:

```python
def test_bootstrap_rejects_invalid_raw_state_before_factory_call():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _calculator_factory()

    invalid = State(
        numbers=np.array([1]),
        positions=np.array([[np.nan, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="initial_state"):
        _bootstrap_minimum(invalid, factory, SSWConfig(), total_force_budget=20)
    assert calls == 0


def test_bootstrap_budget_exhaustion_is_not_converted_to_a_result():
    with pytest.raises(BudgetExceeded):
        _bootstrap_minimum(
            _state(),
            _calculator_factory,
            SSWConfig(quench_maxiter=40),
            total_force_budget=1,
        )
```

Also reject a non-callable factory, a factory product without callable
`evaluate`/`evaluate_flat`, a non-`State`, a wrong SSW config type, and a
nonpositive/boolean total budget.

- [ ] **Step 3: Run bootstrap tests and verify RED**

Run:

```bash
pytest -q tests/integration/test_posterior_exploration_runner.py -k "bootstrap"
```

Expected: import failure because `pamssw.exploration.runner` does not exist.

- [ ] **Step 4: Implement the private bootstrap helper**

Create:

```python
def _bootstrap_minimum(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    *,
    total_force_budget: int,
) -> tuple[State, float, EvaluationCounts]:
    if not isinstance(initial_state, State):
        raise TypeError("initial_state must be a State")
    if not GeometryValidator().is_valid_state(initial_state):
        raise ValueError("initial_state has invalid geometry")
    if not callable(calculator_factory):
        raise TypeError("calculator_factory must be callable")
    if not isinstance(ssw_config, SSWConfig):
        raise TypeError("ssw_config must be an SSWConfig")
    total_force_budget = _positive_int(total_force_budget, "total_force_budget")

    calculator = calculator_factory()
    if not callable(getattr(calculator, "evaluate", None)) or not callable(
        getattr(calculator, "evaluate_flat", None)
    ):
        raise TypeError("calculator_factory must return a calculator")
    counter = EvalCounter(calculator, max_force_evals=total_force_budget)
    relaxer = Relaxer(counter.evaluate_flat, optimizer=ssw_config.quench_optimizer)
    with counter.purpose(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH):
        relaxed = relaxer.relax(
            initial_state,
            fmax=ssw_config.quench_fmax,
            maxiter=ssw_config.quench_maxiter,
        )
    with counter.purpose(EvaluationPurpose.POST_RELAX_VALIDATION):
        valid = GeometryValidator().is_valid_evaluation(relaxed.state, counter)
    if not valid or not np.isfinite(relaxed.energy):
        raise ValueError("bootstrap produced an invalid minimum")
    return deepcopy(relaxed.state), float(relaxed.energy), counter.snapshot()
```

Define one local strict-positive-integer validator; do not import a private
validator from `campaign.py`.

- [ ] **Step 5: Run bootstrap and nearby integration tests**

Run:

```bash
pytest -q \
  tests/integration/test_posterior_exploration_runner.py -k "bootstrap" \
  tests/integration/test_epam_accounting.py \
  tests/unit/test_relax.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add pamssw/exploration/runner.py \
  tests/integration/test_posterior_exploration_runner.py
git commit -m "Add exact posterior campaign bootstrap"
```

---

### Task 3: Compose the minimal fixed-budget ThreadPool campaign

**Files:**
- Modify: `pamssw/exploration/runner.py`
- Modify: `pamssw/exploration/__init__.py`
- Modify: `pamssw/__init__.py`
- Modify: `tests/integration/test_posterior_exploration_runner.py`

- [ ] **Step 1: Write the failing public runner integration test**

Create a small side-effect-free SSW configuration:

```python
def _ssw_config(seed: int = 0) -> SSWConfig:
    return SSWConfig(
        rng_seed=seed,
        max_trials=1,
        max_steps_per_walk=2,
        oracle_candidates=2,
        proposal_relax_steps=3,
        quench_maxiter=40,
        proposal_pool_size=1,
        max_force_evals=None,
    )
```

Run a real ThreadPool campaign:

```python
def test_posterior_ssw_runs_real_threadpool_batches_with_exact_budget(tmp_path):
    config = PosteriorExplorationConfig(
        policy_name="uniform",
        batch_size=3,
        max_workers=2,
        action_force_budget=30,
        total_force_budget=140,
        master_seed=11,
        run_directory=tmp_path / "run",
    )

    result = run_posterior_ssw(
        _state(),
        _calculator_factory,
        _ssw_config(),
        config,
    )

    assert result.policy_name == "uniform"
    assert result.completed_batches >= 1
    assert result.total_evaluations <= config.total_force_budget
    assert result.total_evaluations == result.purpose_counts.total
    assert result.bootstrap_evaluations + result.action_evaluations == result.total_evaluations
    assert result.unused_force_budget < config.action_force_budget
    assert result.stop_reason is CampaignStopReason.BUDGET_TAIL
    assert result.purpose_counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert (config.run_directory / "events.jsonl").is_file()
```

- [ ] **Step 2: Write failing reservation, terminal, and directory tests**

Test:

- existing run directory is rejected before bootstrap factory creation;
- a missing parent directory is rejected;
- bootstrap success followed by a factory that always fails commits one exact
  zero-cost batch and returns `ZERO_COST_STALL`;
- the zero-cost result is benchmark-ineligible with an explicit reason;
- the number of actions in each parsed event-log batch equals the width
  reserved from reconstructed remaining cost;
- no action's `force_budget` differs from
  `exploration_config.action_force_budget`;
- bootstrap-only budget tail returns zero attempts and a valid result;
- `run_posterior_ls_ssw` rejects plain `SSWConfig` and accepts `LSSSWConfig`;
- `run_posterior_ssw` rejects `LSSSWConfig` to keep the public choice explicit.

- [ ] **Step 3: Write failing reproducibility and policy-path tests**

For each of:

```python
("uniform", "posterior_proportional", "minimal_ucb")
```

run the same seed/configuration in two different fresh directories and compare:

```python
def _result_fingerprint(result):
    return (
        result.policy_name,
        tuple((entry.energy, entry.parent_id) for entry in result.archive.entries),
        tuple(result.posterior.counts(entry.entry_id) for entry in result.archive.entries),
        result.completed_batches,
        result.completed_attempts,
        result.failed_attempts,
        result.posterior_observed_attempts,
        result.bootstrap_evaluations,
        result.action_evaluations,
        result.purpose_counts,
        result.unused_force_budget,
        result.stop_reason,
    )
```

Assert identical fingerprints for paired repeats. Do not compare event-log
paths or claim that different policies should produce different results.

- [ ] **Step 4: Run the public tests and verify RED**

Run:

```bash
pytest -q tests/integration/test_posterior_exploration_runner.py -k "posterior_ssw or zero_cost or reproduc"
```

Expected: failure because the public functions do not exist.

- [ ] **Step 5: Implement one private campaign loop**

Implement:

```python
def _run_posterior_campaign(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
    *,
    softening_enabled: bool,
) -> PosteriorExplorationResult:
    if not isinstance(exploration_config, PosteriorExplorationConfig):
        raise TypeError("exploration_config must be a PosteriorExplorationConfig")
    run_directory = exploration_config.run_directory
    if run_directory.exists():
        raise FileExistsError(f"run directory already exists: {run_directory}")
    if not run_directory.parent.is_dir():
        raise FileNotFoundError(
            f"run directory parent does not exist: {run_directory.parent}"
        )

    bootstrap_state, bootstrap_energy, bootstrap_counts = _bootstrap_minimum(
        initial_state,
        calculator_factory,
        ssw_config,
        total_force_budget=exploration_config.total_force_budget,
    )
    budget = CampaignBudget(
        exploration_config.total_force_budget,
        exploration_config.action_force_budget,
    )
    budget.record_bootstrap(bootstrap_counts)

    run_directory.mkdir()
    event_log = ExplorationEventLog(run_directory / "events.jsonl")
    archive = MinimaArchive(
        energy_tol=ssw_config.dedup_energy_tol,
        rmsd_tol=ssw_config.dedup_rmsd_tol,
        max_prototypes=ssw_config.max_prototypes,
    )
    archive.add(deepcopy(bootstrap_state), bootstrap_energy, parent_id=None)
    controller = ExplorationController(
        archive,
        exploration_config.policy_name,
        exploration_config.master_seed,
        event_log,
        require_exact_cost=True,
    )
    worker = SSWAttemptWorker(
        calculator_factory,
        ssw_config,
        softening_enabled=softening_enabled,
    )
    outcomes: list[CreditedOutcome] = []
    with ThreadPoolExecutor(max_workers=exploration_config.max_workers) as executor:
        while True:
            width = budget.next_batch_size(exploration_config.batch_size)
            if width == 0:
                break
            committed = controller.run_batch(
                executor,
                worker,
                width,
                exploration_config.action_force_budget,
            )
            batch_counts = tuple(outcome.evaluation_counts for outcome in committed)
            if any(not isinstance(counts, EvaluationCounts) for counts in batch_counts):
                raise RuntimeError("committed outcomes require evaluation counts")
            budget.commit_batch(batch_counts)
            outcomes.extend(committed)

    if budget.stop_reason is None:
        raise RuntimeError("campaign terminated without a stop reason")
    purpose_counts = budget.bootstrap_counts + budget.action_counts
    reasons: set[str] = set()
    if any(not outcome.posterior_observed for outcome in outcomes):
        reasons.add("non_posterior_observed_attempt")
    if purpose_counts.count(EvaluationPurpose.UNATTRIBUTED):
        reasons.add("unattributed_evaluations")
    if budget.stop_reason is CampaignStopReason.ZERO_COST_STALL:
        reasons.add("zero_cost_stall")
    completed_attempts = sum(
        outcome.status is AttemptStatus.COMPLETED for outcome in outcomes
    )
    return PosteriorExplorationResult(
        archive=controller.archive,
        posterior=controller.posterior,
        policy_name=exploration_config.policy_name,
        completed_batches=budget.committed_batches,
        completed_attempts=completed_attempts,
        failed_attempts=len(outcomes) - completed_attempts,
        posterior_observed_attempts=sum(
            outcome.posterior_observed for outcome in outcomes
        ),
        bootstrap_evaluations=budget.bootstrap_counts.total,
        action_evaluations=budget.action_counts.total,
        total_evaluations=budget.spent,
        purpose_counts=purpose_counts,
        total_force_budget=budget.total,
        unused_force_budget=budget.unused,
        stop_reason=budget.stop_reason,
        benchmark_eligible=not reasons,
        benchmark_ineligibility_reasons=tuple(sorted(reasons)),
        run_directory=run_directory,
    )
```

The implementation must perform these operations in order:

1. validate all public argument types and the exact SSW/LS-SSW pairing;
2. reject an existing run directory or missing parent without creating a
   calculator;
3. call `_bootstrap_minimum`;
4. create the run directory and `ExplorationEventLog`;
5. create `CampaignBudget`, record bootstrap counts, and initialize a
   one-entry `MinimaArchive`;
6. create `ExplorationController(require_exact_cost=True)` and
   `SSWAttemptWorker`;
7. enter one `ThreadPoolExecutor(max_workers=max_workers)`;
8. repeatedly call `next_batch_size(batch_size)`, `controller.run_batch`, and
   `budget.commit_batch(tuple(outcome.evaluation_counts for outcome in outcomes))`;
9. accumulate terminal outcomes only after a successful batch commit;
10. derive the complete `PosteriorExplorationResult`.

Use:

```python
reasons: set[str] = set()
if any(not outcome.posterior_observed for outcome in outcomes):
    reasons.add("non_posterior_observed_attempt")
if purpose_counts.count(EvaluationPurpose.UNATTRIBUTED):
    reasons.add("unattributed_evaluations")
if budget.stop_reason is CampaignStopReason.ZERO_COST_STALL:
    reasons.add("zero_cost_stall")
```

Sort reasons before passing the tuple. Strict controller mode already rejects
unknown costs; do not add another cost model.

- [ ] **Step 6: Add the two public wrappers and exports**

Implement wrappers that only select the type/softening path:

```python
def run_posterior_ssw(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    if type(ssw_config) is not SSWConfig:
        raise TypeError("run_posterior_ssw requires an SSWConfig")
    return _run_posterior_campaign(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        softening_enabled=False,
    )


def run_posterior_ls_ssw(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: LSSSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    if not isinstance(ssw_config, LSSSWConfig):
        raise TypeError("run_posterior_ls_ssw requires an LSSSWConfig")
    return _run_posterior_campaign(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        softening_enabled=True,
    )
```

Export both from `pamssw.exploration` and the top-level `pamssw` package.
Do not modify legacy `run_ssw`, `run_ls_ssw`, or `pamssw/runner.py`.

- [ ] **Step 7: Run runner, controller, adapter, and campaign suites**

Run:

```bash
pytest -q \
  tests/integration/test_posterior_exploration_runner.py \
  tests/integration/test_exploration_controller.py \
  tests/integration/test_ssw_attempt_worker_integration.py \
  tests/unit/test_exploration_campaign.py \
  tests/unit/test_ssw_attempt_worker.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add pamssw/exploration/runner.py pamssw/exploration/__init__.py \
  pamssw/__init__.py tests/integration/test_posterior_exploration_runner.py
git commit -m "Run fixed-budget posterior SSW campaigns"
```

---

### Task 4: Add the raw paired-policy analytic harness

**Files:**
- Create: `benchmarks/posterior_policy_compare.py`
- Create: `tests/unit/test_posterior_policy_compare.py`

- [ ] **Step 1: Write failing harness schema and pairing tests**

Load the script as a module and test a pure orchestration function:

```python
records = run_comparison(
    output_root=tmp_path / "runs",
    seeds=(3, 7),
    policies=("uniform", "posterior_proportional", "minimal_ucb"),
    total_force_budget=120,
    action_force_budget=30,
    batch_size=2,
    max_workers=2,
)
```

Require exactly one record for every `(seed, policy)` pair and the exact
reader-facing fields:

```text
schema_version
calculator_label
potential_parameters
seed
policy
batch_size
max_workers
action_force_budget
total_force_budget
best_energy
unique_minima
completed_batches
completed_attempts
failed_attempts
posterior_observed_attempts
bootstrap_evaluations
action_evaluations
total_evaluations
purpose_counts
unused_force_budget
stop_reason
benchmark_eligible
benchmark_ineligibility_reasons
```

Assert records are ordered by seed and then the supplied policy order. Assert
there is no `winner`, `ranking`, `p_value`, or `score` field.

- [ ] **Step 2: Run the harness tests and verify RED**

Run:

```bash
pytest -q tests/unit/test_posterior_policy_compare.py
```

Expected: failure because the harness does not exist.

- [ ] **Step 3: Implement the harness**

Use a fixed analytic `DoubleWell2D` factory and raw initial state. For every
pair, build a fresh run directory and call `run_posterior_ssw`. Convert
`EvaluationCounts.as_dict()` directly; do not collapse purpose counts.

Provide:

```python
def run_comparison(
    *,
    output_root: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...] = (
        "uniform",
        "posterior_proportional",
        "minimal_ucb",
    ),
    total_force_budget: int,
    action_force_budget: int,
    batch_size: int,
    max_workers: int,
) -> tuple[dict[str, object], ...]:
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.mkdir()
    records: list[dict[str, object]] = []
    for seed in seeds:
        for policy in policies:
            run_directory = output_root / f"seed-{seed:08d}-{policy}"
            config = PosteriorExplorationConfig(
                policy_name=policy,
                batch_size=batch_size,
                max_workers=max_workers,
                action_force_budget=action_force_budget,
                total_force_budget=total_force_budget,
                master_seed=seed,
                run_directory=run_directory,
            )
            result = run_posterior_ssw(
                State(
                    numbers=np.array([1]),
                    positions=np.array([[-0.8, 0.0, 0.0]]),
                ),
                lambda: AnalyticCalculator(DoubleWell2D()),
                SSWConfig(
                    rng_seed=0,
                    max_trials=1,
                    max_steps_per_walk=2,
                    oracle_candidates=2,
                    proposal_relax_steps=3,
                    quench_maxiter=40,
                    proposal_pool_size=1,
                    max_force_evals=None,
                ),
                config,
            )
            records.append(
                {
                    "schema_version": 1,
                    "calculator_label": "analytic-double-well-2d-v1",
                    "potential_parameters": {
                        "x": "(x^2-1)^2",
                        "y": "0.5*y^2",
                        "z": "0.25*z^2",
                    },
                    "seed": seed,
                    "policy": policy,
                    "batch_size": batch_size,
                    "max_workers": max_workers,
                    "action_force_budget": action_force_budget,
                    "total_force_budget": total_force_budget,
                    "best_energy": min(
                        entry.energy for entry in result.archive.entries
                    ),
                    "unique_minima": len(result.archive.entries),
                    "completed_batches": result.completed_batches,
                    "completed_attempts": result.completed_attempts,
                    "failed_attempts": result.failed_attempts,
                    "posterior_observed_attempts": (
                        result.posterior_observed_attempts
                    ),
                    "bootstrap_evaluations": result.bootstrap_evaluations,
                    "action_evaluations": result.action_evaluations,
                    "total_evaluations": result.total_evaluations,
                    "purpose_counts": result.purpose_counts.as_dict(),
                    "unused_force_budget": result.unused_force_budget,
                    "stop_reason": result.stop_reason.value,
                    "benchmark_eligible": result.benchmark_eligible,
                    "benchmark_ineligibility_reasons": list(
                        result.benchmark_ineligibility_reasons
                    ),
                }
            )
    return tuple(records)
```

The CLI accepts the same execution controls plus `--output`. Parse
`--seeds` with `nargs="+"`. Derive the per-run root without another option:

```python
output_path = Path(args.output)
output_root = output_path.parent / f"{output_path.stem}-runs"
```

Reject the command before running if either `output_path` or `output_root`
already exists. Write:

```python
json.dumps(
    {"schema_version": 1, "records": list(records)},
    sort_keys=True,
    indent=2,
    allow_nan=False,
) + "\n"
```

to `output_path`. Do not compute aggregate policy statistics.

- [ ] **Step 4: Run the harness and inspect real JSON**

Run:

```bash
python benchmarks/posterior_policy_compare.py \
  --output /tmp/pamssw-posterior-policy-smoke.json \
  --seeds 3 \
  --total-force-budget 120 \
  --action-force-budget 30 \
  --batch-size 2 \
  --max-workers 2
```

Then run:

```bash
python -m json.tool /tmp/pamssw-posterior-policy-smoke.json
pytest -q tests/unit/test_posterior_policy_compare.py
```

Expected: valid JSON with three raw records and all tests passing.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/posterior_policy_compare.py \
  tests/unit/test_posterior_policy_compare.py
git commit -m "Add paired posterior policy harness"
```

---

### Task 5: Document the minimal claim and retire the recovery plan

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-25-recoverable-budgeted-posterior-runner.md`

- [ ] **Step 1: Add the supersession marker**

Immediately before old Task 6, add:

```markdown
> **Superseded remainder:** Tasks 6–11 below are not part of the active
> implementation scope. They were replaced by
> `docs/superpowers/specs/2026-07-26-minimal-posterior-ablation-runner-design.md`
> and
> `docs/superpowers/plans/2026-07-26-minimal-posterior-ablation-runner.md`.
> Phase-3 completion now means a non-recoverable analytic ThreadPool runner
> and raw three-policy harness; no manifest/resume/replay layer is required.
```

Keep the old text as historical design context rather than deleting it.

- [ ] **Step 2: Update the README claim boundary**

Replace the obsolete statements that no SSW runner/global budget exists.
Document:

- the two opt-in runner functions;
- raw-State bootstrap included in the campaign budget;
- fixed action fidelity and exact per-purpose accounting;
- synchronous complete-batch ThreadPool semantics;
- the three unchanged policies and propensity support boundary;
- compact event logging;
- analytic backend validation only;
- no recovery/resume, async racing, MACE/GPU/process validation, canonical
  sampling, or superiority claim.

Do not modify production preset recommendations.

- [ ] **Step 3: Run documentation and import checks**

Run:

```bash
python - <<'PY'
import pamssw
assert callable(pamssw.run_posterior_ssw)
assert callable(pamssw.run_posterior_ls_ssw)
PY
rg -n "run_posterior_ssw|fixed action|no recovery|no resume" README.md
git diff --check
```

- [ ] **Step 4: Commit**

```bash
git add README.md \
  docs/superpowers/plans/2026-07-25-recoverable-budgeted-posterior-runner.md
git commit -m "Document minimal posterior ablation runner"
```

---

### Task 6: Completion audit and branch verification

**Files:**
- Review all files changed since `6b2a289`
- Do not add production features

- [ ] **Step 1: Run focused statistical and accounting suites**

Run:

```bash
pytest -q \
  tests/unit/test_accounting.py \
  tests/unit/test_exploration_actions.py \
  tests/unit/test_exploration_batch.py \
  tests/unit/test_exploration_campaign.py \
  tests/unit/test_exploration_event_log.py \
  tests/unit/test_exploration_policies.py \
  tests/unit/test_exploration_posterior.py \
  tests/unit/test_ssw_attempt_worker.py \
  tests/integration/test_exploration_controller.py \
  tests/integration/test_ssw_attempt_worker_integration.py \
  tests/integration/test_posterior_exploration_runner.py \
  tests/unit/test_posterior_policy_compare.py
```

- [ ] **Step 2: Run the complete repository suite**

Run:

```bash
pytest -q
```

Record the exact pass/fail/skip count and exit code. A narrow suite cannot
support the branch-completion claim.

- [ ] **Step 3: Run static and source-boundary checks**

Run:

```bash
python -m compileall -q pamssw benchmarks/posterior_policy_compare.py
git diff --check
git status --short
git diff --stat 6b2a289..HEAD
```

Verify:

- no `run_store.py`, resume/session/manifest production module was added;
- `pamssw/exploration/policies.py` and
  `pamssw/exploration/posterior.py` are unchanged from Phase 1;
- numerical direction, bias, and step formulas are unchanged from the
  accounting-only Phase-3 baseline;
- legacy `run_ssw` and `run_ls_ssw` imports still resolve.

- [ ] **Step 4: Obtain independent reviews**

Run a specification review against the reduced design and this plan, then a
separate code-quality review. Resolve every blocker and rerun the relevant
suite after each fix.

- [ ] **Step 5: Commit audit-only corrections**

If reviews require corrections, commit only the verified fixes:

```bash
git add README.md benchmarks/posterior_policy_compare.py \
  docs/superpowers/plans/2026-07-25-recoverable-budgeted-posterior-runner.md \
  pamssw/__init__.py pamssw/exploration/__init__.py \
  pamssw/exploration/campaign.py pamssw/exploration/runner.py \
  tests/integration/test_posterior_exploration_runner.py \
  tests/unit/test_exploration_campaign.py \
  tests/unit/test_posterior_policy_compare.py
git commit -m "Complete posterior ablation runner audit"
```

Do not create an empty audit commit.

- [ ] **Step 6: Push and create the Phase-3 pull request**

Push:

```bash
git push -u pam feature/recoverable-budgeted-posterior-runner
```

Create a pull request whose body states:

- exact Phase-3 implementation scope;
- the removed recovery/resume scope;
- focused and full-suite results;
- analytic ThreadPool-only runtime claim;
- no policy-performance or unbiased-sampling claim;
- remaining representation, direction, uphill-policy, and scientific
  benchmark work.
