# Staged Direction-Efficiency Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute the preregistered C60 momentum, direction-count, bias-step, and conditional proposal-relax-cap ablations with exact terminal outcomes and force-evaluation accounting.

**Architecture:** Preserve the existing `SurfaceWalker`, direction scorer, bias equations, safe-LBFGS, strict true-quench, and locked C60 starters. Add one zero-cost candidate-source diagnostic to the native discrete oracle, then place all stage scheduling, evidence validation, and decision gates in a self-contained experimental package under `runs/`; no reusable production abstraction or posterior selector is added.

**Tech Stack:** Python 3.12, NumPy, pytest, ASE, MACE CUDA calculator, existing `pamssw` accounting/archive/walker APIs.

---

## Scope boundary

This plan implements only the non-learning M/K/B/L campaign in
`docs/superpowers/specs/2026-07-29-staged-direction-efficiency-and-posterior-gates-design.md`.

It does not implement:

- Thompson sampling or another UCB-like rule;
- MACE-feature training;
- direction crossover;
- an online posterior;
- PdO validation;
- a 200-macro-step production campaign.

The final task evaluates and records the posterior-feasibility checklist.  If
that checklist passes, data collection and model comparison require a
separate design and implementation plan.

## File map

### Production code

- Modify `pamssw/walker.py`
  - record the source composition of native candidates already evaluated by
    the discrete direction oracle;
  - add no calculator call and change no score or selected direction.

### Unit tests

- Modify `tests/unit/test_direction_candidate_budget.py`
  - prove candidate-source counts and HVP cost remain exact.
- Create `tests/unit/test_staged_direction_efficiency_protocol.py`
  - test case scheduling, repeat-stable sets, stage decisions, and Stage-L
    entry.
- Create `tests/unit/test_staged_direction_efficiency_runner.py`
  - test frozen config projection, trace/accounting validation, stage
    prerequisites, atomic evidence publication, and no selector fields.

### Experimental package

- Create `runs/20260729-staged-direction-efficiency-ablation/protocol.py`
  - immutable stage/arm/case descriptions;
  - alternating paired schedule;
  - meaningful outcome and repeat-stable set;
  - M/K/B/L decisions and Stage-L entry;
  - deterministic evidence and conclusion construction.
- Create `runs/20260729-staged-direction-efficiency-ablation/run_stage.py`
  - load the existing locked C60 states and MACE calculator;
  - apply only preregistered config overrides;
  - run one proposal and strict true-quench per case;
  - validate direction and purpose ledgers;
  - save per-case structures, hashes, summaries, partial raw evidence, final
    evidence, and conclusion.
- Create `runs/20260729-staged-direction-efficiency-ablation/README.md`
  - exact commands, stage dependencies, budgets, and claim ceiling.

### Runtime evidence

- Create `runs/20260729-direction-efficiency-momentum/`
- Create `runs/20260729-direction-efficiency-candidate-count/`
- Create `runs/20260729-direction-efficiency-bias-steps/`
- Conditionally create `runs/20260729-direction-efficiency-relax-cap/`
- Create `runs/20260729-staged-direction-efficiency-ablation/final_report.md`
- Create `runs/20260729-staged-direction-efficiency-ablation/posterior_gate.json`

## Task 1: Add zero-cost candidate-source diagnostics

**Files:**

- Modify: `pamssw/walker.py:1356-1550`
- Modify: `tests/unit/test_direction_candidate_budget.py`

- [ ] **Step 1: Write the failing native-oracle diagnostic test**

Append this test to `tests/unit/test_direction_candidate_budget.py`:

```python
def test_discrete_choice_records_exact_evaluated_candidate_source_counts():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    calculator = EvalCounter(AnalyticCalculator(Quadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(7),
        candidates=4,
        bond_pairs=[(0, 1)],
        n_bond_pairs=0,
        enable_momentum_candidate=True,
    )

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=np.ones(state.positions.size),
    )

    assert choice.candidate_count == 4
    assert choice.diagnostics["evaluated_candidate_kind_counts"] == {
        "bond": 1,
        "momentum": 1,
        "random": 2,
    }
    assert sum(
        choice.diagnostics["evaluated_candidate_kind_counts"].values()
    ) == choice.candidate_count
    assert calculator.force_evaluations == 2 * choice.candidate_count
```

- [ ] **Step 2: Run the focused test and verify the diagnostic is absent**

Run:

```bash
pytest tests/unit/test_direction_candidate_budget.py::test_discrete_choice_records_exact_evaluated_candidate_source_counts -v
```

Expected: `FAIL` with `KeyError: 'evaluated_candidate_kind_counts'`.

- [ ] **Step 3: Add the diagnostic without changing selection**

Add the import near the top of `pamssw/walker.py`:

```python
from collections import Counter
```

Immediately after `candidates.extend(archive_momentum_candidates)` in
`SoftModeOracle.choose_direction`, compute:

```python
evaluated_candidate_kind_counts = dict(
    sorted(Counter(candidate.kind.value for candidate in candidates).items())
)
```

Change only the final native/discrete `DirectionChoice` construction to pass:

```python
diagnostics={
    "evaluated_candidate_kind_counts": (
        evaluated_candidate_kind_counts
    ),
},
```

Do not add this field to block-Krylov or exact-anchor paths; the name means
native candidates whose HVPs were actually evaluated.

- [ ] **Step 4: Run focused and neighboring oracle tests**

Run:

```bash
pytest tests/unit/test_direction_candidate_budget.py tests/unit/test_walker_policy.py -q
```

Expected: all tests pass; the existing HVP-count assertions remain unchanged.

- [ ] **Step 5: Commit the isolated observability change**

```bash
git add pamssw/walker.py tests/unit/test_direction_candidate_budget.py
git commit -m "test: expose discrete direction source composition"
```

## Task 2: Implement pure stage scheduling and decision gates

**Files:**

- Create: `runs/20260729-staged-direction-efficiency-ablation/protocol.py`
- Create: `tests/unit/test_staged_direction_efficiency_protocol.py`

- [ ] **Step 1: Write failing schedule tests**

Create `tests/unit/test_staged_direction_efficiency_protocol.py` with a module
loader matching other `runs/` tests and these assertions:

```python
def test_stage_case_counts_and_alternating_repeat_order():
    module = load_protocol()
    retained = module.RetainedSettings()

    assert len(module.case_matrix("momentum", retained)) == 24
    assert len(module.case_matrix("candidate_count", retained)) == 36
    assert len(module.case_matrix("bias_steps", retained)) == 24
    assert len(module.case_matrix("relax_cap", retained)) == 24

    block = [
        case.arm
        for case in module.case_matrix("momentum", retained)
        if case.state_id == "intermediate_accepted"
        and case.seed == 42
        and case.repeat == 0
    ]
    reversed_block = [
        case.arm
        for case in module.case_matrix("momentum", retained)
        if case.state_id == "intermediate_accepted"
        and case.seed == 42
        and case.repeat == 1
    ]
    assert block == ["momentum_on", "momentum_off"]
    assert reversed_block == list(reversed(block))
```

Add a row helper and tests proving:

```python
def test_repeat_stable_set_requires_both_exact_repeats():
    module = load_protocol()
    rows = make_complete_rows(module, "momentum")
    rows_by_key = {
        (row["state_id"], row["seed"], row["arm"], row["repeat"]): row
        for row in rows
    }
    rows_by_key[
        ("intermediate_accepted", 42, "momentum_on", 1)
    ]["landing_delta_eV"] = 0.1

    stable = module.repeat_stable_meaningful_set(
        rows,
        "momentum_on",
    )
    assert ("intermediate_accepted", 42) not in stable
```

```python
def test_stage_l_entry_uses_measured_cost_and_cap_hit_fractions():
    module = load_protocol()
    rows = make_complete_rows(module, "bias_steps", retained_arm="b5")
    for row in rows:
        row["force_evaluations"] = 100
        row["purpose_counts"]["biased_proposal_relax"] = 60
        row["optimizer_diagnostics"]["proposal_relax_count"] = 5
        row["optimizer_diagnostics"][
            "proposal_relax_termination_maxiter"
        ] = 1

    gate = module.stage_l_entry(rows, retained_arm="b5")
    assert gate == {
        "entered": True,
        "proposal_relax_force_fraction": 0.6,
        "proposal_relax_cap_hit_fraction": 0.2,
        "proposal_relax_calls": 60,
        "proposal_relax_cap_hits": 12,
    }
```

Add a decision test in which `l40` has a superset stable-outcome set and lower
total FE but equal `biased_proposal_relax` FE to `l80`; assert
`decide_stage("relax_cap", ...)` retains
`proposal_relax_steps == 80`.  Then reduce the `l40` proposal-relax FE in both
repeats and assert it selects 40.  This prevents total-cost savings elsewhere
from being misreported as an optimizer-cap improvement.

- [ ] **Step 2: Run the new test file and verify import failure**

Run:

```bash
pytest tests/unit/test_staged_direction_efficiency_protocol.py -v
```

Expected: `FAIL` because `protocol.py` does not exist.

- [ ] **Step 3: Add immutable protocol records and exact arms**

Create `protocol.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
REPEATS = (0, 1)
MEANINGFUL_ENERGY_DROP_EV = 0.001

STAGE_ARMS = {
    "momentum": ("momentum_on", "momentum_off"),
    "candidate_count": ("k4", "k8", "k12"),
    "bias_steps": ("b5", "b8"),
    "relax_cap": ("l40", "l80"),
}


@dataclass(frozen=True)
class RetainedSettings:
    enable_momentum_candidate: bool = True
    oracle_candidates: int = 12
    max_steps_per_walk: int = 8
    proposal_relax_steps: int = 80


@dataclass(frozen=True)
class CaseSpec:
    stage: str
    state_id: str
    seed: int
    arm: str
    repeat: int
    settings: RetainedSettings

    @property
    def key(self) -> str:
        return (
            f"{self.state_id}-seed{self.seed}-{self.arm}-"
            f"repeat{self.repeat}"
        )


def arm_settings(
    stage: str,
    arm: str,
    retained: RetainedSettings,
) -> RetainedSettings:
    if stage == "momentum":
        return replace(
            retained,
            enable_momentum_candidate=arm == "momentum_on",
        )
    if stage == "candidate_count":
        return replace(retained, oracle_candidates=int(arm[1:]))
    if stage == "bias_steps":
        return replace(retained, max_steps_per_walk=int(arm[1:]))
    if stage == "relax_cap":
        return replace(retained, proposal_relax_steps=int(arm[1:]))
    raise ValueError(f"unknown stage: {stage}")


def case_matrix(
    stage: str,
    retained: RetainedSettings,
) -> list[CaseSpec]:
    arms = STAGE_ARMS.get(stage)
    if arms is None:
        raise ValueError(f"unknown stage: {stage}")
    cases = []
    for state_id in STATE_IDS:
        for seed in SEEDS:
            for repeat in REPEATS:
                ordered = arms if repeat == 0 else tuple(reversed(arms))
                for arm in ordered:
                    cases.append(
                        CaseSpec(
                            stage=stage,
                            state_id=state_id,
                            seed=seed,
                            arm=arm,
                            repeat=repeat,
                            settings=arm_settings(
                                stage,
                                arm,
                                retained,
                            ),
                        )
                    )
    return cases
```

- [ ] **Step 4: Add parameter-free outcome and gate functions**

Continue `protocol.py` with:

```python
def is_meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("certificate") is True
        and row.get("is_new_basin") is True
        and float(row["landing_delta_eV"])
        <= -MEANINGFUL_ENERGY_DROP_EV
    )


def repeat_stable_meaningful_set(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> set[tuple[str, int]]:
    return {
        (state_id, seed)
        for state_id in STATE_IDS
        for seed in SEEDS
        if all(
            is_meaningful(
                next(
                    row
                    for row in rows
                    if row["state_id"] == state_id
                    and int(row["seed"]) == seed
                    and row["arm"] == arm
                    and int(row["repeat"]) == repeat
                )
            )
            for repeat in REPEATS
        )
    }


def _repeat_totals(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    field: str,
) -> dict[int, int]:
    return {
        repeat: sum(
            int(row[field])
            for row in rows
            if row["arm"] == arm and int(row["repeat"]) == repeat
        )
        for repeat in REPEATS
    }


def _repeat_meaningful_counts(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[int, int]:
    return {
        repeat: sum(
            is_meaningful(row)
            for row in rows
            if row["arm"] == arm and int(row["repeat"]) == repeat
        )
        for repeat in REPEATS
    }


def _all_certified(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> bool:
    selected = [row for row in rows if row["arm"] == arm]
    return bool(selected) and all(
        row.get("certificate") is True for row in selected
    )


def _lower_cost_each_repeat(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
) -> bool:
    candidate_cost = _repeat_totals(
        rows,
        candidate,
        "force_evaluations",
    )
    baseline_cost = _repeat_totals(
        rows,
        baseline,
        "force_evaluations",
    )
    return all(
        candidate_cost[repeat] < baseline_cost[repeat]
        for repeat in REPEATS
    )


def _lower_proposal_cost_each_repeat(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
) -> bool:
    def totals(arm: str) -> dict[int, int]:
        return {
            repeat: sum(
                int(
                    row["purpose_counts"][
                        "biased_proposal_relax"
                    ]
                )
                for row in rows
                if row["arm"] == arm
                and int(row["repeat"]) == repeat
            )
            for repeat in REPEATS
        }

    candidate_cost = totals(candidate)
    baseline_cost = totals(baseline)
    return all(
        candidate_cost[repeat] < baseline_cost[repeat]
        for repeat in REPEATS
    )


def decide_stage(
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    retained: RetainedSettings,
) -> dict[str, Any]:
    stable = {
        arm: repeat_stable_meaningful_set(rows, arm)
        for arm in STAGE_ARMS[stage]
    }
    if stage == "momentum":
        on_counts = _repeat_meaningful_counts(rows, "momentum_on")
        off_counts = _repeat_meaningful_counts(rows, "momentum_off")
        on_dominates = (
            stable["momentum_on"] > stable["momentum_off"]
            and all(on_counts[r] >= off_counts[r] for r in REPEATS)
        )
        off_dominates = (
            stable["momentum_off"] > stable["momentum_on"]
            and all(off_counts[r] >= on_counts[r] for r in REPEATS)
        )
        keep = not off_dominates
        status = (
            "positive"
            if on_dominates
            else "removal_candidate"
            if off_dominates
            else "unproven_retained"
        )
        selected = replace(
            retained,
            enable_momentum_candidate=keep,
        )
    else:
        baseline = {
            "candidate_count": "k12",
            "bias_steps": "b8",
            "relax_cap": "l80",
        }[stage]
        candidates = {
            "candidate_count": ("k4", "k8"),
            "bias_steps": ("b5",),
            "relax_cap": ("l40",),
        }[stage]
        winner = baseline
        for candidate in candidates:
            if (
                stable[candidate] >= stable[baseline]
                and _all_certified(rows, candidate)
                and _lower_cost_each_repeat(
                    rows,
                    candidate,
                    baseline,
                )
                and (
                    stage != "relax_cap"
                    or _lower_proposal_cost_each_repeat(
                        rows,
                        candidate,
                        baseline,
                    )
                )
            ):
                winner = candidate
                break
        selected = arm_settings(stage, winner, retained)
        status = "reduced" if winner != baseline else "baseline_retained"
    return {
        "stage": stage,
        "status": status,
        "repeat_stable_sets": {
            arm: sorted([list(key) for key in values])
            for arm, values in stable.items()
        },
        "retained_settings": asdict(selected),
    }
```

Add Stage-L entry exactly as:

```python
def stage_l_entry(
    rows: Sequence[Mapping[str, Any]],
    retained_arm: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == retained_arm]
    total = sum(int(row["force_evaluations"]) for row in selected)
    proposal = sum(
        int(row["purpose_counts"]["biased_proposal_relax"])
        for row in selected
    )
    calls = sum(
        int(row["optimizer_diagnostics"]["proposal_relax_count"])
        for row in selected
    )
    cap_hits = sum(
        int(
            row["optimizer_diagnostics"][
                "proposal_relax_termination_maxiter"
            ]
        )
        for row in selected
    )
    if total <= 0 or calls <= 0:
        raise ValueError(
            "Stage-L entry requires positive total FE and relax calls"
        )
    force_fraction = proposal / total
    cap_fraction = cap_hits / calls
    return {
        "entered": force_fraction > 0.5 and cap_fraction >= 0.2,
        "proposal_relax_force_fraction": force_fraction,
        "proposal_relax_cap_hit_fraction": cap_fraction,
        "proposal_relax_calls": calls,
        "proposal_relax_cap_hits": cap_hits,
    }
```

- [ ] **Step 5: Run protocol tests**

Run:

```bash
pytest tests/unit/test_staged_direction_efficiency_protocol.py -q
```

Expected: all schedule, stable-set, dominance, cost, and Stage-L-entry tests
pass.

- [ ] **Step 6: Commit the pure protocol**

```bash
git add runs/20260729-staged-direction-efficiency-ablation/protocol.py tests/unit/test_staged_direction_efficiency_protocol.py
git commit -m "test: define staged direction ablation gates"
```

## Task 3: Implement frozen config and direction-ledger validation

**Files:**

- Create: `runs/20260729-staged-direction-efficiency-ablation/run_stage.py`
- Create: `tests/unit/test_staged_direction_efficiency_runner.py`

- [ ] **Step 1: Write failing config-projection tests**

The test must load `run_stage.py`, use a fake base runner returning the locked
C60 production config, and assert:

```python
source, effective, diff = runner.config_projection(
    case=protocol.CaseSpec(
        stage="candidate_count",
        state_id="intermediate_accepted",
        seed=42,
        arm="k4",
        repeat=0,
        settings=protocol.RetainedSettings(
            enable_momentum_candidate=True,
            oracle_candidates=4,
            max_steps_per_walk=8,
            proposal_relax_steps=80,
        ),
    ),
    case_dir=tmp_path,
    base_runner=fake_base_runner,
)

assert effective["direction_selection_mode"] == "discrete"
assert effective["direction_synthesis_mode"] == "none"
assert effective["direction_type_ucb_enabled"] is False
assert effective["archive_escape_momentum_enabled"] is False
assert effective["proposal_optimizer"] == "safe-lbfgs-total"
assert effective["proposal_fmax"] == pytest.approx(0.05)
assert effective["enable_momentum_candidate"] is True
assert effective["oracle_candidates"] == 4
assert effective["max_steps_per_walk"] == 8
assert effective["proposal_relax_steps"] == 80
assert effective["quench_optimizer"] == "ase-lbfgs"
assert effective["quench_fallback_optimizer"] == "ase-fire"
assert effective["quench_fmax"] == pytest.approx(0.01)
assert effective["quench_maxiter"] == 400
assert set(diff) == {
    "max_trials",
    "oracle_candidates",
    "quench_fallback_optimizer",
    "quench_optimizer",
}
```

Also parameterize forbidden drift in `direction_selection_mode`,
`direction_synthesis_mode`, `direction_type_ucb_enabled`,
`archive_escape_momentum_enabled`, `proposal_optimizer`, `proposal_fmax`, and
`quench_fmax`; expect `RuntimeError("frozen protocol drifted")`.

- [ ] **Step 2: Write failing direction-trace tests**

Use a two-row trace with `K=4`:

```python
rows = [
    {
        "step": 0,
        "selected_kind": "bond",
        "candidate_count": 4,
        "evaluated_candidate_kind_counts": {
            "bond": 2,
            "random": 2,
        },
        "oracle_selection_force_evaluations_delta": 8,
        "oracle_direction_force_evaluations_delta": 8,
    },
    {
        "step": 1,
        "selected_kind": "momentum",
        "candidate_count": 4,
        "evaluated_candidate_kind_counts": {
            "bond": 2,
            "momentum": 1,
            "random": 1,
        },
        "oracle_selection_force_evaluations_delta": 8,
        "oracle_direction_force_evaluations_delta": 8,
    },
]
audit = runner.validate_direction_trace(rows, oracle_candidates=4)
assert audit == {
    "selection_count": 2,
    "candidate_count": 8,
    "candidate_kind_counts": {
        "bond": 4,
        "momentum": 1,
        "random": 3,
    },
    "selected_kind_counts": {"bond": 1, "momentum": 1},
    "direction_oracle_force_evaluations": 16,
}
```

Mutation tests must fail when a source-count sum differs from
`candidate_count`, the selected kind is absent, or either FE delta differs
from `2 * oracle_candidates`.

- [ ] **Step 3: Implement exact config projection**

In `run_stage.py`, import the approved protocol via a local normal import and
the existing locked runtime via `importlib`.  Define:

```python
FROZEN_FIELDS = {
    "direction_selection_mode": "discrete",
    "direction_synthesis_mode": "none",
    "direction_type_ucb_enabled": False,
    "archive_escape_momentum_enabled": False,
    "choice_aligned_softening_enabled": False,
    "direction_probe_enabled": False,
    "plateau_evolution_enabled": False,
    "direction_curvature_source": "inner",
    "proposal_optimizer": "safe-lbfgs-total",
    "proposal_fmax": 0.05,
    "quench_fmax": 0.01,
}

BASELINE_FIELDS = {
    "enable_momentum_candidate": True,
    "oracle_candidates": 12,
    "max_steps_per_walk": 8,
    "proposal_relax_steps": 80,
}

COMMON_OVERRIDES = {
    "max_trials": 1,
    "max_force_evals": None,
    "quench_optimizer": "ase-lbfgs",
    "quench_fallback_optimizer": "ase-fire",
    "quench_fmax": 0.01,
    "quench_maxiter": 400,
}


def config_projection(*, case, case_dir, base_runner):
    source_config = base_runner.build_config("c60", case_dir)
    source = asdict(source_config)
    for field, expected in FROZEN_FIELDS.items():
        if source[field] != expected:
            raise RuntimeError(
                f"frozen protocol drifted: {field}"
            )
    for field, expected in BASELINE_FIELDS.items():
        if source[field] != expected:
            raise RuntimeError(
                f"frozen baseline drifted: {field}"
            )
    effective_config = replace(
        source_config,
        **COMMON_OVERRIDES,
        rng_seed=case.seed,
        enable_momentum_candidate=(
            case.settings.enable_momentum_candidate
        ),
        oracle_candidates=case.settings.oracle_candidates,
        max_steps_per_walk=case.settings.max_steps_per_walk,
        proposal_relax_steps=case.settings.proposal_relax_steps,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(
            case_dir / "direction_trace.jsonl"
        ),
    )
    effective = asdict(effective_config)
    diff = {
        key: [source[key], effective[key]]
        for key in source
        if source[key] != effective[key]
    }
    return source, effective, diff
```

The production C60 builder already supplies \(B=8\), \(K=12\), \(L=80\).
Tests must fail if those source values drift before applying a stage arm.

- [ ] **Step 4: Implement exact native trace validation**

Add:

```python
def validate_direction_trace(
    rows,
    *,
    oracle_candidates: int,
):
    if not rows:
        raise RuntimeError("direction trace is empty")
    candidate_kinds = Counter()
    selected_kinds = Counter()
    for index, row in enumerate(rows):
        if int(row["candidate_count"]) != oracle_candidates:
            raise RuntimeError(
                f"direction row {index} candidate count drifted"
            )
        counts = row.get("evaluated_candidate_kind_counts")
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"direction row {index} lacks source counts"
            )
        if sum(int(value) for value in counts.values()) != oracle_candidates:
            raise RuntimeError(
                f"direction row {index} source counts do not close"
            )
        selected = str(row["selected_kind"])
        if int(counts.get(selected, 0)) <= 0:
            raise RuntimeError(
                f"direction row {index} selected source was not evaluated"
            )
        expected_fe = 2 * oracle_candidates
        for field in (
            "oracle_selection_force_evaluations_delta",
            "oracle_direction_force_evaluations_delta",
        ):
            if int(row[field]) != expected_fe:
                raise RuntimeError(
                    f"direction row {index} {field} does not close"
                )
        candidate_kinds.update(
            {key: int(value) for key, value in counts.items()}
        )
        selected_kinds[selected] += 1
    return {
        "selection_count": len(rows),
        "candidate_count": len(rows) * oracle_candidates,
        "candidate_kind_counts": dict(sorted(candidate_kinds.items())),
        "selected_kind_counts": dict(sorted(selected_kinds.items())),
        "direction_oracle_force_evaluations": (
            2 * oracle_candidates * len(rows)
        ),
    }
```

This validates candidate-ranking HVP cost.  Keep
`escape_true_pes_check` separate in the purpose ledger; do not relabel true
energy checks as direction-oracle cost.

- [ ] **Step 5: Run config and trace tests**

Run:

```bash
pytest tests/unit/test_staged_direction_efficiency_runner.py -k 'projection or trace' -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the frozen protocol seam**

```bash
git add runs/20260729-staged-direction-efficiency-ablation/run_stage.py tests/unit/test_staged_direction_efficiency_runner.py
git commit -m "test: freeze direction efficiency runtime protocol"
```

## Task 4: Implement one-proposal case execution and evidence publication

**Files:**

- Modify: `runs/20260729-staged-direction-efficiency-ablation/run_stage.py`
- Modify: `runs/20260729-staged-direction-efficiency-ablation/protocol.py`
- Modify: `tests/unit/test_staged_direction_efficiency_runner.py`
- Create: `runs/20260729-staged-direction-efficiency-ablation/README.md`

- [ ] **Step 1: Write failing per-case closure test**

Use a fake locked runtime, calculator counter, walker, archive, and strict
landing result following the existing fixed-starter runner tests.  Assert the
case summary contains:

```python
assert row["status"] == "completed"
assert row["stage"] == "momentum"
assert row["repeat"] == 0
assert row["certificate"] is True
assert row["is_new_basin"] is True
assert row["meaningful"] is True
assert row["purpose_counts"]["unattributed"] == 0
assert sum(row["purpose_counts"].values()) == row["force_evaluations"]
assert row["direction_audit"]["direction_oracle_force_evaluations"] == (
    row["purpose_counts"]["direction_oracle"]
)
assert row["effective_config"]["proposal_optimizer"] == "safe-lbfgs-total"
assert row["optimizer_diagnostics"]["proposal_relax_count"] > 0
assert len(row["starter_file_sha256"]) == 64
assert len(row["escape_sha256"]) == 64
assert len(row["landing_sha256"]) == 64
```

Add negative tests for unattributed calls, a non-converged true quench, and a
direction ledger mismatch; no final `summary.json` may be published after
failure.

- [ ] **Step 2: Implement `_run_case` using the existing physical path**

The function must:

1. create a fresh `SurfaceWalker` and two-entry `MinimaArchive`;
2. evaluate the locked starter under `ESCAPE_TRUE_PES_CHECK`;
3. call `_proposal_pool(..., allow_duplicate_rescue=False)[0]`;
4. evaluate the escape under `ESCAPE_TRUE_PES_CHECK`;
5. call `relax_true_minimum` once;
6. require `has_force_convergence_certificate`;
7. classify the landing in the fresh archive;
8. validate the direction trace and purpose ledger;
9. save and hash starter/escape/landing structures;
10. atomically write the per-case summary.

Use this exact invariant block before publication:

```python
purpose_counts = walker.calculator.snapshot().as_dict()
force_evaluations = sum(int(value) for value in purpose_counts.values())
if purpose_counts["unattributed"] != 0:
    raise RuntimeError("case contains unattributed force evaluations")
if (
    purpose_counts["direction_oracle"]
    != direction_audit["direction_oracle_force_evaluations"]
):
    raise RuntimeError("direction purpose ledger does not close")
if purpose_counts["landing_true_quench"] <= 0:
    raise RuntimeError("strict terminal quench did no physical work")
if not certificate:
    raise RuntimeError("terminal quench lacks a strict certificate")
```

Store the entire scalar `walker.relaxation_diagnostics()` dictionary as
`optimizer_diagnostics`.  Its existing fields
`proposal_relax_count`, `proposal_relax_median_iterations`,
`proposal_relax_p90_iterations`, `proposal_relax_max_iterations`, and
`proposal_relax_termination_maxiter` supply the Stage-L measurements without
opening walker internals or adding raw-list state.

Also store `selection_probability=1.0`: every case in the frozen exhaustive
matrix is executed, so this is a known design propensity rather than a learned
selector probability.

Store a bounded `proposal_trace` object rather than full optimizer coordinate
trajectories:

```python
proposal_trace = {
    "direction_steps": direction_rows,
    "attempted_bias_steps": int(
        optimizer_diagnostics["proposal_relax_count"]
    ),
    "configured_bias_step_cap": config.max_steps_per_walk,
    "termination_reason": (
        "reached_bias_step_cap"
        if int(optimizer_diagnostics["proposal_relax_count"])
        == config.max_steps_per_walk
        else "early_exit"
    ),
    "optimizer_termination_counts": {
        key.removeprefix("proposal_relax_termination_"): int(value)
        for key, value in optimizer_diagnostics.items()
        if key.startswith("proposal_relax_termination_")
    },
}
```

This records the scientifically relevant step/source/optimizer termination
path without enabling full coordinate trajectories or adding calculator work.

- [ ] **Step 3: Implement complete-cohort evidence validation**

Add `build_evidence(stage, retained, rows)` to `protocol.py`.  It must:

- require exact equality between the observed case keys and `case_matrix`;
- reject duplicate or missing cases;
- require `status == "completed"`, strict certificate, exact starter
  provenance, zero unattributed calls, and closed purpose/direction ledgers;
- compute continuous landing deltas, meaningful flags, stable sets,
  per-repeat FE totals, source-composition totals, selected-source totals,
  failure counts, and the stage decision;
- include `production_default_changed: false`;
- include the exact 0.001 eV operational margin and claim ceiling.

Do not place `reward`, `selector`, `posterior`, `UCB`, or `TS` fields in the
evidence schema.

- [ ] **Step 4: Implement resumable, exact stage orchestration**

`run_stage.py` must accept:

```text
--stage {momentum,candidate_count,bias_steps,relax_cap}
--output-dir PATH
--expected-git-commit SHA
--prior-evidence PATH
```

Rules:

- `momentum` forbids `--prior-evidence`;
- `candidate_count` requires momentum evidence and uses its
  `retained_settings`;
- `bias_steps` requires candidate-count evidence;
- `relax_cap` requires bias-step evidence and refuses execution unless its
  saved `stage_l_entry.entered` is true;
- every prior evidence file is hashed into the next stage provenance;
- the tracked worktree must be clean and `HEAD` must match the CLI commit;
- an existing case may be reused only after its summary, artifact hashes,
  case spec, config, execution commit, and ledgers revalidate;
- `raw.json` is rewritten atomically after every completed case;
- `evidence.json` and `conclusion.md` appear only after the complete cohort
  validates.
- the first invocation copies the exact `run_stage.py` and `protocol.py` into
  the output root and records their hashes; resume requires those copies to
  match;
- each case writes its complete config to
  `effective_configs/<case-key>.json`.
- `raw.json` and `evidence.json` retain the locked-state, model, calculator,
  CUDA/runtime, runner-hash, and execution-commit provenance returned by the
  existing locked runtime/preflight path.

Use one shared MACE calculator sequentially but a fresh walker/counter per
case.  Do not add ThreadPool execution on one GPU in this campaign.

The parser must also define `--record-not-entered`.  It is valid only for
`--stage relax_cap` when the prior saved Stage-L entry gate is false.

- [ ] **Step 5: Write the experiment README**

The README must state:

- the four exact commands from Tasks 6--9;
- 24/36/24/conditional-24 proposal ceilings;
- one-proposal fixed-starter semantics;
- the distinction between `2*K` candidate-ranking FE and the full purpose
  ledger;
- exact-repeat semantics;
- that wall time is descriptive;
- that no C60 result changes a production default.

- [ ] **Step 6: Run all new runner tests**

Run:

```bash
pytest tests/unit/test_staged_direction_efficiency_protocol.py tests/unit/test_staged_direction_efficiency_runner.py tests/unit/test_direction_candidate_budget.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the complete runner**

```bash
git add runs/20260729-staged-direction-efficiency-ablation tests/unit/test_staged_direction_efficiency_protocol.py tests/unit/test_staged_direction_efficiency_runner.py
git commit -m "feat: add staged direction efficiency ablation"
```

## Task 5: Verify the implementation before consuming GPU budget

**Files:**

- Modify only if a verification failure identifies a real defect.

- [ ] **Step 1: Run formatting and whitespace checks**

```bash
git diff --check HEAD~3..HEAD
python -m compileall -q pamssw runs/20260729-staged-direction-efficiency-ablation tests/unit
```

Expected: both commands exit 0.

- [ ] **Step 2: Run the focused test cohort**

```bash
pytest tests/unit/test_direction_candidate_budget.py tests/unit/test_staged_direction_efficiency_protocol.py tests/unit/test_staged_direction_efficiency_runner.py -q
```

Expected: all pass.

- [ ] **Step 3: Run the full unit suite**

```bash
pytest tests/unit -q
```

Expected: all pass with no new warning class.

- [ ] **Step 4: Run a no-output CUDA preflight**

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

Expected: exit 0 and a CUDA device name.  If sandbox visibility blocks the
device, rerun the same command with the required execution escalation; do not
replace the production calculator with CPU.

- [ ] **Step 5: Record the clean execution commit**

```bash
git status --short
git rev-parse HEAD
```

Expected: empty status and one 40-character commit SHA.

## Task 6: Execute and conclude Stage M

**Files:**

- Create: `runs/20260729-direction-efficiency-momentum/`

- [ ] **Step 1: Run the 24-case momentum cohort**

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage momentum \
  --output-dir runs/20260729-direction-efficiency-momentum \
  --expected-git-commit "$(git rev-parse HEAD)"
```

Expected: 24 completed case summaries, `raw.json`, `evidence.json`, and
`conclusion.md`; all direction and purpose ledgers close.

- [ ] **Step 2: Re-run as a resume audit**

Run the identical command.

Expected: all 24 cases are revalidated and reused, zero new MACE force
evaluations are executed, and final evidence is byte-identical.

- [ ] **Step 3: Inspect the preregistered decision**

```bash
python -m json.tool runs/20260729-direction-efficiency-momentum/evidence.json
```

Expected: status is exactly one of `positive`, `removal_candidate`, or
`unproven_retained`; `retained_settings` follows the Stage-M gate.

- [ ] **Step 4: Commit Stage-M evidence**

```bash
git add runs/20260729-direction-efficiency-momentum
git commit -m "exp: conclude C60 momentum ablation"
```

## Task 7: Execute and conclude Stage K

**Files:**

- Create: `runs/20260729-direction-efficiency-candidate-count/`

- [ ] **Step 1: Run the 36-case candidate-count cohort**

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage candidate_count \
  --output-dir runs/20260729-direction-efficiency-candidate-count \
  --prior-evidence runs/20260729-direction-efficiency-momentum/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)"
```

Expected: 36 completed cases; each direction selection uses 8, 16, or 24
candidate-ranking FE for K=4, 8, or 12 respectively.

- [ ] **Step 2: Verify the chosen K obeys set and cost gates**

```bash
python -m json.tool runs/20260729-direction-efficiency-candidate-count/evidence.json
```

Expected: K=4 or K=8 is retained only if its stable set contains the K=12 set,
all certificates pass, and both repeat-level total FE sums are lower;
otherwise K=12 remains.

- [ ] **Step 3: Commit Stage-K evidence**

```bash
git add runs/20260729-direction-efficiency-candidate-count
git commit -m "exp: conclude C60 direction-count ablation"
```

## Task 8: Execute and conclude Stage B

**Files:**

- Create: `runs/20260729-direction-efficiency-bias-steps/`

- [ ] **Step 1: Run the 24-case bias-step cohort**

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage bias_steps \
  --output-dir runs/20260729-direction-efficiency-bias-steps \
  --prior-evidence runs/20260729-direction-efficiency-candidate-count/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)"
```

Expected: 24 completed cases with no stopping heuristic; arms differ only in
`max_steps_per_walk=5` versus 8.

- [ ] **Step 2: Verify Stage-B decision and Stage-L entry**

```bash
python -m json.tool runs/20260729-direction-efficiency-bias-steps/evidence.json
```

Expected: the evidence retains B=5 only under the stable-set, certificate, and
paired-cost gate.  It also contains:

```text
stage_l_entry.entered
stage_l_entry.proposal_relax_force_fraction
stage_l_entry.proposal_relax_cap_hit_fraction
stage_l_entry.proposal_relax_calls
stage_l_entry.proposal_relax_cap_hits
```

- [ ] **Step 3: Commit Stage-B evidence**

```bash
git add runs/20260729-direction-efficiency-bias-steps
git commit -m "exp: conclude C60 bias-step ablation"
```

## Task 9: Apply the conditional Stage-L gate

**Files:**

- Conditionally create: `runs/20260729-direction-efficiency-relax-cap/`

- [ ] **Step 1: Read the saved entry decision without overriding it**

```bash
python -c "import json; p=json.load(open('runs/20260729-direction-efficiency-bias-steps/evidence.json')); print(json.dumps(p['stage_l_entry'], indent=2))"
```

Expected: one immutable gate record.

- [ ] **Step 2a: If `entered` is true, run the 24-case relax-cap cohort**

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage relax_cap \
  --output-dir runs/20260729-direction-efficiency-relax-cap \
  --prior-evidence runs/20260729-direction-efficiency-bias-steps/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)"
```

Expected: 24 completed cases differing only in
`proposal_relax_steps=40` versus 80.

- [ ] **Step 2b: If `entered` is false, create a deterministic skip record**

Run:

```bash
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --stage relax_cap \
  --output-dir runs/20260729-direction-efficiency-relax-cap \
  --prior-evidence runs/20260729-direction-efficiency-bias-steps/evidence.json \
  --expected-git-commit "$(git rev-parse HEAD)" \
  --record-not-entered
```

Expected: `evidence.json` and `conclusion.md` state `not_entered`, copy the
measured entry fractions and prior-evidence hash, contain zero cases, and
perform zero MACE evaluations.

- [ ] **Step 3: Commit the executed or skipped Stage-L evidence**

```bash
git add runs/20260729-direction-efficiency-relax-cap
git commit -m "exp: apply C60 proposal-relax gate"
```

## Task 10: Produce the consolidated scientific conclusion

**Files:**

- Create: `runs/20260729-staged-direction-efficiency-ablation/final_report.md`
- Create: `runs/20260729-staged-direction-efficiency-ablation/posterior_gate.json`

- [ ] **Step 1: Add a deterministic report command**

Extend `run_stage.py` with:

```text
--summarize-campaign
--momentum-evidence PATH
--candidate-count-evidence PATH
--bias-step-evidence PATH
--relax-cap-evidence PATH
```

It must verify every evidence hash and stage dependency, then write the final
report and posterior checklist.  It must not fit a model.

- [ ] **Step 2: Test the summary command**

Add a fake four-stage evidence chain to
`tests/unit/test_staged_direction_efficiency_runner.py` and assert:

```python
assert gate["checks"] == {
    "two_fixed_direction_families_with_five_meaningful_each": False,
    "two_starter_classes_with_positive_outcomes": True,
    "complete_action_context_cost_certificate_records": True,
    "held_out_residual_signal_demonstrated": False,
}
assert gate["posterior_ready"] is False
assert "model" not in gate
assert "scalar_reward" not in gate
```

The fixed-direction-family check must remain false: these arms execute mixed
native portfolios across bias microsteps, so their terminal outcomes cannot be
credited to two fixed direction families.  The held-out residual check must
also remain false because this campaign does not run the separate feasibility
model.

- [ ] **Step 3: Generate the final report**

```bash
python runs/20260729-staged-direction-efficiency-ablation/run_stage.py \
  --summarize-campaign \
  --momentum-evidence runs/20260729-direction-efficiency-momentum/evidence.json \
  --candidate-count-evidence runs/20260729-direction-efficiency-candidate-count/evidence.json \
  --bias-step-evidence runs/20260729-direction-efficiency-bias-steps/evidence.json \
  --relax-cap-evidence runs/20260729-direction-efficiency-relax-cap/evidence.json
```

Expected report sections:

1. verified execution and accounting;
2. momentum causal result;
3. candidate-count result and FE savings;
4. bias-step result and FE savings;
5. Stage-L result or measured skip;
6. continuous landing-energy table;
7. source-composition and selected-source table;
8. invalid/damage/non-convergence taxonomy;
9. posterior-feasibility checklist;
10. proven, unproven, and next scientifically justified experiment.

- [ ] **Step 4: Run final verification**

```bash
pytest tests/unit -q
git diff --check
git status --short
```

Expected: tests pass; only the final report, gate file, and intended test/code
updates are uncommitted.

- [ ] **Step 5: Commit and push the completed campaign**

```bash
git add runs/20260729-staged-direction-efficiency-ablation tests/unit/test_staged_direction_efficiency_runner.py
git commit -m "docs: conclude staged direction efficiency campaign"
git push pam feature/posterior-terminal-outcome-validation
```

The handoff must report exact proposal count, force evaluations by purpose,
energy changes, wall time, retained settings, null/mixed findings, Stage-L
entry outcome, and why the posterior gate did or did not open.
