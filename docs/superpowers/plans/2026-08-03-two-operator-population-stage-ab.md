# Two-Operator PES Population Stage A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, before building a batch relaxer, whether direct displacement plus strict true quench and the current H8 SSW action form a repeatable non-dominated two-operator PES portfolio.

**Architecture:** Keep all new code inside one research run directory and leave `pamssw` production behavior unchanged.  Stage A replays already-recorded evidence with zero new force evaluations; only if it passes does Stage B execute 18 exact shared-input pairs across C60, PdO, and CuO and apply the preregistered decision.  Stage C GPU active-set batching is explicitly excluded and receives a new plan only after Stage B passes.

**Tech Stack:** Python 3.12, NumPy, ASE, MACE/torch CUDA, existing `pamssw` evaluator/accounting/walker APIs, pytest, JSON/JSONL evidence.

---

## Scope and file map

Create one bounded research package:

```text
runs/20260803-two-operator-population-gate/
  .gitignore                 generated GPU outputs only
  protocol.py                immutable cohort, pure labels, ledger checks, decisions
  replay_stage_a.py          zero-new-FE source verification and feasibility replay
  run_stage_b.py             exact shared-input direct/SSW paired execution
  analyze_stage_b.py         compact evidence and preregistered Stage-B decision
  conclusion.md              generated only after a completed gate
  evidence.json              compact tracked evidence after execution
  output/                    ignored raw Stage-A/Stage-B artifacts
tests/unit/
  test_two_operator_population_protocol.py
  test_two_operator_population_stage_a.py
  test_two_operator_population_stage_b.py
```

Do not modify `pamssw/walker.py`, configuration defaults, production presets,
archive matching, starter selectors, or optimizer implementations.  Reuse the
current research-only APIs `initial_direction_choice`, `UphillWalkTrace`,
`SurfaceWalker.relax_true_minimum`, `CartesianCoordinates`, and
`TangentVector`.

The frozen starter contexts are existing immutable artifacts:

```text
runs/20260801-uphill-action-observability/confirm-output/{system}/seed-00000049/bootstrap/minimum.xyz
runs/20260802-fixed-h4-h8-equal-budget/output/{system}/seed-00000049/h8/best_minimum.xyz
```

They are named `bootstrap` and `h8_best`.  `protocol.py` records and verifies
their SHA-256 values before Stage B.  Seeds are `55, 56, 57`; replacing a
missing or censored case with another seed is forbidden.

The action caps are censoring limits fixed before execution, not tuned search
parameters.  The `1,000`-FE direct cap exceeds the maximum `526` FE observed in
the locked G-UP0 explicit-quench corpus.  The `2,300`-FE SSW-plus-shared-prefix
cap exceeds three times the maximum `696` FE complete action in the two locked
U-O1 executions.  Eighteen reserved pair caps consume `59,400 <= 60,000` FE,
so the full cohort can be admitted without data-dependent budget reallocation.

### Task 1: Pure protocol and preregistered decisions

**Files:**
- Create: `runs/20260803-two-operator-population-gate/protocol.py`
- Create: `tests/unit/test_two_operator_population_protocol.py`

- [ ] **Step 1: Write the failing cohort and decision tests**

```python
# tests/unit/test_two_operator_population_protocol.py
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/20260803-two-operator-population-gate/protocol.py"


def load_protocol():
    spec = importlib.util.spec_from_file_location("_two_operator_protocol", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(system, starter, seed, family, *, certified=True, new=True, fe=100):
    return {
        "system": system,
        "starter_context": starter,
        "seed": seed,
        "operator_family": family,
        "certified": certified,
        "same_starter_basin": not new,
        "geometry_valid": True,
        "fragmented": False,
        "budget_censored": False,
        "landing_delta_eV": -1.0 if new else 0.0,
        "force_evaluations": fe,
        "purpose_counts": {"landing_true_quench": fe, "unattributed": 0},
    }


def test_case_matrix_is_exactly_18_pairs():
    protocol = load_protocol()
    matrix = protocol.case_matrix()
    assert len(matrix) == 18
    assert {item.system for item in matrix} == {"c60", "pdo", "cuo"}
    assert {item.starter_context for item in matrix} == {"bootstrap", "h8_best"}
    assert {item.seed for item in matrix} == {55, 56, 57}


def test_stage_b_passes_only_with_direct_viability_ssw_support_and_certificates():
    protocol = load_protocol()
    rows = []
    for system in protocol.SYSTEMS:
        for starter in protocol.STARTERS:
            for seed in protocol.SEEDS:
                direct_new = starter == "bootstrap"
                ssw_new = True
                rows.append(row(system, starter, seed, "direct", new=direct_new, fe=80))
                rows.append(row(system, starter, seed, "ssw", new=ssw_new, fe=300))
    decision = protocol.decide_stage_b(rows)
    assert decision["decision"] == "ADMIT_STAGE_C_DESIGN"
    assert decision["direct_viability_contexts"] == 3
    assert decision["ssw_exclusive_support_contexts"] == 3
    assert decision["numerical_acceptability"] is True


def test_stage_b_closes_when_ssw_has_no_exclusive_support():
    protocol = load_protocol()
    rows = []
    for item in protocol.case_matrix():
        rows.append(row(item.system, item.starter_context, item.seed, "direct", fe=80))
        rows.append(row(item.system, item.starter_context, item.seed, "ssw", fe=300))
    assert protocol.decide_stage_b(rows)["decision"] == "CLOSE_TWO_OPERATOR_PORTFOLIO"


def test_unattributed_work_fails_closed():
    protocol = load_protocol()
    bad = row("c60", "bootstrap", 55, "direct")
    bad["purpose_counts"] = {"landing_true_quench": 99, "unattributed": 1}
    try:
        protocol.validate_row(bad)
    except ValueError as exc:
        assert "unattributed" in str(exc)
    else:
        raise AssertionError("unattributed work was accepted")
```

- [ ] **Step 2: Run the tests and verify the protocol is absent**

Run:

```bash
pytest -q tests/unit/test_two_operator_population_protocol.py
```

Expected: collection or import failure because `protocol.py` does not exist.

- [ ] **Step 3: Implement the minimal immutable protocol**

```python
# runs/20260803-two-operator-population-gate/protocol.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")
STARTERS = ("bootstrap", "h8_best")
SEEDS = (55, 56, 57)
FAMILIES = ("direct", "ssw")
FIXED_ATOM_COUNTS = {"c60": 0, "pdo": 40, "cuo": 24}
MAX_FORCE_EVALUATIONS = 60_000
DIRECT_ACTION_CAP = 1_000
SSW_AND_SHARED_PREFIX_CAP = 2_300
PAIR_SUBMISSION_CAP = DIRECT_ACTION_CAP + SSW_AND_SHARED_PREFIX_CAP


@dataclass(frozen=True)
class CaseSpec:
    system: str
    starter_context: str
    seed: int


def case_matrix() -> tuple[CaseSpec, ...]:
    return tuple(
        CaseSpec(system, starter, seed)
        for system in SYSTEMS
        for starter in STARTERS
        for seed in SEEDS
    )


def validate_row(row: Mapping[str, Any]) -> None:
    if row["system"] not in SYSTEMS:
        raise ValueError("unknown system")
    if row["starter_context"] not in STARTERS:
        raise ValueError("unknown starter context")
    if int(row["seed"]) not in SEEDS:
        raise ValueError("unknown seed")
    if row["operator_family"] not in FAMILIES:
        raise ValueError("unknown operator family")
    counts = {str(k): int(v) for k, v in row["purpose_counts"].items()}
    if counts.get("unattributed", 0) != 0:
        raise ValueError("unattributed force evaluations are forbidden")
    if sum(counts.values()) != int(row["force_evaluations"]):
        raise ValueError("action purpose ledger does not close")
    fully_loaded = int(row.get("fully_loaded_force_evaluations", row["force_evaluations"]))
    if fully_loaded < int(row["force_evaluations"]):
        raise ValueError("fully loaded action cost is smaller than exclusive cost")
    if float(row.get("wall_time_s", 0.0)) < 0.0:
        raise ValueError("action wall time is negative")
    if float(row.get("fully_loaded_wall_time_s", row.get("wall_time_s", 0.0))) < float(
        row.get("wall_time_s", 0.0)
    ):
        raise ValueError("fully loaded wall time is smaller than exclusive wall time")


def _context_rows(rows: Sequence[Mapping[str, Any]]):
    grouped = defaultdict(list)
    for row in rows:
        validate_row(row)
        grouped[(row["system"], row["starter_context"], row["operator_family"])].append(row)
    return grouped


def decide_stage_b(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = len(case_matrix()) * len(FAMILIES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, observed {len(rows)}")
    keys = [
        (
            str(row["system"]),
            str(row["starter_context"]),
            int(row["seed"]),
            str(row["operator_family"]),
        )
        for row in rows
    ]
    if len(set(keys)) != expected:
        raise ValueError("Stage-B action keys are not unique")
    grouped = _context_rows(rows)
    direct_viability = 0
    ssw_support = 0
    for system in SYSTEMS:
        for starter in STARTERS:
            direct = grouped[(system, starter, "direct")]
            ssw = grouped[(system, starter, "ssw")]
            direct_new = sum(
                bool(row["certified"]) and not bool(row["same_starter_basin"])
                for row in direct
            )
            if direct_new >= 2 and median(
                row.get("fully_loaded_force_evaluations", row["force_evaluations"])
                for row in direct
            ) < median(
                row.get("fully_loaded_force_evaluations", row["force_evaluations"])
                for row in ssw
            ):
                direct_viability += 1
            exclusive = 0
            for seed in SEEDS:
                d = next(row for row in direct if int(row["seed"]) == seed)
                s = next(row for row in ssw if int(row["seed"]) == seed)
                direct_failed = (
                    not bool(d["certified"])
                    or bool(d["same_starter_basin"])
                    or bool(d["budget_censored"])
                )
                ssw_escaped = bool(s["certified"]) and not bool(s["same_starter_basin"])
                exclusive += int(direct_failed and ssw_escaped)
            if exclusive >= 2:
                ssw_support += 1
    certificate_counts = {
        (system, family): sum(
            bool(row["certified"])
            for row in rows
            if row["system"] == system and row["operator_family"] == family
        )
        for system in SYSTEMS
        for family in FAMILIES
    }
    numerical = all(value >= 5 for value in certificate_counts.values())
    passed = direct_viability >= 2 and ssw_support >= 1 and numerical
    return {
        "decision": "ADMIT_STAGE_C_DESIGN" if passed else "CLOSE_TWO_OPERATOR_PORTFOLIO",
        "direct_viability_contexts": direct_viability,
        "ssw_exclusive_support_contexts": ssw_support,
        "certificate_counts": {
            f"{system}:{family}": value
            for (system, family), value in sorted(certificate_counts.items())
        },
        "numerical_acceptability": numerical,
    }
```

- [ ] **Step 4: Run the protocol tests**

Run:

```bash
pytest -q tests/unit/test_two_operator_population_protocol.py
```

Expected: `4 passed`.

- [ ] **Step 5: Commit the protocol**

```bash
git add runs/20260803-two-operator-population-gate/protocol.py \
  tests/unit/test_two_operator_population_protocol.py
git commit -m "Add two-operator population gate protocol"
```

### Task 2: Stage A zero-new-FE replay

**Files:**
- Create: `runs/20260803-two-operator-population-gate/replay_stage_a.py`
- Create: `runs/20260803-two-operator-population-gate/.gitignore`
- Create: `tests/unit/test_two_operator_population_stage_a.py`

- [ ] **Step 1: Write source-integrity and zero-FE replay tests**

```python
# tests/unit/test_two_operator_population_stage_a.py
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/20260803-two-operator-population-gate"


def load_runner():
    path = RUN / "replay_stage_a.py"
    spec = importlib.util.spec_from_file_location("_two_operator_stage_a", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_replay_rejects_source_hash_drift(tmp_path):
    runner = load_runner()
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    try:
        runner.verify_source(source, "0" * 64)
    except RuntimeError as exc:
        assert "SHA256" in str(exc)
    else:
        raise AssertionError("drifted source was accepted")


def test_summarize_g_up0_records_no_new_force_evaluations():
    runner = load_runner()
    pairs = [
        {
            "system": "c60",
            "state_id": "plateau",
            "seed": 42,
            "causal_outcome": "RELAXED_ONLY_ESCAPE",
            "explicit_label": "RETURN_STARTER",
            "relaxed_label": "ESCAPED_CERTIFIED",
            "new_force_evaluations": 80,
            "explicit_purpose_counts": {
                "landing_true_quench": 79,
                "post_relax_validation": 1,
                "unattributed": 0,
            },
        }
    ]
    result = runner.summarize_pairs(pairs)
    assert result["new_force_evaluations"] == 0
    assert result["relaxed_only_escape_pairs"] == 1
    assert result["direct_quench_observed_force_evaluations"] == 80
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q tests/unit/test_two_operator_population_stage_a.py
```

Expected: import failure because `replay_stage_a.py` does not exist.

- [ ] **Step 3: Implement verified source loading and descriptive replay**

```python
# runs/20260803-two-operator-population-gate/replay_stage_a.py
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
G_UP0_SUMMARY = REPO_ROOT / "runs/20260731-uphill-relax-counterfactual-gate/evidence.json"
G_UP0_SUMMARY_SHA256 = "0a05e4bd7b1a4f1b83e288c47159a7faa1e15775181150cff1fdbb00e6f05a33"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_source(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if digest(path) != expected_sha256:
        raise RuntimeError(f"source SHA256 drifted: {path}")


def summarize_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcomes = Counter(str(row["causal_outcome"]) for row in pairs)
    direct_costs = [int(row["new_force_evaluations"]) for row in pairs]
    for row in pairs:
        counts = {str(k): int(v) for k, v in row["explicit_purpose_counts"].items()}
        if counts.get("unattributed", 0) != 0 or sum(counts.values()) != int(
            row["new_force_evaluations"]
        ):
            raise RuntimeError("G-UP0 direct ledger does not close")
    return {
        "new_force_evaluations": 0,
        "pair_count": len(pairs),
        "relaxed_only_escape_pairs": outcomes["RELAXED_ONLY_ESCAPE"],
        "explicit_only_escape_pairs": outcomes["EXPLICIT_ONLY_ESCAPE"],
        "same_escaped_landing_pairs": outcomes["SAME_ESCAPED_LANDING"],
        "different_escaped_landing_pairs": outcomes["DIFFERENT_ESCAPED_LANDINGS"],
        "direct_quench_observed_force_evaluations": sum(direct_costs),
        "direct_quench_median_force_evaluations": median(direct_costs),
    }


def run(output: Path) -> dict[str, Any]:
    verify_source(G_UP0_SUMMARY, G_UP0_SUMMARY_SHA256)
    summary = json.loads(G_UP0_SUMMARY.read_text(encoding="utf-8"))
    raw = Path(summary["raw_evidence_path"])
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    verify_source(raw, str(summary["raw_evidence_sha256"]))
    payload = json.loads(raw.read_text(encoding="utf-8"))
    replay = summarize_pairs(payload["pairs"])
    replay.update(
        {
            "schema_version": 1,
            "source_path": str(raw.relative_to(REPO_ROOT)),
            "source_sha256": digest(raw),
            "decision": (
                "ADMIT_FRESH_STAGE_B"
                if replay["relaxed_only_escape_pairs"] > 0
                and (
                    replay["same_escaped_landing_pairs"]
                    + replay["different_escaped_landing_pairs"]
                    + replay["explicit_only_escape_pairs"]
                ) > 0
                else "CLOSE_TWO_OPERATOR_PORTFOLIO"
            ),
            "claim_ceiling": (
                "descriptive zero-new-FE replay of the existing C60 G-UP0 corpus; "
                "full fresh action cost is deferred to Stage B"
            ),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "output/stage-a.json")
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create the output ignore file:

```gitignore
# runs/20260803-two-operator-population-gate/.gitignore
output/
smoke-output/
__pycache__/
```

- [ ] **Step 4: Run Stage A unit tests**

Run:

```bash
pytest -q tests/unit/test_two_operator_population_stage_a.py
```

Expected: `2 passed`.

- [ ] **Step 5: Run the zero-new-FE replay and inspect the decision**

Run:

```bash
python runs/20260803-two-operator-population-gate/replay_stage_a.py
```

Expected: exit `0`, `new_force_evaluations: 0`, a verified source SHA-256, and
either `ADMIT_FRESH_STAGE_B` or `CLOSE_TWO_OPERATOR_PORTFOLIO`.

If the decision is `CLOSE_TWO_OPERATOR_PORTFOLIO`, write the compact conclusion
in Task 6, skip Tasks 3--5, and do not create a batch-executor plan.

- [ ] **Step 6: Commit Stage A before any GPU work**

```bash
git add runs/20260803-two-operator-population-gate/.gitignore \
  runs/20260803-two-operator-population-gate/replay_stage_a.py \
  tests/unit/test_two_operator_population_stage_a.py
git commit -m "Add zero-FE two-operator feasibility replay"
```

### Task 3: Stage B exact paired action executor

**Condition:** Execute only when Stage A returns `ADMIT_FRESH_STAGE_B`.

**Files:**
- Create: `runs/20260803-two-operator-population-gate/run_stage_b.py`
- Create: `tests/unit/test_two_operator_population_stage_b.py`

- [ ] **Step 1: Write failing tests for exact starter and shared-input semantics**

```python
# tests/unit/test_two_operator_population_stage_b.py
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/20260803-two-operator-population-gate/run_stage_b.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("_two_operator_stage_b", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_positions_are_exact_sigma_direction_displacement():
    runner = load_runner()
    positions = np.zeros((2, 3))
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    observed = runner.direct_positions(positions, direction, 0.2)
    np.testing.assert_allclose(observed.reshape(-1), 0.2 * direction)


def test_starter_artifacts_are_frozen_for_all_systems():
    runner = load_runner()
    specs = runner.starter_artifacts()
    assert len(specs) == 6
    assert {(row.system, row.context) for row in specs} == {
        (system, context)
        for system in ("c60", "pdo", "cuo")
        for context in ("bootstrap", "h8_best")
    }
    assert all(row.path.is_file() for row in specs)


def test_locked_starter_restores_the_original_constraint_mask():
    runner = load_runner()
    c60 = runner.load_locked_starter("c60", "bootstrap", cuo_resources=None)
    pdo = runner.load_locked_starter("pdo", "bootstrap", cuo_resources=None)
    assert int(c60.fixed_mask.sum()) == 0
    assert int(pdo.fixed_mask.sum()) == 40
    assert pdo.pbc == (True, True, False)


def test_build_config_preserves_the_frozen_per_system_direction_width(tmp_path):
    runner = load_runner()
    observed = {
        system: runner.build_config(system, tmp_path / system, seed=55)
        for system in ("c60", "pdo", "cuo")
    }
    assert {system: cfg.oracle_candidates for system, cfg in observed.items()} == {
        "c60": 4,
        "pdo": 8,
        "cuo": 8,
    }
    assert all(cfg.max_steps_per_walk == 8 for cfg in observed.values())
    assert all(cfg.local_softening_scope == "oracle" for cfg in observed.values())


def test_split_shared_direction_cost_does_not_duplicate_hvp_work():
    runner = load_runner()
    pair = runner.split_pair_cost(
        shared_direction_fe=8,
        direct_counts={"landing_true_quench": 30, "unattributed": 0},
        ssw_counts={"direction_oracle": 24, "biased_proposal_relax": 90,
                    "landing_true_quench": 40, "unattributed": 0},
    )
    assert pair["shared_direction_force_evaluations"] == 8
    assert pair["direct_force_evaluations"] == 30
    assert pair["ssw_force_evaluations"] == 146
    assert pair["pair_force_evaluations"] == 184
```

- [ ] **Step 2: Run the tests and verify failure**

Run:

```bash
pytest -q tests/unit/test_two_operator_population_stage_b.py
```

Expected: import failure because `run_stage_b.py` does not exist.

- [ ] **Step 3: Implement the immutable artifact and pure-cost helpers**

Add these exact public helpers first:

```python
# beginning of run_stage_b.py
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
OBSERVABILITY = REPO_ROOT / "runs/20260801-uphill-action-observability/run_gate.py"
PROTOCOL = RUN_ROOT / "protocol.py"


@dataclass(frozen=True)
class StarterArtifact:
    system: str
    context: str
    path: Path
    sha256: str


def load_module(path: Path, name: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


observability = load_module(OBSERVABILITY, "_two_operator_observability")
protocol = load_module(PROTOCOL, "_two_operator_protocol_runner")


def make_calculator(system, cuo_resources):
    return observability.base.source.source._calculator(system, cuo_resources)


def load_locked_starter(system, context, cuo_resources):
    from pamssw.io import read_state

    artifact = next(
        row for row in starter_artifacts()
        if row.system == system and row.context == context
    )
    if sha256(artifact.path.read_bytes()).hexdigest() != artifact.sha256:
        raise RuntimeError(f"starter SHA256 drifted: {artifact.path}")
    coordinates = read_state(artifact.path)
    template = observability.base.source.source._load_state(system, cuo_resources)
    if not np.array_equal(coordinates.numbers, template.numbers):
        raise RuntimeError("starter atom ordering drifted from the frozen template")
    if coordinates.pbc != template.pbc:
        raise RuntimeError("starter PBC drifted from the frozen template")
    if coordinates.cell is None or template.cell is None:
        if coordinates.cell is not None or template.cell is not None:
            raise RuntimeError("starter cell rank drifted from the frozen template")
    elif not np.allclose(coordinates.cell, template.cell, rtol=0.0, atol=1.0e-10):
        raise RuntimeError("starter cell drifted from the frozen template")
    return template.with_flat_positions(coordinates.positions.reshape(-1))


def build_config(system, case_directory, *, seed):
    base = observability.build_config(
        system,
        Path(case_directory),
        seed=int(seed),
        force_budget=protocol.SSW_AND_SHARED_PREFIX_CAP,
    )
    return replace(
        base,
        max_trials=1,
        max_force_evals=protocol.SSW_AND_SHARED_PREFIX_CAP,
        max_steps_per_walk=8,
        direction_selection_mode="discrete",
        direction_ranking_mode="static_score",
        local_softening_scope="oracle",
        write_relaxation_trajectories=False,
        direction_diagnostics_enabled=False,
        proposal_pool_size=1,
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def starter_artifacts() -> tuple[StarterArtifact, ...]:
    hashes = {
        ("c60", "bootstrap"): "8c0ee78c143f9e285397a0fd5f6e2a0313b278811c8157afea30316f2b94fdc3",
        ("pdo", "bootstrap"): "97efa24720cd1cf9942e417f736b8988b123bacb61915a04a3825c0ba89925f4",
        ("cuo", "bootstrap"): "df3ee1d666d02ab0e3229d2aef210f56f9715ef9fbe3df56bbe714503b4a89a2",
        ("c60", "h8_best"): "15c3d6bf022382bf4da22144accb639cf72980c3c55e89ff658f336b86827424",
        ("pdo", "h8_best"): "b1aa09190b2e110e2ba2c6cd3264d234d148ec588155ad4ca29379c1343dcd4c",
        ("cuo", "h8_best"): "1d3a732ec470a8fe455950f604f70023bcc5cf6edae042f62970fe030175f771",
    }
    rows = []
    for system in ("c60", "pdo", "cuo"):
        rows.append(StarterArtifact(
            system,
            "bootstrap",
            REPO_ROOT / (
                "runs/20260801-uphill-action-observability/confirm-output/"
                f"{system}/seed-00000049/bootstrap/minimum.xyz"
            ),
            hashes[(system, "bootstrap")],
        ))
        rows.append(StarterArtifact(
            system,
            "h8_best",
            REPO_ROOT / (
                "runs/20260802-fixed-h4-h8-equal-budget/output/"
                f"{system}/seed-00000049/h8/best_minimum.xyz"
            ),
            hashes[(system, "h8_best")],
        ))
    return tuple(rows)


def direct_positions(positions, direction, sigma):
    return np.asarray(positions, dtype=float) + float(sigma) * np.asarray(
        direction, dtype=float
    ).reshape((-1, 3))


def split_pair_cost(*, shared_direction_fe, direct_counts, ssw_counts):
    direct = sum(int(v) for v in direct_counts.values())
    ssw_all = {str(k): int(v) for k, v in ssw_counts.items()}
    if ssw_all.get("direction_oracle", 0) < int(shared_direction_fe):
        raise ValueError("shared direction cost exceeds SSW direction ledger")
    ssw_all["direction_oracle"] -= int(shared_direction_fe)
    ssw = sum(ssw_all.values())
    return {
        "shared_direction_force_evaluations": int(shared_direction_fe),
        "direct_force_evaluations": direct,
        "ssw_force_evaluations": ssw,
        "pair_force_evaluations": int(shared_direction_fe) + direct + ssw,
    }


def counts_delta(after, before):
    after_map = after.as_dict()
    before_map = before.as_dict()
    delta = {name: int(after_map[name]) - int(before_map[name]) for name in after_map}
    if any(value < 0 for value in delta.values()):
        raise ValueError("evaluation counters moved backwards")
    return delta


def without_shared_direction(counts, shared_direction_fe):
    result = {str(k): int(v) for k, v in counts.items()}
    result["direction_oracle"] -= int(shared_direction_fe)
    if result["direction_oracle"] < 0:
        raise ValueError("shared direction cost exceeds action direction cost")
    return result


def landing_action_row(
    *, system, starter_context, seed, family, starter_state, starter_energy,
    landing, walker, counts, config, wall_time_s, budget_censored=False,
):
    from pamssw.archive import MinimaArchive
    from pamssw.relax import has_force_convergence_certificate

    counts = {str(k): int(v) for k, v in counts.items()}
    if landing is None:
        return {
            "system": system,
            "starter_context": starter_context,
            "seed": int(seed),
            "operator_family": family,
            "certified": False,
            "same_starter_basin": False,
            "geometry_valid": False,
            "fragmented": False,
            "budget_censored": bool(budget_censored),
            "landing_delta_eV": None,
            "improved_global_best": False,
            "force_evaluations": sum(counts.values()),
            "wall_time_s": float(wall_time_s),
            "purpose_counts": counts,
        }
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    starter_entry = archive.add(starter_state, float(starter_energy), parent_id=None)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=starter_entry.entry_id,
    )
    geometry_valid = bool(walker.geometry_validator.is_valid_state(landing.state))
    fragmented = bool(walker._is_fragmented_cluster(starter_state, landing.state))
    certified = bool(
        geometry_valid
        and not fragmented
        and has_force_convergence_certificate(landing, config.quench_fmax)
    )
    return {
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "operator_family": family,
        "certified": certified,
        "same_starter_basin": bool(landing_entry.entry_id == starter_entry.entry_id),
        "geometry_valid": geometry_valid,
        "fragmented": fragmented,
        "budget_censored": bool(budget_censored),
        "landing_delta_eV": float(landing.energy) - float(starter_energy),
        "improved_global_best": bool(float(landing.energy) < float(starter_energy)),
        "force_evaluations": sum(counts.values()),
        "wall_time_s": float(wall_time_s),
        "purpose_counts": counts,
    }


def serialize_pair(
    *, system, starter_context, seed, starter_state, starter_energy,
    direction, sigma, ssw_trace, ssw_landing, ssw_walker, ssw_counts,
    direct_landing, direct_walker, direct_counts, config, starter_artifact,
    starter_validation_counts, ssw_budget_censored=False,
    direct_budget_censored=False, starter_validation_wall_time_s=0.0,
    initial_direction_wall_time_s=0.0, ssw_wall_time_s=0.0,
    direct_wall_time_s=0.0,
):
    shared_direction_fe = int(
        ssw_trace.steps[0].direction_oracle_force_evaluations
    )
    ssw_exclusive = without_shared_direction(ssw_counts, shared_direction_fe)
    actions = [
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="direct",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=direct_landing,
            walker=direct_walker,
            counts=direct_counts,
            config=config,
            wall_time_s=direct_wall_time_s,
            budget_censored=direct_budget_censored,
        ),
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="ssw",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=ssw_landing,
            walker=ssw_walker,
            counts=ssw_exclusive,
            config=config,
            wall_time_s=max(0.0, ssw_wall_time_s - initial_direction_wall_time_s),
            budget_censored=ssw_budget_censored,
        ),
    ]
    for action in actions:
        action["fully_loaded_force_evaluations"] = (
            int(action["force_evaluations"]) + shared_direction_fe
        )
        action["fully_loaded_wall_time_s"] = (
            float(action["wall_time_s"]) + float(initial_direction_wall_time_s)
        )
    return {
        "schema_version": 1,
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "starter_path": str(starter_artifact.path.relative_to(REPO_ROOT)),
        "starter_sha256": starter_artifact.sha256,
        "fixed_atom_count": int(np.count_nonzero(starter_state.fixed_mask)),
        "direction_sha256": sha256(
            np.asarray(direction, dtype=np.float64).tobytes()
        ).hexdigest(),
        "execution_sigma": float(sigma),
        "shared_direction_force_evaluations": shared_direction_fe,
        "shared_initial_direction_wall_time_s": float(initial_direction_wall_time_s),
        "starter_validation_purpose_counts": dict(starter_validation_counts),
        "starter_validation_wall_time_s": float(starter_validation_wall_time_s),
        "actions": actions,
        "pair_force_evaluations": (
            sum(int(v) for v in starter_validation_counts.values())
            + shared_direction_fe
            + sum(int(row["force_evaluations"]) for row in actions)
        ),
        "pair_wall_time_s": (
            float(starter_validation_wall_time_s)
            + float(ssw_wall_time_s)
            + float(direct_wall_time_s)
        ),
        "effective_config": asdict(config),
    }


def write_prefix_censored_pair(
    *, system, starter_context, seed, starter_artifact, starter_state,
    starter_energy, starter_validation_counts, ssw_counts, ssw_walker, config,
    output_directory, starter_validation_wall_time_s, prefix_wall_time_s,
):
    from pamssw.io import write_state

    zero_counts = {name: 0 for name in ssw_counts}
    prefix_budget_exhausted = bool(ssw_walker.calculator.exhausted())
    actions = [
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="direct",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=None,
            walker=ssw_walker,
            counts=zero_counts,
            config=config,
            wall_time_s=0.0,
            budget_censored=prefix_budget_exhausted,
        ),
        landing_action_row(
            system=system,
            starter_context=starter_context,
            seed=seed,
            family="ssw",
            starter_state=starter_state,
            starter_energy=starter_energy,
            landing=None,
            walker=ssw_walker,
            counts=ssw_counts,
            config=config,
            wall_time_s=prefix_wall_time_s,
            budget_censored=prefix_budget_exhausted,
        ),
    ]
    for action in actions:
        action["fully_loaded_force_evaluations"] = int(action["force_evaluations"])
        action["fully_loaded_wall_time_s"] = float(action["wall_time_s"])
    row = {
        "schema_version": 1,
        "system": system,
        "starter_context": starter_context,
        "seed": int(seed),
        "starter_path": str(starter_artifact.path.relative_to(REPO_ROOT)),
        "starter_sha256": starter_artifact.sha256,
        "fixed_atom_count": int(np.count_nonzero(starter_state.fixed_mask)),
        "direction_sha256": None,
        "execution_sigma": None,
        "paired_input_censored": True,
        "shared_direction_force_evaluations": 0,
        "shared_initial_direction_wall_time_s": 0.0,
        "starter_validation_purpose_counts": dict(starter_validation_counts),
        "starter_validation_wall_time_s": float(starter_validation_wall_time_s),
        "actions": actions,
        "pair_force_evaluations": (
            sum(int(v) for v in starter_validation_counts.values())
            + sum(int(v) for v in ssw_counts.values())
        ),
        "pair_wall_time_s": (
            float(starter_validation_wall_time_s) + float(prefix_wall_time_s)
        ),
        "effective_config": asdict(config),
    }
    write_state(output_directory / "starter.xyz", starter_state)
    write_json(output_directory / "pair.json", row)
    return row
```

- [ ] **Step 4: Implement one exact shared-input pair without changing `pamssw`**

Use the existing observability loader/config/calculator helpers.  The executor
must perform these operations in order:

```python
def run_pair(*, system, starter_context, seed, output_directory, cuo_resources):
    from pamssw.accounting import BudgetExceeded, EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.coordinates import CartesianCoordinates, TangentVector
    from pamssw.io import read_state, write_state
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    artifact = next(
        row for row in starter_artifacts()
        if row.system == system and row.context == starter_context
    )
    starter = load_locked_starter(system, starter_context, cuo_resources)
    if int(np.count_nonzero(starter.fixed_mask)) != protocol.FIXED_ATOM_COUNTS[system]:
        raise RuntimeError("locked starter constraint count drifted")
    config = build_config(system, output_directory, seed=seed)
    ssw = SurfaceWalker(
        calculator=make_calculator(system, cuo_resources),
        config=config,
        softening_enabled=True,
    )
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    starter_validation_started = perf_counter()
    with ssw.calculator.purpose(EvaluationPurpose.STARTER_TRUE_QUENCH):
        starter_evaluation = ssw.calculator.evaluate(starter)
    starter_validation_wall_time_s = perf_counter() - starter_validation_started
    starter_state = starter
    starter_energy = float(starter_evaluation.energy)
    starter_max_force = float(
        np.max(np.linalg.norm(-starter_evaluation.gradient, axis=1)[starter.movable_mask])
    )
    if starter_max_force > config.quench_fmax:
        raise RuntimeError(
            f"locked starter lost its force certificate: {starter_max_force}"
        )
    starter_entry = archive.add(starter_state, starter_energy, parent_id=None)
    starter_validation_counts = ssw.calculator.snapshot().as_dict()

    captured = {}
    original_choose = ssw.oracle.choose_direction

    def capture_choice(*args, **kwargs):
        direction_started = perf_counter()
        choice = original_choose(*args, **kwargs)
        if "choice" not in captured:
            captured["choice"] = deepcopy(choice)
            captured["initial_direction_wall_time_s"] = (
                perf_counter() - direction_started
            )
        return choice

    ssw.oracle.choose_direction = capture_choice
    prefix_traces = []
    continuations = []
    before_ssw = ssw.calculator.snapshot()
    prefix_started = perf_counter()
    try:
        ssw._walk_candidate_from_seed(
            starter_state,
            archive,
            ssw.step_target_controller.target(archive),
            trial_index=0,
            proposal_index=0,
            seed_entry_id=starter_entry.entry_id,
            trace_sink=prefix_traces,
            pause_after_step=0,
            continuation_sink=continuations,
        )
    except BudgetExceeded:
        prefix_wall_time_s = perf_counter() - prefix_started
        return write_prefix_censored_pair(
            system=system,
            starter_context=starter_context,
            seed=seed,
            starter_artifact=artifact,
            starter_state=starter_state,
            starter_energy=starter_energy,
            starter_validation_counts=starter_validation_counts,
            ssw_counts=counts_delta(ssw.calculator.snapshot(), before_ssw),
            ssw_walker=ssw,
            config=config,
            output_directory=output_directory,
            starter_validation_wall_time_s=starter_validation_wall_time_s,
            prefix_wall_time_s=prefix_wall_time_s,
        )
    prefix_wall_time_s = perf_counter() - prefix_started
    if (
        len(prefix_traces) != 1
        or len(prefix_traces[0].steps) != 1
        or len(continuations) != 1
        or "choice" not in captured
    ):
        raise RuntimeError("SSW arm did not expose one complete shared input")
    first_step = prefix_traces[0].steps[0]
    direction = np.asarray(captured["choice"].direction, dtype=float)
    sigma = float(first_step.executed_sigma)

    direct_config = replace(config, max_force_evals=protocol.DIRECT_ACTION_CAP)
    direct = SurfaceWalker(
        calculator=make_calculator(system, cuo_resources),
        config=direct_config,
        softening_enabled=False,
    )
    direct_state = CartesianCoordinates.from_state(starter_state).displace(
        TangentVector(direction), sigma
    )
    direct_budget_censored = False
    direct_started = perf_counter()
    if not direct.geometry_validator.is_valid_state(direct_state):
        direct_landing = None
    else:
        try:
            direct_landing = direct.relax_true_minimum(
                direct_state,
                trajectory_name="direct-landing",
            )
        except BudgetExceeded:
            direct_landing = None
            direct_budget_censored = bool(direct.calculator.exhausted())
    direct_wall_time_s = perf_counter() - direct_started

    traces = []
    ssw_budget_censored = False
    continuation_started = perf_counter()
    try:
        escape = ssw._walk_candidate_from_seed(
            starter_state,
            archive,
            trial_index=0,
            proposal_index=0,
            seed_entry_id=starter_entry.entry_id,
            trace_sink=traces,
            continuation=continuations[0],
        )
        ssw_landing = ssw.relax_true_minimum(
            escape,
            trajectory_name="ssw-landing",
        )
    except BudgetExceeded:
        ssw_landing = None
        ssw_budget_censored = bool(ssw.calculator.exhausted())
        if not traces:
            traces = prefix_traces
    continuation_wall_time_s = perf_counter() - continuation_started
    after_ssw = ssw.calculator.snapshot()

    row = serialize_pair(
        system=system,
        starter_context=starter_context,
        seed=seed,
        starter_state=starter_state,
        starter_energy=starter_energy,
        direction=direction,
        sigma=sigma,
        ssw_trace=traces[0],
        ssw_landing=ssw_landing,
        ssw_walker=ssw,
        ssw_counts=counts_delta(after_ssw, before_ssw),
        direct_landing=direct_landing,
        direct_walker=direct,
        direct_counts=direct.calculator.snapshot().as_dict(),
        config=config,
        starter_artifact=artifact,
        starter_validation_counts=starter_validation_counts,
        ssw_budget_censored=ssw_budget_censored,
        direct_budget_censored=direct_budget_censored,
        starter_validation_wall_time_s=starter_validation_wall_time_s,
        initial_direction_wall_time_s=float(
            captured["initial_direction_wall_time_s"]
        ),
        ssw_wall_time_s=prefix_wall_time_s + continuation_wall_time_s,
        direct_wall_time_s=direct_wall_time_s,
    )
    write_state(output_directory / "starter.xyz", starter_state)
    if ssw_landing is not None:
        write_state(output_directory / "ssw-landing.xyz", ssw_landing.state)
    if direct_landing is not None:
        write_state(output_directory / "direct-landing.xyz", direct_landing.state)
    write_json(output_directory / "pair.json", row)
    return row
```

The pause/continuation boundary is deliberately after micro-step zero.  It adds
no PES evaluation: it only makes the already selected direction and executed
width durable before the independent direct arm runs.  Resuming from the
captured continuation restores the SSW RNG and physical state, so the SSW arm
retains the same micro-step sequence it would have had without the direct arm.

The `build_config()` implementation above is the complete allowed override
set.  The direct walker receives a second resolved config made with
`replace(config, max_force_evals=protocol.DIRECT_ACTION_CAP)`.  Before a pair
is submitted, the campaign runner reserves
`protocol.PAIR_SUBMISSION_CAP == 3_300` FE.  These are censoring caps, not
search controls; a cap hit remains a terminal budget-censored outcome.

`write_prefix_censored_pair()` is the only early terminal path: it emits two
action rows, assigns already incurred counts to the SSW arm, marks
`paired_input_censored=true`, and writes an atomic `pair.json`.  It marks
`budget_censored=true` only when the counter is actually exhausted; an invalid
prefix is invalid rather than mislabeled as a budget event.  Later
`BudgetExceeded` exceptions preserve each walker's actual snapshot and mark
only the affected arm budget-censored when its counter is exhausted.  An
invalid direct geometry is terminal invalid.  No invalid or censored arm is
assigned a landing.

- [ ] **Step 5: Add cohort budget and resume behavior**

The CLI must execute cases in `protocol.case_matrix()` order, persist each
`pair.json` atomically, and skip only a pair whose artifact passes schema and
hash validation.  Before submitting a new pair, require that the remaining
campaign budget is at least the configured pair cap; otherwise record both
arms as budget-censored and stop.  It must never replace a failed seed.

Run interface:

```bash
python runs/20260803-two-operator-population-gate/run_stage_b.py \
  --output runs/20260803-two-operator-population-gate/output \
  --max-force-evaluations 60000
```

- [ ] **Step 6: Run CPU analytic/unit tests**

Run:

```bash
pytest -q \
  tests/unit/test_two_operator_population_protocol.py \
  tests/unit/test_two_operator_population_stage_a.py \
  tests/unit/test_two_operator_population_stage_b.py
```

Expected: all tests pass without loading MACE or allocating CUDA.

- [ ] **Step 7: Run one GPU smoke pair**

Run:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python runs/20260803-two-operator-population-gate/run_stage_b.py \
  --output runs/20260803-two-operator-population-gate/smoke-output \
  --systems c60 --starters bootstrap --seeds 55 \
  --max-force-evaluations 4000
```

Expected: exit `0`; two terminal action rows; one shared starter/direction hash;
strictly closed pair and global ledgers; `unattributed == 0`.

- [ ] **Step 8: Commit the Stage B executor after smoke verification**

```bash
git add runs/20260803-two-operator-population-gate/run_stage_b.py \
  tests/unit/test_two_operator_population_stage_b.py
git commit -m "Add exact paired direct and SSW gate executor"
```

### Task 4: Stage B compact analysis

**Condition:** Execute only after the smoke pair passes.

**Files:**
- Create: `runs/20260803-two-operator-population-gate/analyze_stage_b.py`
- Modify: `tests/unit/test_two_operator_population_stage_b.py`

- [ ] **Step 1: Add failing aggregation tests**

Append tests that pass 36 synthetic action rows into
`analyze_stage_b.build_evidence()` and assert:

```python
evidence["purpose_ledger_closes"] is True
evidence["unattributed_force_evaluations"] == 0
evidence["decision"]["decision"] in {
    "ADMIT_STAGE_C_DESIGN",
    "CLOSE_TWO_OPERATOR_PORTFOLIO",
}
len(evidence["system_summaries"]) == 3
evidence["pooled_summary"]["action_count"] == 36
```

Also mutate one row so `force_evaluations != sum(purpose_counts.values())` and
assert that `build_evidence()` raises `ValueError`.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
pytest -q tests/unit/test_two_operator_population_stage_b.py
```

Expected: failure because `analyze_stage_b.py` does not exist.

- [ ] **Step 3: Implement deterministic aggregation**

`build_evidence(rows, provenance)` must:

1. sort rows by `(system, starter_context, seed, operator_family)`;
2. call `protocol.validate_row()` on every row;
3. verify exactly 36 unique action keys;
4. calculate per-system/per-family certificate, new-basin, global-improvement,
   landing-energy, exclusive FE/wall-time, and fully-loaded FE/wall-time
   summaries; the initial direction is added to each arm only for the
   fully-loaded comparison and counted once in the campaign ledger;
5. report direct-only and SSW-only paired outcomes;
6. call `protocol.decide_stage_b(rows)` without alternative thresholds;
7. include git commit, resolved-config hashes, model/input hashes, device,
   dtype, package versions, raw-directory hash manifest, and the 60,000-FE cap;
8. write JSON with `allow_nan=False` and atomic replacement.

The CLI is:

```bash
python runs/20260803-two-operator-population-gate/analyze_stage_b.py \
  --input runs/20260803-two-operator-population-gate/output \
  --output runs/20260803-two-operator-population-gate/output/evidence.json
```

- [ ] **Step 4: Run all focused tests**

Run:

```bash
pytest -q \
  tests/unit/test_two_operator_population_protocol.py \
  tests/unit/test_two_operator_population_stage_a.py \
  tests/unit/test_two_operator_population_stage_b.py
```

Expected: all pass.

- [ ] **Step 5: Commit the analyzer**

```bash
git add runs/20260803-two-operator-population-gate/analyze_stage_b.py \
  tests/unit/test_two_operator_population_stage_b.py
git commit -m "Add two-operator population gate analysis"
```

### Task 5: Execute the complete Stage B GPU gate

**Condition:** Execute only after Stage A admits it and the one-pair smoke passes.

**Files:**
- Generate ignored raw artifacts under: `runs/20260803-two-operator-population-gate/output/`

- [ ] **Step 1: Record preflight provenance**

Run:

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate mace_les
python -c "import torch, mace, ase, numpy, scipy; print(torch.__version__, torch.cuda.is_available(), mace.__version__, ase.__version__, numpy.__version__, scipy.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
git status --porcelain --untracked-files=no
```

Expected: CUDA available; the tracked worktree clean; versions captured in the
run log.  Stop if the tracked worktree is dirty.

- [ ] **Step 2: Execute the frozen 18-pair cohort**

Run:

```bash
python runs/20260803-two-operator-population-gate/run_stage_b.py \
  --output runs/20260803-two-operator-population-gate/output \
  --systems c60 pdo cuo \
  --starters bootstrap h8_best \
  --seeds 55 56 57 \
  --max-force-evaluations 60000
```

Expected: exit `0`; exactly 18 `pair.json` files or explicit terminal
budget-censor records; no replacement seeds; total attributed FE at most
60,000; `unattributed == 0`.

- [ ] **Step 3: Build compact evidence**

Run:

```bash
python runs/20260803-two-operator-population-gate/analyze_stage_b.py \
  --input runs/20260803-two-operator-population-gate/output \
  --output runs/20260803-two-operator-population-gate/output/evidence.json
```

Expected: one preregistered decision, three system summaries, 36 action rows,
closed ledgers, and no NaN/Infinity.

- [ ] **Step 4: Verify raw evidence mechanically**

Run:

```bash
python -m json.tool runs/20260803-two-operator-population-gate/output/evidence.json >/dev/null
pytest -q \
  tests/unit/test_two_operator_population_protocol.py \
  tests/unit/test_two_operator_population_stage_a.py \
  tests/unit/test_two_operator_population_stage_b.py
git diff --check
```

Expected: every command exits `0`.

### Task 6: Write and commit the scientific conclusion

**Files:**
- Create: `runs/20260803-two-operator-population-gate/conclusion.md`
- Create: `runs/20260803-two-operator-population-gate/evidence.json`

- [ ] **Step 1: Copy only compact reconstructable evidence**

Copy the analyzer output to the tracked compact path and verify the raw path and
SHA-256 manifest remain sufficient to reconstruct it:

```bash
cp runs/20260803-two-operator-population-gate/output/evidence.json \
  runs/20260803-two-operator-population-gate/evidence.json
python -m json.tool runs/20260803-two-operator-population-gate/evidence.json >/dev/null
```

- [ ] **Step 2: Write the conclusion without changing the decision**

The conclusion must state, in this order:

1. the exact Stage A and Stage B decisions;
2. whether direct viability, SSW exclusive support, and numerical acceptability
   passed separately;
3. system-level basin support and landing-energy distributions;
4. propagation, quench, and total FE by family;
5. whether direct moves were cheaper because of lower propagation cost or were
   offset by more expensive true quench;
6. the cross-system and finite-seed claim ceiling;
7. `production_default_changed: false`;
8. exactly one next action:
   - if admitted: write a separate Stage C batch-executor spec/plan;
   - if closed: retain the current SSW reference and return to the controlled
     single-coordinate propagator hypothesis.

Do not add a third operator, change thresholds, or reinterpret a failed gate as
a partial pass.

- [ ] **Step 3: Run final verification**

Run:

```bash
pytest -q \
  tests/unit/test_two_operator_population_protocol.py \
  tests/unit/test_two_operator_population_stage_a.py \
  tests/unit/test_two_operator_population_stage_b.py
git diff --check
git status --short
```

Expected: tests pass; only the intended compact evidence/conclusion and any
already-owned untracked experiment outputs are present.

- [ ] **Step 4: Commit the conclusion**

```bash
git add runs/20260803-two-operator-population-gate/evidence.json \
  runs/20260803-two-operator-population-gate/conclusion.md
git commit -m "Close two-operator population scientific gate"
```

## Stop conditions

- Stage A closes the portfolio: stop after Tasks 1, 2, and 6; perform no GPU
  execution.
- Stage B closes the portfolio: do not implement active-set batching, posterior
  allocation, MD, GA, CCQN, or rescue logic.
- Stage B admits the portfolio: write a new Stage C design and implementation
  plan; do not infer batching equivalence from the Stage B scalar execution.
- Any unattributed FE, source hash drift, missing predeclared starter, replacement
  seed, or non-reconstructable compact evidence invalidates the gate.
