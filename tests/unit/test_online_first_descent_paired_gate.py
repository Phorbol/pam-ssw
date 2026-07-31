from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260801-online-first-descent-paired-gate"
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
RUNNER_PATH = RUN_ROOT / "run_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load(PROTOCOL_PATH, "_online_first_descent_protocol_test")


def _runner():
    return _load(RUNNER_PATH, "_online_first_descent_runner_test")


def test_case_matrix_is_fresh_twenty_four_pair_design():
    protocol = _protocol()

    cases = protocol.case_matrix()

    assert len(cases) == 24
    assert {row["seed"] for row in cases} == {45, 46, 47}
    assert {row["system"] for row in cases} == {"c60", "pdo"}
    assert {row["state_id"] for row in cases} == {
        "intermediate_accepted",
        "plateau_accepted",
    }
    assert {row["arm"] for row in cases} == {
        "D0_exact_anchor",
        "K4_discrete",
    }


def test_first_descent_is_strict_at_existing_tolerance():
    protocol = _protocol()
    observer = protocol.FirstDescentObserver(
        starter_energy_eV=-10.0,
        tolerance_eV=0.001,
    )

    assert observer({"step": 1, "true_energy_eV": -10.001}) is None
    assert observer({"step": 2, "true_energy_eV": -10.002}) == (
        "true_energy_descent"
    )
    assert observer.trigger_step == 2
    assert len(observer.rows) == 2


def _prefix_row(step: int) -> dict[str, object]:
    return {
        "step": step,
        "selected_direction_sha256": f"direction-{step}",
        "executed_step_scale": 0.1 * step,
        "uphill_final_bias_weight": 1.0 * step,
        "true_energy_eV": -10.0 + step,
        "state_sha256": f"state-{step}",
    }


def test_prefix_comparison_is_exact_and_structured():
    protocol = _protocol()
    reference = [_prefix_row(1), _prefix_row(2), _prefix_row(3)]
    early = deepcopy(reference[:2])

    assert protocol.compare_prefix(reference, early) == {
        "prefix_valid": True,
        "prefix_length": 2,
        "mismatches": [],
    }

    early[1]["executed_step_scale"] = 0.2000000000001
    result = protocol.compare_prefix(reference, early)
    assert result["prefix_valid"] is False
    assert result["mismatches"] == [
        {
            "step": 2,
            "field": "executed_step_scale",
            "reference": 0.2,
            "early": 0.2000000000001,
        }
    ]


def _admissible_pairs() -> list[dict[str, object]]:
    rows = []
    for index, case in enumerate(_protocol().case_matrix()):
        triggered = index < 2
        rows.append(
            {
                **case,
                "prefix_valid": True,
                "triggered": triggered,
                "early_landing_certified": triggered,
                "early_landing_delta_eV": -1.0 if triggered else None,
                "dedup_energy_tol_eV": 0.001,
                "reference_complete_action_fe": 100,
                "early_complete_action_fe": 80 if triggered else 100,
            }
        )
    return rows


def test_decision_requires_every_preregistered_mechanism_condition():
    protocol = _protocol()
    pairs = _admissible_pairs()

    assert protocol.build_decision(pairs)["decision"] == (
        "ADMIT_EQUAL_BUDGET_G_E2"
    )

    broken = deepcopy(pairs)
    broken[0]["prefix_valid"] = False
    assert protocol.build_decision(broken)["decision"] == "DO_NOT_ADMIT_G_E2"

    broken = deepcopy(pairs)
    broken[1]["triggered"] = False
    assert protocol.build_decision(broken)["decision"] == "DO_NOT_ADMIT_G_E2"

    broken = deepcopy(pairs)
    broken[0]["early_landing_certified"] = False
    assert protocol.build_decision(broken)["decision"] == "DO_NOT_ADMIT_G_E2"

    broken = deepcopy(pairs)
    broken[0]["early_complete_action_fe"] = 130
    broken[1]["early_complete_action_fe"] = 70
    assert protocol.build_decision(broken)["decision"] == "DO_NOT_ADMIT_G_E2"


def test_reused_quench_is_charged_to_action_but_not_executed_twice():
    runner = _runner()
    generation = runner._zero_counts()
    generation["direction_oracle"] = 4
    generation["biased_proposal_relax"] = 20
    quench = runner._zero_counts()
    quench["escape_true_pes_check"] = 1
    quench["landing_true_quench"] = 9

    executed = runner.compose_arm_cost(
        generation_counts=generation,
        quench_counts=quench,
        quench_reused=True,
    )
    fresh = runner.compose_arm_cost(
        generation_counts=generation,
        quench_counts=quench,
        quench_reused=False,
    )

    assert executed["new_executed_force_evaluations"] == 24
    assert executed["complete_action_force_evaluations"] == 34
    assert executed["new_executed_purpose_counts"][
        "landing_true_quench"
    ] == 0
    assert fresh["new_executed_force_evaluations"] == 34
    assert fresh["complete_action_force_evaluations"] == 34


def test_prefix_trace_joins_already_recorded_step_evidence():
    runner = _runner()
    generation = {
        "direction_trace": [
            {
                "selected_direction_sha256": "direction-1",
                "executed_step_scale": 0.2,
                "uphill_final_bias_weight": 10.0,
            }
        ],
        "walk_step_trace": [
            {"step": 1, "true_energy_eV": -11.0}
        ],
    }

    rows = runner.build_prefix_trace(
        generation,
        state_hashes={1: "state-1"},
    )

    assert rows == [
        {
            "step": 1,
            "selected_direction_sha256": "direction-1",
            "executed_step_scale": 0.2,
            "uphill_final_bias_weight": 10.0,
            "true_energy_eV": -11.0,
            "state_sha256": "state-1",
        }
    ]
