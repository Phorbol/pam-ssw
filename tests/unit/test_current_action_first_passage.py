from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-current-action-first-passage"
    / "protocol.py"
)
RUNNER_PATH = PROTOCOL_PATH.with_name("run_gate.py")


def _protocol():
    spec = importlib.util.spec_from_file_location(
        "_current_action_first_passage_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runner():
    spec = importlib.util.spec_from_file_location(
        "_current_action_first_passage_runner",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_matrix_is_the_locked_twenty_four_trajectory_design() -> None:
    protocol = _protocol()

    cases = protocol.case_matrix()

    assert len(cases) == 24
    assert {case["system"] for case in cases} == {"c60", "pdo"}
    assert {case["state_id"] for case in cases} == {
        "intermediate_accepted",
        "plateau_accepted",
    }
    assert {case["seed"] for case in cases} == {42, 43, 44}
    assert {case["arm"] for case in cases} == {
        "D0_exact_anchor",
        "K4_discrete",
    }
    assert protocol.CHECKPOINT_HORIZONS == (1, 2, 4, 8)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"budget_exhausted": True}, "BUDGET_EXHAUSTED"),
        ({"fragmented": True}, "FRAGMENTED"),
        ({"geometry_valid": False}, "INVALID_GEOMETRY"),
        ({"certificate": False}, "QUENCH_UNCONVERGED"),
        (
            {"matcher_same": True, "descriptor_same": False},
            "AMBIGUOUS_MATCH",
        ),
        (
            {"matcher_same": False, "descriptor_same": True},
            "AMBIGUOUS_MATCH",
        ),
        (
            {"matcher_same": True, "descriptor_same": True},
            "RETURN_STARTER",
        ),
        (
            {"matcher_same": False, "descriptor_same": False},
            "ESCAPED_CERTIFIED",
        ),
    ],
)
def test_checkpoint_label_is_mechanistic_and_priority_ordered(
    kwargs,
    expected,
) -> None:
    protocol = _protocol()
    defaults = {
        "budget_exhausted": False,
        "fragmented": False,
        "geometry_valid": True,
        "certificate": True,
        "matcher_same": True,
        "descriptor_same": True,
    }
    defaults.update(kwargs)

    assert protocol.classify_checkpoint(**defaults) == expected


def _checkpoint(horizon: int, label: str) -> dict[str, object]:
    return {
        "horizon": horizon,
        "label": label,
        "force_evaluations": 10,
        "purpose_counts": {
            "bootstrap_true_quench": 0,
            "starter_true_quench": 0,
            "local_softening_pre_relax": 0,
            "direction_oracle": 0,
            "escape_true_pes_check": 1,
            "biased_proposal_relax": 0,
            "landing_true_quench": 8,
            "post_relax_validation": 1,
            "unattributed": 0,
        },
    }


def test_trajectory_flags_require_observed_h8_and_do_not_impute() -> None:
    protocol = _protocol()
    overshoot = [
        _checkpoint(1, "RETURN_STARTER"),
        _checkpoint(2, "ESCAPED_CERTIFIED"),
        _checkpoint(4, "ESCAPED_CERTIFIED"),
        _checkpoint(8, "RETURN_STARTER"),
    ]
    truncated = overshoot[:-1]

    assert protocol.summarize_trajectory(overshoot) == {
        "early_escape_then_h8_return": True,
        "all_horizons_return_starter": False,
        "learnable": True,
        "observed_horizons": [1, 2, 4, 8],
    }
    assert protocol.summarize_trajectory(truncated)[
        "early_escape_then_h8_return"
    ] is False
    assert protocol.summarize_trajectory(truncated)[
        "all_horizons_return_starter"
    ] is False


def _case(case: dict[str, object], labels: list[str]) -> dict[str, object]:
    checkpoints = [
        _checkpoint(horizon, label)
        for horizon, label in zip((1, 2, 4, 8), labels, strict=True)
    ]
    return {
        **case,
        "status": "completed",
        "generation_force_evaluations": 20,
        "generation_purpose_counts": {
            "bootstrap_true_quench": 0,
            "starter_true_quench": 0,
            "local_softening_pre_relax": 0,
            "direction_oracle": 8,
            "escape_true_pes_check": 2,
            "biased_proposal_relax": 10,
            "landing_true_quench": 0,
            "post_relax_validation": 0,
            "unattributed": 0,
        },
        "checkpoints": checkpoints,
        "trajectory_summary": None,
    }


def test_gate_opens_only_for_repeated_same_context_mechanisms() -> None:
    protocol = _protocol()
    rows = []
    for case in protocol.case_matrix():
        labels = ["ESCAPED_CERTIFIED"] * 3 + ["RETURN_STARTER"]
        if case["system"] == "pdo":
            labels = ["RETURN_STARTER"] * 4
        row = _case(case, labels)
        row["trajectory_summary"] = protocol.summarize_trajectory(
            row["checkpoints"]
        )
        rows.append(row)

    evidence = protocol.build_evidence(rows, max_force_evaluations=15_000)

    assert len(evidence["horizon_gate_contexts"]) == 4
    assert len(evidence["action_support_gap_contexts"]) == 2
    assert evidence["numerical_matcher_gate_systems"] == []
    assert evidence["total_force_evaluations"] == 24 * 60


def test_two_ambiguous_trajectories_route_system_to_matcher_gate() -> None:
    protocol = _protocol()
    rows = []
    for case in protocol.case_matrix():
        row = _case(case, ["RETURN_STARTER"] * 4)
        row["trajectory_summary"] = protocol.summarize_trajectory(
            row["checkpoints"]
        )
        rows.append(row)
    corrupted = deepcopy(rows)
    for index in (0, 1):
        corrupted[index]["checkpoints"][0]["label"] = "AMBIGUOUS_MATCH"
        corrupted[index]["trajectory_summary"] = protocol.summarize_trajectory(
            corrupted[index]["checkpoints"]
        )

    evidence = protocol.build_evidence(
        corrupted,
        max_force_evaluations=15_000,
    )

    assert evidence["numerical_matcher_gate_systems"] == ["c60"]


def test_evidence_rejects_unclosed_or_over_budget_runs() -> None:
    protocol = _protocol()
    rows = []
    for case in protocol.case_matrix():
        row = _case(case, ["RETURN_STARTER"] * 4)
        row["trajectory_summary"] = protocol.summarize_trajectory(
            row["checkpoints"]
        )
        rows.append(row)

    broken = deepcopy(rows)
    broken[0]["generation_force_evaluations"] = 19
    with pytest.raises(ValueError, match="generation ledger"):
        protocol.build_evidence(broken, max_force_evaluations=15_000)

    with pytest.raises(ValueError, match="budget"):
        protocol.build_evidence(rows, max_force_evaluations=1_000)


def test_checkpoint_selection_uses_only_preregistered_reachable_horizons() -> None:
    runner = _runner()

    selected = runner.select_reachable_checkpoints(
        ["h1", "h2", "h3", "h4", "h5", "h6"],
    )

    assert selected == [(1, "h1"), (2, "h2"), (4, "h4")]


def test_checkpoint_selection_includes_h8_only_when_reached() -> None:
    runner = _runner()

    selected = runner.select_reachable_checkpoints(
        [f"h{index}" for index in range(1, 9)],
    )

    assert selected == [(1, "h1"), (2, "h2"), (4, "h4"), (8, "h8")]


def test_first_relaxed_geometry_failure_is_an_invalid_h1_attempt() -> None:
    runner = _runner()

    def no_endpoint_match(_attempts, _endpoint, *, tolerance):
        assert tolerance == pytest.approx(1.0e-8)
        raise RuntimeError("proposal endpoint does not match")

    accepted, errors, failed = runner.partition_checkpoint_attempts(
        ["invalid-relaxation"],
        "starter-endpoint",
        termination_reason="relaxed_geometry_invalid",
        _prefix_resolver=no_endpoint_match,
    )

    assert accepted == []
    assert errors == [None]
    assert failed == "invalid-relaxation"


def test_explicit_geometry_failure_without_optimizer_trace_is_invalid_h1() -> None:
    runner = _runner()

    def no_endpoint_match(_attempts, _endpoint, *, tolerance):
        raise RuntimeError("proposal endpoint does not match")

    accepted, errors, failed = runner.partition_checkpoint_attempts(
        ["reconstructed-explicit-trial"],
        "starter-endpoint",
        termination_reason="explicit_geometry_invalid",
        _prefix_resolver=no_endpoint_match,
    )

    assert accepted == []
    assert errors == [None]
    assert failed == "reconstructed-explicit-trial"


def test_unmatched_endpoint_is_not_hidden_for_other_terminations() -> None:
    runner = _runner()

    def no_endpoint_match(_attempts, _endpoint, *, tolerance):
        raise RuntimeError("proposal endpoint does not match")

    with pytest.raises(RuntimeError, match="does not match"):
        runner.partition_checkpoint_attempts(
            ["attempt"],
            "endpoint",
            termination_reason="walk_displacement_clipped",
            _prefix_resolver=no_endpoint_match,
        )
