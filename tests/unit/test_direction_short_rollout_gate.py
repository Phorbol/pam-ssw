from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-short-rollout-gate"
    / "protocol.py"
)
ANALYZE_PATH = PROTOCOL_PATH.with_name("analyze_repeats.py")


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_direction_short_rollout_protocol_test",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "_direction_short_rollout_analyzer_test",
        ANALYZE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_matrix_is_two_system_two_state_three_seed_two_arm_two_horizon() -> None:
    protocol = _load_protocol()
    cases = protocol.case_matrix()

    assert len(cases) == 48
    assert {case["system"] for case in cases} == {"c60", "pdo"}
    assert {case["state_id"] for case in cases} == {
        "intermediate_accepted",
        "plateau_accepted",
    }
    assert {case["seed"] for case in cases} == {42, 43, 44}
    assert {case["arm"] for case in cases} == {
        "static_score",
        "true_curvature",
    }
    assert {case["horizon"] for case in cases} == {1, 2}


def _probe_rows(protocol, *, reverse_pdo: bool = False):
    rows = []
    for repeat in (1, 2):
        for system in protocol.SYSTEMS:
            for state_id in protocol.STATE_IDS:
                for seed in protocol.SEEDS:
                    group_index = (
                        protocol.STATE_IDS.index(state_id)
                        * len(protocol.SEEDS)
                        + protocol.SEEDS.index(seed)
                    )
                    true_should_win = group_index % 2 == 0
                    if reverse_pdo and system == "pdo":
                        true_should_win = not true_should_win
                    for horizon in protocol.HORIZONS:
                        for arm in protocol.ARMS:
                            energy = (
                                2.0
                                if (arm == "true_curvature") == true_should_win
                                else 1.0
                            )
                            rows.append(
                                {
                                    "repeat_id": repeat,
                                    "system": system,
                                    "state_id": state_id,
                                    "seed": seed,
                                    "arm": arm,
                                    "horizon": horizon,
                                    "escape_delta_eV": energy,
                                    "force_evaluations": 40,
                                    "wall_time_s": 1.0,
                                    "geometry_valid": True,
                                    "fragmented": False,
                                    "first_direction_matches_terminal": True,
                                    "direction_step0_force_evaluations": 8,
                                }
                            )
    return rows


def _terminal_rows(protocol):
    rows = []
    for repeat in (1, 2):
        for system in protocol.SYSTEMS:
            for state_id in protocol.STATE_IDS:
                for seed in protocol.SEEDS:
                    group_index = (
                        protocol.STATE_IDS.index(state_id)
                        * len(protocol.SEEDS)
                        + protocol.SEEDS.index(seed)
                    )
                    true_should_win = group_index % 2 == 0
                    for arm in protocol.ARMS:
                        rows.append(
                            {
                                "repeat_id": repeat,
                                "system": system,
                                "state_id": state_id,
                                "seed": seed,
                                "arm": arm,
                                "landing_delta_eV": (
                                    -2.0
                                    if (arm == "true_curvature")
                                    == true_should_win
                                    else -1.0
                                ),
                                "force_evaluations": 200,
                                "certificate": True,
                                "landing_geometry_valid": True,
                            }
                        )
    return rows


def test_primary_h2_gate_requires_same_predictive_rule_in_both_systems() -> None:
    protocol = _load_protocol()
    result = protocol.summarize_campaign(
        _probe_rows(protocol),
        _terminal_rows(protocol),
    )

    assert result["online_racing_stage_allowed"] is True
    assert result["primary_horizon"] == 2
    assert result["by_system"]["c60"]["prediction_accuracy"] == 1.0
    assert result["by_system"]["c60"]["lower_rise_prediction_accuracy"] == 0.0
    assert result["by_system"]["pdo"]["median_regret_eV"] == 0.0
    assert result["by_system"]["pdo"]["mean_difference_vs_static_eV"] < 0.0

    result = protocol.summarize_campaign(
        _probe_rows(protocol, reverse_pdo=True),
        _terminal_rows(protocol),
    )
    assert result["by_system"]["c60"]["gate_passed"] is True
    assert result["by_system"]["pdo"]["gate_passed"] is False
    assert result["online_racing_stage_allowed"] is False


def test_uncertified_terminal_or_mismatched_direction_is_rejected() -> None:
    protocol = _load_protocol()
    probes = _probe_rows(protocol)
    terminals = _terminal_rows(protocol)
    probes[0]["first_direction_matches_terminal"] = False

    try:
        protocol.summarize_campaign(probes, terminals)
    except ValueError as exc:
        assert "direction" in str(exc)
    else:
        raise AssertionError("mismatched first direction was accepted")

    probes = _probe_rows(protocol)
    terminals[0]["certificate"] = False
    try:
        protocol.summarize_campaign(probes, terminals)
    except ValueError as exc:
        assert "certificate" in str(exc)
    else:
        raise AssertionError("uncertified terminal label was accepted")


def test_terminal_labels_are_tagged_with_their_repeat_identity() -> None:
    analyzer = _load_analyzer()
    first = {
        "rows": [
            {
                "system": "c60",
                "state_id": "plateau_accepted",
                "seed": 42,
                "arm": "static_score",
            }
        ]
    }
    second = {
        "rows": [
            {
                "system": "pdo",
                "state_id": "intermediate_accepted",
                "seed": 43,
                "arm": "true_curvature",
            }
        ]
    }

    rows = analyzer._terminal_rows(first, second)

    assert rows[0]["repeat_id"] == 1
    assert rows[1]["repeat_id"] == 2
