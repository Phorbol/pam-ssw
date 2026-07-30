from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-candidate-counterfactual-gate"
    / "protocol.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_direction_candidate_counterfactual_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_matrix_is_the_preregistered_48_case_design() -> None:
    protocol = _load_protocol()

    cases = protocol.case_matrix()

    assert len(cases) == 48
    assert {case["system"] for case in cases} == {"c60", "pdo"}
    assert {case["state_id"] for case in cases} == {
        "intermediate_accepted",
        "plateau_accepted",
    }
    assert {case["seed"] for case in cases} == {42, 43, 44}
    for group in protocol.group_cases(cases).values():
        assert [case["candidate_index"] for case in group] == [0, 1, 2, 3]


def test_summarize_group_measures_static_winner_regret_without_thresholds() -> None:
    protocol = _load_protocol()
    rows = [
        {
            "system": "c60",
            "state_id": "plateau_accepted",
            "seed": 42,
            "candidate_index": 0,
            "static_rank": 1,
            "static_score": 3.0,
            "landing_delta_eV": -0.2,
            "certificate": True,
            "landing_geometry_valid": True,
        },
        {
            "system": "c60",
            "state_id": "plateau_accepted",
            "seed": 42,
            "candidate_index": 1,
            "static_rank": 2,
            "static_score": 2.0,
            "landing_delta_eV": -0.8,
            "certificate": True,
            "landing_geometry_valid": True,
        },
        {
            "system": "c60",
            "state_id": "plateau_accepted",
            "seed": 42,
            "candidate_index": 2,
            "static_rank": 3,
            "static_score": 1.0,
            "landing_delta_eV": 0.1,
            "certificate": False,
            "landing_geometry_valid": True,
        },
        {
            "system": "c60",
            "state_id": "plateau_accepted",
            "seed": 42,
            "candidate_index": 3,
            "static_rank": 4,
            "static_score": 0.0,
            "landing_delta_eV": -1.0,
            "certificate": True,
            "landing_geometry_valid": False,
        },
    ]

    summary = protocol.summarize_group(rows)

    assert summary["static_winner_candidate_index"] == 0
    assert summary["best_valid_candidate_index"] == 1
    assert summary["static_winner_regret_eV"] == pytest.approx(0.6)
    assert summary["static_winner_is_best_valid"] is False
    assert summary["valid_candidate_count"] == 2
    assert summary["static_score_terminal_spearman"] == pytest.approx(-1.0)


def test_classify_gate_distinguishes_selection_generation_and_ambiguous() -> None:
    protocol = _load_protocol()

    selection = protocol.classify_gate(
        [
            {
                "static_winner_is_best_valid": False,
                "static_winner_regret_eV": 0.4,
                "static_score_terminal_spearman": -0.2,
            },
            {
                "static_winner_is_best_valid": False,
                "static_winner_regret_eV": 0.1,
                "static_score_terminal_spearman": 0.0,
            },
        ]
    )
    generation = protocol.classify_gate(
        [
            {
                "static_winner_is_best_valid": True,
                "static_winner_regret_eV": 0.0,
                "static_score_terminal_spearman": 0.8,
            },
            {
                "static_winner_is_best_valid": True,
                "static_winner_regret_eV": 0.0,
                "static_score_terminal_spearman": 0.6,
            },
        ]
    )
    ambiguous = protocol.classify_gate(
        [
            {
                "static_winner_is_best_valid": False,
                "static_winner_regret_eV": 0.2,
                "static_score_terminal_spearman": 0.7,
            },
            {
                "static_winner_is_best_valid": True,
                "static_winner_regret_eV": 0.0,
                "static_score_terminal_spearman": -0.2,
            },
        ]
    )

    assert selection["classification"] == "selection_bottleneck"
    assert selection["posterior_stage_allowed"] is True
    assert generation["classification"] == "candidate_generation_bottleneck"
    assert generation["posterior_stage_allowed"] is False
    assert ambiguous["classification"] == "ambiguous"
    assert ambiguous["posterior_stage_allowed"] is False
