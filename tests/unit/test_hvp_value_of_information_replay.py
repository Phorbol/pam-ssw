from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-hvp-value-of-information-replay"
    / "protocol.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_hvp_value_of_information_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repeat_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for system in ("c60", "pdo"):
        for state_id in ("intermediate_accepted", "plateau_accepted"):
            for seed in (42, 43, 44):
                for candidate_index in range(4):
                    rows.append(
                        {
                            "system": system,
                            "state_id": state_id,
                            "seed": seed,
                            "candidate_index": candidate_index,
                            "kind": (
                                "bond" if candidate_index < 2 else "random"
                            ),
                            "static_rank": candidate_index + 1,
                            "certificate_first": True,
                            "certificate_second": True,
                            "landing_geometry_valid_first": True,
                            "landing_geometry_valid_second": True,
                            "landing_delta_eV_first": float(candidate_index),
                            "landing_delta_eV_second": float(candidate_index + 2),
                            "force_evaluations_first": 100 + candidate_index,
                            "force_evaluations_second": 102 + candidate_index,
                        }
                    )
    return rows


def test_consolidate_repeats_uses_only_paired_certified_landings() -> None:
    protocol = _load_protocol()
    rows = _repeat_rows()
    rows[0]["certificate_second"] = False

    consolidated = protocol.consolidate_repeats(rows)

    first = consolidated[0]
    assert first["repeat_certified"] is False
    assert first["mean_landing_delta_eV"] == pytest.approx(1.0)
    assert first["mean_force_evaluations"] == pytest.approx(101.0)


def test_family_rotation_is_fixed_before_outcomes_and_balanced_per_system() -> None:
    protocol = _load_protocol()

    schedule = protocol.family_rotation_schedule()

    assert len(schedule) == 12
    for system in ("c60", "pdo"):
        families = [
            family
            for (row_system, _state_id, _seed), family in schedule.items()
            if row_system == system
        ]
        assert families.count("bond") == 3
        assert families.count("random") == 3
    assert schedule[("c60", "intermediate_accepted", 42)] == "bond"
    assert schedule[("c60", "intermediate_accepted", 43)] == "random"
    assert schedule[("c60", "plateau_accepted", 42)] == "random"


def test_group_summary_separates_quality_from_initial_hvp_cost() -> None:
    protocol = _load_protocol()
    rows = [
        {
            "candidate_index": index,
            "kind": "bond" if index < 2 else "random",
            "static_rank": index + 1,
            "repeat_certified": True,
            "mean_landing_delta_eV": value,
            "mean_force_evaluations": cost,
        }
        for index, value, cost in (
            (0, 4.0, 100.0),
            (1, 0.0, 120.0),
            (2, 2.0, 80.0),
            (3, 6.0, 140.0),
        )
    ]

    summary = protocol.summarize_group(
        rows,
        rotated_family="bond",
        initial_pool_hvp_cost=8,
    )

    assert summary["static_k4"]["terminal_regret_eV"] == pytest.approx(4.0)
    assert summary["static_k4"]["projected_force_evaluations"] == pytest.approx(
        108.0
    )
    assert summary["uniform_no_hvp"]["terminal_regret_eV"] == pytest.approx(3.0)
    assert summary["uniform_no_hvp"]["projected_force_evaluations"] == pytest.approx(
        110.0
    )
    assert summary["family_rotation_no_hvp"]["terminal_regret_eV"] == pytest.approx(
        2.0
    )
    assert summary["family_rotation_no_hvp"][
        "projected_force_evaluations"
    ] == pytest.approx(110.0)


def test_gate_requires_both_quality_and_cost_on_each_system() -> None:
    protocol = _load_protocol()
    system_summaries = {
        "c60": {
            "static_k4": {
                "median_terminal_regret_eV": 2.0,
                "median_projected_force_evaluations": 200.0,
            },
            "uniform_no_hvp": {
                "median_terminal_regret_eV": 1.5,
                "median_projected_force_evaluations": 190.0,
            },
            "family_rotation_no_hvp": {
                "median_terminal_regret_eV": 2.5,
                "median_projected_force_evaluations": 180.0,
            },
        },
        "pdo": {
            "static_k4": {
                "median_terminal_regret_eV": 1.0,
                "median_projected_force_evaluations": 100.0,
            },
            "uniform_no_hvp": {
                "median_terminal_regret_eV": 1.0,
                "median_projected_force_evaluations": 90.0,
            },
            "family_rotation_no_hvp": {
                "median_terminal_regret_eV": 0.5,
                "median_projected_force_evaluations": 110.0,
            },
        },
    }

    decision = protocol.evaluate_gate(system_summaries)

    assert decision["uniform_no_hvp"]["cross_system_pass"] is True
    assert decision["family_rotation_no_hvp"]["cross_system_pass"] is False
    assert decision["live_selected_only_hvp_gate_allowed"] is True


def test_invalid_group_is_excluded_instead_of_imputed() -> None:
    protocol = _load_protocol()
    rows = protocol.consolidate_repeats(_repeat_rows())
    rows[0]["repeat_certified"] = False

    result = protocol.analyze_candidate_repeats(rows)

    assert result["quality"]["eligible_group_count"] == 11
    assert result["quality"]["excluded_group_count"] == 1
    assert result["quality"]["complete_expected_group_count"] == 12
