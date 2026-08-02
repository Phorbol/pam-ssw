from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "20260802-fixed-h4-h8-equal-budget"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load(RUN_ROOT / "protocol.py", "_fixed_h4_h8_protocol_test")


def test_case_matrix_is_ordered_three_system_h4_h8_gate() -> None:
    assert protocol.case_matrix() == [
        {"system": system, "seed": 49, "horizon": horizon}
        for system in ("c60", "pdo", "cuo")
        for horizon in (4, 8)
    ]


def test_gain_auc_integrates_best_energy_over_complete_budget() -> None:
    value = protocol.gain_auc(
        initial_energy_eV=-10.0,
        accepted_rows=[
            {"force_evaluations": 5_000, "best_energy": -12.0},
            {"force_evaluations": 15_000, "best_energy": -13.0},
        ],
        total_force_budget=20_000,
    )

    assert value == pytest.approx((10_000 * 2.0 + 5_000 * 3.0) / 20_000)


def test_scientific_config_differences_allow_only_horizon_and_output_paths() -> None:
    h4 = {
        "max_steps_per_walk": 4,
        "max_force_evals": 19_900,
        "accepted_structures_log": "/tmp/h4/accepted.jsonl",
        "proposal_optimizer": "safe-lbfgs-total",
        "oracle_candidates": 4,
    }
    h8 = {
        **h4,
        "max_steps_per_walk": 8,
        "accepted_structures_log": "/tmp/h8/accepted.jsonl",
    }
    assert protocol.scientific_config_differences(h4, h8) == {
        "max_steps_per_walk": [4, 8]
    }

    changed = {**h8, "oracle_candidates": 8}
    with pytest.raises(ValueError, match="oracle_candidates"):
        protocol.scientific_config_differences(h4, changed)


def _rows(deltas: tuple[float, float, float]):
    rows = []
    for system, delta in zip(("c60", "pdo", "cuo"), deltas):
        rows.extend(
            [
                {
                    "system": system,
                    "seed": 49,
                    "horizon": 4,
                    "gain_auc_eV": 5.0 + delta,
                    "strict_landing_certificate_rate": 1.0,
                    "strict_landing_failure_count": 0,
                },
                {
                    "system": system,
                    "seed": 49,
                    "horizon": 8,
                    "gain_auc_eV": 5.0,
                    "strict_landing_certificate_rate": 1.0,
                    "strict_landing_failure_count": 0,
                },
            ]
        )
    return rows


def test_decision_advances_only_two_system_positive_median_without_regression() -> None:
    passed = protocol.cohort_decision(_rows((1.0, 0.5, -0.1)))
    assert passed["decision"] == "ADMIT_H4_REPEAT_GATE"
    assert passed["h4_winning_systems"] == ["c60", "pdo"]
    assert passed["median_h4_minus_h8_gain_auc_eV"] == pytest.approx(0.5)

    failed = protocol.cohort_decision(_rows((1.0, -0.1, -0.2)))
    assert failed["decision"] == "RETAIN_H8_STOP_SHORT_HORIZON_BRANCH"

    invalid = _rows((1.0, 0.5, -0.1))
    invalid[0]["strict_landing_failure_count"] = 1
    assert (
        protocol.cohort_decision(invalid)["decision"]
        == "RETAIN_H8_STOP_SHORT_HORIZON_BRANCH"
    )


def test_equal_budget_decision_uses_failure_count_not_attempt_denominator() -> None:
    rows = _rows((1.0, 0.5, 0.1))
    pdo_h4 = next(
        row for row in rows if row["system"] == "pdo" and row["horizon"] == 4
    )
    pdo_h8 = next(
        row for row in rows if row["system"] == "pdo" and row["horizon"] == 8
    )
    pdo_h4["strict_landing_certificate_rate"] = 74 / 76
    pdo_h8["strict_landing_certificate_rate"] = 75 / 77
    pdo_h4["strict_landing_failure_count"] = 2
    pdo_h8["strict_landing_failure_count"] = 2

    result = protocol.cohort_decision(rows)

    assert result["decision"] == "ADMIT_H4_REPEAT_GATE"
    assert result["certificate_regression_systems"] == []


def _repeat_rows(deltas_by_system: dict[str, tuple[float, float, float]]):
    rows = []
    for system in ("c60", "pdo", "cuo"):
        for seed, delta in zip((49, 50, 51), deltas_by_system[system]):
            rows.extend(
                [
                    {
                        "system": system,
                        "seed": seed,
                        "horizon": 4,
                        "gain_auc_eV": 5.0 + delta,
                        "strict_landing_failure_count": 0,
                    },
                    {
                        "system": system,
                        "seed": seed,
                        "horizon": 8,
                        "gain_auc_eV": 5.0,
                        "strict_landing_failure_count": 0,
                    },
                ]
            )
    return rows


def test_repeat_decision_stops_h4_when_any_system_flips_sign() -> None:
    rows = _repeat_rows(
        {
            "c60": (1.0, -0.2, 0.3),
            "pdo": (0.4, 0.2, 0.1),
            "cuo": (0.5, 0.4, 0.2),
        }
    )

    result = protocol.repeated_seed_decision(rows, seeds=(49, 50, 51))

    assert result["decision"] == "RETAIN_H8_STOP_SHORT_HORIZON_BRANCH"
    assert result["sign_flip_systems"] == ["c60"]
    assert result["system_win_counts"]["c60"] == {"h4": 2, "h8": 1}


def test_repeat_decision_requires_complete_consistent_safe_cohort() -> None:
    rows = _repeat_rows(
        {
            "c60": (1.0, 0.2, 0.3),
            "pdo": (0.4, 0.2, 0.1),
            "cuo": (0.5, 0.4, 0.2),
        }
    )
    passed = protocol.repeated_seed_decision(rows, seeds=(49, 50, 51))
    assert passed["decision"] == "ADMIT_H4_PRODUCTION_DEFAULT"

    regressed = [dict(row) for row in rows]
    regressed[0]["strict_landing_failure_count"] = 1
    failed = protocol.repeated_seed_decision(regressed, seeds=(49, 50, 51))
    assert failed["decision"] == "RETAIN_H8_STOP_SHORT_HORIZON_BRANCH"
    assert failed["certificate_regression_pairs"] == ["c60:49"]

    partial = protocol.repeated_seed_decision(rows[:-1], seeds=(49, 50, 51))
    assert partial["decision"] == "NOT_EVALUATED_PARTIAL_REPEAT_COHORT"


def test_runner_builds_only_horizon_difference(tmp_path: Path) -> None:
    runner = _load(RUN_ROOT / "run_gate.py", "_fixed_h4_h8_runner_test")
    h4 = runner.build_config(
        "c60",
        tmp_path / "h4",
        seed=49,
        horizon=4,
        force_budget=1_000,
    )
    h8 = runner.build_config(
        "c60",
        tmp_path / "h8",
        seed=49,
        horizon=8,
        force_budget=1_000,
    )

    assert h4.seed_selection_mode == h8.seed_selection_mode == "metropolis_chain"
    assert h4.direction_selection_mode == h8.direction_selection_mode == "discrete"
    assert h4.direction_ranking_mode == h8.direction_ranking_mode == "static_score"
    assert protocol.scientific_config_differences(asdict(h4), asdict(h8)) == {
        "max_steps_per_walk": [4, 8]
    }
