from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-action-breadth-order-statistic-gate"
    / "protocol.py"
)


def _protocol():
    spec = importlib.util.spec_from_file_location(
        "_action_breadth_order_statistic_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _candidates():
    return [
        {
            "candidate_index": index,
            "landing_delta_eV": float(index),
            "force_evaluations": 10 * (index + 1),
            "valid": True,
            "static_rank": (4, 2, 3, 1)[index],
        }
        for index in range(4)
    ]


def test_uniform_b2_is_exact_over_all_six_subsets() -> None:
    protocol = _protocol()

    result = protocol.uniform_subset_summary(
        _candidates(),
        breadth=2,
        shared_pool_force_evaluations=8,
    )

    assert result["subset_count"] == 6
    assert result["expected_regret_eV"] == pytest.approx(4.0 / 6.0)
    assert result["expected_force_evaluations"] == pytest.approx(58.0)
    assert result["valid_subset_probability"] == pytest.approx(1.0)


def test_static_b1_uses_frozen_static_rank_one() -> None:
    protocol = _protocol()

    result = protocol.static_b1_summary(
        _candidates(),
        shared_pool_force_evaluations=8,
    )

    assert result["candidate_index"] == 3
    assert result["regret_eV"] == pytest.approx(3.0)
    assert result["force_evaluations"] == 48
    assert result["valid_subset_probability"] == pytest.approx(1.0)


def test_invalid_candidates_remain_in_uniform_sampling_support() -> None:
    protocol = _protocol()
    candidates = _candidates()
    candidates[0]["valid"] = False
    candidates[1]["valid"] = False

    result = protocol.uniform_subset_summary(
        candidates,
        breadth=2,
        shared_pool_force_evaluations=8,
    )

    assert result["subset_count"] == 6
    assert result["valid_subset_probability"] == pytest.approx(5.0 / 6.0)


def test_b2_gate_requires_elasticity_in_every_system_campaign_stratum() -> None:
    protocol = _protocol()
    passing = [
        {
            "campaign": campaign,
            "system": system,
            "static_median_regret_eV": 2.0,
            "static_median_force_evaluations": 100.0,
            "static_valid_probability": 1.0,
            "b2_median_regret_eV": 1.0,
            "b2_median_force_evaluations": 140.0,
            "b2_valid_probability": 1.0,
        }
        for campaign in ("first", "second")
        for system in ("c60", "pdo")
    ]

    accepted = protocol.evaluate_live_b2_gate(passing)
    rejected_rows = [dict(row) for row in passing]
    rejected_rows[-1]["b2_median_regret_eV"] = 1.6
    rejected = protocol.evaluate_live_b2_gate(rejected_rows)

    assert accepted["live_b2_gate_allowed"] is True
    assert accepted["strata"][0]["benefit_cost_elasticity"] == pytest.approx(
        1.25
    )
    assert rejected["live_b2_gate_allowed"] is False
