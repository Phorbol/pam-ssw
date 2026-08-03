from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest


ANALYZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260801-selector-support-audit"
    / "analyze.py"
)


def _analyzer():
    name = "_selector_support_audit_test"
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_support_metrics_measure_effective_arms_and_repeat_fraction():
    analyzer = _analyzer()

    metrics = analyzer.support_metrics([0, 1, 0, 2], final_archive_size=4)

    expected_entropy = -(0.5 * math.log(0.5) + 2 * 0.25 * math.log(0.25))
    assert metrics["action_count"] == 4
    assert metrics["unique_starter_count"] == 3
    assert metrics["effective_support"] == pytest.approx(math.exp(expected_entropy))
    assert metrics["effective_support_fraction"] == pytest.approx(
        math.exp(expected_entropy) / 4.0
    )
    assert metrics["max_starter_frequency"] == pytest.approx(0.5)
    assert metrics["repeat_action_fraction"] == pytest.approx(0.25)


def test_trace_projection_excludes_bootstrap_and_closes_final_archive():
    analyzer = _analyzer()
    trace = [
        {
            "trial": 0,
            "starter_entry_id": 0,
            "landing_entry_id": 0,
            "accepted_new_basin": True,
        },
        {
            "trial": 1,
            "starter_entry_id": 0,
            "landing_entry_id": 1,
            "accepted_new_basin": True,
        },
        {
            "trial": 2,
            "starter_entry_id": 1,
            "landing_entry_id": 1,
            "accepted_new_basin": False,
        },
    ]

    projected = analyzer.project_trace(trace, expected_actions=2, expected_archive=2)

    assert projected == {
        "starter_ids": [0, 1],
        "final_archive_size": 2,
        "new_basin_action_count": 1,
    }


def test_trace_projection_rejects_nonconsecutive_trials():
    analyzer = _analyzer()
    trace = [
        {"trial": 0, "starter_entry_id": 0, "landing_entry_id": 0},
        {"trial": 2, "starter_entry_id": 0, "landing_entry_id": 1},
    ]

    with pytest.raises(ValueError, match="consecutive"):
        analyzer.project_trace(trace, expected_actions=1, expected_archive=2)
