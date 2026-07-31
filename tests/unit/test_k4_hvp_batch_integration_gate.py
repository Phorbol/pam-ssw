from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260731-k4-hvp-batch-integration-gate"
)


def _protocol():
    spec = importlib.util.spec_from_file_location(
        "_test_k4_hvp_batch_integration_protocol",
        RUN_ROOT / "protocol.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runner():
    spec = importlib.util.spec_from_file_location(
        "_test_k4_hvp_batch_integration_runner",
        RUN_ROOT / "run_gate.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(batch_time: float = 1.0, *, batch_kind: str = "random"):
    rows = []
    for system in ("c60", "pdo"):
        for context in ("initial", "momentum"):
            for repetition in range(3):
                shared = {
                    "system": system,
                    "context": context,
                    "repetition": repetition,
                    "selected_kind": "random",
                    "curvature": 2.0,
                    "score": -1.0,
                    "force_evaluations": 8,
                    "unattributed": 0,
                }
                rows.append(
                    {
                        **shared,
                        "mode": "serial",
                        "wall_time_s": 2.0,
                        "direction_cosine_to_serial": 1.0,
                    }
                )
                rows.append(
                    {
                        **shared,
                        "mode": "batch",
                        "wall_time_s": batch_time,
                        "selected_kind": batch_kind,
                        "direction_cosine_to_serial": 1.0,
                    }
                )
    return rows


def test_gate_requires_same_direction_cost_and_cross_system_speed():
    gate = _protocol().evaluate_gate(_rows())

    assert gate["passed"] is True
    assert gate["pair_count"] == 12
    assert all(row["median_speedup"] == 2.0 for row in gate["strata"])


def test_gate_rejects_a_faster_batch_that_changes_the_selected_direction_source():
    gate = _protocol().evaluate_gate(_rows(batch_kind="bond"))

    assert gate["passed"] is False
    assert all(row["same_selected_kind"] is False for row in gate["strata"])


def test_gate_rejects_physical_equivalence_without_enough_wall_time_gain():
    gate = _protocol().evaluate_gate(_rows(batch_time=1.8))

    assert gate["passed"] is False
    assert all(row["same_selected_kind"] is True for row in gate["strata"])


def test_runner_computes_a_normalized_cosine_not_a_raw_dot_product():
    import numpy as np

    runner = _runner()
    left = np.array([0.999999, 0.0])
    right = np.array([2.0, 0.0])

    assert runner._direction_cosine(left, right) == 1.0
