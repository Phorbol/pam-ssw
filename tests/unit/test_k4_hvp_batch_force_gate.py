from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-k4-hvp-batch-force-gate"
    / "protocol.py"
)


def _protocol():
    spec = importlib.util.spec_from_file_location(
        "_k4_hvp_batch_force_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_central_hvp_from_force_pairs_has_correct_sign_and_order() -> None:
    protocol = _protocol()
    force_pairs = [
        (
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[-1.0, 0.0, 0.0]]),
        ),
        (
            np.array([[0.0, 2.0, 0.0]]),
            np.array([[0.0, -2.0, 0.0]]),
        ),
    ]

    hvps = protocol.central_hvps_from_forces(force_pairs, epsilon=0.5)

    np.testing.assert_allclose(hvps[0], [[-2.0, 0.0, 0.0]])
    np.testing.assert_allclose(hvps[1], [[0.0, -4.0, 0.0]])


def test_directional_curvature_uses_matching_direction() -> None:
    protocol = _protocol()
    directions = [
        np.array([[1.0, 0.0, 0.0]]),
        np.array([[0.0, 1.0, 0.0]]),
    ]
    hvps = [
        np.array([[3.0, 4.0, 0.0]]),
        np.array([[5.0, 6.0, 0.0]]),
    ]

    curvatures = protocol.directional_curvatures(directions, hvps)

    assert curvatures == pytest.approx([3.0, 6.0])


def test_equivalence_reports_relative_hvp_error_and_curvature_error() -> None:
    protocol = _protocol()
    reference = {
        "energies": np.array([1.0, 2.0]),
        "forces": np.array([[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]]),
        "hvps": np.array([[[2.0, 0.0, 0.0]]]),
        "curvatures": np.array([2.0]),
    }
    candidate = {
        "energies": np.array([1.001, 2.0]),
        "forces": np.array([[[1.0001, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]]),
        "hvps": np.array([[[2.002, 0.0, 0.0]]]),
        "curvatures": np.array([2.002]),
    }

    metrics = protocol.equivalence_metrics(reference, candidate)

    assert metrics["max_abs_energy_error_eV"] == pytest.approx(0.001)
    assert metrics["max_abs_force_error_eV_per_A"] == pytest.approx(0.0001)
    assert metrics["relative_hvp_norm_error"] == pytest.approx(0.001)
    assert metrics["max_abs_curvature_error_eV_per_A2"] == pytest.approx(
        0.002
    )


def test_common_batch_gate_requires_equivalence_and_speed_in_both_systems() -> None:
    protocol = _protocol()
    rows = [
        {
            "system": system,
            "batch_size": batch_size,
            "equivalent": True,
            "median_speedup": speedup,
        }
        for batch_size, speedup in ((2, 1.2), (4, 1.4), (8, 1.5))
        for system in ("c60", "pdo")
    ]
    rows[-1]["median_speedup"] = 1.1

    gate = protocol.evaluate_batch_gate(rows, minimum_speedup=1.3)

    assert gate["surviving_batch_sizes"] == [4]
    assert gate["force_service_gate_allowed"] is True
