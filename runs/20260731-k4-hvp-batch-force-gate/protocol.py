"""Pure numerical protocol for the K4 central-HVP batch micro-gate."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def central_hvps_from_forces(
    force_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    epsilon: float,
) -> list[np.ndarray]:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return [
        -(
            np.asarray(force_plus, dtype=float)
            - np.asarray(force_minus, dtype=float)
        )
        / (2.0 * float(epsilon))
        for force_plus, force_minus in force_pairs
    ]


def directional_curvatures(
    directions: Sequence[np.ndarray],
    hvps: Sequence[np.ndarray],
) -> list[float]:
    if len(directions) != len(hvps):
        raise ValueError("directions and HVPs must have equal length")
    values = []
    for direction, hvp in zip(directions, hvps, strict=True):
        vector = np.asarray(direction, dtype=float).reshape(-1)
        product = np.asarray(hvp, dtype=float).reshape(-1)
        values.append(float(np.dot(vector, product)))
    return values


def equivalence_metrics(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
) -> dict[str, float]:
    energy_error = np.asarray(candidate["energies"], dtype=float) - np.asarray(
        reference["energies"], dtype=float
    )
    force_error = np.asarray(candidate["forces"], dtype=float) - np.asarray(
        reference["forces"], dtype=float
    )
    reference_hvp = np.asarray(reference["hvps"], dtype=float)
    hvp_error = np.asarray(candidate["hvps"], dtype=float) - reference_hvp
    denominator = float(np.linalg.norm(reference_hvp.reshape(-1)))
    relative_hvp_error = float(
        np.linalg.norm(hvp_error.reshape(-1)) / max(denominator, 1.0e-30)
    )
    curvature_error = np.asarray(
        candidate["curvatures"], dtype=float
    ) - np.asarray(reference["curvatures"], dtype=float)
    return {
        "max_abs_energy_error_eV": float(
            np.max(np.abs(energy_error), initial=0.0)
        ),
        "max_abs_force_error_eV_per_A": float(
            np.max(np.abs(force_error), initial=0.0)
        ),
        "relative_hvp_norm_error": relative_hvp_error,
        "max_abs_curvature_error_eV_per_A2": float(
            np.max(np.abs(curvature_error), initial=0.0)
        ),
    }


def evaluate_batch_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_speedup: float,
) -> dict[str, Any]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["batch_size"])].append(row)
    surviving = []
    batches = []
    for batch_size, group in sorted(grouped.items()):
        systems = {str(row["system"]) for row in group}
        passed = (
            systems == {"c60", "pdo"}
            and all(bool(row["equivalent"]) for row in group)
            and all(
                float(row["median_speedup"]) >= float(minimum_speedup)
                for row in group
            )
        )
        batches.append(
            {
                "batch_size": batch_size,
                "systems": sorted(systems),
                "passed": passed,
            }
        )
        if passed:
            surviving.append(batch_size)
    return {
        "minimum_speedup": float(minimum_speedup),
        "batches": batches,
        "surviving_batch_sizes": surviving,
        "force_service_gate_allowed": bool(surviving),
        "production_change_allowed": False,
    }
