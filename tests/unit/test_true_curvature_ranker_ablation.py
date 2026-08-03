from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-true-curvature-ranker-ablation"
    / "protocol.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_true_curvature_ranker_protocol_test",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_matrix_is_two_system_two_state_three_seed_two_arm() -> None:
    protocol = _load_protocol()
    cases = protocol.case_matrix()

    assert len(cases) == 24
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


def test_repeat_gate_requires_energy_and_cost_pareto_improvement_per_system() -> None:
    protocol = _load_protocol()
    first = []
    second = []
    for system in ("c60", "pdo"):
        for seed in (42, 43, 44):
            for arm, delta, force_evaluations in (
                ("static_score", 2.0, 100),
                ("true_curvature", 1.0, 90),
            ):
                row = {
                    "system": system,
                    "state_id": "plateau_accepted",
                    "seed": seed,
                    "arm": arm,
                    "landing_delta_eV": delta,
                    "force_evaluations": force_evaluations,
                    "certificate": True,
                    "landing_geometry_valid": True,
                }
                first.append(row)
                second.append(dict(row))

    result = protocol.summarize_repeats(first, second)

    assert result["promotion_allowed"] is True
    assert result["by_system"]["c60"]["median_energy_difference_eV"] == -1.0
    assert result["by_system"]["pdo"]["force_evaluation_difference"] == -30

    for row in second:
        if row["system"] == "pdo" and row["arm"] == "true_curvature":
            row["force_evaluations"] = 120
    result = protocol.summarize_repeats(first, second)
    assert result["promotion_allowed"] is False
