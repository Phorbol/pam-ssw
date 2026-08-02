from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "20260802-step1-direction-counterfactual"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load(RUN_ROOT / "protocol.py", "_step1_direction_protocol_test")


def _pool_rows(
    *,
    system: str = "c60",
    seed: int = 52,
    best_by_repeat: tuple[int, int] = (1, 1),
):
    kinds = ("momentum", "bond", "bond", "random")
    rows = []
    for repeat, best_index in enumerate(best_by_repeat):
        for candidate_index, kind in enumerate(kinds):
            landing = float(candidate_index)
            if candidate_index == best_index:
                landing = -2.0
            rows.append(
                {
                    "system": system,
                    "seed": seed,
                    "repeat": repeat,
                    "candidate_index": candidate_index,
                    "kind": kind,
                    "static_rank": candidate_index + 1,
                    "static_score": float(4 - candidate_index),
                    "direction_sha256": f"direction-{candidate_index}",
                    "landing_delta_eV": landing,
                    "certificate": True,
                    "landing_geometry_valid": True,
                    "fragmented": False,
                    "prefix_valid": True,
                }
            )
    return rows


def _complete_rows(*, stable_misses: dict[str, int]):
    rows = []
    for system in ("c60", "pdo", "cuo"):
        for offset, seed in enumerate((52, 53, 54)):
            best = 3 if offset < stable_misses[system] else 0
            rows.extend(
                _pool_rows(
                    system=system,
                    seed=seed,
                    best_by_repeat=(best, best),
                )
            )
    return rows


def test_group_matrix_is_three_system_three_seed_order() -> None:
    assert protocol.group_matrix() == [
        {"system": system, "seed": seed}
        for system in ("c60", "pdo", "cuo")
        for seed in (52, 53, 54)
    ]


def test_family_terminal_medians_do_not_double_weight_bond_slots() -> None:
    rows = [row for row in _pool_rows() if row["repeat"] == 0]

    result = protocol.family_terminal_medians(rows)

    assert result == {
        "momentum": pytest.approx(0.0),
        "bond": pytest.approx(0.0),
        "random": pytest.approx(3.0),
    }


def test_repeat_summary_rejects_unstable_best_identity() -> None:
    result = protocol.summarize_repeats(
        _pool_rows(best_by_repeat=(0, 1))
    )

    assert result["decision"] == "STOP_NON_IDENTIFIABLE_DIRECTION_LABELS"
    assert result["unstable_best_pools"] == ["c60:52"]
    assert result["posterior_gate_allowed"] is False


def test_static_bottleneck_requires_every_system_to_pass() -> None:
    result = protocol.summarize_repeats(
        _complete_rows(
            stable_misses={"c60": 3, "pdo": 2, "cuo": 1}
        )
    )

    assert result["static_continuation_bottleneck"] is False
    assert result["posterior_gate_allowed"] is False


def test_cross_system_momentum_dominance_opens_only_source_gate() -> None:
    rows = []
    for system in ("c60", "pdo", "cuo"):
        for seed in (52, 53, 54):
            rows.extend(
                _pool_rows(
                    system=system,
                    seed=seed,
                    best_by_repeat=(0, 0),
                )
            )

    result = protocol.summarize_repeats(rows)

    assert result["dominant_source"] == "momentum"
    assert result["source_only_gate_allowed"] is True
    assert result["posterior_gate_allowed"] is False
    assert result["decision"] == "ALLOW_SOURCE_ONLY_PROSPECTIVE_GATE"
