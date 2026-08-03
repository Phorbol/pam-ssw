from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260731-action-family-transfer-gate"
    / "protocol.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_action_family_transfer_protocol_test",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(c60, pdo):
    rows = []
    for system, values in (("c60", c60), ("pdo", pdo)):
        for state_id in ("intermediate_accepted", "plateau_accepted"):
            for seed in (42, 43, 44):
                for arm, (delta, cost) in values.items():
                    rows.append(
                        {
                            "system": system,
                            "state_id": state_id,
                            "seed": seed,
                            "arm": arm,
                            "landing_delta_eV": delta,
                            "force_evaluations": cost,
                            "certificate": True,
                            "landing_geometry_valid": True,
                            "fragmented": False,
                        }
                    )
    return rows


def test_case_matrix_has_two_systems_two_states_three_seeds_and_three_arms():
    protocol = _load_protocol()

    matrix = protocol.case_matrix()

    assert len(matrix) == 36
    assert {row["system"] for row in matrix} == {"c60", "pdo"}
    assert {row["arm"] for row in matrix} == {
        "D0_exact_anchor",
        "D1_anchor_krylov_d2",
        "K4_discrete",
    }


def test_transfer_gate_passes_when_the_other_system_selects_a_shared_nondominated_winner():
    protocol = _load_protocol()
    values = {
        "D0_exact_anchor": (2.0, 100),
        "D1_anchor_krylov_d2": (-2.0, 130),
        "K4_discrete": (1.0, 180),
    }

    result = protocol.summarize_campaign(_rows(values, values))

    assert result["promotion_allowed"] is True
    assert result["held_out"]["c60"]["selected_arm"] == "D1_anchor_krylov_d2"
    assert result["held_out"]["pdo"]["selected_arm"] == "D1_anchor_krylov_d2"


def test_transfer_gate_fails_when_systems_prefer_different_arms():
    protocol = _load_protocol()
    c60 = {
        "D0_exact_anchor": (-3.0, 100),
        "D1_anchor_krylov_d2": (0.0, 130),
        "K4_discrete": (1.0, 180),
    }
    pdo = {
        "D0_exact_anchor": (2.0, 100),
        "D1_anchor_krylov_d2": (-2.0, 130),
        "K4_discrete": (1.0, 180),
    }

    result = protocol.summarize_campaign(_rows(c60, pdo))

    assert result["promotion_allowed"] is False
    assert result["held_out"]["c60"]["selected_arm"] == "D1_anchor_krylov_d2"
    assert result["held_out"]["c60"]["beats_uniform_mean"] is False
    assert result["held_out"]["pdo"]["selected_arm"] == "D0_exact_anchor"
    assert result["held_out"]["pdo"]["beats_uniform_mean"] is False


def test_transfer_gate_rejects_a_selected_arm_dominated_in_quality_and_cost():
    protocol = _load_protocol()
    training = {
        "D0_exact_anchor": (0.0, 100),
        "D1_anchor_krylov_d2": (-2.0, 130),
        "K4_discrete": (1.0, 180),
    }
    held_out = {
        "D0_exact_anchor": (-2.0, 100),
        "D1_anchor_krylov_d2": (-1.0, 130),
        "K4_discrete": (1.0, 180),
    }

    result = protocol.summarize_campaign(_rows(held_out, training))

    assert result["held_out"]["c60"]["selected_arm"] == "D1_anchor_krylov_d2"
    assert result["held_out"]["c60"]["pareto_dominated"] is True
    assert result["promotion_allowed"] is False


def test_partial_smoke_cohort_is_summarized_but_cannot_promote():
    protocol = _load_protocol()
    values = {
        "D0_exact_anchor": (2.0, 100),
        "D1_anchor_krylov_d2": (-2.0, 130),
        "K4_discrete": (1.0, 180),
    }
    rows = [
        row
        for row in _rows(values, values)
        if row["system"] == "c60"
        and row["state_id"] == "intermediate_accepted"
        and row["seed"] == 42
    ]

    result = protocol.summarize_campaign(rows)

    assert set(result["by_system"]) == {"c60"}
    assert result["held_out"] == {}
    assert result["promotion_allowed"] is False
