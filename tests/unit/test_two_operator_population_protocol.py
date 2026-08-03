from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/20260803-two-operator-population-gate/protocol.py"


def load_protocol():
    spec = importlib.util.spec_from_file_location("_two_operator_protocol", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(system, starter, seed, family, *, certified=True, new=True, fe=100):
    return {
        "system": system,
        "starter_context": starter,
        "seed": seed,
        "operator_family": family,
        "certified": certified,
        "same_starter_basin": not new,
        "geometry_valid": True,
        "fragmented": False,
        "budget_censored": False,
        "landing_delta_eV": -1.0 if new else 0.0,
        "force_evaluations": fe,
        "fully_loaded_force_evaluations": fe,
        "wall_time_s": 1.0,
        "fully_loaded_wall_time_s": 1.0,
        "purpose_counts": {"landing_true_quench": fe, "unattributed": 0},
    }


def test_case_matrix_is_exactly_18_pairs():
    protocol = load_protocol()
    matrix = protocol.case_matrix()
    assert len(matrix) == 18
    assert {item.system for item in matrix} == {"c60", "pdo", "cuo"}
    assert {item.starter_context for item in matrix} == {"bootstrap", "h8_best"}
    assert {item.seed for item in matrix} == {55, 56, 57}


def test_stage_b_passes_only_with_direct_viability_ssw_support_and_certificates():
    protocol = load_protocol()
    rows = []
    for system in protocol.SYSTEMS:
        for starter in protocol.STARTERS:
            for seed in protocol.SEEDS:
                direct_new = starter == "bootstrap"
                rows.append(
                    row(system, starter, seed, "direct", new=direct_new, fe=80)
                )
                rows.append(row(system, starter, seed, "ssw", new=True, fe=300))
    decision = protocol.decide_stage_b(rows)
    assert decision["decision"] == "ADMIT_STAGE_C_DESIGN"
    assert decision["direct_viability_contexts"] == 3
    assert decision["ssw_exclusive_support_contexts"] == 3
    assert decision["numerical_acceptability"] is True


def test_stage_b_closes_when_ssw_has_no_exclusive_support():
    protocol = load_protocol()
    rows = []
    for item in protocol.case_matrix():
        rows.append(row(item.system, item.starter_context, item.seed, "direct", fe=80))
        rows.append(row(item.system, item.starter_context, item.seed, "ssw", fe=300))
    assert protocol.decide_stage_b(rows)["decision"] == "CLOSE_TWO_OPERATOR_PORTFOLIO"


def test_unattributed_work_fails_closed():
    protocol = load_protocol()
    bad = row("c60", "bootstrap", 55, "direct")
    bad["purpose_counts"] = {"landing_true_quench": 99, "unattributed": 1}
    with pytest.raises(ValueError, match="unattributed"):
        protocol.validate_row(bad)


def test_duplicate_action_keys_fail_closed():
    protocol = load_protocol()
    rows = []
    for item in protocol.case_matrix():
        rows.append(row(item.system, item.starter_context, item.seed, "direct"))
        rows.append(row(item.system, item.starter_context, item.seed, "ssw"))
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="not unique"):
        protocol.decide_stage_b(rows)


def test_fully_loaded_cost_cannot_be_smaller_than_exclusive_cost():
    protocol = load_protocol()
    bad = row("c60", "bootstrap", 55, "direct", fe=100)
    bad["fully_loaded_force_evaluations"] = 99
    with pytest.raises(ValueError, match="fully loaded"):
        protocol.validate_row(bad)
