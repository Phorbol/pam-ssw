from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/20260803-bias-history-action-gate/protocol.py"


def load_protocol():
    spec = importlib.util.spec_from_file_location("_bias_history_protocol", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(system, starter, seed, family, *, escaped, fe=300, certified=True):
    return {
        "system": system,
        "starter_context": starter,
        "seed": seed,
        "operator_family": family,
        "certified": certified,
        "same_starter_basin": not escaped,
        "geometry_valid": True,
        "fragmented": False,
        "budget_censored": False,
        "landing_delta_eV": -1.0 if escaped else 0.0,
        "force_evaluations": fe,
        "fully_loaded_force_evaluations": fe + 20,
        "purpose_counts": {"biased_proposal_relax": fe, "unattributed": 0},
    }


def complete_rows(protocol, outcome):
    rows = []
    for case in protocol.case_matrix():
        for family in protocol.FAMILIES:
            escaped, fe = outcome(case, family)
            rows.append(
                row(
                    case.system,
                    case.starter_context,
                    case.seed,
                    family,
                    escaped=escaped,
                    fe=fe,
                )
            )
    return rows


def test_case_matrix_reuses_the_exact_eighteen_shared_inputs():
    protocol = load_protocol()
    matrix = protocol.case_matrix()
    assert len(matrix) == 18
    assert {case.system for case in matrix} == {"c60", "pdo", "cuo"}
    assert {case.starter_context for case in matrix} == {"bootstrap", "h8_best"}
    assert {case.seed for case in matrix} == {55, 56, 57}
    assert protocol.FAMILIES == ("cumulative", "newest_only")


def test_repeated_cumulative_only_escape_retains_history_as_required():
    protocol = load_protocol()

    def outcome(case, family):
        cumulative_only = case.system == "c60" and case.starter_context == "bootstrap"
        escaped = family == "cumulative" if cumulative_only else True
        return escaped, 300 if family == "cumulative" else 250

    decision = protocol.decide(complete_rows(protocol, outcome))
    assert decision["decision"] == "RETAIN_CUMULATIVE_REQUIRED"
    assert decision["cumulative_only_support_contexts"] == 1


def test_newest_only_replaces_history_only_with_no_support_loss_and_lower_cost():
    protocol = load_protocol()
    decision = protocol.decide(
        complete_rows(
            protocol,
            lambda _case, family: (True, 300 if family == "cumulative" else 250),
        )
    )
    assert decision["decision"] == "REPLACE_WITH_NEWEST_ONLY"
    assert decision["aggregate_escape_counts"] == {
        "cumulative": 18,
        "newest_only": 18,
    }


def test_unattributed_work_fails_closed():
    protocol = load_protocol()
    bad = row("c60", "bootstrap", 55, "cumulative", escaped=True)
    bad["purpose_counts"] = {"biased_proposal_relax": 299, "unattributed": 1}
    try:
        protocol.validate_row(bad)
    except ValueError as exc:
        assert "unattributed" in str(exc)
    else:
        raise AssertionError("unattributed work was accepted")
