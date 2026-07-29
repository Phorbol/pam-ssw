from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "runs"
    / "20260729-anchor-consistent-direction-ablation"
    / "run_ablation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_anchor_consistent_direction_ablation",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(module, case):
    selections = 3
    expected_hvp = module.EXPECTED_HVP_PER_SELECTION[case["arm"]]
    expected_fe = 2 * expected_hvp
    direction_fe = selections * expected_fe
    purposes = {
        "direction_oracle": direction_fe,
        "biased_proposal_relax": 20,
        "landing_true_quench": 10,
        "escape_true_pes_check": 5,
        "post_relax_validation": 1,
        "starter_true_quench": 0,
        "bootstrap_true_quench": 0,
        "unattributed": 0,
    }
    return {
        **case,
        "status": "completed",
        "state_sha256": f"state-{case['state_id']}",
        "exact_starter_reference": True,
        "starter_energy_eV": -100.0,
        "escape_energy_eV": -99.0,
        "landing_energy_eV": -101.0,
        "landing_delta_eV": -1.0,
        "certificate": True,
        "final_max_force_eV_per_A": 0.009,
        "is_new_basin": True,
        "fallback_used": False,
        "force_evaluations": sum(purposes.values()),
        "purpose_counts": purposes,
        "direction_selection_count": selections,
        "direction_hvp_count": selections * expected_hvp,
        "direction_trace_valid": True,
        "generation_wall_time_s": 1.0,
        "quench_wall_time_s": 0.5,
        "starter_path": "starter.xyz",
        "starter_file_sha256": "starter-hash",
        "escape_path": "escape.xyz",
        "escape_sha256": "escape-hash",
        "landing_path": "landing.xyz",
        "landing_sha256": "landing-hash",
        "direction_trace": [],
    }


def test_case_matrix_and_arm_contract_are_exact():
    module = _load_module()

    assert module.STATE_IDS == (
        "intermediate_accepted",
        "plateau_accepted",
    )
    assert module.ARMS == {
        "detached_ritz": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
        "exact_anchor": {
            "direction_selection_mode": "exact_anchor",
        },
        "anchor_lanczos": {
            "direction_selection_mode": "anchor_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 12,
        },
    }
    assert module.EXPECTED_HVP_PER_SELECTION == {
        "detached_ritz": 12,
        "exact_anchor": 1,
        "anchor_lanczos": 12,
    }
    assert len(module.case_matrix()) == 18
    assert len(
        {
            (row["state_id"], row["seed"], row["arm"])
            for row in module.case_matrix()
        }
    ) == 18


def test_evidence_requires_complete_certified_closed_cohort():
    module = _load_module()
    rows = [_row(module, case) for case in module.case_matrix()]

    evidence = module.build_evidence(rows)

    assert evidence["cohort"]["completed_cases"] == 18
    assert evidence["certificate_count"] == 18
    assert evidence["meaningful_outcome_count"] == 18
    assert evidence["production_default_changed"] is False
    assert evidence["bootstrap_force_evaluations"] == 0
    assert evidence["totals"]["unattributed_force_evaluations"] == 0
    assert evidence["arm_results"]["exact_anchor"][
        "direction_force_evaluations"
    ] == 36
    assert evidence["arm_results"]["detached_ritz"][
        "direction_force_evaluations"
    ] == 432
    assert evidence["arm_results"]["anchor_lanczos"][
        "direction_force_evaluations"
    ] == 432

    with pytest.raises(ValueError, match="exact 18-case"):
        module.build_evidence(rows[:-1])

    rows[0]["purpose_counts"]["direction_oracle"] += 1
    rows[0]["force_evaluations"] += 1
    with pytest.raises(ValueError, match="direction ledger"):
        module.build_evidence(rows)


def test_evidence_does_not_require_or_contain_selector_feedback():
    module = _load_module()
    rows = [_row(module, case) for case in module.case_matrix()]

    evidence = module.build_evidence(rows)

    assert "selector" not in evidence
    assert "posterior" not in evidence
    assert "reward" not in evidence
    assert all(
        "selector" not in row
        and "posterior" not in row
        and "reward" not in row
        for row in evidence["cases"]
    )
