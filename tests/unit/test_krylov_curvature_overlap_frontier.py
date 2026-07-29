from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "runs"
    / "20260729-krylov-curvature-overlap-frontier"
    / "run_audit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_krylov_curvature_overlap_frontier",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spectrum():
    points = [
        (1.0, 4.0, 0.1, True),
        (2.0, 3.0, 0.4, False),
        (3.0, 2.0, 0.3, False),
        (4.0, 6.0, 0.8, False),
    ]
    return [
        {
            "block_index": 0,
            "ritz_index": index,
            "executed": executed,
            "curvature": curvature,
            "true_curvature": true_curvature,
            "residual_norm": 0.01 * (index + 1),
            "initial_span_overlap": 0.2 * (index + 1),
            "anchor_abs_overlap": overlap,
            "participation_ratio": 2.0 + index,
        }
        for index, (
            curvature,
            true_curvature,
            overlap,
            executed,
        ) in enumerate(points)
    ]


def _row(module, case):
    selections = 1
    direction_hvps = module.EXPECTED_HVP_PER_SELECTION[
        case["arm"]
    ]
    purposes = {
        "direction_oracle": 2 * direction_hvps,
        "biased_proposal_relax": 20,
        "escape_true_pes_check": 3,
        "landing_true_quench": 0,
        "post_relax_validation": 0,
        "starter_true_quench": 0,
        "bootstrap_true_quench": 0,
        "unattributed": 0,
    }
    return {
        **case,
        "status": "completed",
        "state_sha256": f"state-{case['state_id']}",
        "escape_hash_matches": True,
        "selected_trace_matches": True,
        "replay_escape_sha256": f"escape-{case['state_id']}",
        "prior_escape_sha256": f"escape-{case['state_id']}",
        "force_evaluations": sum(purposes.values()),
        "purpose_counts": purposes,
        "direction_selection_count": selections,
        "direction_hvp_count": direction_hvps,
        "proposal_wall_time_s": 1.0,
        "direction_trace": [
            {
                "step": 0,
                "oracle_selection_force_evaluations_delta": (
                    2 * direction_hvps
                ),
                "krylov_hvp_consumed": direction_hvps,
                "krylov_ritz_spectrum": _spectrum(),
            }
        ],
        "prior_terminal_outcome": {
            "certificate": True,
            "landing_delta_eV": -1.0,
            "is_new_basin": True,
            "meaningful_outcome": True,
            "prior_force_evaluations": 100,
            "prior_landing_true_quench_force_evaluations": 10,
        },
    }


def test_case_matrix_and_arm_contract_are_exact():
    module = _load_module()

    assert module.STATE_IDS == (
        "intermediate_accepted",
        "plateau_accepted",
    )
    assert module.SEEDS == (42, 43, 44)
    assert module.ARMS == {
        "detached_ritz": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
        "anchor_lanczos": {
            "direction_selection_mode": "anchor_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 12,
        },
    }
    assert module.EXPECTED_HVP_PER_SELECTION == {
        "detached_ritz": 12,
        "anchor_lanczos": 12,
    }
    assert len(module.case_matrix()) == 12


def test_pareto_frontier_is_parameter_free_and_exact():
    module = _load_module()

    summary = module.summarize_frontier(_spectrum())

    assert summary["frontier_indices"] == [0, 1, 3]
    assert summary["executed_index"] == 0
    assert summary["maximum_overlap_index"] == 3
    assert summary["frontier_size"] == 3
    assert summary["maximum_overlap_curvature_delta"] == pytest.approx(
        3.0
    )
    assert (
        summary["maximum_overlap_true_curvature_delta"]
        == pytest.approx(2.0)
    )


def test_replay_trace_contract_ignores_timing_but_not_direction_values():
    module = _load_module()
    prior = [
        {
            "selected_curvature": 1.0,
            "anchor_cosine": 0.2,
            "oracle_selection_wall_seconds": 1.0,
            "oracle_wall_seconds": 2.0,
        }
    ]
    replay = [
        {
            **prior[0],
            "oracle_selection_wall_seconds": 10.0,
            "oracle_wall_seconds": 20.0,
            "krylov_ritz_spectrum": _spectrum(),
        }
    ]

    assert module.selected_trace_contract(replay) == (
        module.selected_trace_contract(prior)
    )
    replay[0]["selected_curvature"] = 2.0
    assert module.selected_trace_contract(replay) != (
        module.selected_trace_contract(prior)
    )


def test_evidence_requires_exact_replays_and_zero_quench():
    module = _load_module()
    rows = [_row(module, case) for case in module.case_matrix()]

    evidence = module.build_evidence(
        rows,
        prior_evidence_sha256="prior-evidence-hash",
    )

    assert evidence["cohort"]["completed_replays"] == 12
    assert evidence["escape_hash_match_count"] == 12
    assert evidence["selected_trace_match_count"] == 12
    assert evidence["totals"]["purpose_counts"][
        "landing_true_quench"
    ] == 0
    assert evidence["totals"]["purpose_counts"][
        "bootstrap_true_quench"
    ] == 0
    assert evidence["totals"]["purpose_counts"]["unattributed"] == 0
    assert evidence["referenced_prior_terminal_force_evaluations"] == (
        1200
    )
    assert evidence["totals"]["force_evaluations"] == 12 * 47
    assert evidence["production_default_changed"] is False

    with pytest.raises(ValueError, match="exact 12-case"):
        module.build_evidence(
            rows[:-1],
            prior_evidence_sha256="prior-evidence-hash",
        )

    rows[0]["escape_hash_matches"] = False
    with pytest.raises(ValueError, match="replay proof"):
        module.build_evidence(
            rows,
            prior_evidence_sha256="prior-evidence-hash",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_trace_matches", False, "replay proof"),
        ("landing_true_quench", 1, "forbidden purpose"),
        ("bootstrap_true_quench", 1, "forbidden purpose"),
        ("post_relax_validation", 1, "forbidden purpose"),
        ("unattributed", 1, "forbidden purpose"),
    ],
)
def test_evidence_rejects_trace_mismatch_or_forbidden_cost(
    field,
    value,
    message,
):
    module = _load_module()
    rows = [_row(module, case) for case in module.case_matrix()]
    if field == "selected_trace_matches":
        rows[0][field] = value
    else:
        rows[0]["purpose_counts"][field] = value
        rows[0]["force_evaluations"] += value

    with pytest.raises(ValueError, match=message):
        module.build_evidence(
            rows,
            prior_evidence_sha256="prior-evidence-hash",
        )


@pytest.mark.parametrize(
    "mutation",
    ["no_executed", "two_executed", "unsorted", "nonfinite"],
)
def test_evidence_rejects_invalid_spectrum(mutation):
    module = _load_module()
    rows = [_row(module, case) for case in module.case_matrix()]
    spectrum = rows[0]["direction_trace"][0][
        "krylov_ritz_spectrum"
    ]
    if mutation == "no_executed":
        spectrum[0]["executed"] = False
    elif mutation == "two_executed":
        spectrum[1]["executed"] = True
    elif mutation == "unsorted":
        spectrum[1]["curvature"] = 0.5
    else:
        spectrum[0]["anchor_abs_overlap"] = float("nan")

    with pytest.raises(ValueError, match="spectrum"):
        module.build_evidence(
            rows,
            prior_evidence_sha256="prior-evidence-hash",
        )
