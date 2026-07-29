import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260730-direction-mode-continuation"
    / "run_ablation.py"
)
ANALYZER_PATH = RUNNER_PATH.with_name("analyze_ablation.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_direction_mode_continuation_runner",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "_direction_mode_continuation_analyzer",
        ANALYZER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direction_mode_continuation_protocol_is_the_locked_18_case_matrix():
    runner = _load_runner()

    assert runner.STATE_IDS == (
        "intermediate_accepted",
        "plateau_accepted",
    )
    assert runner.SEEDS == (42, 43, 44)
    assert runner.ARMS == {
        "fixed_intent_ritz": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
        "transported_direction": {
            "direction_selection_mode": "transported_direction",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
        "continuation_lanczos": {
            "direction_selection_mode": "continuation_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 12,
        },
    }
    cases = runner.case_matrix()
    assert len(cases) == 18
    assert len(
        {
            (case["state_id"], case["seed"], case["arm"])
            for case in cases
        }
    ) == 18


def _step_zero_row(direction_hash="common"):
    return {
        "step": 0,
        "selected_kind": "block_ritz",
        "candidate_count": 0,
        "krylov_blocks": 1,
        "krylov_depth": 6,
        "krylov_initial_basis_columns": [2],
        "krylov_hvp_requested": 12,
        "krylov_hvp_consumed": 12,
        "krylov_hvp_count": 12,
        "oracle_selection_force_evaluations_delta": 24,
        "selected_direction_sha256": direction_hash,
        "selected_curvature": -0.2,
        "true_curvature": -0.1,
        "direction_participation_ratio": 3.0,
        "selected_to_previous_selected_abs_cosine": None,
    }


def _later_row(arm):
    if arm == "fixed_intent_ritz":
        row = _step_zero_row(direction_hash="later-control")
        row["step"] = 1
        row["selected_to_previous_selected_abs_cosine"] = 0.4
        return row
    if arm == "transported_direction":
        return {
            "step": 1,
            "selected_kind": "transported",
            "candidate_count": 1,
            "direction_hvp_count": 1,
            "oracle_selection_force_evaluations_delta": 2,
            "selected_direction_sha256": "later-transported",
            "selected_curvature": -0.2,
            "true_curvature": -0.1,
            "selected_to_previous_selected_abs_cosine": 0.95,
            "continuation_source": "selected_mode",
        }
    return {
        "step": 1,
        "selected_kind": "continuation_ritz",
        "candidate_count": 0,
        "krylov_blocks": 1,
        "krylov_depth": 12,
        "krylov_initial_basis_columns": [1],
        "krylov_hvp_requested": 12,
        "krylov_hvp_consumed": 12,
        "krylov_hvp_count": 12,
        "oracle_selection_force_evaluations_delta": 24,
        "selected_direction_sha256": "later-continuation",
        "selected_curvature": -0.2,
        "true_curvature": -0.1,
        "direction_participation_ratio": 3.0,
        "selected_to_previous_selected_abs_cosine": 0.9,
        "continuation_source": "selected_mode",
    }


@pytest.mark.parametrize(
    ("arm", "expected_hvps", "expected_force_evaluations"),
    [
        ("fixed_intent_ritz", 24, 48),
        ("transported_direction", 13, 26),
        ("continuation_lanczos", 24, 48),
    ],
)
def test_direction_trace_contract_supports_variable_post_step_zero_cost(
    arm,
    expected_hvps,
    expected_force_evaluations,
):
    runner = _load_runner()

    audit = runner._validate_direction_trace(
        arm=arm,
        direction_rows=[_step_zero_row(), _later_row(arm)],
    )

    assert audit == {
        "selection_count": 2,
        "direction_oracle_force_evaluations": expected_force_evaluations,
        "hvp_count": expected_hvps,
    }


def test_direction_trace_contract_rejects_a_nonidentical_step_zero_shape():
    runner = _load_runner()
    bad = _step_zero_row()
    bad["krylov_depth"] = 12

    with pytest.raises(RuntimeError, match="step-zero"):
        runner._validate_direction_trace(
            arm="continuation_lanczos",
            direction_rows=[bad, _later_row("continuation_lanczos")],
        )


def _fake_case(state_id, seed, arm, *, meaningful=False):
    trace = [_step_zero_row(direction_hash=f"{state_id}-{seed}")]
    trace.append(_later_row(arm))
    direction_cost = sum(
        row["oracle_selection_force_evaluations_delta"]
        for row in trace
    )
    purposes = {
        "unattributed": 0,
        "bootstrap_true_quench": 0,
        "direction_oracle": direction_cost,
        "biased_proposal_relax": 10,
        "escape_true_pes_check": 5,
        "terminal_true_quench": 20,
        "post_relax_validation": 0,
    }
    return {
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "exact_starter_reference": True,
        "certificate": True,
        "landing_geometry_valid": True,
        "fragmented": False,
        "fallback_used": False,
        "continuation_projection_degenerate": 0,
        "is_new_basin": meaningful,
        "landing_delta_eV": -0.01 if meaningful else 0.01,
        "force_evaluations": sum(purposes.values()),
        "purpose_counts": purposes,
        "direction_selection_count": 2,
        "direction_hvp_count": (
            13 if arm == "transported_direction" else 24
        ),
        "direction_trace_valid": True,
        "direction_trace": trace,
        "generation_wall_time_s": 1.0,
        "quench_wall_time_s": 2.0,
    }


def _fake_cohort(*, transported_extra_event=False):
    rows = []
    for state_id in (
        "intermediate_accepted",
        "plateau_accepted",
    ):
        for seed in (42, 43, 44):
            for arm in (
                "fixed_intent_ritz",
                "transported_direction",
                "continuation_lanczos",
            ):
                rows.append(
                    _fake_case(
                        state_id,
                        seed,
                        arm,
                        meaningful=(
                            transported_extra_event
                            and state_id == "intermediate_accepted"
                            and seed == 42
                            and arm == "transported_direction"
                        ),
                    )
                )
    return rows


def test_analysis_stops_when_continuity_improves_without_terminal_gain():
    analyzer = _load_analyzer()

    evidence = analyzer.analyze_rows(_fake_cohort())

    assert evidence["decision"] == "no_survivor"
    assert evidence["survivors"] == []
    assert (
        evidence["arm_results"]["transported_direction"][
            "median_abs_consecutive_mode_cosine"
        ]
        > evidence["arm_results"]["fixed_intent_ritz"][
            "median_abs_consecutive_mode_cosine"
        ]
    )


def test_analysis_advances_transport_only_for_a_paired_terminal_gain():
    analyzer = _load_analyzer()

    evidence = analyzer.analyze_rows(
        _fake_cohort(transported_extra_event=True)
    )

    assert evidence["decision"] == "transported_direction_survives"
    assert evidence["survivors"] == ["transported_direction"]
