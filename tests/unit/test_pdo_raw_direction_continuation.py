import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260730-direction-mode-continuation"
)
RUNNER_PATH = RUN_ROOT / "run_pdo_raw_transfer.py"
ANALYZER_PATH = RUN_ROOT / "analyze_pdo_raw_transfer.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    return _load(RUNNER_PATH, "_pdo_raw_direction_runner")


def _load_analyzer():
    return _load(ANALYZER_PATH, "_pdo_raw_direction_analyzer")


def test_pdo_raw_protocol_is_one_bootstrap_and_six_paired_actions():
    runner = _load_runner()

    assert runner.SYSTEM == "pdo"
    assert runner.STATE_ID == "raw_bootstrap"
    assert runner.SEEDS == (42, 43, 44)
    assert runner.ARMS == (
        "fixed_intent_ritz",
        "transported_direction",
    )
    assert runner.RAW_INPUT_SHA256 == (
        "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0"
    )
    assert runner.case_matrix() == [
        {"state_id": "raw_bootstrap", "seed": seed, "arm": arm}
        for seed in (42, 43, 44)
        for arm in ("fixed_intent_ritz", "transported_direction")
    ]


def _case(
    seed,
    arm,
    *,
    landing_delta,
    is_new_basin,
    certificate=True,
):
    direction_cost = 48 if arm == "fixed_intent_ritz" else 4
    purposes = {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 0,
        "direction_oracle": direction_cost,
        "escape_true_pes_check": 8,
        "biased_proposal_relax": 20,
        "landing_true_quench": 10,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    return {
        "state_id": "raw_bootstrap",
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "certificate": certificate,
        "landing_geometry_valid": True,
        "fragmentation_applicable": False,
        "fragmented": False,
        "fallback_used": False,
        "continuation_projection_degenerate": 0,
        "is_new_basin": is_new_basin,
        "landing_delta_eV": landing_delta,
        "force_evaluations": sum(purposes.values()),
        "purpose_counts": purposes,
        "direction_trace_valid": True,
        "shared_initial_direction_sha256": f"shared-{seed}",
        "generation_wall_time_s": 2.0,
        "quench_wall_time_s": 1.0,
    }


def test_analysis_separates_shared_bootstrap_and_paired_action_costs():
    analyzer = _load_analyzer()
    bootstrap_purposes = {
        "bootstrap_true_quench": 101,
        "starter_true_quench": 0,
        "direction_oracle": 0,
        "escape_true_pes_check": 0,
        "biased_proposal_relax": 0,
        "landing_true_quench": 0,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    bootstrap = {
        "force_evaluations": 102,
        "purpose_counts": bootstrap_purposes,
        "certificate": True,
        "geometry_valid": True,
        "raw_energy_eV": -100.0,
        "bootstrap_energy_eV": -101.0,
        "energy_drop_eV": 1.0,
        "wall_time_s": 3.0,
    }
    shared = [
        {
            "seed": seed,
            "force_evaluations": 24,
            "purpose_counts": {
                **{key: 0 for key in bootstrap_purposes},
                "direction_oracle": 24,
            },
            "direction_sha256": f"shared-{seed}",
        }
        for seed in (42, 43, 44)
    ]
    rows = []
    for seed in (42, 43, 44):
        rows.append(
            _case(
                seed,
                "fixed_intent_ritz",
                landing_delta=-0.002 if seed == 42 else 0.01,
                is_new_basin=seed == 42,
            )
        )
        rows.append(
            _case(
                seed,
                "transported_direction",
                landing_delta=-0.003 if seed == 42 else 0.005,
                is_new_basin=seed == 42,
            )
        )

    evidence = analyzer.analyze(
        {
            "bootstrap": bootstrap,
            "shared_initial_directions": shared,
            "cases": rows,
        }
    )

    assert evidence["bootstrap"]["force_evaluations"] == 102
    assert evidence["shared_initial_direction_force_evaluations"] == 72
    assert evidence["action_force_evaluations"] == sum(
        row["force_evaluations"] for row in rows
    )
    assert evidence["total_force_evaluations"] == (
        102
        + 72
        + sum(row["force_evaluations"] for row in rows)
    )
    assert evidence["arm_results"]["fixed_intent_ritz"][
        "meaningful_outcome_count"
    ] == 1
    assert evidence["arm_results"]["transported_direction"][
        "meaningful_outcome_count"
    ] == 1
    assert evidence["decision"] == "transported_direction_supported"


def test_analysis_rejects_bootstrap_cost_hidden_inside_an_action():
    analyzer = _load_analyzer()
    row = _case(
        42,
        "fixed_intent_ritz",
        landing_delta=0.0,
        is_new_basin=False,
    )
    row["purpose_counts"]["bootstrap_true_quench"] = 1
    row["force_evaluations"] += 1

    with pytest.raises(ValueError, match="bootstrap"):
        analyzer.validate_action_row(row)


def test_analysis_does_not_promote_empty_terminal_evidence_to_full_support():
    analyzer = _load_analyzer()
    purposes = {
        "bootstrap_true_quench": 20,
        "starter_true_quench": 0,
        "direction_oracle": 0,
        "escape_true_pes_check": 0,
        "biased_proposal_relax": 0,
        "landing_true_quench": 0,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    rows = [
        _case(seed, arm, landing_delta=0.01, is_new_basin=False)
        for seed in (42, 43, 44)
        for arm in ("fixed_intent_ritz", "transported_direction")
    ]
    raw = {
        "bootstrap": {
            "force_evaluations": 21,
            "purpose_counts": purposes,
            "certificate": True,
            "geometry_valid": True,
            "raw_energy_eV": -100.0,
            "bootstrap_energy_eV": -101.0,
            "energy_drop_eV": 1.0,
            "wall_time_s": 3.0,
        },
        "shared_initial_directions": [
            {
                "seed": seed,
                "force_evaluations": 24,
                "purpose_counts": {
                    **{key: 0 for key in purposes},
                    "direction_oracle": 24,
                },
                "direction_sha256": f"shared-{seed}",
            }
            for seed in (42, 43, 44)
        ],
        "cases": rows,
    }

    evidence = analyzer.analyze(raw)

    assert evidence["decision"] == (
        "transported_direction_cost_supported_no_terminal_event"
    )


def test_uncertified_terminal_is_recorded_but_never_meaningful():
    analyzer = _load_analyzer()
    row = _case(
        42,
        "transported_direction",
        landing_delta=-1.0,
        is_new_basin=True,
        certificate=False,
    )

    analyzer.validate_action_row(row)

    assert analyzer._meaningful(row) is False


def _supported_raw():
    purpose_names = (
        "bootstrap_true_quench",
        "starter_true_quench",
        "direction_oracle",
        "escape_true_pes_check",
        "biased_proposal_relax",
        "landing_true_quench",
        "post_relax_validation",
        "unattributed",
    )
    rows = []
    for seed in (42, 43, 44):
        rows.append(
            _case(
                seed,
                "fixed_intent_ritz",
                landing_delta=-1.0,
                is_new_basin=True,
            )
        )
        rows.append(
            _case(
                seed,
                "transported_direction",
                landing_delta=-2.0,
                is_new_basin=True,
            )
        )
    return {
        "execution_commit": "abc",
        "raw_input_sha256": "raw",
        "bootstrap": {
            "force_evaluations": 21,
            "purpose_counts": {
                name: (
                    20
                    if name == "bootstrap_true_quench"
                    else 1
                    if name == "post_relax_validation"
                    else 0
                )
                for name in purpose_names
            },
            "certificate": True,
            "geometry_valid": True,
            "raw_energy_eV": -100.0,
            "bootstrap_energy_eV": -101.0,
            "energy_drop_eV": 1.0,
            "wall_time_s": 3.0,
        },
        "shared_initial_directions": [
            {
                "seed": seed,
                "force_evaluations": 24,
                "purpose_counts": {
                    name: 24 if name == "direction_oracle" else 0
                    for name in purpose_names
                },
                "direction_sha256": f"shared-{seed}",
            }
            for seed in (42, 43, 44)
        ],
        "cases": rows,
    }


def test_repeat_comparison_requires_and_reports_paired_terminal_wins():
    analyzer = _load_analyzer()

    comparison = analyzer.compare_repeats(
        _supported_raw(),
        _supported_raw(),
    )

    assert comparison["decision"] == "repeat_stable_transport_support"
    assert comparison["paired_transport_landing_wins"] == 6
    assert comparison["paired_conditions"] == 6
    assert comparison["aggregate"]["direction_force_evaluations_saved"] > 0
    assert comparison["aggregate"]["action_force_evaluations_saved"] > 0
