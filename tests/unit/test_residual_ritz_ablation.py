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
RUNNER_PATH = RUN_ROOT / "run_residual_ritz_ablation.py"
ANALYZER_PATH = RUN_ROOT / "analyze_residual_ritz_ablation.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_protocol_is_exactly_ninety_paired_actions():
    runner = _load(RUNNER_PATH, "_residual_ritz_runner")

    assert runner.SEEDS == tuple(range(42, 52))
    assert runner.ARMS == (
        "transported_direction",
        "residual_ritz2",
        "fixed_intent_ritz",
    )
    assert runner.RESIDUAL_RITZ_CONFIG == {
        "direction_selection_mode": "continuation_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 2,
    }
    assert len(runner.case_matrix("c60")) == 60
    assert len(runner.case_matrix("pdo")) == 30


def _row(system, state_id, seed, arm, *, delta, total, direction):
    purposes = {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 0,
        "direction_oracle": direction,
        "escape_true_pes_check": 1,
        "biased_proposal_relax": total - direction - 3,
        "landing_true_quench": 1,
        "terminal_true_quench": 0,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    return {
        "system": system,
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "certificate": True,
        "landing_geometry_valid": True,
        "landing_delta_eV": delta,
        "force_evaluations": total,
        "purpose_counts": purposes,
        "quench_iterations": 1,
        "generation_wall_time_s": 1.0,
        "quench_wall_time_s": 1.0,
        "relaxation_diagnostics": {
            "proposal_relax_outcome_energy_exploded": 0,
            "proposal_relax_unconverged": 0,
        },
    }


def _cohort(*, residual_wins):
    rows = []
    states = {
        "c60": ("intermediate_accepted", "plateau_accepted"),
        "pdo": ("raw_bootstrap",),
    }
    for system, state_ids in states.items():
        for state_id in state_ids:
            for seed in range(42, 52):
                rows.extend(
                    [
                        _row(
                            system,
                            state_id,
                            seed,
                            "transported_direction",
                            delta=-0.1,
                            total=40,
                            direction=2,
                        ),
                        _row(
                            system,
                            state_id,
                            seed,
                            "residual_ritz2",
                            delta=-0.2 if residual_wins else 0.0,
                            total=45,
                            direction=4,
                        ),
                        _row(
                            system,
                            state_id,
                            seed,
                            "fixed_intent_ritz",
                            delta=-0.3,
                            total=60,
                            direction=24,
                        ),
                    ]
                )
    return rows


def test_analysis_advances_only_if_residual_ritz_beats_transport_in_both_systems():
    analyzer = _load(ANALYZER_PATH, "_residual_ritz_analyzer")

    supported = analyzer.analyze_cases(_cohort(residual_wins=True))
    rejected = analyzer.analyze_cases(_cohort(residual_wins=False))

    assert supported["decision"] == "advance_residual_ritz2"
    assert supported["systems"]["c60"]["residual_vs_transport"][
        "landing_delta_wins_ties_losses"
    ] == [20, 0, 0]
    assert supported["systems"]["pdo"]["residual_vs_transport"][
        "landing_delta_wins_ties_losses"
    ] == [10, 0, 0]
    assert rejected["decision"] == "reject_residual_ritz2"


def test_raw_analysis_closes_shared_and_bootstrap_costs_and_provenance():
    analyzer = _load(ANALYZER_PATH, "_residual_ritz_raw_analyzer")
    rows = _cohort(residual_wins=True)
    shared_purpose = {
        "direction_oracle": 24,
        "unattributed": 0,
    }
    c60 = {
        "execution_commit": "abc",
        "shared_provenance": {"model_sha256": "model"},
        "shared_initial_directions": [
            {
                "force_evaluations": 24,
                "purpose_counts": shared_purpose,
            }
            for _ in range(20)
        ],
        "cases": [
            {key: value for key, value in row.items() if key != "system"}
            for row in rows
            if row["system"] == "c60"
        ],
    }
    bootstrap_purpose = {
        "bootstrap_true_quench": 85,
        "unattributed": 0,
    }
    pdo = {
        "execution_commit": "abc",
        "model_sha256": "model",
        "raw_input_sha256": "pdo-input",
        "bootstrap": {
            "certificate": True,
            "geometry_valid": True,
            "force_evaluations": 85,
            "purpose_counts": bootstrap_purpose,
        },
        "shared_initial_directions": [
            {
                "force_evaluations": 24,
                "purpose_counts": shared_purpose,
            }
            for _ in range(10)
        ],
        "cases": [
            {key: value for key, value in row.items() if key != "system"}
            for row in rows
            if row["system"] == "pdo"
        ],
    }

    evidence = analyzer.analyze_raw(c60, pdo)

    assert evidence["provenance"]["execution_commit"] == "abc"
    assert evidence["cost_scope"]["c60_shared_direction_fe"] == 480
    assert evidence["cost_scope"]["pdo_bootstrap_fe"] == 85
    assert evidence["cost_scope"]["pdo_shared_direction_fe"] == 240

    pdo["execution_commit"] = "different"
    with pytest.raises(ValueError, match="execution commit"):
        analyzer.analyze_raw(c60, pdo)
