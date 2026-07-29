"""Tests for the C60 K4 public-profile production analysis."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-c60-k4-profile-200"
    / "analyze_production.py"
)


def load_analyzer(name: str):
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_case(
    root: Path,
    *,
    candidates: int,
    force_evaluations: int,
    direction_evaluations: int,
    best_energy: float,
    minima: int,
    wall_time: float,
    extra_config: dict[str, object] | None = None,
) -> None:
    root.mkdir()
    config = {
        "max_trials": 200,
        "rng_seed": 42,
        "max_steps_per_walk": 8,
        "oracle_candidates": candidates,
        "proposal_relax_steps": 80,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
        "direction_type_ucb_enabled": False,
        "accepted_structures_dir": str(root / "accepted_minima"),
        "accepted_structures_log": str(root / "accepted_structures.jsonl"),
        "direction_diagnostics_path": str(root / "direction_trace.jsonl"),
    }
    config.update(extra_config or {})
    purpose_counts = {
        "biased_proposal_relax": force_evaluations - direction_evaluations - 18,
        "bootstrap_true_quench": 0,
        "direction_oracle": direction_evaluations,
        "escape_true_pes_check": 5,
        "landing_true_quench": 10,
        "post_relax_validation": 1,
        "starter_true_quench": 2,
        "unattributed": 0,
    }
    summary = {
        "execution_commit": ("a" if candidates == 4 else "b") * 40,
        "input_sha256": "c" * 64,
        "model_sha256": "d" * 64,
        "runtime_versions": {"python": "test"},
        "calculator": {"device": "cuda"},
        "effective_config": config,
        "initial_energy_eV": -100.0,
        "best_energy_eV": best_energy,
        "energy_drop_eV": -100.0 - best_energy,
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "timing": {"total_wall_time_s": wall_time},
        "optimizer_telemetry": {
            "proposal_relax_mean_iterations": 10.0,
            "proposal_relax_median_iterations": 8.0,
            "proposal_relax_p90_iterations": 20.0,
            "proposal_relax_termination_maxiter": 2,
            "true_quench_mean_iterations": 12.0,
            "true_quench_median_iterations": 10.0,
            "true_quench_p90_iterations": 25.0,
            "quench_fallback_attempts": 1,
            "quench_fallback_converged": 1,
            "true_quench_termination_converged": 201,
            "true_quench_termination_unconverged": 0,
        },
        "stats": {
            "n_trials": 200,
            "n_minima": minima,
            "duplicate_rate": (201 - minima) / 201,
            "bias_steps": 100,
            "direction_choices": 100,
            "direction_candidate_evaluations": candidates * 100,
            "direction_selected_random": 10,
            "direction_selected_bond": 20,
            "direction_selected_momentum": 70,
        },
    }
    trace = [
        {
            "trial": trial,
            "energy_eV": -100.0 - trial / 200,
            "best_energy_eV": -100.0 + (best_energy + 100.0) * trial / 200,
        }
        for trial in range(201)
    ]
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "energy_trace.json").write_text(json.dumps(trace), encoding="utf-8")


def test_analysis_separates_cost_saving_from_search_quality(tmp_path):
    analyzer = load_analyzer("c60_k4_analysis")
    k4 = tmp_path / "k4"
    k12 = tmp_path / "k12"
    write_case(
        k4,
        candidates=4,
        force_evaluations=800,
        direction_evaluations=80,
        best_energy=-120.0,
        minima=160,
        wall_time=20.0,
        extra_config={"block_krylov_blocks": 2, "block_krylov_depth": 3},
    )
    write_case(
        k12,
        candidates=12,
        force_evaluations=1000,
        direction_evaluations=240,
        best_energy=-125.0,
        minima=140,
        wall_time=18.0,
    )

    evidence = analyzer.analyze(k4, k12)

    assert evidence["protocol"]["scientific_config_diff"] == {
        "oracle_candidates": [12, 4]
    }
    assert evidence["protocol"]["schema_only_k4_fields"] == {
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    }
    assert evidence["comparison"]["total_force_evaluations_saved"] == 200
    assert evidence["comparison"]["direction_force_evaluations_saved"] == 160
    assert evidence["comparison"]["non_direction_force_evaluations_delta"] == -40
    assert evidence["comparison"]["minima_delta"] == 20
    assert evidence["comparison"]["best_energy_delta_eV"] == pytest.approx(5.0)
    assert evidence["decision"]["profile_completed_200_trials"] is True
    assert evidence["decision"]["k4_reduced_total_force_evaluations"] is True
    assert evidence["decision"]["k4_improved_final_best_energy"] is False


def test_analysis_rejects_unregistered_scientific_config_drift(tmp_path):
    analyzer = load_analyzer("c60_k4_analysis_drift")
    k4 = tmp_path / "k4"
    k12 = tmp_path / "k12"
    write_case(
        k4,
        candidates=4,
        force_evaluations=800,
        direction_evaluations=80,
        best_energy=-120.0,
        minima=160,
        wall_time=20.0,
        extra_config={"proposal_fmax": 0.1},
    )
    write_case(
        k12,
        candidates=12,
        force_evaluations=1000,
        direction_evaluations=240,
        best_energy=-125.0,
        minima=140,
        wall_time=18.0,
    )

    with pytest.raises(ValueError, match="production protocol drift"):
        analyzer.analyze(k4, k12)
