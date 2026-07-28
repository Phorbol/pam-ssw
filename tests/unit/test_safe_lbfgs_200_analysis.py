"""Tests for the direct-evidence 200-trial production analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ANALYZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "analyze.py"
)


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "safe_lbfgs_200_analysis", ANALYZER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(system: str, case_dir: Path) -> dict[str, object]:
    config = {
        "max_trials": 200,
        "max_force_evals": None,
        "rng_seed": 42,
        "max_steps_per_walk": 8,
        "proposal_optimizer": "safe-lbfgs-total",
        "quench_optimizer": "scipy-lbfgsb",
        "target_uphill_energy": 0.8,
        "proposal_fmax": 0.05,
        "local_softening_strength": 0.15,
        "local_softening_penalty": "buckingham_repulsive",
        "local_softening_xi": 0.3,
        "accepted_structures_dir": str(case_dir / "accepted_minima"),
        "accepted_structures_log": str(case_dir / "accepted_structures.jsonl"),
        "direction_diagnostics_path": str(case_dir / "direction_trace.jsonl"),
    }
    if system == "c60":
        config.update(
            dedup_rmsd_tol=0.15,
            local_softening_active_count=3,
            local_softening_cutoff_scale=1.3,
            max_step_rms=0.15,
            min_step_scale=0.1,
            oracle_candidates=12,
            proposal_relax_steps=80,
            quench_fmax=0.01,
            target_step_rms=0.08,
            walk_trust_radius=5.0,
        )
    else:
        config.update(
            dedup_rmsd_tol=0.4,
            local_softening_active_count=5,
            local_softening_cutoff_scale=1.15,
            max_step_rms=0.35,
            min_step_scale=0.05,
            oracle_candidates=8,
            proposal_relax_steps=300,
            quench_fmax=0.03,
            target_step_rms=0.15,
            walk_trust_radius=4.0,
        )
    return config


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _production_case(root: Path, system: str) -> None:
    case_dir = root / system
    counts = {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 2,
        "direction_oracle": 20,
        "escape_true_pes_check": 10,
        "biased_proposal_relax": 50,
        "landing_true_quench": 16,
        "post_relax_validation": 2,
        "unattributed": 0,
    }
    converged = 35 if system == "c60" else 201
    unconverged = 201 - converged
    proposal_failures = 4 if system == "c60" else 1
    proposal_count = 40
    stats = {
        "n_trials": 200,
        "configured_max_trials": 200,
        "force_evaluations": 100,
        "n_minima": 21 if system == "c60" else 31,
        "duplicate_rate": 0.2 if system == "c60" else 0.1,
        "proposal_relax_count": proposal_count,
        "proposal_relax_termination_converged": proposal_count
        - proposal_failures,
        "proposal_relax_termination_maxiter": proposal_failures
        if system == "c60"
        else 0,
        "proposal_relax_termination_line_search_failed": proposal_failures
        if system == "pdo"
        else 0,
        "proposal_relax_unconverged": proposal_failures,
        "proposal_relax_outcome_converged_productive": proposal_count
        - proposal_failures,
        "proposal_relax_outcome_energy_exploded": proposal_failures,
        "true_quench_count": 201,
        "true_quench_termination_converged": converged,
        "true_quench_termination_unconverged": unconverged,
        "true_quench_unconverged": unconverged,
        "true_quench_outcome_converged_productive": converged,
        "true_quench_outcome_useful_progress": unconverged,
    }
    telemetry = {
        "proposal_optimizer": "safe-lbfgs-total",
        "quench_optimizer": "scipy-lbfgsb",
        **{
            key: value
            for key, value in stats.items()
            if key.startswith(("proposal_relax_", "true_quench_"))
        },
    }
    walk_records = [
        {
            "trial": trial,
            "seed_entry_id": 0,
            "discovered_entry_id": trial,
            "energy_eV": -10.0 if trial < 2 else -11.0,
            "accepted_new_basin": True,
        }
        for trial in range(1, 201)
    ]
    energy_trace = [
        {
            "trial": trial,
            "energy_eV": -10.0 if trial < 2 else -11.0,
            "best_energy_eV": -10.0 if trial < 2 else -11.0,
            "accepted_new_basin": True,
        }
        for trial in range(201)
    ]
    summary = {
        "schema_version": 1,
        "execution_commit": "e" * 40,
        "system": system,
        "input_path": f"/input/{system}.xyz",
        "input_sha256": system[0] * 64,
        "model_path": "/model/mace.model",
        "model_sha256": "m" * 64,
        "runtime_versions": {"python": "3.test", "mace": "test"},
        "cuda": {"available": True, "device_name": "GPU"},
        "calculator": {
            "device": "cuda",
            "default_dtype": "float32",
            "inference_precision": "float32",
            "enable_cueq": False,
        },
        "safe_lbfgs_default_history_limit": 10,
        "effective_config": _config(system, case_dir),
        "input_state": {
            "atom_count": 60 if system == "c60" else 115,
            "fixed_count": 0 if system == "c60" else 40,
            "pbc": [False, False, False]
            if system == "c60"
            else [True, True, False],
        },
        "initial_energy_eV": -10.0,
        "best_energy_eV": -11.0,
        "energy_drop_eV": 1.0,
        "force_evaluations": 100,
        "purpose_counts": counts,
        "optimizer_telemetry": telemetry,
        "stats": stats,
        "timing": {
            "total_wall_time_s": 50.0,
            "per_trial_wall_time_s": None,
            "per_trial_wall_time_reason": "not exposed",
        },
        "walk_records": walk_records,
    }
    _write_json(case_dir / "summary.json", summary)
    _write_json(case_dir / "energy_trace.json", energy_trace)
    _write_json(case_dir / "walk_records.json", walk_records)
    _write_json(case_dir / "optimizer_diagnostics.json", telemetry)


def _historical_case(root: Path, system: str) -> None:
    case_dir = root / f"{system}_seed42_default8"
    summary = {
        "case_id": f"{system}_seed42_default8",
        "system": system,
        "seed": 42,
        "variant": "default8",
        "initial_energy": -10.0,
        "best_energy": -10.5,
        "energy_drop": 0.5,
        "elapsed_s": 60.0,
        "n_minima": 12,
        "config": {
            "max_force_evals": 58000,
            "proposal_optimizer": "ase-fire",
        },
        "stats": {
            "n_trials": 80,
            "force_evaluations": 58000,
            "budget_exhausted": 1,
            "duplicate_rate": 0.25,
        },
    }
    _write_json(case_dir / "ssw_summary.json", summary)
    _write_json(
        case_dir / "energy_trace.json",
        [
            {
                "trial": trial,
                "energy": -10.0 if trial == 0 else -10.5,
                "best_energy": -10.0 if trial == 0 else -10.5,
                "accepted_new_basin": True,
            }
            for trial in range(81)
        ],
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    production = tmp_path / "production"
    historical = tmp_path / "historical"
    for system in ("c60", "pdo"):
        _production_case(production, system)
        _historical_case(historical, system)
    return production, historical


def test_analysis_reports_direct_metrics_costs_and_best_improvement(tmp_path):
    analyzer = load_analyzer()
    production, historical = _fixture(tmp_path)
    evidence = analyzer.analyze(production, historical)

    c60 = evidence["production"]["c60"]
    assert c60["trials"] == 200
    assert c60["energy_drop_eV"] == pytest.approx(1.0)
    assert c60["force_evaluations"] == 100
    assert c60["wall_time_s"] == pytest.approx(50.0)
    assert c60["archive_entries"] == 21
    assert c60["duplicate_rate"] == pytest.approx(0.2)
    assert c60["cost"]["purpose_counts"]["biased_proposal_relax"] == 50
    assert c60["cost"]["phase_shares"]["proposal_relax"] == pytest.approx(0.5)
    assert c60["proposal_relaxation"]["converged"] == 36
    assert c60["proposal_relaxation"]["failures"] == 4
    assert c60["true_quench"]["converged"] == 35
    assert c60["true_quench"]["unconverged"] == 166
    assert c60["best_improvements"] == [
        {
            "trial": 2,
            "best_energy_eV": -11.0,
            "improvement_eV": 1.0,
        }
    ]
    assert c60["final_best_first_reached_trial"] == 2
    assert evidence["historical_fire_boundary"]["strict_superiority_supported"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("trials", "200 trials"),
        ("purpose_sum", "purpose"),
        ("unattributed", "unattributed"),
        ("config", "config"),
        ("provenance", "provenance"),
    ],
)
def test_analysis_rejects_broken_scientific_contract(
    tmp_path, mutation, message
):
    analyzer = load_analyzer()
    production, historical = _fixture(tmp_path)
    c60_path = production / "c60" / "summary.json"
    c60 = json.loads(c60_path.read_text())
    if mutation == "trials":
        c60["stats"]["n_trials"] = 199
    elif mutation == "purpose_sum":
        c60["purpose_counts"]["direction_oracle"] += 1
    elif mutation == "unattributed":
        c60["purpose_counts"]["unattributed"] = 1
        c60["purpose_counts"]["direction_oracle"] -= 1
    elif mutation == "config":
        c60["effective_config"]["proposal_optimizer"] = "ase-fire"
    else:
        c60["model_sha256"] = "x" * 64
    c60_path.write_text(json.dumps(c60), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        analyzer.analyze(production, historical)


def test_write_outputs_adds_only_evidence_and_conclusion(tmp_path):
    analyzer = load_analyzer()
    production, historical = _fixture(tmp_path)
    before = analyzer.tree_hashes(production)
    output = tmp_path / "analysis"

    evidence = analyzer.write_outputs(production, historical, output)

    assert analyzer.tree_hashes(production) == before
    assert sorted(path.name for path in output.iterdir()) == [
        "conclusion.md",
        "evidence.json",
    ]
    conclusion = (output / "conclusion.md").read_text()
    assert "35/201" in conclusion
    assert "201/201" in conclusion
    assert "不支持严格的 200-step superiority 结论" in conclusion
    assert evidence == json.loads((output / "evidence.json").read_text())
