from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pamssw.walker import StepTargetController


ANALYZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260801-archive-step-target-audit"
    / "analyze.py"
)


def _analyzer():
    name = "_archive_step_target_audit_test"
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_scaled_target_matches_reference_formula():
    analyzer = _analyzer()

    assert analyzer.archive_scaled_target([0.0], fallback=0.8) == pytest.approx(0.8)
    assert analyzer.archive_scaled_target([0.0, -10.0], fallback=0.8) == pytest.approx(2.0)
    assert analyzer.archive_scaled_target([0.0, -100.0], fallback=0.8) == pytest.approx(4.0)


def test_archive_scaled_target_matches_runtime_controller():
    analyzer = _analyzer()
    energies = [0.0, -1.0, -3.0, 0.5]
    archive = SimpleNamespace(
        entries=[SimpleNamespace(energy=energy) for energy in energies]
    )
    runtime = StepTargetController(fallback_target=0.8)

    assert analyzer.archive_scaled_target(energies, fallback=0.8) == pytest.approx(
        runtime.target(archive)
    )


def test_reconstruct_targets_uses_pretrial_archive_and_preserves_duplicate_trial():
    analyzer = _analyzer()
    accepted = [
        {"trial_index": 1, "energy": -1.0},
        {"trial_index": 3, "energy": -3.0},
    ]

    result = analyzer.reconstruct_targets(
        bootstrap_energy_eV=0.0,
        completed_trials=3,
        accepted_rows=accepted,
        fallback_target_eV=0.8,
    )

    assert result["completed_trial_targets_eV"] == pytest.approx([0.8, 0.2, 0.2])
    assert result["next_attempt_target_eV"] == pytest.approx(0.5)
    assert result["reconstructed_archive_entries"] == 3


def test_reconstruct_targets_rejects_unordered_or_out_of_range_trials():
    analyzer = _analyzer()

    with pytest.raises(ValueError, match="strictly increasing"):
        analyzer.reconstruct_targets(
            bootstrap_energy_eV=0.0,
            completed_trials=3,
            accepted_rows=[
                {"trial_index": 2, "energy": -1.0},
                {"trial_index": 1, "energy": -2.0},
            ],
            fallback_target_eV=0.8,
        )
    with pytest.raises(ValueError, match="completed trial range"):
        analyzer.reconstruct_targets(
            bootstrap_energy_eV=0.0,
            completed_trials=2,
            accepted_rows=[{"trial_index": 3, "energy": -1.0}],
            fallback_target_eV=0.8,
        )


def test_reconstruct_targets_rejects_nonfinite_energy():
    analyzer = _analyzer()

    with pytest.raises(ValueError, match="finite"):
        analyzer.reconstruct_targets(
            bootstrap_energy_eV=0.0,
            completed_trials=1,
            accepted_rows=[{"trial_index": 1, "energy": float("nan")}],
            fallback_target_eV=0.8,
        )


def test_analyze_case_closes_archive_target_and_hash(tmp_path: Path):
    analyzer = _analyzer()
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        json.dumps({"trial_index": 1, "energy": -1.0}) + "\n",
        encoding="utf-8",
    )
    expected_hash = analyzer.sha256_file(accepted_path)
    raw_case = {
        "system": "toy",
        "starter_mode": "uniform_archive",
        "seed": 1,
        "bootstrap_energy_eV": 0.0,
        "completed_trials": 1,
        "archive_entries": 2,
        "effective_config": {"target_uphill_energy": 0.8},
        "stats": {
            "adaptive_step_target": 0.2,
            "adaptive_step_multiplier": 1.0,
            "adaptive_progress_boost": 1.0,
        },
    }

    result = analyzer.analyze_case(
        raw_case,
        accepted_path,
        expected_log_sha256=expected_hash,
    )

    assert result["accepted_log_sha256"] == expected_hash
    assert result["completed_trial_targets_eV"] == pytest.approx([0.8])
    assert result["next_attempt_target_eV"] == pytest.approx(0.2)
    assert result["fraction_away_from_reference"] == pytest.approx(0.0)


def test_analyze_case_fails_closed_on_archive_or_hash_mismatch(tmp_path: Path):
    analyzer = _analyzer()
    accepted_path = tmp_path / "accepted.jsonl"
    accepted_path.write_text(
        json.dumps({"trial_index": 1, "energy": -1.0}) + "\n",
        encoding="utf-8",
    )
    raw_case = {
        "system": "toy",
        "starter_mode": "uniform_archive",
        "seed": 1,
        "bootstrap_energy_eV": 0.0,
        "completed_trials": 1,
        "archive_entries": 99,
        "effective_config": {"target_uphill_energy": 0.8},
        "stats": {
            "adaptive_step_target": 0.2,
            "adaptive_step_multiplier": 1.0,
            "adaptive_progress_boost": 1.0,
        },
    }

    with pytest.raises(ValueError, match="archive size"):
        analyzer.analyze_case(raw_case, accepted_path)
    raw_case["archive_entries"] = 2
    with pytest.raises(ValueError, match="SHA-256"):
        analyzer.analyze_case(
            raw_case,
            accepted_path,
            expected_log_sha256="0" * 64,
        )
