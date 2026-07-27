"""Contract tests for the five-trial GPU no-loss validation harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from hashlib import sha256

import pytest


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-direction-oracle-no-loss-gpu-validation"
)
RUNNER_PATH = RUN_ROOT / "run_validation.py"
ANALYZER_PATH = RUN_ROOT / "analyze.py"
REFERENCE_PATH = RUN_ROOT / "reference.json"
FROZEN_REFERENCE_SHA256 = "a8c3b9c5aaa0d5085118a1cfa1d62a4c641c9ad1383df49cb1285982ee607864"


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _direction_rows() -> list[dict[str, object]]:
    return [
        {
            "trial": 0,
            "proposal": 0,
            "step": 0,
            "selected_kind": "random",
            "curvature": 1.0,
            "candidate_count": 2,
            "anchor_cosine": 0.1,
        },
        {
            "trial": 0,
            "proposal": 0,
            "step": 1,
            "selected_kind": "momentum",
            "curvature": 2.0,
            "candidate_count": 3,
            "anchor_cosine": 0.2,
        },
        {
            "trial": 1,
            "proposal": 0,
            "step": 0,
            "selected_kind": "bond",
            "curvature": 3.0,
            "candidate_count": 2,
            "anchor_cosine": 0.3,
        },
    ]


def _reference(system: str) -> dict[str, object]:
    energy_trace = [
        {
            "trial": trial,
            "energy_eV": -10.0 - trial,
            "best_energy_eV": -10.0 - trial,
            "accepted_new_basin": True,
        }
        for trial in range(6)
    ]
    walk_records = [
        {
            "trial": trial,
            "seed_entry_id": trial - 1,
            "discovered_entry_id": trial,
            "energy_eV": -10.0 - trial,
            "accepted_new_basin": True,
        }
        for trial in range(1, 6)
    ]
    return {
        "system": system,
        "old_execution_commit": "b" * 40,
        "old_artifact_sha256": {
            "summary": "s" * 64,
            "energy_trace": "e" * 64,
            "walk_records": "w" * 64,
            "direction_trace": "d" * 64,
        },
        "energy_trace": energy_trace,
        "walk_records": walk_records,
        "direction_trace": _direction_rows(),
    }


def _write_old_case(root: Path, system: str) -> None:
    reference = _reference(system)
    case_dir = root / system
    _write_json(case_dir / "summary.json", {"system": system, "execution_commit": reference["old_execution_commit"]})
    _write_json(case_dir / "energy_trace.json", reference["energy_trace"] + [{"trial": 6}])
    _write_json(case_dir / "walk_records.json", reference["walk_records"] + [{"trial": 6}])
    (case_dir / "direction_trace.jsonl").parent.mkdir(parents=True, exist_ok=True)
    rows = _direction_rows() + [{**_direction_rows()[0], "trial": 6}]
    (case_dir / "direction_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_new_case(root: Path, system: str, reference: dict[str, object]) -> None:
    purpose_counts = {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 1,
        "direction_oracle": 14,
        "escape_true_pes_check": 5,
        "biased_proposal_relax": 20,
        "landing_true_quench": 8,
        "post_relax_validation": 1,
        "unattributed": 0,
    }
    _write_json(
        root / system / "summary.json",
        {
            "system": system,
            "execution_commit": "n" * 40,
            "effective_config": {
                "max_trials": 5,
                "direction_selection_mode": "discrete",
                "direction_synthesis_mode": "none",
                "direction_probe_enabled": False,
                "plateau_evolution_enabled": False,
            },
            "stats": {"n_trials": 5, "configured_max_trials": 5},
            "force_evaluations": sum(purpose_counts.values()),
            "purpose_counts": purpose_counts,
        },
    )
    _write_json(root / system / "energy_trace.json", reference["energy_trace"])
    _write_json(root / system / "walk_records.json", reference["walk_records"])
    (root / system / "direction_trace.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reference["direction_trace"]),
        encoding="utf-8",
    )


def test_freeze_reference_keeps_old_commit_hashes_and_first_five_trials(tmp_path):
    runner = _load(RUNNER_PATH, "direction_oracle_no_loss_runner")
    old_root = tmp_path / "old"
    reference_path = tmp_path / "reference.json"
    for system in runner.SYSTEMS:
        _write_old_case(old_root, system)

    reference = runner.freeze_reference(old_root, reference_path)

    assert reference_path.is_file()
    assert reference["trial_count"] == 5
    assert reference["systems"]["c60"]["old_execution_commit"] == "b" * 40
    assert [row["trial"] for row in reference["systems"]["pdo"]["energy_trace"]] == list(range(6))
    assert [row["trial"] for row in reference["systems"]["c60"]["walk_records"]] == list(range(1, 6))
    assert all(row["trial"] < 5 for row in reference["systems"]["pdo"]["direction_trace"])


def test_real_reference_is_hash_pinned_to_the_old_execution_artifacts():
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    assert sha256(REFERENCE_PATH.read_bytes()).hexdigest() == FROZEN_REFERENCE_SHA256
    assert payload["trial_count"] == 5
    for system in ("c60", "pdo"):
        case = payload["systems"][system]
        assert case["old_execution_commit"] == "b2dc9c46932533f8f5be9262ca89555538e40159"
        assert set(case["old_artifact_sha256"]) == {
            "summary",
            "energy_trace",
            "walk_records",
            "direction_trace",
        }
        assert all(len(value) == 64 for value in case["old_artifact_sha256"].values())


def test_validation_config_changes_only_max_trials(tmp_path):
    runner = _load(RUNNER_PATH, "direction_oracle_no_loss_runner_config")

    c60 = runner.build_config("c60", tmp_path / "c60")
    pdo = runner.build_config("pdo", tmp_path / "pdo")

    assert c60.max_trials == pdo.max_trials == 5
    source_c60 = runner.PRODUCTION_RUNNER.build_config("c60", tmp_path / "c60")
    source_pdo = runner.PRODUCTION_RUNNER.build_config("pdo", tmp_path / "pdo")
    assert {**c60.__dict__, "max_trials": 200} == source_c60.__dict__
    assert {**pdo.__dict__, "max_trials": 200} == source_pdo.__dict__


def test_analyzer_requires_exact_trajectory_and_reports_escape_only_savings(tmp_path):
    analyzer = _load(ANALYZER_PATH, "direction_oracle_no_loss_analyzer")
    reference_path = tmp_path / "reference.json"
    reference = {
        "schema_version": 1,
        "trial_count": 5,
        "systems": {system: _reference(system) for system in analyzer.SYSTEMS},
    }
    _write_json(reference_path, reference)
    output_root = tmp_path / "output"
    for system in analyzer.SYSTEMS:
        _write_new_case(output_root, system, reference["systems"][system])

    evidence = analyzer.analyze(reference_path, output_root)

    c60 = evidence["systems"]["c60"]
    assert c60["trajectory_exact"] is True
    assert c60["savings"]["escape_true_pes_check"] == 7
    assert c60["savings"]["direction_oracle"] == 0
    assert c60["savings"]["biased_proposal_relax"] == 0
    assert c60["mechanism_savings"] == {
        "true_curvature_hvp": 6,
        "true_after_carry": 1,
    }

    broken = _reference("c60")
    broken["energy_trace"][3]["energy_eV"] = -99.0
    _write_json(output_root / "c60" / "energy_trace.json", broken["energy_trace"])
    with pytest.raises(ValueError, match="energy_trace"):
        analyzer.analyze(reference_path, output_root)
