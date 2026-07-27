from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-safe-lbfgs-history-depth-ablation"
)
ANALYZER = RUN_ROOT / "analyze_ablation.py"
RAW_DIR = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
ARMS = ("adaptive-scale-history1", "adaptive-scale-history10")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module():
    assert ANALYZER.is_file()
    spec = importlib.util.spec_from_file_location("safe_lbfgs_history_depth_analysis", ANALYZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(ledger_dir: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--ledger-dir",
            str(ledger_dir),
            "--output-dir",
            str(output_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def _copy_ledger(tmp_path: Path) -> Path:
    destination = tmp_path / "ledger"
    shutil.copytree(RAW_DIR, destination)
    return destination


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


def test_analyzer_is_created_for_the_history_depth_contract() -> None:
    assert ANALYZER.is_file()


def test_certificate_is_derived_only_from_positive_fmax_and_nonnegative_force() -> None:
    analyzer = _module()
    row = {
        "fmax_eV_per_A": 0.05,
        "final_active_max_force_eV_per_A": 0.05,
        "telemetry": {"converged": False},
    }

    assert analyzer.certificate_satisfied(row)
    for fmax, force in (
        (0.0, 0.0),
        (-0.05, 0.0),
        (0.05, -1.0e-12),
        (0.05, 0.0500000001),
        (0.05, float("nan")),
    ):
        mutated = {**row, "fmax_eV_per_A": fmax, "final_active_max_force_eV_per_A": force}
        assert not analyzer.certificate_satisfied(mutated)


def test_real_ledger_yields_minimal_certificate_first_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"
    completed = _run(RAW_DIR, output_dir)
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in output_dir.iterdir()} == {"evidence.json", "conclusion.md"}

    evidence = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["ledger"] == {
        "row_count": 32,
        "task_count": 16,
        "evaluator_closure_validated": True,
    }
    assert evidence["raw_files"] == {
        name: {"sha256": _sha256(RAW_DIR / name)}
        for name in ("summary.json", "c60.json", "pdo.json")
    }
    assert evidence["overall_by_arm"] == {
        "adaptive-scale-history1": {
            "certificate_satisfied_count": 16,
            "row_count": 16,
            "force_evaluations": 1833,
        },
        "adaptive-scale-history10": {
            "certificate_satisfied_count": 16,
            "row_count": 16,
            "force_evaluations": 1494,
        },
    }
    assert evidence["by_system_arm"] == {
        "c60": {
            "adaptive-scale-history1": {
                "certificate_satisfied_count": 8,
                "row_count": 8,
                "force_evaluations": 773,
            },
            "adaptive-scale-history10": {
                "certificate_satisfied_count": 8,
                "row_count": 8,
                "force_evaluations": 718,
            },
        },
        "pdo": {
            "adaptive-scale-history1": {
                "certificate_satisfied_count": 8,
                "row_count": 8,
                "force_evaluations": 1060,
            },
            "adaptive-scale-history10": {
                "certificate_satisfied_count": 8,
                "row_count": 8,
                "force_evaluations": 776,
            },
        },
    }

    c60 = evidence["paired_by_system"]["c60"]
    assert c60["force_evaluation_delta_definition"] == "history10_minus_history1"
    assert c60["force_evaluation_delta_summary"] == {
        "lower_count": 3,
        "tie_count": 1,
        "higher_count": 4,
        "median": 0.5,
    }
    pdo = evidence["paired_by_system"]["pdo"]
    assert pdo["force_evaluation_delta_summary"] == {
        "lower_count": 7,
        "tie_count": 0,
        "higher_count": 1,
        "median": -42.0,
    }
    for system in SYSTEMS:
        pairs = evidence["paired_by_system"][system]["pairs"]
        assert len(pairs) == 8
        assert [pair["seed"] for pair in pairs] == list(range(42, 50))
        for pair in pairs:
            assert set(pair["endpoint_raw_diagnostics"]) == {
                "history10_minus_history1_energy_eV",
                "raw_cartesian_rms_atom_difference_A",
                "raw_cartesian_max_atom_difference_A",
            }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_row", "fixed 32-row matrix"),
        ("broken_evaluator_closure", "evaluator closure"),
    ],
)
def test_analyzer_rejects_incomplete_matrix_and_evaluator_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    ledger_dir = _copy_ledger(tmp_path)
    c60_path = ledger_dir / "c60.json"
    c60 = json.loads(c60_path.read_text(encoding="utf-8"))
    if mutation == "missing_row":
        c60["rows"].pop()
    else:
        c60["rows"][0]["force_evaluations"] += 1
    _write_json(c60_path, c60)

    output_dir = tmp_path / "analysis"
    completed = _run(ledger_dir, output_dir)

    assert completed.returncode != 0
    assert message in completed.stderr
    assert not output_dir.exists()


def test_conclusion_selects_history10_with_scientific_claim_ceiling(tmp_path: Path) -> None:
    output_dir = tmp_path / "analysis"
    completed = _run(RAW_DIR, output_dir)
    assert completed.returncode == 0, completed.stderr

    conclusion = (output_dir / "conclusion.md").read_text(encoding="utf-8")
    for statement in (
        "Both arms satisfied 16/16 force certificates.",
        "History 10 used 1494 evaluator calls; history 1 used 1833.",
        "C60: 718 versus 773 calls",
        "3 lower, 1 tie, and 4 higher",
        "median paired delta +0.5",
        "PdO: 776 versus 1060 calls",
        "7 of 8 paired tasks",
        "median paired delta -42",
        "Advance history 10 to the 200-step validation",
        "C60 effect is not consistent",
        "Endpoints can differ",
        "no endpoint-equivalence claim",
        "no general SSW claim",
    ):
        assert statement in conclusion
