from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pamssw.pbc import mic_displacement


RUN_ROOT = Path(__file__).resolve().parents[2] / "runs" / "20260727-safe-history-capacity-ablation"
ANALYZER = RUN_ROOT / "analyze_ablation.py"
RAW_DIR = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
ARMS = (
    ("safe-total-gradient-history10", 10),
    ("safe-total-gradient-history0", 0),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _rows(ledger_dir: Path) -> list[dict]:
    return [
        row
        for system in SYSTEMS
        for row in json.loads((ledger_dir / f"{system}.json").read_text(encoding="utf-8"))["rows"]
    ]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")


def test_analyzer_is_created_for_the_history_capacity_contract() -> None:
    assert ANALYZER.is_file()


def test_real_ledger_yields_canonical_pairwise_mic_and_certificate_first_evidence(
    tmp_path: Path,
) -> None:
    completed = _run(RAW_DIR, tmp_path / "analysis")
    assert completed.returncode == 0, completed.stderr

    evidence = json.loads((tmp_path / "analysis" / "evidence.json").read_text(encoding="utf-8"))
    summary = json.loads((RAW_DIR / "summary.json").read_text(encoding="utf-8"))
    rows = _rows(RAW_DIR)

    assert evidence["ledger"] == {"row_count": 32, "task_count": 16}
    assert evidence["raw_files"]["summary.json"]["sha256"] == _sha256(RAW_DIR / "summary.json")
    assert evidence["execution"]["git_commit"] == summary["current_git_commit"]
    assert evidence["certificate"]["all_satisfied"] is False
    assert evidence["certificate"]["by_termination_reason"]["maxiter"]["count"] == 12
    assert len(evidence["paired_tasks"]) == 16
    assert "score" not in json.dumps(evidence).lower()

    by_system_arm = evidence["aggregates"]["by_system_arm"]
    for system in SYSTEMS:
        for arm_id, _ in ARMS:
            arm_rows = [row for row in rows if row["system"] == system and row["arm_id"] == arm_id]
            aggregate = by_system_arm[system][arm_id]
            assert aggregate["count"] == len(arm_rows) == 8
            assert aggregate["force_evaluations"]["sum"] == sum(
                row["force_evaluations"] for row in arm_rows
            )
            assert aggregate["telemetry"]["accepted_steps"] == sum(
                row["telemetry"]["accepted_steps"] for row in arm_rows
            )

    pdo_pair = next(
        pair for pair in evidence["paired_tasks"] if pair["system"] == "pdo" and pair["seed"] == 42
    )
    raw_by_arm = {
        row["arm_id"]: row
        for row in rows
        if row["system"] == "pdo" and row["task_id"] == pdo_pair["task_id"]
    }
    source = json.loads(Path(summary["source_summary"]).read_text(encoding="utf-8"))
    source_task = next(
        task["task"]
        for entry in source["systems"]
        if entry["system"] == "pdo"
        for task in entry["tasks"]
        if task["task_id"] == pdo_pair["task_id"]
    )
    state = source_task["initial_state"]
    mic = mic_displacement(
        np.asarray(raw_by_arm["safe-total-gradient-history0"]["endpoint"]["positions"]),
        np.asarray(raw_by_arm["safe-total-gradient-history10"]["endpoint"]["positions"]),
        np.asarray(state["cell"]),
        tuple(state["pbc"]),
    )
    assert pdo_pair["endpoint_delta"]["max_mic_displacement_A"] == pytest.approx(
        float(np.max(np.linalg.norm(mic, axis=1)))
    )
    assert pdo_pair["endpoint_delta"]["pbc"] == [True, True, False]

    conclusion = (tmp_path / "analysis" / "conclusion.md").read_text(encoding="utf-8")
    assert conclusion.index("Certificate outcome") < conclusion.index("Cost outcome")
    assert "positive retained-history contribution" in conclusion
    assert "not retained" in conclusion
    assert "statistical" not in conclusion.lower()
    assert not (tmp_path / "analysis" / "history_capacity_curves.svg").exists()


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ("incomplete", "fixed task-arm matrix"),
        ("duplicate", "fixed task-arm matrix"),
        ("badbool", "boolean"),
        ("nonfinite", "non-finite"),
        ("brokenledger", "trace_records length"),
        ("task_sha", "task sha256"),
    ],
)
def test_fail_closed_schema_rejects_incomplete_duplicate_badbool_nonfinite_and_broken_ledgers(
    tmp_path: Path,
    mutation: str,
    expected_fragment: str,
) -> None:
    ledger_dir = _copy_ledger(tmp_path)
    c60_path = ledger_dir / "c60.json"
    c60 = json.loads(c60_path.read_text(encoding="utf-8"))
    if mutation == "incomplete":
        c60["rows"].pop()
    elif mutation == "duplicate":
        c60["rows"].append(c60["rows"][0])
    elif mutation == "badbool":
        c60["rows"][0]["certificate_satisfied"] = 1
    elif mutation == "nonfinite":
        c60["rows"][0]["trace_records"][0]["total_energy_eV"] = float("nan")
    elif mutation == "brokenledger":
        c60["rows"][0]["trace_records"].pop()
    elif mutation == "task_sha":
        c60["rows"][0]["task_sha256"] = "0" * 64
    else:  # pragma: no cover - protects parametrization edits.
        raise AssertionError(mutation)
    _write_json(c60_path, c60)

    completed = _run(ledger_dir, tmp_path / "analysis")
    assert completed.returncode != 0
    assert expected_fragment in completed.stderr.lower()
    assert not (tmp_path / "analysis" / "evidence.json").exists()


def test_artifacts_are_byte_deterministic_in_two_distinct_output_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_run = _run(RAW_DIR, first)
    second_run = _run(RAW_DIR, second)
    assert first_run.returncode == second_run.returncode == 0

    for name in ("evidence.json", "conclusion.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
