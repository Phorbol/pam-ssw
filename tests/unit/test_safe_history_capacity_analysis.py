from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pamssw import pbc as pbc_module
from pamssw.pbc import mic_displacement


RUN_ROOT = Path(__file__).resolve().parents[2] / "runs" / "20260727-safe-history-capacity-ablation"
ANALYZER = RUN_ROOT / "analyze_ablation.py"
PLOTTER = RUN_ROOT / "plot_ablation.py"
COMMITTED_SVG = RUN_ROOT / "history_capacity_curves.svg"
RUNNER = RUN_ROOT / "run_gpu_ablation.py"
TRACE_RECORDER = RUN_ROOT.parent / "20260727-proposal-energy-traces" / "trace_recorder.py"
RAW_DIR = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
ARMS = (
    ("safe-total-gradient-history10", 10),
    ("safe-total-gradient-history0", 0),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def _plot(ledger_dir: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PLOTTER),
            "--ledger-dir",
            str(ledger_dir),
            "--output",
            str(output_path),
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


def test_plotter_is_created_for_the_history_capacity_contract() -> None:
    assert PLOTTER.is_file()


def test_plot_contains_reviewed_system_metric_arm_and_trace_semantics(tmp_path: Path) -> None:
    output = tmp_path / "history_capacity_curves.svg"
    completed = _plot(RAW_DIR, output)
    assert completed.returncode == 0, completed.stderr

    svg = output.read_text(encoding="utf-8")
    assert all(line == line.rstrip() for line in svg.splitlines())
    for label in (
        "C60",
        "PdO",
        "Total biased energy (eV)",
        "Active max total force (eV/Å)",
        "History 10",
        "History 0",
        "Task ID (color)",
        "Arm (line style)",
        "All exact evaluations",
        "Callback-observed accepted_state (not optimizer acceptance rule)",
        "Callback-nonaccepted evaluation",
        "Explicit finalization recheck",
    ):
        assert label in svg


def test_plotter_rejects_incomplete_ledger_through_shared_validator(tmp_path: Path) -> None:
    ledger_dir = _copy_ledger(tmp_path)
    c60_path = ledger_dir / "c60.json"
    c60 = json.loads(c60_path.read_text(encoding="utf-8"))
    c60["rows"].pop()
    _write_json(c60_path, c60)

    output = tmp_path / "history_capacity_curves.svg"
    completed = _plot(ledger_dir, output)

    assert completed.returncode != 0
    assert "fixed task-arm matrix" in completed.stderr.lower()
    assert not output.exists()


def test_plot_is_byte_deterministic_and_matches_committed_svg(tmp_path: Path) -> None:
    first = tmp_path / "first" / "history_capacity_curves.svg"
    second = tmp_path / "second" / "history_capacity_curves.svg"
    first_run = _plot(RAW_DIR, first)
    second_run = _plot(RAW_DIR, second)
    assert first_run.returncode == second_run.returncode == 0

    assert first.read_bytes() == second.read_bytes()
    assert first.read_bytes() == COMMITTED_SVG.read_bytes()


def test_analysis_anchors_are_fixed_to_the_reviewed_execution() -> None:
    analyzer = _module(ANALYZER, "safe_history_capacity_analysis_anchors")

    assert analyzer.EXPECTED_SOURCE_SUMMARY_SHA256 == (
        "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
    )
    assert analyzer.EXPECTED_KERNEL_DESCRIPTOR_SHA256 == (
        "1846e340e762b79f50897dfacd40a16288ed52bb481f36f24ab01594c4724102"
    )
    assert analyzer.EXPECTED_EXECUTION_COMMIT == "0d651ac31e8e4cd12fd2935a0cfb2d66715d2c48"


def test_analyzer_position_hash_matches_trace_recorder_and_raw_endpoints() -> None:
    analyzer = _module(ANALYZER, "safe_history_capacity_analysis_position_hash")
    recorder = _module(TRACE_RECORDER, "safe_history_capacity_trace_recorder_test")

    for row in _rows(RAW_DIR):
        positions = row["endpoint"]["positions"]
        assert analyzer.position_sha256(positions) == recorder.position_hash(positions)
        assert analyzer.position_sha256(positions) == row["trace_records"][-1]["positions_sha256"]


def test_analyzer_task_sha_matches_runner_canonical_hashes_for_pinned_source_payloads() -> None:
    analyzer = _module(ANALYZER, "safe_history_capacity_analysis")
    runner = _module(RUNNER, "safe_history_capacity_runner_for_analysis")
    summary = json.loads((RAW_DIR / "summary.json").read_text(encoding="utf-8"))
    source = json.loads(Path(summary["source_summary"]).read_text(encoding="utf-8"))
    raw_by_task_arm = {
        (row["system"], row["task_id"], row["arm_id"]): row["task_sha256"]
        for row in _rows(RAW_DIR)
    }

    for entry in source["systems"]:
        for task in entry["tasks"]:
            key = (entry["system"], task["task_id"])
            expected = runner.canonical_task_sha256(task["task"])
            assert analyzer.canonical_task_sha256(task["task"]) == expected
            for arm_id, _ in ARMS:
                assert raw_by_task_arm[(key[0], key[1], arm_id)] == expected


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
    assert evidence["analysis_script_sha256"] == _sha256(ANALYZER)
    assert evidence["execution"]["git_commit"] == summary["current_git_commit"]
    mic_provenance = evidence["provenance"]["mic_implementation"]
    assert mic_provenance == {
        "module": "pamssw.pbc",
        "path": str(Path(pbc_module.__file__).resolve()),
        "sha256": _sha256(Path(pbc_module.__file__).resolve()),
    }
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
    aggregates = evidence["aggregates"]["by_system_arm"]
    history10 = (
        aggregates["c60"]["safe-total-gradient-history10"]["force_evaluations"]["sum"]
        + aggregates["pdo"]["safe-total-gradient-history10"]["force_evaluations"]["sum"]
    )
    history0 = (
        aggregates["c60"]["safe-total-gradient-history0"]["force_evaluations"]["sum"]
        + aggregates["pdo"]["safe-total-gradient-history0"]["force_evaluations"]["sum"]
    )
    assert f"History 10: 16/16 certificate-satisfied rows and {history10} evaluator calls." in conclusion
    assert (
        f"History 0: 4/16 certificate-satisfied rows, 12 finite {chr(96)}maxiter{chr(96)} "
        f"rows, and {history0} evaluator calls."
    ) in conclusion
    assert (
        "Each arm cost includes all 16 rows; the history-0 cost includes its incomplete rows "
        "and is not a cheap-success comparison."
    ) in conclusion
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
        ("task_sha", "task_sha256"),
        ("task_sha_pair", "task_sha256"),
        ("unknown_fields", "ledger row keys"),
        ("source_anchor", "source summary sha256"),
        ("descriptor_anchor", "kernel descriptor sha256"),
        ("commit_anchor", "execution commit"),
        ("schema_float", "schema_version"),
        ("row_count_float", "row_count"),
        ("force_string", "integer"),
        ("force_bool", "integer"),
        ("endpoint_force", "last trace force"),
        ("endpoint_shape", "shape"),
        ("endpoint_coordinates", "position hash"),
        ("endpoint_energy", "last trace total energy"),
        ("trace_all_false", "accepted_state"),
        ("accepted_secants", "secants"),
        ("numeric_string_position", "numeric values"),
        ("unknown_summary", "summary keys"),
        ("unknown_telemetry", "telemetry keys"),
        ("unknown_trace", "trace record keys"),
        ("unknown_purpose", "purpose_counts keys"),
        ("unknown_provenance", "git provenance keys"),
        ("negative_force_closure", "nonnegative"),
        ("negative_trace_force", "nonnegative"),
        ("negative_wall_time", "nonnegative"),
        ("negative_summary_wall_time", "nonnegative"),
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
    elif mutation == "task_sha_pair":
        task_id = c60["rows"][0]["task_id"]
        for row in c60["rows"]:
            if row["task_id"] == task_id:
                row["task_sha256"] = "0" * 64
    elif mutation == "unknown_fields":
        for system in SYSTEMS:
            path = ledger_dir / f"{system}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                row["unreviewed_extra_field"] = True
            _write_json(path, payload)
    elif mutation == "source_anchor":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_path = ledger_dir / "tampered-source-summary.json"
        source = json.loads(Path(summary["source_summary"]).read_text(encoding="utf-8"))
        source["unreviewed_extra_field"] = True
        _write_json(source_path, source)
        summary["source_summary"] = str(source_path)
        summary["source_summary_sha256"] = _sha256(source_path)
        _write_json(summary_path, summary)
    elif mutation == "descriptor_anchor":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        descriptor = summary["safe_kernel_descriptor"]
        descriptor["kernel_constants"]["_SAFE_LBFGS_BACKTRACK"] = 0.25
        serialized = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        summary["safe_kernel_descriptor_sha256"] = hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()
        _write_json(summary_path, summary)
        for system in SYSTEMS:
            path = ledger_dir / f"{system}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                row["objective_descriptor"] = descriptor
            _write_json(path, payload)
    elif mutation == "commit_anchor":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        fake_commit = "1" * 40
        summary["current_git_commit"] = fake_commit
        summary["git_provenance"]["actual_git_commit"] = fake_commit
        summary["git_provenance"]["expected_git_commit"] = fake_commit
        _write_json(summary_path, summary)
    elif mutation == "schema_float":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["schema_version"] = 1.0
        _write_json(summary_path, summary)
    elif mutation == "row_count_float":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["row_count"] = 32.0
        _write_json(summary_path, summary)
    elif mutation == "force_string":
        c60["rows"][0]["force_evaluations"] = str(c60["rows"][0]["force_evaluations"])
    elif mutation == "force_bool":
        c60["rows"][0]["force_evaluations"] = True
    elif mutation == "endpoint_force":
        c60["rows"][0]["endpoint"]["max_active_atom_force_eV_per_A"] = 999.0
    elif mutation == "endpoint_shape":
        c60["rows"][0]["endpoint"]["positions"] = [[0.0, 0.0, 0.0]]
    elif mutation == "endpoint_coordinates":
        c60["rows"][0]["endpoint"]["positions"][0][0] += 0.1
    elif mutation == "endpoint_energy":
        c60["rows"][0]["endpoint"]["biased_energy_eV"] += 1.0
    elif mutation == "trace_all_false":
        for trace in c60["rows"][0]["trace_records"]:
            trace["accepted_state"] = False
    elif mutation == "accepted_secants":
        c60["rows"][0]["telemetry"]["accepted_secants"] = 0
    elif mutation == "numeric_string_position":
        c60["rows"][0]["endpoint"]["positions"][0][0] = "1.25"
    elif mutation == "unknown_summary":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["unreviewed_extra_field"] = True
        _write_json(summary_path, summary)
    elif mutation == "unknown_telemetry":
        c60["rows"][0]["telemetry"]["unreviewed_extra_field"] = 0
    elif mutation == "unknown_trace":
        c60["rows"][0]["trace_records"][0]["unreviewed_extra_field"] = 0
    elif mutation == "unknown_purpose":
        c60["rows"][0]["purpose_counts"]["unreviewed_extra_field"] = 0
    elif mutation == "unknown_provenance":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["git_provenance"]["unreviewed_extra_field"] = True
        _write_json(summary_path, summary)
    elif mutation == "negative_force_closure":
        c60["rows"][0]["endpoint"]["max_active_atom_force_eV_per_A"] = -1.0
        c60["rows"][0]["trace_records"][-1]["active_max_total_force_eV_per_A"] = -1.0
    elif mutation == "negative_trace_force":
        c60["rows"][0]["trace_records"][0]["active_max_total_force_eV_per_A"] = -1.0
    elif mutation == "negative_wall_time":
        c60["rows"][0]["wall_time_s"] = -999999.0
    elif mutation == "negative_summary_wall_time":
        summary_path = ledger_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["wall_time_total_s"] = -999999.0
        _write_json(summary_path, summary)
    else:  # pragma: no cover - protects parametrization edits.
        raise AssertionError(mutation)
    if mutation not in {
        "unknown_fields",
        "source_anchor",
        "descriptor_anchor",
        "commit_anchor",
        "schema_float",
        "row_count_float",
        "unknown_summary",
        "unknown_provenance",
        "negative_summary_wall_time",
    }:
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
