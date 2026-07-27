from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_ANALYZER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-proposal-energy-traces"
    / "analyze_traces.py"
)


def _analyzer_module():
    spec = importlib.util.spec_from_file_location("proposal_energy_trace_analyzer", _ANALYZER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(index: int, total_energy: float, *, accepted: bool) -> dict[str, object]:
    return {
        "evaluation_index": index,
        "positions_sha256": f"position-{index}",
        "true_energy_eV": total_energy - 1.0,
        "bias_energy_eV": 1.0,
        "softening_energy_eV": 0.0,
        "total_energy_eV": total_energy,
        "active_max_total_force_eV_per_A": 5.0 / index,
        "accepted_state": accepted,
    }


def _row(*, backend: str, task_id: str, records: list[dict[str, object]], telemetry: dict[str, object]):
    accepted = sum(bool(record["accepted_state"]) for record in records)
    return {
        "system": "pdo",
        "task_id": task_id,
        "seed": 42,
        "backend": backend,
        "force_evaluations": len(records),
        "trace_records": records,
        "accepted_state_count": accepted,
        "non_accepted_evaluation_count": len(records) - accepted,
        "certificate_satisfied": True,
        "wall_time_s": 1.25,
        "final_biased_energy_eV": records[-1]["total_energy_eV"],
        "final_max_active_atom_force_eV_per_A": records[-1]["active_max_total_force_eV_per_A"],
        "telemetry": {
            "evaluator_calls": len(records),
            "termination_reason": "converged",
            "accepted_steps": 0,
            "rejected_steps": 0,
            "line_search_evaluations": 0,
            "explicit_finalization_calls": 0,
            "reporting_evaluator_calls": 0,
            **telemetry,
        },
        "reference_comparison": {
            "all_reference_fields_equal": False,
            "call_delta": 0,
            "certificate_equal": True,
            "termination_equal": True,
            "energy_delta_eV": 0.0,
            "energy_within_measured_neighborhood": True,
            "max_mic_displacement_A": 0.0,
            "position_within_measured_neighborhood": True,
            "rms_mic_displacement_A": 0.0,
            "strict_reference_match": False,
        },
    }


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "current_git_commit": "runner-commit",
        "source_summary_sha256": "source-hash",
        "reference_summary_sha256": "reference-hash",
        "cuda_model_provenance": {"device": "cuda"},
        "systems": [{"system": "pdo", "rows": rows}],
    }


def test_analyzer_separates_safe_finalization_recheck_from_accepted_step_monotonicity():
    analyzer = _analyzer_module()
    records = [
        _record(1, 0.0, accepted=True),
        _record(2, -1.0, accepted=True),
        _record(3, -0.5, accepted=False),
        _record(4, -1.5, accepted=True),
        _record(5, -1.49993896484375, accepted=True),
    ]
    row = _row(
        backend="safe-lbfgs-total",
        task_id="pdo-seed-42-bias-1",
        records=records,
        telemetry={
            "accepted_steps": 2,
            "rejected_steps": 1,
            "line_search_evaluations": 3,
            "explicit_finalization_calls": 1,
            "reporting_evaluator_calls": 1,
        },
    )

    evidence = analyzer.analyze_summary(_summary([row]), input_summary_sha256="ledger-hash")

    task = evidence["tasks"][0]
    assert task["accepted_state_count"] == 4
    assert task["non_accepted_evaluation_count"] == 1
    assert task["non_accepted_minus_telemetry_rejected_steps"] == 0
    assert task["accepted_total_energy_increase_count_including_finalization"] == 1
    assert task["callback_observed_total_energy_increase_count_excluding_explicit_finalization"] == 0
    assert task["safe_lbfgs_accepted_step_total_energy_increase_count"] == 0
    assert task["safe_lbfgs_accepted_step_total_energy_is_monotone_nonincreasing"] is True
    assert task["explicit_finalization_record_count"] == 1
    assert task["finalization_total_energy_delta_eV"] == pytest.approx(6.103515625e-05)


def test_analyzer_rejects_missing_or_inconsistent_accepted_state_labels():
    analyzer = _analyzer_module()
    records = [_record(1, 0.0, accepted=True)]
    del records[0]["accepted_state"]
    row = _row(
        backend="ase-fire",
        task_id="pdo-seed-42-bias-1",
        records=[_record(1, 0.0, accepted=True)],
        telemetry={},
    )
    row["trace_records"] = records

    with pytest.raises(ValueError, match="accepted_state"):
        analyzer.analyze_summary(_summary([row]), input_summary_sha256="ledger-hash")


def test_analyzer_writes_compact_evidence_and_conclusion_without_copying_raw_trace(tmp_path):
    analyzer = _analyzer_module()
    fire = _row(
        backend="ase-fire",
        task_id="pdo-seed-42-bias-1",
        records=[_record(1, 0.0, accepted=True), _record(2, -2.0, accepted=True)],
        telemetry={},
    )
    safe = _row(
        backend="safe-lbfgs-total",
        task_id="pdo-seed-42-bias-1",
        records=[_record(1, 0.0, accepted=True), _record(2, -2.1, accepted=True)],
        telemetry={"accepted_steps": 1},
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary([fire, safe])), encoding="utf-8")
    output_dir = tmp_path / "analysis"

    analyzer.write_analysis_artifacts(summary_path, output_dir, make_plot=False)

    evidence = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["input_summary_sha256"]
    assert evidence["source_run_git_commit"] == "runner-commit"
    assert len(evidence["tasks"]) == 2
    assert "trace_records" not in json.dumps(evidence)
    conclusion = (output_dir / "conclusion.md").read_text(encoding="utf-8")
    assert "Execution/protocol" in conclusion
    assert "does not establish" in conclusion


def test_cli_records_a_repo_relative_reproducible_generator_command(monkeypatch, tmp_path):
    analyzer = _analyzer_module()
    captured: dict[str, object] = {}

    def fake_write(summary_path, output_dir, *, make_plot, command):
        captured.update(
            summary_path=summary_path,
            output_dir=output_dir,
            make_plot=make_plot,
            command=command,
        )
        return {"validation": {"trace_row_count": 0}}

    monkeypatch.setattr(analyzer, "write_analysis_artifacts", fake_write)

    assert analyzer.main(["--summary", "raw.json", "--output-dir", str(tmp_path), "--no-plot"]) == 0
    assert captured["command"] == (
        "python runs/20260727-proposal-energy-traces/analyze_traces.py "
        f"--summary raw.json --output-dir {tmp_path} --no-plot"
    )
