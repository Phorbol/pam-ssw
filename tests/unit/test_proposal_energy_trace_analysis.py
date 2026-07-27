from __future__ import annotations

from copy import deepcopy
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
_SYSTEMS = ("c60", "pdo")
_BACKENDS = ("ase-fire", "safe-lbfgs-total")


def _task_ids(system: str) -> tuple[str, ...]:
    return tuple(f"{system}-seed-{seed}-bias-1" for seed in range(42, 50))


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


def _row(
    *,
    backend: str,
    task_id: str,
    records: list[dict[str, object]],
    telemetry: dict[str, object],
    system: str = "pdo",
):
    accepted = sum(bool(record["accepted_state"]) for record in records)
    return {
        "system": system,
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
        "current_git_commit": "a" * 40,
        "source_summary": "source.json",
        "source_summary_sha256": "b" * 64,
        "reference_summary": "reference.json",
        "reference_summary_sha256": "c" * 64,
        "observation_maxiter": 400,
        "wall_time_total_s": 1.0,
        "historical_reference_neighborhoods": {
            "final_biased_energy_absolute_eV": 5.0e-4,
            "final_positions_max_mic_displacement_A": 2.0e-3,
        },
        "cuda_model_provenance": {
            "cuda_device": "fixture GPU",
            "device": "cuda",
            "model": "fixture.model",
            "model_declared_sha256": "d" * 64,
            "model_sha256": "d" * 64,
            "torch_version": "fixture",
            "source_system_inputs": {
                system: {
                    "input": f"{system}.xyz",
                    "declared_sha256": "e" * 64,
                    "sha256": "e" * 64,
                }
                for system in _SYSTEMS
            },
        },
        "systems": [{"system": "pdo", "task_ids": _task_ids("pdo"), "rows": rows}],
    }


def _full_summary(
    replacements: dict[tuple[str, str, str], dict[str, object]] | None = None,
) -> dict[str, object]:
    replacements = {} if replacements is None else replacements
    systems = []
    for system in _SYSTEMS:
        rows = []
        for task_id in _task_ids(system):
            for backend in _BACKENDS:
                row = _row(
                    system=system,
                    backend=backend,
                    task_id=task_id,
                    records=[_record(1, 0.0, accepted=True)],
                    telemetry={},
                )
                row = deepcopy(replacements.get((system, task_id, backend), row))
                row["system"] = system
                rows.append(row)
        systems.append({"system": system, "task_ids": list(_task_ids(system)), "rows": rows})
    summary = _summary([])
    summary["systems"] = systems
    return summary


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

    evidence = analyzer.analyze_summary(
        _full_summary({("pdo", "pdo-seed-42-bias-1", "safe-lbfgs-total"): row}),
        input_summary_sha256="ledger-hash",
    )

    task = next(task for task in evidence["tasks"] if task["backend"] == "safe-lbfgs-total" and task["task_id"] == "pdo-seed-42-bias-1")
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
        analyzer.analyze_summary(
            _full_summary({("pdo", "pdo-seed-42-bias-1", "ase-fire"): row}),
            input_summary_sha256="ledger-hash",
        )


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
    summary_path.write_text(
        json.dumps(
            _full_summary(
                {
                    ("pdo", "pdo-seed-42-bias-1", "ase-fire"): fire,
                    ("pdo", "pdo-seed-42-bias-1", "safe-lbfgs-total"): safe,
                }
            )
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"

    analyzer.write_analysis_artifacts(summary_path, output_dir, make_plot=False)

    evidence = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["input_summary_sha256"]
    assert evidence["source_run_git_commit"] == "a" * 40
    assert len(evidence["tasks"]) == 32
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


def test_analyzer_rejects_a_partial_frozen_task_matrix():
    analyzer = _analyzer_module()
    summary = _full_summary()
    summary["systems"][0]["rows"].pop()

    with pytest.raises(ValueError, match="frozen task matrix"):
        analyzer.analyze_summary(summary, input_summary_sha256="ledger-hash")


def test_analyzer_rejects_non_boolean_reference_comparison_fields():
    analyzer = _analyzer_module()
    summary = _full_summary()
    summary["systems"][0]["rows"][0]["reference_comparison"]["certificate_equal"] = "false"

    with pytest.raises(ValueError, match="reference_comparison.certificate_equal"):
        analyzer.analyze_summary(summary, input_summary_sha256="ledger-hash")


def test_analyzer_rejects_malformed_required_provenance():
    analyzer = _analyzer_module()
    summary = _full_summary()
    summary["source_summary_sha256"] = "not-a-sha256"

    with pytest.raises(ValueError, match="source_summary_sha256"):
        analyzer.analyze_summary(summary, input_summary_sha256="ledger-hash")


def test_svg_marks_finalization_rechecks_and_is_byte_deterministic(tmp_path):
    analyzer = _analyzer_module()
    records = [
        _record(1, 0.0, accepted=True),
        _record(2, -1.0, accepted=True),
        _record(3, -0.99993896484375, accepted=True),
    ]
    row = _row(
        backend="safe-lbfgs-total",
        task_id="pdo-seed-47-bias-1",
        records=records,
        telemetry={"accepted_steps": 1, "explicit_finalization_calls": 1, "reporting_evaluator_calls": 1},
    )
    summary = _full_summary({("pdo", "pdo-seed-47-bias-1", "safe-lbfgs-total"): row})

    mask = analyzer._explicit_finalization_mask(records, row["telemetry"], label="pdo wrap fixture")
    assert mask.tolist() == [False, False, True]
    first = tmp_path / "first.svg"
    second = tmp_path / "second.svg"
    assert analyzer._write_curve_plot(summary, first) is None
    assert analyzer._write_curve_plot(summary, second) is None
    assert first.read_bytes() == second.read_bytes()
    svg = first.read_text(encoding="utf-8")
    assert "2026-07-27T00:00:00Z" in svg
    assert "explicit finalization recheck" in svg
    assert "explicit-finalization-recheck" in svg
