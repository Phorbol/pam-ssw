from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.accounting import EvalCounter
from pamssw.state import State


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-c60-post-scipy-rescue-ablation"
)
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
ANALYZER_PATH = RUN_ROOT / "analyze_ablation.py"


def _module(path: Path, name: str):
    assert path.is_file()
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _state(offset: float = 0.0) -> State:
    return State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [offset, 0.0, 0.0],
                [1.0 + offset, 0.0, 0.0],
                [offset, 1.0, 0.0],
            ]
        ),
        cell=None,
        pbc=(False, False, False),
        fixed_mask=None,
    )


def _task(runner, index: int):
    return runner.EndpointTask(
        task_index=index,
        trial_index=index + 1,
        discovered_entry_id=index + 1,
        seed_entry_id=index,
        source_path=Path(f"trial{index + 1:04d}_entry{index + 1:04d}_accepted.xyz"),
        source_sha256=f"{index:064x}",
        source_archive_energy_eV=-10.0 - index,
        state=_state(float(index)),
    )


def _preflight_metadata(runner) -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_commit": "e" * 40,
        "source": {
            "production_summary_sha256": "1" * 64,
            "accepted_structures_sha256": "2" * 64,
            "accepted_input_manifest_sha256": "3" * 64,
            "accepted_input_count": 143,
            "ordering": "trial_index,discovered_entry_id",
            "entries": [],
        },
        "model": {"path": "/model/mace.model", "sha256": "4" * 64},
        "runtime_versions": {"python": "3.test", "mace": "test"},
        "cuda": {"available": True, "device_name": "test GPU"},
        "calculator": dict(runner.CALCULATOR_CONFIG),
        "safe_lbfgs_default_history_limit": 10,
    }


def _raw_row(
    runner,
    *,
    task_index: int,
    arm,
    initial_force: float,
    final_energy: float,
    final_positions: list[list[float]],
    calls: int = 4,
) -> dict[str, object]:
    telemetry = {
        "backend": arm.optimizer,
        "evaluator_calls": calls,
        "converged": True,
        "termination_reason": "converged",
    }
    initial_positions = _state(float(task_index)).positions.tolist()
    return {
        "task_index": task_index,
        "trial_index": task_index + 1,
        "discovered_entry_id": task_index + 1,
        "seed_entry_id": task_index,
        "source_path": f"trial{task_index + 1:04d}_accepted.xyz",
        "source_sha256": f"{task_index:064x}",
        "source_archive_energy_eV": -10.0 - task_index,
        "arm_id": arm.arm_id,
        "optimizer": arm.optimizer,
        "safe_history_limit": arm.safe_history_limit,
        "fmax_eV_per_A": 0.01,
        "maxiter": 400,
        "coordinate_trust_radius_A": None,
        "objective": "true_mace_pes_no_bias_no_softening",
        "initial": {
            "energy_eV": -10.0 - task_index,
            "max_active_force_eV_per_A": initial_force,
            "positions_sha256": f"initial-{task_index}",
            "positions": initial_positions,
        },
        "final": {
            "energy_eV": final_energy,
            "max_active_force_eV_per_A": 0.005,
            "positions_sha256": f"final-{arm.arm_id}-{task_index}",
            "positions": final_positions,
        },
        "n_iter": 3,
        "termination_reason": "converged",
        "outcome_class": "converged_productive",
        "telemetry": telemetry,
        "evaluator_calls": calls,
        "purpose_count_delta": {"post_relax_validation": calls, "unattributed": 0},
        "wall_time_s": 0.1,
    }


def _write_fake_output(tmp_path: Path, runner) -> Path:
    output = tmp_path / "output"
    output.mkdir(parents=True)
    rows = []
    for arm in runner.ARMS:
        for index in range(143):
            if index < 10:
                initial_force = 0.005
            elif index < 30:
                initial_force = 0.03
            elif index < 60:
                initial_force = 0.08
            else:
                initial_force = 0.2
            scipy_positions = _state(float(index)).positions
            final_positions = scipy_positions.copy()
            final_energy = -20.0 - index
            if arm.arm_id == "safe-lbfgs-total-rescue":
                rotation = np.array(
                    [
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                final_positions = scipy_positions @ rotation + np.array([3.0, -2.0, 1.0])
                final_energy += 5.0e-4
                if index == 0:
                    final_energy += 2.0e-3
                if index == 1:
                    final_positions[0, 2] += 1.0
            rows.append(
                _raw_row(
                    runner,
                    task_index=index,
                    arm=arm,
                    initial_force=initial_force,
                    final_energy=final_energy,
                    final_positions=final_positions.tolist(),
                )
            )
    summary = {
        **_preflight_metadata(runner),
        "task_count": 143,
        "row_count": 286,
        "protocol": {
            "fmax_eV_per_A": 0.01,
            "maxiter": 400,
            "coordinate_trust_radius_A": None,
            "objective": "true_mace_pes_no_bias_no_softening",
            "task_selection": "all_accepted_endpoints_without_filtering",
        },
        "arms": [asdict(arm) for arm in runner.ARMS],
        "rows_file": "rows.json",
        "interpretation_scope": "post_scipy_accepted_endpoint_refinement_rescue",
    }
    (output / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return output


def test_runner_and_analyzer_are_created() -> None:
    assert RUNNER_PATH.is_file()
    assert ANALYZER_PATH.is_file()


def test_runner_freezes_all_143_endpoints_and_exact_two_arm_protocol() -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_contract")

    assert [(arm.arm_id, arm.optimizer, arm.safe_history_limit) for arm in runner.ARMS] == [
        ("scipy-lbfgsb-restart", "scipy-lbfgsb", None),
        ("safe-lbfgs-total-rescue", "safe-lbfgs-total", 10),
    ]
    assert runner.EXPECTED_ACCEPTED_COUNT == 143
    assert runner.FMAX == pytest.approx(0.01)
    assert runner.MAXITER == 400
    assert runner.COORDINATE_TRUST_RADIUS is None
    assert runner.OBJECTIVE == "true_mace_pes_no_bias_no_softening"


def test_preflight_binds_commit_clean_manifest_model_runtime_and_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_preflight")
    model = tmp_path / "model"
    model.write_bytes(b"model")
    monkeypatch.setattr(runner, "MODEL_PATH", model)
    monkeypatch.setattr(runner, "EXPECTED_MODEL_SHA256", runner._sha256(model))
    monkeypatch.setattr(runner, "_current_commit", lambda: "e" * 40)
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "_safe_lbfgs_default_history_limit", lambda: 10)
    tasks = tuple(_task(runner, index) for index in range(143))
    manifest = _preflight_metadata(runner)["source"]

    checked = runner.preflight(
        expected_git_commit="e" * 40,
        input_loader=lambda: (tasks, manifest),
        runtime_probe=lambda: {"python": "3.test", "mace": "test"},
        cuda_probe=lambda: {"available": True, "device_name": "test GPU"},
    )

    assert checked.tasks == tasks
    assert checked.metadata["execution_commit"] == "e" * 40
    assert checked.metadata["model"]["sha256"] == runner._sha256(model)
    assert checked.metadata["source"] == manifest
    assert checked.metadata["cuda"]["available"] is True
    assert checked.metadata["safe_lbfgs_default_history_limit"] == 10

    model.write_bytes(b"different model")
    with pytest.raises(RuntimeError, match="model.*SHA256"):
        runner.preflight(
            expected_git_commit="e" * 40,
            input_loader=lambda: (tasks, manifest),
            runtime_probe=lambda: {},
            cuda_probe=lambda: {"available": True},
        )
    model.write_bytes(b"model")
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: False)
    with pytest.raises(RuntimeError, match="clean"):
        runner.preflight(
            expected_git_commit="e" * 40,
            input_loader=lambda: (tasks, manifest),
            runtime_probe=lambda: {},
            cuda_probe=lambda: {"available": True},
        )


@pytest.mark.parametrize("arm_index", [0, 1])
def test_task_execution_records_initial_and_final_true_pes_with_closed_counter_delta(
    arm_index: int,
) -> None:
    runner = _module(RUNNER_PATH, f"c60_post_scipy_rescue_runner_task_{arm_index}")

    class QuadraticCalculator:
        def evaluate_flat(self, flat_positions, template):
            flat = np.asarray(flat_positions, dtype=float)
            return 0.5 * float(np.dot(flat, flat)), flat.copy()

    counter = EvalCounter(QuadraticCalculator())
    row = runner.execute_task(_task(runner, 0), runner.ARMS[arm_index], counter)

    assert row["objective"] == "true_mace_pes_no_bias_no_softening"
    assert row["fmax_eV_per_A"] == pytest.approx(0.01)
    assert row["maxiter"] == 400
    assert row["coordinate_trust_radius_A"] is None
    assert row["initial"]["energy_eV"] == pytest.approx(1.0)
    assert row["initial"]["max_active_force_eV_per_A"] == pytest.approx(1.0)
    assert row["final"]["max_active_force_eV_per_A"] <= 0.01
    assert row["evaluator_calls"] == row["telemetry"]["evaluator_calls"]
    assert row["purpose_count_delta"]["post_relax_validation"] == row["evaluator_calls"]
    assert row["purpose_count_delta"]["unattributed"] == 0
    assert len(row["initial"]["positions"]) == len(row["final"]["positions"]) == 3


def test_run_reuses_one_calculator_per_arm_and_publishes_partial_by_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_publish")
    tasks = tuple(_task(runner, index) for index in range(143))
    preflight = runner.Preflight(
        tasks=tasks,
        metadata=_preflight_metadata(runner),
    )
    events: list[str] = []
    calculator_count = 0

    def preflight_fn(**kwargs):
        events.append("preflight")
        return preflight

    def calculator_factory():
        nonlocal calculator_count
        calculator_count += 1
        events.append(f"calculator-{calculator_count}")
        return object()

    def execute_task(task, arm, counter):
        events.append(f"{arm.arm_id}-{task.task_index}")
        return {"task_index": task.task_index, "arm_id": arm.arm_id}

    monkeypatch.setattr(runner, "execute_task", execute_task)
    output = tmp_path / "output"
    summary = runner.run(
        output_dir=output,
        expected_git_commit="e" * 40,
        preflight_fn=preflight_fn,
        calculator_factory=calculator_factory,
        counter_factory=lambda calculator: SimpleCounter(calculator),
    )

    assert events[0] == "preflight"
    assert calculator_count == 2
    assert summary["task_count"] == 143
    assert summary["row_count"] == 286
    assert output.is_dir()
    assert not output.with_name("output.partial").exists()
    rows = json.loads((output / "rows.json").read_text(encoding="utf-8"))
    assert [(row["arm_id"], row["task_index"]) for row in rows[:143]] == [
        ("scipy-lbfgsb-restart", index) for index in range(143)
    ]
    assert [(row["arm_id"], row["task_index"]) for row in rows[143:]] == [
        ("safe-lbfgs-total-rescue", index) for index in range(143)
    ]


def test_preflight_only_does_not_construct_calculator_or_write_output(tmp_path: Path) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_preflight_only")
    checked = runner.Preflight(
        tasks=tuple(_task(runner, index) for index in range(143)),
        metadata=_preflight_metadata(runner),
    )
    output = tmp_path / "output"

    result = runner.run(
        output_dir=output,
        expected_git_commit="e" * 40,
        preflight_only=True,
        preflight_fn=lambda **_: checked,
        calculator_factory=lambda: pytest.fail("calculator must not be constructed"),
    )

    assert result == checked.metadata
    assert not output.exists()
    assert not output.with_name("output.partial").exists()


class SimpleCounter:
    def __init__(self, calculator):
        self.calculator = calculator


def test_analyzer_derives_loose_initial_certificates_and_current_archive_same_basin(
    tmp_path: Path,
) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_analysis_fixture")
    analyzer = _module(ANALYZER_PATH, "c60_post_scipy_rescue_analyzer")
    raw_dir = _write_fake_output(tmp_path, runner)
    evidence = analyzer.analyze(raw_dir)

    assert evidence["ledger"] == {
        "task_count": 143,
        "row_count": 286,
        "counter_telemetry_closure_validated": True,
    }
    assert evidence["initial_force_certificate_counts_by_arm"] == {
        "scipy-lbfgsb-restart": {"0.10": 60, "0.05": 30, "0.01": 10},
        "safe-lbfgs-total-rescue": {"0.10": 60, "0.05": 30, "0.01": 10},
    }
    assert evidence["terminal_pairing"]["same_basin_definition"] == (
        "abs_energy_delta_eV<=0.001_and_kabsch_rmsd_A<=0.15"
    )
    assert evidence["terminal_pairing"]["same_basin_count"] == 141
    assert evidence["terminal_pairing"]["different_basin_count"] == 2
    assert len(evidence["terminal_pairing"]["pairs"]) == 143


def test_analyzer_rejects_counter_mismatch_and_states_claim_ceiling(tmp_path: Path) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_bad_fixture")
    analyzer = _module(ANALYZER_PATH, "c60_post_scipy_rescue_analyzer_bad_fixture")
    raw_dir = _write_fake_output(tmp_path, runner)
    rows_path = raw_dir / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows[0]["evaluator_calls"] += 1
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="counter.*telemetry"):
        analyzer.analyze(raw_dir)

    good_raw = _write_fake_output(tmp_path / "good", runner)
    evidence = analyzer.analyze(good_raw)
    conclusion = analyzer.render_conclusion(evidence)
    assert "post-SciPy accepted-endpoint refinement/rescue" in conclusion
    assert "not a fair replacement comparison against the original landing process" in conclusion
    assert "current archive semantics" in conclusion
    assert "no third optimizer arm" in conclusion


def test_analyzer_adds_reproducible_cost_force_energy_and_strict_pair_aggregates(
    tmp_path: Path,
) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_aggregate_fixture")
    analyzer = _module(ANALYZER_PATH, "c60_post_scipy_rescue_analyzer_aggregates")
    raw_dir = _write_fake_output(tmp_path, runner)
    rows_path = raw_dir / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    for row in rows:
        index = int(row["task_index"])
        safe = row["arm_id"] == "safe-lbfgs-total-rescue"
        calls = (2 if safe else 1) * (index + 1)
        row["evaluator_calls"] = calls
        row["telemetry"]["evaluator_calls"] = calls
        row["purpose_count_delta"]["post_relax_validation"] = calls
        row["wall_time_s"] = 0.2 if safe else 0.1
        row["final"]["energy_eV"] = row["initial"]["energy_eV"] - (
            0.002 if safe else 0.001
        )
        row["final"]["max_active_force_eV_per_A"] = (
            (0.005 if index < 5 or index >= 133 else 0.2)
            if safe
            else (index + 1) / 1000.0
        )
        row["telemetry"].update(
            accepted_steps=0,
            rejected_steps=0,
            line_search_evaluations=0,
        )
        if safe and index in (0, 1):
            row["termination_reason"] = "line_search_failed"
            row["telemetry"].update(
                termination_reason="line_search_failed",
                accepted_steps=index + 2,
                rejected_steps=index + 4,
                line_search_evaluations=index + 6,
            )
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    evidence = analyzer.analyze(raw_dir)
    scipy = evidence["terminal_by_arm"]["scipy-lbfgsb-restart"]
    safe = evidence["terminal_by_arm"]["safe-lbfgs-total-rescue"]

    assert scipy["evaluator_calls"] == {
        "total": 10296,
        "median": 72.0,
        "p90": 128.8,
        "max": 143,
    }
    assert safe["evaluator_calls"] == {
        "total": 20592,
        "median": 144.0,
        "p90": 257.6,
        "max": 286,
    }
    assert scipy["wall_time_s_total"] == pytest.approx(14.3)
    assert safe["wall_time_s_total"] == pytest.approx(28.6)
    assert scipy["terminal_force_eV_per_A"] == {
        "median": pytest.approx(0.072),
        "p90": pytest.approx(0.1288),
        "max": pytest.approx(0.143),
    }
    assert scipy["energy_change_eV"] == {
        "definition": "final_energy_eV_minus_initial_energy_eV",
        "total": pytest.approx(-0.143),
        "median": pytest.approx(-0.001),
        "p90": pytest.approx(-0.001),
        "min": pytest.approx(-0.001),
        "max": pytest.approx(-0.001),
        "decreased_count": 143,
        "unchanged_count": 0,
        "increased_count": 0,
    }
    assert evidence["terminal_pairing"][
        "strict_force_certificate_0.01_contingency"
    ] == {
        "safe_only": 10,
        "scipy_only": 5,
        "both": 5,
        "neither": 123,
    }
    assert evidence["safe_line_search_failed"] == {
        "count": 2,
        "evaluator_calls_total": 6,
        "evaluator_calls_median": 3.0,
        "accepted_steps_total": 5,
        "rejected_steps_total": 9,
        "line_search_evaluations_total": 13,
    }

    conclusion = analyzer.render_conclusion(evidence)
    assert "59/143 versus 32/143" not in conclusion
    assert "20,592 versus 10,296" in conclusion
    assert "2 line-search failures consumed 6 evaluator calls" in conclusion
    assert "does not equal the float32 ULP" in conclusion


def test_analyzer_reports_float32_energy_resolution_as_inference_not_root_cause(
    tmp_path: Path,
) -> None:
    runner = _module(RUNNER_PATH, "c60_post_scipy_rescue_runner_resolution_fixture")
    analyzer = _module(ANALYZER_PATH, "c60_post_scipy_rescue_analyzer_resolution")
    raw_dir = _write_fake_output(tmp_path, runner)
    rows_path = raw_dir / "rows.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    ulp = abs(float(np.spacing(np.float32(-500.0))))
    for index, row in enumerate(rows):
        row["initial"]["energy_eV"] = -500.0
        row["final"]["energy_eV"] = -500.0 if index == 0 else -500.0 - ulp
    rows_path.write_text(json.dumps(rows), encoding="utf-8")

    evidence = analyzer.analyze(raw_dir)

    assert evidence["energy_resolution"] == {
        "minimum_nonzero_absolute_final_minus_initial_energy_eV": ulp,
        "float32_reference_energy_eV": -500.0,
        "float32_ulp_at_reference_energy_eV": ulp,
        "minimum_step_equals_float32_ulp": True,
        "interpretation": (
            "This numerical match supports the inference that energy quantization "
            "may contribute to Armijo stalling near minima; it does not prove the "
            "root cause."
        ),
    }
    conclusion = analyzer.render_conclusion(evidence)
    assert "supports the inference" in conclusion
    assert "may contribute to Armijo stalling near minima" in conclusion
    assert "does not prove the root cause" in conclusion
