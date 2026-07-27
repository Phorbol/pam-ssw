"""Contract tests for the fixed raw-landing tiered true-quench ablation."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from ase import Atoms
from ase.io import write
import numpy as np
import pytest

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.result import RelaxOutcomeClass, RelaxResult, RelaxTelemetry
from pamssw.state import State


ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-true-quench-tiered-ablation"
)
RUNNER_PATH = ROOT / "run_ablation.py"
ANALYZER_PATH = ROOT / "analyze_ablation.py"


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_protocol_is_fixed_to_16_captured_trials_and_two_three_arm_tiers():
    runner = module(RUNNER_PATH, "tiered_runner_contract")
    assert runner.SYSTEMS == ("c60", "pdo")
    assert runner.CAPTURE_TRIALS == 16
    assert runner.MAXITER == 400
    assert runner.STAGES == (("loose", 0.05), ("refine", 0.01))
    assert [(arm.arm_id, arm.optimizer, arm.safe_history_limit) for arm in runner.ARMS] == [
        ("scipy-lbfgsb", "scipy-lbfgsb", None),
        ("safe-lbfgs-total", "safe-lbfgs-total", 10),
        ("ase-fire2", "ase-fire2", None),
    ]


def test_capture_config_is_production_config_plus_only_capture_controls(tmp_path):
    runner = module(RUNNER_PATH, "tiered_runner_capture_config")
    production = runner._production_module()
    for system in runner.SYSTEMS:
        case_dir = tmp_path / system
        baseline = asdict(production.build_config(system, case_dir))
        captured = asdict(runner.build_capture_config(system, case_dir))
        changed = {
            key: (baseline[key], captured[key])
            for key in baseline
            if baseline[key] != captured[key]
        }
        assert changed == {
            "max_trials": (200, 16),
            "write_relaxation_trajectories": (False, True),
            "relaxation_trajectory_dir": (
                None,
                str(case_dir / "trajectories"),
            ),
        }
        assert captured["proposal_optimizer"] == "safe-lbfgs-total"
        assert captured["rng_seed"] == 42
        assert captured["proposal_pool_size"] == 1
        assert captured["proposal_duplicate_rescue_optimizer"] is None


def test_extracts_exact_first_true_quench_frame_per_trial_and_reapplies_fixed_mask(
    tmp_path,
):
    runner = module(RUNNER_PATH, "tiered_runner_corpus")
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    template = State(
        numbers=np.array([46, 8]),
        positions=np.zeros((2, 3)),
        cell=np.diag([5.0, 5.0, 10.0]),
        pbc=(True, True, False),
        fixed_mask=np.array([True, False]),
    )
    for trial in range(1, 17):
        first = Atoms(
            numbers=[46, 8],
            positions=[[trial, 0, 0], [0, 0, 1]],
            cell=template.cell,
            pbc=template.pbc,
        )
        second = first.copy()
        second.positions += 9.0
        write(
            trajectory_dir
            / f"trial{trial:04d}_proposal001_true_quench.xyz",
            [first, second],
        )
    write(
        trajectory_dir / "trial0001_proposal001_step001_proposal_relax.xyz",
        Atoms("PdO", positions=np.zeros((2, 3))),
    )

    tasks, manifest = runner.extract_raw_landing_corpus(
        "pdo", trajectory_dir, template
    )
    assert len(tasks) == manifest["task_count"] == 16
    assert [task.trial_index for task in tasks] == list(range(1, 17))
    assert tasks[0].state.positions[0, 0] == pytest.approx(1.0)
    assert np.array_equal(tasks[0].state.fixed_mask, template.fixed_mask)
    assert all(task.source_frame_index == 0 for task in tasks)
    assert manifest["capture_policy_conditioned"] is True
    assert all(
        ".partial" not in entry["source_trajectory_path"]
        and entry["source_trajectory_path"].startswith("stage_a/pdo/trajectories/")
        for entry in manifest["entries"]
    )

    write(
        trajectory_dir / "trial0001_proposal002_true_quench.xyz",
        Atoms(
            numbers=[46, 8],
            positions=[[1, 0, 0], [0, 0, 1]],
            cell=template.cell,
            pbc=template.pbc,
        ),
    )
    with pytest.raises(ValueError, match="one true-quench trajectory"):
        runner.extract_raw_landing_corpus("pdo", trajectory_dir, template)


def _task(runner, *, system: str = "c60", index: int = 0):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[1.0, 0, 0], [0, 1.0, 0]]),
    )
    return runner.LandingTask(
        system=system,
        task_index=index,
        trial_index=index + 1,
        proposal_index=1,
        source_trajectory_path=Path(f"{system}-{index}.xyz"),
        source_trajectory_sha256="a" * 64,
        source_frame_index=0,
        initial_positions_sha256=runner.position_sha256(state.positions),
        state=state,
    )


@pytest.mark.parametrize("stage", ["loose", "refine"])
@pytest.mark.parametrize("arm_index", [0, 1, 2])
def test_true_quench_rows_use_one_real_pes_contract_and_close_counter(
    monkeypatch, stage, arm_index
):
    runner = module(
        RUNNER_PATH, f"tiered_runner_execute_{stage}_{arm_index}"
    )
    calls = []

    class FakeRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, **kwargs):
            calls.append((self.optimizer, dict(kwargs)))
            energy, gradient = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=energy,
                gradient_norm=0.005,
                n_iter=3,
                outcome_class=RelaxOutcomeClass.CONVERGED_PRODUCTIVE,
                telemetry=RelaxTelemetry(
                    backend=self.optimizer,
                    evaluator_calls=1,
                    backend_evaluations=1,
                    converged=True,
                    termination_reason="converged",
                ),
            )

    monkeypatch.setattr(runner, "Relaxer", FakeRelaxer)

    class Quadratic:
        def evaluate_flat(self, flat, template):
            flat = np.asarray(flat, dtype=float)
            return 0.5 * float(flat @ flat), flat.copy()

    counter = EvalCounter(Quadratic())
    arm = runner.ARMS[arm_index]
    row = runner.execute_true_quench(
        _task(runner),
        stage,
        dict(runner.STAGES)[stage],
        arm,
        counter,
    )
    optimizer, kwargs = calls[0]
    assert optimizer == arm.optimizer
    assert kwargs["fmax"] == dict(runner.STAGES)[stage]
    assert kwargs["maxiter"] == 400
    assert kwargs["coordinate_trust_radius"] is None
    if arm.optimizer == "safe-lbfgs-total":
        assert kwargs["_safe_lbfgs_history_limit"] == 10
    else:
        assert "_safe_lbfgs_history_limit" not in kwargs
    assert row["objective"] == "true_mace_pes_no_bias_no_softening"
    assert row["evaluator_calls"] == row["telemetry"]["evaluator_calls"] == 1
    assert row["purpose_count_delta"]["landing_true_quench"] == 1
    assert row["purpose_count_delta"]["unattributed"] == 0


def _preflight(runner):
    tasks = {
        system: tuple(_task(runner, system=system, index=index) for index in range(16))
        for system in runner.SYSTEMS
    }
    return runner.Preflight(
        metadata={
            "schema_version": 1,
            "execution_commit": "e" * 40,
            "safe_lbfgs_default_history_limit": 10,
        },
        production_states={
            system: tasks[system][0].state for system in runner.SYSTEMS
        },
    )


def test_run_preflights_before_calculator_and_writes_complete_tier_matrix(
    tmp_path, monkeypatch
):
    runner = module(RUNNER_PATH, "tiered_runner_run")
    events = []
    checked = _preflight(runner)

    def preflight_fn(**kwargs):
        events.append("preflight")
        return checked

    def calculator_factory():
        events.append("calculator")
        return object()

    def capture(system, state, config, counter):
        events.append(f"capture-{system}")
        tasks = tuple(
            _task(runner, system=system, index=index) for index in range(16)
        )
        return tasks, {
            "system": system,
            "task_count": 16,
            "capture_policy_conditioned": True,
        }, {"n_trials": 16, "force_evaluations": 0}, EvaluationCounts.zero()

    def execute(task, stage, fmax, arm, counter):
        return {
            "system": task.system,
            "task_index": task.task_index,
            "trial_index": task.trial_index,
            "stage": stage,
            "arm_id": arm.arm_id,
            "optimizer": arm.optimizer,
            "safe_history_limit": arm.safe_history_limit,
            "fmax_eV_per_A": fmax,
            "maxiter": 400,
            "objective": "true_mace_pes_no_bias_no_softening",
            "initial": {
                "energy_eV": 1.0,
                "max_active_force_eV_per_A": 1.0,
                "positions_sha256": task.initial_positions_sha256,
                "state": runner.state_payload(task.state),
            },
            "final": {
                "energy_eV": 0.0,
                "max_active_force_eV_per_A": 0.0,
                "positions_sha256": task.initial_positions_sha256,
                "state": runner.state_payload(task.state),
            },
            "termination_reason": "converged",
            "outcome_class": "converged_productive",
            "n_iter": 1,
            "telemetry": {"evaluator_calls": 1},
            "evaluator_calls": 1,
            "purpose_count_delta": {
                purpose.value: int(purpose is EvaluationPurpose.LANDING_TRUE_QUENCH)
                for purpose in EvaluationPurpose
            },
            "wall_time_s": 0.1,
        }

    monkeypatch.setattr(runner, "capture_stage_a", capture)
    monkeypatch.setattr(runner, "execute_true_quench", execute)
    output = tmp_path / "output"
    summary = runner.run(
        output_dir=output,
        expected_git_commit="e" * 40,
        preflight_fn=preflight_fn,
        calculator_factory=calculator_factory,
    )
    assert events[0] == "preflight"
    assert summary["stage_a_task_count"] == 32
    assert summary["stage_b_row_count"] == 96
    assert summary["stage_c_row_count"] == 96
    assert ".partial" not in json.dumps(summary["stage_a"])
    rows = json.loads((output / "rows.json").read_text())
    assert len(rows) == 192
    for system in runner.SYSTEMS:
        for task_index in range(16):
            refine = [
                row
                for row in rows
                if row["system"] == system
                and row["task_index"] == task_index
                and row["stage"] == "refine"
            ]
            loose_scipy = next(
                row
                for row in rows
                if row["system"] == system
                and row["task_index"] == task_index
                and row["stage"] == "loose"
                and row["arm_id"] == "scipy-lbfgsb"
            )
            assert {
                row["initial"]["positions_sha256"] for row in refine
            } == {loose_scipy["final"]["positions_sha256"]}
    assert output.is_dir()
    assert not output.with_name("output.partial").exists()


def test_preflight_only_constructs_no_calculator_or_output(tmp_path):
    runner = module(RUNNER_PATH, "tiered_runner_preflight_only")
    output = tmp_path / "output"
    checked = _preflight(runner)
    result = runner.run(
        output_dir=output,
        expected_git_commit="e" * 40,
        preflight_only=True,
        preflight_fn=lambda **_: checked,
        calculator_factory=lambda: pytest.fail("no calculator in preflight"),
    )
    assert result == checked.metadata
    assert not output.exists()
    assert not output.with_name("output.partial").exists()


def test_analyzer_reports_certificates_cost_same_basin_and_claim_boundary(
    tmp_path, monkeypatch
):
    runner = module(RUNNER_PATH, "tiered_runner_analysis_fixture")
    analyzer = module(ANALYZER_PATH, "tiered_analyzer")
    checked = _preflight(runner)
    def capture(system, state, config, counter):
        tasks = tuple(
            _task(runner, system=system, index=index) for index in range(16)
        )
        return (
            tasks,
            {
                "system": system,
                "task_count": 16,
                "capture_policy_conditioned": True,
                "entries": [
                    {
                        "system": system,
                        "task_index": task.task_index,
                        "trial_index": task.trial_index,
                        "proposal_index": task.proposal_index,
                        "source_trajectory_path": str(
                            task.source_trajectory_path
                        ),
                        "source_trajectory_sha256": (
                            task.source_trajectory_sha256
                        ),
                        "source_frame_index": 0,
                        "initial_positions_sha256": (
                            task.initial_positions_sha256
                        ),
                        "state": runner.state_payload(task.state),
                    }
                    for task in tasks
                ],
            },
            {"n_trials": 16, "force_evaluations": 0},
            EvaluationCounts.zero(),
        )

    monkeypatch.setattr(runner, "capture_stage_a", capture)

    def row(task, stage, fmax, arm, counter):
        final_state = task.state
        return {
            "system": task.system,
            "task_index": task.task_index,
            "trial_index": task.trial_index,
            "stage": stage,
            "arm_id": arm.arm_id,
            "optimizer": arm.optimizer,
            "safe_history_limit": arm.safe_history_limit,
            "fmax_eV_per_A": fmax,
            "maxiter": 400,
            "objective": "true_mace_pes_no_bias_no_softening",
            "initial": {"energy_eV": 1.0, "max_active_force_eV_per_A": 1.0, "positions_sha256": task.initial_positions_sha256, "state": runner.state_payload(task.state)},
            "final": {"energy_eV": 0.0, "max_active_force_eV_per_A": 0.5 * fmax, "positions_sha256": runner.position_sha256(final_state.positions), "state": runner.state_payload(final_state)},
            "termination_reason": "converged",
            "outcome_class": "converged_productive",
            "n_iter": 1,
            "telemetry": {"evaluator_calls": 2},
            "evaluator_calls": 2,
            "purpose_count_delta": {purpose.value: 2 if purpose is EvaluationPurpose.LANDING_TRUE_QUENCH else 0 for purpose in EvaluationPurpose},
            "wall_time_s": 0.1,
        }

    monkeypatch.setattr(runner, "execute_true_quench", row)
    raw = tmp_path / "raw"
    runner.run(
        output_dir=raw,
        expected_git_commit="e" * 40,
        preflight_fn=lambda **_: checked,
        calculator_factory=lambda: object(),
    )
    evidence = analyzer.analyze(raw)
    assert evidence["ledger"] == {
        "stage_a_tasks": 32,
        "stage_b_rows": 96,
        "stage_c_rows": 96,
        "purpose_closure_validated": True,
    }
    assert evidence["by_stage_system_arm"]["loose"]["c60"]["scipy-lbfgsb"][
        "certificate_count"
    ] == 16
    assert evidence["paired"]["loose"]["c60"]["safe-lbfgs-total"][
        "same_basin_count"
    ] == 16
    finite_pair = evidence["paired"]["loose"]["c60"]["safe-lbfgs-total"][
        "pairs"
    ][0]
    assert finite_pair["terminal_rmsd_status"] == "finite"
    assert finite_pair["terminal_rmsd_A"] == pytest.approx(0.0)
    assert "capture policy" in evidence["claim_boundary"]
    assert "SciPy loose endpoint" in evidence["claim_boundary"]

    rows_path = raw / "rows.json"
    rows = json.loads(rows_path.read_text())
    rows[0]["initial"]["positions_sha256"] = "0" * 64
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="position hash"):
        analyzer.analyze(raw)


def test_distance_signature_screened_rmsd_is_json_null_and_not_same_basin(
    monkeypatch,
):
    runner = module(RUNNER_PATH, "tiered_runner_nonfinite_rmsd_fixture")
    analyzer = module(ANALYZER_PATH, "tiered_analyzer_nonfinite_rmsd")
    state = _task(runner).state
    row = {
        "final": {
            "energy_eV": -1.0,
            "state": runner.state_payload(state),
        }
    }
    monkeypatch.setattr(
        analyzer.MinimaArchive,
        "_rmsd",
        staticmethod(lambda first, second: float("inf")),
    )

    same, energy_delta, rmsd, *status = analyzer._same_basin(
        row,
        row,
        energy_tol=1.0e-3,
        rmsd_tol=0.15,
    )
    pair = {
        "same_basin_current_archive_semantics": same,
        "absolute_terminal_energy_delta_eV": energy_delta,
        "terminal_rmsd_A": rmsd,
        "terminal_rmsd_status": status[0] if status else None,
    }

    json.dumps(pair, allow_nan=False)
    assert pair == {
        "same_basin_current_archive_semantics": False,
        "absolute_terminal_energy_delta_eV": 0.0,
        "terminal_rmsd_A": None,
        "terminal_rmsd_status": "distance_signature_screened_or_nonfinite",
    }


def test_cli_entrypoints_exist_and_parse_help():
    runner = module(RUNNER_PATH, "tiered_runner_cli")
    analyzer = module(ANALYZER_PATH, "tiered_analyzer_cli")
    with pytest.raises(SystemExit):
        runner._parse_args(["--help"])
    with pytest.raises(SystemExit):
        analyzer._parse_args(["--help"])
