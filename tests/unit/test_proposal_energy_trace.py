from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from pamssw.accounting import EvaluationCounts
from pamssw.bias import GaussianBiasTerm
from pamssw.result import RelaxResult, RelaxTelemetry
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


_TRACE_RECORDER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-proposal-energy-traces"
    / "trace_recorder.py"
)

_TRACE_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-proposal-energy-traces"
    / "run_gpu_traces.py"
)


def _trace_recorder_module():
    spec = importlib.util.spec_from_file_location("proposal_energy_trace_recorder", _TRACE_RECORDER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace_runner_module():
    spec = importlib.util.spec_from_file_location("proposal_energy_trace_runner", _TRACE_RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CountingAnalyticCalculator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self.calls += 1
        del flat_positions, template
        return 2.5, np.array([50.0, 0.0, 0.0, 3.0, 4.0, 0.0])


class _QuadraticCountingCalculator:
    """Finite analytic PES used to prove tracing does not add evaluations."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self.calls += 1
        target = np.zeros_like(flat_positions, dtype=float)
        target[3] = 0.35
        displacement = np.asarray(flat_positions, dtype=float) - target
        del template
        return float(0.5 * np.dot(displacement, displacement)), displacement


def _state() -> State:
    return State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.2, 0.0, 0.0], [1.0, 0.2, 0.0]]),
        fixed_mask=np.array([True, False]),
    )


def _bias() -> GaussianBiasTerm:
    return GaussianBiasTerm(
        center=np.zeros(6),
        direction=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        sigma=1.0,
        weight=0.5,
    )


def test_position_hash_is_deterministic_for_state_and_position_array():
    trace_recorder = _trace_recorder_module()
    state = _state()

    from_state = trace_recorder.position_hash(state)
    from_array = trace_recorder.position_hash(state.positions.copy())

    assert from_state == from_array
    assert len(from_state) == 64
    assert trace_recorder.position_hash(state.positions + 0.01) != from_state


def test_recorder_records_existing_component_evaluation_without_extra_calculator_calls():
    trace_recorder = _trace_recorder_module()
    state = _state()
    calculator = _CountingAnalyticCalculator()
    recorder = trace_recorder.RecordingProposalPotential(calculator, biases=[_bias()])

    evaluation = recorder.evaluate_parts(state.flatten_positions(), state)

    assert calculator.calls == 1
    assert len(recorder.records) == 1
    record = recorder.records[0]
    assert record["evaluation_index"] == 1
    assert record["positions_sha256"] == trace_recorder.position_hash(state)
    assert record["true_energy_eV"] == pytest.approx(2.5)
    assert record["bias_energy_eV"] == pytest.approx(evaluation.bias_energy)
    assert record["softening_energy_eV"] == pytest.approx(0.0)
    assert record["total_energy_eV"] == pytest.approx(
        record["true_energy_eV"] + record["bias_energy_eV"] + record["softening_energy_eV"]
    )
    expected_active_force = float(
        np.linalg.norm(evaluation.total_gradient.reshape(-1, 3)[state.movable_mask], axis=1).max()
    )
    assert record["active_max_total_force_eV_per_A"] == pytest.approx(expected_active_force)
    assert record["active_max_total_force_eV_per_A"] < 10.0

    recorder.evaluate_parts(state.flatten_positions() + 0.1, state)

    assert calculator.calls == 2
    assert [record["evaluation_index"] for record in recorder.records] == [1, 2]


def test_mark_accepted_state_evaluations_marks_only_coordinate_hash_matches_and_is_defensive():
    trace_recorder = _trace_recorder_module()
    accepted = _state()
    accepted_by_hash = accepted.positions + 0.1
    rejected = accepted.positions + 0.2
    records = [
        {"positions_sha256": trace_recorder.position_hash(accepted), "nested": {"values": [1]}},
        {"positions_sha256": trace_recorder.position_hash(accepted_by_hash)},
        {"positions_sha256": trace_recorder.position_hash(rejected)},
    ]

    marked = trace_recorder.mark_accepted_state_evaluations(
        records,
        [accepted, trace_recorder.position_hash(accepted_by_hash)],
    )

    assert [record["accepted_state"] for record in marked] == [True, True, False]
    assert all("accepted_state" not in record for record in records)
    assert marked[0] is not records[0]
    marked[0]["nested"]["values"].append(2)
    assert records[0]["nested"]["values"] == [1]


def test_trace_runner_current_commit_resolves_the_worktree_head():
    trace_runner = _trace_runner_module()
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_TRACE_RUNNER_PATH.parents[2],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert trace_runner._current_commit() == expected


def test_trace_runner_binds_the_reviewed_source_and_cap400_reference_hashes():
    trace_runner = _trace_runner_module()

    assert trace_runner.EXPECTED_SOURCE_SUMMARY_SHA256 == (
        "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
    )
    assert trace_runner.EXPECTED_REFERENCE_SUMMARY_SHA256 == (
        "a11cd8ee1a1dae9cc71cac0038008149b1c0f0cb67ae72ceba8afc821cbdf370"
    )


def test_reference_match_uses_measured_mic_float32_equivalence_and_rejects_excess_drift():
    trace_runner = _trace_runner_module()
    reference_positions = np.array([[0.2, 0.0, 0.0]])
    state = State(
        numbers=np.array([1]),
        positions=reference_positions,
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=(True, False, False),
        fixed_mask=np.array([False]),
    )
    reference = {
        "force_evaluations": 75,
        "certificate_satisfied": True,
        "termination_reason": "converged",
        "final_biased_energy_eV": -464.6608064065233,
        "final_positions": reference_positions.tolist(),
    }

    def replay_with(*, energy_delta=0.0, position_delta=0.0):
        result = RelaxResult(
            state=State(
                numbers=state.numbers.copy(),
                positions=state.positions + position_delta,
                cell=state.cell.copy(),
                pbc=state.pbc,
                fixed_mask=state.fixed_mask.copy(),
            ),
            energy=reference["final_biased_energy_eV"] + energy_delta,
            gradient_norm=0.01,
            n_iter=1,
            telemetry=RelaxTelemetry(termination_reason="converged"),
        )
        return trace_runner.TraceReplayResult(
            result=result,
            evaluation_counts=EvaluationCounts.unattributed(75),
            wall_time_s=0.0,
            trace_records=[],
            accepted_callback_hashes=(),
            certificate_satisfied=True,
        )

    assert trace_runner.ENERGY_ABSOLUTE_TOLERANCE_EV == pytest.approx(5.0e-4)
    assert trace_runner.POSITION_MAX_MIC_DISPLACEMENT_A == pytest.approx(2.0e-3)
    trace_runner._require_reference_match(
        system="c60",
        task_id="c60-seed-42-bias-1",
        backend="ase-fire",
        replay=replay_with(energy_delta=4.9e-4, position_delta=np.array([[10.0, 0.0, 0.0]])),
        reference=reference,
    )

    with pytest.raises(
        trace_runner.ReplayMismatchError,
        match=r"final biased energy.*actual=.*reference=.*delta=",
    ):
        trace_runner._require_reference_match(
            system="c60",
            task_id="c60-seed-42-bias-1",
            backend="ase-fire",
            replay=replay_with(energy_delta=5.0001e-4),
            reference=reference,
        )
    with pytest.raises(
        trace_runner.ReplayMismatchError,
        match=r"final positions.*actual=.*reference=.*delta=",
    ):
        trace_runner._require_reference_match(
            system="c60",
            task_id="c60-seed-42-bias-1",
            backend="ase-fire",
            replay=replay_with(position_delta=2.0001e-3),
            reference=reference,
        )


def test_model_provenance_hashes_local_files_before_any_calculator_warmup(tmp_path):
    trace_runner = _trace_runner_module()
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "structure.xyz"
    model_path.write_bytes(b"trusted model bytes")
    input_path.write_bytes(b"trusted structure bytes")
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    calculator_factory_calls = 0

    def forbidden_calculator_factory():
        nonlocal calculator_factory_calls
        calculator_factory_calls += 1
        raise AssertionError("calculator construction must follow provenance validation")

    source = {
        "MODEL": model_path,
        "MODEL_SHA256": model_sha256,
        "SYSTEMS": {"fixture": {"input": input_path, "sha256": input_sha256}},
        "_calculator": forbidden_calculator_factory,
    }

    provenance = trace_runner._verified_model_provenance(source)

    assert provenance["model"] == str(model_path)
    assert provenance["model_sha256"] == model_sha256
    assert provenance["source_system_inputs"]["fixture"]["input"] == str(input_path)
    assert provenance["source_system_inputs"]["fixture"]["sha256"] == input_sha256
    assert calculator_factory_calls == 0

    model_path.write_bytes(b"tampered model bytes")

    with pytest.raises(ValueError, match="model SHA256 mismatch"):
        trace_runner._verified_model_provenance(source)

    assert calculator_factory_calls == 0


def test_ledger_publish_failure_leaves_no_final_or_staging_directory(tmp_path):
    trace_runner = _trace_runner_module()
    output_dir = tmp_path / "ledger"
    write_attempts: list[str] = []

    def fail_on_summary(path, payload):
        del payload
        write_attempts.append(path.name)
        if path.name == "summary.json":
            raise OSError("injected summary write failure")
        trace_runner._write_json_atomic(path, {"written": path.name})

    with pytest.raises(OSError, match="injected summary write failure"):
        trace_runner._publish_ledger_atomically(
            output_dir,
            [{"system": "c60"}, {"system": "pdo"}],
            {"schema_version": 1},
            write_json=fail_on_summary,
        )

    assert write_attempts == ["c60.json", "pdo.json", "summary.json"]
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".ledger.staging-*"))


def test_ledger_publish_refuses_existing_output_without_touching_it(tmp_path):
    trace_runner = _trace_runner_module()
    output_dir = tmp_path / "ledger"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        trace_runner._publish_ledger_atomically(
            output_dir,
            [{"system": "c60"}],
            {"schema_version": 1},
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("optimizer", ["ase-fire", "safe-lbfgs-total"])
def test_replay_task_with_trace_closes_all_ledgers_without_trace_only_pes_calls(optimizer):
    trace_recorder = _trace_recorder_module()
    trace_runner = _trace_runner_module()
    task = ProposalRelaxationTask(
        initial_state=_state(),
        biases=(_bias(),),
        softening=None,
        fmax=1.0e-3,
        maxiter=80,
        coordinate_trust_radius=None,
    )
    calculator = _QuadraticCountingCalculator()

    replay = trace_runner.replay_task_with_trace(task, calculator, optimizer)

    records = replay.trace_records
    assert records
    assert len(records) == replay.evaluation_counts.total
    assert len(records) == replay.result.telemetry.evaluator_calls
    assert calculator.calls == len(records)
    assert replay.evaluation_counts.as_dict()["biased_proposal_relax"] == len(records)
    assert replay.evaluation_counts.as_dict()["unattributed"] == 0
    assert replay.accepted_callback_hashes
    assert all("accepted_state" in record for record in records)
    assert any(record["accepted_state"] for record in records)
    assert np.all(
        np.isfinite(
            [
                value
                for record in records
                for value in (
                    record["true_energy_eV"],
                    record["bias_energy_eV"],
                    record["softening_energy_eV"],
                    record["total_energy_eV"],
                    record["active_max_total_force_eV_per_A"],
                )
            ]
        )
    )
    assert records[-1]["positions_sha256"] == trace_recorder.position_hash(replay.result.state)
    assert records[-1]["total_energy_eV"] == pytest.approx(replay.result.energy)
