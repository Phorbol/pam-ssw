from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pamssw.bias import GaussianBiasTerm
from pamssw.state import State


_TRACE_RECORDER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-proposal-energy-traces"
    / "trace_recorder.py"
)


def _trace_recorder_module():
    spec = importlib.util.spec_from_file_location("proposal_energy_trace_recorder", _TRACE_RECORDER_PATH)
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
