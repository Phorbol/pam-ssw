from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pamssw.accounting import EvaluationPurpose
from pamssw.bias import GaussianBiasTerm
from pamssw.calculators import AnalyticCalculator
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "20260730-uphill-feedback-factorial-u2"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _task() -> ProposalRelaxationTask:
    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.5, 0.0, 0.0]]),
    )
    bias = GaussianBiasTerm(
        center=np.zeros(3),
        direction=np.array([1.0, 0.0, 0.0]),
        sigma=0.5,
        weight=0.5,
    )
    return ProposalRelaxationTask(
        initial_state=state,
        biases=(bias,),
        softening=None,
        fmax=0.05,
        maxiter=10,
        coordinate_trust_radius=1.0,
    )


def test_factorial_arms_separate_width_and_effective_curvature():
    runner = _load("uphill_u2_runner", "run_factorial.py")
    source = _task()
    expected = {
        "feedback_on_on": (0.5, 0.5),
        "feedback_on_off": (0.5, 1.0),
        "feedback_off_on": (1.0, 2.0),
        "feedback_off_off": (1.0, 4.0),
    }

    for arm_id, (sigma, weight) in expected.items():
        task = runner.build_factorial_task(
            source,
            arm_id=arm_id,
            base_sigma=1.0,
            inner_curvature=3.0,
            target_negative_curvature=1.0,
            bias_weight_min=0.0,
            bias_weight_max=10.0,
        )
        assert task.biases[-1].sigma == pytest.approx(sigma)
        assert task.biases[-1].weight == pytest.approx(weight)


class _Quadratic:
    def energy_gradient(self, flat_positions, state):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(flat @ flat), flat.copy()


def test_factorial_replay_closes_four_arm_matrix_without_unattributed_calls():
    runner = _load("uphill_u2_runner_replay", "run_factorial.py")
    analyzer = _load("uphill_u2_analyzer", "analyze_factorial.py")

    rows = runner.run_factorial_arms(
        _task(),
        system="analytic",
        task_id="task-0",
        calculator_factory=lambda: AnalyticCalculator(_Quadratic()),
        optimizer="ase-fire",
        base_sigma=1.0,
        inner_curvature=3.0,
        target_negative_curvature=1.0,
        bias_weight_min=0.0,
        bias_weight_max=10.0,
    )
    summary = analyzer.analyze_rows(rows)

    assert [row["arm_id"] for row in rows] == list(runner.ARM_IDS)
    assert len(rows) == 4
    assert all(
        row["purpose_counts"][EvaluationPurpose.UNATTRIBUTED.value] == 0
        for row in rows
    )
    assert summary["systems"]["analytic"]["task_count"] == 1
    assert set(
        summary["systems"]["analytic"]["paired_differences_vs_current"]
    ) == set(runner.ARM_IDS[1:])


def test_gpu_factorial_protocol_is_late_prefix_c60_only():
    screen = _load("uphill_u2_gpu", "run_gpu_factorial.py")

    assert screen.SYSTEM == "c60"
    assert screen.EVALUATION_SPECS == (
        (2002, 3),
        (2003, 5),
        (2004, 8),
        (2006, 3),
        (2007, 5),
        (2008, 8),
    )
