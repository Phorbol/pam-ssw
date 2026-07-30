from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.bias import GaussianBiasTerm, QuadraticBiasTerm
from pamssw.calculators import AnalyticCalculator
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs" / "20260730-uphill-bias-shape-u3"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _task() -> ProposalRelaxationTask:
    bias = GaussianBiasTerm(
        center=np.zeros(3),
        direction=np.array([1.0, 0.0, 0.0]),
        sigma=0.5,
        weight=0.5,
    )
    return ProposalRelaxationTask(
        initial_state=State(
            numbers=np.array([1]),
            positions=np.array([[0.5, 0.0, 0.0]]),
        ),
        biases=(bias,),
        softening=None,
        fmax=0.05,
        maxiter=20,
        coordinate_trust_radius=1.0,
    )


class _QuadraticPES:
    def energy_gradient(self, flat_positions, state):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(flat @ flat), flat.copy()


def test_shape_arms_change_only_newest_bias_function():
    runner = _load("uphill_shape_runner", "run_shape_ablation.py")
    task = _task()

    gaussian = runner.biases_for_arm(task, "gaussian")
    quadratic = runner.biases_for_arm(task, "quadratic")

    assert isinstance(gaussian[-1], GaussianBiasTerm)
    assert isinstance(quadratic[-1], QuadraticBiasTerm)
    assert quadratic[-1].sigma == gaussian[-1].sigma
    assert quadratic[-1].weight == gaussian[-1].weight
    np.testing.assert_allclose(
        quadratic[-1].center,
        gaussian[-1].center,
    )
    np.testing.assert_allclose(
        quadratic[-1].direction,
        gaussian[-1].direction,
    )
    assert quadratic[-1].directional_curvature_shift() == (
        gaussian[-1].directional_curvature_shift()
    )


def test_shape_replay_closes_paired_matrix_without_unattributed_calls():
    runner = _load("uphill_shape_runner_replay", "run_shape_ablation.py")
    analyzer = _load("uphill_shape_analyzer", "analyze_shape_ablation.py")

    rows = runner.run_shape_arms(
        _task(),
        system="analytic",
        task_id="task-0",
        calculator_factory=lambda: AnalyticCalculator(_QuadraticPES()),
        optimizer="ase-fire",
    )
    summary = analyzer.analyze_rows(rows)

    assert [row["arm_id"] for row in rows] == ["gaussian", "quadratic"]
    assert all(
        row["purpose_counts"][EvaluationPurpose.UNATTRIBUTED.value] == 0
        for row in rows
    )
    assert summary["systems"]["analytic"]["task_count"] == 1
    assert "quadratic_minus_gaussian" in summary["systems"]["analytic"]


def test_gpu_shape_protocol_is_bounded_and_cross_system():
    screen = _load("uphill_shape_gpu", "run_gpu_shape.py")

    assert screen.SYSTEM_SPECS == {
        "c60": (
            (2002, 3),
            (2003, 5),
            (2004, 8),
            (2006, 3),
            (2007, 5),
            (2008, 8),
        ),
        "pdo": (
            (2001, 1),
            (2005, 1),
        ),
    }
