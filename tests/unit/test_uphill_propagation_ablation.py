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
RUN_ROOT = ROOT / "runs" / "20260730-uphill-propagation-u0-u1"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, RUN_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _task() -> ProposalRelaxationTask:
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        fixed_mask=np.array([False, True]),
    )
    prefix = GaussianBiasTerm(
        center=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sigma=0.2,
        weight=0.1,
    )
    newest = GaussianBiasTerm(
        center=np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sigma=0.3,
        weight=0.2,
    )
    return ProposalRelaxationTask(
        initial_state=state,
        biases=(prefix, newest),
        softening=None,
        fmax=0.05,
        maxiter=10,
        coordinate_trust_radius=1.0,
    )


def test_arm_construction_changes_only_the_last_gaussian():
    runner = _load("uphill_u0_u1_runner", "run_ablation.py")
    source = _task()

    current = runner.build_arm_task(
        source,
        arm_id="current_full",
    )
    matched = runner.build_arm_task(
        source,
        arm_id="curvature_matched_no_feedback",
        base_sigma=0.25,
        inner_curvature=0.2,
        target_negative_curvature=0.05,
    )
    fixed = runner.build_arm_task(
        source,
        arm_id="fixed_calibrated",
        fixed_sigma=0.4,
        fixed_weight=0.6,
    )

    assert runner.ARM_IDS == (
        "current_full",
        "curvature_matched_no_feedback",
        "fixed_calibrated",
    )
    assert current is source
    assert matched.biases[-1].sigma == pytest.approx(0.25)
    assert matched.biases[-1].weight == pytest.approx(
        0.25**2 * (0.2 + 0.05)
    )
    assert fixed.biases[-1].sigma == pytest.approx(0.4)
    assert fixed.biases[-1].weight == pytest.approx(0.6)
    for candidate in (matched, fixed):
        assert candidate.biases[0].sigma == source.biases[0].sigma
        assert candidate.biases[0].weight == source.biases[0].weight
        np.testing.assert_allclose(
            candidate.biases[0].center,
            source.biases[0].center,
        )


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        (
            {
                "arm_id": "fixed_calibrated",
                "fixed_sigma": np.nan,
                "fixed_weight": 0.1,
            },
            "sigma",
        ),
        (
            {
                "arm_id": "curvature_matched_no_feedback",
                "base_sigma": 0.2,
                "inner_curvature": np.inf,
                "target_negative_curvature": 0.05,
            },
            "curvature",
        ),
        ({"arm_id": "unknown"}, "arm"),
    ],
)
def test_arm_construction_fails_closed(kwargs, fragment):
    runner = _load("uphill_u0_u1_runner_invalid", "run_ablation.py")
    with pytest.raises((TypeError, ValueError), match=fragment):
        runner.build_arm_task(_task(), **kwargs)


def test_fixed_calibration_uses_only_finite_source_tasks_and_component_medians():
    runner = _load("uphill_u0_u1_runner_calibration", "run_ablation.py")
    records = [
        {
            "system": "c60",
            "task_id": "c60-cal-0",
            "sigma": 0.8,
            "weight": 0.2,
            "source_task_sha256": "a" * 64,
        },
        {
            "system": "c60",
            "task_id": "c60-cal-1",
            "sigma": 1.2,
            "weight": 0.6,
            "source_task_sha256": "b" * 64,
        },
    ]

    calibration = runner.calibrate_fixed_parameters(records)

    assert calibration["c60"]["sigma"] == pytest.approx(1.0)
    assert calibration["c60"]["weight"] == pytest.approx(0.4)
    assert calibration["c60"]["task_ids"] == [
        "c60-cal-0",
        "c60-cal-1",
    ]
    with pytest.raises(ValueError, match="at least two"):
        runner.calibrate_fixed_parameters(records[:1])


def _row(task_id: str, arm_id: str) -> dict:
    return {
        "schema_version": 1,
        "system": "c60",
        "task_id": task_id,
        "arm_id": arm_id,
        "source_task_sha256": "a" * 64,
        "certificate_satisfied": True,
        "biased_proposal_relax_force_evaluations": 10,
        "wall_time_s": 0.1,
        "initial": {
            "true_energy": 1.0,
            "bias_energy": 0.5,
            "softening_energy": 0.0,
            "total_energy": 1.5,
        },
        "final": {
            "true_energy": 1.2,
            "bias_energy": 0.1,
            "softening_energy": 0.0,
            "total_energy": 1.3,
        },
        "direction_progress": 0.3,
        "orthogonal_displacement_norm": 0.2,
        "endpoint_position_sha256": "b" * 64,
        "observer_only_force_evaluations": 0,
    }


def test_analyzer_requires_one_closed_three_arm_matrix_per_source_task():
    runner = _load("uphill_u0_u1_runner_rows", "run_ablation.py")
    analyzer = _load("uphill_u0_u1_analyzer", "analyze_results.py")
    rows = [_row("eval-0", arm) for arm in runner.ARM_IDS]

    summary = analyzer.analyze_rows(rows)

    assert summary["row_count"] == 3
    assert summary["systems"]["c60"]["task_count"] == 1
    assert set(summary["systems"]["c60"]["arms"]) == set(runner.ARM_IDS)
    with pytest.raises(ValueError, match="three-arm matrix"):
        analyzer.analyze_rows(rows[:-1])


class _Quadratic:
    def energy_gradient(self, flat_positions, state):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(flat @ flat), flat.copy()


def test_run_task_arms_writes_one_exactly_accounted_row_per_arm():
    runner = _load("uphill_u0_u1_runner_matrix", "run_ablation.py")
    analyzer = _load("uphill_u0_u1_analyzer_matrix", "analyze_results.py")
    source = _task()

    rows = runner.run_task_arms(
        source,
        system="analytic",
        task_id="eval-0",
        calculator_factory=lambda: AnalyticCalculator(_Quadratic()),
        optimizer="ase-fire",
        base_sigma=0.25,
        inner_curvature=0.2,
        target_negative_curvature=0.05,
        fixed_sigma=0.4,
        fixed_weight=0.6,
    )

    assert [row["arm_id"] for row in rows] == list(runner.ARM_IDS)
    assert len({row["source_task_sha256"] for row in rows}) == 1
    assert all(
        row["observer_only_force_evaluations"] == 0 for row in rows
    )
    assert all(
        isinstance(
            row["biased_proposal_relax_force_evaluations"],
            int,
        )
        for row in rows
    )
    summary = analyzer.analyze_rows(rows)
    assert summary["row_count"] == 3
    assert summary["systems"]["analytic"]["task_count"] == 1


def test_base_sigma_and_inner_curvature_are_recomputed_from_frozen_prefix():
    runner = _load("uphill_u0_u1_runner_characterize", "run_ablation.py")
    source = _task()

    base_sigma = runner.base_sigma_from_task(
        source,
        target_step_rms=0.15,
        max_step_rms=0.35,
        step_rms_scope="all_atoms",
        active_threshold=1.0e-4,
    )
    curvature, counts = runner.measure_inner_curvature(
        source,
        AnalyticCalculator(_Quadratic()),
        hvp_epsilon=1.0e-4,
    )

    assert base_sigma == pytest.approx(0.15)
    expected_prefix_shift = float(
        np.dot(
            source.biases[0].direction,
            source.biases[0].hvp_contribution(
                source.biases[-1].direction,
                source.biases[-1].center,
            ),
        )
    )
    assert curvature == pytest.approx(
        1.0 + expected_prefix_shift,
        abs=1.0e-5,
    )
    assert counts.total == 2
    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 2
