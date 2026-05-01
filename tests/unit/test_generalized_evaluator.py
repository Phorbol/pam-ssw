from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pamssw.calculators import EnergyResult
from pamssw.generalized_coordinates import GeneralizedCoordinates
from pamssw.generalized_evaluator import CalculatorGeneralizedEvaluator, verify_stress_gradient
from pamssw.state import State


def state() -> State:
    return State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5]]),
        cell=np.diag([4.0, 4.0, 4.0]),
        pbc=(True, True, True),
    )


@dataclass
class FractionalAndVolumeCalculator:
    k_atom: float = 2.0
    k_vol: float = 0.01
    target_volume: float = 60.0

    def evaluate(self, candidate: State) -> EnergyResult:
        frac = candidate.positions @ np.linalg.inv(candidate.cell)
        energy_atom = 0.5 * self.k_atom * float(np.sum(frac * frac))
        grad_frac = self.k_atom * frac
        grad_cart = grad_frac @ np.linalg.inv(candidate.cell).T
        volume = float(np.linalg.det(candidate.cell))
        energy_vol = 0.5 * self.k_vol * (volume - self.target_volume) ** 2
        stress = self.k_vol * (volume - self.target_volume) * np.eye(3)
        return EnergyResult(energy=energy_atom + energy_vol, gradient=grad_cart, stress=stress)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        result = self.evaluate(template.with_flat_positions(flat_positions))
        return result.energy, result.gradient.reshape(-1)


@dataclass
class NoStressCalculator(FractionalAndVolumeCalculator):
    def evaluate(self, candidate: State) -> EnergyResult:
        result = super().evaluate(candidate)
        return EnergyResult(result.energy, result.gradient, None)


def test_atomic_gradient_converts_cartesian_to_fractional() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "fixed_cell")
    q = gcoord.to_q(initial)
    _, grad = CalculatorGeneralizedEvaluator(FractionalAndVolumeCalculator()).evaluate_q(q, gcoord)
    frac = initial.positions @ np.linalg.inv(initial.cell)
    expected = (2.0 * frac)[initial.movable_mask].reshape(-1)
    np.testing.assert_allclose(grad[: gcoord.atomic_size], expected)


def test_cell_gradient_matches_finite_difference() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "volume_only")
    q = gcoord.to_q(initial)
    stress_eval = CalculatorGeneralizedEvaluator(FractionalAndVolumeCalculator())
    fd_eval = CalculatorGeneralizedEvaluator(FractionalAndVolumeCalculator(), finite_diff_cell_gradient=True)
    _, stress_grad = stress_eval.evaluate_q(q, gcoord)
    _, fd_grad = fd_eval.evaluate_q(q, gcoord)
    np.testing.assert_allclose(stress_grad[gcoord.atomic_size :], fd_grad[gcoord.atomic_size :], rtol=1e-4, atol=1e-5)


def test_missing_stress_fails_closed() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "volume_only")
    with pytest.raises(RuntimeError):
        CalculatorGeneralizedEvaluator(NoStressCalculator(), requires_stress=True).evaluate_q(gcoord.to_q(initial), gcoord)


def test_verify_stress_gradient_reports_match() -> None:
    result = verify_stress_gradient(FractionalAndVolumeCalculator(), state(), cell_dof_mode="volume_only")
    assert result["components_match"] is True


def test_verify_stress_gradient_uses_noise_robust_default_eps_and_denominator_floor(monkeypatch) -> None:
    observed: dict[str, float] = {}

    def fake_evaluate_q(
        self: CalculatorGeneralizedEvaluator,
        q: np.ndarray,
        gcoord: GeneralizedCoordinates,
    ) -> tuple[float, np.ndarray]:
        grad = np.zeros(gcoord.size, dtype=float)
        if self.finite_diff_cell_gradient:
            observed["eps"] = self.finite_diff_eps
            grad[gcoord.atomic_size :] = 0.02
        return 0.0, grad

    monkeypatch.setattr(CalculatorGeneralizedEvaluator, "evaluate_q", fake_evaluate_q)

    result = verify_stress_gradient(FractionalAndVolumeCalculator(), state(), cell_dof_mode="volume_only")

    assert observed["eps"] == pytest.approx(1e-3)
    assert result["max_relative_error"] == pytest.approx(0.02)
    assert result["components_match"] is True


def test_verify_stress_gradient_allows_small_noisy_components_against_global_scale(monkeypatch) -> None:
    lhs = np.array([10.0, 10.0, 20.0, 10.0, 27.0, 2.0])
    rhs = np.array([9.0, 9.0, 19.0, 9.0, 27.0, -0.2])

    def fake_evaluate_q(
        self: CalculatorGeneralizedEvaluator,
        q: np.ndarray,
        gcoord: GeneralizedCoordinates,
    ) -> tuple[float, np.ndarray]:
        grad = np.zeros(gcoord.size, dtype=float)
        grad[gcoord.atomic_size :] = rhs if self.finite_diff_cell_gradient else lhs
        return 0.0, grad

    monkeypatch.setattr(CalculatorGeneralizedEvaluator, "evaluate_q", fake_evaluate_q)

    result = verify_stress_gradient(FractionalAndVolumeCalculator(), state(), cell_dof_mode="shape_6")

    assert result["max_relative_error"] > 1.0
    assert result["absolute_tol"] == pytest.approx(2.7)
    assert result["components_match"] is True
