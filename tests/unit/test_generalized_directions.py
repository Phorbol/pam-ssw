from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pamssw.calculators import EnergyResult
from pamssw.generalized_coordinates import GeneralizedCoordinates
from pamssw.generalized_directions import (
    GeneralizedCandidateDirectionGenerator,
    GeneralizedDirectionCandidateKind,
    GeneralizedDirectionScorer,
    GeneralizedSoftModeOracle,
    generalized_directional_curvature,
)
from pamssw.generalized_evaluator import CalculatorGeneralizedEvaluator
from pamssw.state import State


def state() -> State:
    return State(
        numbers=np.array([6, 6, 6]),
        positions=np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]]),
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=(True, True, True),
    )


@dataclass
class QuadraticQCalculator:
    def evaluate(self, candidate: State) -> EnergyResult:
        frac = candidate.positions @ np.linalg.inv(candidate.cell)
        energy = 0.5 * float(np.sum(frac * frac))
        grad_cart = frac @ np.linalg.inv(candidate.cell).T
        return EnergyResult(energy=energy, gradient=grad_cart, stress=np.zeros((3, 3)))

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        result = self.evaluate(template.with_flat_positions(flat_positions))
        return result.energy, result.gradient.reshape(-1)


def test_candidate_parts_are_in_expected_subspaces() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "shape_6")
    generator = GeneralizedCandidateDirectionGenerator(
        np.random.default_rng(1),
        gcoord,
        n_atomic_random=1,
        n_cell_random=1,
        n_coupled_random=1,
        n_bond_pairs=0,
    )
    candidates = generator.generate(initial, None)
    by_kind = {candidate.kind: candidate.direction for candidate in candidates}
    assert np.linalg.norm(by_kind[GeneralizedDirectionCandidateKind.ATOMIC_RANDOM][gcoord.atomic_size :]) == 0.0
    assert np.linalg.norm(by_kind[GeneralizedDirectionCandidateKind.CELL_RANDOM][: gcoord.atomic_size]) == 0.0
    assert np.linalg.norm(by_kind[GeneralizedDirectionCandidateKind.COUPLED_RANDOM][: gcoord.atomic_size]) > 0.0
    assert np.linalg.norm(by_kind[GeneralizedDirectionCandidateKind.COUPLED_RANDOM][gcoord.atomic_size :]) > 0.0


def test_bond_direction_has_zero_cell_component() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "shape_6")
    generator = GeneralizedCandidateDirectionGenerator(np.random.default_rng(1), gcoord, n_atomic_random=0, n_cell_random=0, n_coupled_random=0, bond_pairs=[(0, 1)])
    candidates = generator.generate(initial, None)
    bond = [candidate for candidate in candidates if candidate.kind == GeneralizedDirectionCandidateKind.BOND][0]
    assert np.linalg.norm(bond.direction[gcoord.atomic_size :]) == 0.0
    assert abs(gcoord.metric.norm(bond.direction) - 1.0) < 1e-10


def test_generalized_curvature_is_finite() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "shape_6")
    evaluator = CalculatorGeneralizedEvaluator(QuadraticQCalculator())
    q = gcoord.to_q(initial)
    direction = np.zeros(gcoord.size)
    direction[0] = 1.0
    curvature = generalized_directional_curvature(evaluator, gcoord, q, direction, 1e-4)
    assert np.isfinite(curvature)
    assert curvature > 0.0


def test_oracle_can_select_cell_or_coupled_candidate() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "shape_6")
    evaluator = CalculatorGeneralizedEvaluator(QuadraticQCalculator())
    oracle = GeneralizedSoftModeOracle(
        evaluator,
        np.random.default_rng(2),
        gcoord,
        candidates=1,
        n_cell_random=1,
        n_coupled_random=1,
    )
    choice = oracle.choose_direction(initial, gcoord.to_q(initial), None, score_sigma=0.1)
    assert choice.candidate_count >= 3
    assert abs(gcoord.metric.norm(choice.direction) - 1.0) < 1e-10


def test_direction_scorer_uses_generalized_metric_for_continuity() -> None:
    initial = state()
    gcoord = GeneralizedCoordinates.from_state(initial, "shape_6", cell_metric_weight=100.0)
    scorer = GeneralizedDirectionScorer(continuity_weight=1.0)
    previous = np.zeros(gcoord.size)
    previous[0] = 1.0
    current = np.zeros(gcoord.size)
    current[0] = 1.0
    current[gcoord.atomic_size] = 1.0
    score = scorer.score(
        curvature=0.0,
        sigma=1.0,
        direction=current,
        previous_direction=previous,
        anchor_direction=None,
        damage_risk=0.0,
        metric=gcoord.metric,
    )
    expected_penalty = gcoord.metric.norm_sq(gcoord.metric.normalized(current) - gcoord.metric.normalized(previous))
    euclidean_penalty = np.linalg.norm(
        current / (np.linalg.norm(current) + 1e-12) - previous / (np.linalg.norm(previous) + 1e-12)
    ) ** 2
    assert score == -expected_penalty
    assert not np.isclose(expected_penalty, euclidean_penalty)
