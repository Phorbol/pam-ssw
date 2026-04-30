from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generalized_coordinates import GeneralizedCoordinates
from .generalized_evaluator import GeneralizedEvaluator
from .softening import LocalSofteningModel


@dataclass(frozen=True)
class GeneralizedGaussianBiasTerm:
    center_q: np.ndarray
    direction_q: np.ndarray
    sigma: float
    weight: float
    gcoord: GeneralizedCoordinates

    def __post_init__(self) -> None:
        center = np.asarray(self.center_q, dtype=float)
        direction = self.gcoord.metric.normalized(np.asarray(self.direction_q, dtype=float))
        if center.shape != (self.gcoord.size,) or direction.shape != (self.gcoord.size,):
            raise ValueError("generalized bias vectors have the wrong size")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        object.__setattr__(self, "center_q", center)
        object.__setattr__(self, "direction_q", direction)

    def evaluate_q(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        delta = self.gcoord.delta_q(np.asarray(q, dtype=float), self.center_q)
        projection = self.gcoord.metric.dot(delta, self.direction_q)
        energy = float(self.weight * np.exp(-0.5 * (projection / self.sigma) ** 2))
        gradient = -(energy * projection / (self.sigma * self.sigma)) * self.direction_q
        return energy, gradient


class GeneralizedProposalPotential:
    def __init__(
        self,
        evaluator: GeneralizedEvaluator,
        gcoord: GeneralizedCoordinates,
        biases: list[GeneralizedGaussianBiasTerm] | None = None,
        softening: LocalSofteningModel | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.gcoord = gcoord
        self.biases = biases or []
        self.softening = softening

    def evaluate_q(self, q: np.ndarray, gcoord: GeneralizedCoordinates | None = None) -> tuple[float, np.ndarray]:
        gcoord = self.gcoord if gcoord is None else gcoord
        energy, gradient = self.evaluator.evaluate_q(q, gcoord)
        total_energy = float(energy)
        total_gradient = np.asarray(gradient, dtype=float).copy()
        for bias in self.biases:
            bias_energy, bias_gradient = bias.evaluate_q(q)
            total_energy += bias_energy
            total_gradient += bias_gradient
        if self.softening is not None:
            state = gcoord.to_state(q)
            soft_energy, soft_gradient = self.softening.evaluate(state.flatten_positions())
            total_energy += soft_energy
            total_gradient += cartesian_gradient_to_q(soft_gradient.reshape(state.n_atoms, 3), gcoord, state)
        return total_energy, total_gradient


def cartesian_gradient_to_q(gradient: np.ndarray, gcoord: GeneralizedCoordinates, state) -> np.ndarray:
    gradient = np.asarray(gradient, dtype=float).reshape(state.n_atoms, 3)
    grad_q = np.zeros(gcoord.size, dtype=float)
    grad_fractional = gradient @ np.asarray(state.cell, dtype=float).T
    if gcoord.atomic_size:
        grad_q[: gcoord.atomic_size] = grad_fractional[state.movable_mask].reshape(-1)
    return grad_q
