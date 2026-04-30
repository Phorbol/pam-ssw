from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .fingerprint import structural_descriptor
from .generalized_coordinates import (
    GeneralizedCoordinates,
    GeneralizedTangentVector,
    project_out_generalized_rigid_modes,
)
from .generalized_evaluator import GeneralizedEvaluator
from .pbc import mic_displacement, mic_distance_matrix
from .state import State


@dataclass(frozen=True)
class GeneralizedDirectionScorer:
    damage_weight: float = 1.0
    continuity_weight: float = 0.1
    anchor_weight: float = 0.5
    novelty_weight: float = 0.5
    history_push_weight: float = 0.1
    novelty_probe_scales: tuple[float, ...] = (1.0,)

    def score(
        self,
        curvature: float,
        sigma: float,
        direction: np.ndarray,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
        damage_risk: float,
        history_push: float = 0.0,
        continuity_weight: float | None = None,
        metric=None,
    ) -> float:
        continuity = self.continuity_weight if continuity_weight is None else continuity_weight
        energy_cost = 0.5 * sigma * sigma * curvature
        discontinuity = 0.0
        if previous_direction is not None:
            prev = _normalized(previous_direction, metric)
            cur = _normalized(direction, metric)
            discontinuity = _norm_sq(cur - prev, metric)
        anchor_penalty = 0.0
        if anchor_direction is not None:
            anchor = _normalized(anchor_direction, metric)
            cur = _normalized(direction, metric)
            anchor_penalty = _norm_sq(cur - anchor, metric)
        return float(
            -energy_cost
            - self.damage_weight * damage_risk
            - continuity * discontinuity
            - self.anchor_weight * anchor_penalty
            + self.history_push_weight * history_push
        )


class GeneralizedDirectionCandidateKind(str, Enum):
    ATOMIC_RANDOM = "atomic_random"
    BOND = "bond"
    CELL_RANDOM = "cell_random"
    CELL_SOFT = "cell_soft"
    COUPLED_RANDOM = "coupled_random"
    COUPLED_SOFT = "coupled_soft"
    MOMENTUM = "momentum"


@dataclass(frozen=True)
class GeneralizedDirectionCandidate:
    kind: GeneralizedDirectionCandidateKind
    direction: np.ndarray
    damage_risk: float = 0.0


@dataclass(frozen=True)
class GeneralizedDirectionChoice:
    direction: np.ndarray
    curvature: float
    kind: GeneralizedDirectionCandidateKind
    candidate_count: int
    mean_rigid_body_overlap: float = 0.0
    mean_post_projection_rigid_body_overlap: float = 0.0


@dataclass
class GeneralizedCandidateDirectionGenerator:
    rng: np.random.Generator
    gcoord: GeneralizedCoordinates
    n_atomic_random: int = 8
    n_cell_random: int = 2
    n_coupled_random: int = 2
    bond_pairs: list[tuple[int, int]] | None = None
    n_bond_pairs: int = 2
    bond_distance_threshold: float | None = None
    enable_momentum_candidate: bool = True
    cell_soft_mode_enabled: bool = True
    coupled_soft_mode_enabled: bool = True

    def __post_init__(self) -> None:
        self.bond_pairs = self.bond_pairs or []
        self.last_random_bond_pairs_requested = 0
        self.last_random_bond_pairs_generated = 0
        self.last_fallback_bond_pairs_generated = 0
        self.last_random_bond_candidates_valid = 0

    def generate(
        self,
        state: State,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None = None,
        anchor_mixing_alpha: float | None = None,
        n_bond_pairs: int | None = None,
    ) -> list[GeneralizedDirectionCandidate]:
        candidates: list[GeneralizedDirectionCandidate] = []
        if self.enable_momentum_candidate and previous_direction is not None:
            candidates.append(
                self._candidate(
                    GeneralizedDirectionCandidateKind.MOMENTUM,
                    self._anchor_mixed_direction(previous_direction, anchor_direction, anchor_mixing_alpha),
                )
            )
        for atom_i, atom_j in self.bond_pairs:
            direction = self._bond_direction(state, atom_i, atom_j)
            if direction is not None:
                candidates.append(self._candidate(GeneralizedDirectionCandidateKind.BOND, direction))
        requested_pairs = self.n_bond_pairs if n_bond_pairs is None else n_bond_pairs
        pairs = self._random_non_neighbor_pairs(state, requested_pairs, self.bond_distance_threshold)
        self.last_random_bond_pairs_requested = requested_pairs
        self.last_random_bond_pairs_generated = len(pairs)
        self.last_random_bond_candidates_valid = 0
        for atom_i, atom_j in pairs:
            direction = self._bond_direction(state, atom_i, atom_j)
            if direction is not None:
                candidates.append(self._candidate(GeneralizedDirectionCandidateKind.BOND, direction))
                self.last_random_bond_candidates_valid += 1
        for _ in range(max(0, self.n_atomic_random - len(pairs))):
            direction = np.zeros(self.gcoord.size, dtype=float)
            direction[: self.gcoord.atomic_size] = self.rng.normal(size=self.gcoord.atomic_size)
            candidates.append(self._candidate(GeneralizedDirectionCandidateKind.ATOMIC_RANDOM, direction))
        for _ in range(self.n_cell_random):
            direction = np.zeros(self.gcoord.size, dtype=float)
            direction[self.gcoord.atomic_size :] = self.rng.normal(size=self.gcoord.cell_dof)
            candidates.append(self._candidate(GeneralizedDirectionCandidateKind.CELL_RANDOM, direction))
        for _ in range(self.n_coupled_random):
            direction = self.rng.normal(size=self.gcoord.size)
            candidates.append(self._candidate(GeneralizedDirectionCandidateKind.COUPLED_RANDOM, direction))
        return candidates

    def generate_initial_direction(
        self,
        state: State,
        step_index: int,
        max_steps: int,
        lambda_start: float,
        lambda_end: float,
        n_bond_pairs: int,
        bond_distance_threshold: float | None,
    ) -> np.ndarray:
        random_direction = np.zeros(self.gcoord.size, dtype=float)
        random_direction[: self.gcoord.atomic_size] = self.rng.normal(size=self.gcoord.atomic_size)
        random_direction = self._candidate(GeneralizedDirectionCandidateKind.ATOMIC_RANDOM, random_direction).direction
        bond_direction = np.zeros_like(random_direction)
        pairs = self._random_non_neighbor_pairs(state, n_bond_pairs, bond_distance_threshold)
        if pairs:
            atom_i, atom_j = pairs[int(self.rng.integers(0, len(pairs)))]
            raw = self._bond_direction(state, atom_i, atom_j)
            if raw is not None:
                bond_direction = self._candidate(GeneralizedDirectionCandidateKind.BOND, raw).direction
        progress = step_index / max(1, max_steps - 1)
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress
        return self.gcoord.metric.normalized(random_direction + lambda_t * bond_direction)

    def soft_candidates(self, evaluator: GeneralizedEvaluator, q: np.ndarray, epsilon: float) -> list[GeneralizedDirectionCandidate]:
        candidates: list[GeneralizedDirectionCandidate] = []
        if self.cell_soft_mode_enabled and self.gcoord.cell_dof > 0:
            pool = [
                self._candidate(
                    GeneralizedDirectionCandidateKind.CELL_RANDOM,
                    np.concatenate([np.zeros(self.gcoord.atomic_size), self.rng.normal(size=self.gcoord.cell_dof)]),
                )
                for _ in range(max(1, self.n_cell_random))
            ]
            candidates.append(self._softest(evaluator, q, epsilon, GeneralizedDirectionCandidateKind.CELL_SOFT, pool))
        if self.coupled_soft_mode_enabled:
            pool = [
                self._candidate(GeneralizedDirectionCandidateKind.COUPLED_RANDOM, self.rng.normal(size=self.gcoord.size))
                for _ in range(max(1, self.n_coupled_random))
            ]
            candidates.append(self._softest(evaluator, q, epsilon, GeneralizedDirectionCandidateKind.COUPLED_SOFT, pool))
        return candidates

    def _softest(
        self,
        evaluator: GeneralizedEvaluator,
        q: np.ndarray,
        epsilon: float,
        kind: GeneralizedDirectionCandidateKind,
        pool: list[GeneralizedDirectionCandidate],
    ) -> GeneralizedDirectionCandidate:
        ranked = [
            (generalized_directional_curvature(evaluator, self.gcoord, q, candidate.direction, epsilon), candidate)
            for candidate in pool
        ]
        _, best = min(ranked, key=lambda item: item[0])
        return GeneralizedDirectionCandidate(kind, best.direction, best.damage_risk)

    def _candidate(self, kind: GeneralizedDirectionCandidateKind, direction: np.ndarray) -> GeneralizedDirectionCandidate:
        direction = self.gcoord.metric.normalized(direction)
        direction = project_out_generalized_rigid_modes(self.gcoord, direction)
        return GeneralizedDirectionCandidate(kind=kind, direction=self.gcoord.metric.normalized(direction))

    def _anchor_mixed_direction(
        self,
        direction: np.ndarray,
        anchor_direction: np.ndarray | None,
        alpha: float | None,
    ) -> np.ndarray:
        if anchor_direction is None or alpha is None:
            return direction
        alpha = float(np.clip(alpha, 0.0, 1.0))
        anchor = self.gcoord.metric.normalized(anchor_direction)
        base = self.gcoord.metric.normalized(direction)
        perpendicular = base - self.gcoord.metric.dot(base, anchor) * anchor
        if self.gcoord.metric.norm(perpendicular) <= 1e-12:
            return anchor
        perpendicular = self.gcoord.metric.normalized(perpendicular)
        mixed = alpha * anchor + np.sqrt(max(0.0, 1.0 - alpha * alpha)) * perpendicular
        return self.gcoord.metric.normalized(mixed)

    def _bond_direction(self, state: State, atom_i: int, atom_j: int) -> np.ndarray | None:
        if atom_i < 0 or atom_j < 0 or atom_i >= state.n_atoms or atom_j >= state.n_atoms or atom_i == atom_j:
            return None
        active_index = {int(atom): idx for idx, atom in enumerate(np.where(state.movable_mask)[0])}
        if atom_i not in active_index and atom_j not in active_index:
            return None
        delta = mic_displacement(
            state.positions[atom_j : atom_j + 1],
            state.positions[atom_i : atom_i + 1],
            state.cell,
            state.pbc,
        )[0]
        norm = np.linalg.norm(delta)
        if norm <= 1e-12 or state.cell is None:
            return None
        axis_frac = (delta / norm) @ np.linalg.inv(state.cell)
        axis_norm = np.linalg.norm(axis_frac)
        if axis_norm <= 1e-12:
            return None
        axis_frac = axis_frac / axis_norm
        direction = np.zeros(self.gcoord.size, dtype=float)
        atomic = direction[: self.gcoord.atomic_size].reshape(-1, 3)
        if atom_i in active_index:
            atomic[active_index[atom_i]] -= axis_frac
        if atom_j in active_index:
            atomic[active_index[atom_j]] += axis_frac
        if self.gcoord.metric.norm(direction) <= 1e-12:
            return None
        return self.gcoord.metric.normalized(direction)

    def _random_non_neighbor_pairs(
        self,
        state: State,
        n_pairs: int,
        distance_threshold: float | None,
    ) -> list[tuple[int, int]]:
        self.last_fallback_bond_pairs_generated = 0
        if n_pairs <= 0:
            return []
        movable = np.where(state.movable_mask)[0]
        if len(movable) < 2:
            return []
        threshold = self._adaptive_non_neighbor_threshold(state, distance_threshold)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(pairs) < n_pairs and attempts < max(50, n_pairs * 50):
            atom_i, atom_j = self.rng.choice(movable, size=2, replace=False)
            pair = tuple(sorted((int(atom_i), int(atom_j))))
            if pair not in seen:
                distance = float(
                    np.linalg.norm(
                        mic_displacement(
                            state.positions[pair[1] : pair[1] + 1],
                            state.positions[pair[0] : pair[0] + 1],
                            state.cell,
                            state.pbc,
                        )[0]
                    )
                )
                if distance > threshold:
                    seen.add(pair)
                    pairs.append(pair)
            attempts += 1
        if len(pairs) < max(1, n_pairs // 2):
            fallback = self._closest_mic_pairs(state, n_pairs - len(pairs), exclude=set(pairs))
            pairs.extend(fallback)
            self.last_fallback_bond_pairs_generated = len(fallback)
        return pairs

    def _closest_mic_pairs(
        self,
        state: State,
        n_pairs: int,
        exclude: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        if n_pairs <= 0:
            return []
        exclude = exclude or set()
        movable = np.where(state.movable_mask)[0]
        ranked: list[tuple[float, tuple[int, int]]] = []
        for left, atom_i in enumerate(movable):
            for atom_j in movable[left + 1 :]:
                pair = tuple(sorted((int(atom_i), int(atom_j))))
                if pair in exclude:
                    continue
                distance = float(
                    np.linalg.norm(
                        mic_displacement(
                            state.positions[pair[1] : pair[1] + 1],
                            state.positions[pair[0] : pair[0] + 1],
                            state.cell,
                            state.pbc,
                        )[0]
                    )
                )
                ranked.append((distance, pair))
        ranked.sort(key=lambda item: item[0])
        return [pair for _, pair in ranked[:n_pairs]]

    @staticmethod
    def _adaptive_non_neighbor_threshold(state: State, configured_threshold: float | None) -> float:
        if configured_threshold is not None:
            return configured_threshold
        if state.n_atoms < 2:
            return 0.0
        distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        finite = nearest[np.isfinite(nearest)]
        if finite.size == 0:
            return 0.0
        return float(1.5 * np.median(finite))


class GeneralizedSoftModeOracle:
    def __init__(
        self,
        evaluator: GeneralizedEvaluator,
        rng: np.random.Generator,
        gcoord: GeneralizedCoordinates,
        candidates: int,
        *,
        bond_pairs: list[tuple[int, int]] | None = None,
        n_bond_pairs: int = 0,
        bond_distance_threshold: float | None = None,
        anchor_weight: float = 0.5,
        continuity_weight: float = 0.1,
        history_push_weight: float = 0.1,
        novelty_probe_scales: tuple[float, ...] = (1.0,),
        enable_momentum_candidate: bool = True,
        anchor_mixing_alpha: float | None = None,
        hvp_epsilon: float = 1e-3,
        n_cell_random: int = 2,
        n_coupled_random: int = 2,
        cell_soft_mode_enabled: bool = True,
        coupled_soft_mode_enabled: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.gcoord = gcoord
        self.hvp_epsilon = hvp_epsilon
        self.anchor_mixing_alpha = anchor_mixing_alpha
        self.generator = GeneralizedCandidateDirectionGenerator(
            rng=rng,
            gcoord=gcoord,
            n_atomic_random=candidates,
            n_cell_random=n_cell_random,
            n_coupled_random=n_coupled_random,
            bond_pairs=bond_pairs,
            n_bond_pairs=n_bond_pairs,
            bond_distance_threshold=bond_distance_threshold,
            enable_momentum_candidate=enable_momentum_candidate,
            cell_soft_mode_enabled=cell_soft_mode_enabled,
            coupled_soft_mode_enabled=coupled_soft_mode_enabled,
        )
        self.scorer = GeneralizedDirectionScorer(
            anchor_weight=anchor_weight,
            continuity_weight=continuity_weight,
            history_push_weight=history_push_weight,
            novelty_probe_scales=novelty_probe_scales,
        )

    def choose_direction(
        self,
        state: State,
        q: np.ndarray,
        previous_direction: np.ndarray | None,
        *,
        anchor_direction: np.ndarray | None = None,
        archive=None,
        history_gradient: np.ndarray | None = None,
        continuity_weight: float | None = None,
        n_bond_pairs: int | None = None,
        score_sigma: float | None = None,
        score_sigma_fn=None,
    ) -> GeneralizedDirectionChoice:
        candidates = self.generator.generate(
            state,
            previous_direction,
            anchor_direction=anchor_direction,
            anchor_mixing_alpha=self.anchor_mixing_alpha,
            n_bond_pairs=n_bond_pairs,
        )
        candidates.extend(self.generator.soft_candidates(self.evaluator, q, self.hvp_epsilon))
        best: tuple[float, float, GeneralizedDirectionCandidate] | None = None
        scoring_anchor = None if self.anchor_mixing_alpha is not None else anchor_direction
        for candidate in candidates:
            curvature = generalized_directional_curvature(self.evaluator, self.gcoord, q, candidate.direction, self.hvp_epsilon)
            sigma = float(score_sigma_fn(curvature)) if score_sigma_fn is not None else float(score_sigma or 1.0)
            history_push = 0.0 if history_gradient is None else -self.gcoord.metric.dot(history_gradient, candidate.direction)
            score = self.scorer.score(
                curvature=curvature,
                sigma=sigma,
                direction=candidate.direction,
                previous_direction=previous_direction,
                anchor_direction=scoring_anchor,
                damage_risk=candidate.damage_risk,
                history_push=history_push,
                continuity_weight=continuity_weight,
                metric=self.gcoord.metric,
            )
            if archive is not None:
                trial_state = self.gcoord.displace(q, GeneralizedTangentVector(candidate.direction), sigma)
                descriptor = (
                    archive.descriptor_for(trial_state)
                    if hasattr(archive, "descriptor_for")
                    else structural_descriptor(trial_state)
                )
                score += self.scorer.novelty_weight * archive.coverage_gain(descriptor)
            if best is None or score > best[0]:
                best = (score, curvature, candidate)
        if best is None:
            raise RuntimeError("no generalized direction candidates were generated")
        _, curvature, candidate = best
        return GeneralizedDirectionChoice(candidate.direction, curvature, candidate.kind, len(candidates))


def generalized_directional_curvature(
    evaluator: GeneralizedEvaluator,
    gcoord: GeneralizedCoordinates,
    q: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
) -> float:
    tangent = gcoord.metric.normalized(direction)
    q_plus = gcoord.fractional_wrap(np.asarray(q, dtype=float) + epsilon * tangent)
    q_minus = gcoord.fractional_wrap(np.asarray(q, dtype=float) - epsilon * tangent)
    _, grad_plus = evaluator.evaluate_q(q_plus, gcoord)
    _, grad_minus = evaluator.evaluate_q(q_minus, gcoord)
    hvp = (grad_plus - grad_minus) / (2.0 * epsilon)
    return float(gcoord.metric.dot(hvp, tangent))


def _normalized(values: np.ndarray, metric) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if metric is None:
        return values / (np.linalg.norm(values) + 1e-12)
    norm = metric.norm(values)
    if norm <= 1e-12:
        return values.copy()
    return values / norm


def _norm_sq(values: np.ndarray, metric) -> float:
    values = np.asarray(values, dtype=float)
    if metric is None:
        return float(np.linalg.norm(values) ** 2)
    return metric.norm_sq(values)
