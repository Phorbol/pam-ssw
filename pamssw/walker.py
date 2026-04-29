from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .acquisition import AcquisitionPolicy, BanditSelector, ProposalOutcome, ProposalScorer
from .accounting import BudgetExceeded, EvalCounter
from .bias import GaussianBiasTerm
from .config import LSSSWConfig, RelaxConfig, SSWConfig
from .coordinates import CartesianCoordinates, TangentVector
from .fingerprint import structural_descriptor
from .pbc import mic_displacement, mic_distance_matrix, wrap_positions
from .relax import Relaxer
from .result import RelaxResult, SearchResult, WalkRecord
from .rigid import project_out_rigid_body_modes, rigid_body_overlap
from .softening import LocalSofteningModel
from .state import State


class ProposalPotential:
    def __init__(
        self,
        calculator,
        biases: list[GaussianBiasTerm] | None = None,
        softening: LocalSofteningModel | None = None,
    ) -> None:
        self.calculator = calculator
        self.biases = biases or []
        self.softening = softening

    def evaluate(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        energy, gradient = self.calculator.evaluate_flat(flat_positions, template)
        total_gradient = gradient.copy()
        total_energy = energy
        for bias in self.biases:
            bias_energy, bias_gradient = bias.evaluate(flat_positions)
            total_energy += bias_energy
            total_gradient += bias_gradient
        if self.softening is not None:
            soft_energy, soft_gradient = self.softening.evaluate(flat_positions)
            total_energy += soft_energy
            total_gradient += soft_gradient
        return float(total_energy), total_gradient


@dataclass(frozen=True)
class GeometryValidator:
    min_distance: float = 0.5

    def is_valid_state(self, state: State) -> bool:
        if not np.all(np.isfinite(state.positions)):
            return False
        if state.cell is not None and not np.all(np.isfinite(state.cell)):
            return False
        if any(state.pbc):
            return True
        if state.n_atoms < 2:
            return True
        distances = np.linalg.norm(state.positions[:, None, :] - state.positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return bool(np.min(distances) >= self.min_distance)

    def is_valid_evaluation(self, state: State, calculator) -> bool:
        if not self.is_valid_state(state):
            return False
        try:
            energy, gradient = calculator.evaluate_flat(state.flatten_positions(), state)
        except Exception:
            return False
        return bool(np.isfinite(energy) and np.all(np.isfinite(gradient)))


@dataclass
class DirectionChoice:
    direction: np.ndarray
    curvature: float
    kind: DirectionCandidateKind
    candidate_count: int
    mean_rigid_body_overlap: float = 0.0
    mean_post_projection_rigid_body_overlap: float = 0.0


@dataclass(frozen=True)
class TrustRegionUpdate:
    predicted_delta: float
    true_delta: float
    model_error: float
    damaged: bool
    sigma_scale: float
    weight_scale: float
    action: str


@dataclass(frozen=True)
class TrustRegionBiasController:
    error_tolerance: float = 1.0
    gamma_down: float = 0.5
    gamma_up: float = 1.15
    min_scale: float = 0.25
    max_scale: float = 2.0
    damage_ratio: float = 8.0
    epsilon: float = 1e-8

    def update(
        self,
        curvature: float,
        sigma: float,
        true_delta: float,
        sigma_scale: float,
        weight_scale: float,
        g_parallel: float = 0.0,
        error_floor: float = 0.0,
    ) -> TrustRegionUpdate:
        predicted_delta = self.predicted_delta(curvature, sigma, g_parallel=g_parallel)
        denominator = max(abs(predicted_delta), float(error_floor)) + self.epsilon
        model_error = abs(true_delta - predicted_delta) / denominator
        damaged = true_delta > max(1.0, self.damage_ratio * denominator)
        if damaged or model_error > self.error_tolerance:
            return TrustRegionUpdate(
                predicted_delta=predicted_delta,
                true_delta=float(true_delta),
                model_error=float(model_error),
                damaged=damaged,
                sigma_scale=self._clip(sigma_scale * self.gamma_down),
                weight_scale=self._clip(weight_scale * self.gamma_down),
                action="shrink",
            )
        return TrustRegionUpdate(
            predicted_delta=predicted_delta,
            true_delta=float(true_delta),
            model_error=float(model_error),
            damaged=False,
            sigma_scale=self._clip(sigma_scale * self.gamma_up),
            weight_scale=self._clip(weight_scale * self.gamma_up),
            action="expand",
        )

    @staticmethod
    def predicted_delta(curvature: float, sigma: float, g_parallel: float = 0.0) -> float:
        return float(sigma * g_parallel + 0.5 * sigma * sigma * curvature)

    def _clip(self, value: float) -> float:
        return float(np.clip(value, self.min_scale, self.max_scale))


@dataclass
class StepTargetController:
    fallback_target: float
    eta_energy_scale: float = 0.2
    min_fraction: float = 0.05
    max_factor: float = 5.0
    target_escape_rate: float = 0.2
    damage_tolerance: float = 0.3
    feedback_warmup_trials: int = 4
    gamma_up: float = 1.1
    gamma_down: float = 0.7

    def __post_init__(self) -> None:
        if self.fallback_target <= 0.0:
            raise ValueError("fallback_target must be positive")
        self.min_target = self.min_fraction * self.fallback_target
        self.max_target = self.max_factor * self.fallback_target
        self.multiplier = 1.0
        self.trials = 0
        self.escapes = 0
        self.damage_events = 0
        self.last_target = self.fallback_target

    def target(self, archive=None) -> float:
        raw_target = self._archive_target(archive)
        self.last_target = float(np.clip(raw_target * self.multiplier, self.min_target, self.max_target))
        return self.last_target

    def record_trial(self, escaped: bool, damaged: bool) -> None:
        self.trials += 1
        self.escapes += int(escaped)
        self.damage_events += int(damaged)
        escape_rate = self.escapes / self.trials
        damage_rate = self.damage_events / self.trials
        if self.trials >= self.feedback_warmup_trials and damage_rate > self.damage_tolerance:
            self.multiplier = float(np.clip(self.multiplier * self.gamma_down, 0.75, 4.0))
        elif escape_rate < self.target_escape_rate:
            self.multiplier = float(np.clip(self.multiplier * self.gamma_up, 0.75, 4.0))

    def stats(self) -> dict[str, float | int]:
        escape_rate = self.escapes / self.trials if self.trials else 0.0
        damage_rate = self.damage_events / self.trials if self.trials else 0.0
        return {
            "adaptive_step_target": float(self.last_target),
            "adaptive_step_multiplier": float(self.multiplier),
            "adaptive_escape_rate": float(escape_rate),
            "adaptive_damage_rate": float(damage_rate),
        }

    def _archive_target(self, archive) -> float:
        if archive is None or len(archive.entries) < 2:
            return self.fallback_target
        energies = np.asarray([entry.energy for entry in archive.entries], dtype=float)
        best = float(energies.min())
        deltas = energies - best
        median = float(np.median(deltas))
        mad = float(np.median(np.abs(deltas - median)))
        scale = max(mad, float(np.median(deltas[deltas > 1e-12])) if np.any(deltas > 1e-12) else 0.0)
        if scale <= 1e-12:
            return self.fallback_target
        return float(np.clip(self.eta_energy_scale * scale, self.min_target, self.max_target))


class DirectionCandidateKind(str, Enum):
    SOFT = "soft"
    RANDOM = "random"
    BOND = "bond"


@dataclass(frozen=True)
class DirectionCandidate:
    kind: DirectionCandidateKind
    direction: np.ndarray
    damage_risk: float = 0.0
    rigid_body_overlap: float = 0.0
    post_projection_rigid_body_overlap: float = 0.0


@dataclass(frozen=True)
class DirectionScorer:
    damage_weight: float = 1.0
    continuity_weight: float = 0.1
    anchor_weight: float = 0.5
    novelty_weight: float = 0.5

    def score(
        self,
        curvature: float,
        sigma: float,
        direction: np.ndarray,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
        damage_risk: float,
    ) -> float:
        energy_cost = 0.5 * sigma * sigma * curvature
        discontinuity = 0.0
        if previous_direction is not None:
            prev = previous_direction / (np.linalg.norm(previous_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            discontinuity = float(np.linalg.norm(cur - prev) ** 2)
        anchor_penalty = 0.0
        if anchor_direction is not None:
            anchor = anchor_direction / (np.linalg.norm(anchor_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            anchor_penalty = float(np.linalg.norm(cur - anchor) ** 2)
        return float(
            -energy_cost
            - self.damage_weight * damage_risk
            - self.continuity_weight * discontinuity
            - self.anchor_weight * anchor_penalty
        )

    def score_candidate(
        self,
        state: State,
        candidate: DirectionCandidate,
        curvature: float,
        sigma: float,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
        archive,
    ) -> float:
        score = self.score(
            curvature=curvature,
            sigma=sigma,
            direction=candidate.direction,
            previous_direction=previous_direction,
            anchor_direction=anchor_direction,
            damage_risk=candidate.damage_risk,
        )
        if archive is None:
            return score
        probe = CartesianCoordinates.from_state(state).displace(TangentVector(candidate.direction), sigma)
        novelty_gain = archive.coverage_gain(structural_descriptor(probe))
        return float(score + self.novelty_weight * novelty_gain)


class CandidateDirectionGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        n_random: int,
        bond_pairs: list[tuple[int, int]] | None = None,
        n_bond_pairs: int = 0,
        bond_distance_threshold: float | None = None,
    ) -> None:
        self.rng = rng
        self.n_random = n_random
        self.bond_pairs = bond_pairs or []
        self.n_bond_pairs = n_bond_pairs
        self.bond_distance_threshold = bond_distance_threshold
        self.last_initial_bond_pair: tuple[int, int] | None = None
        self.last_random_bond_pairs_requested = 0
        self.last_random_bond_pairs_generated = 0
        self.last_random_bond_candidates_valid = 0

    def generate(self, state: State, previous_direction: np.ndarray | None) -> list[DirectionCandidate]:
        coordinates = CartesianCoordinates.from_state(state)
        candidates: list[DirectionCandidate] = []
        if previous_direction is not None:
            candidates.append(self._candidate(state, DirectionCandidateKind.SOFT, previous_direction))
        for atom_i, atom_j in self.bond_pairs:
            direction = self._bond_direction(state, atom_i, atom_j)
            if direction is not None:
                candidates.append(self._candidate(state, DirectionCandidateKind.BOND, direction))
        dynamic_pairs = self._random_non_neighbor_pairs(
            state,
            n_pairs=self.n_bond_pairs,
            distance_threshold=self.bond_distance_threshold,
        )
        self.last_random_bond_pairs_requested = self.n_bond_pairs
        self.last_random_bond_pairs_generated = len(dynamic_pairs)
        self.last_random_bond_candidates_valid = 0
        for atom_i, atom_j in dynamic_pairs:
            direction = self._bond_direction(state, atom_i, atom_j)
            if direction is not None:
                candidates.append(self._candidate(state, DirectionCandidateKind.BOND, direction))
                self.last_random_bond_candidates_valid += 1
        n_random = max(0, self.n_random - len(dynamic_pairs))
        for _ in range(n_random):
            active = self.rng.normal(size=coordinates.active_size)
            active /= np.linalg.norm(active) + 1e-12
            direction = coordinates.full_tangent_from_active(active).values
            candidates.append(self._candidate(state, DirectionCandidateKind.RANDOM, direction))
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
        coordinates = CartesianCoordinates.from_state(state)
        active = self.rng.normal(size=coordinates.active_size)
        active /= np.linalg.norm(active) + 1e-12
        random_direction = coordinates.full_tangent_from_active(active).values
        random_direction = self._candidate(state, DirectionCandidateKind.RANDOM, random_direction).direction

        bond_direction = np.zeros_like(random_direction)
        self.last_initial_bond_pair = None
        pairs = self._random_non_neighbor_pairs(
            state,
            n_pairs=n_bond_pairs,
            distance_threshold=bond_distance_threshold,
        )
        if pairs:
            atom_i, atom_j = pairs[int(self.rng.integers(0, len(pairs)))]
            raw_bond = self._bond_direction(state, atom_i, atom_j)
            if raw_bond is not None:
                self.last_initial_bond_pair = (atom_i, atom_j)
                bond_direction = self._candidate(state, DirectionCandidateKind.BOND, raw_bond).direction

        progress = step_index / max(1, max_steps - 1)
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress
        mixed = random_direction + lambda_t * bond_direction
        if np.linalg.norm(mixed) <= 1e-12:
            return random_direction
        return self._normalized(mixed)

    def _candidate(self, state: State, kind: DirectionCandidateKind, direction: np.ndarray) -> DirectionCandidate:
        raw = self._normalized(direction)
        overlap = rigid_body_overlap(state, raw)
        projected = project_out_rigid_body_modes(state, raw)
        if np.linalg.norm(projected) <= 1e-12:
            projected = raw
        projected = self._normalized(projected)
        post_overlap = rigid_body_overlap(state, projected)
        return DirectionCandidate(
            kind=kind,
            direction=projected,
            rigid_body_overlap=overlap,
            post_projection_rigid_body_overlap=post_overlap,
        )

    @staticmethod
    def _normalized(direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return direction / (np.linalg.norm(direction) + 1e-12)

    def _bond_direction(self, state: State, atom_i: int, atom_j: int) -> np.ndarray | None:
        if atom_i < 0 or atom_j < 0 or atom_i >= state.n_atoms or atom_j >= state.n_atoms or atom_i == atom_j:
            return None
        delta = mic_displacement(
            state.positions[atom_j : atom_j + 1],
            state.positions[atom_i : atom_i + 1],
            state.cell,
            state.pbc,
        )[0]
        norm = np.linalg.norm(delta)
        if norm <= 1e-12:
            return None
        axis = delta / norm
        values = np.zeros(state.n_atoms * 3, dtype=float).reshape(state.n_atoms, 3)
        if state.movable_mask[atom_i]:
            values[atom_i] -= axis
        if state.movable_mask[atom_j]:
            values[atom_j] += axis
        flat = values.reshape(-1)
        if np.linalg.norm(flat) <= 1e-12:
            return None
        return self._normalized(flat)

    def _random_non_neighbor_pairs(
        self,
        state: State,
        n_pairs: int,
        distance_threshold: float | None = None,
    ) -> list[tuple[int, int]]:
        if n_pairs <= 0 or all(state.pbc):
            return []
        movable_indices = np.where(state.movable_mask)[0]
        if len(movable_indices) < 2:
            return []
        threshold = self._adaptive_non_neighbor_threshold(state, distance_threshold)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max(50, n_pairs * 50)
        while len(pairs) < n_pairs and attempts < max_attempts:
            atom_i, atom_j = self.rng.choice(movable_indices, size=2, replace=False)
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
        return pairs

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


@dataclass(frozen=True)
class CandidateProposal:
    label: str
    state: State


class SoftModeOracle:
    def __init__(
        self,
        calculator,
        rng: np.random.Generator,
        candidates: int,
        bond_pairs: list[tuple[int, int]] | None = None,
        n_bond_pairs: int = 0,
        bond_distance_threshold: float | None = None,
        anchor_weight: float = 0.5,
        hvp_epsilon: float = 1e-3,
    ) -> None:
        self.calculator = calculator
        self.rng = rng
        self.candidates = candidates
        self.hvp_epsilon = hvp_epsilon
        self.generator = CandidateDirectionGenerator(
            rng,
            candidates,
            bond_pairs=bond_pairs,
            n_bond_pairs=n_bond_pairs,
            bond_distance_threshold=bond_distance_threshold,
        )
        self.scorer = DirectionScorer(anchor_weight=anchor_weight)

    def choose_direction(
        self,
        state: State,
        proposal: ProposalPotential,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None = None,
        step_scale_fn=None,
        archive=None,
    ) -> DirectionChoice:
        best_direction: np.ndarray | None = None
        best_curvature: float | None = None
        best_score: float | None = None
        candidates = self.generator.generate(state, previous_direction)
        best_kind: DirectionCandidateKind | None = None
        rigid_overlap_sum = 0.0
        post_projection_rigid_overlap_sum = 0.0
        for candidate in candidates:
            rigid_overlap_sum += candidate.rigid_body_overlap
            post_projection_rigid_overlap_sum += candidate.post_projection_rigid_body_overlap
            curvature = self._directional_curvature(state, proposal, candidate.direction)
            sigma = self._step_scale_from_curvature(curvature) if step_scale_fn is None else step_scale_fn(curvature)
            score = self.scorer.score_candidate(
                state=state,
                candidate=candidate,
                curvature=curvature,
                sigma=sigma,
                previous_direction=previous_direction,
                anchor_direction=anchor_direction,
                archive=archive,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_curvature = curvature
                best_direction = candidate.direction
                best_kind = candidate.kind
        assert best_direction is not None and best_curvature is not None and best_kind is not None

        return DirectionChoice(
            best_direction,
            best_curvature,
            best_kind,
            len(candidates),
            rigid_overlap_sum / len(candidates),
            post_projection_rigid_overlap_sum / len(candidates),
        )

    def _directional_curvature(
        self,
        state: State,
        proposal: ProposalPotential,
        direction: np.ndarray,
        epsilon: float | None = None,
    ) -> float:
        epsilon = self.hvp_epsilon if epsilon is None else epsilon
        coordinates = CartesianCoordinates.from_state(state)
        tangent = TangentVector(direction)
        plus = coordinates.displace(tangent, epsilon)
        minus = coordinates.displace(tangent, -epsilon)
        _, grad_plus = proposal.evaluate(plus.flatten_positions(), plus)
        _, grad_minus = proposal.evaluate(minus.flatten_positions(), minus)
        hvp = (grad_plus - grad_minus) / (2.0 * epsilon)
        return float(np.dot(hvp, direction))

    @staticmethod
    def _step_scale_from_curvature(curvature: float) -> float:
        effective = max(abs(curvature), 1e-4)
        return float(np.sqrt(2.0 / effective))


class SurfaceWalker:
    def __init__(self, calculator, config: SSWConfig, softening_enabled: bool) -> None:
        self.calculator = EvalCounter(calculator, max_force_evals=config.max_force_evals)
        self.config = config
        self.softening_enabled = softening_enabled
        self.rng = np.random.default_rng(config.rng_seed)
        bond_pairs = config.local_softening_pairs if softening_enabled and isinstance(config, LSSSWConfig) else []
        self.oracle = SoftModeOracle(
            calculator,
            self.rng,
            config.oracle_candidates,
            bond_pairs=bond_pairs,
            n_bond_pairs=config.n_bond_pairs,
            bond_distance_threshold=config.bond_distance_threshold,
            anchor_weight=config.anchor_weight,
            hvp_epsilon=config.hvp_epsilon,
        )
        self.proposal_scorer = ProposalScorer.for_mode(config.search_mode)
        self.selector = BanditSelector(
            policy=AcquisitionPolicy(
                archive_density_weight=config.archive_density_weight,
                novelty_weight=config.novelty_weight,
                frontier_weight=config.frontier_weight,
                exploration_weight=config.bandit_exploration_weight,
                baseline_probability=config.baseline_selection_probability,
                beta_energy=config.bandit_energy_weight,
            )
        )
        self.trust_controller = TrustRegionBiasController()
        self.step_target_controller = StepTargetController(config.target_uphill_energy)
        self.geometry_validator = GeometryValidator()
        self._reset_trust_stats()
        self._reset_direction_stats()
        self._reset_relax_stats()
        self._reset_bias_stats()
        self._reset_local_softening_stats()

    def relax_true_minimum(self, state: State) -> RelaxResult:
        if not self.geometry_validator.is_valid_state(state):
            raise BudgetExceeded("invalid geometry before true relaxation")
        relaxer = Relaxer(self.calculator.evaluate_flat, optimizer=self.config.quench_optimizer)
        relax_config = RelaxConfig(fmax=self.config.quench_fmax, maxiter=400)
        result = relaxer.relax(state, fmax=relax_config.fmax, maxiter=relax_config.maxiter)
        if not self.geometry_validator.is_valid_evaluation(result.state, self.calculator):
            raise BudgetExceeded("invalid geometry after true relaxation")
        self._record_relax_result("true_quench", result, relax_config.fmax)
        return result

    def run(self, initial_state: State):
        from .archive import MinimaArchive

        self._reset_trust_stats()
        self._reset_direction_stats()
        self._reset_relax_stats()
        self._reset_bias_stats()
        self._reset_local_softening_stats()
        self._reset_accepted_structure_log()
        initial = self.relax_true_minimum(initial_state)
        archive = MinimaArchive(
            energy_tol=self.config.dedup_energy_tol,
            rmsd_tol=self.config.dedup_rmsd_tol,
            max_prototypes=self.config.max_prototypes,
        )
        best_entry = archive.add(initial.state, initial.energy, parent_id=None)
        walk_history: list[WalkRecord] = []
        local_relaxations = 1

        completed_trials = 0
        budget_exhausted = False
        for trial_index in range(self.config.max_trials):
            if self.calculator.exhausted():
                budget_exhausted = True
                break
            step_target = self.step_target_controller.target(archive)
            damage_events_before = self._trust_damage_events
            if self.config.use_archive_acquisition:
                seed_entry = archive.select_seed(self.selector, self.rng)
            else:
                seed_entry = archive.next_seed()
                seed_entry.visits += 1
                seed_entry.node_trials += 1
            try:
                proposals = self._proposal_pool(seed_entry.state, archive, trial_index, step_target)
            except BudgetExceeded:
                budget_exhausted = True
                break
            best_discovered = None
            best_rank_key: tuple[float, ...] | None = None
            best_reward = 0.0
            any_new = False
            duplicate_failures = 0
            previous_best_energy = best_entry.energy
            for proposal in proposals:
                try:
                    candidate = self.relax_true_minimum(proposal.state)
                except BudgetExceeded:
                    budget_exhausted = True
                    break
                local_relaxations += 1
                if self._is_fragmented_cluster(seed_entry.state, candidate.state):
                    self._fragment_rejections += 1
                    duplicate_failures += 1
                    continue
                descriptor = structural_descriptor(candidate.state)
                coverage_gain = archive.coverage_gain(descriptor)
                before_count = len(archive.entries)
                discovered = archive.add(candidate.state, candidate.energy, parent_id=seed_entry.entry_id)
                is_new = len(archive.entries) > before_count
                is_duplicate = not is_new
                duplicate_failures += int(is_duplicate)
                any_new = any_new or is_new
                outcome = ProposalOutcome(
                    energy=candidate.energy,
                    previous_best_energy=previous_best_energy,
                    is_new_minimum=is_new,
                    is_duplicate=is_duplicate,
                    descriptor_coverage_gain=coverage_gain,
                )
                reward = self.proposal_scorer.score(outcome)
                rank_key = self.proposal_scorer.rank_key(outcome)
                if discovered.energy < best_entry.energy:
                    best_entry = discovered
                if is_new:
                    self._record_accepted_structure(
                        trial_index=trial_index + 1,
                        seed_entry_id=seed_entry.entry_id,
                        discovered_entry_id=discovered.entry_id,
                        energy=discovered.energy,
                        best_energy=best_entry.energy,
                    )
                if best_rank_key is None or rank_key > best_rank_key:
                    best_rank_key = rank_key
                    best_reward = reward
                    best_discovered = discovered
            if best_discovered is None:
                if budget_exhausted:
                    break
                self.step_target_controller.record_trial(
                    escaped=False,
                    damaged=self._trust_damage_events > damage_events_before,
                )
                archive.record_success(seed_entry, 0.0, duplicate_failures=max(1, duplicate_failures))
                completed_trials += 1
                continue
            self.step_target_controller.record_trial(
                escaped=any_new,
                damaged=self._trust_damage_events > damage_events_before,
            )
            archive.record_success(seed_entry, best_reward, duplicate_failures=duplicate_failures)
            walk_history.append(
                WalkRecord(
                    seed_entry_id=seed_entry.entry_id,
                    discovered_entry_id=best_discovered.entry_id,
                    energy=best_discovered.energy,
                    accepted_new_basin=best_discovered.entry_id != seed_entry.entry_id,
                )
            )
            completed_trials += 1

        archive.refresh_frontier_status()
        prototype_stats = archive.prototype_occupancy()
        frontier_stats = archive.frontier_diagnostics()
        return SearchResult(
            best_state=best_entry.state,
            best_energy=best_entry.energy,
            archive=archive,
            walk_history=walk_history,
            stats={
                "n_trials": completed_trials,
                "configured_max_trials": self.config.max_trials,
                "n_minima": len(archive.entries),
                "local_relaxations": local_relaxations,
                "force_evaluations": self.calculator.force_evaluations,
                "energy_evaluations": self.calculator.energy_evaluations,
                "max_force_evals": self.config.max_force_evals if self.config.max_force_evals is not None else 0,
                "budget_exhausted": int(budget_exhausted or self.calculator.exhausted()),
                "duplicate_rate": archive.duplicate_rate(),
                "descriptor_degeneracy_rate": archive.descriptor_degeneracy_rate(),
                "archive_prototypes": prototype_stats["n_prototypes"],
                "archive_max_prototypes": prototype_stats["max_prototypes"],
                "archive_max_prototype_weight": prototype_stats["max_prototype_weight"],
                "archive_mean_prototype_weight": prototype_stats["mean_prototype_weight"],
                "frontier_nodes": frontier_stats["frontier_nodes"],
                "dead_nodes": frontier_stats["dead_nodes"],
                "mean_frontier_score": frontier_stats["mean_frontier_score"],
                "mean_node_duplicate_failure_rate": frontier_stats["mean_node_duplicate_failure_rate"],
                "max_node_duplicate_failure_rate": frontier_stats["max_node_duplicate_failure_rate"],
                "coordinate_system": "cartesian_fixed_cell",
                "variable_cell_supported": 0,
                "quench_optimizer": self.config.quench_optimizer,
                "proposal_optimizer": self.config.proposal_optimizer,
                "local_softening_terms_last": self._local_softening_terms_last,
                "local_softening_terms_total": self._local_softening_terms_built_total,
                "local_softening_builds": self._local_softening_builds,
                "local_softening_terms_built_total": self._local_softening_terms_built_total,
                **self._trust_stats_summary(),
                **self._direction_stats_summary(),
                **self._relax_stats_summary(),
                **self.step_target_controller.stats(),
            },
        )

    def _proposal_pool(self, seed_state: State, archive, trial_index: int, step_target: float | None = None) -> list[CandidateProposal]:
        return [
            CandidateProposal("ssw_walk", self._walk_candidate_from_seed(seed_state, archive, step_target))
            for _ in range(self.config.proposal_pool_size)
        ]

    def _walk_candidate_from_seed(self, seed_state: State, archive=None, step_target: float | None = None) -> State:
        current = seed_state
        previous_direction: np.ndarray | None = None
        anchor_direction: np.ndarray | None = None
        biases: list[GaussianBiasTerm] = []
        softening: LocalSofteningModel | None = None
        softening_built = False
        sigma_scale = 1.0
        weight_scale = 1.0

        for step_index in range(self.config.max_steps_per_walk):
            if anchor_direction is None:
                anchor_direction = self.oracle.generator.generate_initial_direction(
                    current,
                    step_index=step_index,
                    max_steps=self.config.max_steps_per_walk,
                    lambda_start=self.config.lambda_bond_start,
                    lambda_end=self.config.lambda_bond_end,
                    n_bond_pairs=self.config.n_bond_pairs,
                    bond_distance_threshold=self.config.bond_distance_threshold,
                )
            if not softening_built:
                softening = self._build_softening(seed_state, anchor_direction)
                softening_built = True
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            choice = self.oracle.choose_direction(
                current,
                proposal,
                previous_direction,
                anchor_direction=anchor_direction,
                step_scale_fn=lambda curvature: self._scaled_step_scale(
                    curvature,
                    sigma_scale,
                    step_target=step_target,
                ),
                archive=archive,
            )
            self._record_direction_choice(choice)
            true_curvature = self._true_directional_curvature(current, choice.direction)
            sigma = self._scaled_step_scale(true_curvature, sigma_scale, step_target=step_target)
            weight = self._bias_weight(choice.curvature, sigma) * weight_scale
            self._record_bias_weight(weight)
            true_before = self.calculator.evaluate(current)
            true_energy_before = true_before.energy
            g_parallel = float(np.dot(true_before.gradient.reshape(-1), choice.direction))
            biases.append(
                GaussianBiasTerm(
                    center=current.flatten_positions(),
                    direction=choice.direction,
                    sigma=sigma,
                    weight=weight,
                )
            )
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            trial_state = CartesianCoordinates.from_state(current).displace(TangentVector(choice.direction), sigma)
            if not self.geometry_validator.is_valid_state(trial_state):
                break
            proposal_relax = Relaxer(proposal.evaluate, optimizer=self.config.proposal_optimizer).relax(
                trial_state,
                fmax=self.config.proposal_fmax,
                maxiter=self.config.proposal_relax_steps,
                coordinate_trust_radius=self.config.proposal_trust_radius,
            )
            self._record_relax_result("proposal_relax", proposal_relax, self.config.proposal_fmax)
            current_candidate, clipped = self._clip_walk_displacement(
                reference=seed_state,
                candidate=proposal_relax.state,
                max_displacement=self.config.walk_trust_radius,
            )
            self._walk_displacement_clips += int(clipped)
            if not self.geometry_validator.is_valid_state(current_candidate):
                break
            true_energy_after = self.calculator.evaluate(current_candidate).energy
            if not np.isfinite(true_energy_after):
                break
            trust_update = self.trust_controller.update(
                curvature=true_curvature,
                sigma=sigma,
                true_delta=true_energy_after - true_energy_before,
                sigma_scale=sigma_scale,
                weight_scale=weight_scale,
                g_parallel=g_parallel,
                error_floor=0.1 * (step_target if step_target is not None else self.config.target_uphill_energy),
            )
            sigma_scale = trust_update.sigma_scale
            weight_scale = trust_update.weight_scale
            self._record_trust_update(trust_update)
            displacement = mic_displacement(
                current_candidate.positions,
                current.positions,
                current.cell,
                current.pbc,
            ).reshape(-1)
            if np.linalg.norm(displacement) > 1e-8:
                previous_direction = displacement / np.linalg.norm(displacement)
            current = current_candidate
            if clipped:
                break
        return current

    def _walk_from_seed(self, seed_state: State) -> RelaxResult:
        return self.relax_true_minimum(self._walk_candidate_from_seed(seed_state))

    def _step_scale(self, curvature: float) -> float:
        effective = max(abs(curvature), 1e-4)
        sigma = np.sqrt(2.0 * self.config.target_uphill_energy / effective)
        return float(np.clip(sigma, self.config.min_step_scale, self.config.max_step_scale))

    def _scaled_step_scale(self, curvature: float, sigma_scale: float, step_target: float | None = None) -> float:
        target = self.config.target_uphill_energy if step_target is None else step_target
        effective = max(abs(curvature), 1e-4)
        sigma = np.sqrt(2.0 * target / effective) * sigma_scale
        return float(np.clip(sigma, self.config.min_step_scale, self.config.max_step_scale))

    def _bias_weight(self, curvature: float, sigma: float) -> float:
        raw = sigma * sigma * max(curvature + self.config.target_negative_curvature, 0.0)
        return float(np.clip(raw, self.config.bias_weight_min, self.config.bias_weight_max))

    def _true_directional_curvature(self, state: State, direction: np.ndarray) -> float:
        proposal = ProposalPotential(self.calculator)
        return self.oracle._directional_curvature(state, proposal, direction)

    def _reset_trust_stats(self) -> None:
        self._trust_steps = 0
        self._trust_model_error_sum = 0.0
        self._trust_shrink_steps = 0
        self._trust_expand_steps = 0
        self._trust_damage_events = 0

    def _reset_direction_stats(self) -> None:
        self._direction_choices = 0
        self._direction_candidate_evaluations = 0
        self._direction_selected = {kind: 0 for kind in DirectionCandidateKind}
        self._direction_rigid_overlap_sum = 0.0
        self._direction_post_projection_rigid_overlap_sum = 0.0
        self._direction_bond_pairs_requested = 0
        self._direction_bond_pairs_generated = 0
        self._direction_bond_candidates_valid = 0
        self._walk_displacement_clips = 0
        self._fragment_rejections = 0

    def _reset_relax_stats(self) -> None:
        self._relax_stats = {
            "true_quench": {
                "count": 0,
                "unconverged": 0,
                "max_gradient": 0.0,
                "n_iter_sum": 0,
                "n_iter_values": [],
                "bound_fraction_sum": 0.0,
                "max_bound_fraction": 0.0,
                "displacement_rms_sum": 0.0,
                "max_displacement": 0.0,
            },
            "proposal_relax": {
                "count": 0,
                "unconverged": 0,
                "max_gradient": 0.0,
                "n_iter_sum": 0,
                "n_iter_values": [],
                "bound_fraction_sum": 0.0,
                "max_bound_fraction": 0.0,
                "displacement_rms_sum": 0.0,
                "max_displacement": 0.0,
            },
        }

    def _reset_bias_stats(self) -> None:
        self._bias_steps = 0
        self._bias_zero_steps = 0
        self._bias_weight_sum = 0.0
        self._bias_weight_max = 0.0

    def _reset_local_softening_stats(self) -> None:
        self._local_softening_terms_last = 0
        self._local_softening_terms_total = 0
        self._local_softening_builds = 0
        self._local_softening_terms_built_total = 0

    def _accepted_structure_log_path(self) -> Path | None:
        path = self.config.accepted_structures_log
        return Path(path) if path is not None else None

    def _reset_accepted_structure_log(self) -> None:
        path = self._accepted_structure_log_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def _record_accepted_structure(
        self,
        *,
        trial_index: int,
        seed_entry_id: int,
        discovered_entry_id: int,
        energy: float,
        best_energy: float,
    ) -> None:
        path = self._accepted_structure_log_path()
        if path is None:
            return
        payload = {
            "trial_index": int(trial_index),
            "seed_entry_id": int(seed_entry_id),
            "discovered_entry_id": int(discovered_entry_id),
            "energy": float(energy),
            "best_energy": float(best_energy),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _record_relax_result(self, label: str, result: RelaxResult, fmax: float) -> None:
        stats = self._relax_stats[label]
        stats["count"] += 1
        stats["n_iter_sum"] += result.n_iter
        stats["n_iter_values"].append(result.n_iter)
        stats["max_gradient"] = max(float(stats["max_gradient"]), result.gradient_norm)
        stats["unconverged"] += int(result.gradient_norm > fmax)
        stats["bound_fraction_sum"] += result.active_bound_fraction
        stats["max_bound_fraction"] = max(float(stats["max_bound_fraction"]), result.active_bound_fraction)
        stats["displacement_rms_sum"] += result.displacement_rms
        stats["max_displacement"] = max(float(stats["max_displacement"]), result.displacement_max)

    def _record_bias_weight(self, weight: float) -> None:
        self._bias_steps += 1
        self._bias_zero_steps += int(abs(weight) <= 1e-14)
        self._bias_weight_sum += float(weight)
        self._bias_weight_max = max(self._bias_weight_max, float(weight))

    def _record_direction_choice(self, choice: DirectionChoice) -> None:
        self._direction_choices += 1
        self._direction_candidate_evaluations += choice.candidate_count
        self._direction_selected[choice.kind] += 1
        self._direction_rigid_overlap_sum += choice.mean_rigid_body_overlap
        self._direction_post_projection_rigid_overlap_sum += choice.mean_post_projection_rigid_body_overlap
        self._direction_bond_pairs_requested += self.oracle.generator.last_random_bond_pairs_requested
        self._direction_bond_pairs_generated += self.oracle.generator.last_random_bond_pairs_generated
        self._direction_bond_candidates_valid += self.oracle.generator.last_random_bond_candidates_valid

    def _record_trust_update(self, update: TrustRegionUpdate) -> None:
        self._trust_steps += 1
        self._trust_model_error_sum += update.model_error
        if update.action == "shrink":
            self._trust_shrink_steps += 1
        if update.action == "expand":
            self._trust_expand_steps += 1
        if update.damaged:
            self._trust_damage_events += 1

    def _trust_stats_summary(self) -> dict[str, float | int]:
        mean_error = self._trust_model_error_sum / self._trust_steps if self._trust_steps else 0.0
        return {
            "trust_region_steps": self._trust_steps,
            "trust_model_error_mean": float(mean_error),
            "trust_shrink_steps": self._trust_shrink_steps,
            "trust_expand_steps": self._trust_expand_steps,
            "trust_damage_events": self._trust_damage_events,
        }

    def _direction_stats_summary(self) -> dict[str, float | int]:
        mean_pool_size = (
            self._direction_candidate_evaluations / self._direction_choices if self._direction_choices else 0.0
        )
        mean_rigid_overlap = self._direction_rigid_overlap_sum / self._direction_choices if self._direction_choices else 0.0
        mean_post_projection_overlap = (
            self._direction_post_projection_rigid_overlap_sum / self._direction_choices
            if self._direction_choices
            else 0.0
        )
        return {
            "direction_choices": self._direction_choices,
            "direction_candidate_evaluations": self._direction_candidate_evaluations,
            "direction_mean_candidate_pool_size": float(mean_pool_size),
            "direction_rigid_body_overlap_mean": float(mean_rigid_overlap),
            "direction_post_projection_rigid_body_overlap_mean": float(mean_post_projection_overlap),
            "direction_selected_soft": self._direction_selected[DirectionCandidateKind.SOFT],
            "direction_selected_random": self._direction_selected[DirectionCandidateKind.RANDOM],
            "direction_selected_bond": self._direction_selected[DirectionCandidateKind.BOND],
            "direction_bond_pairs_requested": self._direction_bond_pairs_requested,
            "direction_bond_pairs_generated": self._direction_bond_pairs_generated,
            "direction_bond_candidates_valid": self._direction_bond_candidates_valid,
            "walk_displacement_clips": self._walk_displacement_clips,
            "fragment_rejections": self._fragment_rejections,
        }

    def _relax_stats_summary(self) -> dict[str, float | int]:
        summary: dict[str, float | int] = {}
        for label, stats in self._relax_stats.items():
            count = int(stats["count"])
            summary[f"{label}_count"] = count
            summary[f"{label}_unconverged"] = int(stats["unconverged"])
            summary[f"{label}_max_gradient"] = float(stats["max_gradient"])
            summary[f"{label}_mean_iterations"] = float(stats["n_iter_sum"] / count) if count else 0.0
            n_iter_values = np.asarray(stats["n_iter_values"], dtype=float)
            summary[f"{label}_min_iterations"] = float(np.min(n_iter_values)) if count else 0.0
            summary[f"{label}_median_iterations"] = float(np.median(n_iter_values)) if count else 0.0
            summary[f"{label}_p90_iterations"] = float(np.percentile(n_iter_values, 90)) if count else 0.0
            summary[f"{label}_max_iterations"] = float(np.max(n_iter_values)) if count else 0.0
            summary[f"{label}_active_bound_fraction_mean"] = (
                float(stats["bound_fraction_sum"] / count) if count else 0.0
            )
            summary[f"{label}_active_bound_fraction_max"] = float(stats["max_bound_fraction"])
            summary[f"{label}_displacement_rms_mean"] = (
                float(stats["displacement_rms_sum"] / count) if count else 0.0
            )
            summary[f"{label}_displacement_max"] = float(stats["max_displacement"])
        summary["bias_steps"] = self._bias_steps
        summary["bias_zero_weight_steps"] = self._bias_zero_steps
        summary["bias_zero_weight_fraction"] = (
            float(self._bias_zero_steps / self._bias_steps) if self._bias_steps else 0.0
        )
        summary["bias_weight_mean"] = float(self._bias_weight_sum / self._bias_steps) if self._bias_steps else 0.0
        summary["bias_weight_max"] = float(self._bias_weight_max)
        return summary

    def _build_softening(self, seed_state: State, direction: np.ndarray | None = None) -> LocalSofteningModel | None:
        if not self.softening_enabled or not isinstance(self.config, LSSSWConfig):
            self._local_softening_terms_last = 0
            return None
        if self.config.local_softening_mode == "manual" and not self.config.local_softening_pairs:
            self._local_softening_terms_last = 0
            return None
        softening = LocalSofteningModel.from_state(
            seed_state,
            pairs=self.config.local_softening_pairs,
            strength=self.config.local_softening_strength,
            mode=self.config.local_softening_mode,
            cutoff_scale=self.config.local_softening_cutoff_scale,
            active_indices=self._softening_active_indices(seed_state, direction),
            penalty=self.config.local_softening_penalty,
            xi=self.config.local_softening_xi,
            cutoff=self.config.local_softening_cutoff,
            adaptive_strength=self.config.local_softening_adaptive_strength,
            max_strength_scale=self.config.local_softening_max_strength_scale,
            deviation_scale=self.config.local_softening_deviation_scale,
        )
        self._local_softening_terms_last = len(softening.terms)
        if self._local_softening_terms_last == 0:
            return None
        self._local_softening_builds += 1
        self._local_softening_terms_built_total += self._local_softening_terms_last
        self._local_softening_terms_total = self._local_softening_terms_built_total
        return softening

    def _softening_active_indices(self, seed_state: State, direction: np.ndarray | None = None) -> np.ndarray | None:
        if not isinstance(self.config, LSSSWConfig) or self.config.local_softening_mode != "active_neighbors":
            return None
        movable_indices = np.where(seed_state.movable_mask)[0]
        active_count = self.config.local_softening_active_count
        if active_count is None:
            return movable_indices
        if direction is None:
            return movable_indices[:active_count]
        displacement = np.asarray(direction, dtype=float).reshape(seed_state.n_atoms, 3)
        scores = np.linalg.norm(displacement, axis=1)
        movable_scores = scores[movable_indices]
        selected_positions = np.argsort(-movable_scores, kind="stable")[:active_count]
        selected = movable_indices[selected_positions]
        return np.sort(selected)

    @staticmethod
    def _clip_walk_displacement(reference: State, candidate: State, max_displacement: float) -> tuple[State, bool]:
        displacement = mic_displacement(candidate.positions, reference.positions, reference.cell, reference.pbc)
        norms = np.linalg.norm(displacement, axis=1)
        movable = candidate.movable_mask
        clipped = movable & (norms > max_displacement)
        if not np.any(clipped):
            return candidate, False
        scale = np.ones(candidate.n_atoms, dtype=float)
        scale[clipped] = max_displacement / (norms[clipped] + 1e-12)
        positions = reference.positions + displacement * scale[:, None]
        if np.any(candidate.fixed_mask):
            positions[candidate.fixed_mask] = reference.positions[candidate.fixed_mask]
        positions = wrap_positions(positions, candidate.cell, candidate.pbc)
        return (
            State(
                numbers=candidate.numbers.copy(),
                positions=positions,
                cell=None if candidate.cell is None else candidate.cell.copy(),
                pbc=candidate.pbc,
                fixed_mask=candidate.fixed_mask.copy(),
                metadata=candidate.metadata.copy(),
            ),
            True,
        )

    def _is_fragmented_cluster(self, reference: State, candidate: State) -> bool:
        if self.config.fragment_guard_factor is None:
            return False
        if all(reference.pbc) or all(candidate.pbc) or candidate.n_atoms < 2:
            return False
        reference_scale = self._max_nearest_neighbor_distance(reference)
        candidate_scale = self._max_nearest_neighbor_distance(candidate)
        if reference_scale <= 1e-12:
            return False
        candidate_median = self._median_nearest_neighbor_distance(candidate)
        parent_fragmented = candidate_scale > self.config.fragment_guard_factor * reference_scale
        self_fragmented = (
            candidate_median > 1e-12
            and candidate_scale > self.config.fragment_guard_factor * candidate_median
        )
        return bool(parent_fragmented or self_fragmented)

    @staticmethod
    def _max_nearest_neighbor_distance(state: State) -> float:
        nearest = SurfaceWalker._nearest_neighbor_distances(state)
        return float(np.max(nearest, initial=0.0))

    @staticmethod
    def _median_nearest_neighbor_distance(state: State) -> float:
        nearest = SurfaceWalker._nearest_neighbor_distances(state)
        return float(np.median(nearest)) if nearest.size else 0.0

    @staticmethod
    def _nearest_neighbor_distances(state: State) -> np.ndarray:
        distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
        np.fill_diagonal(distances, np.inf)
        return np.min(distances, axis=1)
