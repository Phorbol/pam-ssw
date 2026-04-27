from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .acquisition import BanditSelector, ProposalOutcome, ProposalScorer
from .accounting import BudgetExceeded, EvalCounter
from .bias import GaussianBiasTerm
from .config import LSSSWConfig, RelaxConfig, SSWConfig
from .coordinates import CartesianCoordinates, TangentVector
from .fingerprint import structural_descriptor
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
    ) -> TrustRegionUpdate:
        predicted_delta = self.predicted_delta(curvature, sigma)
        denominator = abs(predicted_delta) + self.epsilon
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
    def predicted_delta(curvature: float, sigma: float) -> float:
        return float(0.5 * sigma * sigma * curvature)

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
    CELL = "cell"


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
    novelty_weight: float = 0.5

    def score(
        self,
        curvature: float,
        sigma: float,
        direction: np.ndarray,
        previous_direction: np.ndarray | None,
        damage_risk: float,
    ) -> float:
        energy_cost = 0.5 * sigma * sigma * curvature
        discontinuity = 0.0
        if previous_direction is not None:
            prev = previous_direction / (np.linalg.norm(previous_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            discontinuity = float(np.linalg.norm(cur - prev) ** 2)
        return float(-energy_cost - self.damage_weight * damage_risk - self.continuity_weight * discontinuity)

    def score_candidate(
        self,
        state: State,
        candidate: DirectionCandidate,
        curvature: float,
        sigma: float,
        previous_direction: np.ndarray | None,
        archive,
    ) -> float:
        score = self.score(
            curvature=curvature,
            sigma=sigma,
            direction=candidate.direction,
            previous_direction=previous_direction,
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
    ) -> None:
        self.rng = rng
        self.n_random = n_random
        self.bond_pairs = bond_pairs or []

    def generate(self, state: State, previous_direction: np.ndarray | None) -> list[DirectionCandidate]:
        coordinates = CartesianCoordinates.from_state(state)
        candidates: list[DirectionCandidate] = []
        if previous_direction is not None:
            candidates.append(self._candidate(state, DirectionCandidateKind.SOFT, previous_direction))
        for atom_i, atom_j in self.bond_pairs:
            direction = self._bond_direction(state, atom_i, atom_j)
            if direction is not None:
                candidates.append(self._candidate(state, DirectionCandidateKind.BOND, direction))
        for _ in range(self.n_random):
            active = self.rng.normal(size=coordinates.active_size)
            active /= np.linalg.norm(active) + 1e-12
            direction = coordinates.full_tangent_from_active(active).values
            candidates.append(self._candidate(state, DirectionCandidateKind.RANDOM, direction))
        return candidates

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
        delta = state.positions[atom_j] - state.positions[atom_i]
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
    ) -> None:
        self.calculator = calculator
        self.rng = rng
        self.candidates = candidates
        self.generator = CandidateDirectionGenerator(rng, candidates, bond_pairs=bond_pairs)
        self.scorer = DirectionScorer()

    def choose_direction(
        self,
        state: State,
        proposal: ProposalPotential,
        previous_direction: np.ndarray | None,
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
            sigma = self._step_scale_from_curvature(curvature)
            score = self.scorer.score_candidate(
                state=state,
                candidate=candidate,
                curvature=curvature,
                sigma=sigma,
                previous_direction=previous_direction,
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
        epsilon: float = 1e-3,
    ) -> float:
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
        self.oracle = SoftModeOracle(calculator, self.rng, config.oracle_candidates, bond_pairs=bond_pairs)
        self.proposal_scorer = ProposalScorer.for_mode(config.search_mode)
        self.selector = BanditSelector()
        self.trust_controller = TrustRegionBiasController()
        self.step_target_controller = StepTargetController(config.target_uphill_energy)
        self._reset_trust_stats()
        self._reset_direction_stats()

    def relax_true_minimum(self, state: State) -> RelaxResult:
        relaxer = Relaxer(self.calculator.evaluate_flat)
        relax_config = RelaxConfig(fmax=self.config.quench_fmax, maxiter=400)
        return relaxer.relax(state, fmax=relax_config.fmax, maxiter=relax_config.maxiter)

    def run(self, initial_state: State):
        from .archive import MinimaArchive

        self._reset_trust_stats()
        self._reset_direction_stats()
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
                if best_rank_key is None or rank_key > best_rank_key:
                    best_rank_key = rank_key
                    best_reward = reward
                    best_discovered = discovered
            if best_discovered is None:
                break
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
                **self._trust_stats_summary(),
                **self._direction_stats_summary(),
                **self.step_target_controller.stats(),
            },
        )

    def _proposal_pool(self, seed_state: State, archive, trial_index: int, step_target: float | None = None) -> list[CandidateProposal]:
        return [CandidateProposal("ssw_walk", self._walk_candidate_from_seed(seed_state, archive, step_target))]

    def _walk_candidate_from_seed(self, seed_state: State, archive=None, step_target: float | None = None) -> State:
        current = seed_state
        previous_direction: np.ndarray | None = None
        biases: list[GaussianBiasTerm] = []
        softening = self._build_softening(seed_state)
        sigma_scale = 1.0
        weight_scale = 1.0

        for _ in range(self.config.max_steps_per_walk):
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            choice = self.oracle.choose_direction(current, proposal, previous_direction, archive=archive)
            self._record_direction_choice(choice)
            sigma = self._scaled_step_scale(choice.curvature, sigma_scale, step_target=step_target)
            weight = self._bias_weight(choice.curvature, sigma) * weight_scale
            true_energy_before = self.calculator.evaluate(current).energy
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
            proposal_relax = Relaxer(proposal.evaluate).relax(
                trial_state,
                fmax=self.config.proposal_fmax,
                maxiter=self.config.proposal_relax_steps,
            )
            true_energy_after = self.calculator.evaluate(proposal_relax.state).energy
            trust_update = self.trust_controller.update(
                curvature=choice.curvature,
                sigma=sigma,
                true_delta=true_energy_after - true_energy_before,
                sigma_scale=sigma_scale,
                weight_scale=weight_scale,
            )
            sigma_scale = trust_update.sigma_scale
            weight_scale = trust_update.weight_scale
            self._record_trust_update(trust_update)
            displacement = proposal_relax.state.flatten_positions() - current.flatten_positions()
            if np.linalg.norm(displacement) > 1e-8:
                previous_direction = displacement / np.linalg.norm(displacement)
            current = proposal_relax.state
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
        return float(sigma * sigma * max(curvature + self.config.target_negative_curvature, 0.0))

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

    def _record_direction_choice(self, choice: DirectionChoice) -> None:
        self._direction_choices += 1
        self._direction_candidate_evaluations += choice.candidate_count
        self._direction_selected[choice.kind] += 1
        self._direction_rigid_overlap_sum += choice.mean_rigid_body_overlap
        self._direction_post_projection_rigid_overlap_sum += choice.mean_post_projection_rigid_body_overlap

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
            "direction_selected_cell": self._direction_selected[DirectionCandidateKind.CELL],
        }

    def _build_softening(self, seed_state: State) -> LocalSofteningModel | None:
        if not self.softening_enabled or not isinstance(self.config, LSSSWConfig):
            return None
        if not self.config.local_softening_pairs:
            return None
        return LocalSofteningModel.from_state(
            seed_state,
            pairs=self.config.local_softening_pairs,
            strength=self.config.local_softening_strength,
        )
