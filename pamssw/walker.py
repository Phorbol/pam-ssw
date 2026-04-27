from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import BanditSelector, ProposalOutcome, ProposalScorer
from .bias import GaussianBiasTerm
from .config import LSSSWConfig, RelaxConfig, SSWConfig
from .coordinates import CartesianCoordinates, TangentVector
from .fingerprint import structural_descriptor
from .relax import Relaxer
from .result import RelaxResult, SearchResult, WalkRecord
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


@dataclass(frozen=True)
class CandidateProposal:
    label: str
    state: State


class SoftModeOracle:
    def __init__(self, calculator, rng: np.random.Generator, candidates: int) -> None:
        self.calculator = calculator
        self.rng = rng
        self.candidates = candidates

    def choose_direction(
        self,
        state: State,
        proposal: ProposalPotential,
        previous_direction: np.ndarray | None,
    ) -> DirectionChoice:
        coordinates = CartesianCoordinates.from_state(state)
        best_direction: np.ndarray | None = None
        best_curvature: float | None = None
        candidate_vectors: list[np.ndarray] = []
        if previous_direction is not None:
            candidate_vectors.append(previous_direction)
        for _ in range(self.candidates):
            active = self.rng.normal(size=coordinates.active_size)
            active /= np.linalg.norm(active) + 1e-12
            candidate_vectors.append(coordinates.full_tangent_from_active(active).values)
        for vector in candidate_vectors:
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            curvature = self._directional_curvature(state, proposal, vector)
            if best_curvature is None or curvature < best_curvature:
                best_curvature = curvature
                best_direction = vector
        assert best_direction is not None and best_curvature is not None

        return DirectionChoice(best_direction, best_curvature)

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


class SurfaceWalker:
    def __init__(self, calculator, config: SSWConfig, softening_enabled: bool) -> None:
        self.calculator = calculator
        self.config = config
        self.softening_enabled = softening_enabled
        self.rng = np.random.default_rng(config.rng_seed)
        self.oracle = SoftModeOracle(calculator, self.rng, config.oracle_candidates)
        self.proposal_scorer = ProposalScorer()
        self.selector = BanditSelector()

    def relax_true_minimum(self, state: State) -> RelaxResult:
        relaxer = Relaxer(self.calculator.evaluate_flat)
        relax_config = RelaxConfig(fmax=self.config.quench_fmax, maxiter=400)
        return relaxer.relax(state, fmax=relax_config.fmax, maxiter=relax_config.maxiter)

    def run(self, initial_state: State):
        from .archive import MinimaArchive

        initial = self.relax_true_minimum(initial_state)
        archive = MinimaArchive(
            energy_tol=self.config.dedup_energy_tol,
            rmsd_tol=self.config.dedup_rmsd_tol,
        )
        best_entry = archive.add(initial.state, initial.energy, parent_id=None)
        walk_history: list[WalkRecord] = []
        local_relaxations = 1

        for trial_index in range(self.config.max_trials):
            if self.config.use_archive_acquisition:
                seed_entry = archive.select_seed(self.selector, self.rng)
            else:
                seed_entry = archive.next_seed()
                seed_entry.visits += 1
                seed_entry.node_trials += 1
            proposals = self._proposal_pool(seed_entry.state, archive, trial_index)
            best_discovered = None
            best_score = -float("inf")
            best_reward = 0.0
            previous_best_energy = best_entry.energy
            for proposal in proposals:
                candidate = self.relax_true_minimum(proposal.state)
                local_relaxations += 1
                descriptor = structural_descriptor(candidate.state)
                coverage_gain = archive.coverage_gain(descriptor)
                before_count = len(archive.entries)
                discovered = archive.add(candidate.state, candidate.energy, parent_id=seed_entry.entry_id)
                is_new = len(archive.entries) > before_count
                is_duplicate = not is_new
                outcome = ProposalOutcome(
                    energy=candidate.energy,
                    previous_best_energy=previous_best_energy,
                    is_new_minimum=is_new,
                    is_duplicate=is_duplicate,
                    descriptor_coverage_gain=coverage_gain,
                )
                reward = self.proposal_scorer.score(outcome)
                if discovered.energy < best_entry.energy:
                    best_entry = discovered
                if reward > best_score or best_discovered is None:
                    best_score = reward
                    best_reward = reward
                    best_discovered = discovered
            assert best_discovered is not None
            archive.record_success(seed_entry, best_reward)
            walk_history.append(
                WalkRecord(
                    seed_entry_id=seed_entry.entry_id,
                    discovered_entry_id=best_discovered.entry_id,
                    energy=best_discovered.energy,
                    accepted_new_basin=best_discovered.entry_id != seed_entry.entry_id,
                )
            )

        return SearchResult(
            best_state=best_entry.state,
            best_energy=best_entry.energy,
            archive=archive,
            walk_history=walk_history,
            stats={
                "n_trials": self.config.max_trials,
                "n_minima": len(archive.entries),
                "local_relaxations": local_relaxations,
                "duplicate_rate": archive.duplicate_rate(),
                "descriptor_degeneracy_rate": archive.descriptor_degeneracy_rate(),
                "coordinate_system": "cartesian_fixed_cell",
                "variable_cell_supported": 0,
            },
        )

    def _proposal_pool(self, seed_state: State, archive, trial_index: int) -> list[CandidateProposal]:
        return [CandidateProposal("ssw_walk", self._walk_candidate_from_seed(seed_state))]

    def _walk_candidate_from_seed(self, seed_state: State) -> State:
        current = seed_state
        previous_direction: np.ndarray | None = None
        biases: list[GaussianBiasTerm] = []
        softening = self._build_softening(seed_state)

        for _ in range(self.config.max_steps_per_walk):
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            choice = self.oracle.choose_direction(current, proposal, previous_direction)
            sigma = self._step_scale(choice.curvature)
            weight = self._bias_weight(choice.curvature, sigma)
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

    def _bias_weight(self, curvature: float, sigma: float) -> float:
        return float(sigma * sigma * max(curvature + self.config.target_negative_curvature, 0.0))

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
