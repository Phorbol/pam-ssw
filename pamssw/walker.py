from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bias import GaussianBiasTerm
from .config import LSSSWConfig, RelaxConfig, SSWConfig
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
        flat_size = state.flatten_positions().shape[0]
        best_direction: np.ndarray | None = None
        best_curvature: float | None = None
        candidate_vectors: list[np.ndarray] = []
        if previous_direction is not None:
            candidate_vectors.append(previous_direction)
        for _ in range(self.candidates):
            vector = np.zeros(flat_size, dtype=float)
            active = self.rng.normal(size=state.flatten_active().shape[0])
            active /= np.linalg.norm(active) + 1e-12
            vector.reshape(state.n_atoms, 3)[state.movable_mask] = active.reshape(-1, 3)
            candidate_vectors.append(vector)
        for vector in candidate_vectors:
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            curvature = self._directional_curvature(state, proposal, vector)
            if best_curvature is None or curvature < best_curvature:
                best_curvature = curvature
                best_direction = vector
        assert best_direction is not None and best_curvature is not None

        base_direction = candidate_vectors[-1] / (np.linalg.norm(candidate_vectors[-1]) + 1e-12)
        mixed_direction = 0.7 * base_direction + 0.3 * best_direction
        mixed_direction /= np.linalg.norm(mixed_direction) + 1e-12
        mixed_curvature = self._directional_curvature(state, proposal, mixed_direction)
        return DirectionChoice(mixed_direction, mixed_curvature)

    def _directional_curvature(
        self,
        state: State,
        proposal: ProposalPotential,
        direction: np.ndarray,
        epsilon: float = 1e-3,
    ) -> float:
        plus = state.displaced(direction, epsilon)
        minus = state.displaced(direction, -epsilon)
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

        for trial_index in range(self.config.max_trials):
            seed_entry = archive.next_seed()
            if self._should_cluster_reseed(seed_entry.state, trial_index):
                candidate = self.relax_true_minimum(self._random_cluster_state_like(seed_entry.state))
            else:
                candidate = self._walk_from_seed(seed_entry.state)
            discovered = archive.add(candidate.state, candidate.energy, parent_id=seed_entry.entry_id)
            if discovered.energy < best_entry.energy:
                best_entry = discovered
            walk_history.append(
                WalkRecord(
                    seed_entry_id=seed_entry.entry_id,
                    discovered_entry_id=discovered.entry_id,
                    energy=discovered.energy,
                    accepted_new_basin=discovered.entry_id != seed_entry.entry_id,
                )
            )

        return SearchResult(
            best_state=best_entry.state,
            best_energy=best_entry.energy,
            archive=archive,
            walk_history=walk_history,
            stats={"n_trials": self.config.max_trials, "n_minima": len(archive.entries)},
        )

    def _walk_from_seed(self, seed_state: State) -> RelaxResult:
        current = seed_state
        previous_direction: np.ndarray | None = None
        biases: list[GaussianBiasTerm] = []
        softening = self._build_softening(seed_state)

        for _ in range(self.config.max_steps_per_walk):
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            choice = self.oracle.choose_direction(current, proposal, previous_direction)
            sigma = self._step_scale(choice.curvature)
            weight = max(0.05, sigma * sigma * max(choice.curvature + 0.25, 0.05))
            biases.append(
                GaussianBiasTerm(
                    center=current.flatten_positions(),
                    direction=choice.direction,
                    sigma=sigma,
                    weight=weight,
                )
            )
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            trial_state = current.displaced(choice.direction, sigma)
            proposal_relax = Relaxer(proposal.evaluate).relax(
                trial_state,
                fmax=self.config.proposal_fmax,
                maxiter=self.config.proposal_relax_steps,
            )
            displacement = proposal_relax.state.flatten_positions() - current.flatten_positions()
            if np.linalg.norm(displacement) > 1e-8:
                previous_direction = displacement / np.linalg.norm(displacement)
            current = proposal_relax.state

        return self.relax_true_minimum(current)

    def _step_scale(self, curvature: float) -> float:
        effective = max(abs(curvature), 1e-4)
        sigma = np.sqrt(2.0 * self.config.target_uphill_energy / effective)
        return float(np.clip(sigma, self.config.min_step_scale, self.config.max_step_scale))

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

    def _should_cluster_reseed(self, state: State, trial_index: int) -> bool:
        if self.config.cluster_reseed_interval <= 0:
            return False
        if state.n_atoms < 4 or state.cell is not None or any(state.pbc):
            return False
        return (trial_index + 1) % self.config.cluster_reseed_interval == 0

    def _random_cluster_state_like(self, state: State) -> State:
        positions = self._random_compact_positions(state.n_atoms)
        return State(numbers=state.numbers.copy(), positions=positions)

    def _random_compact_positions(self, n_atoms: int) -> np.ndarray:
        radius = 0.75 * n_atoms ** (1.0 / 3.0)
        min_distance = 0.72
        positions: list[np.ndarray] = []
        attempts = 0
        while len(positions) < n_atoms and attempts < 5000:
            attempts += 1
            point = self.rng.normal(size=3)
            point /= np.linalg.norm(point) + 1e-12
            point *= radius * self.rng.random() ** (1.0 / 3.0)
            if all(np.linalg.norm(point - existing) >= min_distance for existing in positions):
                positions.append(point)
        if len(positions) < n_atoms:
            positions = [self.rng.normal(scale=radius / 2.0, size=3) for _ in range(n_atoms)]
        array = np.asarray(positions, dtype=float)
        return array - array.mean(axis=0, keepdims=True)
