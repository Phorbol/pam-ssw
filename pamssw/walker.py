from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acquisition import BanditSelector, ProposalOutcome, ProposalScorer
from .bias import GaussianBiasTerm
from .config import LSSSWConfig, RelaxConfig, SSWConfig
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
        self.proposal_scorer = ProposalScorer()
        self.selector = BanditSelector(
            archive_density_weight=config.archive_density_weight,
            novelty_weight=config.novelty_weight,
            frontier_weight=config.frontier_weight,
            exploration_weight=config.bandit_exploration_weight,
            baseline_probability=config.baseline_selection_probability,
        )

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
                "proposal_pool_size": self.config.proposal_pool_size,
                "local_relaxations": local_relaxations,
                "duplicate_rate": archive.duplicate_rate(),
                "descriptor_degeneracy_rate": archive.descriptor_degeneracy_rate(),
            },
        )

    def _proposal_pool(self, seed_state: State, archive, trial_index: int) -> list[CandidateProposal]:
        proposals = [CandidateProposal("ssw_walk", self._walk_candidate_from_seed(seed_state))]
        if seed_state.n_atoms < 4 or seed_state.cell is not None or any(seed_state.pbc):
            return proposals
        while len(proposals) < self.config.proposal_pool_size:
            label = self._next_proposal_label(len(proposals), archive, trial_index)
            if label == "compact_reseed":
                state = self._random_cluster_state_like(seed_state)
            elif label == "archive_novel_reseed":
                state = self._archive_novel_reseed(seed_state, archive)
            elif label == "surface_relocation":
                state = self._surface_relocation(seed_state)
            elif label == "graph_recombination":
                state = self._graph_recombination(seed_state, archive)
            else:
                state = self._random_cluster_state_like(seed_state)
            proposals.append(CandidateProposal(label, state))
        return proposals[: self.config.proposal_pool_size]

    def _next_proposal_label(self, slot: int, archive, trial_index: int) -> str:
        if self._should_cluster_reseed(archive.entries[0].state, trial_index) or slot == 1:
            return "compact_reseed"
        if slot == 2:
            return "archive_novel_reseed"
        if slot == 3:
            return "surface_relocation"
        if len(archive.entries) >= 2:
            return "graph_recombination"
        return "compact_reseed"

    def _walk_candidate_from_seed(self, seed_state: State) -> State:
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
        return current

    def _walk_from_seed(self, seed_state: State) -> RelaxResult:
        return self.relax_true_minimum(self._walk_candidate_from_seed(seed_state))

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

    def _archive_novel_reseed(self, state: State, archive) -> State:
        best_state = self._motif_cluster_state_like(state) or self._random_cluster_state_like(state)
        best_score = -float("inf")
        descriptor = structural_descriptor(best_state)
        best_score = archive.coverage_gain(descriptor)
        for _ in range(6):
            candidate = self._random_cluster_state_like(state)
            descriptor = structural_descriptor(candidate)
            score = archive.coverage_gain(descriptor)
            if score > best_score:
                best_score = score
                best_state = candidate
        return best_state

    def _motif_cluster_state_like(self, state: State) -> State | None:
        if state.cell is not None or any(state.pbc):
            return None
        try:
            if state.n_atoms == 38:
                from ase.cluster import Octahedron

                atoms = Octahedron("Ar", length=4, cutoff=1)
            elif state.n_atoms in {13, 55}:
                from ase.cluster import Icosahedron

                atoms = Icosahedron("Ar", 2 if state.n_atoms == 13 else 3)
            else:
                return None
        except Exception:
            return None
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if positions.shape != state.positions.shape:
            return None
        positions -= positions.mean(axis=0, keepdims=True)
        min_distance = self._min_pair_distance(positions)
        if min_distance <= 1e-12:
            return None
        positions *= 1.12 / min_distance
        positions = self._random_rotate(positions, self.rng)
        positions -= positions.mean(axis=0, keepdims=True)
        return State(numbers=state.numbers.copy(), positions=positions)

    def _surface_relocation(self, state: State) -> State:
        if state.n_atoms < 4 or state.cell is not None or any(state.pbc):
            return self._random_cluster_state_like(state)
        positions = state.positions - state.positions.mean(axis=0, keepdims=True)
        distances = np.linalg.norm(positions, axis=1)
        moved_index = int(np.argmax(distances))
        compact = self._random_compact_positions(state.n_atoms)
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-12
        compact[moved_index] = direction * max(1.0, np.percentile(np.linalg.norm(compact, axis=1), 75))
        compact -= compact.mean(axis=0, keepdims=True)
        if self._min_pair_distance(compact) < 0.55:
            return self._random_cluster_state_like(state)
        return State(numbers=state.numbers.copy(), positions=compact)

    def _graph_recombination(self, state: State, archive) -> State:
        if len(archive.entries) < 2 or state.n_atoms < 2:
            return self._random_cluster_state_like(state)
        parents = sorted(archive.entries, key=lambda entry: (entry.energy, entry.entry_id))[: min(6, len(archive.entries))]
        first = parents[int(self.rng.integers(0, len(parents)))]
        diverse = max(
            parents,
            key=lambda entry: np.linalg.norm(
                (first.descriptor if first.descriptor is not None else np.zeros(1))
                - (entry.descriptor if entry.descriptor is not None else np.zeros(1))
            ),
        )
        coords_a = self._random_rotate(first.state.positions - first.state.positions.mean(axis=0), self.rng)
        coords_b = self._random_rotate(diverse.state.positions - diverse.state.positions.mean(axis=0), self.rng)
        direction = self.rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-12
        order_a = np.argsort(coords_a @ direction)
        order_b = np.argsort(coords_b @ direction)
        split = int(self.rng.integers(1, state.n_atoms))
        child = np.vstack([coords_a[order_a[:split]], coords_b[order_b[-(state.n_atoms - split) :]]])
        child += self.rng.normal(scale=0.05, size=child.shape)
        child -= child.mean(axis=0, keepdims=True)
        if self._min_pair_distance(child) < 0.55:
            return self._random_cluster_state_like(state)
        return State(numbers=state.numbers.copy(), positions=child)

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

    @staticmethod
    def _min_pair_distance(positions: np.ndarray) -> float:
        if positions.shape[0] < 2:
            return float("inf")
        best = float("inf")
        for atom_i in range(positions.shape[0]):
            delta = positions[atom_i + 1 :] - positions[atom_i]
            if delta.size:
                best = min(best, float(np.linalg.norm(delta, axis=1).min()))
        return best

    @staticmethod
    def _random_rotate(positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        q = rng.normal(size=4)
        q /= np.linalg.norm(q) + 1e-12
        w, x, y, z = q
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return positions @ rotation.T
