from __future__ import annotations

import argparse
import json
import math
import os
import signal
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.lj import LennardJones
from ase.io import read
from ase.optimize import FIRE
from ase.optimize.basin import BasinHopping

from pamssw import SSWConfig, State, relax_minimum, run_ssw
from pamssw.calculators import ASECalculator
from pamssw.config import RelaxConfig
from pamssw.result import RelaxResult

CCD_GLOBAL_MINIMA = {
    13: -44.326801,
    38: -173.928427,
    55: -279.248470,
    75: -397.492331,
}


@dataclass
class RunSummary:
    algorithm: str
    size: int
    seed: int
    best_energy: float
    energy_gap: float
    success: bool
    local_relaxations: int
    force_evaluations: int | None = None
    budget_exhausted: bool | None = None
    n_minima: int | None = None
    duplicate_rate: float | None = None
    frontier_nodes: int | None = None
    dead_nodes: int | None = None
    mean_node_duplicate_failure_rate: float | None = None
    max_node_duplicate_failure_rate: float | None = None
    direction_choices: int | None = None
    direction_candidate_evaluations: int | None = None
    direction_mean_candidate_pool_size: float | None = None
    direction_rigid_body_overlap_mean: float | None = None
    direction_post_projection_rigid_body_overlap_mean: float | None = None
    direction_selected_soft: int | None = None
    direction_selected_random: int | None = None
    direction_selected_bond: int | None = None
    direction_selected_cell: int | None = None
    true_quench_count: int | None = None
    true_quench_unconverged: int | None = None
    true_quench_max_gradient: float | None = None
    true_quench_mean_iterations: float | None = None
    proposal_relax_count: int | None = None
    proposal_relax_unconverged: int | None = None
    proposal_relax_max_gradient: float | None = None
    proposal_relax_mean_iterations: float | None = None


@dataclass
class EnergyTrace:
    algorithm: str
    size: int
    seed: int
    points: list[dict[str, float | int]]


def random_cluster_state(size: int, seed: int) -> State:
    rng = np.random.RandomState(seed)
    radius = size ** (1.0 / 3.0)
    positions = rng.uniform(-radius, radius, size=(size, 3))
    return State(numbers=np.full(size, 18), positions=positions)


def make_calculator() -> ASECalculator:
    return ASECalculator(LennardJones())


def make_ase_lj_calculator() -> LennardJones:
    return LennardJones()


def state_to_atoms(state: State) -> Atoms:
    atoms = Atoms(numbers=state.numbers, positions=state.positions, cell=state.cell, pbc=state.pbc)
    atoms.calc = make_ase_lj_calculator()
    return atoms


def atoms_to_state(atoms: Atoms) -> State:
    return State(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=np.array(atoms.cell) if np.any(atoms.cell) else None,
        pbc=tuple(bool(x) for x in atoms.pbc),
    )


def attach_single_point(atoms: Atoms, energy: float) -> Atoms:
    stored = atoms.copy()
    stored.calc = SinglePointCalculator(stored, energy=float(energy))
    stored.info.setdefault("key_value_pairs", {})
    stored.info["key_value_pairs"]["raw_score"] = -float(energy)
    stored.info.setdefault("data", {})
    return stored


def atoms_energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except RuntimeError:
        raw_score = atoms.info.get("key_value_pairs", {}).get("raw_score")
        if raw_score is None:
            raise
        return -float(raw_score)


def set_cluster_cell(atoms: Atoms, box_length: float) -> Atoms:
    atoms.set_cell(np.eye(3) * box_length, scale_atoms=False)
    atoms.set_pbc(False)
    return atoms


class OperatorTimeout(RuntimeError):
    pass


def call_with_timeout(function, timeout_seconds: int):
    def _handler(signum, frame):
        raise OperatorTimeout(f"operator exceeded {timeout_seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_seconds)
    try:
        return function()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def quench(state: State, calculator: ASECalculator, fmax: float = 1e-3, maxiter: int = 400):
    return relax_minimum(state, calculator, RelaxConfig(fmax=fmax, maxiter=maxiter))


def ase_relax_atoms(atoms: Atoms, fmax: float = 1e-3, steps: int = 400) -> tuple[Atoms, float]:
    relaxed = atoms.copy()
    relaxed.calc = make_ase_lj_calculator()
    optimizer = FIRE(relaxed, logfile=os.devnull)
    optimizer.run(fmax=fmax, steps=steps)
    energy = float(relaxed.get_potential_energy())
    return attach_single_point(relaxed, energy), energy


def repo_relax_atoms(atoms: Atoms, fmax: float = 1e-3, maxiter: int = 400) -> tuple[Atoms, float]:
    result = quench(atoms_to_state(atoms), make_calculator(), fmax=fmax, maxiter=maxiter)
    relaxed = Atoms(
        numbers=result.state.numbers,
        positions=result.state.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    return attach_single_point(relaxed, result.energy), result.energy


def run_ssw_trial(
    size: int,
    seed: int,
    budget: int,
    steps_per_walk: int = 8,
    proposal_relax_steps: int = 40,
    minima_output_dir: Path | None = None,
) -> RunSummary:
    max_trials = max(1, budget - 1)
    calculator = make_calculator()
    result = run_ssw(
        random_cluster_state(size, seed),
        calculator,
        SSWConfig(
            max_trials=max_trials,
            max_steps_per_walk=steps_per_walk,
            target_uphill_energy=1.2,
            quench_fmax=1e-3,
            proposal_relax_steps=proposal_relax_steps,
            dedup_rmsd_tol=0.2,
            rng_seed=seed,
        ),
    )
    target = CCD_GLOBAL_MINIMA[size]
    gap = result.best_energy - target
    if minima_output_dir is not None:
        minima = [
            (entry.state, entry.energy, {"entry_id": entry.entry_id, "visits": entry.visits})
            for entry in result.archive.entries
        ]
        write_minima_xyz(minima_output_dir, "ssw", size, seed, minima)
    return RunSummary(
        algorithm="ssw",
        size=size,
        seed=seed,
        best_energy=result.best_energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=int(result.stats.get("local_relaxations", 0)),
        force_evaluations=int(result.stats.get("force_evaluations", 0)),
        budget_exhausted=bool(result.stats.get("budget_exhausted", 0)),
        n_minima=int(result.stats.get("n_minima", 0)),
        duplicate_rate=float(result.stats.get("duplicate_rate", 0.0)),
        frontier_nodes=int(result.stats.get("frontier_nodes", 0)),
        dead_nodes=int(result.stats.get("dead_nodes", 0)),
        mean_node_duplicate_failure_rate=float(result.stats.get("mean_node_duplicate_failure_rate", 0.0)),
        max_node_duplicate_failure_rate=float(result.stats.get("max_node_duplicate_failure_rate", 0.0)),
        direction_choices=int(result.stats.get("direction_choices", 0)),
        direction_candidate_evaluations=int(result.stats.get("direction_candidate_evaluations", 0)),
        direction_mean_candidate_pool_size=float(result.stats.get("direction_mean_candidate_pool_size", 0.0)),
        direction_rigid_body_overlap_mean=float(result.stats.get("direction_rigid_body_overlap_mean", 0.0)),
        direction_post_projection_rigid_body_overlap_mean=float(
            result.stats.get("direction_post_projection_rigid_body_overlap_mean", 0.0)
        ),
        direction_selected_soft=int(result.stats.get("direction_selected_soft", 0)),
        direction_selected_random=int(result.stats.get("direction_selected_random", 0)),
        direction_selected_bond=int(result.stats.get("direction_selected_bond", 0)),
        direction_selected_cell=int(result.stats.get("direction_selected_cell", 0)),
        true_quench_count=int(result.stats.get("true_quench_count", 0)),
        true_quench_unconverged=int(result.stats.get("true_quench_unconverged", 0)),
        true_quench_max_gradient=float(result.stats.get("true_quench_max_gradient", 0.0)),
        true_quench_mean_iterations=float(result.stats.get("true_quench_mean_iterations", 0.0)),
        proposal_relax_count=int(result.stats.get("proposal_relax_count", 0)),
        proposal_relax_unconverged=int(result.stats.get("proposal_relax_unconverged", 0)),
        proposal_relax_max_gradient=float(result.stats.get("proposal_relax_max_gradient", 0.0)),
        proposal_relax_mean_iterations=float(result.stats.get("proposal_relax_mean_iterations", 0.0)),
    )


def run_ssw_trace(
    size: int,
    seed: int,
    budget: int,
    steps_per_walk: int = 8,
    proposal_relax_steps: int = 40,
) -> EnergyTrace:
    max_trials = max(1, budget - 1)
    calculator = make_calculator()
    result = run_ssw(
        random_cluster_state(size, seed),
        calculator,
        SSWConfig(
            max_trials=max_trials,
            max_steps_per_walk=steps_per_walk,
            target_uphill_energy=1.2,
            quench_fmax=1e-3,
            proposal_relax_steps=proposal_relax_steps,
            dedup_rmsd_tol=0.2,
            rng_seed=seed,
        ),
    )
    target = CCD_GLOBAL_MINIMA[size]
    initial_energy = result.archive.entries[0].energy if result.archive.entries else result.best_energy
    best = initial_energy
    points = [_trace_point(1, best, target)]
    for index, record in enumerate(result.walk_history, start=2):
        best = min(best, record.energy)
        points.append(_trace_point(index, best, target))
    return EnergyTrace("ssw", size, seed, _pad_trace(points, budget))


def run_bh_trial(size: int, seed: int, budget: int, minima_output_dir: Path | None = None) -> RunSummary:
    rng = np.random.RandomState(seed)
    calculator = make_calculator()
    current = quench(random_cluster_state(size, seed), calculator)
    minima: list[tuple[State, float, dict[str, int | float]]] = [(current.state, current.energy, {"step": 1})]
    best = current
    current_energy = current.energy
    temperature, step_scale = bh_parameters(size)
    for step in range(2, budget + 1):
        displacement = rng.normal(scale=step_scale, size=current.state.positions.shape)
        trial_positions = current.state.positions + displacement
        trial_positions -= trial_positions.mean(axis=0, keepdims=True)
        trial_state = State(numbers=current.state.numbers, positions=trial_positions)
        trial = quench(trial_state, calculator)
        minima.append((trial.state, trial.energy, {"step": step}))
        delta = trial.energy - current_energy
        if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
            current = trial
            current_energy = trial.energy
        if trial.energy < best.energy:
            best = trial
    target = CCD_GLOBAL_MINIMA[size]
    gap = best.energy - target
    if minima_output_dir is not None:
        write_minima_xyz(minima_output_dir, "bh", size, seed, minima)
    return RunSummary(
        algorithm="bh",
        size=size,
        seed=seed,
        best_energy=best.energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=budget,
    )


def run_ase_bh_trial(size: int, seed: int, budget: int, minima_output_dir: Path | None = None) -> RunSummary:
    np.random.seed(seed)
    initial = state_to_atoms(random_cluster_state(size, seed))
    temperature, step_scale = bh_parameters(size)
    with tempfile.TemporaryDirectory(prefix=f"ase_bh_LJ{size}_seed{seed}_") as tmp:
        tmp_path = Path(tmp)
        local_minima_path = tmp_path / "local_minima.traj"
        lowest_path = tmp_path / "lowest.traj"
        bh = BasinHopping(
            initial,
            temperature=temperature,
            optimizer=FIRE,
            fmax=1e-3,
            dr=step_scale,
            logfile=os.devnull,
            trajectory=str(lowest_path),
            optimizer_logfile=os.devnull,
            local_minima_trajectory=str(local_minima_path),
            adjust_cm=True,
        )
        bh.run(max(0, budget - 1))
        best_energy, best_atoms = bh.get_minimum()
        minima_atoms = _read_traj_atoms(local_minima_path)

    target = CCD_GLOBAL_MINIMA[size]
    gap = float(best_energy) - target
    if minima_output_dir is not None:
        minima = [
            (atoms_to_state(atoms), atoms_energy(atoms), {"step": index + 1})
            for index, atoms in enumerate(minima_atoms)
        ]
        write_minima_xyz(minima_output_dir, "ase_bh", size, seed, minima)
    return RunSummary(
        algorithm="ase_bh",
        size=size,
        seed=seed,
        best_energy=float(best_energy),
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=len(minima_atoms),
        n_minima=len(minima_atoms),
    )


def run_ase_bh_trace(size: int, seed: int, budget: int) -> EnergyTrace:
    np.random.seed(seed)
    initial = state_to_atoms(random_cluster_state(size, seed))
    temperature, step_scale = bh_parameters(size)
    target = CCD_GLOBAL_MINIMA[size]
    with tempfile.TemporaryDirectory(prefix=f"ase_bh_trace_LJ{size}_seed{seed}_") as tmp:
        tmp_path = Path(tmp)
        local_minima_path = tmp_path / "local_minima.traj"
        bh = BasinHopping(
            initial,
            temperature=temperature,
            optimizer=FIRE,
            fmax=1e-3,
            dr=step_scale,
            logfile=os.devnull,
            trajectory=str(tmp_path / "lowest.traj"),
            optimizer_logfile=os.devnull,
            local_minima_trajectory=str(local_minima_path),
            adjust_cm=True,
        )
        bh.run(max(0, budget - 1))
        minima_atoms = _read_traj_atoms(local_minima_path)
    points: list[dict[str, float | int]] = []
    best = math.inf
    for index, atoms in enumerate(minima_atoms, start=1):
        best = min(best, atoms_energy(atoms))
        points.append(_trace_point(index, best, target))
    return EnergyTrace("ase_bh", size, seed, _pad_trace(points, budget))


def run_bh_trace(size: int, seed: int, budget: int) -> EnergyTrace:
    rng = np.random.RandomState(seed)
    calculator = make_calculator()
    current = quench(random_cluster_state(size, seed), calculator)
    best = current
    current_energy = current.energy
    target = CCD_GLOBAL_MINIMA[size]
    points = [_trace_point(1, best.energy, target)]
    temperature, step_scale = bh_parameters(size)
    for step in range(2, budget + 1):
        displacement = rng.normal(scale=step_scale, size=current.state.positions.shape)
        trial_positions = current.state.positions + displacement
        trial_positions -= trial_positions.mean(axis=0, keepdims=True)
        trial_state = State(numbers=current.state.numbers, positions=trial_positions)
        trial = quench(trial_state, calculator)
        delta = trial.energy - current_energy
        if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
            current = trial
            current_energy = trial.energy
        if trial.energy < best.energy:
            best = trial
        points.append(_trace_point(step, best.energy, target))
    return EnergyTrace("bh", size, seed, points)


def run_ga_trial(size: int, seed: int, budget: int, minima_output_dir: Path | None = None) -> RunSummary:
    rng = np.random.default_rng(seed)
    calculator = make_calculator()
    population_size = min(budget, 10, max(4, budget // 4))
    population = []
    minima: list[tuple[State, float, dict[str, int | float]]] = []
    relaxations = 0
    for offset in range(population_size):
        relaxed = quench(random_cluster_state(size, seed * 100 + offset), calculator)
        relaxations += 1
        minima.append((relaxed.state, relaxed.energy, {"step": relaxations, "initial_offset": offset}))
        population.append(relaxed)
    population.sort(key=lambda item: item.energy)
    best = population[0]

    while relaxations < budget:
        parent_a = tournament_select(population, rng)
        parent_b = tournament_select(population, rng)
        child_state = cut_and_splice(parent_a.state, parent_b.state, rng)
        child = quench(child_state, calculator)
        relaxations += 1
        minima.append((child.state, child.energy, {"step": relaxations}))
        population.append(child)
        population.sort(key=lambda item: item.energy)
        population = population[:population_size]
        if child.energy < best.energy:
            best = child

    target = CCD_GLOBAL_MINIMA[size]
    gap = best.energy - target
    if minima_output_dir is not None:
        write_minima_xyz(minima_output_dir, "ga", size, seed, minima)
    return RunSummary(
        algorithm="ga",
        size=size,
        seed=seed,
        best_energy=best.energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=relaxations,
    )


def run_ase_ga_trial(size: int, seed: int, budget: int, minima_output_dir: Path | None = None) -> RunSummary:
    minima, best_atoms, best_energy = _run_ase_ga(size, seed, budget)
    target = CCD_GLOBAL_MINIMA[size]
    gap = best_energy - target
    if minima_output_dir is not None:
        minima_states = [
            (atoms_to_state(atoms), atoms_energy(atoms), {"step": index + 1})
            for index, atoms in enumerate(minima)
        ]
        write_minima_xyz(minima_output_dir, "ase_ga", size, seed, minima_states)
    return RunSummary(
        algorithm="ase_ga",
        size=size,
        seed=seed,
        best_energy=best_energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=len(minima),
        n_minima=len(minima),
    )


def run_ase_ga_trace(size: int, seed: int, budget: int) -> EnergyTrace:
    minima, _, _ = _run_ase_ga(size, seed, budget)
    target = CCD_GLOBAL_MINIMA[size]
    points: list[dict[str, float | int]] = []
    best = math.inf
    for index, atoms in enumerate(minima, start=1):
        best = min(best, atoms_energy(atoms))
        points.append(_trace_point(index, best, target))
    return EnergyTrace("ase_ga", size, seed, _pad_trace(points, budget))


def run_ga_trace(size: int, seed: int, budget: int) -> EnergyTrace:
    rng = np.random.default_rng(seed)
    calculator = make_calculator()
    population_size = min(budget, 10, max(4, budget // 4))
    population = []
    target = CCD_GLOBAL_MINIMA[size]
    points: list[dict[str, float | int]] = []
    best = None
    relaxations = 0
    for offset in range(population_size):
        relaxed = quench(random_cluster_state(size, seed * 100 + offset), calculator)
        relaxations += 1
        population.append(relaxed)
        if best is None or relaxed.energy < best.energy:
            best = relaxed
        points.append(_trace_point(relaxations, best.energy, target))
    population.sort(key=lambda item: item.energy)
    assert best is not None

    while relaxations < budget:
        parent_a = tournament_select(population, rng)
        parent_b = tournament_select(population, rng)
        child_state = cut_and_splice(parent_a.state, parent_b.state, rng)
        child = quench(child_state, calculator)
        relaxations += 1
        population.append(child)
        population.sort(key=lambda item: item.energy)
        population = population[:population_size]
        if child.energy < best.energy:
            best = child
        points.append(_trace_point(relaxations, best.energy, target))
    return EnergyTrace("ga", size, seed, _pad_trace(points, budget))


def run_algorithm_traces(
    size: int,
    seeds: list[int],
    budget: int,
    ssw_steps_per_walk: int = 8,
    ssw_proposal_relax_steps: int = 40,
    baseline_source: str = "internal",
) -> list[EnergyTrace]:
    traces: list[EnergyTrace] = []
    for seed in seeds:
        traces.append(
            run_ssw_trace(
                size,
                seed,
                budget,
                steps_per_walk=ssw_steps_per_walk,
                proposal_relax_steps=ssw_proposal_relax_steps,
            )
        )
        if baseline_source == "ase":
            traces.append(run_ase_bh_trace(size, seed, budget))
            traces.append(run_ase_ga_trace(size, seed, budget))
        else:
            traces.append(run_bh_trace(size, seed, budget))
            traces.append(run_ga_trace(size, seed, budget))
    return traces


def run_all_trials(
    sizes: list[int],
    seeds: list[int],
    budget: int,
    ssw_steps_per_walk: int = 8,
    ssw_proposal_relax_steps: int = 40,
    minima_output_dir: Path | None = None,
    baseline_source: str = "internal",
) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for size in sizes:
        if size not in CCD_GLOBAL_MINIMA:
            raise ValueError(f"No CCD target energy registered for LJ{size}")
        for seed in seeds:
            runs.append(
                run_ssw_trial(
                    size,
                    seed,
                    budget,
                    steps_per_walk=ssw_steps_per_walk,
                    proposal_relax_steps=ssw_proposal_relax_steps,
                    minima_output_dir=minima_output_dir,
                )
            )
            if baseline_source == "ase":
                runs.append(run_ase_bh_trial(size, seed, budget, minima_output_dir=minima_output_dir))
                runs.append(run_ase_ga_trial(size, seed, budget, minima_output_dir=minima_output_dir))
            else:
                runs.append(run_bh_trial(size, seed, budget, minima_output_dir=minima_output_dir))
                runs.append(run_ga_trial(size, seed, budget, minima_output_dir=minima_output_dir))
    return runs


def _trace_point(step: int, best_energy: float, target: float) -> dict[str, float | int]:
    return {"step": int(step), "best_energy": float(best_energy), "energy_gap": float(best_energy - target)}


def _pad_trace(points: list[dict[str, float | int]], budget: int) -> list[dict[str, float | int]]:
    if not points:
        return points
    padded = list(points)
    last = dict(padded[-1])
    for step in range(int(last["step"]) + 1, budget + 1):
        filled = dict(last)
        filled["step"] = step
        padded.append(filled)
    return padded


def tournament_select(population, rng: np.random.Generator, tournament_size: int = 3):
    picks = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    return min((population[index] for index in picks), key=lambda item: item.energy)


def cut_and_splice(state_a: State, state_b: State, rng: np.random.Generator) -> State:
    coords_a = random_rotate(center_positions(state_a.positions), rng)
    coords_b = random_rotate(center_positions(state_b.positions), rng)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction) + 1e-12
    order_a = np.argsort(coords_a @ direction)
    order_b = np.argsort(coords_b @ direction)
    split = int(rng.integers(1, state_a.n_atoms))
    chosen_a = coords_a[order_a[:split]]
    chosen_b = coords_b[order_b[-(state_a.n_atoms - split) :]]
    child = np.vstack([chosen_a, chosen_b])
    child += rng.normal(scale=0.08, size=child.shape)
    child -= child.mean(axis=0, keepdims=True)
    return State(numbers=state_a.numbers.copy(), positions=child)


def center_positions(positions: np.ndarray) -> np.ndarray:
    return positions - positions.mean(axis=0, keepdims=True)


def random_rotate(positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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


def bh_parameters(size: int) -> tuple[float, float]:
    # Conservative lightweight baseline, not literature-tuned basin hopping.
    return 0.8, 0.35


def _read_traj_atoms(path: Path) -> list[Atoms]:
    if not path.exists():
        return []
    frames = read(str(path), ":")
    if isinstance(frames, Atoms):
        return [frames]
    return list(frames)


def _run_ase_ga(size: int, seed: int, budget: int) -> tuple[list[Atoms], Atoms, float]:
    try:
        from ase_ga.cutandsplicepairing import CutAndSplicePairing
        from ase_ga.data import DataConnection, PrepareDB
        from ase_ga.population import Population
        from ase_ga.standard_comparators import InteratomicDistanceComparator
        from ase_ga.startgenerator import StartGenerator
    except ImportError as exc:
        raise RuntimeError("ASE GA baseline requires the separate `ase-ga` package.") from exc

    rng = np.random.RandomState(seed)
    population_size = min(budget, 10, max(4, budget // 4))
    box_length = 2.4 * size ** (1.0 / 3.0)
    slab = Atoms("", cell=np.eye(3) * box_length, pbc=False)
    box = [
        np.array([-0.5 * box_length, -0.5 * box_length, -0.5 * box_length]),
        [np.array([box_length, 0.0, 0.0]), np.array([0.0, box_length, 0.0]), np.array([0.0, 0.0, box_length])],
    ]
    blmin = {(18, 18): 0.75}
    starter = StartGenerator(
        slab,
        [(18, size)],
        blmin,
        box_to_place_in=box,
        test_dist_to_slab=False,
        test_too_far=True,
        rng=rng,
    )
    pairing = CutAndSplicePairing(
        slab,
        size,
        blmin,
        test_dist_to_slab=False,
        use_tags=False,
        rng=rng,
    )
    comparator = InteratomicDistanceComparator(
        n_top=size,
        pair_cor_cum_diff=0.03,
        pair_cor_max=0.7,
        dE=0.02,
        mic=False,
    )

    minima: list[Atoms] = []
    best_atoms: Atoms | None = None
    best_energy = math.inf
    with tempfile.TemporaryDirectory(prefix=f"ase_ga_LJ{size}_seed{seed}_") as tmp:
        db_path = Path(tmp) / "ga.db"
        prep = PrepareDB(str(db_path), simulation_cell=slab)
        for index in range(population_size):
            candidate = starter.get_new_candidate(maxiter=10000)
            if candidate is None:
                candidate = state_to_atoms(random_cluster_state(size, seed * 100 + index))
            candidate = set_cluster_cell(candidate, box_length)
            relaxed, energy = repo_relax_atoms(candidate)
            relaxed = set_cluster_cell(relaxed, box_length)
            prep.add_relaxed_candidate(relaxed, description=f"initial {index}")
            minima.append(relaxed.copy())
            if energy < best_energy:
                best_energy = energy
                best_atoms = relaxed.copy()

        dc = DataConnection(str(db_path))
        population = Population(dc, population_size=population_size, comparator=comparator, rng=rng)
        relaxations = len(minima)
        while relaxations < budget:
            parents = population.get_two_candidates()
            if parents is None:
                candidate = starter.get_new_candidate(maxiter=10000)
                description = "mutation: random-start-fallback"
            else:
                try:
                    candidate, description = call_with_timeout(
                        lambda: pairing.get_new_individual(parents),
                        timeout_seconds=5,
                    )
                except OperatorTimeout:
                    candidate, description = None, "mutation: pairing-timeout-fallback"
            if candidate is None:
                candidate = starter.get_new_candidate(maxiter=10000)
                description = "mutation: random-start-fallback"
            if candidate is None:
                candidate = state_to_atoms(random_cluster_state(size, seed * 1000 + relaxations))
                description = "mutation: uniform-fallback"

            candidate = set_cluster_cell(candidate, box_length)
            candidate.info.setdefault("key_value_pairs", {})
            candidate.info.setdefault("data", {})
            dc.add_unrelaxed_candidate(candidate, description=description)
            relaxed, energy = repo_relax_atoms(candidate)
            relaxed = set_cluster_cell(relaxed, box_length)
            relaxed.info["confid"] = candidate.info["confid"]
            relaxed.info["data"] = dict(candidate.info.get("data", {}))
            relaxed.info.setdefault("key_value_pairs", {})
            relaxed.info["key_value_pairs"]["raw_score"] = -energy
            dc.add_relaxed_step(relaxed)
            population.update([relaxed])
            minima.append(relaxed.copy())
            relaxations += 1
            if energy < best_energy:
                best_energy = energy
                best_atoms = relaxed.copy()

    assert best_atoms is not None
    return minima, best_atoms, float(best_energy)


def write_minima_xyz(
    output_dir: Path,
    algorithm: str,
    size: int,
    seed: int,
    minima: list[tuple[State, float, dict[str, int | float]]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{algorithm}_LJ{size}_seed{seed}_minima.xyz"
    lines: list[str] = []
    for index, (state, energy, metadata) in enumerate(minima):
        lines.append(str(state.n_atoms))
        fields = [
            f"algorithm={algorithm}",
            f"size={size}",
            f"seed={seed}",
            f"index={index}",
            f"energy={energy:.16g}",
        ]
        fields.extend(f"{key}={value}" for key, value in sorted(metadata.items()))
        lines.append(" ".join(fields))
        for number, position in zip(state.numbers, state.positions):
            symbol = "Ar" if int(number) == 18 else f"X{int(number)}"
            lines.append(f"{symbol} {position[0]:.16g} {position[1]:.16g} {position[2]:.16g}")
    path.write_text("\n".join(lines) + "\n")
    return path


def aggregate(runs: list[RunSummary]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[RunSummary]] = {}
    for run in runs:
        grouped.setdefault((run.algorithm, run.size), []).append(run)
    summary: dict[str, object] = {}
    for (algorithm, size), items in sorted(grouped.items()):
        key = f"{algorithm}_LJ{size}"
        summary[key] = {
            "success_rate": sum(item.success for item in items) / len(items),
            "best_energy": min(item.best_energy for item in items),
            "mean_energy_gap": sum(item.energy_gap for item in items) / len(items),
            "runs": [asdict(item) for item in items],
        }
    return summary


def write_trace_plot(traces: list[EnergyTrace], output: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    sizes = sorted({trace.size for trace in traces})
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4), squeeze=False)
    colors = {"ssw": "#1f77b4", "bh": "#ff7f0e", "ga": "#2ca02c"}
    colors.update({"ase_bh": "#d62728", "ase_ga": "#9467bd"})
    for axis, size in zip(axes[0], sizes):
        for algorithm in ["ssw", "bh", "ga", "ase_bh", "ase_ga"]:
            selected = [trace for trace in traces if trace.size == size and trace.algorithm == algorithm]
            if not selected:
                continue
            max_step = max(int(point["step"]) for trace in selected for point in trace.points)
            means = []
            steps = list(range(1, max_step + 1))
            for step in steps:
                values = []
                for trace in selected:
                    point = trace.points[min(step, len(trace.points)) - 1]
                    values.append(float(point["energy_gap"]))
                means.append(float(np.mean(values)))
            axis.plot(steps, means, label=algorithm.upper(), color=colors[algorithm], linewidth=2)
        axis.set_title(f"LJ{size}")
        axis.set_xlabel("local relaxations / generation step")
        axis.set_ylabel("best energy gap")
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SSW, BH, and GA on LJ cluster benchmarks.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[13, 38, 55])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=60, help="Approximate local-relaxation budget per run.")
    parser.add_argument("--ssw-steps-per-walk", type=int, default=8)
    parser.add_argument("--ssw-proposal-relax-steps", type=int, default=40)
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--minima-output-dir", type=Path, default=None)
    parser.add_argument("--baseline-source", choices=["internal", "ase"], default="internal")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results_lj.json"))
    args = parser.parse_args()

    runs = run_all_trials(
        sizes=args.sizes,
        seeds=args.seeds,
        budget=args.budget,
        ssw_steps_per_walk=args.ssw_steps_per_walk,
        ssw_proposal_relax_steps=args.ssw_proposal_relax_steps,
        minima_output_dir=args.minima_output_dir,
        baseline_source=args.baseline_source,
    )

    payload = {
        "sizes": args.sizes,
        "seeds": args.seeds,
        "budget": args.budget,
        "ssw_steps_per_walk": args.ssw_steps_per_walk,
        "ssw_proposal_relax_steps": args.ssw_proposal_relax_steps,
        "baseline_source": args.baseline_source,
        "minima_output_dir": str(args.minima_output_dir) if args.minima_output_dir is not None else None,
        "ccd_global_minima": CCD_GLOBAL_MINIMA,
        "summary": aggregate(runs),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    if args.trace_output is not None or args.plot_output is not None:
        traces: list[EnergyTrace] = []
        for size in args.sizes:
            traces.extend(
                run_algorithm_traces(
                    size,
                    args.seeds,
                    args.budget,
                    ssw_steps_per_walk=args.ssw_steps_per_walk,
                    ssw_proposal_relax_steps=args.ssw_proposal_relax_steps,
                    baseline_source=args.baseline_source,
                )
            )
        trace_payload = {"traces": [asdict(trace) for trace in traces]}
        if args.trace_output is not None:
            args.trace_output.write_text(json.dumps(trace_payload, indent=2))
        if args.plot_output is not None:
            write_trace_plot(traces, args.plot_output)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
