from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from ase.calculators.lj import LennardJones

from pamssw import SSWConfig, State, relax_minimum, run_ssw
from pamssw.calculators import ASECalculator
from pamssw.config import RelaxConfig

CCD_GLOBAL_MINIMA = {
    13: -44.326801,
    38: -173.928427,
    55: -279.248470,
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


def random_cluster_state(size: int, seed: int) -> State:
    rng = np.random.default_rng(seed)
    radius = size ** (1.0 / 3.0)
    positions = rng.uniform(-radius, radius, size=(size, 3))
    return State(numbers=np.full(size, 18), positions=positions)


def make_calculator() -> ASECalculator:
    return ASECalculator(LennardJones())


def quench(state: State, calculator: ASECalculator, fmax: float = 1e-3, maxiter: int = 400):
    return relax_minimum(state, calculator, RelaxConfig(fmax=fmax, maxiter=maxiter))


def run_ssw_trial(size: int, seed: int, budget: int) -> RunSummary:
    steps_per_walk = 8
    proposal_pool_size = 3
    max_trials = max(1, (budget - 1) // proposal_pool_size)
    calculator = make_calculator()
    result = run_ssw(
        random_cluster_state(size, seed),
        calculator,
        SSWConfig(
            max_trials=max_trials,
            max_steps_per_walk=steps_per_walk,
            proposal_pool_size=proposal_pool_size,
            target_uphill_energy=1.2,
            quench_fmax=1e-3,
            dedup_rmsd_tol=0.2,
            rng_seed=seed,
        ),
    )
    target = CCD_GLOBAL_MINIMA[size]
    gap = result.best_energy - target
    return RunSummary(
        algorithm="ssw",
        size=size,
        seed=seed,
        best_energy=result.best_energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=int(result.stats.get("local_relaxations", 0)),
    )


def run_bh_trial(size: int, seed: int, budget: int) -> RunSummary:
    rng = np.random.default_rng(seed)
    calculator = make_calculator()
    current = quench(random_cluster_state(size, seed), calculator)
    best = current
    current_energy = current.energy
    temperature = 0.8
    step_scale = 0.35
    for _ in range(max(0, budget - 1)):
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
    target = CCD_GLOBAL_MINIMA[size]
    gap = best.energy - target
    return RunSummary(
        algorithm="bh",
        size=size,
        seed=seed,
        best_energy=best.energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=budget,
    )


def run_ga_trial(size: int, seed: int, budget: int) -> RunSummary:
    rng = np.random.default_rng(seed)
    calculator = make_calculator()
    population_size = min(10, max(4, budget // 4))
    population = []
    relaxations = 0
    for offset in range(population_size):
        relaxed = quench(random_cluster_state(size, seed * 100 + offset), calculator)
        relaxations += 1
        population.append(relaxed)
    population.sort(key=lambda item: item.energy)
    best = population[0]

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

    target = CCD_GLOBAL_MINIMA[size]
    gap = best.energy - target
    return RunSummary(
        algorithm="ga",
        size=size,
        seed=seed,
        best_energy=best.energy,
        energy_gap=gap,
        success=gap <= 1e-3,
        local_relaxations=relaxations,
    )


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SSW, BH, and GA on LJ cluster benchmarks.")
    parser.add_argument("--sizes", nargs="+", type=int, default=[13, 38, 55])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=60, help="Approximate local-relaxation budget per run.")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results_lj.json"))
    args = parser.parse_args()

    runs: list[RunSummary] = []
    for size in args.sizes:
        if size not in CCD_GLOBAL_MINIMA:
            raise ValueError(f"No CCD target energy registered for LJ{size}")
        for seed in args.seeds:
            runs.append(run_ssw_trial(size, seed, args.budget))
            runs.append(run_bh_trial(size, seed, args.budget))
            runs.append(run_ga_trial(size, seed, args.budget))

    payload = {
        "sizes": args.sizes,
        "seeds": args.seeds,
        "budget": args.budget,
        "ccd_global_minima": CCD_GLOBAL_MINIMA,
        "summary": aggregate(runs),
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
