from __future__ import annotations

import argparse
import json
import math
import os
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
    rng = np.random.default_rng(seed)
    radius = size ** (1.0 / 3.0)
    positions = rng.uniform(-radius, radius, size=(size, 3))
    return State(numbers=np.full(size, 18), positions=positions)


def make_calculator() -> ASECalculator:
    return ASECalculator(LennardJones())


def quench(state: State, calculator: ASECalculator, fmax: float = 1e-3, maxiter: int = 400):
    return relax_minimum(state, calculator, RelaxConfig(fmax=fmax, maxiter=maxiter))


def run_ssw_trial(
    size: int,
    seed: int,
    budget: int,
    steps_per_walk: int = 8,
    proposal_relax_steps: int = 40,
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


def run_bh_trace(size: int, seed: int, budget: int) -> EnergyTrace:
    rng = np.random.default_rng(seed)
    calculator = make_calculator()
    current = quench(random_cluster_state(size, seed), calculator)
    best = current
    current_energy = current.energy
    target = CCD_GLOBAL_MINIMA[size]
    points = [_trace_point(1, best.energy, target)]
    temperature = 0.8
    step_scale = 0.35
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


def run_ga_trial(size: int, seed: int, budget: int) -> RunSummary:
    rng = np.random.default_rng(seed)
    calculator = make_calculator()
    population_size = min(budget, 10, max(4, budget // 4))
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
        traces.append(run_bh_trace(size, seed, budget))
        traces.append(run_ga_trace(size, seed, budget))
    return traces


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
    for axis, size in zip(axes[0], sizes):
        for algorithm in ["ssw", "bh", "ga"]:
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
    parser.add_argument("--output", type=Path, default=Path("benchmark_results_lj.json"))
    args = parser.parse_args()

    runs: list[RunSummary] = []
    for size in args.sizes:
        if size not in CCD_GLOBAL_MINIMA:
            raise ValueError(f"No CCD target energy registered for LJ{size}")
        for seed in args.seeds:
            runs.append(
                run_ssw_trial(
                    size,
                    seed,
                    args.budget,
                    steps_per_walk=args.ssw_steps_per_walk,
                    proposal_relax_steps=args.ssw_proposal_relax_steps,
                )
            )
            runs.append(run_bh_trial(size, seed, args.budget))
            runs.append(run_ga_trial(size, seed, args.budget))

    payload = {
        "sizes": args.sizes,
        "seeds": args.seeds,
        "budget": args.budget,
        "ssw_steps_per_walk": args.ssw_steps_per_walk,
        "ssw_proposal_relax_steps": args.ssw_proposal_relax_steps,
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
