"""Paired posterior-policy comparison on the analytic double-well potential."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.exploration import PosteriorExplorationConfig, run_posterior_ssw
from pamssw.exploration.policies import SUPPORTED_POLICIES
from pamssw.potentials import DoubleWell2D
from pamssw.state import State


SCHEMA_VERSION = 1
CALCULATOR_LABEL = "analytic-double-well-2d-v1"
POTENTIAL_PARAMETERS = {
    "energy_expression": "(x^2 - 1)^2 + 0.5*y^2 + 0.25*z^2",
    "x_well_positions": [-1.0, 1.0],
    "y_quadratic_coefficient": 0.5,
    "z_quadratic_coefficient": 0.25,
}


def _initial_state() -> State:
    return State(
        numbers=np.array([1], dtype=int),
        positions=np.array([[-0.8, 0.0, 0.0]], dtype=float),
    )


def _calculator_factory() -> AnalyticCalculator:
    return AnalyticCalculator(DoubleWell2D())


def _ssw_config() -> SSWConfig:
    return SSWConfig(
        rng_seed=0,
        max_trials=1,
        max_steps_per_walk=2,
        oracle_candidates=2,
        proposal_relax_steps=3,
        quench_maxiter=40,
        proposal_pool_size=1,
        max_force_evals=None,
    )


def _preflight_output_root(output_root: Path) -> Path:
    output_root = Path(output_root)
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output_root already exists: {output_root}")
    parent = output_root.parent
    if not parent.exists():
        raise FileNotFoundError(f"output_root parent does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"output_root parent is not a directory: {parent}")
    return output_root


def _validated_seeds(seeds: object) -> tuple[int, ...]:
    try:
        seeds = tuple(seeds)
    except TypeError as error:
        raise TypeError("seeds must be an iterable of integers") from error
    if not seeds:
        raise ValueError("seeds must be nonempty")
    seen: set[int] = set()
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seeds must contain integers")
        if seed < 0:
            raise ValueError("seeds must contain nonnegative integers")
        if seed in seen:
            raise ValueError("seeds must be unique")
        seen.add(seed)
    return seeds


def _validated_policies(policies: object) -> tuple[str, ...]:
    try:
        policies = tuple(policies)
    except TypeError as error:
        raise TypeError("policies must be an iterable of strings") from error
    if not policies:
        raise ValueError("policies must be nonempty")
    seen: set[str] = set()
    for policy in policies:
        if not isinstance(policy, str):
            raise TypeError("policies must contain strings")
        if policy not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported policy: {policy}")
        if policy in seen:
            raise ValueError("policies must be unique")
        seen.add(policy)
    return policies


def _preflight_exploration_configs(
    *,
    output_root: Path,
    seeds: object,
    policies: object,
    total_force_budget: int,
    action_force_budget: int,
    batch_size: int,
    max_workers: int,
) -> tuple[tuple[int, str, PosteriorExplorationConfig], ...]:
    validated_seeds = _validated_seeds(seeds)
    validated_policies = _validated_policies(policies)
    planned: list[tuple[int, str, PosteriorExplorationConfig]] = []
    for seed in validated_seeds:
        for policy in validated_policies:
            exploration_config = PosteriorExplorationConfig(
                policy_name=policy,
                batch_size=batch_size,
                max_workers=max_workers,
                action_force_budget=action_force_budget,
                total_force_budget=total_force_budget,
                master_seed=seed,
                run_directory=output_root / f"seed-{seed:08d}-{policy}",
            )
            planned.append((seed, policy, exploration_config))
    return tuple(planned)


def _record(result, *, seed: int, policy: str, exploration_config: PosteriorExplorationConfig) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "calculator_label": CALCULATOR_LABEL,
        "potential_parameters": POTENTIAL_PARAMETERS,
        "seed": seed,
        "policy": policy,
        "batch_size": exploration_config.batch_size,
        "max_workers": exploration_config.max_workers,
        "action_force_budget": exploration_config.action_force_budget,
        "total_force_budget": exploration_config.total_force_budget,
        "best_energy": min(entry.energy for entry in result.archive.entries),
        "unique_minima": len(result.archive.entries),
        "completed_batches": result.completed_batches,
        "completed_attempts": result.completed_attempts,
        "failed_attempts": result.failed_attempts,
        "posterior_observed_attempts": result.posterior_observed_attempts,
        "bootstrap_evaluations": result.bootstrap_evaluations,
        "action_evaluations": result.action_evaluations,
        "total_evaluations": result.total_evaluations,
        "purpose_counts": result.purpose_counts.as_dict(),
        "unused_force_budget": result.unused_force_budget,
        "stop_reason": result.stop_reason.value,
        "benchmark_eligible": result.benchmark_eligible,
        "benchmark_ineligibility_reasons": list(result.benchmark_ineligibility_reasons),
    }


def run_comparison(
    *,
    output_root: Path,
    seeds: tuple[int, ...],
    policies: tuple[str, ...] = ("uniform", "posterior_proportional", "minimal_ucb"),
    total_force_budget: int,
    action_force_budget: int,
    batch_size: int,
    max_workers: int,
) -> tuple[dict[str, object], ...]:
    output_root = Path(output_root)
    planned_configs = _preflight_exploration_configs(
        output_root=output_root,
        seeds=seeds,
        policies=policies,
        total_force_budget=total_force_budget,
        action_force_budget=action_force_budget,
        batch_size=batch_size,
        max_workers=max_workers,
    )
    output_root = _preflight_output_root(output_root)
    output_root.mkdir()
    records: list[dict[str, object]] = []
    for seed, policy, exploration_config in planned_configs:
        result = run_posterior_ssw(
            _initial_state(),
            _calculator_factory,
            _ssw_config(),
            exploration_config,
        )
        records.append(
            _record(
                result,
                seed=seed,
                policy=policy,
                exploration_config=exploration_config,
            )
        )
    return tuple(records)


def _preflight_output(output: Path) -> Path:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    parent = output.parent
    if not parent.exists():
        raise FileNotFoundError(f"output parent does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {parent}")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--total-force-budget", type=int, required=True)
    parser.add_argument("--action-force-budget", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-workers", type=int, required=True)
    args = parser.parse_args(argv)

    output = _preflight_output(args.output)
    output_root = output.parent / f"{output.stem}-runs"
    records = run_comparison(
        output_root=output_root,
        seeds=tuple(args.seeds),
        total_force_budget=args.total_force_budget,
        action_force_budget=args.action_force_budget,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(
            {"schema_version": SCHEMA_VERSION, "records": records},
            stream,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
