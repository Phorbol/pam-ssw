#!/usr/bin/env python3
"""Run one frozen 200-step direction-transport production campaign."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FROZEN_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
SEEDS = (42,)
MAX_TRIALS = 200
ARMS: dict[str, dict[str, object]] = {
    "fixed_intent_ritz": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
    "transported_direction": {
        "direction_selection_mode": "transported_direction",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
}


def case_matrix() -> list[dict[str, Any]]:
    return [
        {"system": system, "seed": seed, "arm": arm}
        for system in SYSTEMS
        for seed in SEEDS
        for arm in ARMS
    ]


def _load_module(path: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _base_runner():
    return _load_module(
        FROZEN_RUNNER_PATH,
        "_direction_transport_long_frozen_runner",
    )


def build_config(base_runner, system: str, arm: str, seed: int, case_dir: Path):
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return replace(
        base_runner.build_config(system, case_dir),
        max_trials=MAX_TRIALS,
        max_force_evals=None,
        rng_seed=seed,
        quench_optimizer="ase-lbfgs",
        quench_fallback_optimizer="ase-fire",
        quench_fmax=0.01,
        **ARMS[arm],
    )


def _energy_trace(result) -> list[dict[str, Any]]:
    initial = float(result.archive.entries[0].energy)
    best = initial
    rows = [
        {
            "recorded_walk": 0,
            "energy_eV": initial,
            "best_energy_eV": initial,
            "best_energy_improvement_eV": 0.0,
        }
    ]
    for index, record in enumerate(result.walk_history, start=1):
        best = min(best, float(record.energy))
        rows.append(
            {
                "recorded_walk": index,
                "energy_eV": float(record.energy),
                "best_energy_eV": best,
                "best_energy_improvement_eV": initial - best,
            }
        )
    return rows


def _mean_best_improvement(
    energy_trace: Sequence[dict[str, Any]],
    *,
    completed_trials: int,
) -> float | None:
    if len(energy_trace) != completed_trials + 1:
        return None
    return float(
        sum(float(row["best_energy_improvement_eV"]) for row in energy_trace)
        / len(energy_trace)
    )


def run(
    *,
    system: str,
    arm: str,
    seed: int,
    output_dir: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    from pamssw.calculators import ASECalculator
    from pamssw.walker import SurfaceWalker

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    base = _base_runner()
    provenance = dict(
        base.preflight(
            system=system,
            expected_git_commit=expected_git_commit,
        )
    )
    output_dir.mkdir(parents=True)
    config = build_config(base, system, arm, seed, output_dir)
    state = base.load_state(system)
    walker = SurfaceWalker(
        calculator=ASECalculator(base._calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    wall_time = float(perf_counter() - started)

    counts = walker.calculator.snapshot()
    purposes = counts.as_dict()
    completed_trials = int(result.stats["n_trials"])
    if (
        completed_trials != MAX_TRIALS
        or int(result.stats["force_evaluations"]) != counts.total
        or sum(purposes.values()) != counts.total
        or purposes["unattributed"] != 0
    ):
        raise RuntimeError("long-campaign completion or purpose ledger failed")
    trace = _energy_trace(result)
    diagnostics = walker.relaxation_diagnostics()
    initial_energy = float(result.archive.entries[0].energy)
    summary = {
        **provenance,
        "schema_version": 1,
        "system": system,
        "seed": seed,
        "arm": arm,
        "completed_trials": completed_trials,
        "recorded_walks": len(result.walk_history),
        "trajectory_complete": (
            len(result.walk_history) == completed_trials
        ),
        "initial_energy_eV": initial_energy,
        "best_energy_eV": float(result.best_energy),
        "best_energy_drop_eV": initial_energy - float(result.best_energy),
        "mean_best_energy_improvement_eV": _mean_best_improvement(
            trace,
            completed_trials=completed_trials,
        ),
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.stats["duplicate_rate"]),
        "force_evaluations": counts.total,
        "purpose_counts": purposes,
        "wall_time_s": wall_time,
        "optimizer_telemetry": diagnostics,
        "stats": result.stats,
        "effective_config": asdict(config),
    }
    base.write_state(output_dir / "best_minimum.xyz", result.best_state)
    base._write_json(output_dir / "energy_trace.json", trace)
    base._write_json(output_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                system=args.system,
                arm=args.arm,
                seed=args.seed,
                output_dir=args.output,
                expected_git_commit=args.expected_git_commit,
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
