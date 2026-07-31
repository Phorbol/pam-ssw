#!/usr/bin/env python3
"""Matched C60/PdO gate for uniform, archive-UCB-like, and Metropolis starters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.mace_batch import MACEBatchCalculator
from pamssw.walker import SurfaceWalker
from pamssw.io import write_state


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_GATE = (
    RUN_ROOT.parent / "20260730-starter-cell-online-gate" / "run_gate.py"
)
SYSTEMS = ("c60", "pdo")
STARTER_MODES = ("uniform_archive", "archive_ucb", "metropolis_chain")
MAX_TRIALS = 10_000
DEFAULT_FORCE_BUDGET = 20_000
_SOURCE_MODULE_NAME = "_starter_selection_source_gate"


def _load_source_gate():
    module = sys.modules.get(_SOURCE_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_SOURCE_MODULE_NAME, SOURCE_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen source gate: {SOURCE_GATE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_config(
    system: str,
    case_directory: Path,
    *,
    seed: int,
    starter_mode: str,
    force_budget: int,
):
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS!r}")
    if starter_mode not in STARTER_MODES:
        raise ValueError(f"starter_mode must be one of {STARTER_MODES!r}")
    if isinstance(force_budget, bool) or not isinstance(force_budget, int):
        raise ValueError("force_budget must be a positive integer")
    if force_budget <= 0:
        raise ValueError("force_budget must be a positive integer")
    source_gate = _load_source_gate()
    frozen = source_gate.build_production_config(
        system,
        Path(case_directory),
        master_seed=seed,
    )
    return replace(
        frozen,
        max_trials=MAX_TRIALS,
        max_force_evals=force_budget,
        rng_seed=seed,
        seed_selection_mode=starter_mode,
    )


def _preflight(expected_commit: str) -> dict[str, Any]:
    import mace
    import torch

    actual_commit = _git_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    source_gate = _load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    inputs = {
        system: {
            "path": str(production.INPUT_PATHS[system]),
            "sha256": _sha256(production.INPUT_PATHS[system]),
        }
        for system in SYSTEMS
    }
    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mace": getattr(mace, "__version__", "unknown"),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": str(torch.cuda.get_device_name(0)),
        "model_path": str(production.MODEL_PATH),
        "model_sha256": _sha256(production.MODEL_PATH),
        "calculator": dict(production.CALCULATOR_CONFIG),
        "inputs": inputs,
        "starter_modes": list(STARTER_MODES),
        "random_stream_protocol": {
            "physical_action_stream": "rng_seed",
            "starter_selection_stream": "SeedSequence([rng_seed, 0x535357])",
            "reason": (
                "starter draws must not shift random, bond, or momentum direction draws"
            ),
        },
        "mechanism_frozen": (
            "direction generation, local softening, Gaussian-bias propagation, "
            "proposal relaxation, true-PES quench"
        ),
    }


def _calculator():
    from mace.calculators import MACECalculator

    source_gate = _load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    mace_calculator = MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )
    return MACEBatchCalculator(mace_calculator)


def _energy_trace(result) -> list[dict[str, Any]]:
    initial_energy = float(result.archive.entries[0].energy)
    best_energy = initial_energy
    trace = [
        {
            "trial": 0,
            "landing_energy_eV": initial_energy,
            "best_energy_eV": initial_energy,
            "starter_entry_id": 0,
            "landing_entry_id": 0,
            "accepted_new_basin": True,
        }
    ]
    for trial, record in enumerate(result.walk_history, start=1):
        best_energy = min(best_energy, float(record.energy))
        trace.append(
            {
                "trial": trial,
                "landing_energy_eV": float(record.energy),
                "best_energy_eV": best_energy,
                "starter_entry_id": int(record.seed_entry_id),
                "landing_entry_id": int(record.discovered_entry_id),
                "accepted_new_basin": bool(record.accepted_new_basin),
            }
        )
    return trace


def _run_case(
    *,
    system: str,
    seed: int,
    starter_mode: str,
    case_directory: Path,
    force_budget: int,
) -> dict[str, Any]:
    source_gate = _load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    state = production.load_state(system)
    config = build_config(
        system,
        case_directory,
        seed=seed,
        starter_mode=starter_mode,
        force_budget=force_budget,
    )
    walker = SurfaceWalker(
        calculator=_calculator(),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    wall_time_s = perf_counter() - started
    counts = walker.calculator.snapshot()
    if counts.total != int(result.stats["force_evaluations"]):
        raise RuntimeError("force-evaluation ledger does not close")
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("force-evaluation ledger contains unattributed work")
    if counts.total > force_budget:
        raise RuntimeError("case exceeded its force-evaluation budget")

    initial_energy = float(result.archive.entries[0].energy)
    best_energy = float(result.best_energy)
    summary = {
        "schema_version": 1,
        "system": system,
        "seed": seed,
        "starter_mode": starter_mode,
        "effective_config": asdict(config),
        "initial_energy_eV": initial_energy,
        "best_energy_eV": best_energy,
        "energy_drop_eV": initial_energy - best_energy,
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "completed_trials": int(result.stats["n_trials"]),
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.archive.duplicate_rate()),
        "budget_exhausted": bool(result.stats["budget_exhausted"]),
        "wall_time_s": wall_time_s,
        "stats": result.stats,
    }
    case_directory.mkdir(parents=True, exist_ok=False)
    write_state(case_directory / "best_minimum.xyz", result.best_state)
    _write_json(case_directory / "energy_trace.json", _energy_trace(result))
    _write_json(case_directory / "summary.json", summary)
    return summary


def run_gate(
    *,
    output_directory: Path,
    expected_commit: str,
    systems: Sequence[str],
    seeds: Sequence[int],
    starter_modes: Sequence[str],
    force_budget: int,
    preflight_only: bool = False,
) -> dict[str, Any]:
    systems = tuple(systems)
    seeds = tuple(seeds)
    starter_modes = tuple(starter_modes)
    if not systems or any(system not in SYSTEMS for system in systems):
        raise ValueError(f"systems must be drawn from {SYSTEMS!r}")
    if not seeds or any(isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if not starter_modes or any(mode not in STARTER_MODES for mode in starter_modes):
        raise ValueError(f"starter_modes must be drawn from {STARTER_MODES!r}")
    provenance = _preflight(expected_commit)
    manifest = {
        **provenance,
        "systems": list(systems),
        "seeds": list(seeds),
        "starter_modes": list(starter_modes),
        "force_budget_per_case": force_budget,
        "max_trials_guard": MAX_TRIALS,
        "primary_metric": "best energy at the common force-evaluation budget",
        "secondary_metrics": [
            "energy drop",
            "archive entries",
            "duplicate rate",
            "completed trials",
            "purpose-resolved force evaluations",
            "wall time",
        ],
    }
    if preflight_only:
        return manifest
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "manifest.json", manifest)

    cases = []
    for system in systems:
        for seed in seeds:
            for starter_mode in starter_modes:
                relative = Path(system) / f"seed-{seed:08d}" / starter_mode
                summary = _run_case(
                    system=system,
                    seed=seed,
                    starter_mode=starter_mode,
                    case_directory=output_directory / relative,
                    force_budget=force_budget,
                )
                cases.append(
                    {
                        "system": system,
                        "seed": seed,
                        "starter_mode": starter_mode,
                        "summary": str(relative / "summary.json"),
                        "best_energy_eV": summary["best_energy_eV"],
                        "force_evaluations": summary["force_evaluations"],
                    }
                )
                _write_json(
                    output_directory / "index.json",
                    {"schema_version": 1, "manifest": "manifest.json", "cases": cases},
                )
    return {"schema_version": 1, "manifest": manifest, "cases": cases}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--starter-modes",
        nargs="+",
        choices=STARTER_MODES,
        default=STARTER_MODES,
    )
    parser.add_argument("--force-budget", type=int, default=DEFAULT_FORCE_BUDGET)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_gate(
        output_directory=args.output,
        expected_commit=args.expected_commit,
        systems=args.systems,
        seeds=args.seeds,
        starter_modes=args.starter_modes,
        force_budget=args.force_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
