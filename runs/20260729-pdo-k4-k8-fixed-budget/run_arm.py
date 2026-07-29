#!/usr/bin/env python3
"""Run one PdO K4/K8 arm under an exact 20,000-force-evaluation budget."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np

from pamssw import write_state
from pamssw.accounting import EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FROZEN_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
ARMS = {"k4": 4, "k8": 8}
SEEDS = (42, 43)
TOTAL_FORCE_BUDGET = 20_000
MAX_TRIALS = 200
EXPECTED_OUTPUT_FILES = (
    "best_minimum.xyz",
    "energy_trace.json",
    "walk_records.json",
    "optimizer_diagnostics.json",
    "summary.json",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_pdo_k4_k8_frozen_runner", FROZEN_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen PdO production runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), allow_nan=False))


def config_projection(
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    base_runner: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    source_config = base_runner.build_config("pdo", Path(output_dir))
    effective_config = replace(
        source_config,
        max_trials=MAX_TRIALS,
        max_force_evals=TOTAL_FORCE_BUDGET,
        oracle_candidates=ARMS[arm],
        rng_seed=seed,
    )
    source = _json_mapping(asdict(source_config))
    effective = _json_mapping(asdict(effective_config))
    if set(source) != set(effective):
        raise RuntimeError("source and effective config fields differ")
    diff = {
        field: [source[field], effective[field]]
        for field in sorted(source)
        if source[field] != effective[field]
    }
    expected_fields = {"max_force_evals", "oracle_candidates", "rng_seed"}
    actual_fields = set(diff)
    allowed_fields = {
        field
        for field in expected_fields
        if source.get(field) != effective.get(field)
    }
    if actual_fields != allowed_fields:
        raise RuntimeError(f"unexpected PdO protocol drift: {diff}")

    required = {
        "max_trials": 200,
        "max_force_evals": TOTAL_FORCE_BUDGET,
        "max_steps_per_walk": 8,
        "proposal_relax_steps": 300,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "quench_optimizer": "scipy-lbfgsb",
        "quench_fallback_optimizer": None,
        "quench_fmax": 0.03,
        "quench_maxiter": 400,
        "direction_type_ucb_enabled": False,
        "direction_probe_enabled": False,
        "plateau_evolution_enabled": False,
        "archive_escape_momentum_enabled": False,
    }
    drift = {
        field: [expected, effective.get(field)]
        for field, expected in required.items()
        if effective.get(field) != expected
    }
    if drift:
        raise RuntimeError(f"PdO fixed-budget protocol drift: {drift}")
    return source, effective, diff


def preflight(
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    expected_git_commit: str,
    base_runner: ModuleType | None = None,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if expected_git_commit != actual_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not FROZEN_RUNNER_PATH.is_file():
        raise FileNotFoundError(FROZEN_RUNNER_PATH)
    base = _base_runner() if base_runner is None else base_runner
    base_preflight = dict(
        base.preflight(system="pdo", expected_git_commit=expected_git_commit)
    )
    source, effective, diff = config_projection(
        arm=arm,
        seed=seed,
        output_dir=output_dir,
        base_runner=base,
    )
    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "system": "pdo",
        "arm": arm,
        "seed": seed,
        "total_force_budget": TOTAL_FORCE_BUDGET,
        "base_runner": {
            "path": str(FROZEN_RUNNER_PATH),
            "sha256": _sha256(FROZEN_RUNNER_PATH),
        },
        "base_preflight": base_preflight,
        "source_config": source,
        "effective_config": effective,
        "config_diff": diff,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _energy_trace(result) -> list[dict[str, Any]]:
    initial_energy = float(result.archive.entries[0].energy)
    best_energy = initial_energy
    points = [
        {
            "trial": 0,
            "energy_eV": initial_energy,
            "best_energy_eV": initial_energy,
            "accepted_new_basin": True,
        }
    ]
    for trial, record in enumerate(result.walk_history, start=1):
        best_energy = min(best_energy, float(record.energy))
        points.append(
            {
                "trial": trial,
                "energy_eV": float(record.energy),
                "best_energy_eV": best_energy,
                "accepted_new_basin": bool(record.accepted_new_basin),
            }
        )
    return points


def _walk_records(result) -> list[dict[str, Any]]:
    return [
        {
            "trial": trial,
            "seed_entry_id": int(record.seed_entry_id),
            "discovered_entry_id": int(record.discovered_entry_id),
            "energy_eV": float(record.energy),
            "accepted_new_basin": bool(record.accepted_new_basin),
        }
        for trial, record in enumerate(result.walk_history, start=1)
    ]


def run(
    *,
    arm: str,
    seed: int,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    base = _base_runner()
    checked = preflight(
        arm=arm,
        seed=seed,
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
        base_runner=base,
    )
    if preflight_only:
        return checked

    output_dir.mkdir(parents=True)
    state = base.load_state("pdo")
    config = replace(
        base.build_config("pdo", output_dir),
        max_trials=MAX_TRIALS,
        max_force_evals=TOTAL_FORCE_BUDGET,
        oracle_candidates=ARMS[arm],
        rng_seed=seed,
    )
    if _json_mapping(asdict(config)) != checked["effective_config"]:
        raise RuntimeError("runtime config differs from preflight")
    walker = SurfaceWalker(
        calculator=ASECalculator(base._calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    wall_time_s = perf_counter() - started

    counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    force_evaluations = int(result.stats["force_evaluations"])
    if counts.total != force_evaluations:
        raise RuntimeError("force-evaluation total does not close")
    if force_evaluations > TOTAL_FORCE_BUDGET:
        raise RuntimeError("fixed force-evaluation budget was exceeded")
    if purpose_counts[EvaluationPurpose.UNATTRIBUTED.value] != 0:
        raise RuntimeError("purpose ledger contains unattributed evaluations")
    if purpose_counts[EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH.value] <= 0:
        raise RuntimeError("top-level raw-State bootstrap was not accounted")
    if purpose_counts[EvaluationPurpose.STARTER_TRUE_QUENCH.value] != 0:
        raise RuntimeError("top-level run contains dispatched-starter costs")
    n_trials = int(result.stats["n_trials"])
    budget_exhausted = bool(result.stats["budget_exhausted"])
    if not budget_exhausted and n_trials != MAX_TRIALS:
        raise RuntimeError("run stopped before its trial or force budget")

    energy_trace = _energy_trace(result)
    walk_records = _walk_records(result)
    telemetry = walker.relaxation_diagnostics()
    initial_energy = float(energy_trace[0]["energy_eV"])
    summary = {
        **checked["base_preflight"],
        "experiment": checked,
        "effective_config": asdict(config),
        "input_state": {
            "atom_count": int(len(state.numbers)),
            "fixed_count": int(np.count_nonzero(state.fixed_mask)),
            "pbc": list(state.pbc),
        },
        "initial_energy_eV": initial_energy,
        "best_energy_eV": float(result.best_energy),
        "energy_drop_eV": initial_energy - float(result.best_energy),
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "optimizer_telemetry": telemetry,
        "stats": result.stats,
        "timing": {"total_wall_time_s": wall_time_s},
        "walk_records": walk_records,
    }

    write_state(output_dir / "best_minimum.xyz", result.best_state)
    _write_json(output_dir / "energy_trace.json", energy_trace)
    _write_json(output_dir / "walk_records.json", walk_records)
    _write_json(output_dir / "optimizer_diagnostics.json", telemetry)
    _atomic_write_json(output_dir / "summary.json", summary)
    for name in EXPECTED_OUTPUT_FILES:
        if not (output_dir / name).is_file():
            raise RuntimeError(f"missing output: {name}")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(
        arm=args.arm,
        seed=args.seed,
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
