#!/usr/bin/env python3
"""Run the paired transport / 2D Ritz / fresh Ritz screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
C60_RUNNER_PATH = RUN_ROOT / "run_ablation.py"
PDO_RUNNER_PATH = RUN_ROOT / "run_pdo_raw_transfer.py"
SEEDS = tuple(range(42, 52))
ARMS = (
    "transported_direction",
    "residual_ritz2",
    "fixed_intent_ritz",
)
RESIDUAL_RITZ_CONFIG = {
    "direction_selection_mode": "continuation_krylov",
    "block_krylov_blocks": 1,
    "block_krylov_depth": 2,
}
SYSTEMS = ("c60", "pdo")
STATE_IDS = {
    "c60": ("intermediate_accepted", "plateau_accepted"),
    "pdo": ("raw_bootstrap",),
}
MAX_TOTAL_FORCE_EVALUATIONS = 100_000


def case_matrix(system: str) -> list[dict[str, Any]]:
    if system not in STATE_IDS:
        raise ValueError(f"unknown system: {system}")
    return [
        {
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "arm": arm,
        }
        for state_id in STATE_IDS[system]
        for seed in SEEDS
        for arm in ARMS
    ]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(
    *,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    c60_locked_source: Path | None = None,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    output_dir = Path(output_dir)
    if system == "c60":
        if c60_locked_source is None:
            raise ValueError("c60_locked_source is required for C60")
        runner = _load_module(
            C60_RUNNER_PATH, "_residual_ritz_c60_runner"
        )
        runner.SEEDS = SEEDS
        runner.ARMS["residual_ritz2"] = RESIDUAL_RITZ_CONFIG
        runner.MAX_TOTAL_FORCE_EVALUATIONS = (
            MAX_TOTAL_FORCE_EVALUATIONS
        )
        return runner.run(
            output_dir,
            locked_source=Path(c60_locked_source),
            arms=ARMS,
        )
    if system == "pdo":
        runner = _load_module(
            PDO_RUNNER_PATH, "_residual_ritz_pdo_runner"
        )
        runner.SEEDS = SEEDS
        runner.ARMS = ARMS
        runner.EXTRA_DIRECTION_ARMS = {
            "residual_ritz2": RESIDUAL_RITZ_CONFIG
        }
        runner.MAX_TOTAL_FORCE_EVALUATIONS = (
            MAX_TOTAL_FORCE_EVALUATIONS
        )
        return runner.run(output_dir)
    raise ValueError(f"unknown system: {system}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=SYSTEMS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--c60-locked-source", type=Path)
    args = parser.parse_args(argv)
    result = run(
        system=args.system,
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        c60_locked_source=args.c60_locked_source,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
