#!/usr/bin/env python3
"""Run the transport-only residual observation cohort."""

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
ARM = "transported_direction"
SYSTEMS = ("c60", "pdo")
C60_STATE_IDS = ("intermediate_accepted", "plateau_accepted")
PDO_STATE_ID = "raw_bootstrap"
MAX_TOTAL_FORCE_EVALUATIONS = 50_000


def case_matrix(system: str) -> list[dict[str, Any]]:
    if system == "c60":
        state_ids = C60_STATE_IDS
    elif system == "pdo":
        state_ids = (PDO_STATE_ID,)
    else:
        raise ValueError(f"unknown system: {system}")
    return [
        {
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "arm": ARM,
        }
        for state_id in state_ids
        for seed in SEEDS
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
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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
            C60_RUNNER_PATH,
            "_residual_observation_c60_runner",
        )
        runner.SEEDS = SEEDS
        runner.MAX_TOTAL_FORCE_EVALUATIONS = (
            MAX_TOTAL_FORCE_EVALUATIONS
        )
        return runner.run(
            output_dir,
            locked_source=Path(c60_locked_source),
            arms=(ARM,),
        )
    if system == "pdo":
        runner = _load_module(
            PDO_RUNNER_PATH,
            "_residual_observation_pdo_runner",
        )
        runner.SEEDS = SEEDS
        runner.ARMS = (ARM,)
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
    print(
        json.dumps(
            run(
                system=args.system,
                output_dir=args.output,
                expected_git_commit=args.expected_git_commit,
                c60_locked_source=args.c60_locked_source,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
