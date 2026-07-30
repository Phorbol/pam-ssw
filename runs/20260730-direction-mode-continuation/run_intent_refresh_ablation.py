#!/usr/bin/env python3
"""Run the paired two-vector intent-refresh direction screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
BASE_RUNNER_PATH = RUN_ROOT / "run_residual_ritz_ablation.py"
SEEDS = tuple(range(42, 52))
ARMS = (
    "transported_direction",
    "continuation_intent_ritz2",
    "fixed_intent_ritz",
)
INTENT_REFRESH_CONFIG = {
    "direction_selection_mode": "continuation_intent_krylov",
    "block_krylov_blocks": 1,
    "block_krylov_depth": 1,
}
SYSTEMS = ("c60", "pdo")
STATE_IDS = {
    "c60": ("intermediate_accepted", "plateau_accepted"),
    "pdo": ("raw_bootstrap",),
}


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


def _load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "_intent_refresh_base_runner", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(
    *,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    c60_locked_source: Path | None = None,
) -> dict[str, Any]:
    base = _load_base_runner()
    return base.run_screen(
        system=system,
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
        arms=ARMS,
        candidate_arm="continuation_intent_ritz2",
        candidate_config=INTENT_REFRESH_CONFIG,
        c60_locked_source=c60_locked_source,
    )


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
