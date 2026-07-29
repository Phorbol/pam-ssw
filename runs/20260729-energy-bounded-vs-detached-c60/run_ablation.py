#!/usr/bin/env python3
"""Run the paired C60 energy-bounded-anchor versus detached-Ritz ablation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
BASE_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-energy-bounded-anchor-c60"
    / "run_ablation.py"
)
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
    "detached_ritz": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
    "energy_bounded_anchor": {
        "direction_selection_mode": "energy_bounded_anchor",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
}
EXPECTED_HVP_PER_SELECTION = {
    "detached_ritz": 12,
    "energy_bounded_anchor": 12,
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case_matrix() -> list[dict[str, Any]]:
    return [
        {"state_id": state_id, "seed": seed, "arm": arm}
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    ]


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _validate_detached_trace(
    direction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not direction_rows:
        raise RuntimeError("direction diagnostics contain no selections")
    direction_force_evaluations = 0
    for index, row in enumerate(direction_rows, start=1):
        selection_delta = int(
            row.get("oracle_selection_force_evaluations_delta", -1)
        )
        if (
            selection_delta != 24
            or row.get("selected_kind") != "block_ritz"
            or row.get("candidate_count") != 0
            or row.get("krylov_blocks") != 1
            or row.get("krylov_depth") != 6
            or row.get("krylov_initial_basis_columns") != [2]
            or row.get("krylov_hvp_requested") != 12
            or row.get("krylov_hvp_consumed") != 12
            or row.get("krylov_hvp_count") != 12
        ):
            raise RuntimeError(
                f"direction row {index} violates detached-Ritz contract"
            )
        _finite(row.get("anchor_cosine"), "anchor cosine")
        _finite(row.get("selected_curvature"), "selected curvature")
        _finite(row.get("true_curvature"), "true curvature")
        direction_force_evaluations += selection_delta
    return {
        "selection_count": len(direction_rows),
        "direction_oracle_force_evaluations": (
            direction_force_evaluations
        ),
        "hvp_count": len(direction_rows) * 12,
    }


def _configured_base_runner():
    base = _load_module(
        BASE_RUNNER_PATH,
        "_energy_bounded_vs_detached_base_runner",
    )
    energy_bounded_validator = base._validate_direction_trace

    def validate_direction_trace(
        *,
        arm: str,
        direction_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if arm == "detached_ritz":
            return _validate_detached_trace(direction_rows)
        return energy_bounded_validator(
            arm=arm,
            direction_rows=direction_rows,
        )

    base.ARMS = ARMS
    base.EXPECTED_HVP_PER_SELECTION = EXPECTED_HVP_PER_SELECTION
    base._validate_direction_trace = validate_direction_trace
    return base


def load_and_validate_evidence(path: Path) -> dict[str, Any]:
    return _configured_base_runner().load_and_validate_evidence(path)


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    return _configured_base_runner().run(
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
    )


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "output",
    )
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = run(
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
    )
    print(
        json.dumps(
            {
                "cohort": evidence["cohort"],
                "arm_results": evidence["arm_results"],
                "totals": evidence["totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
