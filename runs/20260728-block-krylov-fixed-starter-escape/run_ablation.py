#!/usr/bin/env python3
"""Run the preregistered fixed-starter productive-escape ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FIXED_AUDIT_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-block-krylov-direction-audit"
    / "run_fixed_state_audit.py"
)
DIRECTION_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-block-krylov-direction-gpu-ablation"
    / "run_ablation.py"
)
STATE_IDS = ("bootstrap_quenched", "intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
    "discrete": {"direction_selection_mode": "discrete"},
    "variational_breadth": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 6,
        "block_krylov_depth": 1,
    },
    "balanced_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    },
    "deep_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
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


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_fixed_states():
    audit = _load_module(FIXED_AUDIT_PATH, "_fixed_starter_source_audit")
    strict_wrapper, base_runner = audit._load_frozen_runtime()
    registry = audit.FIXED_STATE_REGISTRY["c60"]
    origin_summary = audit._validate_origin("c60", registry)
    bootstrap, shared_calculator, provenance, template = audit._reconstruct_bootstrap(
        system="c60",
        entry=registry[0],
        strict_wrapper=strict_wrapper,
        base_runner=base_runner,
        origin_summary=origin_summary,
    )
    states = {
        "bootstrap_quenched": (bootstrap, provenance),
    }
    for entry in registry[1:]:
        state, state_provenance = audit._load_locked_state(entry, template)
        states[str(entry["state_id"])] = (state, state_provenance)
    if tuple(states) != STATE_IDS:
        raise RuntimeError("fixed-state registry drifted")
    return states, shared_calculator, base_runner, provenance


def _run_case(
    *,
    state,
    state_provenance: Mapping[str, Any],
    shared_calculator,
    base_runner,
    state_id: str,
    seed: int,
    arm: str,
    case_dir: Path,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.fingerprint import descriptor_distance, structural_descriptor
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    direction_runner = _load_module(
        DIRECTION_RUNNER_PATH,
        "_fixed_starter_direction_trace_contract",
    )
    config = replace(
        base_runner.build_config("c60", case_dir),
        max_trials=1,
        max_force_evals=None,
        rng_seed=seed,
        **ARMS[arm],
    )
    walker = SurfaceWalker(
        calculator=shared_calculator,
        config=config,
        softening_enabled=True,
    )
    walker._reset_direction_diagnostics()
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )

    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        starter_evaluation = walker.calculator.evaluate(state)
    starter_energy = float(starter_evaluation.energy)
    seed_entry = archive.add(state, starter_energy, parent_id=None)
    proposal = walker._proposal_pool(
        state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=seed_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    escape_state = proposal.state
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    landing = walker.relax_true_minimum(
        escape_state,
        trajectory_name=f"{state_id}-{seed}-{arm}-landing",
    )
    wall_time = float(perf_counter() - started)

    before_count = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=seed_entry.entry_id,
    )
    is_new_basin = len(archive.entries) > before_count
    fragmented = walker._is_fragmented_cluster(state, landing.state)
    direction_rows = _read_direction_rows(Path(config.direction_diagnostics_path))
    direction_audit = direction_runner.validate_direction_trace(
        arm=arm,
        direction_rows=direction_rows,
    )
    purpose_counts = walker.calculator.snapshot().as_dict()
    force_evaluations = int(sum(purpose_counts.values()))
    if purpose_counts["unattributed"] != 0:
        raise RuntimeError("case contains unattributed evaluations")
    if purpose_counts["direction_oracle"] != direction_audit["direction_oracle_force_evaluations"]:
        raise RuntimeError("direction trace does not close against the purpose ledger")
    if any(
        int(row["oracle_selection_force_evaluations_delta"]) != 24
        for row in direction_rows
    ):
        raise RuntimeError("a completed direction selection did not use exactly 24 force evaluations")

    starter_descriptor = structural_descriptor(state)
    landing_descriptor = structural_descriptor(landing.state)
    case_dir.mkdir(parents=True, exist_ok=True)
    base_runner.write_state(case_dir / "escape.xyz", escape_state)
    base_runner.write_state(case_dir / "landing.xyz", landing.state)
    row = {
        "state_id": state_id,
        "state_sha256": state_provenance["state_sha256"],
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "is_new_basin": is_new_basin,
        "landing_entry_id": int(landing_entry.entry_id),
        "descriptor_delta": float(descriptor_distance(starter_descriptor, landing_descriptor)),
        "fragmented": bool(fragmented),
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "direction_selection_count": int(direction_audit["selection_count"]),
        "direction_hvp_count": int(
            sum(
                int(row.get("krylov_hvp_consumed", 12))
                for row in direction_rows
            )
        ),
        "direction_trace_valid": True,
        "quench_converged": bool(
            has_force_convergence_certificate(landing, config.quench_fmax)
        ),
        "quench_iterations": int(landing.n_iter),
        "wall_time_s": wall_time,
        "effective_config": asdict(config),
        "direction_trace": direction_rows,
    }
    _write_json(case_dir / "summary.json", row)
    return row


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def build_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        (row["state_id"], row["seed"], row["arm"])
        for row in case_matrix()
    }
    if len(rows) != 36:
        raise ValueError("evidence requires exactly 36 cases")
    observed = {
        (row.get("state_id"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if observed != expected or len(observed) != len(rows):
        raise ValueError("case cohort differs from the preregistration")
    by_key = {
        (str(row["state_id"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    for key, row in by_key.items():
        if row.get("status") != "completed" or row.get("direction_trace_valid") is not True:
            raise ValueError(f"case {key} is not complete")
        purposes = row.get("purpose_counts")
        if (
            not isinstance(purposes, Mapping)
            or sum(int(value) for value in purposes.values()) != row.get("force_evaluations")
            or purposes.get("unattributed") != 0
            or purposes.get("direction_oracle") != 24 * row.get("direction_selection_count")
        ):
            raise ValueError(f"case {key} purpose ledger does not close")
        if row.get("direction_hvp_count") != 12 * row.get("direction_selection_count"):
            raise ValueError(f"case {key} HVP ledger does not close")

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [by_key[(state_id, seed, arm)] for state_id in STATE_IDS for seed in SEEDS]
        paired = []
        if arm != "discrete":
            for state_id in STATE_IDS:
                for seed in SEEDS:
                    block = by_key[(state_id, seed, arm)]
                    discrete = by_key[(state_id, seed, "discrete")]
                    paired.append(
                        _finite(block["landing_delta_eV"], "block landing delta")
                        - _finite(discrete["landing_delta_eV"], "discrete landing delta")
                    )
        arm_results[arm] = {
            "completed_cases": len(arm_rows),
            "new_basin_count": sum(bool(row["is_new_basin"]) for row in arm_rows),
            "downhill_landing_count": sum(
                _finite(row["landing_delta_eV"], "landing delta") < 0.0
                for row in arm_rows
            ),
            "median_landing_delta_eV": statistics.median(
                _finite(row["landing_delta_eV"], "landing delta")
                for row in arm_rows
            ),
            "median_force_evaluations": statistics.median(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "mean_direction_force_evaluations": statistics.mean(
                int(row["purpose_counts"]["direction_oracle"])
                for row in arm_rows
            ),
            "paired_better_count": None if arm == "discrete" else sum(value < 0.0 for value in paired),
            "median_paired_landing_delta_eV": None if arm == "discrete" else statistics.median(paired),
            "state_results": {
                state_id: {
                    "new_basin_count": sum(
                        bool(by_key[(state_id, seed, arm)]["is_new_basin"])
                        for seed in SEEDS
                    ),
                    "median_landing_delta_eV": statistics.median(
                        _finite(
                            by_key[(state_id, seed, arm)]["landing_delta_eV"],
                            "landing delta",
                        )
                        for seed in SEEDS
                    ),
                }
                for state_id in STATE_IDS
            },
        }
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "completed_cases": len(rows),
        },
        "arm_results": arm_results,
        "production_default_changed": False,
        "claim_ceiling": (
            "descriptive fixed-starter mechanism audit; no arm is promoted "
            "without a separate preregistered decision experiment"
        ),
        "cases": list(rows),
    }


def run(*, output_dir: Path, expected_git_commit: str) -> dict[str, Any]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    states, shared_calculator, base_runner, bootstrap_provenance = _load_fixed_states()
    rows: list[dict[str, Any]] = []
    for case in case_matrix():
        state, state_provenance = states[case["state_id"]]
        case_dir = (
            output_dir
            / "cases"
            / f"{case['state_id']}-seed{case['seed']}-{case['arm']}"
        )
        print(
            f"[escape-audit] {case['state_id']} seed={case['seed']} arm={case['arm']}",
            flush=True,
        )
        rows.append(
            _run_case(
                state=state,
                state_provenance=state_provenance,
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                state_id=case["state_id"],
                seed=case["seed"],
                arm=case["arm"],
                case_dir=case_dir,
            )
        )
        _write_json(
            output_dir / "raw.json",
            {
                "schema_version": 1,
                "execution_commit": actual_commit,
                "bootstrap_provenance": bootstrap_provenance,
                "cases": rows,
            },
        )
    evidence = build_evidence(rows)
    evidence["execution_commit"] = actual_commit
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "output")
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = run(
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
    )
    print(json.dumps(evidence["arm_results"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
