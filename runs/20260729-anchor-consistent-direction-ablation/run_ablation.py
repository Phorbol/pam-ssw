#!/usr/bin/env python3
"""Run the preregistered anchor-consistent C60 direction ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
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
LOCKED_STATE_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-direction-conditioned-checkpoint-shooting"
    / "run_audit.py"
)
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
    "detached_ritz": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
    "exact_anchor": {
        "direction_selection_mode": "exact_anchor",
    },
    "anchor_lanczos": {
        "direction_selection_mode": "anchor_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
}
EXPECTED_HVP_PER_SELECTION = {
    "detached_ritz": 12,
    "exact_anchor": 1,
    "anchor_lanczos": 12,
}
MEANINGFUL_ENERGY_DROP_EV = 0.001


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state) -> str:
    digest = sha256()
    for values in (
        np.asarray(state.numbers, dtype="<i8"),
        np.asarray(state.positions, dtype="<f8"),
        np.asarray(state.fixed_mask, dtype=np.uint8),
        np.asarray(
            (
                state.cell
                if state.cell is not None
                else np.zeros((3, 3))
            ),
            dtype="<f8",
        ),
        np.asarray(state.pbc, dtype=np.uint8),
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(
            f"direction diagnostics file was not written: {path}"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _load_locked_runtime():
    source = _load_module(
        LOCKED_STATE_RUNNER_PATH,
        "_anchor_consistent_locked_state_source",
    )
    return source._load_locked_states_and_calculator()


def _validate_direction_trace(
    *,
    arm: str,
    direction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not direction_rows:
        raise RuntimeError("direction diagnostics contain no selections")
    expected_hvp = EXPECTED_HVP_PER_SELECTION[arm]
    direction_force_evaluations = 0
    hvp_count = 0
    for index, row in enumerate(direction_rows, start=1):
        selection_delta = int(
            row.get("oracle_selection_force_evaluations_delta", -1)
        )
        if selection_delta != 2 * expected_hvp:
            raise RuntimeError(
                f"direction row {index} has unexpected selection cost"
            )
        if arm == "exact_anchor":
            if (
                row.get("selected_kind") != "anchor"
                or row.get("candidate_count") != 1
                or row.get("direction_hvp_count") != 1
                or abs(abs(_finite(row["anchor_cosine"], "anchor cosine")) - 1.0)
                > 1.0e-12
            ):
                raise RuntimeError(
                    f"direction row {index} violates exact-anchor contract"
                )
            row_hvp = 1
        else:
            expected_depth = int(ARMS[arm]["block_krylov_depth"])
            expected_basis = [2] if arm == "detached_ritz" else [1]
            if (
                row.get("selected_kind") != "block_ritz"
                or row.get("candidate_count") != 0
                or row.get("krylov_blocks") != 1
                or row.get("krylov_depth") != expected_depth
                or row.get("krylov_initial_basis_columns")
                != expected_basis
                or row.get("krylov_hvp_requested") != expected_hvp
                or row.get("krylov_hvp_consumed") != expected_hvp
                or row.get("krylov_hvp_count") != expected_hvp
            ):
                raise RuntimeError(
                    f"direction row {index} violates Krylov contract"
                )
            _finite(row["anchor_cosine"], "anchor cosine")
            _finite(row["selected_curvature"], "selected curvature")
            _finite(row["true_curvature"], "true curvature")
            _finite(
                row["direction_participation_ratio"],
                "participation ratio",
            )
            row_hvp = expected_hvp
        direction_force_evaluations += selection_delta
        hvp_count += row_hvp
    return {
        "selection_count": len(direction_rows),
        "direction_oracle_force_evaluations": (
            direction_force_evaluations
        ),
        "hvp_count": hvp_count,
    }


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
    from pamssw.fingerprint import (
        descriptor_distance,
        structural_descriptor,
    )
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    config = replace(
        base_runner.build_config("c60", case_dir),
        max_trials=1,
        max_force_evals=None,
        rng_seed=seed,
        quench_optimizer="ase-lbfgs",
        quench_fallback_optimizer="ase-fire",
        quench_fmax=0.01,
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

    generation_started = perf_counter()
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
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
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    generation_wall_time = float(perf_counter() - generation_started)

    quench_started = perf_counter()
    landing = walker.relax_true_minimum(
        escape_state,
        trajectory_name=f"{state_id}-{seed}-{arm}-landing",
    )
    quench_wall_time = float(perf_counter() - quench_started)

    before_count = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=seed_entry.entry_id,
    )
    direction_rows = _read_direction_rows(
        Path(config.direction_diagnostics_path)
    )
    direction_audit = _validate_direction_trace(
        arm=arm,
        direction_rows=direction_rows,
    )
    purposes = walker.calculator.snapshot().as_dict()
    force_evaluations = int(sum(purposes.values()))
    if (
        purposes["unattributed"] != 0
        or purposes["bootstrap_true_quench"] != 0
        or purposes["direction_oracle"]
        != direction_audit["direction_oracle_force_evaluations"]
    ):
        raise RuntimeError("case purpose ledger does not close")
    certificate = bool(
        has_force_convergence_certificate(
            landing,
            config.quench_fmax,
        )
    )
    if not certificate:
        raise RuntimeError("terminal quench lacks a strict certificate")

    case_dir.mkdir(parents=True, exist_ok=True)
    starter_path = case_dir / "starter.xyz"
    escape_path = case_dir / "escape.xyz"
    landing_path = case_dir / "landing.xyz"
    base_runner.write_state(starter_path, state)
    base_runner.write_state(escape_path, escape_state)
    base_runner.write_state(landing_path, landing.state)
    state_hash = _state_sha256(state)
    if state_hash != state_provenance["state_sha256"]:
        raise RuntimeError("starter state hash differs from provenance")
    diagnostics = walker.relaxation_diagnostics()
    row = {
        "state_id": state_id,
        "state_sha256": state_hash,
        "exact_starter_reference": True,
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "certificate": certificate,
        "final_max_force_eV_per_A": float(landing.gradient_norm),
        "is_new_basin": len(archive.entries) > before_count,
        "landing_entry_id": int(landing_entry.entry_id),
        "descriptor_delta": float(
            descriptor_distance(
                structural_descriptor(state),
                structural_descriptor(landing.state),
            )
        ),
        "fragmented": bool(
            walker._is_fragmented_cluster(state, landing.state)
        ),
        "fallback_used": bool(
            diagnostics["quench_fallback_attempts"]
        ),
        "quench_iterations": int(landing.n_iter),
        "termination_reason": landing.telemetry.termination_reason,
        "force_evaluations": force_evaluations,
        "purpose_counts": purposes,
        "direction_selection_count": int(
            direction_audit["selection_count"]
        ),
        "direction_hvp_count": int(direction_audit["hvp_count"]),
        "direction_trace_valid": True,
        "generation_wall_time_s": generation_wall_time,
        "quench_wall_time_s": quench_wall_time,
        "starter_path": str(starter_path),
        "starter_file_sha256": _sha256(starter_path),
        "escape_path": str(escape_path),
        "escape_sha256": _sha256(escape_path),
        "landing_path": str(landing_path),
        "landing_sha256": _sha256(landing_path),
        "effective_config": asdict(config),
        "direction_trace": direction_rows,
    }
    _write_json(case_dir / "summary.json", row)
    return row


def _meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row["certificate"]
        and row["is_new_basin"]
        and _finite(row["landing_delta_eV"], "landing delta")
        < -MEANINGFUL_ENERGY_DROP_EV
    )


def _median_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def build_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        (case["state_id"], case["seed"], case["arm"])
        for case in case_matrix()
    }
    observed = {
        (row.get("state_id"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if (
        len(rows) != 18
        or len(observed) != len(rows)
        or observed != expected
    ):
        raise ValueError("evidence requires the exact 18-case cohort")

    for row in rows:
        arm = str(row["arm"])
        purposes = row.get("purpose_counts")
        if (
            row.get("status") != "completed"
            or row.get("certificate") is not True
            or row.get("exact_starter_reference") is not True
            or row.get("direction_trace_valid") is not True
            or not isinstance(purposes, Mapping)
            or sum(int(value) for value in purposes.values())
            != row.get("force_evaluations")
            or purposes.get("unattributed") != 0
            or purposes.get("bootstrap_true_quench") != 0
        ):
            raise ValueError("case completion or purpose ledger does not close")
        selections = int(row["direction_selection_count"])
        expected_hvp = EXPECTED_HVP_PER_SELECTION[arm]
        if (
            purposes.get("direction_oracle")
            != 2 * expected_hvp * selections
            or row.get("direction_hvp_count")
            != expected_hvp * selections
        ):
            raise ValueError("case direction ledger does not close")

    by_key = {
        (str(row["state_id"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [
            by_key[(state_id, seed, arm)]
            for state_id in STATE_IDS
            for seed in SEEDS
        ]
        trace_rows = [
            trace
            for row in arm_rows
            for trace in row["direction_trace"]
        ]
        paired = [
            _finite(
                by_key[(state_id, seed, arm)]["landing_delta_eV"],
                "arm landing delta",
            )
            - _finite(
                by_key[
                    (state_id, seed, "detached_ritz")
                ]["landing_delta_eV"],
                "detached landing delta",
            )
            for state_id in STATE_IDS
            for seed in SEEDS
        ]
        arm_results[arm] = {
            "completed_cases": len(arm_rows),
            "certificate_count": sum(
                bool(row["certificate"]) for row in arm_rows
            ),
            "fallback_count": sum(
                bool(row["fallback_used"]) for row in arm_rows
            ),
            "new_basin_count": sum(
                bool(row["is_new_basin"]) for row in arm_rows
            ),
            "meaningful_outcome_count": sum(
                _meaningful(row) for row in arm_rows
            ),
            "direction_force_evaluations": sum(
                int(row["purpose_counts"]["direction_oracle"])
                for row in arm_rows
            ),
            "total_force_evaluations": sum(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "median_force_evaluations": float(
                statistics.median(
                    int(row["force_evaluations"])
                    for row in arm_rows
                )
            ),
            "median_landing_delta_eV": float(
                statistics.median(
                    _finite(
                        row["landing_delta_eV"],
                        "landing delta",
                    )
                    for row in arm_rows
                )
            ),
            "paired_better_than_detached_count": sum(
                value < 0.0 for value in paired
            ),
            "median_paired_landing_delta_eV": float(
                statistics.median(paired)
            ),
            "median_abs_anchor_cosine": _median_or_none(
                [
                    abs(
                        _finite(
                            trace["anchor_cosine"],
                            "anchor cosine",
                        )
                    )
                    for trace in trace_rows
                ]
            ),
            "state_results": {
                state_id: {
                    "meaningful_outcome_count": sum(
                        _meaningful(
                            by_key[(state_id, seed, arm)]
                        )
                        for seed in SEEDS
                    ),
                    "new_basin_count": sum(
                        bool(
                            by_key[
                                (state_id, seed, arm)
                            ]["is_new_basin"]
                        )
                        for seed in SEEDS
                    ),
                    "median_landing_delta_eV": float(
                        statistics.median(
                            _finite(
                                by_key[
                                    (state_id, seed, arm)
                                ]["landing_delta_eV"],
                                "landing delta",
                            )
                            for seed in SEEDS
                        )
                    ),
                }
                for state_id in STATE_IDS
            },
        }

    purpose_totals = {
        purpose: sum(
            int(row["purpose_counts"][purpose]) for row in rows
        )
        for purpose in rows[0]["purpose_counts"]
    }
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "completed_cases": len(rows),
        },
        "certificate_count": sum(
            bool(row["certificate"]) for row in rows
        ),
        "meaningful_outcome_count": sum(
            _meaningful(row) for row in rows
        ),
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "bootstrap_force_evaluations": 0,
        "arm_results": arm_results,
        "totals": {
            "force_evaluations": sum(
                int(row["force_evaluations"]) for row in rows
            ),
            "purpose_counts": purpose_totals,
            "unattributed_force_evaluations": purpose_totals[
                "unattributed"
            ],
            "generation_wall_time_s": sum(
                _finite(
                    row["generation_wall_time_s"],
                    "generation wall time",
                )
                for row in rows
            ),
            "quench_wall_time_s": sum(
                _finite(
                    row["quench_wall_time_s"],
                    "quench wall time",
                )
                for row in rows
            ),
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "descriptive paired three-seed direction-mechanism audit; "
            "no direction mode or selector is promoted"
        ),
        "cases": list(rows),
    }


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    (
        states,
        shared_calculator,
        base_runner,
        state_provenance,
        shared_provenance,
    ) = _load_locked_runtime()
    rows: list[dict[str, Any]] = []
    for case in case_matrix():
        state_id = str(case["state_id"])
        seed = int(case["seed"])
        arm = str(case["arm"])
        case_dir = (
            output_dir
            / "cases"
            / f"{state_id}-seed{seed}-{arm}"
        )
        print(
            f"[anchor-audit] {state_id} seed={seed} arm={arm}",
            flush=True,
        )
        rows.append(
            _run_case(
                state=states[state_id],
                state_provenance=state_provenance[state_id],
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                state_id=state_id,
                seed=seed,
                arm=arm,
                case_dir=case_dir,
            )
        )
        _write_json(
            output_dir / "raw.json",
            {
                "schema_version": 1,
                "execution_commit": actual_commit,
                "shared_provenance": shared_provenance,
                "state_provenance": state_provenance,
                "cases": rows,
            },
        )
    evidence = build_evidence(rows)
    evidence["execution_commit"] = actual_commit
    evidence["shared_provenance"] = shared_provenance
    evidence["state_provenance"] = state_provenance
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


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
