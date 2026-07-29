#!/usr/bin/env python3
"""Run the preregistered C60 direction-continuation ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
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
MAX_TOTAL_FORCE_EVALUATIONS = 10_000
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
    "continuation_lanczos": {
        "direction_selection_mode": "continuation_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
}


def case_matrix(
    arms: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    selected_arms = tuple(ARMS) if arms is None else tuple(arms)
    if (
        not selected_arms
        or len(set(selected_arms)) != len(selected_arms)
        or any(arm not in ARMS for arm in selected_arms)
    ):
        raise ValueError("arms must be unique known direction arms")
    return [
        {"state_id": state_id, "seed": seed, "arm": arm}
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in selected_arms
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
            state.cell if state.cell is not None else np.zeros((3, 3)),
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


def _load_persisted_c60_state(path: Path):
    from ase.io import read

    from pamssw.state import State

    atoms = read(path)
    cell = np.asarray(atoms.cell.array, dtype=float)
    return State(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=None if not np.any(cell) else cell,
        pbc=tuple(bool(value) for value in atoms.pbc),
        fixed_mask=np.zeros(len(atoms), dtype=bool),
        metadata=dict(atoms.info),
    )


def _load_persisted_locked_runtime(source_dir: Path):
    from pamssw.calculators import ASECalculator

    checkpoint_source = _load_module(
        LOCKED_STATE_RUNNER_PATH,
        "_direction_continuation_checkpoint_source",
    )
    audit = _load_module(
        checkpoint_source.FIXED_STATE_AUDIT_PATH,
        "_direction_continuation_fixed_state_audit",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    raw = json.loads((source_dir / "raw.json").read_text(encoding="utf-8"))
    states = {}
    provenance = {}
    for state_id in STATE_IDS:
        case_dir = (
            source_dir
            / "cases"
            / f"{state_id}-seed42-detached_ritz"
        )
        summary = json.loads(
            (case_dir / "summary.json").read_text(encoding="utf-8")
        )
        starter_path = case_dir / "starter.xyz"
        if _sha256(starter_path) != summary["starter_file_sha256"]:
            raise RuntimeError(
                f"persisted starter checksum drifted for {state_id}"
            )
        state = _load_persisted_c60_state(starter_path)
        state_hash = _state_sha256(state)
        if (
            state_hash != summary["state_sha256"]
            or state_hash
            != raw["state_provenance"][state_id]["state_sha256"]
        ):
            raise RuntimeError(
                f"persisted starter state hash drifted for {state_id}"
            )
        states[state_id] = state
        provenance[state_id] = {
            **raw["state_provenance"][state_id],
            "persisted_starter_path": str(starter_path),
            "persisted_starter_sha256": _sha256(starter_path),
        }
    shared = {
        **raw["shared_provenance"],
        "bootstrap_force_evaluations": 0,
        "bootstrap_skipped": True,
        "locked_source_output": str(source_dir),
        "selection_rationale": (
            "exact state-hash-matched starters persisted by the preceding "
            "validated 18-case cohort"
        ),
    }
    return (
        states,
        ASECalculator(base_runner._calculator()),
        base_runner,
        provenance,
        shared,
    )


def _load_locked_runtime(source_dir: Path | None = None):
    if source_dir is not None:
        return _load_persisted_locked_runtime(Path(source_dir))
    source = _load_module(
        LOCKED_STATE_RUNNER_PATH,
        "_direction_continuation_locked_state_source",
    )
    return source._load_locked_states_and_calculator()


def _validate_direction_trace(
    *,
    arm: str,
    direction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not direction_rows:
        raise RuntimeError("direction diagnostics contain no selections")

    force_evaluations = 0
    hvp_count = 0
    for index, row in enumerate(direction_rows):
        if index == 0:
            if (
                row.get("step") != 0
                or row.get("selected_kind") != "block_ritz"
                or row.get("candidate_count") != 0
                or row.get("krylov_blocks") != 1
                or row.get("krylov_depth") != 6
                or row.get("krylov_initial_basis_columns") != [2]
                or row.get("krylov_hvp_requested") != 12
                or row.get("krylov_hvp_consumed") != 12
                or row.get("krylov_hvp_count") != 12
                or row.get("oracle_selection_force_evaluations_delta")
                != 0
                or row.get("shared_initial_direction") is not True
            ):
                raise RuntimeError(
                    "direction trace violates the common step-zero contract"
                )
            row_hvps = 0
        elif arm == "transported_direction":
            if (
                row.get("selected_kind") != "transported"
                or row.get("candidate_count") != 1
                or row.get("direction_hvp_count") != 1
                or row.get("oracle_selection_force_evaluations_delta")
                != 2
                or row.get("continuation_source") != "selected_mode"
            ):
                raise RuntimeError(
                    f"direction row {index} violates transported contract"
                )
            row_hvps = 1
        else:
            expected_kind = (
                "block_ritz"
                if arm == "fixed_intent_ritz"
                else "continuation_ritz"
            )
            expected_depth = (
                6 if arm == "fixed_intent_ritz" else 12
            )
            expected_columns = (
                [2] if arm == "fixed_intent_ritz" else [1]
            )
            if (
                row.get("selected_kind") != expected_kind
                or row.get("candidate_count") != 0
                or row.get("krylov_blocks") != 1
                or row.get("krylov_depth") != expected_depth
                or row.get("krylov_initial_basis_columns")
                != expected_columns
                or row.get("krylov_hvp_requested") != 12
                or row.get("krylov_hvp_consumed") != 12
                or row.get("krylov_hvp_count") != 12
                or row.get("oracle_selection_force_evaluations_delta")
                != 24
            ):
                raise RuntimeError(
                    f"direction row {index} violates Krylov contract"
                )
            row_hvps = 12
        if not row.get("selected_direction_sha256"):
            raise RuntimeError(
                f"direction row {index} lacks a direction hash"
            )
        force_evaluations += int(
            row["oracle_selection_force_evaluations_delta"]
        )
        hvp_count += row_hvps

    return {
        "selection_count": len(direction_rows),
        "direction_oracle_force_evaluations": force_evaluations,
        "hvp_count": hvp_count,
    }


def _precompute_shared_initial_direction(
    *,
    state,
    shared_calculator,
    base_runner,
    state_id: str,
    seed: int,
    shared_dir: Path,
    system: str = "c60",
):
    from pamssw.accounting import EvaluationPurpose
    from pamssw.walker import ProposalPotential, SurfaceWalker

    config = replace(
        base_runner.build_config(system, shared_dir),
        max_trials=1,
        max_force_evals=None,
        rng_seed=seed,
        direction_selection_mode="block_krylov",
        block_krylov_blocks=1,
        block_krylov_depth=6,
    )
    walker = SurfaceWalker(
        calculator=shared_calculator,
        config=config,
        softening_enabled=True,
    )
    anchor_direction, krylov_intents = (
        walker._initialize_walk_direction_context(
            state,
            trial_index=0,
        )
    )
    softening = walker._build_softening(state, anchor_direction)
    proposal = ProposalPotential(
        walker.calculator,
        biases=[],
        softening=softening,
    )
    scoring_proposal = walker._direction_scoring_proposal(proposal)
    with walker.calculator.purpose(
        EvaluationPurpose.DIRECTION_ORACLE
    ):
        choice = walker.oracle._choose_block_krylov_direction(
            state,
            scoring_proposal,
            krylov_intents,
            anchor_direction,
            None,
            None,
            depth=6,
        )
    purposes = walker.calculator.snapshot().as_dict()
    if (
        purposes["direction_oracle"] != 24
        or sum(purposes.values()) != 24
        or choice.diagnostics.get("krylov_hvp_consumed") != 12
    ):
        raise RuntimeError(
            "shared initial direction did not consume exactly 12 HVPs"
        )
    direction = np.asarray(choice.direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    direction_hash = sha256(
        np.asarray(direction, dtype="<f8").tobytes()
    ).hexdigest()
    return choice, {
        "state_id": state_id,
        "seed": seed,
        "direction_sha256": direction_hash,
        "direction": direction.tolist(),
        "curvature": float(choice.curvature),
        "true_curvature": float(choice.true_curvature),
        "force_evaluations": 24,
        "purpose_counts": purposes,
        "diagnostics": choice.diagnostics,
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
    initial_direction_choice,
    system: str = "c60",
    fragmentation_applicable: bool = True,
    require_terminal_certificate: bool = True,
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
        base_runner.build_config(system, case_dir),
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
    escape_state = walker._walk_candidate_from_seed(
        state,
        archive,
        walker.step_target_controller.target(archive),
        trial_index=0,
        proposal_index=0,
        seed_entry_id=seed_entry.entry_id,
        initial_direction_choice=initial_direction_choice,
    )
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
        has_force_convergence_certificate(landing, config.quench_fmax)
    )
    if require_terminal_certificate and not certificate:
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
        "fragmentation_applicable": fragmentation_applicable,
        "fragmented": bool(
            walker._is_fragmented_cluster(state, landing.state)
            if fragmentation_applicable
            else False
        ),
        "landing_geometry_valid": bool(
            walker.geometry_validator.is_valid_state(landing.state)
        ),
        "fallback_used": bool(
            diagnostics["quench_fallback_attempts"]
        ),
        "continuation_projection_degenerate": int(
            walker._direction_stats_summary()[
                "continuation_projection_degenerate"
            ]
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
        "shared_initial_direction_sha256": direction_rows[0][
            "selected_direction_sha256"
        ],
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


def run(
    output_dir: Path,
    *,
    locked_source: Path | None = None,
    arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    execution_commit = _current_commit()
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
    ) = _load_locked_runtime(locked_source)
    rows: list[dict[str, Any]] = []
    selected_arms = tuple(ARMS) if arms is None else tuple(arms)
    case_matrix(selected_arms)
    total_force_evaluations = 0
    shared_initial_directions: list[dict[str, Any]] = []
    for state_id in STATE_IDS:
        for seed in SEEDS:
            initial_choice, shared_record = (
                _precompute_shared_initial_direction(
                    state=states[state_id],
                    shared_calculator=shared_calculator,
                    base_runner=base_runner,
                    state_id=state_id,
                    seed=seed,
                    shared_dir=(
                        output_dir
                        / "shared-initial"
                        / f"{state_id}-seed{seed}"
                    ),
                )
            )
            shared_initial_directions.append(shared_record)
            total_force_evaluations += int(
                shared_record["force_evaluations"]
            )
            for arm in selected_arms:
                case_dir = (
                    output_dir
                    / "cases"
                    / f"{state_id}-seed{seed}-{arm}"
                )
                print(
                    f"[continuation-audit] {state_id} seed={seed} "
                    f"arm={arm}",
                    flush=True,
                )
                row = _run_case(
                    state=states[state_id],
                    state_provenance=state_provenance[state_id],
                    shared_calculator=shared_calculator,
                    base_runner=base_runner,
                    state_id=state_id,
                    seed=seed,
                    arm=arm,
                    case_dir=case_dir,
                    initial_direction_choice=initial_choice,
                )
                rows.append(row)
                total_force_evaluations += int(
                    row["force_evaluations"]
                )
                if (
                    total_force_evaluations
                    > MAX_TOTAL_FORCE_EVALUATIONS
                ):
                    raise RuntimeError(
                        "C1 exceeded the 10000 force-evaluation stop"
                    )
                _write_json(
                    output_dir / "raw.json",
                    {
                        "schema_version": 2,
                        "execution_commit": execution_commit,
                        "shared_provenance": shared_provenance,
                        "state_provenance": state_provenance,
                        "arms": list(selected_arms),
                        "shared_initial_directions": (
                            shared_initial_directions
                        ),
                        "cases": rows,
                    },
                )
    return {
        "execution_commit": execution_commit,
        "completed_cases": len(rows),
        "force_evaluations": total_force_evaluations,
    }


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "output",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARMS),
        default=None,
    )
    parser.add_argument(
        "--locked-source",
        type=Path,
        default=None,
        help=(
            "validated preceding cohort output containing exact persisted "
            "starter states"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(
        json.dumps(
            run(
                args.output,
                locked_source=args.locked_source,
                arms=args.arms,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
