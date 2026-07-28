#!/usr/bin/env python3
"""Run the preregistered direction-conditioned checkpoint shooting audit."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FIXED_STATE_AUDIT_PATH = (
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
CHECKPOINT_NAME = re.compile(
    r"^trial0001_proposal001_step(?P<step>[0-9]{3})_"
    r"proposal_relax[.]xyz$"
)
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
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
MEANINGFUL_ENERGY_DROP_EV = 0.001


def case_matrix() -> list[dict[str, Any]]:
    return [
        {"state_id": state_id, "seed": seed, "arm": arm}
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    ]


def classify_trajectory(
    checkpoints: Sequence[Mapping[str, Any]],
) -> str:
    if not checkpoints:
        raise ValueError("trajectory requires at least one checkpoint")
    observed = [int(row["step_index"]) for row in checkpoints]
    expected = list(range(1, len(checkpoints) + 1))
    if observed != expected:
        raise ValueError("checkpoint indices must be consecutive and ordered")
    productive = [bool(row["productive"]) for row in checkpoints]
    final_productive = productive[-1]
    earlier_productive = any(productive[:-1])
    if final_productive and earlier_productive:
        return "productive_earlier_and_final"
    if final_productive:
        return "productive_final"
    if earlier_productive:
        return "overshoot"
    return "no_productive_checkpoint"


def discover_checkpoint_paths(directory: Path) -> list[Path]:
    directory = Path(directory)
    paths = sorted(directory.glob("*proposal_relax.xyz"))
    if not paths:
        raise ValueError("trajectory directory contains no checkpoints")
    indexed: list[tuple[int, Path]] = []
    for path in paths:
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected checkpoint trajectory: {path.name}")
        indexed.append((int(match.group("step")), path))
    indexed.sort()
    steps = [step for step, _ in indexed]
    if steps != list(range(1, len(indexed) + 1)):
        raise ValueError("checkpoint trajectory steps are not consecutive")
    return [path for _, path in indexed]


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checkpoints = [
        checkpoint
        for row in rows
        for checkpoint in row["checkpoints"]
    ]
    return {
        "case_count": len(rows),
        "checkpoint_count": len(checkpoints),
        "certificate_count": sum(
            bool(checkpoint["certificate"]) for checkpoint in checkpoints
        ),
        "meaningful_checkpoint_count": sum(
            bool(checkpoint["productive"]) for checkpoint in checkpoints
        ),
        "generation_force_evaluations": sum(
            int(row["generation_force_evaluations"]) for row in rows
        ),
        "shooting_force_evaluations": sum(
            int(checkpoint["force_evaluations"])
            for checkpoint in checkpoints
        ),
        "classification_counts": dict(
            sorted(Counter(row["classification"] for row in rows).items())
        ),
    }


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
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("evidence requires the exact 12-case cohort")

    for row in rows:
        generation_purposes = row.get("generation_purpose_counts")
        if (
            row.get("status") != "completed"
            or not isinstance(generation_purposes, Mapping)
            or sum(int(value) for value in generation_purposes.values())
            != row.get("generation_force_evaluations")
            or generation_purposes.get("unattributed") != 0
        ):
            raise ValueError("generation purpose ledger does not close")
        if generation_purposes.get("direction_oracle") != (
            24 * int(row.get("direction_selection_count", -1))
        ):
            raise ValueError("direction ledger does not close")

        checkpoints = row.get("checkpoints")
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("case requires at least one checkpoint")
        expected_classification = classify_trajectory(checkpoints)
        if row.get("classification") != expected_classification:
            raise ValueError("trajectory classification does not close")

        for checkpoint in checkpoints:
            purposes = checkpoint.get("purpose_counts")
            if (
                checkpoint.get("status") != "completed"
                or not isinstance(checkpoint.get("certificate"), bool)
                or not isinstance(checkpoint.get("productive"), bool)
                or not isinstance(purposes, Mapping)
                or sum(int(value) for value in purposes.values())
                != checkpoint.get("force_evaluations")
                or purposes.get("unattributed") != 0
                or purposes.get("direction_oracle") != 0
                or purposes.get("biased_proposal_relax") != 0
                or purposes.get("landing_true_quench", 0) <= 0
            ):
                raise ValueError("checkpoint purpose ledger does not close")
            expected_productive = bool(
                checkpoint["certificate"]
                and checkpoint.get("is_new_basin")
                and _finite(
                    checkpoint["landing_delta_eV"],
                    "checkpoint landing delta",
                )
                < -MEANINGFUL_ENERGY_DROP_EV
            )
            if checkpoint["productive"] != expected_productive:
                raise ValueError("checkpoint productive flag does not close")

    classification_counts = dict(
        sorted(Counter(row["classification"] for row in rows).items())
    )
    aggregate = _aggregate(rows)
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "completed_cases": len(rows),
            "checkpoint_count": aggregate["checkpoint_count"],
        },
        "classification_counts": classification_counts,
        "certificate_count": aggregate["certificate_count"],
        "meaningful_checkpoint_count": aggregate[
            "meaningful_checkpoint_count"
        ],
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "state_arm_results": {
            state_id: {
                arm: _aggregate(
                    [
                        row
                        for row in rows
                        if row["state_id"] == state_id
                        and row["arm"] == arm
                    ]
                )
                for arm in ARMS
            }
            for state_id in STATE_IDS
        },
        "totals": aggregate,
        "production_default_changed": False,
        "claim_ceiling": (
            "descriptive direction-conditioned checkpoint shooting audit; "
            "no stopping rule, direction arm, or selector is promoted"
        ),
        "cases": list(rows),
    }


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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_locked_states_and_calculator():
    from pamssw.calculators import ASECalculator

    audit = _load_module(
        FIXED_STATE_AUDIT_PATH,
        "_checkpoint_shooting_fixed_state_audit",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    registry = audit.FIXED_STATE_REGISTRY["c60"]
    origin_summary = audit._validate_origin("c60", registry)
    runtime_files = audit._validate_bootstrap_runtime_files(
        system="c60",
        entry=registry[0],
        origin_summary=origin_summary,
        base_runner=base_runner,
    )
    template = base_runner.load_state("c60")
    states = {}
    provenance = {}
    by_id = {str(entry["state_id"]): entry for entry in registry}
    for state_id in STATE_IDS:
        state, state_provenance = audit._load_locked_state(
            by_id[state_id],
            template,
        )
        states[state_id] = state
        provenance[state_id] = state_provenance
    calculator = ASECalculator(base_runner._calculator())
    shared = {
        **runtime_files,
        "bootstrap_force_evaluations": 0,
        "bootstrap_skipped": True,
        "selection_rationale": (
            "only locked accepted states are used; no bootstrap minimum "
            "is reconstructed"
        ),
    }
    return states, calculator, base_runner, provenance, shared


def _state_from_checkpoint(path: Path, starter_state):
    from ase.io import read

    from pamssw.state import State

    atoms = read(path, index=-1)
    return State(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=(
            None
            if starter_state.cell is None
            else starter_state.cell.copy()
        ),
        pbc=starter_state.pbc,
        fixed_mask=starter_state.fixed_mask.copy(),
        metadata={"checkpoint_path": str(path)},
    )


def effective_checkpoint_state(
    starter_state,
    raw_checkpoint_state,
    *,
    max_displacement: float,
    _clipper=None,
):
    if _clipper is None:
        from pamssw.walker import SurfaceWalker

        _clipper = SurfaceWalker._clip_walk_displacement
    return _clipper(
        starter_state,
        raw_checkpoint_state,
        max_displacement,
    )


def _run_checkpoint(
    *,
    checkpoint_path: Path,
    step_index: int,
    starter_state,
    starter_energy: float,
    shared_calculator,
    base_runner,
    config,
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

    checkpoint_state = _state_from_checkpoint(
        checkpoint_path,
        starter_state,
    )
    checkpoint_config = replace(
        config,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        quench_optimizer="ase-lbfgs",
        quench_fallback_optimizer="ase-fire",
        quench_fmax=0.01,
        quench_maxiter=400,
    )
    walker = SurfaceWalker(
        calculator=shared_calculator,
        config=checkpoint_config,
        softening_enabled=True,
    )
    started = perf_counter()
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        checkpoint_evaluation = walker.calculator.evaluate(checkpoint_state)
    landing = walker.relax_true_minimum(
        checkpoint_state,
        trajectory_name=None,
    )
    wall_time = float(perf_counter() - started)

    archive = MinimaArchive(
        energy_tol=checkpoint_config.dedup_energy_tol,
        rmsd_tol=checkpoint_config.dedup_rmsd_tol,
        max_prototypes=checkpoint_config.max_prototypes,
    )
    seed_entry = archive.add(
        starter_state,
        starter_energy,
        parent_id=None,
    )
    before_count = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=seed_entry.entry_id,
    )
    certificate = bool(
        has_force_convergence_certificate(
            landing,
            checkpoint_config.quench_fmax,
        )
    )
    is_new_basin = len(archive.entries) > before_count
    landing_delta = float(landing.energy) - starter_energy
    purpose_counts = walker.calculator.snapshot().as_dict()
    force_evaluations = int(sum(purpose_counts.values()))
    if (
        purpose_counts["unattributed"] != 0
        or purpose_counts["direction_oracle"] != 0
        or purpose_counts["biased_proposal_relax"] != 0
        or purpose_counts["escape_true_pes_check"] != 1
        or purpose_counts["landing_true_quench"] <= 0
    ):
        raise RuntimeError("checkpoint purpose ledger does not close")
    diagnostics = walker.relaxation_diagnostics()
    landing_path = (
        case_dir
        / "checkpoint_landings"
        / f"step{step_index:03d}_landing.xyz"
    )
    landing_path.parent.mkdir(parents=True, exist_ok=True)
    base_runner.write_state(landing_path, landing.state)
    return {
        "step_index": step_index,
        "status": "completed",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_energy_eV": float(checkpoint_evaluation.energy),
        "checkpoint_delta_eV": (
            float(checkpoint_evaluation.energy) - starter_energy
        ),
        "landing_path": str(landing_path),
        "landing_sha256": _sha256(landing_path),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": landing_delta,
        "certificate": certificate,
        "final_max_force_eV_per_A": float(landing.gradient_norm),
        "is_new_basin": is_new_basin,
        "landing_entry_id": int(landing_entry.entry_id),
        "descriptor_delta": float(
            descriptor_distance(
                structural_descriptor(starter_state),
                structural_descriptor(landing.state),
            )
        ),
        "productive": bool(
            certificate
            and is_new_basin
            and landing_delta < -MEANINGFUL_ENERGY_DROP_EV
        ),
        "fallback_used": bool(
            diagnostics["quench_fallback_attempts"]
        ),
        "quench_iterations": int(landing.n_iter),
        "termination_reason": landing.telemetry.termination_reason,
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "wall_time_s": wall_time,
    }


def _run_case(
    *,
    state_id: str,
    seed: int,
    arm: str,
    starter_state,
    state_provenance: Mapping[str, Any],
    shared_calculator,
    base_runner,
    output_dir: Path,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.walker import SurfaceWalker

    direction_runner = _load_module(
        DIRECTION_RUNNER_PATH,
        "_checkpoint_shooting_direction_contract",
    )
    case_dir = (
        output_dir / "cases" / f"{state_id}-seed{seed}-{arm}"
    )
    trajectory_dir = case_dir / "proposal_trajectories"
    config = replace(
        base_runner.build_config("c60", case_dir),
        max_trials=1,
        max_force_evals=None,
        rng_seed=seed,
        write_relaxation_trajectories=True,
        relaxation_trajectory_dir=str(trajectory_dir),
        relaxation_trajectory_stride=1,
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
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        starter_evaluation = walker.calculator.evaluate(starter_state)
    starter_energy = float(starter_evaluation.energy)
    seed_entry = archive.add(
        starter_state,
        starter_energy,
        parent_id=None,
    )
    proposal = walker._proposal_pool(
        starter_state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=seed_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    generation_wall_time = float(perf_counter() - started)
    raw_checkpoint_paths = discover_checkpoint_paths(trajectory_dir)
    checkpoint_paths = []
    checkpoint_source_records = []
    for step_index, raw_checkpoint_path in enumerate(
        raw_checkpoint_paths,
        start=1,
    ):
        raw_checkpoint = _state_from_checkpoint(
            raw_checkpoint_path,
            starter_state,
        )
        effective_checkpoint, clipped = effective_checkpoint_state(
            starter_state,
            raw_checkpoint,
            max_displacement=config.walk_trust_radius,
        )
        if clipped and step_index != len(raw_checkpoint_paths):
            raise RuntimeError(
                "a nonterminal macro checkpoint requires walk clipping"
            )
        correction = float(
            np.max(
                np.abs(
                    effective_checkpoint.positions
                    - raw_checkpoint.positions
                )
            )
        )
        effective_path = (
            case_dir
            / "macro_checkpoints"
            / f"step{step_index:03d}_checkpoint.xyz"
        )
        effective_path.parent.mkdir(parents=True, exist_ok=True)
        base_runner.write_state(effective_path, effective_checkpoint)
        checkpoint_paths.append(effective_path)
        checkpoint_source_records.append(
            {
                "raw_optimizer_checkpoint_path": str(
                    raw_checkpoint_path
                ),
                "raw_optimizer_checkpoint_sha256": _sha256(
                    raw_checkpoint_path
                ),
                "walk_trust_radius_clipped": bool(clipped),
                "walk_trust_radius_correction_A": correction,
            }
        )
    final_checkpoint = _state_from_checkpoint(
        checkpoint_paths[-1],
        starter_state,
    )
    final_position_error = float(
        np.max(
            np.abs(
                final_checkpoint.positions - proposal.state.positions
            )
        )
    )
    if final_position_error > 1.0e-8:
        raise RuntimeError(
            "last checkpoint differs from proposal endpoint: "
            f"{final_position_error}"
        )

    direction_rows = _read_direction_rows(
        Path(config.direction_diagnostics_path)
    )
    direction_audit = direction_runner.validate_direction_trace(
        arm=arm,
        direction_rows=direction_rows,
    )
    generation_purpose_counts = (
        walker.calculator.snapshot().as_dict()
    )
    generation_force_evaluations = int(
        sum(generation_purpose_counts.values())
    )
    if (
        generation_purpose_counts["unattributed"] != 0
        or generation_purpose_counts["landing_true_quench"] != 0
        or generation_purpose_counts["direction_oracle"]
        != direction_audit["direction_oracle_force_evaluations"]
    ):
        raise RuntimeError("generation purpose ledger does not close")
    if any(
        int(row["oracle_selection_force_evaluations_delta"]) != 24
        for row in direction_rows
    ):
        raise RuntimeError(
            "a direction selection did not use exactly 24 evaluations"
        )

    checkpoints = []
    for step_index, checkpoint_path in enumerate(
        checkpoint_paths,
        start=1,
    ):
        print(
            f"[shoot] {state_id} seed={seed} arm={arm} "
            f"step={step_index}/{len(checkpoint_paths)}",
            flush=True,
        )
        checkpoint = _run_checkpoint(
            checkpoint_path=checkpoint_path,
            step_index=step_index,
            starter_state=starter_state,
            starter_energy=starter_energy,
            shared_calculator=shared_calculator,
            base_runner=base_runner,
            config=config,
            case_dir=case_dir,
        )
        checkpoint.update(checkpoint_source_records[step_index - 1])
        checkpoints.append(checkpoint)
        _write_json(
            case_dir / "partial_checkpoints.json",
            checkpoints,
        )
    classification = classify_trajectory(checkpoints)
    row = {
        "state_id": state_id,
        "state_sha256": state_provenance["state_sha256"],
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "starter_energy_eV": starter_energy,
        "generation_force_evaluations": (
            generation_force_evaluations
        ),
        "generation_purpose_counts": generation_purpose_counts,
        "generation_wall_time_s": generation_wall_time,
        "direction_selection_count": int(
            direction_audit["selection_count"]
        ),
        "direction_hvp_count": int(
            sum(
                int(row.get("krylov_hvp_consumed", 12))
                for row in direction_rows
            )
        ),
        "direction_trace": direction_rows,
        "effective_config": asdict(config),
        "checkpoint_count": len(checkpoints),
        "final_checkpoint_position_error_A": final_position_error,
        "checkpoints": checkpoints,
        "classification": classification,
    }
    _write_json(case_dir / "summary.json", row)
    return row


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
    ) = _load_locked_states_and_calculator()
    rows: list[dict[str, Any]] = []
    for case in case_matrix():
        row = _run_case(
            state_id=case["state_id"],
            seed=case["seed"],
            arm=case["arm"],
            starter_state=states[case["state_id"]],
            state_provenance=state_provenance[case["state_id"]],
            shared_calculator=shared_calculator,
            base_runner=base_runner,
            output_dir=output_dir,
        )
        rows.append(row)
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
                "classification_counts": evidence[
                    "classification_counts"
                ],
                "state_arm_results": evidence[
                    "state_arm_results"
                ],
                "totals": evidence["totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
