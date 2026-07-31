#!/usr/bin/env python3
"""Run the frozen C60/PdO D0/K4 escape first-passage gate."""

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
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
ACTION_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-action-family-transfer-gate"
    / "run_experiment.py"
)
SHOOTING_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-direction-conditioned-checkpoint-shooting"
    / "run_audit.py"
)
FIXED_AUDIT_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-block-krylov-direction-audit"
    / "run_fixed_state_audit.py"
)
CONFIG_GATE_PATH = (
    REPO_ROOT
    / "runs"
    / "20260730-starter-cell-online-gate"
    / "run_gate.py"
)
LOCKED_GATE_RELATIVE = (
    Path("runs")
    / "20260731-direction-candidate-counterfactual-gate"
    / "output"
    / "groups"
)
MAX_FORCE_EVALUATIONS = 15_000


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    PROTOCOL_PATH,
    "_current_action_first_passage_protocol_runner",
)


def select_reachable_checkpoints(
    checkpoints: Sequence[Any],
) -> list[tuple[int, Any]]:
    """Select only preregistered horizons that the generated path reached."""

    return [
        (horizon, checkpoints[horizon - 1])
        for horizon in protocol.CHECKPOINT_HORIZONS
        if horizon <= len(checkpoints)
    ]


def partition_checkpoint_attempts(
    attempted_states: Sequence[Any],
    proposal_endpoint,
    *,
    termination_reason: str,
    _prefix_resolver,
) -> tuple[list[Any], list[float | None], Any | None]:
    """Separate accepted macro states from one rejected invalid relaxation.

    Optimizer trajectory callbacks also persist a relaxation that the walker
    subsequently rejects as invalid.  Such a state is an attempted horizon,
    not the returned proposal endpoint.
    """

    try:
        accepted, errors = _prefix_resolver(
            attempted_states,
            proposal_endpoint,
            tolerance=1.0e-8,
        )
    except RuntimeError:
        if (
            termination_reason == "relaxed_geometry_invalid"
            and len(attempted_states) == 1
        ):
            return [], [None], attempted_states[0]
        raise
    failed = None
    if (
        termination_reason == "relaxed_geometry_invalid"
        and len(attempted_states) > len(accepted)
    ):
        failed = attempted_states[len(accepted)]
    return list(accepted), list(errors), failed


def _current_commit() -> str:
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _direction_hash(direction: np.ndarray) -> str:
    values = np.asarray(direction, dtype=float).reshape(-1)
    values = values / np.linalg.norm(values)
    return sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_starter(
    *,
    state_source_root: Path,
    system: str,
    state_id: str,
    audit,
    base_runner,
):
    from pamssw.io import read_state

    root = state_source_root / LOCKED_GATE_RELATIVE / system / state_id
    source = root / "seed-00000042" / "candidate-0"
    summary_path = source / "summary.json"
    state_path = source / "starter.xyz"
    if not summary_path.is_file() or not state_path.is_file():
        raise FileNotFoundError(source)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    template = base_runner.load_state(system)
    state = read_state(state_path, fixed_mask=template.fixed_mask)
    state_hash = audit._state_sha256(state)
    if state_hash != summary["state_sha256"]:
        raise RuntimeError(f"{system} {state_id} starter hash drifted")
    for seed in protocol.SEEDS:
        peer = (
            root
            / f"seed-{seed:08d}"
            / "candidate-0"
            / "summary.json"
        )
        if not peer.is_file():
            raise FileNotFoundError(peer)
        if json.loads(peer.read_text(encoding="utf-8"))[
            "state_sha256"
        ] != state_hash:
            raise RuntimeError(f"{system} {state_id} peer starter drifted")
    return state, {
        "state_sha256": state_hash,
        "source_summary_path": str(summary_path),
        "source_state_path": str(state_path),
    }


def _generate_action_path(
    *,
    state,
    provenance: Mapping[str, Any],
    calculator,
    base_config,
    system: str,
    state_id: str,
    seed: int,
    arm: str,
    case_dir: Path,
    remaining_budget: int,
    action_runner,
    shooting_runner,
    base_runner,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.walker import SurfaceWalker

    case_dir.mkdir(parents=True, exist_ok=False)
    trajectory_dir = case_dir / "proposal_trajectories"
    direction_path = case_dir / "direction.jsonl"
    config = replace(
        base_config,
        max_force_evals=remaining_budget,
        write_relaxation_trajectories=True,
        relaxation_trajectory_dir=str(trajectory_dir),
        relaxation_trajectory_stride=1,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(direction_path),
        **action_runner.ARM_CONFIGS[arm],
    )

    class AnchorAuditWalker(SurfaceWalker):
        gate_anchor_sha256: str | None = None

        def _initialize_walk_direction_context(self, current, *, trial_index):
            anchor, intents = super()._initialize_walk_direction_context(
                current,
                trial_index=trial_index,
            )
            anchor_hash = _direction_hash(anchor)
            if self.gate_anchor_sha256 is None:
                self.gate_anchor_sha256 = anchor_hash
            elif self.gate_anchor_sha256 != anchor_hash:
                raise RuntimeError("one action regenerated a different anchor")
            return anchor, intents

    walker = AnchorAuditWalker(
        calculator=calculator,
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
    starter_entry = archive.add(state, starter_energy, parent_id=None)
    proposal = walker._proposal_pool(
        state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=starter_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    generation_wall_time = float(perf_counter() - started)

    raw_paths = shooting_runner.discover_checkpoint_paths(trajectory_dir)
    attempted_paths: list[Path] = []
    attempted_states = []
    source_records = []
    for step_index, raw_path in enumerate(raw_paths, start=1):
        raw_state = shooting_runner._state_from_checkpoint(raw_path, state)
        effective_state, clipped = shooting_runner.effective_checkpoint_state(
            state,
            raw_state,
            max_displacement=config.walk_trust_radius,
        )
        if clipped and step_index != len(raw_paths):
            raise RuntimeError("nonterminal macro checkpoint was clipped")
        effective_path = (
            case_dir
            / "macro_checkpoints"
            / f"step{step_index:03d}_checkpoint.xyz"
        )
        effective_path.parent.mkdir(parents=True, exist_ok=True)
        base_runner.write_state(effective_path, effective_state)
        attempted_paths.append(effective_path)
        attempted_states.append(effective_state)
        source_records.append(
            {
                "raw_optimizer_checkpoint_path": str(raw_path),
                "raw_optimizer_checkpoint_sha256": _file_sha256(raw_path),
                "walk_trust_radius_clipped": bool(clipped),
            }
        )
    termination_reason = walker._walk_termination_last_reason
    accepted_states, endpoint_errors, failed_attempt = (
        partition_checkpoint_attempts(
            attempted_states,
            proposal.state,
            termination_reason=termination_reason,
            _prefix_resolver=(
                shooting_runner.accepted_checkpoint_prefix
            ),
        )
    )
    accepted_count = len(accepted_states)
    accepted_paths = attempted_paths[:accepted_count]
    accepted_sources = source_records[:accepted_count]

    direction_rows = _read_jsonl(direction_path)
    direction_audit = action_runner._validate_direction_trace(
        arm,
        direction_rows,
    )
    counts = walker.calculator.snapshot().as_dict()
    force_evaluations = walker.calculator.force_evaluations
    if (
        counts["unattributed"] != 0
        or sum(counts.values()) != force_evaluations
        or counts["direction_oracle"]
        != direction_audit["direction_force_evaluations"]
    ):
        raise RuntimeError("generation purpose ledger does not close")
    selected = []
    for horizon, path in select_reachable_checkpoints(accepted_paths):
        record = dict(accepted_sources[horizon - 1])
        record.update(
            {
                "horizon": horizon,
                "checkpoint_path": str(path),
                "checkpoint_sha256": _file_sha256(path),
                "endpoint_position_error_A": endpoint_errors[horizon - 1],
            }
        )
        selected.append(record)
    failed_attempt_record = None
    if failed_attempt is not None:
        failed_index = accepted_count
        failed_horizon = failed_index + 1
        failed_path = attempted_paths[failed_index]
        failed_attempt_record = {
            **source_records[failed_index],
            "horizon": failed_horizon,
            "checkpoint_path": str(failed_path),
            "checkpoint_sha256": _file_sha256(failed_path),
            "endpoint_position_error_A": endpoint_errors[failed_index],
            "preclassified_label": "INVALID_GEOMETRY",
        }
        if failed_horizon in protocol.CHECKPOINT_HORIZONS:
            selected.append(failed_attempt_record)
    return {
        "system": system,
        "state_id": state_id,
        "state_sha256": provenance["state_sha256"],
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "anchor_sha256": walker.gate_anchor_sha256,
        "starter_energy_eV": starter_energy,
        "generation_force_evaluations": force_evaluations,
        "generation_purpose_counts": counts,
        "generation_wall_time_s": generation_wall_time,
        "walk_termination_reason": termination_reason,
        "reached_macro_steps": accepted_count,
        "attempted_macro_steps": len(attempted_paths),
        "terminal_failed_attempt": failed_attempt_record,
        "selected_checkpoints": selected,
        "direction_audit": direction_audit,
        "direction_trace": direction_rows,
        "effective_config": asdict(config),
    }


def _checkpoint_failure_row(
    *,
    source: Mapping[str, Any],
    label: str,
    purpose_counts: Mapping[str, int],
    wall_time_s: float,
) -> dict[str, Any]:
    return {
        **source,
        "status": "completed",
        "label": label,
        "checkpoint_energy_eV": None,
        "checkpoint_delta_eV": None,
        "landing_energy_eV": None,
        "landing_delta_eV": None,
        "certificate": False,
        "geometry_valid": label != "INVALID_GEOMETRY",
        "fragmented": label == "FRAGMENTED",
        "matcher_same": None,
        "descriptor_same": None,
        "descriptor_delta": None,
        "force_evaluations": sum(purpose_counts.values()),
        "purpose_counts": dict(purpose_counts),
        "wall_time_s": wall_time_s,
    }


def _quench_checkpoint(
    *,
    source: Mapping[str, Any],
    starter_state,
    starter_energy: float,
    calculator,
    config,
    system: str,
    case_dir: Path,
    remaining_budget: int,
    base_runner,
) -> dict[str, Any]:
    from pamssw.accounting import BudgetExceeded, EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.fingerprint import (
        descriptor_distance,
        structural_descriptor,
    )
    from pamssw.io import read_state
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    if source.get("preclassified_label") == "INVALID_GEOMETRY":
        zero_counts = {
            "bootstrap_true_quench": 0,
            "starter_true_quench": 0,
            "local_softening_pre_relax": 0,
            "direction_oracle": 0,
            "escape_true_pes_check": 0,
            "biased_proposal_relax": 0,
            "landing_true_quench": 0,
            "post_relax_validation": 0,
            "unattributed": 0,
        }
        return _checkpoint_failure_row(
            source=source,
            label="INVALID_GEOMETRY",
            purpose_counts=zero_counts,
            wall_time_s=0.0,
        )

    checkpoint_state = read_state(
        Path(source["checkpoint_path"]),
        fixed_mask=starter_state.fixed_mask,
    )
    checkpoint_config = replace(
        config,
        max_force_evals=remaining_budget,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
    )
    walker = SurfaceWalker(
        calculator=calculator,
        config=checkpoint_config,
        softening_enabled=True,
    )
    started = perf_counter()
    if not walker.geometry_validator.is_valid_state(checkpoint_state):
        counts = walker.calculator.snapshot().as_dict()
        return _checkpoint_failure_row(
            source=source,
            label="INVALID_GEOMETRY",
            purpose_counts=counts,
            wall_time_s=float(perf_counter() - started),
        )

    checkpoint_energy = None
    try:
        with walker.calculator.purpose(
            EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
        ):
            checkpoint_energy = float(
                walker.calculator.evaluate(checkpoint_state).energy
            )
        landing = walker.relax_true_minimum(
            checkpoint_state,
            trajectory_name=None,
        )
    except BudgetExceeded as error:
        counts = walker.calculator.snapshot().as_dict()
        label = (
            "INVALID_GEOMETRY"
            if "invalid geometry" in str(error)
            else "BUDGET_EXHAUSTED"
        )
        row = _checkpoint_failure_row(
            source=source,
            label=label,
            purpose_counts=counts,
            wall_time_s=float(perf_counter() - started),
        )
        row["checkpoint_energy_eV"] = checkpoint_energy
        row["checkpoint_delta_eV"] = (
            None
            if checkpoint_energy is None
            else checkpoint_energy - starter_energy
        )
        return row

    certificate = bool(
        has_force_convergence_certificate(
            landing,
            checkpoint_config.quench_fmax,
        )
    )
    geometry_valid = bool(
        walker.geometry_validator.is_valid_state(landing.state)
    )
    fragmented = bool(
        walker._is_fragmented_cluster(starter_state, landing.state)
        if system == "c60"
        else False
    )
    archive = MinimaArchive(
        energy_tol=checkpoint_config.dedup_energy_tol,
        rmsd_tol=checkpoint_config.dedup_rmsd_tol,
        max_prototypes=checkpoint_config.max_prototypes,
    )
    archive.add(starter_state, starter_energy, parent_id=None)
    matcher_same = (
        archive.find_match(landing.state, float(landing.energy)) is not None
    )
    descriptor_delta = float(
        descriptor_distance(
            structural_descriptor(starter_state),
            structural_descriptor(landing.state),
        )
    )
    descriptor_same = (
        descriptor_delta < checkpoint_config.min_escape_descriptor_delta
    )
    label = protocol.classify_checkpoint(
        budget_exhausted=False,
        fragmented=fragmented,
        geometry_valid=geometry_valid,
        certificate=certificate,
        matcher_same=matcher_same,
        descriptor_same=descriptor_same,
    )
    landing_path = (
        case_dir
        / "checkpoint_landings"
        / f"h{int(source['horizon']):02d}_landing.xyz"
    )
    landing_path.parent.mkdir(parents=True, exist_ok=True)
    base_runner.write_state(landing_path, landing.state)
    counts = walker.calculator.snapshot().as_dict()
    force_evaluations = walker.calculator.force_evaluations
    if (
        counts["unattributed"] != 0
        or counts["direction_oracle"] != 0
        or counts["biased_proposal_relax"] != 0
        or sum(counts.values()) != force_evaluations
    ):
        raise RuntimeError("checkpoint purpose ledger does not close")
    return {
        **source,
        "status": "completed",
        "label": label,
        "checkpoint_energy_eV": checkpoint_energy,
        "checkpoint_delta_eV": checkpoint_energy - starter_energy,
        "landing_path": str(landing_path),
        "landing_sha256": _file_sha256(landing_path),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "certificate": certificate,
        "geometry_valid": geometry_valid,
        "fragmented": fragmented,
        "matcher_same": matcher_same,
        "descriptor_same": descriptor_same,
        "descriptor_delta": descriptor_delta,
        "final_max_force_eV_per_A": float(landing.gradient_norm),
        "quench_iterations": int(landing.n_iter),
        "termination_reason": landing.telemetry.termination_reason,
        "force_evaluations": force_evaluations,
        "purpose_counts": counts,
        "wall_time_s": float(perf_counter() - started),
    }


def run(
    *,
    output_dir: Path,
    expected_commit: str,
    state_source_root: Path,
    systems: Sequence[str],
    state_ids: Sequence[str],
    seeds: Sequence[int],
    arms: Sequence[str],
    max_force_evaluations: int,
) -> dict[str, Any]:
    if _current_commit() != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    from pamssw.calculators import ASECalculator

    audit = _load_module(FIXED_AUDIT_PATH, "_first_passage_fixed_audit")
    config_gate = _load_module(CONFIG_GATE_PATH, "_first_passage_config_gate")
    action_runner = _load_module(
        ACTION_RUNNER_PATH,
        "_first_passage_action_runner",
    )
    shooting_runner = _load_module(
        SHOOTING_RUNNER_PATH,
        "_first_passage_shooting_runner",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    state_source_root = state_source_root.resolve()
    if not (state_source_root / LOCKED_GATE_RELATIVE).is_dir():
        raise FileNotFoundError(state_source_root / LOCKED_GATE_RELATIVE)
    calculator = ASECalculator(base_runner._calculator())

    started = perf_counter()
    total_force_evaluations = 0
    rows: list[dict[str, Any]] = []
    for system in systems:
        for state_id in state_ids:
            state, provenance = _load_starter(
                state_source_root=state_source_root,
                system=system,
                state_id=state_id,
                audit=audit,
                base_runner=base_runner,
            )
            for seed in seeds:
                anchor_hashes = {}
                for arm in arms:
                    if total_force_evaluations >= max_force_evaluations:
                        raise RuntimeError(
                            "budget exhausted before all generation paths"
                        )
                    case_dir = (
                        output_dir
                        / "cases"
                        / system
                        / state_id
                        / f"seed-{seed:08d}"
                        / arm
                    )
                    print(
                        f"[first-passage] {system} {state_id} "
                        f"seed={seed} arm={arm}",
                        flush=True,
                    )
                    base_config = action_runner._base_config(
                        config_gate,
                        system,
                        seed,
                        case_dir,
                    )
                    generation = _generate_action_path(
                        state=state,
                        provenance=provenance,
                        calculator=calculator,
                        base_config=base_config,
                        system=system,
                        state_id=state_id,
                        seed=seed,
                        arm=arm,
                        case_dir=case_dir,
                        remaining_budget=(
                            max_force_evaluations - total_force_evaluations
                        ),
                        action_runner=action_runner,
                        shooting_runner=shooting_runner,
                        base_runner=base_runner,
                    )
                    total_force_evaluations += int(
                        generation["generation_force_evaluations"]
                    )
                    checkpoints = []
                    for source in generation.pop("selected_checkpoints"):
                        checkpoint = _quench_checkpoint(
                            source=source,
                            starter_state=state,
                            starter_energy=generation["starter_energy_eV"],
                            calculator=calculator,
                            config=base_config,
                            system=system,
                            case_dir=case_dir,
                            remaining_budget=max(
                                0,
                                max_force_evaluations
                                - total_force_evaluations,
                            ),
                            base_runner=base_runner,
                        )
                        checkpoints.append(checkpoint)
                        total_force_evaluations += int(
                            checkpoint["force_evaluations"]
                        )
                        if checkpoint["label"] == "BUDGET_EXHAUSTED":
                            break
                    generation["checkpoints"] = checkpoints
                    generation["trajectory_summary"] = (
                        protocol.summarize_trajectory(checkpoints)
                    )
                    anchor_hashes[arm] = generation["anchor_sha256"]
                    rows.append(generation)
                    _write_json(case_dir / "summary.json", generation)
                    _write_json(
                        output_dir / "partial.json",
                        {
                            "schema_version": 1,
                            "execution_commit": expected_commit,
                            "cases": rows,
                            "total_force_evaluations": (
                                total_force_evaluations
                            ),
                        },
                    )
                if len(set(anchor_hashes.values())) != 1:
                    raise RuntimeError(
                        f"anchor mismatch for {system} {state_id} seed {seed}"
                    )

    full_matrix = (
        tuple(systems) == protocol.SYSTEMS
        and tuple(state_ids) == protocol.STATE_IDS
        and tuple(seeds) == protocol.SEEDS
        and tuple(arms) == protocol.ARMS
    )
    if full_matrix:
        evidence = protocol.build_evidence(
            rows,
            max_force_evaluations=max_force_evaluations,
        )
    else:
        evidence = {
            "schema_version": 1,
            "smoke_only": True,
            "cases": rows,
            "total_force_evaluations": total_force_evaluations,
            "max_force_evaluations": max_force_evaluations,
        }
    evidence.update(
        {
            "execution_commit": expected_commit,
            "state_source_root": str(state_source_root),
            "wall_time_s": float(perf_counter() - started),
        }
    )
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--state-source-root", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(protocol.SYSTEMS))
    parser.add_argument("--state-ids", nargs="+", default=list(protocol.STATE_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(protocol.SEEDS))
    parser.add_argument("--arms", nargs="+", default=list(protocol.ARMS))
    parser.add_argument(
        "--max-force-evaluations",
        type=int,
        default=MAX_FORCE_EVALUATIONS,
    )
    args = parser.parse_args()
    evidence = run(
        output_dir=args.output_dir.resolve(),
        expected_commit=args.expected_commit,
        state_source_root=args.state_source_root,
        systems=tuple(args.systems),
        state_ids=tuple(args.state_ids),
        seeds=tuple(args.seeds),
        arms=tuple(args.arms),
        max_force_evaluations=args.max_force_evaluations,
    )
    print(
        json.dumps(
            {
                "smoke_only": evidence.get("smoke_only", False),
                "total_force_evaluations": evidence[
                    "total_force_evaluations"
                ],
                "wall_time_s": evidence["wall_time_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
