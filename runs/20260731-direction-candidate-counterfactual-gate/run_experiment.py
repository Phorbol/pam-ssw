#!/usr/bin/env python3
"""Execute the preregistered shared-K4 candidate counterfactual gate."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
MAX_TOTAL_FORCE_EVALUATIONS = 60_000


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
    "_direction_candidate_counterfactual_protocol_runner",
)


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


def _direction_hash(direction: np.ndarray) -> str:
    normalized = np.asarray(direction, dtype=float).reshape(-1)
    normalized = normalized / np.linalg.norm(normalized)
    return sha256(np.asarray(normalized, dtype="<f8").tobytes()).hexdigest()


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"direction diagnostics were not written: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _base_config(
    *,
    config_gate,
    system: str,
    seed: int,
    group_dir: Path,
):
    config = config_gate.build_production_config(
        system,
        group_dir,
        master_seed=seed,
    )
    return replace(
        config,
        max_trials=1,
        max_force_evals=None,
        oracle_candidates=4,
        direction_selection_mode="discrete",
        direction_synthesis_mode="none",
        direction_probe_enabled=False,
        direction_type_ucb_enabled=False,
        direction_archive_enabled=False,
        direction_archive_path=None,
        plateau_evolution_enabled=False,
        accepted_structures_log=None,
        accepted_structures_dir=None,
        write_proposal_minima=False,
        proposal_minima_dir=None,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
        proposal_pool_size=1,
        proposal_duplicate_rescue_optimizer=None,
    )


def _precompute_pool(
    *,
    state,
    calculator,
    config,
):
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.walker import DirectionChoice, ProposalPotential, SurfaceWalker

    walker = SurfaceWalker(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )
    if walker.rng is not walker.oracle.rng:
        raise RuntimeError("walker and direction oracle no longer share one RNG")
    current, frozen_softening = walker._prepare_frozen_local_softening(state)
    if current is not state or frozen_softening is not None:
        raise RuntimeError(
            "counterfactual gate requires the moving-reference LS protocol"
        )
    anchor_direction, _ = walker._initialize_walk_direction_context(
        current,
        trial_index=0,
    )
    softening = walker._build_softening(current, anchor_direction)
    oracle_softening = (
        softening if walker._softening_scope_enabled("oracle") else None
    )
    proposal = ProposalPotential(
        walker.calculator,
        biases=[],
        softening=oracle_softening,
    )
    scoring_proposal = walker._direction_scoring_proposal(proposal)
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    archive.add(current, 0.0, parent_id=None)
    candidates = walker.oracle.generator.generate(
        current,
        None,
        anchor_direction=anchor_direction,
        anchor_mixing_alpha=config.anchor_mixing_alpha,
        n_bond_pairs=config.n_bond_pairs,
    )
    if len(candidates) != 4:
        raise RuntimeError(
            f"shared native pool must contain four candidates, got {len(candidates)}"
        )
    score_sigma_fn = walker._direction_score_sigma_fn(1.0)
    score_sigma = (
        None
        if config.direction_score_sigma_mode == "adaptive"
        else walker._direction_score_sigma(1.0)
    )
    scoring_anchor = (
        None
        if config.anchor_mixing_alpha is not None
        else anchor_direction
    )
    evaluated: list[dict[str, Any]] = []
    with walker.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        for index, candidate in enumerate(candidates):
            total_hvp, true_hvp = walker.oracle._candidate_directional_hvps(
                current,
                scoring_proposal,
                candidate.direction,
            )
            curvature = float(np.dot(total_hvp, candidate.direction))
            true_curvature = float(
                np.dot(true_hvp, candidate.direction)
            )
            sigma = walker.oracle._candidate_score_sigma(
                curvature=curvature,
                score_sigma=score_sigma,
                score_sigma_fn=score_sigma_fn,
                step_scale_fn=None,
            )
            score = walker.oracle.scorer.score_candidate(
                state=current,
                candidate=candidate,
                curvature=curvature,
                sigma=sigma,
                previous_direction=None,
                anchor_direction=scoring_anchor,
                archive=archive,
                history_push=0.0,
                continuity_weight=config.continuity_weight,
            )
            evaluated.append(
                {
                    "candidate_index": index,
                    "candidate": candidate,
                    "curvature": curvature,
                    "true_curvature": true_curvature,
                    "score_sigma": sigma,
                    "static_score": float(score),
                    "direction_sha256": _direction_hash(
                        candidate.direction
                    ),
                }
            )
    direction_evaluations = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )
    if direction_evaluations != 8:
        raise RuntimeError(
            "four native central-FD candidate HVPs must cost exactly 8 force evaluations"
        )
    ordered = sorted(
        evaluated,
        key=lambda row: (
            -float(row["static_score"]),
            int(row["candidate_index"]),
        ),
    )
    ranks = {
        int(row["candidate_index"]): rank
        for rank, row in enumerate(ordered, start=1)
    }
    choices: list[DirectionChoice] = []
    records: list[dict[str, Any]] = []
    for row in evaluated:
        candidate = row.pop("candidate")
        rank = ranks[int(row["candidate_index"])]
        diagnostics = {
            "direction_hvp_count": 1,
            "counterfactual_shared_pool": True,
            "counterfactual_candidate_index": int(
                row["candidate_index"]
            ),
            "counterfactual_static_rank": rank,
            "counterfactual_score_sigma": float(row["score_sigma"]),
        }
        choices.append(
            DirectionChoice(
                direction=np.asarray(candidate.direction, dtype=float).copy(),
                curvature=float(row["curvature"]),
                kind=candidate.kind,
                candidate_count=4,
                mean_rigid_body_overlap=float(
                    candidate.rigid_body_overlap
                ),
                mean_post_projection_rigid_body_overlap=float(
                    candidate.post_projection_rigid_body_overlap
                ),
                score=float(row["static_score"]),
                true_curvature=float(row["true_curvature"]),
                diagnostics=diagnostics,
            )
        )
        records.append(
            {
                **row,
                "kind": candidate.kind.value,
                "static_rank": rank,
                "rigid_body_overlap": float(
                    candidate.rigid_body_overlap
                ),
                "post_projection_rigid_body_overlap": float(
                    candidate.post_projection_rigid_body_overlap
                ),
            }
        )
    return {
        "choices": choices,
        "records": records,
        "anchor_sha256": _direction_hash(anchor_direction),
        "post_pool_rng_state": deepcopy(walker.rng.bit_generator.state),
        "purpose_counts": walker.calculator.snapshot().as_dict(),
        "force_evaluations": walker.calculator.force_evaluations,
    }


def _run_forced_candidate(
    *,
    state,
    state_provenance: Mapping[str, Any],
    calculator,
    config,
    pool: Mapping[str, Any],
    candidate_record: Mapping[str, Any],
    candidate_choice,
    case_dir: Path,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.io import write_state
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    class PostPoolRNGWalker(SurfaceWalker):
        def _initialize_walk_direction_context(
            self,
            current,
            *,
            trial_index,
        ):
            anchor, intents = super()._initialize_walk_direction_context(
                current,
                trial_index=trial_index,
            )
            if _direction_hash(anchor) != pool["anchor_sha256"]:
                raise RuntimeError(
                    "forced arm regenerated a different initial anchor"
                )
            self.rng.bit_generator.state = deepcopy(
                pool["post_pool_rng_state"]
            )
            return anchor, intents

    case_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_path = case_dir / "direction.jsonl"
    case_config = replace(
        config,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(diagnostics_path),
    )
    walker = PostPoolRNGWalker(
        calculator=calculator,
        config=case_config,
        softening_enabled=True,
    )
    walker._reset_direction_diagnostics()
    archive = MinimaArchive(
        energy_tol=case_config.dedup_energy_tol,
        rmsd_tol=case_config.dedup_rmsd_tol,
        max_prototypes=case_config.max_prototypes,
    )
    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        starter_evaluation = walker.calculator.evaluate(state)
    starter_energy = float(starter_evaluation.energy)
    starter_entry = archive.add(state, starter_energy, parent_id=None)
    escape_state = walker._walk_candidate_from_seed(
        state,
        archive,
        walker.step_target_controller.target(archive),
        trial_index=0,
        proposal_index=0,
        seed_entry_id=starter_entry.entry_id,
        initial_direction_choice=candidate_choice,
    )
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    generation_wall_time = float(perf_counter() - started)
    quench_started = perf_counter()
    landing = walker.relax_true_minimum(
        escape_state,
        trajectory_name="counterfactual-landing",
    )
    quench_wall_time = float(perf_counter() - quench_started)
    before_count = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=starter_entry.entry_id,
    )
    direction_rows = _read_direction_rows(diagnostics_path)
    expected_hash = str(candidate_record["direction_sha256"])
    if (
        not direction_rows
        or direction_rows[0].get("selected_direction_sha256")
        != expected_hash
        or direction_rows[0].get("shared_initial_direction") is not True
        or direction_rows[0].get(
            "oracle_selection_force_evaluations_delta"
        )
        != 0
    ):
        raise RuntimeError(
            "forced case did not execute the exact shared candidate at step zero"
        )
    purpose_counts = walker.calculator.snapshot().as_dict()
    if purpose_counts["unattributed"] != 0:
        raise RuntimeError("forced case contains unattributed evaluations")
    if sum(purpose_counts.values()) != walker.calculator.force_evaluations:
        raise RuntimeError("forced case purpose ledger does not close")
    certificate = bool(
        has_force_convergence_certificate(
            landing,
            case_config.quench_fmax,
        )
    )
    geometry_valid = bool(
        walker.geometry_validator.is_valid_state(landing.state)
    )
    fragmented = bool(
        walker._is_fragmented_cluster(state, landing.state)
        if candidate_record["system"] == "c60"
        else False
    )
    write_state(case_dir / "starter.xyz", state)
    write_state(case_dir / "escape.xyz", escape_state)
    write_state(case_dir / "landing.xyz", landing.state)
    row = {
        **{
            key: candidate_record[key]
            for key in (
                "system",
                "state_id",
                "seed",
                "candidate_index",
                "kind",
                "static_rank",
                "static_score",
                "score_sigma",
                "curvature",
                "true_curvature",
                "direction_sha256",
            )
        },
        "status": "completed",
        "state_sha256": state_provenance["state_sha256"],
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "certificate": certificate,
        "landing_geometry_valid": geometry_valid,
        "fragmented": fragmented,
        "is_new_basin": len(archive.entries) > before_count,
        "landing_entry_id": int(landing_entry.entry_id),
        "landing_iterations": int(landing.n_iter),
        "landing_gradient_norm_eV_per_A": float(
            landing.gradient_norm
        ),
        "landing_termination_reason": (
            landing.telemetry.termination_reason
        ),
        "walk_termination_reason": walker._walk_termination_last_reason,
        "direction_selection_count": len(direction_rows),
        "purpose_counts": purpose_counts,
        "force_evaluations": walker.calculator.force_evaluations,
        "generation_wall_time_s": generation_wall_time,
        "quench_wall_time_s": quench_wall_time,
        "wall_time_s": generation_wall_time + quench_wall_time,
        "effective_config": asdict(case_config),
        "direction_trace": direction_rows,
    }
    _write_json(case_dir / "summary.json", row)
    return row


def _render_conclusion(evidence: Mapping[str, Any]) -> str:
    analysis = evidence["analysis"]
    lines = [
        "# Shared-candidate direction gate result",
        "",
        "## Decision",
        "",
        f"- Overall classification: **{analysis['overall']['classification']}**.",
        (
            "- Cross-system posterior stage: "
            f"**{analysis['overall']['cross_system_posterior_stage_allowed']}**."
        ),
        (
            f"- Total force evaluations: {evidence['total_force_evaluations']} "
            f"(shared pools {evidence['shared_pool_force_evaluations']}, "
            f"forced cases {evidence['case_force_evaluations']})."
        ),
        f"- Total wall time: {evidence['wall_time_s']:.1f} s.",
        "",
        "## System-level readout",
        "",
        "| System | Classification | Pools | Winner misses | Miss fraction | Median regret (eV) | Median Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for system, result in analysis["by_system"].items():
        lines.append(
            f"| {system} | {result['classification']} | "
            f"{result['comparable_group_count']} | "
            f"{result['static_winner_miss_count']} | "
            f"{result['static_winner_miss_fraction']:.3f} | "
            f"{result['median_static_winner_regret_eV']:.6f} | "
            f"{result['median_static_score_terminal_spearman']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The gate uses exact shared candidate pools and terminal true-PES",
            "quenches. It does not claim that a direction-family posterior is",
            "useful unless both systems independently satisfy the preregistered",
            "selection-bottleneck rule.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    output_dir: Path,
    expected_commit: str,
    state_source_root: Path,
    systems: Sequence[str],
    state_ids: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    if _current_commit() != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if any(system not in protocol.SYSTEMS for system in systems):
        raise ValueError("unknown system")
    if any(state_id not in protocol.STATE_IDS for state_id in state_ids):
        raise ValueError("unknown state id")
    output_dir.mkdir(parents=True)

    from pamssw.calculators import ASECalculator

    audit = _load_module(
        FIXED_AUDIT_PATH,
        "_counterfactual_fixed_state_audit",
    )
    config_gate = _load_module(
        CONFIG_GATE_PATH,
        "_counterfactual_config_gate",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    state_source_root = Path(state_source_root).resolve()
    if not state_source_root.is_dir():
        raise FileNotFoundError(state_source_root)
    # Only provenance-bearing historical outputs are resolved from this
    # explicit root.  Code and runtime configuration remain pinned to the
    # execution commit in the current worktree.
    audit.REPO_ROOT = state_source_root
    calculator = ASECalculator(base_runner._calculator())
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    pool_records: list[dict[str, Any]] = []
    shared_pool_force_evaluations = 0
    case_force_evaluations = 0

    for system in systems:
        registry = audit.FIXED_STATE_REGISTRY[system]
        audit._validate_origin(system, registry)
        template = base_runner.load_state(system)
        entries = {
            entry["state_id"]: entry
            for entry in registry
            if entry["state_id"] in state_ids
        }
        if set(entries) != set(state_ids):
            raise RuntimeError(f"{system} locked state registry is incomplete")
        for state_id in state_ids:
            state, provenance = audit._load_locked_state(
                entries[state_id],
                template,
            )
            for seed in seeds:
                group_dir = (
                    output_dir
                    / "groups"
                    / system
                    / state_id
                    / f"seed-{seed:08d}"
                )
                config = _base_config(
                    config_gate=config_gate,
                    system=system,
                    seed=seed,
                    group_dir=group_dir,
                )
                pool = _precompute_pool(
                    state=state,
                    calculator=calculator,
                    config=config,
                )
                shared_pool_force_evaluations += int(
                    pool["force_evaluations"]
                )
                records = []
                for record in pool["records"]:
                    records.append(
                        {
                            **record,
                            "system": system,
                            "state_id": state_id,
                            "seed": seed,
                        }
                    )
                pool_record = {
                    "system": system,
                    "state_id": state_id,
                    "seed": seed,
                    "state_sha256": provenance["state_sha256"],
                    "anchor_sha256": pool["anchor_sha256"],
                    "force_evaluations": pool["force_evaluations"],
                    "purpose_counts": pool["purpose_counts"],
                    "candidates": records,
                }
                pool_records.append(pool_record)
                _write_json(group_dir / "shared_pool.json", pool_record)
                for record, choice in zip(records, pool["choices"]):
                    candidate_index = int(record["candidate_index"])
                    print(
                        f"[counterfactual] {system} {state_id} "
                        f"seed={seed} candidate={candidate_index} "
                        f"kind={record['kind']} rank={record['static_rank']}",
                        flush=True,
                    )
                    case_dir = (
                        group_dir / f"candidate-{candidate_index}"
                    )
                    row = _run_forced_candidate(
                        state=state,
                        state_provenance=provenance,
                        calculator=calculator,
                        config=config,
                        pool=pool,
                        candidate_record=record,
                        candidate_choice=choice,
                        case_dir=case_dir,
                    )
                    rows.append(row)
                    case_force_evaluations += int(
                        row["force_evaluations"]
                    )
                    if (
                        shared_pool_force_evaluations
                        + case_force_evaluations
                        > MAX_TOTAL_FORCE_EVALUATIONS
                    ):
                        raise RuntimeError(
                            "campaign exceeded its 60000 force-evaluation stop"
                        )
                    _write_json(
                        output_dir / "partial.json",
                        {
                            "execution_commit": expected_commit,
                            "shared_pools": pool_records,
                            "rows": rows,
                        },
                    )

    analysis = protocol.summarize_campaign(rows)
    wall_time = float(perf_counter() - started)
    evidence = {
        "schema_version": 1,
        "execution_commit": expected_commit,
        "state_source_root": str(state_source_root),
        "systems": list(systems),
        "state_ids": list(state_ids),
        "seeds": list(seeds),
        "candidate_count": 4,
        "shared_pool_count": len(pool_records),
        "case_count": len(rows),
        "shared_pool_force_evaluations": (
            shared_pool_force_evaluations
        ),
        "case_force_evaluations": case_force_evaluations,
        "total_force_evaluations": (
            shared_pool_force_evaluations
            + case_force_evaluations
        ),
        "wall_time_s": wall_time,
        "shared_pools": pool_records,
        "rows": rows,
        "analysis": analysis,
    }
    _write_json(output_dir / "evidence.json", evidence)
    (output_dir / "conclusion.md").write_text(
        _render_conclusion(evidence),
        encoding="utf-8",
    )
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--state-source-root",
        required=True,
        type=Path,
        help="repository root containing the checksum-pinned historical outputs",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=protocol.SYSTEMS,
        default=list(protocol.SYSTEMS),
    )
    parser.add_argument(
        "--state-ids",
        nargs="+",
        choices=protocol.STATE_IDS,
        default=list(protocol.STATE_IDS),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(protocol.SEEDS),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = run(
        output_dir=args.output,
        expected_commit=args.expected_commit,
        state_source_root=args.state_source_root,
        systems=args.systems,
        state_ids=args.state_ids,
        seeds=args.seeds,
    )
    print(
        json.dumps(
            {
                "case_count": evidence["case_count"],
                "total_force_evaluations": (
                    evidence["total_force_evaluations"]
                ),
                "wall_time_s": evidence["wall_time_s"],
                "analysis": evidence["analysis"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
