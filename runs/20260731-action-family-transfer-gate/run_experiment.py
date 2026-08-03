#!/usr/bin/env python3
"""Run the preregistered C60/PdO action-family transfer gate."""

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
ARM_CONFIGS: dict[str, dict[str, object]] = {
    "D0_exact_anchor": {
        "direction_selection_mode": "exact_anchor",
    },
    "D1_anchor_krylov_d2": {
        "direction_selection_mode": "anchor_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 2,
    },
    "K4_discrete": {
        "direction_selection_mode": "discrete",
        "oracle_candidates": 4,
    },
}
EXPECTED_SELECTION_EVALUATIONS = {
    "D0_exact_anchor": 2,
    "D1_anchor_krylov_d2": 4,
    "K4_discrete": 8,
}


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
    "_action_family_transfer_protocol_runner",
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
    values = np.asarray(direction, dtype=float).reshape(-1)
    values = values / np.linalg.norm(values)
    return sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"missing direction diagnostics: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _base_config(config_gate, system: str, seed: int, case_dir: Path):
    config = config_gate.build_production_config(
        system,
        case_dir,
        master_seed=seed,
    )
    return replace(
        config,
        max_trials=1,
        max_force_evals=None,
        max_steps_per_walk=8,
        direction_synthesis_mode="none",
        direction_probe_enabled=False,
        direction_type_ucb_enabled=False,
        direction_archive_enabled=False,
        direction_archive_path=None,
        plateau_evolution_enabled=False,
        archive_escape_momentum_enabled=False,
        accepted_structures_log=None,
        accepted_structures_dir=None,
        write_proposal_minima=False,
        proposal_minima_dir=None,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        proposal_pool_size=1,
        proposal_duplicate_rescue_optimizer=None,
    )


def _validate_direction_trace(
    arm: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("direction diagnostics contain no selections")
    expected_selection = EXPECTED_SELECTION_EVALUATIONS[arm]
    expected_kind = {
        "D0_exact_anchor": "anchor",
        "D1_anchor_krylov_d2": "block_ritz",
    }.get(arm)
    for index, row in enumerate(rows):
        if int(row["oracle_selection_force_evaluations_delta"]) != expected_selection:
            raise RuntimeError(
                f"{arm} direction row {index} has unexpected selection cost"
            )
        if expected_kind is not None and row["selected_kind"] != expected_kind:
            raise RuntimeError(
                f"{arm} direction row {index} has unexpected selected kind"
            )
        if arm == "D1_anchor_krylov_d2" and (
            int(row["krylov_depth"]) != 2
            or int(row["krylov_hvp_consumed"]) != 2
            or row["krylov_initial_basis_columns"] != [1]
        ):
            raise RuntimeError("D1 did not execute the depth-two anchor Krylov arm")
        if arm == "K4_discrete" and int(row["candidate_count"]) != 4:
            raise RuntimeError("K4 did not evaluate exactly four candidates")
    return {
        "selection_count": len(rows),
        "selection_force_evaluations": sum(
            int(row["oracle_selection_force_evaluations_delta"])
            for row in rows
        ),
        "direction_force_evaluations": sum(
            int(row["oracle_direction_force_evaluations_delta"])
            for row in rows
        ),
    }


def _run_case(
    *,
    state,
    provenance: Mapping[str, Any],
    calculator,
    config,
    system: str,
    state_id: str,
    seed: int,
    arm: str,
    case_dir: Path,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.io import write_state
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    case_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_path = case_dir / "direction.jsonl"
    case_config = replace(
        config,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(diagnostics_path),
        **ARM_CONFIGS[arm],
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
    proposal = walker._proposal_pool(
        state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=starter_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    escape_state = proposal.state
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    generation_wall_time = float(perf_counter() - started)

    quench_started = perf_counter()
    landing = walker.relax_true_minimum(
        escape_state,
        trajectory_name="action-family-landing",
    )
    quench_wall_time = float(perf_counter() - quench_started)
    before_count = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=starter_entry.entry_id,
    )

    direction_rows = _read_jsonl(diagnostics_path)
    direction_audit = _validate_direction_trace(arm, direction_rows)
    purpose_counts = walker.calculator.snapshot().as_dict()
    if (
        purpose_counts["unattributed"] != 0
        or sum(purpose_counts.values()) != walker.calculator.force_evaluations
        or purpose_counts["direction_oracle"]
        != direction_audit["direction_force_evaluations"]
    ):
        raise RuntimeError("purpose-resolved force ledger does not close")
    certificate = bool(
        has_force_convergence_certificate(landing, case_config.quench_fmax)
    )
    geometry_valid = bool(
        walker.geometry_validator.is_valid_state(landing.state)
    )
    fragmented = bool(
        walker._is_fragmented_cluster(state, landing.state)
        if system == "c60"
        else False
    )

    write_state(case_dir / "starter.xyz", state)
    write_state(case_dir / "escape.xyz", escape_state)
    write_state(case_dir / "landing.xyz", landing.state)
    row = {
        "system": system,
        "state_id": state_id,
        "state_sha256": provenance["state_sha256"],
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "anchor_sha256": walker.gate_anchor_sha256,
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
        "landing_gradient_norm_eV_per_A": float(landing.gradient_norm),
        "landing_termination_reason": landing.telemetry.termination_reason,
        "walk_termination_reason": walker._walk_termination_last_reason,
        "force_evaluations": walker.calculator.force_evaluations,
        "purpose_counts": purpose_counts,
        "generation_wall_time_s": generation_wall_time,
        "quench_wall_time_s": quench_wall_time,
        "wall_time_s": generation_wall_time + quench_wall_time,
        "direction_audit": direction_audit,
        "direction_trace": direction_rows,
        "effective_config": asdict(case_config),
    }
    _write_json(case_dir / "summary.json", row)
    return row


def _render_conclusion(evidence: Mapping[str, Any]) -> str:
    analysis = evidence["analysis"]
    lines = [
        "# Action-family transfer gate conclusion",
        "",
        "## Decision",
        "",
        f"- Posterior promotion allowed: **{analysis['promotion_allowed']}**.",
        f"- Completed cases: {len(evidence['cases'])}/36.",
        f"- Total force evaluations: {evidence['total_force_evaluations']}.",
        f"- Total wall time: {evidence['wall_time_s']:.1f} s.",
        "",
        "## Arm summaries",
        "",
        "| System | Arm | Mean landing delta (eV) | Median delta (eV) | Mean FE | Lower landings |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for system, arms in analysis["by_system"].items():
        for arm, summary in arms.items():
            lines.append(
                f"| {system} | {arm} | "
                f"{summary['mean_landing_delta_eV']:.6f} | "
                f"{summary['median_landing_delta_eV']:.6f} | "
                f"{summary['mean_force_evaluations']:.1f} | "
                f"{summary['lower_landing_count']}/{summary['count']} |"
            )
    lines.extend(
        [
            "",
            "## Leave-one-system-out transfer",
            "",
            "| Held out | Trained on | Selected arm | Improvement over uniform (eV) | Dominated | Pass |",
            "|---|---|---|---:|:---:|:---:|",
        ]
    )
    for system, summary in analysis["held_out"].items():
        lines.append(
            f"| {system} | {summary['training_system']} | "
            f"{summary['selected_arm']} | "
            f"{summary['improvement_over_uniform_eV']:.6f} | "
            f"{summary['pareto_dominated']} | {summary['passes']} |"
        )
    lines.extend(
        [
            "",
            "This gate tests whether a coarse repeated action family transfers.",
            "It does not tune a posterior, UCB coefficient, representation, or",
            "scalar energy/cost reward. Failure closes the online posterior",
            "stage for these three arms.",
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
    output_dir.mkdir(parents=True)

    from pamssw.calculators import ASECalculator

    audit = _load_module(FIXED_AUDIT_PATH, "_action_family_fixed_audit")
    config_gate = _load_module(CONFIG_GATE_PATH, "_action_family_config_gate")
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    state_source_root = Path(state_source_root).resolve()
    if not state_source_root.is_dir():
        raise FileNotFoundError(state_source_root)
    locked_gate_root = state_source_root / LOCKED_GATE_RELATIVE
    if not locked_gate_root.is_dir():
        raise FileNotFoundError(locked_gate_root)
    calculator = ASECalculator(base_runner._calculator())

    started = perf_counter()
    cases: list[dict[str, Any]] = []
    for system in systems:
        template = base_runner.load_state(system)
        for state_id in state_ids:
            source_case = (
                locked_gate_root
                / system
                / state_id
                / "seed-00000042"
                / "candidate-0"
            )
            source_summary_path = source_case / "summary.json"
            source_state_path = source_case / "starter.xyz"
            if not source_summary_path.is_file() or not source_state_path.is_file():
                raise FileNotFoundError(source_case)
            source_summary = json.loads(
                source_summary_path.read_text(encoding="utf-8")
            )
            from pamssw.io import read_state

            state = read_state(
                source_state_path,
                fixed_mask=template.fixed_mask,
            )
            state_hash = audit._state_sha256(state)
            if state_hash != source_summary["state_sha256"]:
                raise RuntimeError(
                    f"{system} {state_id} locked starter hash drifted"
                )
            for source_seed in protocol.SEEDS:
                peer_summary_path = (
                    locked_gate_root
                    / system
                    / state_id
                    / f"seed-{source_seed:08d}"
                    / "candidate-0"
                    / "summary.json"
                )
                if not peer_summary_path.is_file():
                    raise FileNotFoundError(peer_summary_path)
                peer_summary = json.loads(
                    peer_summary_path.read_text(encoding="utf-8")
                )
                if peer_summary["state_sha256"] != state_hash:
                    raise RuntimeError(
                        f"{system} {state_id} locked peer starter drifted"
                    )
            provenance = {
                "state_sha256": state_hash,
                "source_summary_path": str(source_summary_path),
                "source_state_path": str(source_state_path),
            }
            for seed in seeds:
                anchor_hashes = {}
                for arm in protocol.ARMS:
                    case_dir = (
                        output_dir
                        / "cases"
                        / system
                        / state_id
                        / f"seed-{seed:08d}"
                        / arm
                    )
                    print(
                        f"[action-family] {system} {state_id} "
                        f"seed={seed} arm={arm}",
                        flush=True,
                    )
                    config = _base_config(
                        config_gate,
                        system,
                        seed,
                        case_dir,
                    )
                    row = _run_case(
                        state=state,
                        provenance=provenance,
                        calculator=calculator,
                        config=config,
                        system=system,
                        state_id=state_id,
                        seed=seed,
                        arm=arm,
                        case_dir=case_dir,
                    )
                    cases.append(row)
                    anchor_hashes[arm] = row["anchor_sha256"]
                if len(set(anchor_hashes.values())) != 1:
                    raise RuntimeError(
                        f"anchor mismatch for {system} {state_id} seed {seed}"
                    )

    expected_cases = [
        row
        for row in protocol.case_matrix()
        if row["system"] in systems
        and row["state_id"] in state_ids
        and row["seed"] in seeds
    ]
    if len(cases) != len(expected_cases):
        raise RuntimeError("executed case matrix is incomplete")
    analysis = protocol.summarize_campaign(cases)
    evidence = {
        "schema_version": 1,
        "execution_commit": expected_commit,
        "state_source_root": str(state_source_root),
        "cases": cases,
        "analysis": analysis,
        "total_force_evaluations": sum(
            int(row["force_evaluations"]) for row in cases
        ),
        "wall_time_s": float(perf_counter() - started),
    }
    _write_json(output_dir / "evidence.json", evidence)
    (output_dir / "conclusion.md").write_text(
        _render_conclusion(evidence),
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--state-source-root", type=Path, required=True)
    parser.add_argument("--systems", nargs="+", default=list(protocol.SYSTEMS))
    parser.add_argument("--state-ids", nargs="+", default=list(protocol.STATE_IDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(protocol.SEEDS))
    args = parser.parse_args()
    evidence = run(
        output_dir=args.output_dir,
        expected_commit=args.expected_commit,
        state_source_root=args.state_source_root,
        systems=tuple(args.systems),
        state_ids=tuple(args.state_ids),
        seeds=tuple(args.seeds),
    )
    print(json.dumps(
        {
            "promotion_allowed": evidence["analysis"]["promotion_allowed"],
            "total_force_evaluations": evidence["total_force_evaluations"],
            "wall_time_s": evidence["wall_time_s"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
