#!/usr/bin/env python3
"""Run one repeat of the preregistered H1/H2 direction-racing probes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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
COUNTERFACTUAL_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-candidate-counterfactual-gate"
    / "run_experiment.py"
)
MAX_TOTAL_FORCE_EVALUATIONS = 12_000


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
    "_direction_short_rollout_protocol_runner",
)
counterfactual = _load_module(
    COUNTERFACTUAL_RUNNER_PATH,
    "_direction_short_rollout_counterfactual_runtime",
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


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"direction diagnostics were not written: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _terminal_index(
    payload: Mapping[str, Any],
) -> dict[tuple[str, str, int, str], Mapping[str, Any]]:
    rows = {
        (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
            str(row["arm"]),
        ): row
        for row in payload["rows"]
    }
    if len(rows) != 24:
        raise RuntimeError("terminal label campaign must contain 24 cases")
    return rows


def _run_case(
    *,
    state,
    state_provenance: Mapping[str, Any],
    calculator,
    config,
    system: str,
    state_id: str,
    seed: int,
    arm: str,
    horizon: int,
    repeat_id: int,
    terminal_row: Mapping[str, Any],
    case_dir: Path,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
    from pamssw.io import write_state
    from pamssw.pbc import mic_displacement
    from pamssw.walker import SurfaceWalker

    case_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_path = case_dir / "direction.jsonl"
    case_config = replace(
        config,
        max_steps_per_walk=horizon,
        direction_ranking_mode=arm,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(diagnostics_path),
    )
    walker = SurfaceWalker(
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
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
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
    )
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    wall_time = float(perf_counter() - started)
    direction_rows = _read_direction_rows(diagnostics_path)
    if not direction_rows:
        raise RuntimeError("short rollout contains no direction selection")
    terminal_trace = terminal_row["direction_trace"]
    if not terminal_trace:
        raise RuntimeError("terminal label contains no direction selection")
    first_direction_matches_terminal = bool(
        direction_rows[0]["selected_direction_sha256"]
        == terminal_trace[0]["selected_direction_sha256"]
    )
    if not first_direction_matches_terminal:
        raise RuntimeError(
            "short rollout does not reproduce terminal first direction"
        )
    if any(
        row.get("direction_ranking_mode") != arm
        for row in direction_rows
    ):
        raise RuntimeError("short rollout trace has the wrong ranker")
    purpose_counts = walker.calculator.snapshot().as_dict()
    if purpose_counts["unattributed"] != 0:
        raise RuntimeError("short rollout contains unattributed evaluations")
    if sum(purpose_counts.values()) != walker.calculator.force_evaluations:
        raise RuntimeError("short rollout purpose ledger does not close")
    geometry_valid = bool(
        walker.geometry_validator.is_valid_state(escape_state)
    )
    fragmented = bool(
        walker._is_fragmented_cluster(state, escape_state)
        if system == "c60"
        else False
    )
    displacement = mic_displacement(
        escape_state.positions,
        state.positions,
        state.cell,
        state.pbc,
    )
    free = ~state.fixed_mask
    free_displacement = displacement[free]
    free_gradient = np.asarray(
        escape_evaluation.gradient,
        dtype=float,
    )[free]
    displacement_rms = float(
        np.sqrt(np.mean(free_displacement * free_displacement))
        if free_displacement.size
        else 0.0
    )
    max_force = float(
        np.max(np.linalg.norm(free_gradient, axis=1))
        if free_gradient.size
        else 0.0
    )
    write_state(case_dir / "starter.xyz", state)
    write_state(case_dir / "escape.xyz", escape_state)
    row = {
        "repeat_id": repeat_id,
        "system": system,
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "horizon": horizon,
        "state_sha256": state_provenance["state_sha256"],
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "escape_delta_eV": (
            float(escape_evaluation.energy) - starter_energy
        ),
        "escape_displacement_rms_A": displacement_rms,
        "escape_max_force_eV_per_A": max_force,
        "geometry_valid": geometry_valid,
        "fragmented": fragmented,
        "first_direction_matches_terminal": (
            first_direction_matches_terminal
        ),
        "first_direction_sha256": direction_rows[0][
            "selected_direction_sha256"
        ],
        "direction_step0_force_evaluations": int(
            direction_rows[0][
                "oracle_selection_force_evaluations_delta"
            ]
        ),
        "direction_selection_count": len(direction_rows),
        "selected_direction_kinds": [
            trace["selected_kind"] for trace in direction_rows
        ],
        "walk_termination_reason": (
            walker._walk_termination_last_reason
        ),
        "purpose_counts": purpose_counts,
        "force_evaluations": walker.calculator.force_evaluations,
        "wall_time_s": wall_time,
        "effective_config": asdict(case_config),
        "direction_trace": direction_rows,
    }
    _write_json(case_dir / "summary.json", row)
    return row


def run(
    *,
    output_dir: Path,
    expected_commit: str,
    state_source_root: Path,
    terminal_evidence_path: Path,
    repeat_id: int,
) -> dict[str, Any]:
    if _current_commit() != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    from pamssw.calculators import ASECalculator

    terminal_payload = json.loads(
        terminal_evidence_path.read_text(encoding="utf-8")
    )
    terminal_rows = _terminal_index(terminal_payload)
    audit = _load_module(
        counterfactual.FIXED_AUDIT_PATH,
        f"_short_rollout_fixed_state_audit_{repeat_id}",
    )
    config_gate = _load_module(
        counterfactual.CONFIG_GATE_PATH,
        f"_short_rollout_config_gate_{repeat_id}",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    state_source_root = Path(state_source_root).resolve()
    if not state_source_root.is_dir():
        raise FileNotFoundError(state_source_root)
    audit.REPO_ROOT = state_source_root
    calculator = ASECalculator(base_runner._calculator())
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    total_force_evaluations = 0

    for system in protocol.SYSTEMS:
        registry = audit.FIXED_STATE_REGISTRY[system]
        audit._validate_origin(system, registry)
        template = base_runner.load_state(system)
        entries = {
            entry["state_id"]: entry
            for entry in registry
            if entry["state_id"] in protocol.STATE_IDS
        }
        for state_id in protocol.STATE_IDS:
            state, provenance = audit._load_locked_state(
                entries[state_id],
                template,
            )
            for seed in protocol.SEEDS:
                group_dir = (
                    output_dir
                    / "groups"
                    / system
                    / state_id
                    / f"seed-{seed:08d}"
                )
                config = counterfactual._base_config(
                    config_gate=config_gate,
                    system=system,
                    seed=seed,
                    group_dir=group_dir,
                )
                for arm in protocol.ARMS:
                    terminal_row = terminal_rows[
                        (system, state_id, seed, arm)
                    ]
                    for horizon in protocol.HORIZONS:
                        print(
                            f"[short-rollout] repeat={repeat_id} "
                            f"{system} {state_id} seed={seed} "
                            f"arm={arm} H={horizon}",
                            flush=True,
                        )
                        row = _run_case(
                            state=state,
                            state_provenance=provenance,
                            calculator=calculator,
                            config=config,
                            system=system,
                            state_id=state_id,
                            seed=seed,
                            arm=arm,
                            horizon=horizon,
                            repeat_id=repeat_id,
                            terminal_row=terminal_row,
                            case_dir=(
                                group_dir / arm / f"horizon-{horizon}"
                            ),
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
                                "short-rollout campaign exceeded "
                                "its preregistered force budget"
                            )
                        _write_json(
                            output_dir / "partial.json",
                            {
                                "schema_version": 1,
                                "execution_commit": expected_commit,
                                "repeat_id": repeat_id,
                                "rows": rows,
                            },
                        )
    evidence = {
        "schema_version": 1,
        "execution_commit": expected_commit,
        "terminal_execution_commit": terminal_payload[
            "execution_commit"
        ],
        "repeat_id": repeat_id,
        "state_source_root": str(state_source_root),
        "terminal_evidence_path": str(
            terminal_evidence_path.resolve()
        ),
        "case_count": len(rows),
        "force_evaluations": total_force_evaluations,
        "wall_time_s": float(perf_counter() - started),
        "rows": rows,
    }
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--state-source-root", required=True, type=Path)
    parser.add_argument(
        "--terminal-evidence",
        required=True,
        type=Path,
    )
    parser.add_argument("--repeat-id", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = run(
        output_dir=args.output,
        expected_commit=args.expected_commit,
        state_source_root=args.state_source_root,
        terminal_evidence_path=args.terminal_evidence,
        repeat_id=args.repeat_id,
    )
    print(
        json.dumps(
            {
                "case_count": evidence["case_count"],
                "force_evaluations": evidence["force_evaluations"],
                "wall_time_s": evidence["wall_time_s"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
