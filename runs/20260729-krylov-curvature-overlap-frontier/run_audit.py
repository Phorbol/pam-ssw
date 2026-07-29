#!/usr/bin/env python3
"""Replay locked C60 proposals and expose their complete Krylov frontiers."""

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
PRIOR_RUN_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-anchor-consistent-direction-ablation"
    / "run_ablation.py"
)
PRIOR_EVIDENCE_PATH = (
    PRIOR_RUN_PATH.parent / "output" / "evidence.json"
)
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
    "detached_ritz": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
    "anchor_lanczos": {
        "direction_selection_mode": "anchor_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
}
EXPECTED_HVP_PER_SELECTION = {
    "detached_ritz": 12,
    "anchor_lanczos": 12,
}
REPLAY_PURPOSES = {
    "direction_oracle",
    "biased_proposal_relax",
    "escape_true_pes_check",
}
FORBIDDEN_PURPOSES = {
    "landing_true_quench",
    "post_relax_validation",
    "starter_true_quench",
    "bootstrap_true_quench",
    "unattributed",
}
TRACE_DIAGNOSTIC_ONLY_KEYS = {
    "krylov_ritz_spectrum",
    "oracle_selection_wall_seconds",
    "oracle_wall_seconds",
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


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _validate_spectrum(
    spectrum: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(spectrum, list) or not spectrum:
        raise ValueError("spectrum must be a non-empty list")
    executed_count = 0
    previous_curvature = -math.inf
    for index, point in enumerate(spectrum):
        if not isinstance(point, Mapping):
            raise ValueError("spectrum point must be a mapping")
        curvature = _finite(
            point.get("curvature"),
            "spectrum curvature",
        )
        if curvature < previous_curvature:
            raise ValueError("spectrum curvature must be ordered")
        previous_curvature = curvature
        for key in (
            "true_curvature",
            "residual_norm",
            "initial_span_overlap",
            "anchor_abs_overlap",
            "participation_ratio",
        ):
            _finite(point.get(key), f"spectrum {key}")
        overlap = float(point["anchor_abs_overlap"])
        if overlap < 0.0 or overlap > 1.0 + 1.0e-12:
            raise ValueError("spectrum anchor overlap is outside [0, 1]")
        if float(point["participation_ratio"]) <= 0.0:
            raise ValueError("spectrum participation ratio must be positive")
        if point.get("block_index") != 0:
            raise ValueError("spectrum replay requires one Krylov block")
        if point.get("ritz_index") != index:
            raise ValueError("spectrum Ritz indices must be contiguous")
        executed = point.get("executed")
        if type(executed) is not bool:
            raise ValueError("spectrum executed flag must be boolean")
        executed_count += int(executed)
    if executed_count != 1 or spectrum[0].get("executed") is not True:
        raise ValueError(
            "spectrum must identify the lowest Ritz point as executed"
        )


def summarize_frontier(
    spectrum: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_spectrum(spectrum)
    frontier_indices: list[int] = []
    best_overlap = -math.inf
    for index, point in enumerate(spectrum):
        overlap = float(point["anchor_abs_overlap"])
        if overlap > best_overlap:
            frontier_indices.append(index)
            best_overlap = overlap
    executed_index = next(
        index
        for index, point in enumerate(spectrum)
        if point["executed"]
    )
    maximum_overlap_index = max(
        range(len(spectrum)),
        key=lambda index: (
            float(spectrum[index]["anchor_abs_overlap"]),
            -index,
        ),
    )
    executed = spectrum[executed_index]
    maximum = spectrum[maximum_overlap_index]
    return {
        "frontier_indices": frontier_indices,
        "frontier_size": len(frontier_indices),
        "executed_index": executed_index,
        "maximum_overlap_index": maximum_overlap_index,
        "executed_anchor_abs_overlap": float(
            executed["anchor_abs_overlap"]
        ),
        "maximum_anchor_abs_overlap": float(
            maximum["anchor_abs_overlap"]
        ),
        "executed_curvature": float(executed["curvature"]),
        "maximum_overlap_curvature": float(
            maximum["curvature"]
        ),
        "maximum_overlap_curvature_delta": float(
            maximum["curvature"]
        )
        - float(executed["curvature"]),
        "executed_true_curvature": float(
            executed["true_curvature"]
        ),
        "maximum_overlap_true_curvature": float(
            maximum["true_curvature"]
        ),
        "maximum_overlap_true_curvature_delta": float(
            maximum["true_curvature"]
        )
        - float(executed["true_curvature"]),
        "maximum_overlap_is_executed": (
            maximum_overlap_index == executed_index
        ),
    }


def _meaningful_prior(row: Mapping[str, Any]) -> bool:
    outcome = row["prior_terminal_outcome"]
    return bool(outcome.get("meaningful_outcome"))


def _validate_row(row: Mapping[str, Any]) -> None:
    arm = str(row.get("arm"))
    if arm not in ARMS:
        raise ValueError("unknown replay arm")
    purposes = row.get("purpose_counts")
    if (
        row.get("status") != "completed"
        or row.get("escape_hash_matches") is not True
        or row.get("selected_trace_matches") is not True
    ):
        raise ValueError("replay proof is incomplete")
    if not isinstance(purposes, Mapping):
        raise ValueError("purpose ledger is missing")
    if any(int(purposes.get(key, -1)) != 0 for key in FORBIDDEN_PURPOSES):
        raise ValueError("forbidden purpose has nonzero cost")
    if (
        sum(int(value) for value in purposes.values())
        != row.get("force_evaluations")
    ):
        raise ValueError("purpose ledger does not close")
    selections = int(row.get("direction_selection_count", -1))
    expected_hvp = EXPECTED_HVP_PER_SELECTION[arm]
    if (
        selections <= 0
        or row.get("direction_hvp_count")
        != selections * expected_hvp
        or purposes.get("direction_oracle")
        != selections * expected_hvp * 2
    ):
        raise ValueError("direction replay ledger does not close")
    trace = row.get("direction_trace")
    if not isinstance(trace, list) or len(trace) != selections:
        raise ValueError("direction trace count does not close")
    for direction_row in trace:
        if (
            direction_row.get(
                "oracle_selection_force_evaluations_delta"
            )
            != 2 * expected_hvp
            or direction_row.get("krylov_hvp_consumed")
            != expected_hvp
        ):
            raise ValueError("direction trace budget does not close")
        _validate_spectrum(
            direction_row.get("krylov_ritz_spectrum")
        )
    prior = row.get("prior_terminal_outcome")
    if (
        not isinstance(prior, Mapping)
        or prior.get("certificate") is not True
        or int(prior.get("prior_force_evaluations", -1)) <= 0
    ):
        raise ValueError("prior terminal outcome is invalid")


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def build_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    prior_evidence_sha256: str,
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
        len(rows) != 12
        or len(observed) != len(rows)
        or observed != expected
    ):
        raise ValueError("evidence requires the exact 12-case cohort")
    for row in rows:
        _validate_row(row)

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        selection_summaries = [
            summarize_frontier(trace["krylov_ritz_spectrum"])
            for row in arm_rows
            for trace in row["direction_trace"]
        ]
        arm_results[arm] = {
            "completed_replays": len(arm_rows),
            "direction_selection_count": len(
                selection_summaries
            ),
            "ritz_point_count": sum(
                len(trace["krylov_ritz_spectrum"])
                for row in arm_rows
                for trace in row["direction_trace"]
            ),
            "meaningful_prior_outcome_count": sum(
                _meaningful_prior(row) for row in arm_rows
            ),
            "total_replay_force_evaluations": sum(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "median_frontier_size": _median(
                [
                    float(summary["frontier_size"])
                    for summary in selection_summaries
                ]
            ),
            "median_executed_anchor_abs_overlap": _median(
                [
                    summary["executed_anchor_abs_overlap"]
                    for summary in selection_summaries
                ]
            ),
            "median_maximum_anchor_abs_overlap": _median(
                [
                    summary["maximum_anchor_abs_overlap"]
                    for summary in selection_summaries
                ]
            ),
            "median_maximum_overlap_curvature_delta": _median(
                [
                    summary[
                        "maximum_overlap_curvature_delta"
                    ]
                    for summary in selection_summaries
                ]
            ),
            "median_maximum_overlap_true_curvature_delta": (
                _median(
                    [
                        summary[
                            "maximum_overlap_true_curvature_delta"
                        ]
                        for summary in selection_summaries
                    ]
                )
            ),
            "maximum_overlap_is_executed_count": sum(
                summary["maximum_overlap_is_executed"]
                for summary in selection_summaries
            ),
            "state_results": {
                state_id: {
                    "direction_selection_count": sum(
                        len(row["direction_trace"])
                        for row in arm_rows
                        if row["state_id"] == state_id
                    ),
                    "meaningful_prior_outcome_count": sum(
                        _meaningful_prior(row)
                        for row in arm_rows
                        if row["state_id"] == state_id
                    ),
                }
                for state_id in STATE_IDS
            },
        }

    purpose_names = tuple(rows[0]["purpose_counts"])
    purpose_totals = {
        purpose: sum(
            int(row["purpose_counts"][purpose]) for row in rows
        )
        for purpose in purpose_names
    }
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "completed_replays": len(rows),
        },
        "prior_evidence_sha256": prior_evidence_sha256,
        "escape_hash_match_count": sum(
            bool(row["escape_hash_matches"]) for row in rows
        ),
        "selected_trace_match_count": sum(
            bool(row["selected_trace_matches"]) for row in rows
        ),
        "referenced_prior_terminal_force_evaluations": sum(
            int(
                row["prior_terminal_outcome"][
                    "prior_force_evaluations"
                ]
            )
            for row in rows
        ),
        "arm_results": arm_results,
        "totals": {
            "force_evaluations": sum(
                int(row["force_evaluations"]) for row in rows
            ),
            "purpose_counts": purpose_totals,
            "proposal_wall_time_s": sum(
                _finite(
                    row["proposal_wall_time_s"],
                    "proposal wall time",
                )
                for row in rows
            ),
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "descriptive exact-proposal replay of already paid Krylov "
            "subspaces; unexecuted Ritz points have no terminal label"
        ),
        "cases": list(rows),
    }


def _load_prior_evidence() -> tuple[dict[str, Any], str]:
    if not PRIOR_EVIDENCE_PATH.is_file():
        raise RuntimeError(
            f"prior evidence does not exist: {PRIOR_EVIDENCE_PATH}"
        )
    payload = json.loads(PRIOR_EVIDENCE_PATH.read_text())
    if payload.get("cohort", {}).get("completed_cases") != 18:
        raise RuntimeError("prior evidence cohort is incomplete")
    return payload, _sha256(PRIOR_EVIDENCE_PATH)


def _prior_case_index(
    prior_evidence: Mapping[str, Any],
) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    index = {
        (
            str(row["state_id"]),
            int(row["seed"]),
            str(row["arm"]),
        ): row
        for row in prior_evidence["cases"]
        if row["arm"] in ARMS
    }
    if len(index) != 12:
        raise RuntimeError("prior refined-arm cohort is incomplete")
    return index


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"direction trace was not written: {path}")
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def selected_trace_contract(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key not in TRACE_DIAGNOSTIC_ONLY_KEYS
        }
        for row in rows
    ]


def _run_replay_case(
    *,
    state,
    state_provenance: Mapping[str, Any],
    shared_calculator,
    base_runner,
    prior_case: Mapping[str, Any],
    state_id: str,
    seed: int,
    arm: str,
    case_dir: Path,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.archive import MinimaArchive
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

    started = perf_counter()
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        starter_evaluation = walker.calculator.evaluate(state)
    seed_entry = archive.add(
        state,
        float(starter_evaluation.energy),
        parent_id=None,
    )
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
    proposal_wall_time = float(perf_counter() - started)

    direction_rows = _read_direction_rows(
        Path(config.direction_diagnostics_path)
    )
    selected_trace_matches = (
        selected_trace_contract(direction_rows)
        == selected_trace_contract(prior_case["direction_trace"])
    )
    purposes = walker.calculator.snapshot().as_dict()
    prior_purposes = prior_case["purpose_counts"]
    if any(
        purposes[purpose] != prior_purposes[purpose]
        for purpose in REPLAY_PURPOSES
    ):
        raise RuntimeError("replay purpose cost differs from prior case")
    if any(purposes[purpose] != 0 for purpose in FORBIDDEN_PURPOSES):
        raise RuntimeError("replay unexpectedly used a forbidden purpose")
    force_evaluations = int(sum(purposes.values()))

    case_dir.mkdir(parents=True, exist_ok=True)
    escape_path = case_dir / "escape.xyz"
    base_runner.write_state(escape_path, escape_state)
    replay_escape_sha256 = _sha256(escape_path)
    escape_hash_matches = (
        replay_escape_sha256 == prior_case["escape_sha256"]
    )
    prior_terminal_outcome = {
        "certificate": bool(prior_case["certificate"]),
        "landing_delta_eV": float(
            prior_case["landing_delta_eV"]
        ),
        "is_new_basin": bool(prior_case["is_new_basin"]),
        "meaningful_outcome": bool(
            prior_case["certificate"]
            and prior_case["is_new_basin"]
            and float(prior_case["landing_delta_eV"]) < -0.001
        ),
        "landing_sha256": prior_case["landing_sha256"],
        "prior_force_evaluations": int(
            prior_case["force_evaluations"]
        ),
        "prior_landing_true_quench_force_evaluations": int(
            prior_purposes["landing_true_quench"]
        ),
    }
    row = {
        "state_id": state_id,
        "state_sha256": state_provenance["state_sha256"],
        "seed": seed,
        "arm": arm,
        "status": "completed",
        "starter_energy_eV": float(starter_evaluation.energy),
        "escape_energy_eV": float(escape_evaluation.energy),
        "escape_hash_matches": escape_hash_matches,
        "selected_trace_matches": selected_trace_matches,
        "replay_escape_path": str(escape_path),
        "replay_escape_sha256": replay_escape_sha256,
        "prior_escape_sha256": prior_case["escape_sha256"],
        "force_evaluations": force_evaluations,
        "purpose_counts": purposes,
        "direction_selection_count": len(direction_rows),
        "direction_hvp_count": sum(
            int(row["krylov_hvp_consumed"])
            for row in direction_rows
        ),
        "proposal_wall_time_s": proposal_wall_time,
        "direction_trace": direction_rows,
        "prior_terminal_outcome": prior_terminal_outcome,
        "effective_config": asdict(config),
    }
    _validate_row(row)
    if not escape_hash_matches or not selected_trace_matches:
        raise RuntimeError("replay proof does not match prior case")
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

    prior_evidence, prior_evidence_hash = _load_prior_evidence()
    prior_index = _prior_case_index(prior_evidence)
    prior_module = _load_module(
        PRIOR_RUN_PATH,
        "_krylov_frontier_prior_ablation",
    )
    (
        states,
        shared_calculator,
        base_runner,
        state_provenance,
        shared_provenance,
    ) = prior_module._load_locked_runtime()

    rows: list[dict[str, Any]] = []
    for case in case_matrix():
        state_id = str(case["state_id"])
        seed = int(case["seed"])
        arm = str(case["arm"])
        print(
            f"[krylov-frontier] {state_id} seed={seed} arm={arm}",
            flush=True,
        )
        case_dir = (
            output_dir
            / "cases"
            / f"{state_id}-seed{seed}-{arm}"
        )
        rows.append(
            _run_replay_case(
                state=states[state_id],
                state_provenance=state_provenance[state_id],
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                prior_case=prior_index[(state_id, seed, arm)],
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
                "prior_evidence_path": str(PRIOR_EVIDENCE_PATH),
                "prior_evidence_sha256": prior_evidence_hash,
                "prior_execution_commit": prior_evidence[
                    "execution_commit"
                ],
                "shared_provenance": shared_provenance,
                "state_provenance": state_provenance,
                "cases": rows,
            },
        )

    evidence = build_evidence(
        rows,
        prior_evidence_sha256=prior_evidence_hash,
    )
    evidence.update(
        {
            "execution_commit": actual_commit,
            "prior_evidence_path": str(PRIOR_EVIDENCE_PATH),
            "prior_execution_commit": prior_evidence[
                "execution_commit"
            ],
            "shared_provenance": shared_provenance,
            "state_provenance": state_provenance,
        }
    )
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
