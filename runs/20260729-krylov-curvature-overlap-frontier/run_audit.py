#!/usr/bin/env python3
"""Run the self-contained C60 Krylov curvature-overlap frontier audit."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence


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
FORBIDDEN_PURPOSES = {
    "starter_true_quench",
    "bootstrap_true_quench",
    "unattributed",
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


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _validate_spectrum(
    spectrum: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(spectrum, list) or not spectrum:
        raise ValueError("spectrum must be a non-empty list")
    previous_curvature = -math.inf
    executed_count = 0
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
        if (
            point.get("block_index") != 0
            or point.get("ritz_index") != index
        ):
            raise ValueError(
                "spectrum requires one contiguous Krylov block"
            )
        if type(point.get("executed")) is not bool:
            raise ValueError("spectrum executed flag must be boolean")
        executed_count += int(point["executed"])
    if executed_count != 1 or spectrum[0]["executed"] is not True:
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


def _meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row["certificate"]
        and row["is_new_basin"]
        and _finite(row["landing_delta_eV"], "landing delta")
        < -MEANINGFUL_ENERGY_DROP_EV
    )


def _validate_row(row: Mapping[str, Any]) -> None:
    arm = str(row.get("arm"))
    if arm not in ARMS:
        raise ValueError("unknown frontier arm")
    if row.get("status") != "completed":
        raise ValueError("case is not complete")
    if row.get("certificate") is not True:
        raise ValueError("terminal quench certificate is missing")
    purposes = row.get("purpose_counts")
    if not isinstance(purposes, Mapping):
        raise ValueError("purpose ledger is missing")
    if any(
        int(purposes.get(purpose, -1)) != 0
        for purpose in FORBIDDEN_PURPOSES
    ):
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
        raise ValueError("direction ledger does not close")
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
            "direction_selection_count": len(
                selection_summaries
            ),
            "ritz_point_count": sum(
                len(trace["krylov_ritz_spectrum"])
                for row in arm_rows
                for trace in row["direction_trace"]
            ),
            "total_force_evaluations": sum(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "direction_force_evaluations": sum(
                int(row["purpose_counts"]["direction_oracle"])
                for row in arm_rows
            ),
            "median_landing_delta_eV": _median(
                [
                    _finite(
                        row["landing_delta_eV"],
                        "landing delta",
                    )
                    for row in arm_rows
                ]
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
            "median_maximum_overlap_true_curvature_delta": _median(
                [
                    summary[
                        "maximum_overlap_true_curvature_delta"
                    ]
                    for summary in selection_summaries
                ]
            ),
            "maximum_overlap_is_executed_count": sum(
                summary["maximum_overlap_is_executed"]
                for summary in selection_summaries
            ),
            "state_results": {
                state_id: {
                    "meaningful_outcome_count": sum(
                        _meaningful(row)
                        for row in arm_rows
                        if row["state_id"] == state_id
                    ),
                    "direction_selection_count": sum(
                        len(row["direction_trace"])
                        for row in arm_rows
                        if row["state_id"] == state_id
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
        "prior_evidence_sha256": prior_evidence_sha256,
        "certificate_count": sum(
            bool(row["certificate"]) for row in rows
        ),
        "meaningful_outcome_count": sum(
            _meaningful(row) for row in rows
        ),
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "arm_results": arm_results,
        "totals": {
            "force_evaluations": sum(
                int(row["force_evaluations"]) for row in rows
            ),
            "purpose_counts": purpose_totals,
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
            "descriptive paired three-seed frontier audit; "
            "unexecuted Ritz points have no terminal label"
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
        row = prior_module._run_case(
            state=states[state_id],
            state_provenance=state_provenance[state_id],
            shared_calculator=shared_calculator,
            base_runner=base_runner,
            state_id=state_id,
            seed=seed,
            arm=arm,
            case_dir=case_dir,
        )
        _validate_row(row)
        rows.append(row)
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
