#!/usr/bin/env python3
"""Freeze ASE-LBFGS strict-certificate failures for fallback replay."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_ROOT = (
    REPO_ROOT / "runs" / "20260728-true-quench-raw-strict-replay" / "output"
)
SOURCE_ROWS_PATH = SOURCE_ROOT / "rows.json"
SOURCE_SUMMARY_PATH = SOURCE_ROOT / "summary.json"
CORPUS_PATH = RUN_ROOT / "corpus.json"
EXPECTED_ROWS_SHA256 = (
    "b9c12ca40f2fa183d8573a46b66ec630aeb25ab54ea8eacfc3e88640d163e1ac"
)
EXPECTED_SUMMARY_SHA256 = (
    "3e15c150ef20ad85e143346a55918aa7510ec3eeadae727168cd36a6e33c9b36"
)
EXPECTED_FAILURE_KEYS = (("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4))
SYSTEMS = ("c60", "pdo")
ARM_IDS = (
    "scipy-lbfgsb",
    "safe-lbfgs-total",
    "ase-fire",
    "ase-fire2",
    "ase-lbfgs",
)
PRIMARY_ARM_ID = "ase-lbfgs"
STRICT_FMAX = 0.01
MAXITER = 400
OBJECTIVE = "true_mace_pes_no_bias_no_softening"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field} must be a lowercase Git commit")
    return value


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_summary(summary: object) -> Mapping[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("summary root must be an object")
    if summary.get("row_count") != 160:
        raise ValueError("summary does not describe a complete 160-row matrix")
    protocol = summary.get("strict_protocol")
    if protocol != {
        "fmax_eV_per_A": STRICT_FMAX,
        "maxiter": MAXITER,
        "objective": OBJECTIVE,
    }:
        raise ValueError("summary strict protocol mismatch")
    if [arm.get("arm_id") for arm in summary.get("arms", [])] != list(ARM_IDS):
        raise ValueError("summary arm matrix mismatch")
    _commit(summary.get("execution_commit"), "source execution commit")
    model = summary.get("model")
    if not isinstance(model, dict):
        raise ValueError("summary model provenance is missing")
    _digest(model.get("sha256"), "source model hash")
    corpus = summary.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("summary raw corpus provenance is missing")
    _digest(corpus.get("sha256"), "source raw corpus hash")
    inputs = summary.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(SYSTEMS):
        raise ValueError("summary input provenance is incomplete")
    for system in SYSTEMS:
        if not isinstance(inputs[system], dict):
            raise ValueError(f"summary {system} input provenance is missing")
        _digest(inputs[system].get("sha256"), f"source {system} input hash")
    return summary


def _validate_rows(rows: object) -> list[Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 160:
        raise ValueError("rows do not form a complete 160-row matrix")
    expected = {
        (system, task_index, arm_id)
        for system in SYSTEMS
        for arm_id in ARM_IDS
        for task_index in range(16)
    }
    seen: set[tuple[str, int, str]] = set()
    checked: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row must be an object")
        key = (
            str(row.get("system")),
            int(row.get("task_index", -1)),
            str(row.get("arm_id")),
        )
        if key in seen:
            raise ValueError("rows contain a duplicate matrix key")
        seen.add(key)
        if (
            row.get("fmax_eV_per_A") != STRICT_FMAX
            or row.get("maxiter") != MAXITER
            or row.get("coordinate_trust_radius_A") is not None
            or row.get("objective") != OBJECTIVE
        ):
            raise ValueError("row strict protocol mismatch")
        force_evaluations = row.get("force_evaluations")
        purpose = row.get("purpose_count_delta")
        if (
            not isinstance(force_evaluations, int)
            or force_evaluations <= 0
            or not isinstance(purpose, dict)
            or purpose.get("landing_true_quench") != force_evaluations
            or purpose.get("unattributed") != 0
            or sum(int(value) for value in purpose.values()) != force_evaluations
        ):
            raise ValueError("row evaluation ledger does not close")
        final = row.get("final")
        if not isinstance(final, dict) or not isinstance(final.get("state"), dict):
            raise ValueError("row final state is missing")
        positions_hash = _digest(
            final.get("positions_sha256"), "row final positions hash"
        )
        if position_sha256(final["state"].get("positions")) != positions_hash:
            raise ValueError("row final positions hash mismatch")
        checked.append(row)
    if seen != expected:
        raise ValueError("rows do not form a complete 160-row matrix")
    return checked


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _primary_cohort(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_system: dict[str, dict[str, int | float]] = {}
    for system in SYSTEMS:
        system_rows = [
            row
            for row in rows
            if row["arm_id"] == PRIMARY_ARM_ID and row["system"] == system
        ]
        trigger_count = sum(
            float(row["final"]["max_active_force_eV_per_A"]) > STRICT_FMAX
            for row in system_rows
        )
        by_system[system] = {
            "task_count": len(system_rows),
            "certificate_success_count": len(system_rows) - trigger_count,
            "trigger_count": trigger_count,
            "total_force_evaluations": sum(
                int(row["force_evaluations"]) for row in system_rows
            ),
            "total_wall_time_s": sum(float(row["wall_time_s"]) for row in system_rows),
        }
    return {
        "overall": {
            field: sum(system_values[field] for system_values in by_system.values())
            for field in (
                "task_count",
                "certificate_success_count",
                "trigger_count",
                "total_force_evaluations",
                "total_wall_time_s",
            )
        },
        "by_system": by_system,
    }


def build_corpus(
    *,
    rows_path: Path = SOURCE_ROWS_PATH,
    summary_path: Path = SOURCE_SUMMARY_PATH,
    output_path: Path = CORPUS_PATH,
) -> dict[str, Any]:
    """Validate the complete replay and freeze exactly its four primary failures."""

    rows_path = Path(rows_path)
    summary_path = Path(summary_path)
    output_path = Path(output_path)
    rows_hash = _sha256(rows_path)
    summary_hash = _sha256(summary_path)
    if rows_hash != EXPECTED_ROWS_SHA256:
        raise ValueError("source rows SHA-256 mismatch")
    if summary_hash != EXPECTED_SUMMARY_SHA256:
        raise ValueError("source summary SHA-256 mismatch")
    summary = _validate_summary(_load_json(summary_path))
    rows = _validate_rows(_load_json(rows_path))
    primary_rows = [row for row in rows if row["arm_id"] == PRIMARY_ARM_ID]
    failures = [
        row
        for row in primary_rows
        if float(row["final"]["max_active_force_eV_per_A"]) > STRICT_FMAX
    ]
    failure_keys = tuple((str(row["system"]), int(row["task_index"])) for row in failures)
    if failure_keys != EXPECTED_FAILURE_KEYS:
        raise ValueError(
            f"ASE-LBFGS certificate-failure set mismatch: {failure_keys!r}"
        )

    entries = []
    for row in failures:
        final = row["final"]
        entries.append(
            {
                "system": row["system"],
                "task_index": row["task_index"],
                "trial_index": row["trial_index"],
                "proposal_index": row["proposal_index"],
                "source_trajectory_path": row["source_trajectory_path"],
                "source_trajectory_sha256": row["source_trajectory_sha256"],
                "source_frame_index": row["source_frame_index"],
                "primary": {
                    "arm_id": row["arm_id"],
                    "optimizer": row["optimizer"],
                    "force_evaluations": row["force_evaluations"],
                    "wall_time_s": row["wall_time_s"],
                    "initial_energy_eV": row["initial"]["energy_eV"],
                    "final_energy_eV": final["energy_eV"],
                    "initial_max_active_force_eV_per_A": row["initial"][
                        "max_active_force_eV_per_A"
                    ],
                    "final_max_active_force_eV_per_A": final[
                        "max_active_force_eV_per_A"
                    ],
                    "certificate_passed": False,
                    "termination_reason": row["termination_reason"],
                },
                "fallback_start": {
                    "energy_eV": final["energy_eV"],
                    "max_active_force_eV_per_A": final[
                        "max_active_force_eV_per_A"
                    ],
                    "positions_sha256": final["positions_sha256"],
                    "state": final["state"],
                },
            }
        )

    payload = {
        "schema_version": 1,
        "selection": {
            "primary_arm_id": PRIMARY_ARM_ID,
            "strict_fmax_eV_per_A": STRICT_FMAX,
            "expected_failure_keys": [list(key) for key in EXPECTED_FAILURE_KEYS],
            "primary_cohort": _primary_cohort(rows),
        },
        "source": {
            "rows": {"path": _display_path(rows_path), "sha256": rows_hash},
            "summary": {"path": _display_path(summary_path), "sha256": summary_hash},
            "execution_commit": summary["execution_commit"],
            "raw_landing_corpus_sha256": summary["corpus"]["sha256"],
            "model": summary["model"],
            "inputs": summary["inputs"],
            "calculator": summary["calculator"],
            "runtime_versions": summary["runtime_versions"],
            "cuda": summary["cuda"],
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=SOURCE_ROWS_PATH)
    parser.add_argument("--summary", type=Path, default=SOURCE_SUMMARY_PATH)
    parser.add_argument("--output", type=Path, default=CORPUS_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = build_corpus(
        rows_path=args.rows,
        summary_path=args.summary,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
