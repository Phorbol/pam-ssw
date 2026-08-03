#!/usr/bin/env python3
"""Zero-FE audit of starter support in existing selector campaigns."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_ROOTS = {
    "c60": (
        REPO_ROOT
        / "runs"
        / "20260731-starter-selection-mechanism-gate"
        / "seed42-output"
        / "c60"
        / "seed-00000042"
    ),
    "pdo": (
        REPO_ROOT
        / "runs"
        / "20260731-starter-selection-mechanism-gate"
        / "seed42-output"
        / "pdo"
        / "seed-00000042"
    ),
    "cuo": (
        REPO_ROOT
        / "runs"
        / "20260731-starter-selection-mechanism-gate"
        / "seed42-cuo-shared-bootstrap-output"
        / "cuo"
        / "seed-00000042"
    ),
}
MODES = ("uniform_archive", "archive_ucb", "metropolis_chain")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def support_metrics(
    starter_ids: Sequence[int],
    *,
    final_archive_size: int,
) -> dict[str, float | int]:
    if final_archive_size <= 0:
        raise ValueError("final_archive_size must be positive")
    ids = [int(value) for value in starter_ids]
    if not ids:
        raise ValueError("starter_ids must not be empty")
    if min(ids) < 0 or max(ids) >= final_archive_size:
        raise ValueError("starter ID lies outside the final archive")
    counts = Counter(ids)
    total = len(ids)
    entropy = -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
    )
    effective = math.exp(entropy)
    seen: set[int] = set()
    repeated = 0
    for entry_id in ids:
        repeated += int(entry_id in seen)
        seen.add(entry_id)
    return {
        "action_count": total,
        "unique_starter_count": len(counts),
        "unique_starter_fraction_of_final_archive": (
            len(counts) / final_archive_size
        ),
        "shannon_entropy_nats": entropy,
        "effective_support": effective,
        "effective_support_fraction": effective / final_archive_size,
        "max_starter_frequency": max(counts.values()) / total,
        "repeat_action_fraction": repeated / total,
    }


def project_trace(
    trace: Sequence[Mapping[str, Any]],
    *,
    expected_actions: int,
    expected_archive: int,
) -> dict[str, Any]:
    rows = [dict(row) for row in trace]
    if len(rows) != expected_actions + 1:
        raise ValueError("trace action count does not close")
    if [int(row["trial"]) for row in rows] != list(range(len(rows))):
        raise ValueError("trace trials must be consecutive from bootstrap")
    if int(rows[0]["starter_entry_id"]) != 0:
        raise ValueError("bootstrap starter must be archive entry zero")
    starter_ids = [int(row["starter_entry_id"]) for row in rows[1:]]
    landing_ids = [int(row["landing_entry_id"]) for row in rows]
    inferred_archive = max(landing_ids) + 1
    if inferred_archive != expected_archive:
        raise ValueError("trace landing IDs do not close final archive")
    return {
        "starter_ids": starter_ids,
        "final_archive_size": inferred_archive,
        "new_basin_action_count": sum(
            bool(row.get("accepted_new_basin", False)) for row in rows[1:]
        ),
    }


def analyze_case(system: str, mode: str) -> dict[str, Any]:
    case_root = SOURCE_ROOTS[system] / mode
    summary_path = case_root / "summary.json"
    trace_path = case_root / "energy_trace.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    projected = project_trace(
        trace,
        expected_actions=int(summary["completed_trials"]),
        expected_archive=int(summary["archive_entries"]),
    )
    return {
        "system": system,
        "seed": int(summary["seed"]),
        "mode": mode,
        "source_summary_path": summary_path.relative_to(REPO_ROOT).as_posix(),
        "source_summary_sha256": _sha256(summary_path),
        "source_trace_path": trace_path.relative_to(REPO_ROOT).as_posix(),
        "source_trace_sha256": _sha256(trace_path),
        "final_archive_size": projected["final_archive_size"],
        "new_basin_action_count": projected["new_basin_action_count"],
        "best_energy_eV": float(summary["best_energy_eV"]),
        "energy_drop_eV": float(summary["energy_drop_eV"]),
        "force_evaluations": int(summary["force_evaluations"]),
        **support_metrics(
            projected["starter_ids"],
            final_archive_size=projected["final_archive_size"],
        ),
    }


def build_evidence() -> dict[str, Any]:
    cases = [
        analyze_case(system, mode)
        for system in SOURCE_ROOTS
        for mode in MODES
    ]
    by_mode: dict[str, dict[str, float]] = {}
    for mode in MODES:
        rows = [row for row in cases if row["mode"] == mode]
        by_mode[mode] = {
            "mean_effective_support_fraction": sum(
                float(row["effective_support_fraction"]) for row in rows
            )
            / len(rows),
            "mean_unique_starter_fraction_of_final_archive": sum(
                float(row["unique_starter_fraction_of_final_archive"])
                for row in rows
            )
            / len(rows),
            "mean_repeat_action_fraction": sum(
                float(row["repeat_action_fraction"]) for row in rows
            )
            / len(rows),
        }
    return {
        "schema_version": 1,
        "new_force_evaluations": 0,
        "case_count": len(cases),
        "cases": cases,
        "aggregate_by_mode": by_mode,
        "claim_ceiling": (
            "descriptive starter-support audit of nine existing seed-42 "
            "20k-FE traces; not a selector reward or asymptotic claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "evidence.json",
    )
    args = parser.parse_args()
    evidence = build_evidence()
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence["aggregate_by_mode"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
