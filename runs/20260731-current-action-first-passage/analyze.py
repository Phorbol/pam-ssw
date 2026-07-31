#!/usr/bin/env python3
"""Compact and render the current-action first-passage evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_RAW = RUN_ROOT / "output" / "evidence.json"


def _context_summaries(
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (
                str(row["system"]),
                str(row["state_id"]),
                str(row["arm"]),
            )
            for row in cases
        }
    )
    summaries = []
    for system, state_id, arm in keys:
        rows = [
            row
            for row in cases
            if row["system"] == system
            and row["state_id"] == state_id
            and row["arm"] == arm
        ]
        labels_by_horizon = {}
        for horizon in (1, 2, 4, 8):
            labels = Counter(
                checkpoint["label"]
                for row in rows
                for checkpoint in row["checkpoints"]
                if int(checkpoint["horizon"]) == horizon
            )
            labels_by_horizon[str(horizon)] = dict(sorted(labels.items()))
        summaries.append(
            {
                "system": system,
                "state_id": state_id,
                "arm": arm,
                "case_count": len(rows),
                "reached_h8_count": sum(
                    any(
                        int(checkpoint["horizon"]) == 8
                        for checkpoint in row["checkpoints"]
                    )
                    for row in rows
                ),
                "early_escape_then_h8_return_seed_count": sum(
                    row["trajectory_summary"][
                        "early_escape_then_h8_return"
                    ]
                    for row in rows
                ),
                "all_horizons_return_seed_count": sum(
                    row["trajectory_summary"][
                        "all_horizons_return_starter"
                    ]
                    for row in rows
                ),
                "generation_force_evaluations": sum(
                    int(row["generation_force_evaluations"])
                    for row in rows
                ),
                "checkpoint_force_evaluations": sum(
                    int(checkpoint["force_evaluations"])
                    for row in rows
                    for checkpoint in row["checkpoints"]
                ),
                "labels_by_horizon": labels_by_horizon,
            }
        )
    return summaries


def _purpose_totals(cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in cases:
        totals.update(
            {
                key: int(value)
                for key, value in row["generation_purpose_counts"].items()
            }
        )
        for checkpoint in row["checkpoints"]:
            totals.update(
                {
                    key: int(value)
                    for key, value in checkpoint["purpose_counts"].items()
                }
            )
    return dict(sorted(totals.items()))


def _render(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Current D0/K4 escape first-passage result",
        "",
        "## Decision",
        "",
        (
            "- Discrete horizon gate contexts: "
            f"**{len(evidence['horizon_gate_contexts'])}**."
        ),
        (
            "- Reproducible action-support gap contexts: "
            f"**{len(evidence['action_support_gap_contexts'])}**."
        ),
        (
            "- Numerical/matcher gate systems: "
            f"**{', '.join(evidence['numerical_matcher_gate_systems']) or 'none'}**."
        ),
        "- Production default changed: **False**.",
        (
            f"- Force evaluations: {evidence['total_force_evaluations']}"
            f"/{evidence['max_force_evaluations']}."
        ),
        f"- GPU kernel wall time: {evidence['wall_time_s']:.1f} s.",
        "",
        "## Context readout",
        "",
        "| System | Starter | Arm | H8 reached | Early escape→H8 return | All four return | Generation FE | Checkpoint FE |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in evidence["context_summaries"]:
        lines.append(
            f"| {row['system']} | {row['state_id']} | {row['arm']} | "
            f"{row['reached_h8_count']}/3 | "
            f"{row['early_escape_then_h8_return_seed_count']}/3 | "
            f"{row['all_horizons_return_seed_count']}/3 | "
            f"{row['generation_force_evaluations']} | "
            f"{row['checkpoint_force_evaluations']} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint labels",
            "",
            "| Label | Count |",
            "|---|---:|",
        ]
    )
    for label, count in evidence["label_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "The gate classifies whether the frozen current actions leave and",
            "return to their starter basins. It does not optimize a horizon,",
            "alter Gaussian bias parameters, or train a selector/posterior.",
            "",
        ]
    )
    return "\n".join(lines)


def run(raw_path: Path) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if raw.get("smoke_only"):
        raise ValueError("cannot publish a smoke-only first-passage result")
    cases = raw["cases"]
    compact = {
        key: raw[key]
        for key in (
            "schema_version",
            "execution_commit",
            "state_source_root",
            "cohort",
            "label_counts",
            "horizon_gate_contexts",
            "action_support_gap_contexts",
            "numerical_matcher_gate_systems",
            "unlearnable_trajectory_count_by_system",
            "total_force_evaluations",
            "max_force_evaluations",
            "unattributed_force_evaluations",
            "production_default_changed",
            "claim_ceiling",
            "wall_time_s",
        )
    }
    compact.update(
        {
            "raw_evidence_path": str(raw_path.relative_to(REPO_ROOT)),
            "raw_evidence_sha256": sha256(raw_path.read_bytes()).hexdigest(),
            "purpose_counts": _purpose_totals(cases),
            "context_summaries": _context_summaries(cases),
        }
    )
    (RUN_ROOT / "evidence.json").write_text(
        json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "conclusion.md").write_text(
        _render(compact),
        encoding="utf-8",
    )
    return compact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-evidence", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    result = run(args.raw_evidence.resolve())
    print(
        json.dumps(
            {
                "horizon_gate_contexts": len(
                    result["horizon_gate_contexts"]
                ),
                "action_support_gap_contexts": len(
                    result["action_support_gap_contexts"]
                ),
                "total_force_evaluations": result[
                    "total_force_evaluations"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
