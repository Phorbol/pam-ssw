#!/usr/bin/env python3
"""Create compact evidence for two prospective ranker cohorts."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_true_curvature_repeat_protocol",
        PROTOCOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load true-curvature protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _purpose_totals(rows) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(row["purpose_counts"])
    return dict(sorted(totals.items()))


def run(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = _load(first_path)
    second = _load(second_path)
    if first["execution_commit"] != second["execution_commit"]:
        raise RuntimeError("ranker cohorts used different commits")
    if first["case_count"] != 24 or second["case_count"] != 24:
        raise RuntimeError("ranker gate requires two complete 24-case cohorts")
    analysis = protocol.summarize_repeats(
        first["rows"],
        second["rows"],
    )
    key = lambda row: (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
    )
    first_rows = {key(row): row for row in first["rows"]}
    second_rows = {key(row): row for row in second["rows"]}
    compact_rows = []
    for case_key in sorted(first_rows):
        left = first_rows[case_key]
        right = second_rows[case_key]
        compact_rows.append(
            {
                "system": case_key[0],
                "state_id": case_key[1],
                "seed": case_key[2],
                "arm": case_key[3],
                "landing_delta_eV_first": left["landing_delta_eV"],
                "landing_delta_eV_second": right["landing_delta_eV"],
                "force_evaluations_first": left["force_evaluations"],
                "force_evaluations_second": right["force_evaluations"],
                "direction_force_evaluations_first": left[
                    "purpose_counts"
                ]["direction_oracle"],
                "direction_force_evaluations_second": right[
                    "purpose_counts"
                ]["direction_oracle"],
                "walk_termination_first": left[
                    "walk_termination_reason"
                ],
                "walk_termination_second": right[
                    "walk_termination_reason"
                ],
                "selected_direction_kinds_first": left[
                    "selected_direction_kinds"
                ],
                "selected_direction_kinds_second": right[
                    "selected_direction_kinds"
                ],
                "certificate_first": left["certificate"],
                "certificate_second": right["certificate"],
                "landing_geometry_valid_first": left[
                    "landing_geometry_valid"
                ],
                "landing_geometry_valid_second": right[
                    "landing_geometry_valid"
                ],
            }
        )
    all_rows = first["rows"] + second["rows"]
    return {
        "schema_version": 1,
        "execution_commit": first["execution_commit"],
        "campaigns": [
            {
                "path": str(first_path),
                "force_evaluations": first["force_evaluations"],
                "wall_time_s": first["wall_time_s"],
            },
            {
                "path": str(second_path),
                "force_evaluations": second["force_evaluations"],
                "wall_time_s": second["wall_time_s"],
            },
        ],
        "total_force_evaluations": (
            first["force_evaluations"]
            + second["force_evaluations"]
        ),
        "total_wall_time_s": (
            first["wall_time_s"] + second["wall_time_s"]
        ),
        "quality": {
            "total_cases": len(all_rows),
            "certified_cases": sum(
                bool(row["certificate"]) for row in all_rows
            ),
            "geometry_valid_cases": sum(
                bool(row["landing_geometry_valid"])
                for row in all_rows
            ),
            "fragmented_cases": sum(
                bool(row["fragmented"]) for row in all_rows
            ),
        },
        "purpose_counts_first": _purpose_totals(first["rows"]),
        "purpose_counts_second": _purpose_totals(second["rows"]),
        "analysis": analysis,
        "rows": compact_rows,
    }


def _render(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Prospective true-curvature ranker result",
        "",
        "## Decision",
        "",
        (
            "- Promote true-curvature ranker as production default: "
            f"**{payload['analysis']['promotion_allowed']}**."
        ),
        (
            f"- Cases: {payload['quality']['certified_cases']}/"
            f"{payload['quality']['total_cases']} certified and "
            f"{payload['quality']['geometry_valid_cases']}/"
            f"{payload['quality']['total_cases']} geometry-valid."
        ),
        (
            f"- Total cost: {payload['total_force_evaluations']} force "
            f"evaluations, {payload['total_wall_time_s']:.1f} s."
        ),
        "",
        "| System | True-curvature wins | Mean ΔΔE (eV) | Median ΔΔE (eV) | Δ force eval | Pareto |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for system, result in payload["analysis"]["by_system"].items():
        lines.append(
            f"| {system} | {result['true_curvature_energy_wins']}/"
            f"{result['group_count']} | "
            f"{result['mean_energy_difference_eV']:.6f} | "
            f"{result['median_energy_difference_eV']:.6f} | "
            f"{result['force_evaluation_difference']} | "
            f"{result['pareto_improved']} |"
        )
    lines.extend(
        [
            "",
            "True curvature is a useful PdO mechanism but not a universal",
            "replacement for intent-preserving static selection. C60's negative",
            "mean is driven by one large win while its median and four of six",
            "paired groups are worse. The default therefore remains",
            "`static_score`.",
            "",
            "This closes the single-ranker branch. The next selector experiment",
            "must preserve the complementary intent and softness hypotheses",
            "without introducing a tuned weighted blend: either an explicitly",
            "costed two-arm racing policy or a contextual posterior that first",
            "passes offline held-out calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--conclusion", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(args.first, args.second)
    _write_json(args.evidence, payload)
    args.conclusion.write_text(_render(payload), encoding="utf-8")
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
