#!/usr/bin/env python3
"""Combine two short-rollout repeats with the frozen H8 terminal labels."""

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
        "_direction_short_rollout_repeat_protocol",
        PROTOCOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load short-rollout protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _terminal_rows(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat_id, campaign in enumerate((first, second), start=1):
        rows.extend(
            {**row, "repeat_id": repeat_id}
            for row in campaign["rows"]
        )
    return rows


def _purpose_totals(rows) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in rows:
        totals.update(row["purpose_counts"])
    return dict(sorted(totals.items()))


def run(
    first_probe_path: Path,
    second_probe_path: Path,
    first_terminal_path: Path,
    second_terminal_path: Path,
) -> dict[str, Any]:
    first_probe = _load(first_probe_path)
    second_probe = _load(second_probe_path)
    first_terminal = _load(first_terminal_path)
    second_terminal = _load(second_terminal_path)
    if first_probe["case_count"] != 48 or second_probe["case_count"] != 48:
        raise RuntimeError("short-rollout gate requires two complete 48-case repeats")
    if first_terminal["case_count"] != 24 or second_terminal["case_count"] != 24:
        raise RuntimeError("terminal labels require two complete 24-case repeats")
    probe_rows = first_probe["rows"] + second_probe["rows"]
    terminal_rows = _terminal_rows(first_terminal, second_terminal)
    analysis = protocol.summarize_campaign(probe_rows, terminal_rows)
    return {
        "schema_version": 1,
        "probe_execution_commit": first_probe["execution_commit"],
        "terminal_execution_commits": [
            first_terminal["execution_commit"],
            second_terminal["execution_commit"],
        ],
        "campaigns": [
            {
                "path": str(first_probe_path),
                "force_evaluations": first_probe["force_evaluations"],
                "wall_time_s": first_probe["wall_time_s"],
            },
            {
                "path": str(second_probe_path),
                "force_evaluations": second_probe["force_evaluations"],
                "wall_time_s": second_probe["wall_time_s"],
            },
        ],
        "probe_force_evaluations": (
            int(first_probe["force_evaluations"])
            + int(second_probe["force_evaluations"])
        ),
        "probe_wall_time_s": (
            float(first_probe["wall_time_s"])
            + float(second_probe["wall_time_s"])
        ),
        "purpose_counts": _purpose_totals(probe_rows),
        "quality": {
            "probe_cases": len(probe_rows),
            "geometry_valid_cases": sum(
                bool(row["geometry_valid"]) for row in probe_rows
            ),
            "fragmented_cases": sum(
                bool(row["fragmented"]) for row in probe_rows
            ),
            "first_direction_match_cases": sum(
                bool(row["first_direction_matches_terminal"])
                for row in probe_rows
            ),
        },
        "analysis": analysis,
    }


def _render(payload: Mapping[str, Any]) -> str:
    analysis = payload["analysis"]
    lines = [
        "# Two-arm short-uphill-rollout result",
        "",
        "## Decision",
        "",
        (
            "- Allow implementation of online state-reusing racing: "
            f"**{analysis['online_racing_stage_allowed']}**."
        ),
        (
            f"- Probe quality: {payload['quality']['geometry_valid_cases']}/"
            f"{payload['quality']['probe_cases']} geometry-valid, "
            f"{payload['quality']['first_direction_match_cases']}/"
            f"{payload['quality']['probe_cases']} exact first-direction "
            "matches."
        ),
        (
            f"- Probe cost: {payload['probe_force_evaluations']} force "
            f"evaluations, {payload['probe_wall_time_s']:.1f} s serial wall."
        ),
        "",
        "Primary rule: choose the larger true-PES energy rise after H2.",
        "",
        "| System | Larger-rise accuracy | Lower-rise diagnostic | Stable | Median regret (eV) | Mean vs static (eV) | Median vs static (eV) | Mean FE overhead | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for system, result in analysis["by_system"].items():
        lines.append(
            f"| {system} | {result['prediction_accuracy']:.3f} | "
            f"{result['lower_rise_prediction_accuracy']:.3f} | "
            f"{result['prediction_stability']:.3f} | "
            f"{result['median_regret_eV']:.6f} | "
            f"{result['mean_difference_vs_static_eV']:.6f} | "
            f"{result['median_difference_vs_static_eV']:.6f} | "
            f"{result['mean_estimated_online_overhead_vs_static']:.1f} | "
            f"{result['gate_passed']} |"
        )
    lines.extend(
        [
            "",
            "H1 is diagnostic only. No alternative metric or horizon may be",
            "substituted after seeing this result. A passing result permits",
            "only the next fixed-total-FE end-to-end test; it does not change",
            "the production selector by itself.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-probe", required=True, type=Path)
    parser.add_argument("--second-probe", required=True, type=Path)
    parser.add_argument("--first-terminal", required=True, type=Path)
    parser.add_argument("--second-terminal", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--conclusion", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(
        args.first_probe,
        args.second_probe,
        args.first_terminal,
        args.second_terminal,
    )
    _write_json(args.evidence, payload)
    args.conclusion.write_text(_render(payload), encoding="utf-8")
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
