#!/usr/bin/env python3
"""Derive compact, zero-PES evidence from the raw step-1 gate output."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _step_one_delta(row: Mapping[str, Any]) -> float | None:
    steps = row["trace"]["steps"]
    if len(steps) < 2:
        return None
    step = steps[1]
    return float(step["true_energy_after_eV"] - step["true_energy_before_eV"])


def _projected_rules(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for system in SYSTEMS:
        values = defaultdict(list)
        for group in groups:
            if group["system"] != system or not group["rows"]:
                continue
            for repeat in (0, 1):
                rows = [row for row in group["rows"] if row["repeat"] == repeat]
                best = min(float(row["landing_delta_eV"]) for row in rows)
                static = next(row for row in rows if int(row["static_rank"]) == 1)

                def branch_cost(row):
                    shared_pool_in_reference = 8 if repeat == 0 and row is static else 0
                    return int(row["force_evaluations"]) - shared_pool_in_reference

                families = defaultdict(list)
                for row in rows:
                    families[str(row["kind"])].append(row)
                values["static_regret_eV"].append(float(static["landing_delta_eV"]) - best)
                values["uniform_regret_eV"].append(
                    sum(float(row["landing_delta_eV"]) - best for row in rows)
                    / len(rows)
                )
                values["family_rotation_regret_eV"].append(
                    sum(
                        sum(float(row["landing_delta_eV"]) - best for row in family)
                        / len(family)
                        for family in families.values()
                    )
                    / len(families)
                )
                values["static_projected_FE"].append(branch_cost(static) + 8.0)
                values["uniform_selected_only_HVP_projected_FE"].append(
                    sum(branch_cost(row) for row in rows) / len(rows) + 2.0
                )
                values["family_selected_only_HVP_projected_FE"].append(
                    sum(
                        sum(branch_cost(row) for row in family) / len(family)
                        for family in families.values()
                    )
                    / len(families)
                    + 2.0
                )
        result[system] = {
            name: {
                "median": float(median(samples)) if samples else None,
                "mean": float(sum(samples) / len(samples)) if samples else None,
                "n": len(samples),
            }
            for name, samples in sorted(values.items())
        }
    return result


def derive(raw: Mapping[str, Any], *, raw_sha256: str, raw_bytes: int) -> dict[str, Any]:
    ledger = Counter()
    ledger_repairs = []
    group_summaries = []
    terminal_drift = {system: [] for system in SYSTEMS}
    immediate_drift = {system: [] for system in SYSTEMS}
    compact_rows = []
    rank_one_kinds = []
    quadratic_spreads = []
    closest_prediction_matches = 0
    closest_prediction_pools = 0

    for group in raw["groups"]:
        for name in (
            "bootstrap_purpose_counts",
            "step_zero_pool_purpose_counts",
            "shared_prefix_purpose_counts",
        ):
            counts = group.get(name)
            if counts is None and name == "step_zero_pool_purpose_counts":
                missing = int(group["step_zero_pool_force_evaluations"])
                ledger["direction_oracle"] += missing
                ledger_repairs.append(
                    {
                        "pool": f"{group['system']}:{group['seed']}",
                        "missing_field": name,
                        "reconstructed_direction_oracle_FE": missing,
                        "basis": "frozen step-zero K4 uses four central HVPs",
                    }
                )
            elif counts is not None:
                ledger.update({str(key): int(value) for key, value in counts.items()})
        for row in group["rows"]:
            ledger.update({str(key): int(value) for key, value in row["purpose_counts"].items()})

        candidates = group.get("step_one_candidates", [])
        predicted = [
            0.5 * float(candidate["score_sigma"]) ** 2 * float(candidate["curvature"])
            for candidate in candidates
        ]
        if candidates:
            rank_one = next(candidate for candidate in candidates if int(candidate["static_rank"]) == 1)
            rank_one_kinds.append(str(rank_one["kind"]))
            quadratic_spreads.append(max(predicted) - min(predicted))

        rows_by_candidate = defaultdict(dict)
        for row in group["rows"]:
            rows_by_candidate[int(row["candidate_index"])][int(row["repeat"])] = row
            delta = _step_one_delta(row)
            compact_rows.append(
                {
                    "system": row["system"],
                    "seed": int(row["seed"]),
                    "repeat": int(row["repeat"]),
                    "candidate_index": int(row["candidate_index"]),
                    "kind": row["kind"],
                    "static_rank": int(row["static_rank"]),
                    "static_score": float(row["static_score"]),
                    "quadratic_prediction_eV": (
                        0.5
                        * float(group["step_one_candidates"][int(row["candidate_index"])]["score_sigma"]) ** 2
                        * float(group["step_one_candidates"][int(row["candidate_index"])]["curvature"])
                    ),
                    "step_one_true_delta_eV": delta,
                    "landing_delta_eV": float(row["landing_delta_eV"]),
                    "certificate": bool(row["certificate"]),
                    "walk_termination_reason": row["walk_termination_reason"],
                    "force_evaluations": int(row["force_evaluations"]),
                }
            )
        for repeats in rows_by_candidate.values():
            terminal_drift[str(group["system"])].append(
                abs(float(repeats[0]["landing_delta_eV"]) - float(repeats[1]["landing_delta_eV"]))
            )
            left = _step_one_delta(repeats[0])
            right = _step_one_delta(repeats[1])
            if left is not None and right is not None:
                immediate_drift[str(group["system"])].append(abs(left - right))

        if candidates and group["rows"]:
            prediction = predicted[0]
            closest = []
            for repeat in (0, 1):
                completed = [
                    row
                    for row in group["rows"]
                    if int(row["repeat"]) == repeat and _step_one_delta(row) is not None
                ]
                closest.append(
                    min(
                        completed,
                        key=lambda row: (
                            abs(float(_step_one_delta(row)) - prediction),
                            int(row["candidate_index"]),
                        ),
                    )["candidate_index"]
                )
            if closest[0] == closest[1]:
                closest_prediction_pools += 1
                static_index = next(
                    int(candidate["candidate_index"])
                    for candidate in candidates
                    if int(candidate["static_rank"]) == 1
                )
                closest_prediction_matches += int(int(closest[0]) == static_index)

        group_summaries.append(
            {
                "system": group["system"],
                "seed": int(group["seed"]),
                "right_censored_before_step1": bool(group["right_censored_before_step1"]),
                "terminal_arms": len(group["rows"]),
                "bootstrap_FE": int(group["bootstrap_force_evaluations"]),
                "step_zero_pool_FE": int(group["step_zero_pool_force_evaluations"]),
                "shared_prefix_FE": int(group["shared_prefix_force_evaluations"]),
                "terminal_FE": sum(int(row["force_evaluations"]) for row in group["rows"]),
                "strict_certificates": sum(bool(row["certificate"]) for row in group["rows"]),
                "rank_one_kind": None if not candidates else str(rank_one["kind"]),
                "quadratic_prediction_spread_eV": None if not predicted else max(predicted) - min(predicted),
            }
        )

    drift = {
        system: {
            "terminal_abs_delta_median_eV": (
                float(median(terminal_drift[system])) if terminal_drift[system] else None
            ),
            "terminal_abs_delta_max_eV": max(terminal_drift[system], default=None),
            "immediate_step1_abs_delta_median_eV": (
                float(median(immediate_drift[system])) if immediate_drift[system] else None
            ),
            "immediate_step1_abs_delta_max_eV": max(immediate_drift[system], default=None),
        }
        for system in SYSTEMS
    }
    return {
        "schema_version": 1,
        "raw_evidence": {
            "path": "output/evidence.json",
            "sha256": raw_sha256,
            "bytes": raw_bytes,
            "execution_commit": raw["execution_commit"],
        },
        "aggregate": raw["aggregate"],
        "purpose_counts_reconstructed": dict(sorted(ledger.items())),
        "purpose_ledger_repairs": ledger_repairs,
        "purpose_ledger_closes": sum(ledger.values()) == int(raw["aggregate"]["new_force_evaluations"]),
        "group_summaries": group_summaries,
        "protocol_summary": raw["summary"],
        "mechanism_readout": {
            "usable_pool_count": sum(bool(group["rows"]) for group in raw["groups"]),
            "right_censored_pool_count": sum(not bool(group["rows"]) for group in raw["groups"]),
            "strict_certificate_count": sum(bool(row["certificate"]) for row in raw["rows"]),
            "terminal_arm_count": len(raw["rows"]),
            "completed_step_one_count": sum(_step_one_delta(row) is not None for row in raw["rows"]),
            "rank_one_kinds": rank_one_kinds,
            "all_rank_one_are_momentum": bool(rank_one_kinds and set(rank_one_kinds) == {"momentum"}),
            "maximum_within_pool_quadratic_prediction_spread_eV": max(quadratic_spreads, default=None),
            "repeat_stable_closest_quadratic_prediction_pools": closest_prediction_pools,
            "static_winner_matches_closest_quadratic_prediction_pools": closest_prediction_matches,
            "repeat_drift": drift,
        },
        "continuation_HVP_value_of_information_projection": _projected_rules(raw["groups"]),
        "rows": compact_rows,
        "decision": "STOP_DIRECTION_SOURCE_AND_POSTERIOR_GATES",
        "production_default_changed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_bytes = args.raw.read_bytes()
    raw = json.loads(raw_bytes)
    compact = derive(
        raw,
        raw_sha256=sha256(raw_bytes).hexdigest(),
        raw_bytes=len(raw_bytes),
    )
    _write_json(args.output, compact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
