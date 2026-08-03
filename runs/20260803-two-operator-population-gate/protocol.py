from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")
STARTERS = ("bootstrap", "h8_best")
SEEDS = (55, 56, 57)
FAMILIES = ("direct", "ssw")
FIXED_ATOM_COUNTS = {"c60": 0, "pdo": 40, "cuo": 24}
MAX_FORCE_EVALUATIONS = 60_000
DIRECT_ACTION_CAP = 1_000
SSW_AND_SHARED_PREFIX_CAP = 2_300
PAIR_SUBMISSION_CAP = DIRECT_ACTION_CAP + SSW_AND_SHARED_PREFIX_CAP


@dataclass(frozen=True)
class CaseSpec:
    system: str
    starter_context: str
    seed: int


def case_matrix() -> tuple[CaseSpec, ...]:
    return tuple(
        CaseSpec(system, starter, seed)
        for system in SYSTEMS
        for starter in STARTERS
        for seed in SEEDS
    )


def validate_row(row: Mapping[str, Any]) -> None:
    if row["system"] not in SYSTEMS:
        raise ValueError("unknown system")
    if row["starter_context"] not in STARTERS:
        raise ValueError("unknown starter context")
    if int(row["seed"]) not in SEEDS:
        raise ValueError("unknown seed")
    if row["operator_family"] not in FAMILIES:
        raise ValueError("unknown operator family")

    counts = {str(key): int(value) for key, value in row["purpose_counts"].items()}
    if counts.get("unattributed", 0) != 0:
        raise ValueError("unattributed force evaluations are forbidden")
    exclusive = int(row["force_evaluations"])
    if sum(counts.values()) != exclusive:
        raise ValueError("action purpose ledger does not close")
    fully_loaded = int(row.get("fully_loaded_force_evaluations", exclusive))
    if fully_loaded < exclusive:
        raise ValueError("fully loaded action cost is smaller than exclusive cost")

    wall_time = float(row.get("wall_time_s", 0.0))
    fully_loaded_wall = float(row.get("fully_loaded_wall_time_s", wall_time))
    if wall_time < 0.0:
        raise ValueError("action wall time is negative")
    if fully_loaded_wall < wall_time:
        raise ValueError("fully loaded wall time is smaller than exclusive wall time")


def _context_rows(rows: Sequence[Mapping[str, Any]]):
    grouped = defaultdict(list)
    for row in rows:
        validate_row(row)
        key = (row["system"], row["starter_context"], row["operator_family"])
        grouped[key].append(row)
    return grouped


def _fully_loaded_cost(row: Mapping[str, Any]) -> int:
    return int(row.get("fully_loaded_force_evaluations", row["force_evaluations"]))


def decide_stage_b(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = len(case_matrix()) * len(FAMILIES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, observed {len(rows)}")
    keys = [
        (
            str(row["system"]),
            str(row["starter_context"]),
            int(row["seed"]),
            str(row["operator_family"]),
        )
        for row in rows
    ]
    if len(set(keys)) != expected:
        raise ValueError("Stage-B action keys are not unique")

    grouped = _context_rows(rows)
    direct_viability = 0
    ssw_support = 0
    for system in SYSTEMS:
        for starter in STARTERS:
            direct = grouped[(system, starter, "direct")]
            ssw = grouped[(system, starter, "ssw")]
            direct_new = sum(
                bool(row["certified"]) and not bool(row["same_starter_basin"])
                for row in direct
            )
            if direct_new >= 2 and median(map(_fully_loaded_cost, direct)) < median(
                map(_fully_loaded_cost, ssw)
            ):
                direct_viability += 1

            exclusive_support = 0
            for seed in SEEDS:
                direct_row = next(row for row in direct if int(row["seed"]) == seed)
                ssw_row = next(row for row in ssw if int(row["seed"]) == seed)
                direct_failed = (
                    not bool(direct_row["certified"])
                    or bool(direct_row["same_starter_basin"])
                    or bool(direct_row["budget_censored"])
                )
                ssw_escaped = bool(ssw_row["certified"]) and not bool(
                    ssw_row["same_starter_basin"]
                )
                exclusive_support += int(direct_failed and ssw_escaped)
            if exclusive_support >= 2:
                ssw_support += 1

    certificate_counts = {
        (system, family): sum(
            bool(row["certified"])
            for row in rows
            if row["system"] == system and row["operator_family"] == family
        )
        for system in SYSTEMS
        for family in FAMILIES
    }
    numerical = all(value >= 5 for value in certificate_counts.values())
    passed = direct_viability >= 2 and ssw_support >= 1 and numerical
    return {
        "decision": "ADMIT_STAGE_C_DESIGN" if passed else "CLOSE_TWO_OPERATOR_PORTFOLIO",
        "direct_viability_contexts": direct_viability,
        "ssw_exclusive_support_contexts": ssw_support,
        "certificate_counts": {
            f"{system}:{family}": value
            for (system, family), value in sorted(certificate_counts.items())
        },
        "numerical_acceptability": numerical,
    }
