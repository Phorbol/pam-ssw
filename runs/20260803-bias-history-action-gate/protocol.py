from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")
STARTERS = ("bootstrap", "h8_best")
SEEDS = (55, 56, 57)
FAMILIES = ("cumulative", "newest_only")
FIXED_ATOM_COUNTS = {"c60": 0, "pdo": 40, "cuo": 24}
MAX_FORCE_EVALUATIONS = 60_000
PAIR_SUBMISSION_CAP = 3_300


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
    if float(row.get("wall_time_s", 0.0)) < 0.0:
        raise ValueError("action wall time is negative")


def _escaped(row: Mapping[str, Any]) -> bool:
    return bool(
        row["certified"]
        and not row["same_starter_basin"]
        and not row["budget_censored"]
    )


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        raise ValueError("action keys are not unique")
    for row in rows:
        validate_row(row)

    indexed = {
        (
            str(row["system"]),
            str(row["starter_context"]),
            int(row["seed"]),
            str(row["operator_family"]),
        ): row
        for row in rows
    }
    context_counts: dict[str, dict[str, int]] = {}
    cumulative_only_contexts = 0
    newest_only_contexts = 0
    no_context_loss = True
    for system in SYSTEMS:
        for starter in STARTERS:
            counts = {family: 0 for family in FAMILIES}
            cumulative_only = 0
            newest_only = 0
            for seed in SEEDS:
                cumulative = indexed[(system, starter, seed, "cumulative")]
                newest = indexed[(system, starter, seed, "newest_only")]
                cumulative_escape = _escaped(cumulative)
                newest_escape = _escaped(newest)
                counts["cumulative"] += int(cumulative_escape)
                counts["newest_only"] += int(newest_escape)
                cumulative_only += int(cumulative_escape and not newest_escape)
                newest_only += int(newest_escape and not cumulative_escape)
            context_counts[f"{system}:{starter}"] = counts
            cumulative_only_contexts += int(cumulative_only >= 2)
            newest_only_contexts += int(newest_only >= 2)
            no_context_loss = no_context_loss and (
                counts["newest_only"] >= counts["cumulative"]
            )

    aggregate_escape_counts = {
        family: sum(
            _escaped(row) for row in rows if row["operator_family"] == family
        )
        for family in FAMILIES
    }
    fully_loaded_costs = {
        family: sum(
            int(row.get("fully_loaded_force_evaluations", row["force_evaluations"]))
            for row in rows
            if row["operator_family"] == family
        )
        for family in FAMILIES
    }
    certificate_counts = {
        f"{system}:{family}": sum(
            bool(row["certified"])
            for row in rows
            if row["system"] == system and row["operator_family"] == family
        )
        for system in SYSTEMS
        for family in FAMILIES
    }
    numerical = all(value >= 5 for value in certificate_counts.values())

    if cumulative_only_contexts:
        decision = "RETAIN_CUMULATIVE_REQUIRED"
    elif (
        numerical
        and no_context_loss
        and aggregate_escape_counts["newest_only"]
        >= aggregate_escape_counts["cumulative"]
        and fully_loaded_costs["newest_only"] < fully_loaded_costs["cumulative"]
    ):
        decision = "REPLACE_WITH_NEWEST_ONLY"
    elif numerical and newest_only_contexts >= 2:
        decision = "ADMIT_HISTORY_PORTFOLIO_REPEAT"
    else:
        decision = "RETAIN_CUMULATIVE_CLOSE_NEWEST_ONLY"

    return {
        "decision": decision,
        "cumulative_only_support_contexts": cumulative_only_contexts,
        "newest_only_support_contexts": newest_only_contexts,
        "context_escape_counts": context_counts,
        "aggregate_escape_counts": aggregate_escape_counts,
        "fully_loaded_force_evaluations": fully_loaded_costs,
        "certificate_counts": certificate_counts,
        "numerical_acceptability": numerical,
        "no_context_escape_loss": no_context_loss,
    }
