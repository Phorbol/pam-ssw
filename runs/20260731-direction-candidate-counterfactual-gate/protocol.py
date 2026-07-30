"""Pure preregistered analysis for the shared K=4 candidate-pool gate."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
CANDIDATE_INDICES = (0, 1, 2, 3)


def case_matrix() -> list[dict[str, Any]]:
    return [
        {
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "candidate_index": candidate_index,
        }
        for system in SYSTEMS
        for state_id in STATE_IDS
        for seed in SEEDS
        for candidate_index in CANDIDATE_INDICES
    ]


def group_cases(
    cases: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = (
            str(case["system"]),
            str(case["state_id"]),
            int(case["seed"]),
        )
        grouped[key].append(dict(case))
    return {
        key: sorted(rows, key=lambda row: int(row["candidate_index"]))
        for key, rows in sorted(grouped.items())
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.ptp(left_ranks) == 0.0 or np.ptp(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(CANDIDATE_INDICES):
        raise ValueError("each counterfactual group must contain exactly four candidates")
    ordered = sorted(rows, key=lambda row: int(row["candidate_index"]))
    static_winners = [row for row in ordered if int(row["static_rank"]) == 1]
    if len(static_winners) != 1:
        raise ValueError("each group must have exactly one static winner")
    static_winner = static_winners[0]
    valid = [
        row
        for row in ordered
        if bool(row["certificate"]) and bool(row["landing_geometry_valid"])
    ]
    best = (
        min(valid, key=lambda row: float(row["landing_delta_eV"]))
        if valid
        else None
    )
    valid_scores = [float(row["static_score"]) for row in valid]
    terminal_quality = [-float(row["landing_delta_eV"]) for row in valid]
    winner_valid = static_winner in valid
    regret = (
        float(static_winner["landing_delta_eV"])
        - float(best["landing_delta_eV"])
        if best is not None and winner_valid
        else None
    )
    return {
        "system": str(ordered[0]["system"]),
        "state_id": str(ordered[0]["state_id"]),
        "seed": int(ordered[0]["seed"]),
        "static_winner_candidate_index": int(static_winner["candidate_index"]),
        "best_valid_candidate_index": (
            None if best is None else int(best["candidate_index"])
        ),
        "valid_candidate_count": len(valid),
        "static_winner_valid": winner_valid,
        "static_winner_is_best_valid": bool(
            best is not None
            and winner_valid
            and int(best["candidate_index"])
            == int(static_winner["candidate_index"])
        ),
        "static_winner_regret_eV": regret,
        "static_score_terminal_spearman": _spearman(
            valid_scores,
            terminal_quality,
        ),
    }


def classify_gate(
    group_summaries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not group_summaries:
        raise ValueError("at least one group summary is required")
    comparable = [
        row
        for row in group_summaries
        if row.get("static_winner_regret_eV") is not None
    ]
    correlations = [
        float(row["static_score_terminal_spearman"])
        for row in group_summaries
        if row.get("static_score_terminal_spearman") is not None
    ]
    miss_count = sum(
        not bool(row["static_winner_is_best_valid"])
        for row in comparable
    )
    miss_fraction = miss_count / len(comparable) if comparable else 1.0
    median_regret = (
        float(median(float(row["static_winner_regret_eV"]) for row in comparable))
        if comparable
        else None
    )
    median_correlation = (
        float(median(correlations)) if correlations else None
    )

    # These are structural decision rules rather than tuned acquisition
    # weights.  A selection bottleneck requires a majority of exact shared
    # pools to contain a better non-winner and no positive median ordering
    # signal.  A generation bottleneck requires the static winner to be best
    # in every comparable pool with positive ordering signal.
    if (
        comparable
        and miss_fraction > 0.5
        and median_correlation is not None
        and median_correlation <= 0.0
    ):
        classification = "selection_bottleneck"
    elif (
        comparable
        and miss_count == 0
        and median_correlation is not None
        and median_correlation > 0.0
    ):
        classification = "candidate_generation_bottleneck"
    else:
        classification = "ambiguous"
    return {
        "classification": classification,
        "posterior_stage_allowed": classification == "selection_bottleneck",
        "comparable_group_count": len(comparable),
        "static_winner_miss_count": miss_count,
        "static_winner_miss_fraction": miss_fraction,
        "median_static_winner_regret_eV": median_regret,
        "median_static_score_terminal_spearman": median_correlation,
    }


def summarize_campaign(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = group_cases(rows)
    summaries = [summarize_group(group) for group in grouped.values()]
    present_systems = tuple(
        system
        for system in SYSTEMS
        if any(row["system"] == system for row in summaries)
    )
    by_system = {
        system: classify_gate(
            [row for row in summaries if row["system"] == system]
        )
        for system in present_systems
    }
    overall = classify_gate(summaries)
    overall["cross_system_posterior_stage_allowed"] = bool(
        set(present_systems) == set(SYSTEMS)
        and overall["posterior_stage_allowed"]
        and all(
            result["classification"] == "selection_bottleneck"
            for result in by_system.values()
        )
    )
    return {
        "group_summaries": summaries,
        "by_system": by_system,
        "overall": overall,
    }
