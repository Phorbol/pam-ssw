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


def summarize_repeats(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def case_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
        return (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
            int(row["candidate_index"]),
        )

    first_by_case = {case_key(row): row for row in first_rows}
    second_by_case = {case_key(row): row for row in second_rows}
    if first_by_case.keys() != second_by_case.keys():
        raise ValueError("repeat campaigns do not contain the same cases")
    direction_identity_stable = True
    static_rank_stable = True
    score_differences: list[float] = []
    landing_differences: list[float] = []
    for key in sorted(first_by_case):
        first = first_by_case[key]
        second = second_by_case[key]
        direction_identity_stable &= (
            first["direction_sha256"] == second["direction_sha256"]
        )
        static_rank_stable &= int(first["static_rank"]) == int(
            second["static_rank"]
        )
        score_differences.append(
            abs(float(first["static_score"]) - float(second["static_score"]))
        )
        landing_differences.append(
            abs(
                float(first["landing_delta_eV"])
                - float(second["landing_delta_eV"])
            )
        )

    first_analysis = summarize_campaign(first_rows)
    second_analysis = summarize_campaign(second_rows)

    def group_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
        return (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
        )

    first_groups = {
        group_key(row): row
        for row in first_analysis["group_summaries"]
    }
    second_groups = {
        group_key(row): row
        for row in second_analysis["group_summaries"]
    }
    if first_groups.keys() != second_groups.keys():
        raise ValueError("repeat group summaries do not align")
    stable_misses = [
        key
        for key in first_groups
        if not first_groups[key]["static_winner_is_best_valid"]
        and not second_groups[key]["static_winner_is_best_valid"]
    ]
    stable_best = [
        key
        for key in first_groups
        if first_groups[key]["best_valid_candidate_index"]
        == second_groups[key]["best_valid_candidate_index"]
    ]
    stable_misses_by_system = {
        system: sum(key[0] == system for key in stable_misses)
        for system in sorted({key[0] for key in first_groups})
    }
    groups_by_system = {
        system: sum(key[0] == system for key in first_groups)
        for system in stable_misses_by_system
    }
    mechanism_supported = all(
        stable_misses_by_system[system]
        / groups_by_system[system]
        > 0.5
        for system in groups_by_system
    )
    posterior_allowed = all(
        first_analysis["by_system"][system]["classification"]
        == "selection_bottleneck"
        and second_analysis["by_system"][system]["classification"]
        == "selection_bottleneck"
        for system in groups_by_system
    )
    return {
        "direction_identity_stable": bool(direction_identity_stable),
        "static_rank_stable": bool(static_rank_stable),
        "max_static_score_absolute_difference": max(
            score_differences,
            default=0.0,
        ),
        "median_landing_delta_repeat_difference_eV": float(
            median(landing_differences)
        ),
        "max_landing_delta_repeat_difference_eV": max(
            landing_differences,
            default=0.0,
        ),
        "group_count": len(first_groups),
        "stable_static_winner_miss_count": len(stable_misses),
        "stable_static_winner_miss_by_system": stable_misses_by_system,
        "best_candidate_identity_stable_count": len(stable_best),
        "static_selector_inadequacy_supported": bool(
            mechanism_supported
        ),
        "posterior_promotion_allowed": bool(posterior_allowed),
        "first_analysis": first_analysis,
        "second_analysis": second_analysis,
    }


def evaluate_repeat_rankers(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
        return (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
            int(row["candidate_index"]),
        )

    first = {key(row): row for row in first_rows}
    second = {key(row): row for row in second_rows}
    if first.keys() != second.keys():
        raise ValueError("ranker campaigns do not contain the same cases")
    combined: list[dict[str, Any]] = []
    for case_key in sorted(first):
        left = first[case_key]
        right = second[case_key]
        combined.append(
            {
                "system": case_key[0],
                "state_id": case_key[1],
                "seed": case_key[2],
                "candidate_index": case_key[3],
                "kind": str(left["kind"]),
                "static_score": 0.5
                * (
                    float(left["static_score"])
                    + float(right["static_score"])
                ),
                "curvature": 0.5
                * (
                    float(left["curvature"])
                    + float(right["curvature"])
                ),
                "true_curvature": 0.5
                * (
                    float(left["true_curvature"])
                    + float(right["true_curvature"])
                ),
                "score_sigma": 0.5
                * (
                    float(left["score_sigma"])
                    + float(right["score_sigma"])
                ),
                "landing_delta_eV": 0.5
                * (
                    float(left["landing_delta_eV"])
                    + float(right["landing_delta_eV"])
                ),
            }
        )
    grouped = group_cases(combined)

    def static_score(group):
        return max(group, key=lambda row: float(row["static_score"]))

    def inner_curvature(group):
        return min(group, key=lambda row: float(row["curvature"]))

    def true_curvature(group):
        return min(group, key=lambda row: float(row["true_curvature"]))

    def family_static(group, kind: str):
        return max(
            [row for row in group if row["kind"] == kind],
            key=lambda row: float(row["static_score"]),
        )

    selectors = {
        "static_score": static_score,
        "inner_curvature": inner_curvature,
        "true_curvature": true_curvature,
        "random_then_static": lambda group: family_static(group, "random"),
        "bond_then_static": lambda group: family_static(group, "bond"),
    }

    def metrics(
        selected_groups: Sequence[
            tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]
        ],
    ) -> dict[str, Any]:
        regrets = []
        for group, selected in selected_groups:
            best = min(
                group,
                key=lambda row: float(row["landing_delta_eV"]),
            )
            regrets.append(
                float(selected["landing_delta_eV"])
                - float(best["landing_delta_eV"])
            )
        return {
            "group_count": len(regrets),
            "top1_hits": sum(regret <= 1.0e-12 for regret in regrets),
            "mean_regret_eV": float(np.mean(regrets)),
            "median_regret_eV": float(median(regrets)),
            "max_regret_eV": float(max(regrets)),
        }

    systems = sorted({key[0] for key in grouped})
    results: dict[str, dict[str, Any]] = {
        "overall": {},
        **{system: {} for system in systems},
    }
    for name, selector in selectors.items():
        selections = [
            (group, selector(group))
            for group in grouped.values()
        ]
        results["overall"][name] = metrics(selections)
        for system in systems:
            results[system][name] = metrics(
                [
                    pair
                    for group_key, pair in zip(grouped, selections)
                    if group_key[0] == system
                ]
            )

    posterior_selections = []
    grouped_items = list(grouped.items())
    for held_key, held_group in grouped_items:
        wins = {
            "bond": [1, 2],
            "random": [1, 2],
        }
        for training_key, training_group in grouped_items:
            if training_key == held_key:
                continue
            best = min(
                training_group,
                key=lambda row: float(row["landing_delta_eV"]),
            )
            wins[str(best["kind"])][0] += 1
            wins[str(best["kind"])][1] += 1
        posterior_means = {
            kind: successes / total
            for kind, (successes, total) in wins.items()
        }
        selected_kind = max(
            posterior_means,
            key=posterior_means.get,
        )
        posterior_selections.append(
            (
                held_key,
                held_group,
                family_static(held_group, selected_kind),
                selected_kind,
            )
        )
    results["overall"]["loo_beta_family"] = metrics(
        [
            (group, selected)
            for _, group, selected, _ in posterior_selections
        ]
    )
    for system in systems:
        results[system]["loo_beta_family"] = metrics(
            [
                (group, selected)
                for held_key, group, selected, _ in posterior_selections
                if held_key[0] == system
            ]
        )
    posterior_kind_counts = {
        kind: sum(
            selected_kind == kind
            for _, _, _, selected_kind in posterior_selections
        )
        for kind in ("bond", "random")
    }

    def dominates_static(name: str) -> bool:
        strict = False
        for system in systems:
            baseline = results[system]["static_score"]
            candidate = results[system][name]
            if (
                candidate["top1_hits"] < baseline["top1_hits"]
                or candidate["mean_regret_eV"]
                > baseline["mean_regret_eV"] + 1.0e-12
                or candidate["median_regret_eV"]
                > baseline["median_regret_eV"] + 1.0e-12
            ):
                return False
            strict |= (
                candidate["top1_hits"] > baseline["top1_hits"]
                or candidate["mean_regret_eV"]
                < baseline["mean_regret_eV"] - 1.0e-12
                or candidate["median_regret_eV"]
                < baseline["median_regret_eV"] - 1.0e-12
            )
        return strict

    energy_costs = [
        0.5
        * float(row["score_sigma"]) ** 2
        * float(row["curvature"])
        for row in combined
    ]
    return {
        **results,
        "adaptive_score_energy_cost": {
            "minimum_eV": min(energy_costs),
            "maximum_eV": max(energy_costs),
            "range_eV": max(energy_costs) - min(energy_costs),
        },
        "loo_beta_selected_kind_counts": posterior_kind_counts,
        "prospective_true_curvature_ablation_allowed": dominates_static(
            "true_curvature"
        ),
        "family_posterior_promotion_allowed": bool(
            dominates_static("loo_beta_family")
            and sum(count > 0 for count in posterior_kind_counts.values())
            > 1
        ),
    }
