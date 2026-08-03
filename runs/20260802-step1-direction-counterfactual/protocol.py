"""Pure protocol for the shared micro-step-1 direction counterfactual gate."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


SYSTEMS = ("c60", "pdo", "cuo")
SEEDS = (52, 53, 54)
REPEATS = (0, 1)


def group_matrix(
    systems: Sequence[str] = SYSTEMS,
    seeds: Sequence[int] = SEEDS,
) -> list[dict[str, object]]:
    normalized_systems = tuple(str(system) for system in systems)
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_systems or any(system not in SYSTEMS for system in normalized_systems):
        raise ValueError("invalid systems")
    if not normalized_seeds or any(seed < 0 for seed in normalized_seeds):
        raise ValueError("invalid seeds")
    return [
        {"system": system, "seed": seed}
        for system in normalized_systems
        for seed in normalized_seeds
    ]


def family_terminal_medians(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["kind"])].append(float(row["landing_delta_eV"]))
    return {
        kind: float(median(values))
        for kind, values in sorted(grouped.items())
    }


def summarize_repeats(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    pools: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        pools[(str(row["system"]), int(row["seed"]))].append(row)

    usable_counts = {system: 0 for system in SYSTEMS}
    stable_best_counts = {system: 0 for system in SYSTEMS}
    stable_miss_counts = {system: 0 for system in SYSTEMS}
    spearman_by_system = {
        system: {repeat: [] for repeat in REPEATS}
        for system in SYSTEMS
    }
    source_wins = {
        system: defaultdict(int)
        for system in SYSTEMS
    }
    unstable_best_pools: list[str] = []
    invalid_pools: list[str] = []
    pool_results: list[dict[str, Any]] = []

    for system, seed in [(item["system"], item["seed"]) for item in group_matrix()]:
        key = (str(system), int(seed))
        pool_id = f"{system}:{seed}"
        pool_rows = pools.get(key, [])
        indexed: dict[int, dict[int, Mapping[str, object]]] = defaultdict(dict)
        for row in pool_rows:
            indexed[int(row["candidate_index"])][int(row["repeat"])] = row
        complete = len(indexed) == 4 and all(
            set(repeats) == set(REPEATS)
            for repeats in indexed.values()
        )
        identity_stable = complete and all(
            _candidate_identity(repeats[0]) == _candidate_identity(repeats[1])
            for repeats in indexed.values()
        )
        quality_valid = identity_stable and all(
            _quality_valid(row)
            for repeats in indexed.values()
            for row in repeats.values()
        )
        kinds = [
            str(indexed[index][0]["kind"])
            for index in sorted(indexed)
        ] if complete else []
        composition_valid = (
            kinds.count("momentum") == 1
            and any(kind != "momentum" for kind in kinds)
        )
        usable = bool(quality_valid and composition_valid)
        if not usable:
            invalid_pools.append(pool_id)
            pool_results.append(
                {
                    "pool_id": pool_id,
                    "usable": False,
                    "best_candidate_stable": False,
                }
            )
            continue

        usable_counts[str(system)] += 1
        best_by_repeat: dict[int, int] = {}
        family_best_by_repeat: dict[int, str] = {}
        static_miss_by_repeat: dict[int, bool] = {}
        momentum_regret_by_repeat: dict[int, float] = {}
        for repeat in REPEATS:
            repeated_rows = [indexed[index][repeat] for index in sorted(indexed)]
            best_index = min(
                (int(row["candidate_index"]) for row in repeated_rows),
                key=lambda index: (
                    float(indexed[index][repeat]["landing_delta_eV"]),
                    index,
                ),
            )
            best_by_repeat[repeat] = best_index
            static_index = min(
                (int(row["candidate_index"]) for row in repeated_rows),
                key=lambda index: (
                    int(indexed[index][repeat]["static_rank"]),
                    index,
                ),
            )
            static_miss_by_repeat[repeat] = static_index != best_index
            family_medians = family_terminal_medians(repeated_rows)
            family_best_by_repeat[repeat] = min(
                family_medians,
                key=lambda kind: (family_medians[kind], kind),
            )
            momentum_row = next(row for row in repeated_rows if row["kind"] == "momentum")
            best_nonmomentum = min(
                float(row["landing_delta_eV"])
                for row in repeated_rows
                if row["kind"] != "momentum"
            )
            momentum_regret_by_repeat[repeat] = (
                float(momentum_row["landing_delta_eV"]) - best_nonmomentum
            )
            spearman_by_system[str(system)][repeat].append(
                _spearman(
                    [float(row["static_score"]) for row in repeated_rows],
                    [-float(row["landing_delta_eV"]) for row in repeated_rows],
                )
            )

        stable_best = best_by_repeat[0] == best_by_repeat[1]
        if stable_best:
            stable_best_counts[str(system)] += 1
        else:
            unstable_best_pools.append(pool_id)
        stable_static_miss = stable_best and all(static_miss_by_repeat.values())
        if stable_static_miss:
            stable_miss_counts[str(system)] += 1
        stable_family_best = family_best_by_repeat[0] == family_best_by_repeat[1]
        if stable_family_best:
            source_wins[str(system)][family_best_by_repeat[0]] += 1
        pool_results.append(
            {
                "pool_id": pool_id,
                "usable": True,
                "best_candidate_stable": stable_best,
                "best_candidate_by_repeat": best_by_repeat,
                "static_miss_by_repeat": static_miss_by_repeat,
                "family_best_by_repeat": family_best_by_repeat,
                "momentum_regret_eV_by_repeat": momentum_regret_by_repeat,
            }
        )

    median_spearman = {
        system: {
            str(repeat): (
                float(median(values)) if values else None
            )
            for repeat, values in repeated.items()
        }
        for system, repeated in spearman_by_system.items()
    }
    identifiable = all(
        usable_counts[system] >= 2 and stable_best_counts[system] >= 2
        for system in SYSTEMS
    )
    static_bottleneck = identifiable and all(
        stable_miss_counts[system] >= 2
        and all(
            median_spearman[system][str(repeat)] is not None
            and float(median_spearman[system][str(repeat)]) <= 0.0
            for repeat in REPEATS
        )
        for system in SYSTEMS
    )
    candidate_families = sorted(
        {
            family
            for wins in source_wins.values()
            for family in wins
        }
    )
    dominant = [
        family
        for family in candidate_families
        if all(source_wins[system][family] >= 2 for system in SYSTEMS)
    ]
    dominant_source = dominant[0] if len(dominant) == 1 else None
    source_gate = identifiable and dominant_source is not None
    posterior_gate = bool(static_bottleneck)
    if not identifiable:
        decision = "STOP_NON_IDENTIFIABLE_DIRECTION_LABELS"
    elif source_gate and posterior_gate:
        decision = "ALLOW_SOURCE_AND_CONTEXT_PROSPECTIVE_GATES"
    elif source_gate:
        decision = "ALLOW_SOURCE_ONLY_PROSPECTIVE_GATE"
    elif posterior_gate:
        decision = "ALLOW_CONTEXT_POSTERIOR_PROSPECTIVE_GATE"
    else:
        decision = "STOP_MIXED_DIRECTION_EVIDENCE"
    return {
        "decision": decision,
        "usable_pool_counts": usable_counts,
        "stable_best_pool_counts": stable_best_counts,
        "stable_static_miss_counts": stable_miss_counts,
        "unstable_best_pools": unstable_best_pools,
        "invalid_pools": invalid_pools,
        "median_static_score_terminal_spearman": median_spearman,
        "source_win_counts": {
            system: dict(sorted(wins.items()))
            for system, wins in source_wins.items()
        },
        "dominant_source": dominant_source,
        "source_only_gate_allowed": source_gate,
        "static_continuation_bottleneck": static_bottleneck,
        "posterior_gate_allowed": posterior_gate,
        "pool_results": pool_results,
    }


def _candidate_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        int(row["candidate_index"]),
        str(row["kind"]),
        int(row["static_rank"]),
        str(row["direction_sha256"]),
    )


def _quality_valid(row: Mapping[str, object]) -> bool:
    return bool(
        row["certificate"]
        and row["landing_geometry_valid"]
        and not row["fragmented"]
        and row["prefix_valid"]
        and np.isfinite(float(row["landing_delta_eV"]))
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.allclose(left_ranks, left_ranks[0]) or np.allclose(right_ranks, right_ranks[0]):
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


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
