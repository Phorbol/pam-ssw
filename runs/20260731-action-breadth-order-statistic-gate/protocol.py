"""Pure zero-FE protocol for exact K4 action-breadth counterfactuals."""

from __future__ import annotations

from itertools import combinations
from math import inf
from statistics import fmean
from typing import Any, Mapping, Sequence


def _valid(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate["valid"])


def _best_valid_energy(candidates: Sequence[Mapping[str, Any]]) -> float:
    valid = [
        float(candidate["landing_delta_eV"])
        for candidate in candidates
        if _valid(candidate)
    ]
    if not valid:
        raise ValueError("a fully observed pool requires a valid candidate")
    return min(valid)


def uniform_subset_summary(
    candidates: Sequence[Mapping[str, Any]],
    *,
    breadth: int,
    shared_pool_force_evaluations: int,
) -> dict[str, Any]:
    """Average exactly over all unordered subsets of the requested breadth."""

    if not 1 <= int(breadth) <= len(candidates):
        raise ValueError("breadth must be within the candidate pool")
    best_pool_energy = _best_valid_energy(candidates)
    subsets = list(combinations(candidates, int(breadth)))
    costs = [
        int(shared_pool_force_evaluations)
        + sum(int(candidate["force_evaluations"]) for candidate in subset)
        for subset in subsets
    ]
    valid_best_energies = [
        min(
            float(candidate["landing_delta_eV"])
            for candidate in subset
            if _valid(candidate)
        )
        for subset in subsets
        if any(_valid(candidate) for candidate in subset)
    ]
    valid_probability = len(valid_best_energies) / len(subsets)
    return {
        "breadth": int(breadth),
        "subset_count": len(subsets),
        "valid_subset_probability": valid_probability,
        "expected_regret_eV": (
            None
            if not valid_best_energies
            else fmean(
                energy - best_pool_energy
                for energy in valid_best_energies
            )
        ),
        "expected_force_evaluations": fmean(costs),
    }


def static_b1_summary(
    candidates: Sequence[Mapping[str, Any]],
    *,
    shared_pool_force_evaluations: int,
) -> dict[str, Any]:
    winners = [
        candidate
        for candidate in candidates
        if int(candidate["static_rank"]) == 1
    ]
    if len(winners) != 1:
        raise ValueError("each pool requires exactly one static rank-one winner")
    winner = winners[0]
    valid = _valid(winner)
    return {
        "candidate_index": int(winner["candidate_index"]),
        "valid_subset_probability": float(valid),
        "regret_eV": (
            float(winner["landing_delta_eV"])
            - _best_valid_energy(candidates)
            if valid
            else None
        ),
        "force_evaluations": (
            int(shared_pool_force_evaluations)
            + int(winner["force_evaluations"])
        ),
    }


def evaluate_live_b2_gate(
    strata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not strata:
        raise ValueError("at least one campaign-system stratum is required")
    evaluated = []
    for stratum in strata:
        static_regret = float(stratum["static_median_regret_eV"])
        static_cost = float(stratum["static_median_force_evaluations"])
        b2_regret = float(stratum["b2_median_regret_eV"])
        b2_cost = float(stratum["b2_median_force_evaluations"])
        regret_reduction_fraction = (
            (static_regret - b2_regret) / static_regret
            if static_regret > 0.0
            else 0.0
        )
        cost_increase_fraction = (
            (b2_cost - static_cost) / static_cost
            if static_cost > 0.0
            else inf
        )
        if cost_increase_fraction > 0.0:
            elasticity = (
                regret_reduction_fraction / cost_increase_fraction
            )
        elif regret_reduction_fraction > 0.0:
            elasticity = inf
        else:
            elasticity = 0.0
        validity_not_lower = (
            float(stratum["b2_valid_probability"])
            >= float(stratum["static_valid_probability"])
        )
        passed = (
            static_regret > 0.0
            and b2_regret < static_regret
            and validity_not_lower
            and elasticity >= 1.0
        )
        evaluated.append(
            {
                **dict(stratum),
                "fractional_regret_reduction": regret_reduction_fraction,
                "fractional_force_evaluation_increase": (
                    cost_increase_fraction
                ),
                "benefit_cost_elasticity": elasticity,
                "validity_not_lower": validity_not_lower,
                "passed": passed,
            }
        )
    return {
        "strata": evaluated,
        "live_b2_gate_allowed": all(
            bool(stratum["passed"]) for stratum in evaluated
        ),
        "production_change_allowed": False,
    }
