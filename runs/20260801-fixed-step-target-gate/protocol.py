"""Pure protocol for the fixed-reference macro uphill-target gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")
TARGET_MODES = ("archive_scaled", "fixed_reference")


def case_matrix(
    *,
    systems: Sequence[str] = SYSTEMS,
    seeds: Sequence[int] = (46,),
    target_modes: Sequence[str] = TARGET_MODES,
) -> list[dict[str, Any]]:
    if not systems or any(system not in SYSTEMS for system in systems):
        raise ValueError("invalid systems")
    if not seeds or any(isinstance(seed, bool) or int(seed) < 0 for seed in seeds):
        raise ValueError("invalid seeds")
    if not target_modes or any(mode not in TARGET_MODES for mode in target_modes):
        raise ValueError("invalid target modes")
    return [
        {"system": system, "seed": int(seed), "target_mode": mode}
        for system in systems
        for seed in seeds
        for mode in target_modes
    ]


def gain_auc(
    *,
    initial_energy_eV: float,
    bootstrap_force_evaluations: int,
    accepted_rows: Sequence[Mapping[str, Any]],
    total_force_budget: int,
) -> float:
    if total_force_budget <= 0:
        raise ValueError("total_force_budget must be positive")
    if not 0 <= bootstrap_force_evaluations <= total_force_budget:
        raise ValueError("invalid bootstrap force evaluations")
    best = float(initial_energy_eV)
    previous_fe = 0
    area = 0.0
    for row in accepted_rows:
        force_evaluations = int(row["force_evaluations"])
        if not bootstrap_force_evaluations <= force_evaluations <= total_force_budget:
            raise ValueError("accepted-row force evaluation outside campaign")
        if force_evaluations < previous_fe:
            raise ValueError("accepted rows must be force ordered")
        area += (force_evaluations - previous_fe) * max(
            0.0,
            float(initial_energy_eV) - best,
        )
        best = min(best, float(row["best_energy"]))
        previous_fe = force_evaluations
    area += (total_force_budget - previous_fe) * max(
        0.0,
        float(initial_energy_eV) - best,
    )
    return area / total_force_budget


def ut1_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["system"]), str(row["target_mode"])): float(
            row["gain_auc_eV"]
        )
        for row in rows
        if int(row["seed"]) == 46
    }
    winning = []
    for system in SYSTEMS:
        scaled = indexed.get((system, "archive_scaled"))
        fixed = indexed.get((system, "fixed_reference"))
        if scaled is None or fixed is None:
            raise ValueError("U-T1 rows do not contain the required matrix")
        if fixed > scaled:
            winning.append(system)
    return {
        "decision": "ADMIT_U_T2" if len(winning) >= 2 else "DO_NOT_ADMIT_U_T2",
        "fixed_winning_system_count": len(winning),
        "fixed_winning_systems": winning,
    }


def cohort_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = {
        (str(row["system"]), int(row["seed"]), str(row["target_mode"]))
        for row in rows
    }
    expected = {
        (row["system"], int(row["seed"]), row["target_mode"])
        for row in case_matrix()
    }
    if observed != expected:
        return {
            "decision": "NOT_EVALUATED_PARTIAL_COHORT",
            "fixed_winning_system_count": 0,
            "fixed_winning_systems": [],
        }
    return ut1_decision(rows)
