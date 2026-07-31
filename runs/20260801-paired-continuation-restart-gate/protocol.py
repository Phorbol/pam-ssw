"""Pure protocol for the paired continuation/restart selector gate."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")
STARTER_MODES = (
    "uniform_archive",
    "archive_ucb",
    "metropolis_chain",
    "paired_best_uniform",
)
COMPARATORS = ("archive_ucb", "metropolis_chain")


def case_matrix(
    *,
    systems: Sequence[str] = SYSTEMS,
    seeds: Sequence[int] = (45,),
    starter_modes: Sequence[str] = STARTER_MODES,
) -> list[dict[str, Any]]:
    if not systems or any(system not in SYSTEMS for system in systems):
        raise ValueError("invalid systems")
    if not seeds or any(isinstance(seed, bool) or int(seed) < 0 for seed in seeds):
        raise ValueError("invalid seeds")
    if not starter_modes or any(mode not in STARTER_MODES for mode in starter_modes):
        raise ValueError("invalid starter modes")
    return [
        {"system": system, "seed": int(seed), "starter_mode": mode}
        for system in systems
        for seed in seeds
        for mode in starter_modes
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
            0.0, float(initial_energy_eV) - best
        )
        best = min(best, float(row["best_energy"]))
        previous_fe = force_evaluations
    area += (total_force_budget - previous_fe) * max(
        0.0, float(initial_energy_eV) - best
    )
    return area / total_force_budget


def scr1_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["system"]), str(row["starter_mode"])): float(
            row["gain_auc_eV"]
        )
        for row in rows
        if int(row["seed"]) == 45
    }
    winning = []
    for system in SYSTEMS:
        paired = indexed.get((system, "paired_best_uniform"))
        comparators = [indexed.get((system, mode)) for mode in COMPARATORS]
        if paired is None or any(value is None for value in comparators):
            raise ValueError("S-CR1 rows do not contain the required matrix")
        if all(paired > float(value) for value in comparators):
            winning.append(system)
    return {
        "decision": "ADMIT_S_CR2" if len(winning) >= 2 else "DO_NOT_ADMIT_S_CR2",
        "paired_winning_system_count": len(winning),
        "paired_winning_systems": winning,
    }
