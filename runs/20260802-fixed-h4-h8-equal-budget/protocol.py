"""Pure protocol for the fixed H4 versus H8 equal-budget gate."""

from __future__ import annotations

from statistics import median
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo", "cuo")
HORIZONS = (4, 8)
SEED = 49
_OUTPUT_KEYS = {
    "accepted_structures_log",
    "accepted_structures_dir",
    "direction_diagnostics_path",
    "proposal_minima_dir",
}


def case_matrix(
    *,
    systems: Sequence[str] = SYSTEMS,
    seeds: Sequence[int] = (SEED,),
    horizons: Sequence[int] = HORIZONS,
) -> list[dict[str, Any]]:
    systems = tuple(str(system) for system in systems)
    seeds = tuple(int(seed) for seed in seeds)
    horizons = tuple(int(horizon) for horizon in horizons)
    if not systems or any(system not in SYSTEMS for system in systems):
        raise ValueError("invalid systems")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("invalid seeds")
    if not horizons or any(horizon not in HORIZONS for horizon in horizons):
        raise ValueError("invalid horizons")
    return [
        {"system": system, "seed": seed, "horizon": horizon}
        for system in systems
        for seed in seeds
        for horizon in horizons
    ]


def gain_auc(
    *,
    initial_energy_eV: float,
    accepted_rows: Sequence[Mapping[str, Any]],
    total_force_budget: int,
) -> float:
    if total_force_budget <= 0:
        raise ValueError("total_force_budget must be positive")
    best = float(initial_energy_eV)
    previous_fe = 0
    area = 0.0
    for row in accepted_rows:
        force_evaluations = int(row["force_evaluations"])
        if force_evaluations < previous_fe or force_evaluations > total_force_budget:
            raise ValueError("accepted rows must be ordered within the campaign budget")
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
    return float(area / total_force_budget)


def scientific_config_differences(
    h4: Mapping[str, Any],
    h8: Mapping[str, Any],
) -> dict[str, list[Any]]:
    if set(h4) != set(h8):
        missing = sorted(set(h4).symmetric_difference(h8))
        raise ValueError(f"config keys differ: {missing}")
    differences = {
        key: [h4[key], h8[key]]
        for key in h4
        if key not in _OUTPUT_KEYS and h4[key] != h8[key]
    }
    forbidden = sorted(set(differences) - {"max_steps_per_walk"})
    if forbidden:
        raise ValueError(f"scientific config differs outside horizon: {forbidden}")
    if differences.get("max_steps_per_walk") != [4, 8]:
        raise ValueError("comparison must be H4 versus H8")
    return differences


def cohort_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = [
        {
            "system": str(row["system"]),
            "seed": int(row["seed"]),
            "horizon": int(row["horizon"]),
        }
        for row in rows
    ]
    if observed != case_matrix():
        return {
            "decision": "NOT_EVALUATED_PARTIAL_COHORT",
            "h4_winning_systems": [],
            "median_h4_minus_h8_gain_auc_eV": None,
            "h4_minus_h8_gain_auc_eV": {},
            "certificate_regression_systems": [],
        }
    indexed = {
        (str(row["system"]), int(row["horizon"])): row
        for row in rows
    }
    deltas = {
        system: float(indexed[(system, 4)]["gain_auc_eV"])
        - float(indexed[(system, 8)]["gain_auc_eV"])
        for system in SYSTEMS
    }
    winners = [system for system in SYSTEMS if deltas[system] > 0.0]
    regressions = [
        system
        for system in SYSTEMS
        if _certificate_failure_count(indexed[(system, 4)])
        > _certificate_failure_count(indexed[(system, 8)])
    ]
    median_delta = float(median(deltas.values()))
    admit = len(winners) >= 2 and median_delta > 0.0 and not regressions
    return {
        "decision": (
            "ADMIT_H4_REPEAT_GATE"
            if admit
            else "RETAIN_H8_STOP_SHORT_HORIZON_BRANCH"
        ),
        "h4_winning_systems": winners,
        "median_h4_minus_h8_gain_auc_eV": median_delta,
        "h4_minus_h8_gain_auc_eV": deltas,
        "certificate_regression_systems": regressions,
    }


def _certificate_failure_count(row: Mapping[str, Any]) -> int:
    explicit = row.get("strict_landing_failure_count")
    if explicit is not None:
        return int(explicit)
    attempted = int(row["action_analysis"]["landing_quench_drop_eV"]["count"])
    rate = float(row["strict_landing_certificate_rate"])
    successes = int(round(attempted * rate))
    return attempted - successes
