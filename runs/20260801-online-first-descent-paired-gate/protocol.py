"""Pure protocol for the G-E1 online first-descent paired gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (45, 46, 47)
ARMS = ("D0_exact_anchor", "K4_discrete")
MAX_NEW_FORCE_EVALUATIONS = 25_000
MAX_KERNEL_WALL_TIME_S = 300.0

PREFIX_FIELDS = (
    "selected_direction_sha256",
    "executed_step_scale",
    "uphill_final_bias_weight",
    "true_energy_eV",
    "state_sha256",
)


def case_matrix() -> list[dict[str, Any]]:
    return [
        {
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "arm": arm,
        }
        for system in SYSTEMS
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    ]


@dataclass
class FirstDescentObserver:
    starter_energy_eV: float
    tolerance_eV: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    trigger_step: int | None = None

    def __post_init__(self) -> None:
        if self.tolerance_eV < 0.0:
            raise ValueError("tolerance_eV must be non-negative")

    def __call__(self, record: Mapping[str, Any]) -> str | None:
        row = dict(record)
        self.rows.append(row)
        if self.trigger_step is not None:
            return None
        boundary = float(self.starter_energy_eV) - float(self.tolerance_eV)
        if float(row["true_energy_eV"]) < boundary:
            self.trigger_step = int(row["step"])
            return "true_energy_descent"
        return None


def compare_prefix(
    reference: Sequence[Mapping[str, Any]],
    early: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mismatches = []
    prefix_length = len(early)
    if prefix_length > len(reference):
        mismatches.append(
            {
                "step": None,
                "field": "prefix_length",
                "reference": len(reference),
                "early": prefix_length,
            }
        )
    for index, early_row in enumerate(early[: len(reference)]):
        reference_row = reference[index]
        expected_step = index + 1
        for side, row in (("reference", reference_row), ("early", early_row)):
            if int(row["step"]) != expected_step:
                mismatches.append(
                    {
                        "step": expected_step,
                        "field": f"{side}_step",
                        "reference": expected_step,
                        "early": int(row["step"]),
                    }
                )
        for name in PREFIX_FIELDS:
            if reference_row[name] != early_row[name]:
                mismatches.append(
                    {
                        "step": expected_step,
                        "field": name,
                        "reference": reference_row[name],
                        "early": early_row[name],
                    }
                )
    return {
        "prefix_valid": not mismatches,
        "prefix_length": prefix_length,
        "mismatches": mismatches,
    }


def classify_tradeoff(
    early_landing_delta_eV: float | None,
    reference_landing_delta_eV: float | None,
    tolerance_eV: float,
) -> str:
    if early_landing_delta_eV is None or reference_landing_delta_eV is None:
        return "UNLEARNABLE"
    early = float(early_landing_delta_eV)
    reference = float(reference_landing_delta_eV)
    tolerance = float(tolerance_eV)
    if reference > early + tolerance:
        return "AVOIDED_OVERSHOOT"
    if reference < early - tolerance:
        return "FORGONE_DEEPER_TERMINAL"
    return "ENERGY_EQUIVALENT"


def _case_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
    )


def build_decision(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_keys = {_case_key(row) for row in case_matrix()}
    actual_keys = {_case_key(row) for row in pairs}
    cohort_valid = len(pairs) == 24 and actual_keys == expected_keys
    prefixes_valid = cohort_valid and all(
        bool(row["prefix_valid"]) for row in pairs
    )
    triggered = [row for row in pairs if bool(row["triggered"])]
    lower_certified = len(triggered) >= 2 and all(
        bool(row["early_landing_certified"])
        and row["early_landing_delta_eV"] is not None
        and float(row["early_landing_delta_eV"])
        < -float(row["dedup_energy_tol_eV"])
        for row in triggered
    )
    summed_complete_action_fe_saving = sum(
        int(row["reference_complete_action_fe"])
        - int(row["early_complete_action_fe"])
        for row in triggered
    )
    admitted = bool(
        prefixes_valid
        and len(triggered) >= 2
        and lower_certified
        and summed_complete_action_fe_saving > 0
    )
    return {
        "decision": (
            "ADMIT_EQUAL_BUDGET_G_E2"
            if admitted
            else "DO_NOT_ADMIT_G_E2"
        ),
        "cohort_valid": cohort_valid,
        "prefixes_valid": prefixes_valid,
        "triggered_pair_count": len(triggered),
        "all_triggered_lower_certified": lower_certified,
        "summed_complete_action_fe_saving": (
            summed_complete_action_fe_saving
        ),
    }
