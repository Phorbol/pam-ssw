from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
REPEATS = (0, 1)
ARMS = ("random_only", "bond_only")
MEANINGFUL_ENERGY_DROP_EV = 0.001


@dataclass(frozen=True)
class FamilySettings:
    enable_momentum_candidate: bool
    oracle_candidates: int
    max_steps_per_walk: int
    proposal_relax_steps: int
    n_bond_pairs: int
    stagnation_bond_pair_boost: int
    expected_kind: str


ARM_SETTINGS = {
    "random_only": FamilySettings(
        enable_momentum_candidate=False,
        oracle_candidates=4,
        max_steps_per_walk=8,
        proposal_relax_steps=80,
        n_bond_pairs=0,
        stagnation_bond_pair_boost=0,
        expected_kind="random",
    ),
    "bond_only": FamilySettings(
        enable_momentum_candidate=False,
        oracle_candidates=4,
        max_steps_per_walk=8,
        proposal_relax_steps=80,
        n_bond_pairs=4,
        stagnation_bond_pair_boost=0,
        expected_kind="bond",
    ),
}


@dataclass(frozen=True)
class CaseSpec:
    stage: str
    state_id: str
    seed: int
    arm: str
    repeat: int
    settings: FamilySettings

    @property
    def key(self) -> str:
        return (
            f"{self.state_id}-seed{self.seed}-{self.arm}-"
            f"repeat{self.repeat}"
        )


def case_matrix() -> list[CaseSpec]:
    cases = []
    for state_id in STATE_IDS:
        for seed in SEEDS:
            for repeat in REPEATS:
                arms = ARMS if repeat == 0 else tuple(reversed(ARMS))
                for arm in arms:
                    cases.append(
                        CaseSpec(
                            stage="fixed_direction_family",
                            state_id=state_id,
                            seed=seed,
                            arm=arm,
                            repeat=repeat,
                            settings=ARM_SETTINGS[arm],
                        )
                    )
    return cases


def is_meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("certificate") is True
        and row.get("is_new_basin") is True
        and float(row["landing_delta_eV"])
        <= -MEANINGFUL_ENERGY_DROP_EV
    )


def stable_labels(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["state_id"]), int(row["seed"]), str(row["arm"]))
        grouped.setdefault(key, []).append(row)

    expected_keys = {
        (state_id, seed, arm)
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    }
    if set(grouped) != expected_keys:
        raise ValueError("stable labels require the exact paired cohort")

    labels = []
    for state_id, seed, arm in sorted(grouped):
        pair = grouped[(state_id, seed, arm)]
        if sorted(int(row["repeat"]) for row in pair) != list(REPEATS):
            raise ValueError("stable labels require both exact repeats")
        labels.append(
            {
                "state_id": state_id,
                "seed": seed,
                "arm": arm,
                "meaningful": all(is_meaningful(row) for row in pair),
                "landing_delta_eV_mean": sum(
                    float(row["landing_delta_eV"]) for row in pair
                )
                / len(pair),
            }
        )
    return labels


def _posterior_predictive(
    labels: Sequence[Mapping[str, Any]],
    *,
    state_id: str,
    arm: str | None,
) -> float:
    matched = [
        bool(label["meaningful"])
        for label in labels
        if label["state_id"] == state_id
        and (arm is None or label["arm"] == arm)
    ]
    return (1.0 + sum(matched)) / (2.0 + len(matched))


def held_out_residual_audit(
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    folds = []
    for held_out_seed in SEEDS:
        train = [
            label
            for label in labels
            if int(label["seed"]) != held_out_seed
        ]
        test = [
            label
            for label in labels
            if int(label["seed"]) == held_out_seed
        ]
        baseline_loss = 0.0
        action_loss = 0.0
        for label in test:
            observed = float(bool(label["meaningful"]))
            baseline_probability = _posterior_predictive(
                train,
                state_id=str(label["state_id"]),
                arm=None,
            )
            action_probability = _posterior_predictive(
                train,
                state_id=str(label["state_id"]),
                arm=str(label["arm"]),
            )
            baseline_loss += (baseline_probability - observed) ** 2
            action_loss += (action_probability - observed) ** 2
        folds.append(
            {
                "held_out_seed": held_out_seed,
                "starter_only_brier": baseline_loss / len(test),
                "starter_plus_family_brier": action_loss / len(test),
                "improved": action_loss < baseline_loss,
            }
        )
    baseline = sum(fold["starter_only_brier"] for fold in folds) / len(folds)
    action = sum(
        fold["starter_plus_family_brier"] for fold in folds
    ) / len(folds)
    return {
        "method": (
            "leave-one-seed-out Beta(1,1) Bernoulli posterior predictive"
        ),
        "folds": folds,
        "starter_only_brier": baseline,
        "starter_plus_family_brier": action,
        "all_folds_improved": all(fold["improved"] for fold in folds),
        "held_out_residual_signal": bool(
            action < baseline and all(fold["improved"] for fold in folds)
        ),
    }


def _validate_exact_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (
            case.state_id,
            case.seed,
            case.arm,
            case.repeat,
        ): case
        for case in case_matrix()
    }
    actual = {
        (
            str(row.get("state_id")),
            int(row.get("seed", -1)),
            str(row.get("arm")),
            int(row.get("repeat", -1)),
        ): row
        for row in rows
    }
    if len(rows) != len(expected) or set(actual) != set(expected):
        raise ValueError("evidence requires the exact cohort")

    for key, case in expected.items():
        validate_case_row(case, actual[key])


def validate_case_row(
    case: CaseSpec,
    row: Mapping[str, Any],
) -> None:
    settings = asdict(case.settings)
    audit = row.get("direction_audit", {})
    purposes = row.get("purpose_counts", {})
    expected_kind = case.settings.expected_kind
    expected_candidate_count = int(audit.get("selection_count", -1)) * 4
    if (
        row.get("status") != "completed"
        or row.get("stage") != "fixed_direction_family"
        or row.get("settings") != settings
        or row.get("selection_probability") != 1.0
        or row.get("exact_starter_reference") is not True
        or row.get("certificate") is not True
        or row.get("direction_trace_valid") is not True
        or bool(row.get("meaningful")) != is_meaningful(row)
    ):
        raise ValueError("case outcome contract does not revalidate")
    if audit.get("candidate_kind_counts") != {
        expected_kind: expected_candidate_count
    } or audit.get("selected_kind_counts") != {
        expected_kind: int(audit["selection_count"])
    }:
        raise ValueError("each arm must contain a single expected family")
    if (
        int(audit.get("candidate_count", -1))
        != expected_candidate_count
        or int(audit.get("direction_oracle_force_evaluations", -1))
        != 2 * expected_candidate_count
        or int(purposes.get("direction_oracle", -1))
        != 2 * expected_candidate_count
        or int(purposes.get("unattributed", -1)) != 0
        or sum(int(value) for value in purposes.values())
        != int(row.get("force_evaluations", -1))
    ):
        raise ValueError("case force-evaluation ledger does not close")


def build_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _validate_exact_rows(rows)
    labels = stable_labels(rows)
    stable_successes = Counter(
        label["arm"] for label in labels if label["meaningful"]
    )
    audit = held_out_residual_audit(labels)
    enough_labels = all(stable_successes[arm] >= 5 for arm in ARMS)
    held_out_signal = bool(audit["held_out_residual_signal"])
    enter = bool(enough_labels and held_out_signal)
    if not enough_labels:
        reason = "insufficient_stable_meaningful_labels_per_family"
    elif not held_out_signal:
        reason = "no_held_out_family_residual_signal"
    else:
        reason = "fixed_family_posterior_stage_justified"

    purpose_totals: Counter[str] = Counter()
    for row in rows:
        purpose_totals.update(
            {
                key: int(value)
                for key, value in row["purpose_counts"].items()
            }
        )
    return {
        "schema_version": 1,
        "experiment": "fixed_direction_family_terminal_labels",
        "cohort": {
            "state_ids": list(STATE_IDS),
            "seeds": list(SEEDS),
            "repeats": list(REPEATS),
            "arms": list(ARMS),
            "completed_cases": len(rows),
        },
        "fixed_controls": {
            "oracle_candidates": 4,
            "max_steps_per_walk": 8,
            "proposal_relax_steps": 80,
            "enable_momentum_candidate": False,
            "stagnation_bond_pair_boost": 0,
        },
        "stable_labels": labels,
        "stable_meaningful_by_family": {
            arm: stable_successes[arm] for arm in ARMS
        },
        "held_out_audit": audit,
        "posterior_gate": {
            "minimum_stable_meaningful_per_family": 5,
            "enough_stable_labels": enough_labels,
            "held_out_residual_signal": held_out_signal,
            "enter_posterior_stage": enter,
            "reason": reason,
        },
        "totals": {
            "force_evaluations": sum(
                int(row["force_evaluations"]) for row in rows
            ),
            "purpose_counts": dict(sorted(purpose_totals.items())),
            "meaningful_outcomes": sum(
                bool(row["meaningful"]) for row in rows
            ),
            "fragmented_outcomes": sum(
                bool(row["fragmented"]) for row in rows
            ),
            "fallback_outcomes": sum(
                bool(row["fallback_used"]) for row in rows
            ),
            "generation_wall_time_s": sum(
                float(row["generation_wall_time_s"]) for row in rows
            ),
            "quench_wall_time_s": sum(
                float(row["quench_wall_time_s"]) for row in rows
            ),
        },
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "production_default_changed": False,
        "claim_ceiling": (
            "paired C60 fixed-starter, fixed-cost random-only versus "
            "bond-only terminal outcomes; not a production selector or "
            "cross-system direction-family ranking"
        ),
        "cases": list(rows),
    }
