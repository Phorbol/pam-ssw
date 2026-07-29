from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence


STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
REPEATS = (0, 1)
MEANINGFUL_ENERGY_DROP_EV = 0.001

STAGE_ARMS = {
    "momentum": ("momentum_on", "momentum_off"),
    "candidate_count": ("k4", "k8", "k12"),
    "bias_steps": ("b5", "b8"),
    "relax_cap": ("l40", "l80"),
}


@dataclass(frozen=True)
class RetainedSettings:
    enable_momentum_candidate: bool = True
    oracle_candidates: int = 12
    max_steps_per_walk: int = 8
    proposal_relax_steps: int = 80


@dataclass(frozen=True)
class CaseSpec:
    stage: str
    state_id: str
    seed: int
    arm: str
    repeat: int
    settings: RetainedSettings

    @property
    def key(self) -> str:
        return (
            f"{self.state_id}-seed{self.seed}-{self.arm}-"
            f"repeat{self.repeat}"
        )


def arm_settings(
    stage: str,
    arm: str,
    retained: RetainedSettings,
) -> RetainedSettings:
    if stage == "momentum":
        return replace(
            retained,
            enable_momentum_candidate=arm == "momentum_on",
        )
    if stage == "candidate_count":
        return replace(retained, oracle_candidates=int(arm[1:]))
    if stage == "bias_steps":
        return replace(retained, max_steps_per_walk=int(arm[1:]))
    if stage == "relax_cap":
        return replace(retained, proposal_relax_steps=int(arm[1:]))
    raise ValueError(f"unknown stage: {stage}")


def case_matrix(
    stage: str,
    retained: RetainedSettings,
) -> list[CaseSpec]:
    arms = STAGE_ARMS.get(stage)
    if arms is None:
        raise ValueError(f"unknown stage: {stage}")
    cases = []
    for state_id in STATE_IDS:
        for seed in SEEDS:
            for repeat in REPEATS:
                ordered = arms if repeat == 0 else tuple(reversed(arms))
                for arm in ordered:
                    cases.append(
                        CaseSpec(
                            stage=stage,
                            state_id=state_id,
                            seed=seed,
                            arm=arm,
                            repeat=repeat,
                            settings=arm_settings(
                                stage,
                                arm,
                                retained,
                            ),
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


def repeat_stable_meaningful_set(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> set[tuple[str, int]]:
    return {
        (state_id, seed)
        for state_id in STATE_IDS
        for seed in SEEDS
        if all(
            is_meaningful(
                next(
                    row
                    for row in rows
                    if row["state_id"] == state_id
                    and int(row["seed"]) == seed
                    and row["arm"] == arm
                    and int(row["repeat"]) == repeat
                )
            )
            for repeat in REPEATS
        )
    }


def _repeat_totals(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
    field: str,
) -> dict[int, int]:
    return {
        repeat: sum(
            int(row[field])
            for row in rows
            if row["arm"] == arm and int(row["repeat"]) == repeat
        )
        for repeat in REPEATS
    }


def _repeat_meaningful_counts(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> dict[int, int]:
    return {
        repeat: sum(
            is_meaningful(row)
            for row in rows
            if row["arm"] == arm and int(row["repeat"]) == repeat
        )
        for repeat in REPEATS
    }


def _all_certified(
    rows: Sequence[Mapping[str, Any]],
    arm: str,
) -> bool:
    selected = [row for row in rows if row["arm"] == arm]
    return bool(selected) and all(
        row.get("certificate") is True for row in selected
    )


def _lower_cost_each_repeat(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
) -> bool:
    candidate_cost = _repeat_totals(
        rows,
        candidate,
        "force_evaluations",
    )
    baseline_cost = _repeat_totals(
        rows,
        baseline,
        "force_evaluations",
    )
    return all(
        candidate_cost[repeat] < baseline_cost[repeat]
        for repeat in REPEATS
    )


def _lower_proposal_cost_each_repeat(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
) -> bool:
    def totals(arm: str) -> dict[int, int]:
        return {
            repeat: sum(
                int(
                    row["purpose_counts"][
                        "biased_proposal_relax"
                    ]
                )
                for row in rows
                if row["arm"] == arm
                and int(row["repeat"]) == repeat
            )
            for repeat in REPEATS
        }

    candidate_cost = totals(candidate)
    baseline_cost = totals(baseline)
    return all(
        candidate_cost[repeat] < baseline_cost[repeat]
        for repeat in REPEATS
    )


def decide_stage(
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    retained: RetainedSettings,
) -> dict[str, Any]:
    stable = {
        arm: repeat_stable_meaningful_set(rows, arm)
        for arm in STAGE_ARMS[stage]
    }
    if stage == "momentum":
        on_counts = _repeat_meaningful_counts(rows, "momentum_on")
        off_counts = _repeat_meaningful_counts(rows, "momentum_off")
        on_dominates = (
            stable["momentum_on"] > stable["momentum_off"]
            and all(
                on_counts[repeat] >= off_counts[repeat]
                for repeat in REPEATS
            )
        )
        off_dominates = (
            stable["momentum_off"] > stable["momentum_on"]
            and all(
                off_counts[repeat] >= on_counts[repeat]
                for repeat in REPEATS
            )
        )
        keep = not off_dominates
        status = (
            "positive"
            if on_dominates
            else "removal_candidate"
            if off_dominates
            else "unproven_retained"
        )
        selected = replace(
            retained,
            enable_momentum_candidate=keep,
        )
    else:
        baseline = {
            "candidate_count": "k12",
            "bias_steps": "b8",
            "relax_cap": "l80",
        }[stage]
        candidates = {
            "candidate_count": ("k4", "k8"),
            "bias_steps": ("b5",),
            "relax_cap": ("l40",),
        }[stage]
        winner = baseline
        for candidate in candidates:
            if (
                stable[candidate] >= stable[baseline]
                and _all_certified(rows, candidate)
                and _lower_cost_each_repeat(
                    rows,
                    candidate,
                    baseline,
                )
                and (
                    stage != "relax_cap"
                    or _lower_proposal_cost_each_repeat(
                        rows,
                        candidate,
                        baseline,
                    )
                )
            ):
                winner = candidate
                break
        selected = arm_settings(stage, winner, retained)
        status = (
            "reduced" if winner != baseline else "baseline_retained"
        )
    return {
        "stage": stage,
        "status": status,
        "repeat_stable_sets": {
            arm: sorted([list(key) for key in values])
            for arm, values in stable.items()
        },
        "retained_settings": asdict(selected),
    }


def stage_l_entry(
    rows: Sequence[Mapping[str, Any]],
    retained_arm: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row["arm"] == retained_arm]
    total = sum(int(row["force_evaluations"]) for row in selected)
    proposal = sum(
        int(row["purpose_counts"]["biased_proposal_relax"])
        for row in selected
    )
    calls = sum(
        int(row["optimizer_diagnostics"]["proposal_relax_count"])
        for row in selected
    )
    cap_hits = sum(
        int(
            row["optimizer_diagnostics"][
                "proposal_relax_termination_maxiter"
            ]
        )
        for row in selected
    )
    if total <= 0 or calls <= 0:
        raise ValueError(
            "Stage-L entry requires positive total FE and relax calls"
        )
    force_fraction = proposal / total
    cap_fraction = cap_hits / calls
    return {
        "entered": force_fraction > 0.5 and cap_fraction >= 0.2,
        "proposal_relax_force_fraction": force_fraction,
        "proposal_relax_cap_hit_fraction": cap_fraction,
        "proposal_relax_calls": calls,
        "proposal_relax_cap_hits": cap_hits,
    }
