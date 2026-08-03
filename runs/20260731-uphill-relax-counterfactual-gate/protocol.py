"""Pure protocol for the biased-relaxation counterfactual gate."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


SYSTEMS = ("c60",)
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS = ("D0_exact_anchor", "K4_discrete")
HORIZONS = (1, 2, 4)
MAX_NEW_FORCE_EVALUATIONS = 5_000

LEARNABLE_BASIN_LABELS = {
    "RETURN_STARTER",
    "ESCAPED_CERTIFIED",
}
LANDING_RELATIONS = {
    "NOT_APPLICABLE",
    "NOT_COMPARABLE",
    "SAME_LANDING",
    "DIFFERENT_LANDING",
    "AMBIGUOUS_LANDING",
}
CAUSAL_OUTCOMES = {
    "BOTH_RETURN_STARTER",
    "SAME_ESCAPED_LANDING",
    "DIFFERENT_ESCAPED_LANDINGS",
    "RELAXED_ONLY_ESCAPE",
    "EXPLICIT_ONLY_ESCAPE",
    "UNLEARNABLE",
}


def pair_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
        int(row["horizon"]),
    )


def pair_specs(source_cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Extract every recorded C60 h1/h2/h4 pre-relax frame exactly once."""

    specs: list[dict[str, Any]] = []
    for case in source_cases:
        if str(case.get("system")) not in SYSTEMS:
            continue
        state_id = str(case["state_id"])
        seed = int(case["seed"])
        arm = str(case["arm"])
        if state_id not in STATE_IDS or seed not in SEEDS or arm not in ARMS:
            raise ValueError("source case falls outside the frozen C60 cohort")
        for checkpoint in case.get("checkpoints", []):
            horizon = int(checkpoint["horizon"])
            raw_path = checkpoint.get("raw_optimizer_checkpoint_path")
            if horizon not in HORIZONS or raw_path is None:
                continue
            raw_hash = checkpoint.get("raw_optimizer_checkpoint_sha256")
            if not raw_hash:
                raise ValueError("recorded relaxation trajectory lacks SHA256")
            specs.append(
                {
                    "system": "c60",
                    "state_id": state_id,
                    "seed": seed,
                    "arm": arm,
                    "horizon": horizon,
                    "raw_optimizer_checkpoint_path": str(raw_path),
                    "raw_optimizer_checkpoint_sha256": str(raw_hash),
                    "relaxed_checkpoint": dict(checkpoint),
                }
            )
    keys = [pair_key(spec) for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("source contains duplicate counterfactual pairs")
    return specs


def causal_outcome(
    *,
    explicit_label: str,
    relaxed_label: str,
    landing_relation: str,
) -> str:
    """Classify only certified basin outcomes; numerical failures stay unknown."""

    if landing_relation not in LANDING_RELATIONS:
        raise ValueError("unknown landing relation")
    if (
        explicit_label not in LEARNABLE_BASIN_LABELS
        or relaxed_label not in LEARNABLE_BASIN_LABELS
    ):
        return "UNLEARNABLE"
    if explicit_label == "RETURN_STARTER" and relaxed_label == "RETURN_STARTER":
        return "BOTH_RETURN_STARTER"
    if explicit_label == "RETURN_STARTER":
        return "RELAXED_ONLY_ESCAPE"
    if relaxed_label == "RETURN_STARTER":
        return "EXPLICIT_ONLY_ESCAPE"
    if landing_relation == "SAME_LANDING":
        return "SAME_ESCAPED_LANDING"
    if landing_relation == "DIFFERENT_LANDING":
        return "DIFFERENT_ESCAPED_LANDINGS"
    return "UNLEARNABLE"


def repeated_contexts(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    required_seed_count: int,
    required_context_size: int,
) -> list[dict[str, Any]]:
    """Return complete contexts with a repeated paired outcome."""

    contexts: list[dict[str, Any]] = []
    for state_id in STATE_IDS:
        for arm in ARMS:
            for horizon in HORIZONS:
                context = [
                    row
                    for row in rows
                    if str(row["system"]) == "c60"
                    and str(row["state_id"]) == state_id
                    and str(row["arm"]) == arm
                    and int(row["horizon"]) == horizon
                ]
                seeds = sorted({int(row["seed"]) for row in context})
                if len(seeds) != required_context_size:
                    continue
                matching = sorted(
                    int(row["seed"])
                    for row in context
                    if row["causal_outcome"] == outcome
                )
                if len(matching) >= required_seed_count:
                    contexts.append(
                        {
                            "system": "c60",
                            "state_id": state_id,
                            "arm": arm,
                            "horizon": horizon,
                            "seed_count": len(matching),
                            "seeds": matching,
                        }
                    )
    return contexts


def decide(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply preregistered causal stopping rules without score weights."""

    learnable = [
        row for row in rows if row["causal_outcome"] != "UNLEARNABLE"
    ]
    repeated = repeated_contexts(
        rows,
        outcome="RELAXED_ONLY_ESCAPE",
        required_seed_count=2,
        required_context_size=3,
    )
    if repeated:
        return {
            "decision": "RETAIN_RELAXATION_CAUSAL_SIGNAL",
            "repeated_relaxed_only_contexts": repeated,
        }
    exact_outcomes = {
        "BOTH_RETURN_STARTER",
        "SAME_ESCAPED_LANDING",
    }
    if len(learnable) == len(rows) and all(
        row["causal_outcome"] in exact_outcomes for row in learnable
    ):
        return {
            "decision": "EXACT_REDUNDANCY_SIGNAL",
            "repeated_relaxed_only_contexts": [],
        }
    return {
        "decision": "MIXED_NO_DEFAULT_CHANGE",
        "repeated_relaxed_only_contexts": [],
    }


def _validate_ledger(row: Mapping[str, Any]) -> int:
    counts = row["explicit_purpose_counts"]
    if int(counts.get("unattributed", -1)) != 0:
        raise ValueError("explicit replay contains unattributed evaluations")
    if int(counts.get("direction_oracle", -1)) != 0:
        raise ValueError("explicit replay repeated direction evaluations")
    if int(counts.get("biased_proposal_relax", -1)) != 0:
        raise ValueError("explicit replay performed biased relaxation")
    total = sum(int(value) for value in counts.values())
    if total != int(row["new_force_evaluations"]):
        raise ValueError("explicit replay ledger does not close")
    return total


def build_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_specs: Sequence[Mapping[str, Any]],
    max_new_force_evaluations: int = MAX_NEW_FORCE_EVALUATIONS,
) -> dict[str, Any]:
    expected = {pair_key(spec) for spec in expected_specs}
    observed = {pair_key(row) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("evidence does not contain the exact replay cohort")
    total = 0
    outcomes: Counter[str] = Counter()
    explicit_labels: Counter[str] = Counter()
    relaxed_labels: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "completed":
            raise ValueError("every replay pair must complete")
        outcome = str(row["causal_outcome"])
        if outcome not in CAUSAL_OUTCOMES:
            raise ValueError("unknown causal outcome")
        total += _validate_ledger(row)
        outcomes[outcome] += 1
        explicit_labels[str(row["explicit_label"])] += 1
        relaxed_labels[str(row["relaxed_label"])] += 1
    if total > max_new_force_evaluations:
        raise ValueError("new force-evaluation budget exceeded")
    decision = decide(rows)
    return {
        "schema_version": 1,
        "cohort": {
            "systems": list(SYSTEMS),
            "state_ids": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "horizons": list(HORIZONS),
            "pair_count": len(rows),
        },
        "outcome_counts": dict(sorted(outcomes.items())),
        "explicit_label_counts": dict(sorted(explicit_labels.items())),
        "relaxed_label_counts": dict(sorted(relaxed_labels.items())),
        "new_force_evaluations": total,
        "max_new_force_evaluations": max_new_force_evaluations,
        "unattributed_force_evaluations": 0,
        **decision,
        "production_default_changed": False,
        "claim_ceiling": (
            "paired C60 local counterfactual at recorded SSW macro states; "
            "not an online explicit-only walk or cross-system promotion"
        ),
        "pairs": list(rows),
    }
