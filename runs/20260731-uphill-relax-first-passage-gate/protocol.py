"""Pure protocol for accepted-step first passage through biased relaxation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


TrajectoryKey = tuple[str, int, str, int]

DISCOVERY_KEYS: tuple[TrajectoryKey, ...] = (
    ("plateau_accepted", 42, "D0_exact_anchor", 1),
    ("plateau_accepted", 43, "D0_exact_anchor", 1),
    ("plateau_accepted", 43, "K4_discrete", 2),
    ("plateau_accepted", 44, "K4_discrete", 2),
)
HOLDOUT_KEYS: tuple[TrajectoryKey, ...] = (
    ("intermediate_accepted", 42, "D0_exact_anchor", 4),
    ("plateau_accepted", 42, "K4_discrete", 1),
    ("plateau_accepted", 42, "K4_discrete", 4),
    ("plateau_accepted", 44, "K4_discrete", 4),
)
MAX_NEW_FORCE_EVALUATIONS = 32_000
MAX_KERNEL_WALL_TIME_S = 900.0


def trajectory_key(row: Mapping[str, Any]) -> TrajectoryKey:
    return (
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
        int(row["horizon"]),
    )


def select_pairs(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[TrajectoryKey],
) -> list[dict[str, Any]]:
    by_key = {trajectory_key(row): dict(row) for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("source contains duplicate trajectory keys")
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"source is missing frozen trajectories: {missing}")
    return [by_key[key] for key in keys]


def _suffix_start(rows: Sequence[Mapping[str, Any]], predicate) -> int | None:
    start: int | None = None
    for row in reversed(rows):
        if not predicate(row):
            break
        start = int(row["frame_index"])
    return start


def summarize_trajectory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("trajectory requires at least one frame")
    indices = [int(row["frame_index"]) for row in rows]
    if indices != list(range(len(rows))):
        raise ValueError("frame indices must be complete and consecutive")
    if rows[-1]["label"] != "ESCAPED_CERTIFIED":
        raise ValueError("discovery final frame must be a certified escape")
    if rows[-1]["landing_relation_to_final"] != "SAME_LANDING":
        raise ValueError("discovery final frame must match its final basin")
    first_escape = next(
        (
            int(row["frame_index"])
            for row in rows
            if row["label"] == "ESCAPED_CERTIFIED"
        ),
        None,
    )
    stable_escape = _suffix_start(
        rows,
        lambda row: row["label"] == "ESCAPED_CERTIFIED",
    )
    stable_final = _suffix_start(
        rows,
        lambda row: (
            row["label"] == "ESCAPED_CERTIFIED"
            and row["landing_relation_to_final"] == "SAME_LANDING"
        ),
    )
    return {
        "frame_count": len(rows),
        "final_frame_index": indices[-1],
        "first_escape_step": first_escape,
        "stable_escape_step": stable_escape,
        "stable_final_basin_step": stable_final,
        "unlearnable_frame_count": sum(
            row["label"]
            not in {"RETURN_STARTER", "ESCAPED_CERTIFIED"}
            or row["landing_relation_to_final"] == "AMBIGUOUS_LANDING"
            for row in rows
        ),
    }


def derive_cutoff(summaries: Sequence[Mapping[str, Any]]) -> int:
    if len(summaries) != len(DISCOVERY_KEYS):
        raise ValueError("cutoff requires all four discovery trajectories")
    values = [summary.get("stable_final_basin_step") for summary in summaries]
    if any(value is None for value in values):
        raise ValueError("every discovery trajectory needs stable final-basin arrival")
    return max(int(value) for value in values)


def decide_holdout(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    observed = {trajectory_key(row) for row in rows}
    if len(rows) != len(HOLDOUT_KEYS) or observed != set(HOLDOUT_KEYS):
        raise ValueError("holdout requires the exact four frozen trajectories")
    passed = all(
        row.get("status") == "completed"
        and bool(row.get("has_headroom"))
        and row.get("cutoff_label") == "ESCAPED_CERTIFIED"
        and row.get("cutoff_relation_to_final") == "SAME_LANDING"
        for row in rows
    )
    return {
        "decision": (
            "OPEN_FIXED_CUTOFF_FULL_ACTION_GATE"
            if passed
            else "RETAIN_CURRENT_LENGTH_NO_PROMOTION"
        ),
        "holdout_reproduced_final_basin_count": sum(
            row.get("cutoff_label") == "ESCAPED_CERTIFIED"
            and row.get("cutoff_relation_to_final") == "SAME_LANDING"
            for row in rows
        ),
        "holdout_headroom_count": sum(
            bool(row.get("has_headroom")) for row in rows
        ),
    }

