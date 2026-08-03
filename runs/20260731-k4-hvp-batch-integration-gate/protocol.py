from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Mapping, Sequence


def evaluate_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    minimum_speedup: float = 1.3,
) -> dict[str, object]:
    grouped: dict[tuple[str, str, int], dict[str, Mapping[str, object]]] = (
        defaultdict(dict)
    )
    for row in rows:
        key = (
            str(row["system"]),
            str(row["context"]),
            int(row["repetition"]),
        )
        grouped[key][str(row["mode"])] = row

    pair_rows = []
    for (system, context, repetition), modes in sorted(grouped.items()):
        if set(modes) != {"serial", "batch"}:
            raise ValueError("each comparison requires serial and batch rows")
        serial = modes["serial"]
        batch = modes["batch"]
        serial_time = float(serial["wall_time_s"])
        batch_time = float(batch["wall_time_s"])
        pair_rows.append(
            {
                "system": system,
                "context": context,
                "repetition": repetition,
                "speedup": serial_time / batch_time,
                "same_kind": serial["selected_kind"] == batch["selected_kind"],
                "direction_cosine": float(batch["direction_cosine_to_serial"]),
                "curvature_abs_error": abs(
                    float(batch["curvature"]) - float(serial["curvature"])
                ),
                "score_abs_error": abs(
                    float(batch["score"]) - float(serial["score"])
                ),
                "closed_cost": (
                    int(serial["force_evaluations"])
                    == int(batch["force_evaluations"])
                    == 8
                    and int(serial["unattributed"])
                    == int(batch["unattributed"])
                    == 0
                ),
            }
        )

    strata = []
    for system, context in sorted(
        {(row["system"], row["context"]) for row in pair_rows}
    ):
        selected = [
            row
            for row in pair_rows
            if row["system"] == system and row["context"] == context
        ]
        median_speedup = median(float(row["speedup"]) for row in selected)
        strata.append(
            {
                "system": system,
                "context": context,
                "comparisons": len(selected),
                "median_speedup": median_speedup,
                "minimum_direction_cosine": min(
                    float(row["direction_cosine"]) for row in selected
                ),
                "maximum_curvature_abs_error": max(
                    float(row["curvature_abs_error"]) for row in selected
                ),
                "maximum_score_abs_error": max(
                    float(row["score_abs_error"]) for row in selected
                ),
                "same_selected_kind": all(
                    bool(row["same_kind"]) for row in selected
                ),
                "closed_cost": all(bool(row["closed_cost"]) for row in selected),
                "passed": (
                    median_speedup >= minimum_speedup
                    and all(bool(row["same_kind"]) for row in selected)
                    and all(bool(row["closed_cost"]) for row in selected)
                    and min(
                        float(row["direction_cosine"]) for row in selected
                    )
                    >= 1.0 - 1.0e-12
                ),
            }
        )
    return {
        "minimum_speedup": minimum_speedup,
        "pair_count": len(pair_rows),
        "strata": strata,
        "passed": bool(strata) and all(bool(row["passed"]) for row in strata),
    }
