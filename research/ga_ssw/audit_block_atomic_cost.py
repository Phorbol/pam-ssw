"""Zero-PES accounting audit of completed Gaussian prefixes in block runs."""
import json
from pathlib import Path
import numpy as np


ROOT = Path("research/ga_ssw/fe7c3-block-baseline")


def audit(seed):
    path = ROOT / "comparison" / f"safe_total-seed{seed}" / "result.json"
    data = json.loads(path.read_text())
    outer = data["records"][2]
    atomic = outer["atomic"]
    checkpoint = atomic["checkpoint"]
    reference = float(checkpoint["reference_energy"])
    events = []
    previous = None
    for event in checkpoint["climb"]:
        direction = np.asarray(event["direction"], dtype=float).reshape(-1)
        direction_norm = float(np.linalg.norm(direction))
        row = {
            "index": int(event["index"]),
            "true_energy": float(event["true_energy"]),
            "biased_energy": float(event["biased_energy"]),
            "true_minus_reference": float(event["true_energy"] - reference),
            "biased_minus_reference": float(event["biased_energy"] - reference),
            "bias_weight": float(event["weight"]),
            "direction_norm": direction_norm,
            "rotation_force_requests": int(event["rotation_force_requests"]),
            "height_requests": 1,
            "biased_quench_requests": int(event["quench_requests"]),
            "true_energy_requests": 1,
            # Height is one charged evaluate on the displaced structure and
            # must be included before the true-energy check.
            "event_requests_before_true_check": int(event["rotation_force_requests"] + 1 + event["quench_requests"]),
            "event_requests_including_true_check": int(event["rotation_force_requests"] + 1 + event["quench_requests"] + 1),
        }
        if previous is None:
            row["direction_cosine_previous"] = None
            row["direction_angle_deg_previous"] = None
        else:
            cosine = float(np.dot(direction, previous) / (direction_norm * np.linalg.norm(previous)))
            cosine = float(np.clip(cosine, -1., 1.))
            row["direction_cosine_previous"] = cosine
            row["direction_angle_deg_previous"] = float(np.degrees(np.arccos(cosine)))
        previous = direction
        events.append(row)
    pending = checkpoint.get("pending")
    return {
        "seed": seed,
        "outer_index": int(outer["index"]),
        "outer_status": outer["status"],
        "outer_requests": int(outer["requests"]),
        "outer_start_reference_energy": reference,
        "atomic_status": atomic["status"],
        "atomic_requests": int(atomic["requests"]),
        "completed_gaussians": len(events),
        "events": events,
        "all_completed_true_above_reference": all(e["true_minus_reference"] >= 0. for e in events),
        "any_completed_true_below_reference": any(e["true_minus_reference"] < 0. for e in events),
        "checkpoint": {
            "next_index": int(checkpoint["next_index"]),
            "previous_requests": int(checkpoint["previous_requests"]),
            "pending_index": None if pending is None else pending.get("index"),
            "pending_stage": None if pending is None else pending.get("stage"),
            "pending_has_displaced": bool(pending and "displaced" in pending),
        },
    }


def main():
    payload = {
        "status": "audited",
        "zero_pes": True,
        "formula": "true_minus_reference = completed Gaussian true_energy - outer starting E+pV; bias weight=(forward_force-force_parallel)*width*exp(0.5)",
        "runs": [audit(7), audit(101)],
        "interpretation": "This is an accounting and stopping audit. Energy ordering is not a basin identity test.",
    }
    out = ROOT / "atomic-cost-summary.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(out)


if __name__ == "__main__":
    main()
