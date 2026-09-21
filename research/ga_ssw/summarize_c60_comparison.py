"""Compact, non-recomputing readout of a fixed-budget C60 comparison JSON."""
import argparse
import json
from pathlib import Path


def best_event(events):
    eligible = [event for event in events if event.get("fresh_qualified") and event.get("energy_eV") is not None]
    if not eligible:
        return None
    return min(eligible, key=lambda event: event["energy_eV"])


def geometry_summary(event):
    geometry = event.get("geometry") or {}
    return {
        str(cut): {
            key: value
            for key, value in (geometry.get(str(cut)) or {}).items()
            if key in {"components", "edges", "degree_counts", "three_connected", "planar", "face_counts", "graph_cage_candidate", "ih_graph_match"}
        }
        for cut in (1.64, 1.7, 1.8)
    }


def arm_readout(arm, reference_energy):
    events = arm.get("events", [])
    best = best_event(events)
    summary = arm.get("summary") or {}
    if "search_requests" in summary:
        actual_calls = summary.get("search_requests")
    else:
        actual_calls = arm.get("actual_search_calls")
    fresh_present = sum(bool(event.get("fresh_check_present")) for event in events)
    fresh_qualified = sum(bool(event.get("fresh_qualified")) for event in events)
    return {
        "status": summary.get("status", summary.get("process", {}).get("state")),
        "returncode": summary.get("process", {}).get("returncode"),
        "actual_search_calls": actual_calls,
        "fresh_check_present": fresh_present,
        "fresh_qualified": fresh_qualified,
        "event_count": len(events),
        "unmatched_events": summary.get("unmatched_events"),
        "best": None if best is None else {
            "event_index": best.get("index"),
            "cumulative_search_calls": best.get("cumulative_search_calls"),
            "energy_eV": best.get("energy_eV"),
            "energy_minus_reference_eV": best["energy_eV"] - reference_energy,
            "fmax_eV_per_A": best.get("fmax_eV_per_A"),
            "cage_candidate": best.get("cage_candidate", False),
            "geometry": geometry_summary(best),
        },
        "budgets": [
            {
                key: budget.get(key)
                for key in (
                    "budget_search_calls", "budget_reached", "actual_search_calls",
                    "events_within_budget", "qualified_minima", "best_energy_eV",
                    "cage_candidates", "energy_successes", "cage_energy_intersections",
                    "qualification_basis", "missing_or_truncated",
                )
                if key in budget
            }
            for budget in arm.get("budgets", [])
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.input.read_text())
    reference_energy = source["reference"]["energy_eV"]
    output = {
        "source": str(args.input),
        "scope": source.get("scope"),
        "budgets": source.get("budgets"),
        "reference": source["reference"],
        "cases": {
            seed: {
                method: arm_readout(arm, reference_energy)
                for method, arm in case.items()
            }
            for seed, case in source.get("cases", {}).items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
