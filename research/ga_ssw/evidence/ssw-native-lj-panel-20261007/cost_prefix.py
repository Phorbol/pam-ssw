#!/usr/bin/env python3
"""Read matched request prefixes of a previously analyzed LJ development panel.

No PES, graph matching, interpolation, or extrapolation is performed. Endpoint
energies are only credited after the complete qualified quench has been paid.
"""
import argparse
import json
from pathlib import Path


def curve(run):
    if run["kind"] == "native":
        return [(e["cumulative_paid_ef"], e["energy_eV"])
                for e in run["events"] if e.get("force_domain_qualified")
                and e.get("connectivity", {}).get("single_cluster")]
    minima = {m["index"]: m for m in run["minima"]}
    points = []
    for event in run["progress_events"]:
        index = 0 if event["kind"] == "initial" else event.get("new_minimum_index")
        minimum = minima.get(index)
        if minimum is None:
            continue
        if minimum.get("force_domain_qualified") and minimum.get("connectivity", {}).get("single_cluster"):
            if index:
                assert abs(minimum["energy_eV"] - event["landing_energy_eV"]) < 1e-8
            points.append((event["evaluation_requests"], minimum["energy_eV"]))
    return points


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    args.output.mkdir()
    rows = []
    report = ["# LJ development panel: matched request prefixes", "",
              "Post hoc accounting of the frozen six-arm panel, not independent performance validation.",
              "Prefix credits a connected, force/domain-qualified endpoint only after its full cost; includes each arm's initialization.",
              "Native is the bounded periodic-representation whole-program reference. No native/Python component attribution.", "",
              "| N | Common paid request prefix | Rotation best E | Full/per_atom best E | Native best E |",
              "|---:|---:|---:|---:|---:|"]
    stopping_notes = []
    for size in (55, 38):
        runs = [r for r in data["runs"] if r["n"] == size]
        assert len(runs) == 3
        paid = {r["arm"]: (r["costs"]["surface_requests"] if r["kind"] == "python"
                           else r["costs"]["external_successful_requests"]) for r in runs}
        common = min(paid.values())
        budgets = sorted({b for b in (100, 250, 500, 1000, common) if b <= common})
        curves = {r["arm"]: curve(r) for r in runs}
        for budget in budgets:
            energies = {arm: min((e for cost, e in points if cost <= budget), default=None)
                        for arm, points in curves.items()}
            rows.append(dict(n=size, request_prefix=budget, best_qualified_energy_eV=energies))
            report.append(f"| {size} | {budget} | " + " | ".join(
                "NA" if energies[a] is None else f"{energies[a]:.8f}"
                for a in ("rotation", "full", "native")) + " |")
        stopping_notes.append(f"N{size} total paid requests: {paid}; shared observed prefix {common}.")
    report.extend(["", *stopping_notes, "",
                   "Observed common prefix depends on run stopping and is a development diagnostic, not a preregistered statistical estimator."])
    (args.output / "prefixes.json").write_text(json.dumps(dict(source=str(args.analysis.resolve()),
        qualification="intact force/domain-qualified endpoint, not Hessian index", rows=rows), indent=2)+"\n")
    (args.output / "README.md").write_text("\n".join(report)+"\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
