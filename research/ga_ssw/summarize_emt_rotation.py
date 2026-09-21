"""Offline audit and cost-node summary for the Cu55 EMT rotation control."""
import argparse
import json
import math
from pathlib import Path


FRESH_FMAX = 0.03
COST_NODES = (1500, 3000, 6000)


def read(path):
    return json.loads(Path(path).read_text())


def pair_geometry(atoms, cutoffs=(3.0, 3.2, 3.4)):
    positions = atoms["positions"]
    n = len(positions)
    distances = []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        i, j = find(i), find(j)
        if i != j:
            parent[j] = i

    for i in range(n):
        xi = positions[i]
        for j in range(i + 1, n):
            xj = positions[j]
            d = math.sqrt(sum((xi[k] - xj[k]) ** 2 for k in range(3)))
            distances.append((d, i, j))
    components = {}
    for cutoff in cutoffs:
        parent = list(range(n))
        for d, i, j in distances:
            if d <= cutoff:
                union(i, j)
        components[str(cutoff)] = len({find(i) for i in range(n)})
    return {
        "shortest_pair_distance_A": min(d for d, _, _ in distances),
        "diameter_A": max(d for d, _, _ in distances),
        "components_by_cutoff_A": components,
    }


def summarize_arm(root, row):
    name = f"cu55-octa-seed{row['seed']}-{row['arm']}"
    folder = root / name
    result_path = folder / "result.json"
    if not result_path.exists():
        return {
            "name": name, "seed": row["seed"], "arm": row["arm"],
            "status": row.get("status"), "missing_result": True,
            "search_calls": row.get("search_calls"),
            "fresh_calls": row.get("fresh_calls"),
        }
    result = read(result_path)
    checks = {int(x["index"]): x for x in row.get("fresh_checks", [])}
    initial = result["initial"]
    initial_cost = int(initial["evaluation_requests"])
    cost = initial_cost
    minima_costs = [initial_cost]
    landing_records = []
    for record in result.get("records", []):
        cost += int(record["evaluation_requests"])
        landing = record.get("landing")
        if landing is not None and bool(landing.get("converged")):
            minima_costs.append(cost)
            landing_records.append(record)
    paid_result = int(result["evaluation_requests"])
    paid_summary = int(row.get("search_calls", paid_result))
    accounting = {
        "initial_requests": initial_cost,
        "record_requests": [int(x["evaluation_requests"]) for x in result.get("records", [])],
        "result_requests": paid_result,
        "summary_requests": paid_summary,
        "initial_plus_records_equals_result": cost == paid_result,
        "result_equals_summary": paid_result == paid_summary,
        "terminal_status": result.get("status"),
        "terminal_record_status": (result.get("records") or [{}])[-1].get("status"),
        "terminal_cost_included": True,
        "denials": row.get("denials", 0),
        "boundary": row.get("boundary"),
    }
    minima = result.get("minima", [])
    if len(minima) != len(minima_costs):
        accounting["minima_record_mapping_error"] = {
            "result_minima": len(minima), "mapped_costs": len(minima_costs)
        }
    if cost != paid_result or paid_result != paid_summary or len(minima) != len(minima_costs):
        raise ValueError(f"invalid accounting or minima mapping: {name}")
    minima_rows = []
    for index, minimum in enumerate(minima):
        check = checks.get(index)
        fresh_qualified = bool(check and check.get("fmax_eV_A", math.inf) <= FRESH_FMAX
                               and check.get("qualified", False))
        minima_rows.append({
            "index": index,
            "cost": minima_costs[index] if index < len(minima_costs) else None,
            "energy_eV": check.get("energy_eV") if check else minimum.get("energy"),
            "delta_from_initial_eV": ((check.get("energy_eV") - initial["energy"])
                                       if check and check.get("energy_eV") is not None else None),
            "fresh_fmax_eV_A": check.get("fmax_eV_A") if check else None,
            "fresh_qualified": fresh_qualified,
            "geometry": pair_geometry(minimum["atoms"]),
        })
    node_rows = []
    for node in COST_NODES:
        if node > paid_summary:
            node_rows.append({"search_cost_node": node, "available": False,
                              "completed_landings_excluding_initial": None,
                              "accepted_landings_excluding_initial": None,
                              "eligible_fresh_qualified_minima": None,
                              "best_energy_eV_including_initial": None,
                              "best_delta_from_initial_eV": None,
                              "best_minimum_index": None})
            continue
        eligible = [x for x in minima_rows if x["cost"] is not None and x["cost"] <= node
                    and x["fresh_qualified"] and x["energy_eV"] is not None]
        best = min(eligible, key=lambda x: x["energy_eV"]) if eligible else None
        node_rows.append({
            "search_cost_node": node,
            "available": True,
            "completed_landings_excluding_initial": sum(c <= node for c in minima_costs[1:]),
            "accepted_landings_excluding_initial": sum(bool(r.get("accepted")) for c,r in zip(minima_costs[1:], landing_records) if c <= node),
            "eligible_fresh_qualified_minima": len(eligible),
            "best_energy_eV_including_initial": None if best is None else best["energy_eV"],
            "best_delta_from_initial_eV": None if best is None else best["delta_from_initial_eV"],
            "best_minimum_index": None if best is None else best["index"],
        })
    completed = len(landing_records)
    accepted = sum(bool(x.get("accepted")) for x in landing_records)
    return {
        "name": name, "seed": row["seed"], "arm": row["arm"],
        "status": row.get("status"), "missing_result": False,
        "search_calls": paid_summary, "fresh_calls": row.get("fresh_calls"),
        "initial_energy_eV": initial["energy"],
        "completed_landings_excluding_initial": completed,
        "accepted_landings_excluding_initial": accepted,
        "accounting": accounting,
        "minima": minima_rows,
        "cost_nodes": node_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite {output}")
    summary = read(root / "summary.json")
    plan = read(root / "plan.json")
    expected = {(seed, arm) for seed in plan["seeds"] for arm in plan["arms"]}
    observed = {(r["seed"], r["arm"]) for r in summary}
    missing = sorted(expected - observed)
    rows = [summarize_arm(root, row) for row in summary]
    report = {
        "source": str(root),
        "scope": "Offline EMT control audit only; no new PES calls and no pooling with MH1.",
        "fresh_force_threshold_eV_A": FRESH_FMAX,
        "cost_nodes_search_EF": list(COST_NODES),
        "arms": rows,
        "missing_arms": missing,
        "all_present_results_accounted": not missing and all(
            not row.get("missing_result") and row["accounting"]["initial_plus_records_equals_result"]
            and row["accounting"]["result_equals_summary"] for row in rows
        ),
        "structural_note": "Shortest distance, diameter, and cutoff components are explicit empirical diagnostics; they are not chemical bond assignments or basin counts.",
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"arms": len(rows), "output": str(output),
                      "all_present_results_accounted": report["all_present_results_accounted"]}))


if __name__ == "__main__":
    main()
