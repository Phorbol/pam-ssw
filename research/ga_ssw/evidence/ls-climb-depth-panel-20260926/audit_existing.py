"""Zero-PES locality and graph-change audit of existing SSW/NativeLS toggles."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

EVIDENCE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence")
SPECS = []
for arm, ls_on in (("ssw", False), ("native_ls", True)):
    for seed in (61, 67):
        SPECS.append({
            "environment": "c4h6_mh1_omol", "arm": arm, "ls_on": ls_on, "seed": seed,
            "path": EVIDENCE / "c4h6-mh1-coverage-20260924" / f"{arm}-seed{seed}" / "result.json",
            "cutoffs": {(1, 1): 0.8400000095367432, (1, 6): 1.190000033378601, (6, 6): 1.6399999618530273},
        })
for arm, ls_on in (("baseline", False), ("native_ls", True)):
    base = "mh1-native-ls-equal-budget-20260920" if ls_on else "mh1-equal-budget-rotation-20260920"
    for seed in (17093, 17094):
        SPECS.append({
            "environment": "c60_mh1_omol", "arm": arm, "ls_on": ls_on, "seed": seed,
            "path": EVIDENCE / base / f"c60_{seed}-first-{arm}" / "result.json",
            "cutoffs": {(6, 6): 1.64},
        })


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def serialized(atoms):
    if not isinstance(atoms, dict):
        return None
    return atoms


def graph_edges(atoms, cutoffs):
    numbers = atoms["numbers"]
    positions = atoms["positions"]
    edges = set()
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            key = tuple(sorted((int(numbers[i]), int(numbers[j]))))
            cutoff = cutoffs.get(key)
            if cutoff is None:
                continue
            d2 = sum((float(positions[i][k]) - float(positions[j][k])) ** 2 for k in range(3))
            if d2 <= cutoff * cutoff:
                edges.add((i, j))
    return edges


def component_count(n, edges):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a
    return len({find(i) for i in range(n)})


def direction_locality(direction):
    mag2 = [sum(float(x) ** 2 for x in atom) for atom in direction]
    total = sum(mag2)
    if total <= 0:
        return None
    weights = [v / total for v in mag2]
    participation = 1.0 / sum(w * w for w in weights)
    ordered = sorted(weights, reverse=True)
    running = 0.0
    n90 = 0
    for w in ordered:
        running += w
        n90 += 1
        if running >= 0.9:
            break
    return {"effective_atoms_participation_ratio": participation,
            "atoms_carrying_90pct_direction_norm": n90,
            "fraction_of_atoms_carrying_90pct": n90 / len(weights)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise FileExistsError(output)
    arm_rows = []
    all_hashes = {}
    for spec in SPECS:
        path = spec["path"]
        data = json.loads(path.read_text())
        all_hashes[str(path)] = sha(path)
        records = data.get("records", [])
        if not records:
            raise ValueError(f"no records in {path}")
        current = serialized(data["initial"].get("atoms"))
        if current is None:
            raise ValueError(f"no initial geometry in {path}")
        cumulative = int(data["initial"].get("evaluation_requests", 0))
        attempt_rows = []
        for index, record in enumerate(records):
            if record.get("index") != index:
                raise ValueError(f"record index mismatch in {path} at {index}")
            cost = int(record.get("evaluation_requests", 0))
            cumulative += cost
            landing = record.get("landing")
            landed = serialized(landing.get("atoms")) if isinstance(landing, dict) else None
            climb = record.get("climb")
            row = {
                "record": index,
                "status": record.get("status"),
                "accepted": record.get("accepted"),
                "requests": cost,
                "cumulative_requests": cumulative,
                "ls_preparation_requests": int((record.get("ls_preparation") or {}).get("evaluation_requests", 0)),
                "ls_preparation_share": (
                    int((record.get("ls_preparation") or {}).get("evaluation_requests", 0)) / cost if cost else None
                ),
                "random_anchor_locality": direction_locality(record.get("initial_direction", [])),
                "first_refined_mode_locality": (
                    direction_locality(climb[0].get("direction", []))
                    if isinstance(climb, list) and climb and climb[0].get("direction") is not None else None
                ),
                "refined_mode_locality_by_gaussian": [
                    direction_locality(stage.get("direction", []))
                    for stage in (record.get("climb") or [])
                    if stage.get("direction") is not None
                ],
            }
            if landed is not None:
                before, after = graph_edges(current, spec["cutoffs"]), graph_edges(landed, spec["cutoffs"])
                added, removed = after - before, before - after
                components = component_count(len(landed["numbers"]), after)
                landing_fmax = landing.get("max_force")
                qualified = landing_fmax is not None and float(landing_fmax) <= 0.03
                row.update({
                    "landing_fmax_eV_A": landing_fmax,
                    "force_qualified_003": qualified,
                    "components_at_configured_cutoff": components,
                    "connected_landing": components == 1,
                    "edge_changes_vs_selected_current": len(added) + len(removed),
                    "edge_additions": len(added),
                    "edge_removals": len(removed),
                    "connected_qualified_graph_change": bool(components == 1 and qualified and (added or removed)),
                    "accepted_connected_qualified_graph_change": bool(record.get("accepted") and components == 1 and qualified and (added or removed)),
                })
            else:
                row["landing_status"] = "no physical landing"
            attempt_rows.append(row)
            if record.get("accepted") and landed is not None:
                current = landed
        if cumulative != int(data.get("evaluation_requests", cumulative)):
            raise ValueError(f"result/record request totals disagree in {path}")
        requests_at_200k = next((r["record"] for r in attempt_rows if r["cumulative_requests"] >= 200000), None)
        arm_rows.append({
            "environment": spec["environment"], "arm": spec["arm"], "ls_on": spec["ls_on"], "seed": spec["seed"],
            "source": str(path), "source_sha256": all_hashes[str(path)],
            "records": len(records), "total_search_requests": cumulative,
            "ls_preparation_requests": sum(r["ls_preparation_requests"] for r in attempt_rows),
            "attempt_rows": attempt_rows,
            "first_record": attempt_rows[0],
            "complete_attempts": sum(r["status"] not in ("evaluation_failed", "error", "failed") for r in attempt_rows),
            "landings": sum("edge_changes_vs_selected_current" in r for r in attempt_rows),
            "force_qualified_landings": sum(bool(r.get("force_qualified_003")) for r in attempt_rows),
            "connected_qualified_graph_changes": sum(bool(r.get("connected_qualified_graph_change")) for r in attempt_rows),
            "accepted_connected_qualified_graph_changes": sum(bool(r.get("accepted_connected_qualified_graph_change")) for r in attempt_rows),
            "first_record_cumulative": attempt_rows[0]["cumulative_requests"],
            "record_at_or_crossing_200k": requests_at_200k,
        })
    out = {
        "status": "offline_existing_data_only",
        "pes_evaluations": 0,
        "question": "Do existing NativeLS toggles measurably concentrate the initial escape direction and produce more connected, force-qualified bond-graph changes per paid request?",
        "definitions": {
            "random anchor locality": "inverse participation ratio of per-atom squared initial_direction magnitudes; this is the sampled anchor before mode refinement",
            "refined Gaussian-mode locality": "same metric on climb[i].direction, the solver-refined direction at each Gaussian stage",
            "graph": "same-atom-order edge changes using archived C4H6 HC length+0.10 A cutoffs and C60 1.64 A cutoff; diagnostic proxy, not a minimum identity",
            "request accounting": "initial plus all per-record E/F requests; LS preparation requests shown as a subset, not double-counted",
            "qualification": "saved landing force <= 0.03 eV/A; no new freshness evaluations",
        },
        "source_hashes": all_hashes,
        "arms": arm_rows,
        "toggle_pairs": [],
        "limits": [
            "C4H6 and C60 use same MH-1/omol head but different size and physical objective; interpret paired seed arms only within each system.",
            "Only the first attempt starts from an identical saved geometry across each method toggle. Later same-index events are not paired states after trajectories diverge.",
            "Per-pair LS force/Hessian contributions are not saved. Direction locality and graph changes cannot establish that LS altered local stiffness as the cause.",
            "TiO2/OMAT and AlOH have no matching LS-off trajectory in this evidence set; they are not added as pseudo-controls.",
            "Connected graph change is a proxy. It is not a stable-minimum, phase, reaction-path, or chemical-accuracy certificate.",
        ],
    }
    # The serialized record-level initial_direction is the random anchor; the
    # first refined mode is stored separately at climb[0].direction.
    pairs = {}
    for row in arm_rows:
        pairs.setdefault((row["environment"], row["seed"]), {})[row["arm"]] = row
    for (environment, seed), arms in sorted(pairs.items()):
        if "native_ls" not in arms or not ("ssw" in arms or "baseline" in arms):
            continue
        off = arms.get("ssw", arms.get("baseline"))
        on = arms["native_ls"]
        off_source = json.loads(Path(off["source"]).read_text())
        on_source = json.loads(Path(on["source"]).read_text())
        off_direction = off_source["records"][0]["initial_direction"]
        on_direction = on_source["records"][0]["initial_direction"]
        off_anchor_neff = off["first_record"]["random_anchor_locality"]["effective_atoms_participation_ratio"]
        on_anchor_neff = on["first_record"]["random_anchor_locality"]["effective_atoms_participation_ratio"]
        off_mode_neff = off["first_record"]["first_refined_mode_locality"]["effective_atoms_participation_ratio"]
        on_mode_neff = on["first_record"]["first_refined_mode_locality"]["effective_atoms_participation_ratio"]
        out["toggle_pairs"].append({
            "environment": environment, "seed": seed,
            "off_arm": off["arm"], "on_arm": on["arm"],
            "first_attempt_random_anchor_exactly_equal": off_direction == on_direction,
            "first_attempt_random_anchor_effective_atoms_off": off_anchor_neff,
            "first_attempt_random_anchor_effective_atoms_on": on_anchor_neff,
            "first_attempt_refined_mode_exactly_equal": off_source["records"][0]["climb"][0]["direction"] == on_source["records"][0]["climb"][0]["direction"],
            "first_attempt_refined_mode_effective_atoms_off": off_mode_neff,
            "first_attempt_refined_mode_effective_atoms_on": on_mode_neff,
            "full_run_all_connected_qualified_graph_changes_off": off["connected_qualified_graph_changes"],
            "full_run_all_connected_qualified_graph_changes_on": on["connected_qualified_graph_changes"],
            "full_run_accepted_connected_qualified_graph_change_off": off["accepted_connected_qualified_graph_changes"],
            "full_run_accepted_connected_qualified_graph_change_on": on["accepted_connected_qualified_graph_changes"],
            "full_run_search_requests_off": off["total_search_requests"],
            "full_run_search_requests_on": on["total_search_requests"],
            "full_run_preparation_requests_on": on["ls_preparation_requests"],
            "full_run_preparation_fraction_on": on["ls_preparation_requests"] / on["total_search_requests"],
        })
    out["arm_summaries"] = []
    for environment in sorted({row["environment"] for row in arm_rows}):
        for arm_name in sorted({row["arm"] for row in arm_rows if row["environment"] == environment}):
            selected = [row for row in arm_rows if row["environment"] == environment and row["arm"] == arm_name]
            anchors = [attempt["random_anchor_locality"]["effective_atoms_participation_ratio"]
                       for row in selected for attempt in row["attempt_rows"] if attempt["random_anchor_locality"]]
            modes = [mode["effective_atoms_participation_ratio"]
                     for row in selected for attempt in row["attempt_rows"]
                     for mode in attempt["refined_mode_locality_by_gaussian"] if mode]
            out["arm_summaries"].append({
                "environment": environment, "arm": arm_name,
                "seed_count": len(selected), "record_count": sum(x["records"] for x in selected),
                "random_anchor_effective_atom_count_median": statistics.median(anchors),
                "random_anchor_effective_atom_count_range": [min(anchors), max(anchors)],
                "refined_mode_effective_atom_count_median": statistics.median(modes),
                "refined_mode_effective_atom_count_range": [min(modes), max(modes)],
                "total_search_requests": sum(x["total_search_requests"] for x in selected),
                "ls_preparation_requests": sum(x["ls_preparation_requests"] for x in selected),
                "connected_qualified_graph_change_events": sum(x["connected_qualified_graph_changes"] for x in selected),
                "accepted_connected_qualified_graph_change_events": sum(x["accepted_connected_qualified_graph_changes"] for x in selected),
            })
    with output.open("x") as stream:
        stream.write(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(f"wrote {output}; PES evaluations=0; arms={len(arm_rows)}")
    for row in arm_rows:
        first = row["first_record"]
        print(row["environment"], row["arm"], row["seed"],
              "records", row["records"], "search_EF", row["total_search_requests"],
              "prep_EF", row["ls_preparation_requests"],
              "qual_graph_changes", row["connected_qualified_graph_changes"],
              "first_anchor_neff", (first.get("random_anchor_locality") or {}).get("effective_atoms_participation_ratio"),
              "first_refined_mode_neff", (first.get("first_refined_mode_locality") or {}).get("effective_atoms_participation_ratio"))


if __name__ == "__main__":
    main()
