"""Offline graph/ledger audit for the six-arm C4H6 coverage run.

This script only reads archived JSON/JSONL and ASE G2 structures; it never
constructs a calculator or performs a PES call.
"""
import argparse, json
from pathlib import Path
import networkx as nx
import numpy as np
from ase import Atoms
from ase.collections import g2
from pamssw.standalone.native_ls import HC_BOND_LENGTHS


def atoms_from_dict(d):
    return Atoms(numbers=d["numbers"], positions=d["positions"],
                 cell=d.get("cell"), pbc=d.get("pbc", False))


def graph(atoms):
    g = nx.Graph()
    g.add_nodes_from((i, {"number": int(atoms.numbers[i])}) for i in range(len(atoms)))
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            key = tuple(sorted((int(atoms.numbers[i]), int(atoms.numbers[j]))))
            cutoff = HC_BOND_LENGTHS[key] + 0.1
            if np.linalg.norm(atoms.positions[i] - atoms.positions[j]) <= cutoff:
                g.add_edge(i, j)
    return g


def graph_label(atoms, references):
    g = graph(atoms)
    comps = nx.number_connected_components(g)
    labels = []
    nm = nx.algorithms.isomorphism.categorical_node_match("number", None)
    for name, ref in references.items():
        if nx.is_isomorphic(g, ref, node_match=nm):
            labels.append(name)
    return dict(component_count=comps, reference_graphs=labels,
                graph_degree_sequence=sorted(dict(g.degree()).values()))


def component_formulas(atoms, g):
    return [Atoms(numbers=[int(atoms.numbers[i]) for i in sorted(nodes)]).get_chemical_formula()
            for nodes in nx.connected_components(g)]


def graph_signature(g):
    node_numbers = sorted(int(g.nodes[i]["number"]) for i in g.nodes)
    edges = sorted((min(g.nodes[i]["number"], g.nodes[j]["number"]),
                    max(g.nodes[i]["number"], g.nodes[j]["number"])) for i, j in g.edges)
    return (node_numbers, edges)


def assign_global_classes(entries):
    """Assign IDs by deterministic signature order, then exact isomorphism."""
    ordered = sorted(entries, key=lambda x: (graph_signature(x["graph"]), x["arm"], x["index"]))
    reps = []
    nm = nx.algorithms.isomorphism.categorical_node_match("number", None)
    for entry in ordered:
        found = None
        for class_id, rep in enumerate(reps):
            if nx.is_isomorphic(entry["graph"], rep, node_match=nm):
                found = class_id
                break
        if found is None:
            reps.append(entry["graph"])
            found = len(reps) - 1
        entry["class_id"] = found
    return reps


def landing_match(minimum, landing):
    if not landing or "atoms" not in landing:
        return False
    return (np.isclose(minimum.get("energy"), landing.get("energy"), atol=1e-9) and
            np.allclose(np.asarray(minimum["atoms"]["positions"]),
                        np.asarray(landing["atoms"]["positions"]), atol=1e-7))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = args.evidence.resolve()
    # G2 references are graph labels only. They are not energy or PBE oracles.
    references = {name: graph(g2[name].copy())
                  for name in ("butadiene", "cyclobutene", "2-butyne",
                               "methylenecyclopropane", "bicyclobutane")}
    rows = []
    all_entries = []
    expected = {f"butadiene-{arm}-seed{seed}" for arm in ("ssw", "paper_ls", "native_ls") for seed in (11, 29)}
    present = set()
    incomplete = {}
    for p in root.glob("butadiene-*-seed*"):
        result_file, summary_file, fresh_file = (p / n for n in ("result.json", "summary.json", "fresh-checks.json"))
        if not (result_file.exists() and summary_file.exists() and fresh_file.exists()):
            continue
        try:
            minima_n = len(json.loads(result_file.read_text()).get("minima", []))
            fresh_ids = {int(x["index"]) for x in json.loads(fresh_file.read_text()) if "index" in x}
            if fresh_ids >= set(range(minima_n)):
                present.add(p.name)
            else:
                incomplete[p.name] = {"reason": "fresh index coverage incomplete",
                                       "missing_indices": sorted(set(range(minima_n)) - fresh_ids)}
        except Exception as exc:
            incomplete[p.name] = {"reason": "artifact parse failure", "error": repr(exc)}
    for folder in sorted(root.glob("butadiene-*-seed*")):
        result_path, fresh_path, summary_path = (folder / n for n in
                                                  ("result.json", "fresh-checks.json", "summary.json"))
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text())
        fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else []
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        checks = {int(x["index"]): x for x in fresh if "index" in x}
        minima = result.get("minima", [])
        initial = result.get("initial", {})
        initial_energy = initial.get("energy")
        categories = []
        atoms_by_index = {}
        for index, minimum in enumerate(minima):
            atoms = atoms_from_dict(minimum["atoms"])
            atoms_by_index[index] = atoms
            label = graph_label(atoms, references)
            check = checks.get(index, {})
            g = graph(atoms)
            entry = dict(arm=folder.name, index=index, graph=g)
            all_entries.append(entry)
            row = dict(index=index, energy=minimum.get("energy"),
                       delta_from_initial=(None if initial_energy is None else
                                            minimum.get("energy") - initial_energy),
                       graph=label, fresh=check,
                       fragmented=label["component_count"] > 1,
                       component_formulas=component_formulas(atoms, g),
                       fresh_force_qualified=check.get("force_qualified"),
                       cumulative_search_requests=None)
            categories.append(row)
        records = result.get("records", [])
        cumulative = int(initial.get("evaluation_requests", 0) or 0)
        responses = []
        assert minima and minima[0] == initial
        categories[0]["cumulative_search_requests"] = cumulative
        next_minimum = 1
        prepared_response_records, native_normal_updates = 0, 0
        for record in records:
            cumulative += int(record.get("evaluation_requests", 0) or 0)
            landing = record.get("landing")
            if landing and landing.get("converged") and landing.get("surface") == "true":
                assert next_minimum < len(minima) and minima[next_minimum] == landing
                categories[next_minimum]["cumulative_search_requests"] = cumulative
                next_minimum += 1
            if record.get("energy_response") is not None:
                prepared_response_records += 1
            if isinstance(record.get("energy_response"), (int, float)):
                responses.append(float(record["energy_response"]))
            update = record.get("ls_update")
            if isinstance(update, dict) and "normal_update" in update.get("actions", ()):
                native_normal_updates += 1
        assert next_minimum == len(minima)
        assert cumulative == result["evaluation_requests"]
        ledger = folder / "evaluations.jsonl"
        ledger_rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        rows.append(dict(arm=folder.name, status=summary.get("status", result.get("status")),
            search_requests=summary.get("search_requests", result.get("evaluation_requests")),
            ledger_rows=len(ledger_rows), ledger_search_rows=sum(x.get("kind") in ("search", "search_failure") for x in ledger_rows),
            ledger_failures=sum(x.get("kind") == "search_failure" for x in ledger_rows),
            ledger_denials=sum(x.get("kind") == "search_denial" for x in ledger_rows),
            ledger_matches_requests=(sum(x.get("kind") in ("search", "search_failure") for x in ledger_rows) ==
                                     summary.get("search_requests", result.get("evaluation_requests"))),
            minima_count=len(minima), records=len(records),
            accepted=sum(bool(x.get("accepted")) for x in records),
            nonaccepted_statuses=sorted({x.get("status") for x in records if not x.get("accepted")} - {None}),
            failure_statuses=sorted({x.get("status") for x in records
                                     if str(x.get("status", "")).endswith("failed")}),
            ls_update_records=sum(x.get("ls_update") is not None for x in records),
            ls_response_records=sum(x.get("energy_response") is not None for x in records),
            prepared_response_records=prepared_response_records,
            energy_response_count=len(responses),
            energy_response_min=min(responses) if responses else None,
            energy_response_max=max(responses) if responses else None,
            energy_response_last=responses[-1] if responses else None,
            paper_target_eV_per_atom=0.7,
            native_controller_normal_updates=native_normal_updates,
            ls_controller_note="paper LS records expose no ls_update object; response count is not controller-success count; native count requires normal_update action",
            fresh_requests=summary.get("fresh_requests"), fresh_failures=sum("error" in x for x in fresh),
            graph_categories=categories))
    reps = assign_global_classes(all_entries)
    by_key = {(x["arm"], x["index"]): x for x in all_entries}
    for row in rows:
        for item in row["graph_categories"]:
            x = by_key[(row["arm"], item["index"])]
            item["class_id"] = x["class_id"]
        qualified = [item for item in row["graph_categories"] if item["fresh_force_qualified"] is True]
        counts = {}
        for item in qualified:
            counts[item["class_id"]] = counts.get(item["class_id"], 0) + 1
        row["qualified_class_counts"] = counts
        excluding = {}
        for item in qualified:
            if item["index"] != 0:
                key = str(item["class_id"])
                excluding[key] = excluding.get(key, 0) + 1
        row["qualified_class_counts_excluding_initial"] = excluding
    paired_prefixes = []
    for seed in (11, 29):
        selected = [r for r in rows if r["arm"].endswith(f"seed{seed}")]
        if len(selected) != 3 or any(r["arm"] not in present for r in selected):
            continue
        cap = min(r["search_requests"] for r in selected)
        compared = []
        for row in selected:
            qualified = [x for x in row["graph_categories"]
                         if x["fresh_force_qualified"] is True
                         and x["cumulative_search_requests"] <= cap]
            compared.append(dict(arm=row["arm"], qualified_landings=len(qualified),
                classes=sorted({x["class_id"] for x in qualified}),
                noninitial_classes=sorted({x["class_id"] for x in qualified if x["index"] != 0}),
                best_delta=min(x["delta_from_initial"] for x in qualified),
                fragmented_landings=sum(x["fragmented"] for x in qualified)))
        paired_prefixes.append(dict(seed=seed, common_search_requests=cap, arms=compared))
    output = dict(protocol="GFN2-xTB developmental connectivity audit; not DFT GGA-PBE reproduction",
                  references=list(references), bond_cutoff="HC_BOND_LENGTHS + 0.1 A",
                  complete=(present == expected), missing_arms=sorted(expected - present),
                  incomplete_arms=incomplete,
                  global_graph_class_count=len(reps), paired_common_prefixes=paired_prefixes,
                  note="same graph class does not identify the same conformer/minimum; fragmented structures are reported, not auto-rejected",
                  arms=rows)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__": main()
