"""Inspect first-discovery C4H6 topology representatives from an offline audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import networkx as nx

DEFAULT_AUDIT = Path(__file__).with_name("topology-audit.json")
CUTOFFS = {
    (1, 1): 0.8400000095367432,
    (1, 6): 1.190000033378601,
    (6, 6): 1.6399999618530273,
}
NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("number", None)


def make_graph(atom_data):
    numbers, positions = atom_data["numbers"], atom_data["positions"]
    if len(numbers) != len(positions):
        raise ValueError("atom number/position length mismatch")
    graph = nx.Graph()
    graph.add_nodes_from((i, {"number": int(number)}) for i, number in enumerate(numbers))
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            pair = tuple(sorted((int(numbers[i]), int(numbers[j]))))
            cutoff = CUTOFFS[pair]
            d2 = sum((float(positions[i][k]) - float(positions[j][k])) ** 2 for k in range(3))
            if d2 <= cutoff * cutoff:
                graph.add_edge(i, j)
    return graph


def species_preserving_permutation(numbers):
    """Return a deterministic, nonidentity permutation that never swaps species."""
    groups = {}
    for index, number in enumerate(numbers):
        groups.setdefault(int(number), []).append(index)
    permuted = list(range(len(numbers)))
    for indices in groups.values():
        if len(indices) > 1:
            rotated = indices[1:] + indices[:1]
            for old, new in zip(indices, rotated):
                permuted[new] = old
    return permuted


def permuted_atom_data(atom_data):
    permutation = species_preserving_permutation(atom_data["numbers"])
    return ({"numbers": [atom_data["numbers"][i] for i in permutation],
             "positions": [atom_data["positions"][i] for i in permutation]}, permutation)


def distance(a, b):
    return math.sqrt(sum((float(a[k]) - float(b[k])) ** 2 for k in range(3)))


def element(number):
    return {1: "H", 6: "C"}.get(int(number), f"Z{int(number)}")


def inspect_representative(arm_row, discovery, data):
    record_index = int(discovery["first_record"])
    record = data["records"][record_index]
    landing = record.get("landing")
    if not isinstance(landing, dict) or not isinstance(landing.get("atoms"), dict):
        raise ValueError(f"missing landing geometry for {arm_row['arm']} seed {arm_row['seed']} record {record_index}")
    atoms = landing["atoms"]
    numbers = [int(n) for n in atoms["numbers"]]
    positions = atoms["positions"]
    graph = make_graph(atoms)
    reordered, permutation = permuted_atom_data(atoms)
    permuted_graph = make_graph(reordered)
    reorder_check = nx.is_isomorphic(graph, permuted_graph, node_match=NODE_MATCH)

    degrees = {element(n): [] for n in sorted(set(numbers))}
    for i, number in enumerate(numbers):
        degrees[element(number)].append(int(graph.degree[i]))
    edge_list = [
        {"atoms": [i, j], "elements": [element(numbers[i]), element(numbers[j])]}
        for i, j in sorted(graph.edges)
    ]
    distances = [distance(positions[i], positions[j])
                 for i in range(len(numbers)) for j in range(i + 1, len(numbers))]
    initial_energy = data["initial"].get("energy")
    energy = landing.get("energy")
    delta = None if initial_energy is None or energy is None else float(energy) - float(initial_energy)
    anomaly_flags = []
    for index, number in enumerate(numbers):
        degree = graph.degree[index]
        if number == 1 and degree != 1:
            anomaly_flags.append({"atom": index, "element": "H", "degree": degree, "rule": "H_degree_not_1"})
        if number == 6 and degree > 4:
            anomaly_flags.append({"atom": index, "element": "C", "degree": degree, "rule": "C_degree_above_4"})

    return {
        "arm": arm_row["arm"], "seed": arm_row["seed"],
        "local_class_id": discovery["class_id"], "first_record": record_index,
        "cumulative_EF_requests": discovery["cumulative_EF_requests"],
        "accepted_by_original_MC": record.get("accepted"),
        "energy_eV": energy, "initial_energy_eV": initial_energy,
        "delta_energy_from_initial_eV": delta,
        "saved_max_force_eV_A": landing.get("max_force"),
        "landing_converged": landing.get("converged"), "landing_surface": landing.get("surface"),
        "atom_count": len(numbers),
        "element_counts": {element(n): numbers.count(n) for n in sorted(set(numbers))},
        "bond_graph_edge_count": graph.number_of_edges(), "edges_by_atom_index": edge_list,
        "degree_by_element": degrees,
        "min_pair_distance_A": min(distances) if distances else None,
        "max_pair_distance_A": max(distances) if distances else None,
        "degree_anomaly_flags_for_review_only": anomaly_flags,
        "reindex_consistency": {
            "permutation_new_index_to_old_index": permutation,
            "species_preserving": all(numbers[new] == int(numbers[permutation[new]])
                                       for new in range(len(numbers))),
            "graph_isomorphic_after_reindex": reorder_check,
        },
        "source_record_status": record.get("status"),
    }, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT,
                        help=f"topology audit JSON (default: {DEFAULT_AUDIT})")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON path; existing files are refused")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text())
    representatives = []
    source_hashes = {}
    for arm_row in audit["arms"]:
        source_path = Path(arm_row["source"])
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        source_hashes[str(source_path)] = source_hash
        if arm_row.get("source_sha256") != source_hash:
            raise ValueError(f"source SHA256 differs from topology audit: {source_path}")
        data = json.loads(source_path.read_text())
        for discovery in arm_row.get("all_candidate_noninitial_first_discoveries", []):
            inspected, graph = inspect_representative(arm_row, discovery, data)
            if inspected["reindex_consistency"]["species_preserving"] is not True:
                raise ValueError("reindexing unexpectedly changed species")
            # Shared classes are computed globally by exact labeled-graph isomorphism,
            # independently of the run-local class IDs in topology-audit.json.
            global_id = next((i for i, item in enumerate(representatives)
                              if nx.is_isomorphic(graph, item["graph"], node_match=NODE_MATCH)), None)
            if global_id is None:
                global_id = len(representatives)
                representatives.append({"graph": graph, "members": []})
            representatives[global_id]["members"].append({
                "arm": arm_row["arm"], "seed": arm_row["seed"],
                "local_class_id": discovery["class_id"], "first_record": discovery["first_record"],
            })
            inspected["cross_arm_permutation_invariant_class_id"] = global_id
            representatives[global_id]["inspections"] = representatives[global_id].get("inspections", [])
            representatives[global_id]["inspections"].append(inspected)

    for group in representatives:
        for inspected in group["inspections"]:
            inspected["cross_arm_members_sharing_class"] = group["members"]
            inspected["cross_arm_class_member_count"] = len(group["members"])

    output = {
        "status": "offline_geometry_inspection_only",
        "pes_evaluations": 0,
        "audit_source": str(audit_path),
        "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "trajectory_source_hashes": source_hashes,
        "graph_definition": audit.get("definitions", {}).get("graph"),
        "reindex_consistency_checks_passed": all(
            item["reindex_consistency"]["graph_isomorphic_after_reindex"]
            for group in representatives for item in group.get("inspections", [])
        ),
        "global_isomorphism_classes": [
            {"class_id": i, "members": group["members"], "member_count": len(group["members"]),
             "representative_inspections": group.get("inspections", [])}
            for i, group in enumerate(representatives)
        ],
        "interpretation_limits": [
            "Degree flags identify structures for review only; they do not silently exclude a graph or prove it is unphysical.",
            "The distance graph has no bond order, valence model, geometry optimization, or independent force validation.",
            "Energy and saved force are archived landing diagnostics, not certificates of a minimum or chemical stability.",
            "Cross-arm classes denote shared element-labeled graph topology, not shared geometrical minima.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
