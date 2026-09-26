"""Zero-PES exact topology audit for archived C4H6 SSW/NativeLS trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import networkx as nx

EVIDENCE = Path(
    "/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence"
)
RUNS = [(arm, seed) for arm in ("ssw", "native_ls") for seed in (61, 67)]
CUTOFFS = {
    (1, 1): 0.8400000095367432,
    (1, 6): 1.190000033378601,
    (6, 6): 1.6399999618530273,
}
FORCE_LIMIT = 0.03
NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("number", None)


def make_graph(atom_data):
    numbers, positions = atom_data["numbers"], atom_data["positions"]
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


def assign_class(graph, representatives):
    for class_id, representative in enumerate(representatives):
        if nx.is_isomorphic(graph, representative, node_match=NODE_MATCH):
            return class_id, False
    representatives.append(graph)
    return len(representatives) - 1, True


def analyze_run(arm, seed):
    path = EVIDENCE / "c4h6-mh1-coverage-20260924" / f"{arm}-seed{seed}" / "result.json"
    data = json.loads(path.read_text())
    records = data.get("records", [])
    if len(records) != 400:
        raise ValueError(f"expected 400 attempts, got {len(records)}: {path}")
    initial = data["initial"]
    initial_graph = make_graph(initial["atoms"])
    all_reps, accepted_reps = [initial_graph], [initial_graph]
    cumulative = int(initial.get("evaluation_requests", 0))
    all_new, accepted_new = [], []
    landing_count = qualified_count = accepted_qualified_count = 0

    for index, record in enumerate(records):
        if record.get("index") != index:
            raise ValueError(f"record index mismatch at {index}: {path}")
        cumulative += int(record.get("evaluation_requests", 0) or 0)
        landing = record.get("landing")
        if not isinstance(landing, dict) or not isinstance(landing.get("atoms"), dict):
            continue
        landing_count += 1
        force = landing.get("max_force")
        force_ok = force is not None and math.isfinite(float(force)) and float(force) <= FORCE_LIMIT
        converged_true_surface = landing.get("converged") is True and landing.get("surface") == "true"
        if not (force_ok and converged_true_surface):
            continue
        graph = make_graph(landing["atoms"])
        if not nx.is_connected(graph):
            continue
        qualified_count += 1
        class_id, is_new = assign_class(graph, all_reps)
        if is_new:
            all_new.append({"class_id": class_id, "first_record": index,
                            "cumulative_EF_requests": cumulative})
        if record.get("accepted") is True:
            accepted_qualified_count += 1
            accepted_class_id, accepted_is_new = assign_class(graph, accepted_reps)
            if accepted_is_new:
                accepted_new.append({"class_id": accepted_class_id, "first_record": index,
                                     "cumulative_EF_requests": cumulative})

    if cumulative != int(data.get("evaluation_requests", cumulative)):
        raise ValueError(f"initial + record E/F requests disagree with result total: {path}")
    initial_class, _ = assign_class(initial_graph, [])
    return {
        "arm": arm, "seed": seed, "source": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attempts": len(records), "landing_records": landing_count,
        "qualified_connected_candidates": qualified_count,
        "candidate_events_not_discovering_a_new_noninitial_class": qualified_count - len(all_new),
        "accepted_qualified_connected_candidates": accepted_qualified_count,
        "accepted_events_not_discovering_a_new_noninitial_class": accepted_qualified_count - len(accepted_new),
        "all_candidate_topology_classes_including_initial": len(all_reps),
        "all_candidate_noninitial_topology_classes": len(all_new),
        "all_candidate_noninitial_first_discoveries": all_new,
        "accepted_topology_classes_including_initial": len(accepted_reps),
        "accepted_noninitial_topology_classes": len(accepted_new),
        "accepted_noninitial_first_discoveries": accepted_new,
        "initial_graph_class_id_within_run": initial_class,
        "total_EF_requests_including_initial": cumulative,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new JSON path; omitted means stdout")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    rows = [analyze_run(arm, seed) for arm, seed in RUNS]
    out = {
        "status": "offline_archived_data_only",
        "pes_evaluations": 0,
        "question": "Are the previously counted connected, force-qualified landing events mostly repeats of existing element-labeled graph topologies, or distinct topology classes?",
        "definitions": {
            "candidate": "stored landing with converged=true and surface=true, saved max_force <= 0.03 eV/A, and one connected component under the frozen graph cutoff",
            "graph": "nonperiodic direct Cartesian distances; H-H 0.8400000095367432 A, H-C 1.190000033378601 A, C-C 1.6399999618530273 A; nodes labeled by atomic number",
            "identity": "exact NetworkX graph isomorphism with categorical atomic-number node matching; class IDs are local to each arm and seed",
            "cost": "initial plus cumulative per-record archived E/F evaluation_requests at first qualified discovery; LS preparation requests are already included in record totals",
            "accepted_subset": "same candidate qualification, additionally record.accepted is true; classes/first-discovery costs are recomputed within this subset",
        },
        "arms": rows,
        "limits": [
            "This is exact topology identity only; distinct geometries with the same bond graph are merged, and same topology does not establish a common geometric minimum.",
            "The filter uses archived landing convergence and saved force only; no independent fresh evaluation is performed.",
            "Counts describe these four archived C4H6 trajectories only and do not establish reaction paths, barriers, or general LS efficacy.",
        ],
    }
    rendered = json.dumps(out, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(rendered)


if __name__ == "__main__":
    main()
