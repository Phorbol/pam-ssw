#!/usr/bin/env python3
"""Zero-PES audit of C4H4 + H2 component-labelled minima in a saved run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/"
    "research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/analysis.json"
)
PREFIX_REQUESTS = 200_000
TARGET_COMPONENTS = ["C4H4", "H2"]


def analyze(source: Path) -> dict:
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    data = json.loads(source.read_text())
    # Use the already-frozen campaign graph builder and its H/C distance
    # cutoffs; this remains an offline graph check, with no calculator access.
    source_root = source.parents[4]
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "research" / "ga_ssw"))
    from analyze_c4h6_ls_reaction_coverage import atoms_from_dict, graph
    import networkx as nx

    target_g = nx.Graph()
    target_g.add_nodes_from((i, {"number": 6}) for i in range(4))
    target_g.add_nodes_from((i, {"number": 1}) for i in range(4, 10))
    target_g.add_edges_from(((0, 1), (1, 2), (2, 3)))
    target_g.add_edges_from(((0, 4), (0, 5), (1, 6), (3, 7)))
    target_g.add_edge(8, 9)  # separate H2

    target_h = nx.Graph()
    target_h.add_nodes_from((i, {"number": 6}) for i in range(4))
    target_h.add_nodes_from((i, {"number": 1}) for i in range(4, 10))
    target_h.add_edges_from(((0, 1), (1, 2), (2, 0), (0, 3)))
    target_h.add_edges_from(((1, 4), (2, 5), (3, 6), (3, 7)))
    target_h.add_edge(8, 9)  # separate H2
    node_match = nx.algorithms.isomorphism.categorical_node_match("number", None)

    arms = []
    for arm in data["arms"]:
        prefix_rows = [
            m for m in arm["minima"]
            if m.get("cumulative_requests") is not None
            and m["cumulative_requests"] <= PREFIX_REQUESTS
            and sorted(m.get("component_formulas", [])) == TARGET_COMPONENTS
        ]
        qualified = [
            m for m in prefix_rows
            if m.get("converged") is True
            and m.get("fresh", {}).get("identity_matches") is True
            and m.get("fresh", {}).get("record", {}).get("numerical_qualified") is True
        ]
        classes: dict[int, list[dict]] = defaultdict(list)
        for m in qualified:
            classes[m["graph_class_id"]].append(m)
        class_rows = []
        for class_id, observations in sorted(classes.items(), key=lambda item: min(x["cumulative_requests"] for x in item[1])):
            first = min(observations, key=lambda x: x["cumulative_requests"])
            class_rows.append({
                "graph_class_id": class_id,
                "first_request": first["cumulative_requests"],
                "first_minimum_index": first["index"],
                "force_qualified_observations": len(observations),
                "first_energy_eV": first["energy_eV"],
                "first_fresh_fmax_eV_A": first["fresh"]["record"]["fmax_eV_A"],
            })
        # Coordinates are embedded in the archived per-arm result.json rather
        # than standalone trajectory files. Load one arm at a time and inspect
        # only the preselected formula-qualified prefix minima.
        raw_result_path = source.parent / arm["label"] / "result.json"
        raw_result = json.loads(raw_result_path.read_text()) if qualified else None
        target_matches = []
        for m in qualified:
            raw_min = raw_result["minima"][m["index"]]
            atoms = atoms_from_dict(raw_min["atoms"])
            observed_graph = graph(atoms)
            matches_g = nx.is_isomorphic(observed_graph, target_g, node_match=node_match)
            matches_h = nx.is_isomorphic(observed_graph, target_h, node_match=node_match)
            if matches_g or matches_h:
                target_matches.append({
                    "minimum_index": m["index"],
                    "request": m["cumulative_requests"],
                    "paper_connectivity_candidate": "G" if matches_g else "H",
                    "graph_class_id": m["graph_class_id"],
                    "fresh_fmax_eV_A": m["fresh"]["record"]["fmax_eV_A"],
                })
        del raw_result
        arms.append({
            "arm": arm["arm"],
            "seed": arm["seed"],
            "label": arm["label"],
            "prefix_requests": PREFIX_REQUESTS,
            "component_formula_observations": len(prefix_rows),
            "force_qualified_observations": len(qualified),
            "force_qualified_graph_classes": len(class_rows),
            "classes": class_rows,
            "paper_connectivity_candidates": target_matches,
        })
    return {
        "source": str(source),
        "source_sha256": digest.hexdigest(),
        "method": "component_formulas exactly C4H4 and H2; cumulative request <= common prefix; converged and matched fresh numerical qualification",
        "common_prefix_requests": PREFIX_REQUESTS,
        "limits": [
            "Component formulas and connectivity are graph-class labels, not bond-order, spin, radical, or exact molecular-identity assignments.",
            "Fresh force qualification is numerical qualification only, not Hessian stability or a PBE paper-oracle match.",
            "G/H comparisons are only exact element-labeled connectivity matches under the saved campaign H/C distance cutoff; they do not validate bond order, electronic state, reaction pathway, barrier, or DFT identity.",
        ],
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
