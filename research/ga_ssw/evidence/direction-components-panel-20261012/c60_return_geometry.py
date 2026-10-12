#!/usr/bin/env python3
"""Zero-PES comparison of saved C60 initial and landing geometries."""
import json
from pathlib import Path

import networkx as nx
import numpy as np
from ase.io import read


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "runs-1756515" / "c60"
CUTOFFS = (1.64, 1.70, 1.80)


def graph(atoms, cutoff):
    distances = atoms.get_all_distances(mic=False)
    result = nx.Graph()
    result.add_nodes_from(range(len(atoms)))
    result.add_edges_from(zip(*np.where(np.triu(
        (distances < cutoff) & (distances > 0), 1))))
    return result


def main():
    rows = []
    for path in sorted(RUN_ROOT.glob("*/seed-*/result.json")):
        result = json.loads(path.read_text())
        trajectory = path.with_name("structures.extxyz")
        frames = read(trajectory, ":")
        initial = next(atoms for atoms in frames if atoms.info["role"] == "initial")
        landing = next(atoms for atoms in frames if atoms.info["role"] == "landing")
        x = initial.positions - initial.positions.mean(axis=0)
        y = landing.positions - landing.positions.mean(axis=0)
        u, _, vt = np.linalg.svd(x.T @ y)
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        rms = float(np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1))))
        graphs = []
        for cutoff in CUTOFFS:
            before, after = graph(initial, cutoff), graph(landing, cutoff)
            graphs.append({
                "cutoff_A": cutoff,
                "landing_connected": nx.is_connected(after),
                "landing_edges": after.number_of_edges(),
                "same_index_edges": set(before.edges()) == set(after.edges()),
                "isomorphic_to_initial": nx.is_isomorphic(before, after),
            })
        rows.append({
            "method": result["method"], "seed": result["seed"],
            "result_path": str(path.relative_to(HERE)),
            "trajectory_path": str(trajectory.relative_to(HERE)),
            "initial_frame_role": "initial", "landing_frame_role": "landing",
            "search_requests": result["search_requests"],
            "fresh_landing_minus_initial_energy_eV":
                result["fresh"]["energy_eV"] - result["initial"]["energy_eV"],
            "same_index_proper_rotation_rms_A": rms,
            "graphs": graphs,
        })
    output = {
        "zero_pes_analysis": True,
        "input_run_root": str(RUN_ROOT.relative_to(HERE)),
        "observed_pairs": len(rows),
        "method": "centered, same-atom-index least-squares proper rotation; no permutation in RMS",
        "graph_rule": "nonperiodic pair distance strictly below each existing cutoff and greater than zero",
        "new_classification_threshold": None,
        "limitations": [
            "Small RMS and identical edge sets do not prove exact basin identity.",
            "Different graphs and higher energies do not prove productive escape.",
            "Coordinates are read from the saved extxyz frames, including their serialization precision.",
        ],
        "rows": rows,
    }
    rendered = json.dumps(output, indent=2, allow_nan=False) + "\n"
    (HERE / "c60_return_geometry.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
