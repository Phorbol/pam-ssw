"""Offline C4H6 torsion audit for the archived six-arm LS coverage run."""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from ase.collections import g2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "ga_ssw"))
from analyze_c4h6_ls_reaction_coverage import atoms_from_dict, graph

SOURCE = ROOT / "research/ga_ssw/evidence/c4h6-ls-reaction-coverage-20260912"
ARMS = [f"butadiene-{method}-seed{seed}" for method in
        ("ssw", "paper_ls", "native_ls") for seed in (11, 29)]


def carbon_path(atoms):
    carbon = [i for i, number in enumerate(atoms.numbers) if int(number) == 6]
    g = graph(atoms).subgraph(carbon)
    paths = set()
    for a in carbon:
        for b in carbon:
            if a == b:
                continue
            for c in carbon:
                if c in (a, b):
                    continue
                for d in carbon:
                    if d in (a, b, c):
                        continue
                    seq = (a, b, c, d)
                    if all(g.has_edge(seq[k], seq[k + 1]) for k in range(3)) and not any(
                            g.has_edge(seq[i], seq[j]) for i, j in ((0, 2), (0, 3), (1, 3))):
                        paths.add(min(seq, seq[::-1]))
    if len(paths) != 1:
        raise ValueError(f"expected one induced four-carbon path, found {sorted(paths)}")
    return list(paths)[0]


def torsion(atoms):
    path = carbon_path(atoms)
    angle = float(atoms.get_dihedral(*path))
    return path, angle, float(np.cos(np.deg2rad(angle)))


def reference_diagnostic():
    trans = g2["butadiene"].copy()
    path, trans_angle, trans_cos = torsion(trans)
    cut = graph(trans)
    cut.remove_edge(path[1], path[2])
    branch = nx.node_connected_component(cut, path[2])
    cis = trans.copy()
    cis.set_dihedral(*path, 0.0, indices=sorted(branch))
    cis_angle, cis_cos = torsion(cis)[1:]
    same_graph = nx.is_isomorphic(graph(trans), graph(cis),
        node_match=nx.algorithms.isomorphism.categorical_node_match("number", None))
    reversed_atoms = trans[::-1]
    reversed_cos = torsion(reversed_atoms)[2]
    checks = {"trans_reference_graph_matches_cis_constructed": same_graph,
              "cis_trans_cos_signs_opposite": trans_cos * cis_cos < 0.0,
              "reverse_order_cos_invariant_1e-10": abs(trans_cos - reversed_cos) <= 1e-10}
    if not all(checks.values()):
        raise AssertionError(f"reference geometry diagnostic failed: {checks}")
    return {"path_indices": list(path), "trans_raw_deg": trans_angle,
            "trans_cosphi": trans_cos, "constructed_cis_raw_deg": cis_angle,
            "constructed_cis_cosphi": cis_cos, "old_graph_isomorphic": same_graph,
            "reverse_order_cosphi": reversed_cos, "checks": checks,
            "interpretation": "constructed geometry diagnostic only; not a physical minimum"}


def run():
    if (HERE / "analysis.json").exists() or (HERE / "report.md").exists():
        raise FileExistsError("refusing to overwrite analysis.json or report.md")
    diagnostic = reference_diagnostic()
    reference = graph(g2["butadiene"].copy())
    node_match = nx.algorithms.isomorphism.categorical_node_match("number", None)
    arms, boundary = [], []
    for arm in ARMS:
        folder = SOURCE / arm
        result_path, fresh_path = folder / "result.json", folder / "fresh-checks.json"
        if not result_path.is_file() or not fresh_path.is_file():
            boundary.append({"arm": arm, "kind": "missing_input",
                             "result_exists": result_path.is_file(),
                             "fresh_exists": fresh_path.is_file()})
            continue
        result, fresh = json.loads(result_path.read_text()), json.loads(fresh_path.read_text())
        checks = {}
        for record in fresh:
            if "index" in record:
                index = int(record["index"])
                if index in checks:
                    boundary.append({"arm": arm, "kind": "duplicate_fresh_index", "index": index})
                checks[index] = record
        minima = result.get("minima", [])
        rows, angles = [], []
        for index, minimum in enumerate(minima):
            fresh_row = checks.get(index)
            atoms = atoms_from_dict(minimum["atoms"])
            matches = nx.is_isomorphic(graph(atoms), reference, node_match=node_match)
            row = {"index": index, "converged": minimum.get("converged"),
                   "fresh_present": fresh_row is not None,
                   "fresh_force_qualified": None if fresh_row is None else fresh_row.get("force_qualified"),
                   "fresh_record": fresh_row, "matches_butadiene_element_graph": matches}
            if matches:
                path, raw, cosine = torsion(atoms)
                row.update(carbon_path_indices=list(path), raw_deg=raw, cosphi=cosine)
                angles.append({"index": index, "raw_deg": raw, "cosphi": cosine,
                               "converged": minimum.get("converged"),
                               "fresh_force_qualified": None if fresh_row is None else fresh_row.get("force_qualified")})
            rows.append(row)
        for index in sorted(set(checks) - set(range(len(minima)))):
            boundary.append({"arm": arm, "kind": "fresh_index_out_of_range", "index": index})
        for index in sorted(set(range(len(minima))) - set(checks)):
            boundary.append({"arm": arm, "kind": "fresh_index_missing", "index": index})
        arms.append({"arm": arm, "result_status": result.get("status"),
                     "minima_count": len(minima), "selected_frame_count": len(angles),
                     "raw_angles_deg": [x["raw_deg"] for x in angles],
                     "selected_frames": angles,
                     "cis_like_count": sum(x["cosphi"] > 0 for x in angles),
                     "trans_like_count": sum(x["cosphi"] < 0 for x in angles),
                     "zero_cosphi_count": sum(x["cosphi"] == 0 for x in angles),
                     "frames": rows})
    output = {"source": str(SOURCE.relative_to(ROOT)),
              "selection": "element-labeled full graph isomorphic to ASE G2 butadiene",
              "torsion_sign": "cosphi sign is geometric cis-like/trans-like only; no angle threshold or stable-isomer claim",
              "counts_are": "selected frames, not unique states or discovery rates",
              "reference_diagnostic": diagnostic,
              "boundary_anomalies": boundary, "arms": arms}
    (HERE / "analysis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    lines = ["# C4H6 torsion audit", "",
             "The old element-labeled graph merges cis/trans butadiene geometries. This audit selects frames graph-isomorphic to ASE G2 butadiene and reports the CCCC dihedral and cos(phi). The sign describes geometry only; it does not certify a stable isomer. Counts are frames, not unique states or discovery rates.", "",
             "| Arm | Selected frames | cis-like (cos(phi)>0) | trans-like (cos(phi)<0) | Raw angles (deg) |",
             "|---|---:|---:|---:|---|"]
    for row in arms:
        lines.append(f"| {row['arm']} | {row['selected_frame_count']} | {row['cis_like_count']} | {row['trans_like_count']} | {row['raw_angles_deg']} |")
    lines += ["", f"Reference geometry checks: `{diagnostic['checks']}`.",
              "The cis geometry was constructed by setting the reference dihedral to 0 degrees after cutting the central C-C bond for the ASE mask; it is a geometry diagnostic, not a physical minimum.",
              f"Boundary anomalies retained: {len(boundary)}. Per-frame indices, convergence fields, fresh-check records, and graph-selection decisions are in `analysis.json`."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    if boundary or len(arms) != len(ARMS):
        raise SystemExit(f"audit outputs written with boundary anomalies: {len(boundary)}; "
                         f"complete arms: {len(arms)}/{len(ARMS)}")


if __name__ == "__main__":
    run()
