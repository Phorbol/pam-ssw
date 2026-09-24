#!/usr/bin/env python3
"""Audit C60 source geometries as cutoff graphs and test a one-switch relation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
from ase.build import molecule
from ase.io import read


MEMBERS = {1: "c60/c60-iso-1_opt.xyz", 2: "c60/c60-iso-2_opt.xyz"}
CUTOFFS = (1.64, 1.70, 1.80)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def distance_graph(atoms, cutoff: float) -> nx.Graph:
    import numpy as np

    xyz = atoms.positions
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    g = nx.Graph()
    g.add_nodes_from(range(len(atoms)))
    g.add_edges_from((int(i), int(j)) for i, j in zip(*np.where(np.triu((d < cutoff) & (d > 0), 1))))
    return g


def graph_summary(g: nx.Graph) -> dict:
    planar, embedding = nx.check_planarity(g)
    faces = []
    if planar:
        seen = set()
        for u, v in embedding.edges():
            if (u, v) not in seen:
                faces.append(embedding.traverse_face(u, v, seen))
    degrees = Counter(dict(g.degree()).values())
    connected = nx.is_connected(g)
    face_counts = Counter(map(len, faces))
    structural_candidate = bool(
        len(g) == 60 and g.number_of_edges() == 90 and degrees == Counter({3: 60})
        and connected and planar and face_counts == Counter({5: 12, 6: 20})
    )
    three_connected = bool(structural_candidate and nx.node_connectivity(g) >= 3)
    fullerene = three_connected
    return {
        "nodes": g.number_of_nodes(), "edges": g.number_of_edges(),
        "components": nx.number_connected_components(g),
        "degree_counts": dict(sorted(degrees.items())), "three_connected": three_connected,
        "planar": planar, "face_counts": dict(sorted(face_counts.items())),
        "fullerene_cage": fullerene,
    }


def one_switch_witnesses(reference: nx.Graph, target: nx.Graph) -> list[dict]:
    """Enumerate AB retained; AC and BE removed; AE and BC added."""
    witnesses = []
    for a, b in sorted(tuple(sorted(e)) for e in reference.edges()):
        a_neighbors = sorted(set(reference.neighbors(a)) - {b})
        b_neighbors = sorted(set(reference.neighbors(b)) - {a})
        for c in a_neighbors:
            for e in b_neighbors:
                if len({a, b, c, e}) < 4:
                    continue
                if reference.has_edge(a, e) or reference.has_edge(b, c):
                    continue
                candidate = reference.copy()
                candidate.remove_edges_from(((a, c), (b, e)))
                candidate.add_edges_from(((a, e), (b, c)))
                summary = graph_summary(candidate)
                matcher = nx.isomorphism.GraphMatcher(candidate, target)
                if summary["fullerene_cage"] and matcher.is_isomorphic():
                    mapping = matcher.mapping
                    witnesses.append({
                        "central_edge_retained": [a, b],
                        "removed_edges": [[a, c], [b, e]],
                        "added_edges": [[a, e], [b, c]],
                        "candidate_result_edges": [list(e) for e in sorted(tuple(sorted(x)) for x in candidate.edges())],
                        "candidate_to_target_node_mapping": {str(k): v for k, v in sorted(mapping.items())},
                        "result_graph": summary,
                    })
    return witnesses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    args = ap.parse_args()
    source = args.source_dir.resolve()
    zip_path = source / "41524_2024_1410_MOESM3_ESM.zip"
    csv_path = source / "41524_2024_1410_MOESM2_ESM.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("Cn", "").strip().lower() == "c60" and row.get("#iso", "").strip() in {"1", "2"}:
                rows[int(row["#iso"].strip())] = row
    if set(rows) != {1, 2}:
        raise RuntimeError(f"CSV rows for author C60 IDs 1 and 2 not uniquely found: {sorted(rows)}")

    spec = importlib.util.spec_from_file_location(
        "analyze_c60_random_development",
        args.repo_root / "research/ga_ssw/analyze_c60_random_development.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load existing graph_row helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    atoms = {}
    member_info = {}
    with zipfile.ZipFile(zip_path) as archive:
        archive_names = set(archive.namelist())
        for iso, member in MEMBERS.items():
            if member not in archive_names:
                raise RuntimeError(f"Required source member missing: {member}")
            info = archive.getinfo(member)
            member_info[str(iso)] = {"name": member, "uncompressed_bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
            source_xyz = archive.read(member).decode("utf-8").rstrip() + "\n"
            atoms[iso] = read(io.StringIO(source_xyz), format="xyz")
            if not np.isfinite(atoms[iso].positions).all():
                raise RuntimeError(f"Non-finite coordinate in source member: {member}")

    ih = molecule("C60")
    ih_graphs = {cutoff: distance_graph(ih, cutoff) for cutoff in CUTOFFS}
    target_graphs = {iso: {cutoff: distance_graph(atoms[iso], cutoff) for cutoff in CUTOFFS} for iso in (1, 2)}
    analyses = {}
    for iso in (1, 2):
        analyses[str(iso)] = {
            "source_member": MEMBERS[iso],
            "source_csv_row": rows[iso],
            "atoms": len(atoms[iso]),
            "atomic_numbers": sorted(Counter(map(int, atoms[iso].numbers)).items()),
            "cutoff_graphs": {
                str(cutoff): {
                    "graph_row": helper.graph_row(atoms[iso].numbers, atoms[iso].positions, cutoff, ih_graphs[cutoff]),
                    "fullerene_topology": graph_summary(target_graphs[iso][cutoff]),
                    "isomorphic_to_ASE_Ih_graph": nx.is_isomorphic(target_graphs[iso][cutoff], ih_graphs[cutoff]),
                }
                for cutoff in CUTOFFS
            },
        }

    switch_search = {}
    switch_cache = {}
    for cutoff in CUTOFFS:
        key = (
            frozenset(tuple(sorted(e)) for e in ih_graphs[cutoff].edges()),
            frozenset(tuple(sorted(e)) for e in target_graphs[2][cutoff].edges()),
        )
        signature_reused = key in switch_cache
        if key not in switch_cache:
            switch_cache[key] = one_switch_witnesses(ih_graphs[cutoff], target_graphs[2][cutoff])
        witnesses = switch_cache[key]
        switch_search[str(cutoff)] = {
            "edge_set_signature_reused": signature_reused,
            "candidate_Ih_graph": graph_summary(ih_graphs[cutoff]),
            "target_author_iso_2_graph": graph_summary(target_graphs[2][cutoff]),
            "target_isomorphic_to_Ih": nx.is_isomorphic(ih_graphs[cutoff], target_graphs[2][cutoff]),
            "one_switch_witness_count": len(witnesses),
            "first_witness": witnesses[0] if witnesses else None,
        }

    result = {
        "scope": "source-geometry graph qualification only; no energy evaluation and no physical path inference",
        "source": {
            "dataset_doi": "10.1038/s41524-024-01410-7",
            "zip": str(zip_path), "zip_sha256": sha256(zip_path),
            "csv": str(csv_path), "csv_sha256": sha256(csv_path),
            "source_members": list(MEMBERS.values()),
            "source_member_zip_metadata": member_info,
            "identifier_warning": "IDs 1 and 2 are the author's #iso labels in this supplied dataset; no Atlas-number equivalence is asserted.",
        },
        "ASE_reference": {
            "constructor": "ase.build.molecule('C60')",
            "atoms": len(ih),
            "cutoff_graphs": {
                str(c): {
                    "topology": graph_summary(ih_graphs[c]),
                    "graph_row": helper.graph_row(ih.numbers, ih.positions, c, ih_graphs[c]),
                }
                for c in CUTOFFS
            },
        },
        "structures": analyses,
        "one_edge_switch_search": {
            "definition": "For each central edge A-B, retain A-B; remove A-C and B-E; add A-E and B-C, with four distinct vertices and both added edges absent before the switch.",
            "matches": switch_search,
        },
    }
    (args.output_dir / "graph_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    witness = {
        "dataset_doi": result["source"]["dataset_doi"],
        "author_dataset_id": 2,
        "not_atlas_id": True,
        "cutoff_A": None,
        "one_switch_witness": None,
    }
    found = [(float(c), data["first_witness"]) for c, data in switch_search.items() if data["first_witness"] is not None]
    if found:
        witness["cutoff_A"] = found[0][0]
        witness["one_switch_witness"] = found[0][1]
    (args.output_dir / "witness.json").write_text(json.dumps(witness, indent=2) + "\n")


if __name__ == "__main__":
    main()
