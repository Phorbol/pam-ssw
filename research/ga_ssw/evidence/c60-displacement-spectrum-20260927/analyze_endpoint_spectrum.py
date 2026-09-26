#!/usr/bin/env python3
"""Map qualified C60 endpoints and project their displacement on a saved Hessian.

This is a graph-mapped endpoint comparison, not a path, transition-state, or
barrier calculation. It performs no calculator evaluations.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from ase.build import molecule
from ase.io import read
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925"
OUT = Path(__file__).resolve().parent
CUTOFF = 1.64


def distance_graph(xyz: np.ndarray) -> nx.Graph:
    distances = cdist(xyz, xyz)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(xyz)))
    graph.add_edges_from((int(i), int(j)) for i, j in zip(*np.where(np.triu((distances < CUTOFF) & (distances > 0), 1))))
    return graph


def proper_kabsch(mobile: np.ndarray, reference: np.ndarray):
    """Return proper-rotation fit of row-vector mobile onto reference."""
    mob_center = mobile.mean(axis=0)
    ref_center = reference.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mob_center).T @ (reference - ref_center))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    fitted = (mobile - mob_center) @ rotation + ref_center
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - reference) ** 2, axis=1))))
    return fitted, rotation, rmsd


def edge_set(graph: nx.Graph, mapping: dict[int, int] | None = None) -> set[tuple[int, int]]:
    if mapping is None:
        return {tuple(sorted(edge)) for edge in graph.edges()}
    return {tuple(sorted((mapping[int(i)], mapping[int(j)]))) for i, j in graph.edges()}


def main() -> None:
    witness = json.loads((EVIDENCE / "witness.json").read_text())["one_switch_witness"]
    ase_to_iso2 = {int(k): int(v) for k, v in witness["candidate_to_target_node_mapping"].items()}
    if set(ase_to_iso2) != set(range(60)) or set(ase_to_iso2.values()) != set(range(60)):
        raise RuntimeError("witness mapping is not a bijection on the 60 nodes")

    ase_c60 = molecule("C60")
    ase_graph = distance_graph(ase_c60.positions)
    iso1 = read(EVIDENCE / "qualification/isomer-1/final.extxyz")
    iso2 = read(EVIDENCE / "qualification/isomer-2/final.extxyz")
    iso1_graph = distance_graph(iso1.positions)
    iso2_graph = distance_graph(iso2.positions)
    if not nx.is_isomorphic(ase_graph, iso1_graph):
        raise RuntimeError("qualified isomer-1 graph is not isomorphic to ASE C60 at cutoff")
    if nx.is_isomorphic(ase_graph, iso2_graph):
        raise RuntimeError("expected isomer-2 graph to differ from ASE C60")

    data = np.load(EVIDENCE / "curvature/isomer-2.npz")
    positions = data["positions"]
    basis = data["internal_basis"]
    eigenvalues = data["eigenvalues"]
    eigenvectors = data["eigenvectors"]
    hessian = data["raw_hessian"]
    if positions.shape != (60, 3) or basis.shape != (180, 174) or eigenvectors.shape != (174, 174):
        raise RuntimeError("unexpected saved Hessian dimensions")

    # Enumerate graph isomorphisms from ASE's C60 labels to isomer-1 labels.
    # This is bounded by the Ih graph's 120 automorphisms, then choose the
    # minimum proper-Kabsch displacement against the isomer-2 Hessian geometry.
    matcher = nx.isomorphism.GraphMatcher(ase_graph, iso1_graph)
    fits = []
    mapping_count = 0
    for ase_to_iso1 in matcher.isomorphisms_iter():
        mapping_count += 1
        iso1_xyz = iso1.positions[[ase_to_iso1[i] for i in range(60)]]
        iso2_xyz = positions[[ase_to_iso2[i] for i in range(60)]]
        fitted, rotation, rmsd = proper_kabsch(iso1_xyz, iso2_xyz)
        fits.append((rmsd, ase_to_iso1.copy(), rotation.copy(), fitted.copy()))
    if not fits:
        raise RuntimeError("no graph isomorphism found")
    fits.sort(key=lambda item: item[0])
    endpoint_rmsd, ase_to_iso1, rotation, fitted_iso1 = fits[0]
    tied_fits = [item for item in fits if item[0] <= endpoint_rmsd + 1e-12]

    # Compose ASE->isomer-2 (the saved witness) with ASE->isomer-1. Express
    # endpoint displacement in isomer-2 atom order and Hessian coordinates.
    displacement = np.empty((60, 3), dtype=float)
    iso2_to_iso1 = {}
    for ase_i in range(60):
        i2 = ase_to_iso2[ase_i]
        i1 = ase_to_iso1[ase_i]
        iso2_to_iso1[i2] = i1
        displacement[i2] = fitted_iso1[ase_i] - positions[i2]

    d = displacement.reshape(-1)
    internal = basis.T @ d
    retained_norm = float(np.linalg.norm(internal))
    endpoint_norm = float(np.linalg.norm(d))
    if retained_norm == 0 or not np.isfinite(retained_norm):
        raise RuntimeError("zero/non-finite internal endpoint displacement")
    unit_internal = internal / retained_norm
    coefficients = eigenvectors.T @ unit_internal
    modal_weights = coefficients**2
    rayleigh_eig = float(np.dot(eigenvalues, modal_weights))
    # Independent consistency check using the raw Cartesian Hessian.
    unit_cart = (basis @ unit_internal)
    rayleigh_raw = float(unit_cart @ hessian @ unit_cart)
    fractions = {str(m): float(np.sum(modal_weights[:m])) for m in (1, 5, 10, 20, 50, 174)}

    iso1_edges_as_iso2 = edge_set(iso1_graph, {i1: i2 for i2, i1 in iso2_to_iso1.items()})
    iso2_edges = edge_set(iso2_graph)
    removed = sorted(iso2_edges - iso1_edges_as_iso2)
    added = sorted(iso1_edges_as_iso2 - iso2_edges)
    expected_removed = sorted(tuple(sorted(ase_to_iso2[i] for i in x)) for x in witness["added_edges"])
    expected_added = sorted(tuple(sorted(ase_to_iso2[i] for i in x)) for x in witness["removed_edges"])
    if not np.allclose(iso2.positions, positions, rtol=0.0, atol=1e-10):
        raise RuntimeError("saved Hessian positions do not directly match qualified isomer-2 atom order")
    iso2_hessian_order_max_abs_A = float(np.max(np.abs(iso2.positions - positions)))
    if mapping_count != 120 or len(set(iso2_to_iso1.values())) != 60:
        raise RuntimeError("graph mapping enumeration/composition was incomplete")
    if not (np.isclose(rayleigh_eig, rayleigh_raw, rtol=1e-10, atol=1e-10)
            and np.isclose(modal_weights.sum(), 1.0, rtol=1e-10, atol=1e-10)):
        raise RuntimeError("Rayleigh or modal-weight consistency check failed")
    if not (len(removed) == 2 and len(added) == 2
            and removed == expected_removed and added == expected_added):
        raise RuntimeError("composed graph mapping does not recover the exact four witness edge changes")
    aligned_endpoint_iso2_order = positions + displacement
    expected_aligned_endpoint = np.empty_like(positions)
    for ase_i in range(60):
        expected_aligned_endpoint[ase_to_iso2[ase_i]] = fitted_iso1[ase_i]
    if not np.allclose(aligned_endpoint_iso2_order, expected_aligned_endpoint, rtol=0.0, atol=1e-10):
        raise RuntimeError("saved endpoint is not in isomer-2 atom order")
    aligned_endpoint_graph = distance_graph(aligned_endpoint_iso2_order)
    if edge_set(aligned_endpoint_graph) != iso1_edges_as_iso2:
        raise RuntimeError("saved aligned endpoint does not reproduce mapped isomer-1 edge set")
    edge_distance_checks = []
    for edge in sorted(removed + added):
        i, j = edge
        edge_distance_checks.append({
            "isomer2_edge_indices": [i, j],
            "isomer2_distance_A": float(np.linalg.norm(positions[i] - positions[j])),
            "aligned_isomer1_distance_A": float(np.linalg.norm(aligned_endpoint_iso2_order[i] - aligned_endpoint_iso2_order[j])),
            "isomer2_edge_present": iso2_graph.has_edge(i, j),
            "aligned_isomer1_edge_present": aligned_endpoint_graph.has_edge(i, j),
        })
    tied_projection_fractions = {}
    for _, tied_map, _, tied_fitted in tied_fits:
        tied_d = np.empty((60, 3), dtype=float)
        for ase_i in range(60):
            i2 = ase_to_iso2[ase_i]
            tied_d[i2] = tied_fitted[ase_i] - positions[i2]
        tied_coefficients = eigenvectors.T @ (basis.T @ tied_d.reshape(-1) / retained_norm)
        tied_weights = tied_coefficients**2
        tied_projection_fractions[str(tuple(tied_map[i] for i in range(60))[:4])] = {
            str(m): float(tied_weights[:m].sum()) for m in (1, 5, 10, 20, 50, 174)
        }
    endpoint_path = OUT / "aligned_isomer1_in_isomer2_atom_order.npy"
    np.save(endpoint_path, aligned_endpoint_iso2_order)
    saved_endpoint = np.load(endpoint_path)
    if (saved_endpoint.shape != (60, 3)
            or not np.allclose(saved_endpoint, positions + displacement, rtol=0.0, atol=1e-12)
            or edge_set(distance_graph(saved_endpoint)) != iso1_edges_as_iso2):
        raise RuntimeError("saved endpoint array failed atom-order, displacement, or edge-set verification")
    (OUT / "selected_mapping.json").write_text(json.dumps({
        "ASE_to_isomer1": {str(k): int(v) for k, v in sorted(ase_to_iso1.items())},
        "ASE_to_isomer2_from_witness": {str(k): int(v) for k, v in sorted(ase_to_iso2.items())},
        "isomer2_to_isomer1": {str(k): int(v) for k, v in sorted(iso2_to_iso1.items())},
        "minimum_proper_kabsch_tie_count_within_1e-12_A": len(tied_fits),
        "tied_mappings_have_projection_fractions": tied_projection_fractions,
        "saved_endpoint_array": "aligned_isomer1_in_isomer2_atom_order.npy; positions + displacement, row i matches isomer2/Hessian atom i; reloaded array checked against displacement and graph edges",
        "note": "Two graph mappings tie at minimum proper-Kabsch RMSD; reported projections agree to roundoff.",
    }, indent=2) + "\n")

    result = {
        "scope": "zero-PES endpoint displacement projection; not a path, TS, barrier, or softening-performance test",
        "inputs": {
            "witness": str(EVIDENCE / "witness.json"),
            "isomer1": str(EVIDENCE / "qualification/isomer-1/final.extxyz"),
            "isomer2": str(EVIDENCE / "qualification/isomer-2/final.extxyz"),
            "hessian": str(EVIDENCE / "curvature/isomer-2.npz"),
            "cutoff_A": CUTOFF,
        },
        "mapping": {
            "ASE_to_isomer1_graph_isomorphisms_enumerated": mapping_count,
            "minimum_proper_kabsch_endpoint_rmsd_A": endpoint_rmsd,
            "second_best_proper_kabsch_rmsd_A": fits[1][0] if len(fits) > 1 else None,
            "second_best_minus_best_rmsd_A": fits[1][0] - endpoint_rmsd if len(fits) > 1 else None,
            "minimum_rmsd_tie_count_within_1e-12_A": len(tied_fits),
            "rotation_determinant": float(np.linalg.det(rotation)),
            "ASE_to_isomer2_mapping_from_one_switch_witness": "composed as stored",
            "composition_is_bijective": len(set(iso2_to_iso1.values())) == 60,
            "isomer2_extxyz_hessian_positions_max_abs_difference_A": iso2_hessian_order_max_abs_A,
            "tied_minimum_mappings_projection_fractions": tied_projection_fractions,
            "isomer2_graph_isomorphic_to_ASE_C60": False,
            "isomer1_graph_isomorphic_to_ASE_C60": True,
        },
        "endpoint_displacement": {
            "cartesian_norm_A": endpoint_norm,
            "internal_projected_norm_A": retained_norm,
            "rigid_body_removed_norm_A": float(np.sqrt(max(0.0, endpoint_norm**2 - retained_norm**2))),
            "internal_retained_fraction_of_norm": retained_norm / endpoint_norm,
            "unit_internal_displacement_rayleigh_eV_A2": rayleigh_eig,
            "rayleigh_from_raw_cartesian_hessian_eV_A2": rayleigh_raw,
            "cumulative_projection_fraction": fractions,
            "uniform_internal_direction_expectation": {str(m): m / 174 for m in (1, 5, 10, 20, 50, 174)},
            "lowest_eigenvalues_eV_A2": eigenvalues[:10].tolist(),
            "projection_weights_sum": float(modal_weights.sum()),
        },
        "witness_edge_recovery": {
            "removed_edges_isomer2_to_isomer1": [list(e) for e in removed],
            "added_edges_isomer2_to_isomer1": [list(e) for e in added],
            "exactly_four_edge_changes": len(removed) == 2 and len(added) == 2,
            "matches_composed_one_switch_witness": removed == expected_removed and added == expected_added,
            "saved_endpoint_edge_set_matches_mapped_isomer1": edge_set(aligned_endpoint_graph) == iso1_edges_as_iso2,
            "changed_edge_distance_checks": edge_distance_checks,
            "witness_expected_removed_from_isomer2": [list(e) for e in expected_removed],
            "witness_expected_added_to_isomer1": [list(e) for e in expected_added],
        },
    }
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
