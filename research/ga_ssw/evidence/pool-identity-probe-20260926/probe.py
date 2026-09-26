"""Zero-PES identity stress test for MinimaArchive on archived C4H6 geometries."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from pamssw.archive import MinimaArchive
from pamssw.fingerprint import rdf_histogram_fingerprint, structural_descriptor
from pamssw.state import State


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_from_atoms_payload(payload):
    return State(numbers=np.asarray(payload["numbers"], dtype=int),
                 positions=np.asarray(payload["positions"], dtype=float),
                 cell=np.asarray(payload.get("cell", np.zeros((3, 3))), dtype=float),
                 pbc=tuple(payload.get("pbc", (False, False, False))))


def rotate_translate(state, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(matrix)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    translation = rng.normal(size=3) * 7.0
    return state.with_flat_positions((state.positions @ q + translation).reshape(-1)), translation, q


def same_species_permutation(state, seed):
    rng = np.random.default_rng(seed)
    permutation = np.arange(state.n_atoms)
    for number in np.unique(state.numbers):
        indices = np.flatnonzero(state.numbers == number)
        permutation[indices] = rng.permutation(indices)
    if not np.array_equal(state.numbers, state.numbers[permutation]):
        raise AssertionError("permutation changed species assignment")
    return (State(numbers=state.numbers[permutation], positions=state.positions[permutation],
                  cell=None if state.cell is None else state.cell.copy(), pbc=state.pbc,
                  fixed_mask=state.fixed_mask[permutation].copy()), permutation.tolist())


def describe_query(label, reference, query, energy, transform_seed=None, permutation=None):
    archive = MinimaArchive(energy_tol=0.001, rmsd_tol=0.1)
    archive.add(reference, energy, parent_id=None)
    before_entries = len(archive.entries)
    matched = archive.find_match(query, energy)
    added = archive.add(query, energy, parent_id=0)
    ref_rdf = rdf_histogram_fingerprint(reference)
    query_rdf = rdf_histogram_fingerprint(query)
    ref_desc = structural_descriptor(reference)
    query_desc = structural_descriptor(query)
    return {
        "transformation": label,
        "seed": transform_seed,
        "new_index_to_reference_index": permutation,
        "find_match": matched is not None,
        "matched_entry_id": None if matched is None else matched.entry_id,
        "entry_count_before_query": before_entries,
        "entry_count_after_query": len(archive.entries),
        "query_entry_id": added.entry_id,
        "query_increased_archive_entries": len(archive.entries) > before_entries,
        "rdf_histogram_l2_difference": float(np.linalg.norm(query_rdf - ref_rdf)),
        "rdf_histogram_exact_equal": bool(np.array_equal(query_rdf, ref_rdf)),
        "full_structural_descriptor_l2_difference": float(np.linalg.norm(query_desc - ref_desc)),
        "full_structural_descriptor_exact_equal": bool(np.array_equal(query_desc, ref_desc)),
    }


def load_structures(audit_path):
    audit = json.loads(audit_path.read_text())
    structures = []
    sources = {}
    for arm in audit["arms"]:
        path = Path(arm["source"])
        digest = sha256(path)
        if digest != arm["source_sha256"]:
            raise ValueError(f"source hash mismatch: {path}")
        sources[str(path)] = digest
        data = json.loads(path.read_text())
        structures.append({
            "identity": {"arm": arm["arm"], "seed": arm["seed"], "kind": "initial", "record": None},
            "atoms": data["initial"]["atoms"], "energy_eV": float(data["initial"]["energy"]),
        })
        discoveries = arm.get("all_candidate_noninitial_first_discoveries", [])
        for item in discoveries:
            index = int(item["first_record"])
            landing = data["records"][index].get("landing")
            if not isinstance(landing, dict) or not isinstance(landing.get("atoms"), dict):
                raise ValueError(f"missing landing at {arm['arm']}/seed{arm['seed']}/{index}")
            structures.append({
                "identity": {"arm": arm["arm"], "seed": arm["seed"],
                             "kind": "first_noninitial_topology_representative",
                             "local_topology_class_id": item["class_id"], "record": index},
                "atoms": landing["atoms"], "energy_eV": float(landing["energy"]),
            })
    return audit, structures, sources


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, required=True,
                    help="committed zero-PES topology-audit.json")
    ap.add_argument("--output", type=Path, required=True,
                    help="new JSON result path; overwrite is refused")
    ap.add_argument("--matcher-source-root", type=Path, required=True,
                    help="checkout whose MinimaArchive implementation is imported")
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit_path = args.audit.resolve()
    matcher_root = args.matcher_source_root.resolve()
    matcher_files = [matcher_root / relative for relative in
                     ("pamssw/archive.py", "pamssw/fingerprint.py", "pamssw/state.py")]
    matcher_hashes = {str(path): sha256(path) for path in matcher_files}
    matcher_commit = subprocess.check_output(
        ["git", "-C", str(matcher_root), "rev-parse", "HEAD"], text=True).strip()
    audit, structures, sources = load_structures(audit_path)
    results = []
    for row in structures:
        ref = state_from_atoms_payload(row["atoms"])
        variants = [("exact_copy", None, ref, None)]
        for seed in (1, 2, 3):
            rotated, translation, rotation = rotate_translate(ref, seed)
            variants.append(("rigid_rotation_translation", seed, rotated,
                             {"translation_A": translation.tolist(), "rotation_matrix": rotation.tolist()}))
        for seed in (1, 2, 3):
            permuted, permutation = same_species_permutation(ref, seed)
            variants.append(("same_element_atom_permutation", seed, permuted,
                             {"new_index_to_reference_index": permutation}))
        checks = []
        for label, seed, variant_state, metadata in variants:
            permutation = (metadata["new_index_to_reference_index"]
                           if label == "same_element_atom_permutation" else None)
            check = describe_query(label, ref, variant_state, row["energy_eV"], seed,
                                   permutation=permutation)
            if label == "rigid_rotation_translation":
                check["rigid_transform"] = metadata
            checks.append(check)
        results.append({"identity": row["identity"], "energy_eV": row["energy_eV"],
                        "checks": checks})
    output = {
        "status": "offline_archive_identity_probe",
        "pes_evaluations": 0,
        "source_audit": str(audit_path),
        "source_audit_sha256": sha256(audit_path),
        "trajectory_source_hashes": sources,
        "matcher_source_root": str(matcher_root),
        "matcher_source_commit": matcher_commit,
        "matcher_source_hashes": matcher_hashes,
        "archive_configuration": {"energy_tol_eV": 0.001, "rmsd_tol_A": 0.1,
                                   "source": "existing PoolStarterAdapter PAM pool protocol"},
        "candidate_geometry_count": len(structures),
        "transformation_seeds": [1, 2, 3],
        "method": {
            "archive_identity": "MinimaArchive.find_match and add; exact composition/order, nonperiodic Kabsch RMSD threshold",
            "descriptor": "existing rdf_histogram_fingerprint and structural_descriptor; no PES or refitting",
            "rotation_translation": "deterministic proper orthogonal rotation and arbitrary translation, preserving geometry",
            "permutation": "permute positions only within each atomic-number group; geometry and composition are unchanged",
        },
        "structures": results,
        "summary": {
            label: {
                "queries": sum(check["transformation"] == label for row in results for check in row["checks"]),
                "matched": sum(check["transformation"] == label and check["find_match"]
                               for row in results for check in row["checks"]),
                "descriptor_exact_equal": sum(check["transformation"] == label and
                                               check["full_structural_descriptor_exact_equal"]
                                               for row in results for check in row["checks"]),
                "rdf_max_l2_difference": max((check["rdf_histogram_l2_difference"]
                                               for row in results for check in row["checks"]
                                               if check["transformation"] == label), default=0.0),
            }
            for label in ("exact_copy", "rigid_rotation_translation", "same_element_atom_permutation")
        },
        "interpretation_limits": [
            "A same-element permutation is the same exact labeled geometry and may expose ordered-RMSD identity failure; it is not a distinct minimum.",
            "A common graph or RDF descriptor does not prove two geometries are the same minimum.",
            "This only tests the recorded approximate matcher and descriptor on archived C4H6 geometries; it does not establish a generalized novelty bias.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
