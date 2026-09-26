"""Zero-PES comparison of ordered, ASE and Hungarian C4H6/C60 geometry matchers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.build import molecule as ase_molecule
from ase.geometry import distance as ase_distance
from ase.io import read
from pymatgen.core import Molecule
from pymatgen.core.molecule_matcher import HungarianOrderMatcher

from pamssw.archive import MinimaArchive
from pamssw.fingerprint import rdf_histogram_fingerprint, structural_descriptor
from pamssw.state import State

THRESHOLD_A = 0.1
MACE_ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
C60_FILES = {
    "c60_isomer2_qualified_defect": MACE_ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-2/final.extxyz",
    "c60_native_ls_1101_landing0": MACE_ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/direction-probe/runs/native_ls-1101/landing-0.traj",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def to_state(numbers, positions):
    return State(numbers=np.asarray(numbers, dtype=int), positions=np.asarray(positions, dtype=float),
                 cell=np.zeros((3, 3)), pbc=(False, False, False))


def to_ase(state):
    from ase import Atoms
    return Atoms(numbers=state.numbers, positions=state.positions, pbc=False)


def to_pm(state):
    return Molecule([int(x) for x in state.numbers], state.positions, charge=0, spin_multiplicity=1)


def rotate_translate(state, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    translation = rng.normal(size=3) * 7.0
    positions = state.positions @ q + translation
    return to_state(state.numbers, positions), {"rotation_matrix": q.tolist(), "translation_A": translation.tolist()}


def species_permutation(state, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(state.numbers))
    for number in np.unique(state.numbers):
        group = np.flatnonzero(state.numbers == number)
        indices[group] = rng.permutation(group)
    if not np.array_equal(state.numbers, state.numbers[indices]):
        raise AssertionError("permutation changed atomic species")
    return to_state(state.numbers[indices], state.positions[indices]), indices.tolist()


def load_c4h6(audit_path):
    audit = json.loads(audit_path.read_text())
    structures, source_hashes = [], {}
    for arm in audit["arms"]:
        result_path = Path(arm["source"])
        digest = sha256(result_path)
        if arm.get("source_sha256") != digest:
            raise ValueError(f"C4H6 source hash mismatch: {result_path}")
        source_hashes[str(result_path)] = digest
        data = json.loads(result_path.read_text())
        structures.append({
            "identity": {"system": "C4H6", "arm": arm["arm"], "seed": arm["seed"],
                         "kind": "initial", "record": None},
            "state": to_state(data["initial"]["atoms"]["numbers"],
                              data["initial"]["atoms"]["positions"]),
        })
        for first in arm.get("all_candidate_noninitial_first_discoveries", []):
            index = int(first["first_record"])
            landing = data["records"][index]["landing"]["atoms"]
            structures.append({
                "identity": {"system": "C4H6", "arm": arm["arm"], "seed": arm["seed"],
                             "kind": "first_noninitial_topology_representative",
                             "local_topology_class_id": first["class_id"], "record": index},
                "state": to_state(landing["numbers"], landing["positions"]),
            })
    return structures, source_hashes


def load_c60():
    structures, source_hashes = [], {}
    ideal = ase_molecule("C60")
    structures.append({"identity": {"system": "C60", "kind": "ASE_build_molecule_C60"},
                       "state": to_state(ideal.numbers, ideal.positions)})
    for label, path in C60_FILES.items():
        atoms = read(path, index=0)
        if len(atoms) != 60 or not np.all(atoms.numbers == 6):
            raise ValueError(f"expected a 60-carbon geometry in {path}")
        source_hashes[str(path)] = sha256(path)
        structures.append({"identity": {"system": "C60", "kind": label, "source": str(path)},
                           "state": to_state(atoms.numbers, atoms.positions)})
    return structures, source_hashes


def method_values(reference, query):
    ref_ase, query_ase = to_ase(reference), to_ase(query)
    ref_pm, query_pm = to_pm(reference), to_pm(query)
    archive = MinimaArchive(energy_tol=0.001, rmsd_tol=THRESHOLD_A)
    archive.add(reference, energy=0.0, parent_id=None)
    try:
        ordered_rmsd = float(MinimaArchive._rmsd(reference, query))
    except Exception as exc:
        ordered_rmsd = None
        ordered_setup_error = f"{type(exc).__name__}: {exc}"
    else:
        ordered_setup_error = None
    ordered_start = time.perf_counter()
    try:
        ordered_match = archive.find_match(query, energy=0.0)
        ordered_pass = ordered_match is not None
        ordered_error = ordered_setup_error
    except Exception as exc:
        ordered_pass = None
        ordered_error = f"{type(exc).__name__}: {exc}"
    ordered_seconds = time.perf_counter() - ordered_start

    ase_start = time.perf_counter()
    try:
        ase_rmsd = float(ase_distance(ref_ase, query_ase, permute=True) / math.sqrt(len(reference.numbers)))
        ase_pass = ase_rmsd <= THRESHOLD_A
        ase_error = None
    except Exception as exc:
        ase_rmsd = None
        ase_pass = None
        ase_error = f"{type(exc).__name__}: {exc}"
    ase_seconds = time.perf_counter() - ase_start

    hungarian = HungarianOrderMatcher(ref_pm)
    hungarian_start = time.perf_counter()
    try:
        _inds, _rotation, _translation, hungarian_rmsd = hungarian.match(query_pm)
        hungarian_rmsd = float(hungarian_rmsd)
        hungarian_error = None
    except Exception as exc:  # Retain method failure as a result, never suppress it.
        hungarian_rmsd = None
        hungarian_error = f"{type(exc).__name__}: {exc}"
    hungarian_seconds = time.perf_counter() - hungarian_start

    return {
        "ordered_MinimaArchive": {"rmsd_A": ordered_rmsd,
                                  "passed_0p1A": ordered_pass,
                                  "seconds": ordered_seconds, "error": ordered_error},
        "ASE_geometry_distance_over_sqrtN": {"rmsd_A": ase_rmsd,
                                             "passed_0p1A": ase_pass,
                                             "seconds": ase_seconds, "error": ase_error},
        "pymatgen_HungarianOrderMatcher": {"rmsd_A": hungarian_rmsd,
                                           "passed_0p1A": None if hungarian_rmsd is None else hungarian_rmsd <= THRESHOLD_A,
                                           "seconds": hungarian_seconds,
                                           "error": hungarian_error},
    }


def transformed_queries(reference):
    out = [("exact", None, reference, None)]
    for seed in (1, 2, 3):
        rigid, transform = rotate_translate(reference, seed)
        out.append(("rigid", seed, rigid, transform))
    for seed in (1, 2, 3):
        permuted, indices = species_permutation(reference, seed)
        out.append(("perm", seed, permuted, {"new_index_to_reference_index": indices}))
    for seed in (1, 2, 3):
        rigid, transform = rotate_translate(reference, seed)
        combined, indices = species_permutation(rigid, seed)
        out.append(("rigid_plus_perm", seed, combined,
                    {**transform, "new_index_to_rigid_index": indices}))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True,
                        help="committed C4H6 topology-audit.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output JSON path; existing files are refused")
    parser.add_argument("--matcher-root", type=Path, default=MACE_ROOT,
                        help="checkout containing MinimaArchive")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    matcher_root = args.matcher_root.resolve()
    matcher_commit = subprocess.check_output(["git", "-C", str(matcher_root), "rev-parse", "HEAD"], text=True).strip()
    structures, c4_hashes = load_c4h6(args.audit.resolve())
    c60_structures, c60_hashes = load_c60()
    structures.extend(c60_structures)

    detailed, timings = [], defaultdict(lambda: defaultdict(list))
    for source in structures:
        reference = source["state"]
        for category, seed, query, transform in transformed_queries(reference):
            methods = method_values(reference, query)
            detailed.append({"reference": source["identity"], "transformation": category,
                             "transformation_seed": seed, "transform": transform,
                             "methods": methods})
            for method, values in methods.items():
                timings[category][method].append(values["seconds"])

    summary = {}
    for category, method_map in timings.items():
        summary[category] = {}
        for method, elapsed in method_map.items():
            rows = [row["methods"][method] for row in detailed if row["transformation"] == category]
            passes = sum(row["passed_0p1A"] is True for row in rows)
            failures = sum(row.get("error") is not None for row in rows)
            summary[category][method] = {
                "queries": len(rows), "passes_at_0p1A": passes, "errors": failures,
                "elapsed_s_total": float(sum(elapsed)),
                "elapsed_s_median_per_query": float(np.median(elapsed)) if elapsed else None,
                "elapsed_s_max_per_query": float(max(elapsed)) if elapsed else None,
            }

    # Pair distinct archived C60 references for raw context only; no pair is
    # labeled a true/false minimum match from graph topology alone.
    c60_pairs = []
    for i, first in enumerate(c60_structures):
        for second in c60_structures[i + 1:]:
            c60_pairs.append({"first": first["identity"], "second": second["identity"],
                              "ground_truth_assigned": False,
                              "methods": method_values(first["state"], second["state"])})

    output = {
        "status": "offline_geometry_matcher_qualification",
        "pes_evaluations": 0,
        "threshold_A": THRESHOLD_A,
        "matcher_source": {"root": str(matcher_root), "git_commit": matcher_commit,
                           "files_sha256": {str(matcher_root / rel): sha256(matcher_root / rel)
                                            for rel in ("pamssw/archive.py", "pamssw/state.py")}},
        "input_sources": {"topology_audit": str(args.audit.resolve()),
                           "topology_audit_sha256": sha256(args.audit.resolve()),
                           "c4h6_trajectory_sha256": c4_hashes,
                           "c60_geometry_sha256": c60_hashes},
        "software_versions": {name: importlib.metadata.version(name)
                              for name in ("ase", "pymatgen", "numpy", "scipy")},
        "structure_count": len(structures),
        "transformation_counts": {key: len([r for r in detailed if r["transformation"] == key])
                                  for key in ("exact", "rigid", "perm", "rigid_plus_perm")},
        "timing_scope": "matcher calls only; structure loading and transformations excluded",
        "methods": {
            "ordered_MinimaArchive": "MinimaArchive.find_match; exact species sequence and existing nonperiodic Kabsch RMSD <= 0.1 A",
            "ASE_geometry_distance_over_sqrtN": "ase.geometry.distance(..., permute=True)/sqrt(N) <= 0.1 A; existing inertia-axis/greedy method",
            "pymatgen_HungarianOrderMatcher": "HungarianOrderMatcher(reference).match(query)[-1] <= 0.1 A; no OpenBabel; records exceptions",
        },
        "by_transformation": summary,
        "c60_distinct_reference_pairwise_raw_comparisons": c60_pairs,
        "structures": detailed,
        "interpretation_limits": [
            "Same-geometry transformations test matcher invariance, not basin identity.",
            "Different C60 representatives are reported as raw pairwise comparisons without a topology-derived truth label.",
            "C60 ASE build_molecule is a geometry-only symmetric reference, not a MACE/PES-qualified structure.",
            "Times are environment- and geometry-specific method-call timings; they do not rank search efficiency.",
            "Hungarian principal-axis assignment can be symmetry-degenerate or suboptimal; failures and threshold misses are preserved.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
