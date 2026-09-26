"""Zero-PES public-callback qualification for ordered/ASE pool identity modes.

Prepared against the published ``StarterPoolSnapshot`` callback. Run only after
the target research adapter implementation is reviewed/green.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import molecule as ase_molecule
from ase.io import read

from pamssw.standalone.starter_selection import StarterObservation, StarterPoolSnapshot
from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter

OWN_ROOT = Path(__file__).resolve().parents[4]
MAIN_ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
AUDIT = OWN_ROOT / "research/ga_ssw/evidence/ls-climb-depth-panel-20260926/topology-audit.json"
C60_FILES = {
    "c60_isomer2_qualified_defect": MAIN_ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-2/final.extxyz",
    "c60_native_ls_1101_landing0": MAIN_ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/direction-probe/runs/native_ls-1101/landing-0.traj",
}
ENERGY_TOL = 0.001
RMSD_TOL_A = 0.1
IDENTITY_MODES = ("ordered_v1", "ase_permute_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atoms_from(numbers, positions):
    return Atoms(numbers=np.asarray(numbers, dtype=int),
                 positions=np.asarray(positions, dtype=float), pbc=False)


def load_geometries():
    audit = json.loads(AUDIT.read_text())
    geometries, sources = [], {str(AUDIT): sha256(AUDIT)}
    for arm in audit["arms"]:
        path = Path(arm["source"])
        digest = sha256(path)
        if arm.get("source_sha256") != digest:
            raise ValueError(f"C4H6 source hash mismatch: {path}")
        sources[str(path)] = digest
        run = json.loads(path.read_text())
        geometries.append({
            "identity": {"system": "C4H6", "arm": arm["arm"], "seed": arm["seed"], "kind": "initial"},
            "atoms": atoms_from(run["initial"]["atoms"]["numbers"], run["initial"]["atoms"]["positions"]),
        })
        for discovery in arm.get("all_candidate_noninitial_first_discoveries", []):
            record_index = int(discovery["first_record"])
            atoms = run["records"][record_index]["landing"]["atoms"]
            geometries.append({
                "identity": {"system": "C4H6", "arm": arm["arm"], "seed": arm["seed"],
                             "kind": "first_noninitial_topology_representative",
                             "local_topology_class_id": discovery["class_id"], "record": record_index},
                "atoms": atoms_from(atoms["numbers"], atoms["positions"]),
            })

    ideal = ase_molecule("C60")
    geometries.append({"identity": {"system": "C60", "kind": "ASE_build_molecule_C60"}, "atoms": ideal})
    for name, path in C60_FILES.items():
        path = path.resolve()
        atoms = read(path, index=0)
        if len(atoms) != 60 or not np.all(atoms.numbers == 6):
            raise ValueError(f"expected a 60-carbon input: {path}")
        geometries.append({"identity": {"system": "C60", "kind": name, "source": str(path)}, "atoms": atoms})
        sources[str(path)] = sha256(path)
    if len(geometries) != 33:
        raise ValueError(f"expected previous 33-geometry panel, found {len(geometries)}")
    return geometries, sources


def rigid_transform(atoms, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved = atoms.copy()
    moved.positions = atoms.positions @ q + rng.normal(size=3) * 7.0
    return moved


def relabel_same_species(atoms, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(atoms))
    for z in np.unique(atoms.numbers):
        group = np.flatnonzero(atoms.numbers == z)
        indices[group] = rng.permutation(group)
    if not np.array_equal(atoms.numbers, atoms.numbers[indices]):
        raise AssertionError("relabeling changed species")
    moved = atoms[indices]
    moved.set_pbc(False)
    return moved, indices.tolist()


def transformed_queries(atoms):
    yield "exact", None, atoms.copy(), None
    for seed in (1, 2, 3):
        yield "rigid", seed, rigid_transform(atoms, seed), None
    for seed in (1, 2, 3):
        query, permutation = relabel_same_species(atoms, seed)
        yield "perm", seed, query, {"new_index_to_reference_index": permutation}
    for seed in (1, 2, 3):
        rotated = rigid_transform(atoms, seed)
        query, permutation = relabel_same_species(rotated, seed)
        yield "rigid_plus_perm", seed, query, {"new_index_to_rigid_index": permutation}


def make_adapter(identity_mode=None):
    kwargs = dict(mode="uniform", energy_tol=ENERGY_TOL, rmsd_tol=RMSD_TOL_A)
    if identity_mode is not None:
        kwargs["identity_matcher"] = identity_mode
    return PoolStarterAdapter(**kwargs)


def snapshot(atoms_list, step, *, current=0, landing=None, cost=1):
    observations = tuple(StarterObservation(i, atoms.copy(), 0.0, 0.0)
                        for i, atoms in enumerate(atoms_list))
    return StarterPoolSnapshot(observations, current, landing, step, cost)


def deep_equal(left, right, path="root"):
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or not np.array_equal(left, right):
            raise AssertionError(f"checkpoint differs at {path}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f"checkpoint keys differ at {path}")
        for key in left:
            deep_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"checkpoint sequence differs at {path}")
        for i, (a, b) in enumerate(zip(left, right)):
            deep_equal(a, b, f"{path}[{i}]")
    elif left != right:
        raise AssertionError(f"checkpoint differs at {path}: {left!r} != {right!r}")


def run_pair(reference, query, identity_mode, *, check_default=False):
    continuous = make_adapter(identity_mode)
    continuous_rng = np.random.default_rng(505)
    first = snapshot([reference], 0)
    continuous(first, continuous_rng)
    saved = continuous.export_state()

    # Split continuation uses the documented explicit export/restore contract.
    resumed = make_adapter(identity_mode)
    resumed.restore_state(saved)
    # RNG state is caller-owned; recreate the exact one-step state used above.
    resumed_rng = np.random.default_rng(505)
    resumed_rng.integers(1)

    pair = snapshot([reference, query], 1, current=0, landing=1, cost=2)
    continuous(pair, continuous_rng)
    resumed(pair, resumed_rng)
    deep_equal(continuous.export_state(), resumed.export_state(), "continued_state")

    state = continuous.export_state()
    entry_count = len(state["archive"]["entries"])
    mapping = list(state["mapping"])
    observation_one = entry_count == 1 and mapping == [0, 0]
    if identity_mode == "ase_permute_v1" and not observation_one:
        raise AssertionError(f"ASE matcher did not collapse same-geometry pair: {mapping}")
    if observation_one:
        entry = state["archive"]["entries"][0]
        payload_state = entry["state"]
        if not np.array_equal(payload_state["numbers"], reference.numbers):
            raise AssertionError("stored representative atom order changed")
        if not np.array_equal(payload_state["positions"], reference.positions):
            raise AssertionError("stored representative coordinates changed")
        if entry["node_successes"] != 0:
            raise AssertionError(f"duplicate geometry received node success={entry['node_successes']}")
        if state["outcomes"][1]["is_new_minimum"] or not state["outcomes"][1]["is_duplicate"]:
            raise AssertionError("merged observation outcome is not a duplicate")

    if check_default:
        default = make_adapter(None)
        explicit = make_adapter("ordered_v1")
        default_rng = np.random.default_rng(505)
        explicit_rng = np.random.default_rng(505)
        default(first, default_rng)
        explicit(first, explicit_rng)
        default(pair, default_rng)
        explicit(pair, explicit_rng)
        a, b = default.export_state(), explicit.export_state()
        if a["mapping"] != b["mapping"] or len(a["archive"]["entries"]) != len(b["archive"]["entries"]):
            raise AssertionError("legacy default behavior differs from explicit ordered_v1")
        if a["contract"] != b["contract"]:
            raise AssertionError("legacy default checkpoint contract differs from ordered_v1")

    return {"entry_count": entry_count, "mapping": mapping,
            "representative_order_preserved": observation_one,
            "duplicate_node_successes": (state["archive"]["entries"][0]["node_successes"]
                                          if observation_one else None),
            "continuation_equal": True}


def check_rejections():
    atoms = ase_molecule("C60")
    old = make_adapter("ordered_v1")
    old( snapshot([atoms], 0), np.random.default_rng(505))
    old_payload = old.export_state()
    old_restored = make_adapter("ordered_v1")
    old_restored.restore_state(old_payload)
    deep_equal(old_restored.export_state(), old_payload, "ordered_v1_restore")
    new = make_adapter("ase_permute_v1")
    try:
        new.restore_state(old_payload)
    except ValueError as exc:
        old_mode_rejected = str(exc)
    else:
        raise AssertionError("ASE mode accepted the ordered-v1 checkpoint")

    new( snapshot([atoms], 0), np.random.default_rng(505))
    new_payload = new.export_state()
    expected_ase_version = new_payload["version"]
    if expected_ase_version < 2:
        raise AssertionError(f"ASE matcher checkpoint version must be >=2, got {expected_ase_version}")
    wrong_version = dict(new_payload)
    wrong_version["version"] = 1
    try:
        make_adapter("ase_permute_v1").restore_state(wrong_version)
    except ValueError as exc:
        bad_version_rejected = str(exc)
    else:
        raise AssertionError("ASE mode accepted a v1 checkpoint")

    return {"ordered_checkpoint_rejected_by_ase_mode": old_mode_rejected,
            "ordered_v1_checkpoint_restores_in_ordered_mode": True,
            "wrong_version_rejected_by_ase_mode": bad_version_rejected,
            "ase_payload_version": expected_ase_version,
            "ordered_payload_version": old_payload["version"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new result JSON path; existing paths are refused")
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    geometries, source_hashes = load_geometries()
    results = []
    for geometry in geometries:
        for category, seed, query, transform in transformed_queries(geometry["atoms"]):
            for identity_mode in IDENTITY_MODES:
                output = run_pair(geometry["atoms"], query, identity_mode,
                                  check_default=(category == "exact" and seed is None))
                results.append({"reference": geometry["identity"], "transformation": category,
                                "seed": seed, "transform": transform,
                                "identity_matcher": identity_mode, **output})
    rejection_checks = check_rejections()
    output = {
        "status": "public_callback_pool_adapter_qualification",
        "pes_evaluations": 0,
        "pool_snapshots": "StarterObservation/StarterPoolSnapshot public callback only",
        "reference_count": len(geometries),
        "pair_count_per_identity_mode": len(geometries) * 10,
        "threshold_A": RMSD_TOL_A,
        "energy_tolerance_eV": ENERGY_TOL,
        "input_source_sha256": source_hashes,
        "main_root": str(args.main_root.resolve()),
        "main_commit": __import__("subprocess").check_output(
            ["git", "-C", str(args.main_root.resolve()), "rev-parse", "HEAD"], text=True).strip(),
        "main_source_sha256": {
            str(args.main_root.resolve() / rel): sha256(args.main_root.resolve() / rel)
            for rel in ("research/ga_ssw/pool_starter_adapter.py",
                        "research/ga_ssw/pool_molecular_archive.py",
                        "pamssw/archive.py")
        },
        "software_versions": {name: importlib.metadata.version(name)
                              for name in ("ase", "numpy", "scipy")},
        "checkpoint_rejections": rejection_checks,
        "cases": results,
        "limits": [
            "No PES, calculator, optimization, or physical-minimum validation is involved.",
            "A match tests invariance of one same-geometry pair; distinct structures are not compared or labeled as same/different basins.",
            "The ASE matcher qualification is limited to these 33 inputs and specified transforms.",
            "The test does not reimplement or infer the adapter identity algorithm; it only uses its public callback/export/restore contract.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
