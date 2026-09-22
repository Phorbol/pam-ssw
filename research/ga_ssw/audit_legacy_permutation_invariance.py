"""Offline permutation audit for the legacy GA routing descriptor.

This is an analysis script, not a controller test and not a PES workflow.  It
requires an explicit JSON manifest so every frame and configuration table is
provenanced by the caller.  Each case entry has ``name``, ``frames`` and
``config`` fields.  ``config`` must be an existing GA config containing
``bond_lengths`` and frozen ``references``; ``neighbor_range`` may be supplied
by the manifest when the archived runner kept it in a shared fixture.

For each saved frame the script compares the legacy descriptor/projection of
the original atom order with a deterministic permutation of exactly the same
geometry.  It also reports an experiment-only comparator that reorders rows
by the complete legacy n/d fingerprint rather than count-only keys.  The
comparator applies to candidate and frozen reference rows, is not paper DCCD,
and is never sent to the controller.

Example (must use paths from the approved saved-artifact manifest):

    python research/ga_ssw/audit_legacy_permutation_invariance.py \
      --manifest /path/to/ga-permutation-manifest.json \
      --output /path/to/audit.json

The script performs no calculator or surface calls.  Expected exact signal:
the raw descriptor can drift when count ties are permuted; a full-fingerprint
row order removes only that representation-order drift.  A nonzero projection
change after full sorting would indicate a separate descriptor calculation or
species/channel issue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from pamssw.standalone.legacy_descriptor import (
    cluster_descriptor,
    descriptor_similarity,
)


def _pair_table(raw):
    result = {}
    for key, value in raw.items():
        if isinstance(key, str):
            key = key.strip().strip("()").replace(" ", "")
            first, second = key.split(",")
            pair = (int(first), int(second))
        else:
            pair = tuple(int(item) for item in key)
        result[tuple(sorted(pair))] = float(value)
    return result


def _full_sort(descriptor):
    """Return a descriptor with an experiment-only complete row ordering."""
    n = len(descriptor["n1"])
    rows = []
    for i in range(n):
        key = tuple(
            value
            for name in ("n1", "n2", "n3", "d1", "d2", "d3")
            for value in np.asarray(descriptor[name][i]).reshape(-1).tolist()
        )
        rows.append((key, i))
    order = [i for _, i in sorted(rows, key=lambda item: item[0])]
    return {
        name: [descriptor[name][i] for i in order]
        for name in descriptor
    }


def _projection(descriptor, references, weights):
    return [
        descriptor_similarity(descriptor, reference, weights)
        for reference in references
    ]


def _permuted(atoms):
    # Reverse order is deterministic and exercises atom-label permutation.
    result = atoms.copy()
    result = result[list(range(len(result) - 1, -1, -1))]
    result.calc = None
    return result


def audit_case(case):
    config = json.loads(Path(case["config"]).read_text(encoding="utf-8"))
    bonds = _pair_table(config["bond_lengths"])
    if "neighbor_range" in case:
        neighbor_range = float(case["neighbor_range"])
    elif "neighbor_range" in config:
        neighbor_range = float(config["neighbor_range"])
    else:
        raise ValueError(f"{case['name']}: manifest/config must provide sourced neighbor_range")
    references = config["references"]
    if "weights" not in case:
        raise ValueError(f"{case['name']}: manifest must provide sourced descriptor weights")
    weights = tuple(case["weights"])
    if len(weights) != 6 or any(float(value) < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError(f"{case['name']}: invalid six-component descriptor weights")
    identity_tolerance = case.get("identity_tolerance", config.get("ga", {}).get("projection_tolerance"))
    if identity_tolerance is None:
        raise ValueError(f"{case['name']}: manifest must provide sourced identity_tolerance")
    references_full = [_full_sort(reference) for reference in references]
    rows = []
    for frame in case["frames"]:
        frame_path = frame["path"]
        frame_index = frame["index"]
        atoms = read(frame_path, index=frame_index)
        if bool(np.any(atoms.pbc)):
            raise ValueError(f"{frame_path}: periodic input is unsupported by cluster_descriptor")
        permuted = _permuted(atoms)
        original = cluster_descriptor(atoms.numbers, atoms.positions, bonds, neighbor_range)
        shuffled = cluster_descriptor(permuted.numbers, permuted.positions, bonds, neighbor_range)
        original_full = _full_sort(original)
        shuffled_full = _full_sort(shuffled)
        raw_projection = _projection(original, references, weights)
        perm_projection = _projection(shuffled, references, weights)
        full_projection = _projection(original_full, references_full, weights)
        full_perm_projection = _projection(shuffled_full, references_full, weights)
        rows.append({
            "frame": str(frame_path),
            "frame_index": frame_index,
            "atoms": len(atoms),
            "raw_descriptor_equal": original == shuffled,
            "raw_projection_max_abs_delta": float(np.max(np.abs(np.asarray(raw_projection) - perm_projection))),
            "raw_identity_same_at_tolerance": bool(np.max(np.abs(np.asarray(raw_projection) - perm_projection)) <= identity_tolerance),
            "full_descriptor_equal": original_full == shuffled_full,
            "full_projection_max_abs_delta": float(np.max(np.abs(np.asarray(full_projection) - full_perm_projection))),
            "full_identity_same_at_tolerance": bool(np.max(np.abs(np.asarray(full_projection) - full_perm_projection)) <= identity_tolerance),
            "identity_tolerance": float(identity_tolerance),
        })
    return {
        "name": case["name"],
        "config": str(case["config"]),
        "bond_lengths": config["bond_lengths"],
        "neighbor_range": neighbor_range,
        "weights": list(weights),
        "frames": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = {
        "scope": "offline descriptor permutation audit; zero PES calls",
        "manifest": str(args.manifest),
        "cases": [audit_case(case) for case in manifest["cases"]],
        "interpretation": {
            "raw": "legacy count-only row order; ties may drift under atom permutation",
            "full": "experimental complete n/d row sort; not paper DCCD or Java parity",
            "success_signal": "full_projection_max_abs_delta is zero within floating precision",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
