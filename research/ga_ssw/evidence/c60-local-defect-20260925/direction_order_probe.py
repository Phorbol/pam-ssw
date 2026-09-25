#!/usr/bin/env python3
"""Run the existing C60 escape probe on one physically identical relabeling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
PLAN = HERE / "direction-order-probe" / "plan.json"
PREFLIGHT = HERE / "direction-order-probe" / "preflight.json"

import escape_probe as runner

runner.PLAN = PLAN
runner.PREFLIGHT = PREFLIGHT
_original_preflight = runner.preflight


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_relabeling(plan):
    mapping_artifact = plan["source_artifacts"]["input_relabeling"]
    original_artifact = plan["source_artifacts"]["original_defect_input"]
    new_artifact = plan["source_artifacts"]["defect_input"]
    mapping_path = Path(mapping_artifact["path"])
    original_path = Path(original_artifact["path"])
    new_path = Path(new_artifact["path"])
    for path, expected in ((mapping_path, mapping_artifact["sha256"]),
                           (original_path, original_artifact["sha256"]),
                           (new_path, new_artifact["sha256"])):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"relabeling identity artifact missing or changed: {path}")
    metadata = json.loads(mapping_path.read_text())
    permutation = np.random.default_rng(25092573).permutation(60)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(60)
    if (metadata.get("rng") != "numpy.random.default_rng(25092573).permutation(60)" or
            not np.array_equal(metadata.get("permuted_index_to_original_index"), permutation) or
            not np.array_equal(metadata.get("original_index_to_permuted_index"), inverse)):
        raise RuntimeError("saved atom permutation differs from frozen RNG or its inverse")
    original, relabeled = read(original_path), read(new_path)
    if (len(original) != 60 or len(relabeled) != 60 or
            not np.array_equal(original.numbers, relabeled.numbers[ inverse ]) or
            not np.array_equal(relabeled.positions, original.positions[permutation]) or
            not np.array_equal(original.cell.array, relabeled.cell.array) or
            not np.array_equal(original.pbc, relabeled.pbc) or
            bool(original.constraints) != bool(relabeled.constraints)):
        raise RuntimeError("permuted input is not an exact row reordering of the qualified defect")
    return {
        "status": "physical_input_identity_passed",
        "atom_count": len(original),
        "permutation_seed": 25092573,
        "positions_exact_after_reindexing": True,
        "species_cell_pbc_preserved": True,
        "source_input_sha256": original_artifact["sha256"],
        "permuted_input_sha256": new_artifact["sha256"],
        "mapping_sha256": mapping_artifact["sha256"],
        "wrapper_sha256": sha256(__file__),
    }


def preflight(*, require_preflight=False):
    plan = json.loads(PLAN.read_text())
    identity = verify_relabeling(plan)
    checked, ledger = _original_preflight(require_preflight=require_preflight)
    if require_preflight:
        prior = json.loads(PREFLIGHT.read_text())
        saved = prior.get("input_relabeling_identity", {})
        if saved.get("wrapper_sha256") != identity["wrapper_sha256"]:
            raise RuntimeError("saved relabeling preflight is stale for this wrapper")
    checked["input_relabeling_identity"] = identity
    return checked, ledger


runner.preflight = preflight


if __name__ == "__main__":
    runner.main()
