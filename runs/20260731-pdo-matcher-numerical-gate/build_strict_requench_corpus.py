#!/usr/bin/env python3
"""Freeze the single residual PdO starter/landing pair without PES calls."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
MATCHER_EVIDENCE = RUN_ROOT / "evidence.json"
OUTPUT = RUN_ROOT / "strict_requench_corpus.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyzer = _load_module(
    RUN_ROOT / "analyze.py",
    "_pdo_matcher_corpus_analyzer",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _state_payload(state) -> dict[str, Any]:
    return {
        "numbers": state.numbers.tolist(),
        "positions": state.positions.tolist(),
        "cell": None if state.cell is None else state.cell.tolist(),
        "pbc": list(state.pbc),
        "fixed_mask": state.fixed_mask.tolist(),
    }


def _raw_checkpoint(
    raw: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    matches = [
        case
        for case in raw["cases"]
        if case["system"] == row["system"]
        and case["state_id"] == row["state_id"]
        and int(case["seed"]) == int(row["seed"])
        and case["arm"] == row["arm"]
    ]
    if len(matches) != 1:
        raise RuntimeError("residual pair case identity is not unique")
    case = matches[0]
    checkpoints = [
        checkpoint
        for checkpoint in case["checkpoints"]
        if int(checkpoint["horizon"]) == int(row["horizon"])
    ]
    if len(checkpoints) != 1:
        raise RuntimeError("residual pair checkpoint identity is not unique")
    return case, checkpoints[0]


def build() -> dict[str, Any]:
    from pamssw.io import read_state

    matcher = json.loads(MATCHER_EVIDENCE.read_text(encoding="utf-8"))
    if (
        matcher["gate"]["classification"]
        != "residual_energy_only_local_same"
        or matcher["gate"]["strict_requench_pair_count"] != 1
    ):
        raise RuntimeError("matcher evidence did not open the one-pair gate")
    residual = [
        row
        for row in matcher["pairs"]
        if row["mechanism"] == "energy_only_archive_split"
    ]
    if len(residual) != 1:
        raise RuntimeError("strict re-quench cohort is not exactly one pair")
    row = residual[0]
    raw_path = REPO_ROOT / matcher["raw_evidence_path"]
    if _sha256(raw_path) != matcher["raw_evidence_sha256"]:
        raise RuntimeError("raw first-passage evidence checksum drifted")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    case, checkpoint = _raw_checkpoint(raw, row)

    fixed_mask = analyzer._pdo_fixed_mask()
    starter_path = REPO_ROOT / row["starter_path"]
    landing_path = REPO_ROOT / row["landing_path"]
    if _sha256(landing_path) != checkpoint["landing_sha256"]:
        raise RuntimeError("residual landing checksum drifted")
    starter = read_state(starter_path, fixed_mask=fixed_mask)
    landing = read_state(landing_path, fixed_mask=fixed_mask)
    if not np.array_equal(starter.numbers, landing.numbers):
        raise RuntimeError("residual endpoint compositions differ")

    payload = {
        "schema_version": 1,
        "new_force_evaluations": 0,
        "selection": {
            "system": row["system"],
            "state_id": row["state_id"],
            "seed": row["seed"],
            "arm": row["arm"],
            "horizon": row["horizon"],
            "mechanism": row["mechanism"],
            "matcher_gate": matcher["gate"]["classification"],
        },
        "protocol": {
            "optimizer": "scipy-lbfgsb",
            "original_fmax_eV_per_A": float(
                case["effective_config"]["quench_fmax"]
            ),
            "strict_fmax_eV_per_A": 0.01,
            "maxiter": int(case["effective_config"]["quench_maxiter"]),
            "objective": "true_mace_pes_no_bias_no_softening",
            "energy_tol_eV": float(
                case["effective_config"]["dedup_energy_tol"]
            ),
            "rmsd_tol_A": float(
                case["effective_config"]["dedup_rmsd_tol"]
            ),
            "descriptor_tol": float(
                case["effective_config"]["min_escape_descriptor_delta"]
            ),
        },
        "fixed_mask_provenance": matcher["fixed_mask_provenance"],
        "source": {
            "matcher_evidence_path": str(
                MATCHER_EVIDENCE.relative_to(REPO_ROOT)
            ),
            "matcher_evidence_sha256": _sha256(MATCHER_EVIDENCE),
            "raw_evidence_path": matcher["raw_evidence_path"],
            "raw_evidence_sha256": matcher["raw_evidence_sha256"],
        },
        "original_observation": {
            "starter_energy_eV": float(case["starter_energy_eV"]),
            "landing_energy_eV": float(checkpoint["landing_energy_eV"]),
            "landing_delta_eV": float(checkpoint["landing_delta_eV"]),
            "landing_final_max_force_eV_per_A": float(
                checkpoint["final_max_force_eV_per_A"]
            ),
            "landing_force_evaluations": int(
                checkpoint["purpose_counts"]["landing_true_quench"]
            ),
        },
        "endpoints": [
            {
                "endpoint": "starter",
                "source_path": str(starter_path.relative_to(REPO_ROOT)),
                "source_sha256": _sha256(starter_path),
                "state": _state_payload(starter),
            },
            {
                "endpoint": "landing",
                "source_path": str(landing_path.relative_to(REPO_ROOT)),
                "source_sha256": _sha256(landing_path),
                "state": _state_payload(landing),
            },
        ],
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    built = build()
    print(json.dumps(built["selection"], indent=2, sort_keys=True))
