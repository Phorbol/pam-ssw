#!/usr/bin/env python3
"""Analyze the preregistered one-pair PdO strict re-quench gate."""

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
CORPUS_PATH = RUN_ROOT / "strict_requench_corpus.json"
OUTPUT_DIR = RUN_ROOT / "strict-requench-output"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    RUN_ROOT / "protocol.py",
    "_pdo_matcher_strict_protocol_analysis",
)


def _state(payload: Mapping[str, Any]):
    from pamssw.state import State

    return State(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=(
            None
            if payload["cell"] is None
            else np.asarray(payload["cell"], dtype=float)
        ),
        pbc=tuple(bool(value) for value in payload["pbc"]),
        fixed_mask=np.asarray(payload["fixed_mask"], dtype=bool),
    )


def analyze() -> dict[str, Any]:
    from pamssw.archive import MinimaArchive
    from pamssw.fingerprint import descriptor_distance, structural_descriptor

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    summary_path = OUTPUT_DIR / "summary.json"
    rows_path = OUTPUT_DIR / "rows.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    by_endpoint = {row["endpoint"]: row for row in rows}
    if set(by_endpoint) != {"starter", "landing"}:
        raise RuntimeError("strict re-quench endpoints are incomplete")
    if summary["total_force_evaluations"] != sum(
        int(row["force_evaluations"]) for row in rows
    ):
        raise RuntimeError("strict re-quench summary accounting is open")

    starter = _state(by_endpoint["starter"]["final"]["state"])
    landing = _state(by_endpoint["landing"]["final"]["state"])
    strict_energy_delta = float(
        by_endpoint["landing"]["final"]["energy_eV"]
        - by_endpoint["starter"]["final"]["energy_eV"]
    )
    strict_rmsd = float(MinimaArchive._rmsd(starter, landing))
    strict_descriptor_delta = descriptor_distance(
        structural_descriptor(starter),
        structural_descriptor(landing),
    )
    fmax = float(corpus["protocol"]["strict_fmax_eV_per_A"])
    converged = {
        endpoint: (
            row["termination_reason"] == "converged"
            and float(row["final"]["max_active_force_eV_per_A"]) <= fmax
        )
        for endpoint, row in by_endpoint.items()
    }
    gate = protocol.evaluate_strict_requench(
        starter_converged=converged["starter"],
        landing_converged=converged["landing"],
        strict_energy_delta_eV=strict_energy_delta,
        energy_tol_eV=float(corpus["protocol"]["energy_tol_eV"]),
        strict_indexed_mic_rmsd_A=strict_rmsd,
        rmsd_tol_A=float(corpus["protocol"]["rmsd_tol_A"]),
    )
    evidence = {
        "schema_version": 1,
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "corpus_sha256": sha256(CORPUS_PATH.read_bytes()).hexdigest(),
        "execution_summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "execution_summary_sha256": sha256(
            summary_path.read_bytes()
        ).hexdigest(),
        "new_force_evaluations": int(summary["total_force_evaluations"]),
        "gpu_wall_time_s": float(summary["total_wall_time_s"]),
        "original_observation": corpus["original_observation"],
        "strict_endpoints": {
            endpoint: {
                "converged": converged[endpoint],
                "initial_energy_eV": row["initial"]["energy_eV"],
                "final_energy_eV": row["final"]["energy_eV"],
                "energy_drop_eV": (
                    row["final"]["energy_eV"] - row["initial"]["energy_eV"]
                ),
                "initial_max_force_eV_per_A": row["initial"][
                    "max_active_force_eV_per_A"
                ],
                "final_max_force_eV_per_A": row["final"][
                    "max_active_force_eV_per_A"
                ],
                "iterations": row["n_iter"],
                "force_evaluations": row["force_evaluations"],
                "wall_time_s": row["wall_time_s"],
                "termination_reason": row["termination_reason"],
                "optimizer_success": row["telemetry"]["optimizer_success"],
            }
            for endpoint, row in by_endpoint.items()
        },
        "strict_pair": {
            "energy_delta_eV": strict_energy_delta,
            "indexed_mic_rmsd_A": strict_rmsd,
            "descriptor_delta": strict_descriptor_delta,
            "energy_tol_eV": corpus["protocol"]["energy_tol_eV"],
            "rmsd_tol_A": corpus["protocol"]["rmsd_tol_A"],
            "descriptor_tol": corpus["protocol"]["descriptor_tol"],
        },
        "gate": gate,
    }
    (RUN_ROOT / "strict_requench_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PdO strict re-quench result",
        "",
        f"- Classification: **{gate['classification']}**.",
        f"- Offline first-passage label: **{gate['offline_label']}**.",
        f"- New force evaluations: **{evidence['new_force_evaluations']}**.",
        f"- GPU wall time: **{evidence['gpu_wall_time_s']:.6f} s**.",
        (
            "- Strict endpoint ΔE / MIC RMSD / descriptor Δ: "
            f"**{strict_energy_delta:.6f} eV / {strict_rmsd:.6f} Å / "
            f"{strict_descriptor_delta:.6f}**."
        ),
        (
            "- SciPy reported optimizer success for the landing, but the "
            f"independent raw-force certificate was "
            f"**{converged['landing']}**."
        ),
        "- Production matcher or default change authorized: **False**.",
        "",
    ]
    (RUN_ROOT / "strict_requench_conclusion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return evidence


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
