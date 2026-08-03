#!/usr/bin/env python3
"""Analyze the bounded PdO strict-certificate rescue closure."""

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
OUTPUT_DIR = RUN_ROOT / "certificate-rescue-output"


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
    "_pdo_matcher_certificate_protocol_analysis",
)
runner = _load_module(
    RUN_ROOT / "run_certificate_rescue.py",
    "_pdo_matcher_certificate_runner_analysis",
)
matcher_analyzer = _load_module(
    RUN_ROOT / "analyze.py",
    "_pdo_matcher_closure_analysis",
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


def _final_row(record: Mapping[str, Any]) -> Mapping[str, Any]:
    fallback = record["fallback"]
    return record["primary"] if fallback is None else fallback


def analyze() -> dict[str, Any]:
    from pamssw.archive import MinimaArchive
    from pamssw.fingerprint import descriptor_distance, structural_descriptor

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    summary_path = OUTPUT_DIR / "summary.json"
    records_path = OUTPUT_DIR / "records.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = json.loads(records_path.read_text(encoding="utf-8"))
    by_endpoint = {record["endpoint"]: record for record in records}
    if set(by_endpoint) != {"starter", "landing"}:
        raise RuntimeError("certificate-rescue endpoints are incomplete")
    row_total = sum(
        int(record["primary"]["force_evaluations"])
        + (
            0
            if record["fallback"] is None
            else int(record["fallback"]["force_evaluations"])
        )
        for record in records
    )
    if summary["total_force_evaluations"] != row_total:
        raise RuntimeError("certificate-rescue summary accounting is open")

    finals = {
        endpoint: _final_row(record)
        for endpoint, record in by_endpoint.items()
    }
    states = {
        endpoint: _state(row["final"]["state"])
        for endpoint, row in finals.items()
    }
    energy_delta = float(
        finals["landing"]["final"]["energy_eV"]
        - finals["starter"]["final"]["energy_eV"]
    )
    rmsd = float(
        MinimaArchive._rmsd(states["starter"], states["landing"])
    )
    descriptor_delta = descriptor_distance(
        structural_descriptor(states["starter"]),
        structural_descriptor(states["landing"]),
    )
    fmax = float(summary["protocol"]["fmax_eV_per_A"])
    certified = {
        endpoint: runner.has_strict_certificate(row, fmax=fmax)
        for endpoint, row in finals.items()
    }
    gate = protocol.evaluate_strict_requench(
        starter_converged=certified["starter"],
        landing_converged=certified["landing"],
        strict_energy_delta_eV=energy_delta,
        energy_tol_eV=float(corpus["protocol"]["energy_tol_eV"]),
        strict_indexed_mic_rmsd_A=rmsd,
        rmsd_tol_A=float(corpus["protocol"]["rmsd_tol_A"]),
    )
    matcher_evidence_path = RUN_ROOT / "evidence.json"
    matcher_evidence = json.loads(
        matcher_evidence_path.read_text(encoding="utf-8")
    )
    raw_path = REPO_ROOT / matcher_evidence["raw_evidence_path"]
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    relabel_basis = {
        "global_geometry_descriptor_collisions": sum(
            row["mechanism"] == "descriptor_collision_geometry_split"
            for row in matcher_evidence["pairs"]
        ),
        "strict_requench_certified_splits": int(
            gate["offline_label"] == "ESCAPED_CERTIFIED"
        ),
    }
    relabel_all = (
        relabel_basis["global_geometry_descriptor_collisions"] == 4
        and relabel_basis["strict_requench_certified_splits"] == 1
    )
    counterfactual_first_passage = (
        matcher_analyzer._counterfactual_first_passage(
            raw,
            relabel=relabel_all,
        )
    )
    endpoint_rows: dict[str, Any] = {}
    for endpoint, record in by_endpoint.items():
        final = finals[endpoint]
        endpoint_rows[endpoint] = {
            "certified": certified[endpoint],
            "final_stage": record["final_stage"],
            "primary": {
                "force_evaluations": record["primary"]["force_evaluations"],
                "wall_time_s": record["primary"]["wall_time_s"],
                "termination_reason": record["primary"][
                    "termination_reason"
                ],
                "final_max_force_eV_per_A": record["primary"]["final"][
                    "max_active_force_eV_per_A"
                ],
            },
            "fallback": (
                None
                if record["fallback"] is None
                else {
                    "force_evaluations": record["fallback"][
                        "force_evaluations"
                    ],
                    "wall_time_s": record["fallback"]["wall_time_s"],
                    "termination_reason": record["fallback"][
                        "termination_reason"
                    ],
                    "final_max_force_eV_per_A": record["fallback"][
                        "final"
                    ]["max_active_force_eV_per_A"],
                }
            ),
            "final_energy_eV": final["final"]["energy_eV"],
            "final_max_force_eV_per_A": final["final"][
                "max_active_force_eV_per_A"
            ],
        }
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
        "fallback_count": int(summary["fallback_count"]),
        "endpoints": endpoint_rows,
        "strict_pair": {
            "energy_delta_eV": energy_delta,
            "indexed_mic_rmsd_A": rmsd,
            "descriptor_delta": descriptor_delta,
            "energy_tol_eV": corpus["protocol"]["energy_tol_eV"],
            "rmsd_tol_A": corpus["protocol"]["rmsd_tol_A"],
            "descriptor_tol": corpus["protocol"]["descriptor_tol"],
        },
        "gate": gate,
        "offline_first_passage_closure": {
            "all_five_ambiguities_resolved_as_escape": relabel_all,
            "relabel_basis": relabel_basis,
            "counterfactual": counterfactual_first_passage,
        },
        "production_change_allowed": False,
    }
    evidence_path = RUN_ROOT / "certificate_rescue_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# PdO certificate-rescue closure",
        "",
        f"- Classification: **{gate['classification']}**.",
        f"- Offline first-passage label: **{gate['offline_label']}**.",
        f"- New force evaluations: **{evidence['new_force_evaluations']}**.",
        f"- GPU wall time: **{evidence['gpu_wall_time_s']:.6f} s**.",
        f"- Fallbacks triggered: **{evidence['fallback_count']} / 2**.",
        (
            "- Final endpoint ΔE / MIC RMSD / descriptor Δ: "
            f"**{energy_delta:.6f} eV / {rmsd:.6f} Å / "
            f"{descriptor_delta:.6f}**."
        ),
        (
            "- All five original PdO ambiguities resolved offline as escape: "
            f"**{relabel_all}**."
        ),
        (
            "- Closed first-passage labels (escape / return / invalid): "
            f"**{counterfactual_first_passage['label_counts'].get('ESCAPED_CERTIFIED', 0)} / "
            f"{counterfactual_first_passage['label_counts'].get('RETURN_STARTER', 0)} / "
            f"{counterfactual_first_passage['label_counts'].get('INVALID_GEOMETRY', 0)}**."
        ),
        (
            "- Remaining H8-return or action-support-gap contexts: "
            f"**{len(counterfactual_first_passage['horizon_gate_contexts'])} / "
            f"{len(counterfactual_first_passage['action_support_gap_contexts'])}**."
        ),
        "- Production matcher or optimizer change authorized: **False**.",
        "",
    ]
    (RUN_ROOT / "certificate_rescue_conclusion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return evidence


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
