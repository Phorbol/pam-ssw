#!/usr/bin/env python3
"""Zero-PES geometry readout of saved C60 full-direction/CBD probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read


ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925"
PROBE = EVIDENCE / "direction-probe/runs"
PATH_RUN = ROOT / "research/ga_ssw/evidence/c60-path-diagnostic-20260927/run-1504112"
HESSIAN = EVIDENCE / "curvature/isomer-2.npz"


def projected_cosine(vector: np.ndarray, target: np.ndarray, basis: np.ndarray) -> dict:
    v = basis.T @ np.asarray(vector, dtype=float).reshape(-1)
    t = basis.T @ np.asarray(target, dtype=float).reshape(-1)
    nv, nt = float(np.linalg.norm(v)), float(np.linalg.norm(t))
    signed = None if nv == 0 or nt == 0 else float(np.dot(v, t) / (nv * nt))
    return {"signed_cosine": signed, "unsigned_cosine": None if signed is None else abs(signed),
            "internal_vector_norm": nv}


def analyze() -> dict:
    hessian = np.load(HESSIAN)
    x0 = hessian["positions"]
    basis = hessian["internal_basis"]
    path_images = read(PATH_RUN / "path.extxyz", index=":")
    path0 = np.asarray(path_images[0].positions, dtype=float)
    path1 = np.asarray(path_images[1].positions, dtype=float)
    if not np.allclose(path0, x0, rtol=0.0, atol=1e-8):
        raise RuntimeError("NEB R0 and saved Hessian coordinates/order do not match")
    target = path1 - path0
    reference_norm = float(np.linalg.norm(basis.T @ target.reshape(-1)))
    if reference_norm == 0:
        raise RuntimeError("NEB first segment has zero internal displacement")

    cases = []
    for arm in ("ssw_without_ls-1101", "ssw_without_ls-1102",
                "native_ls-1101", "native_ls-1102"):
        result = json.loads((PROBE / arm / "result.json").read_text())
        summary = json.loads((PROBE / arm / "summary.json").read_text())
        checks = {int(row["role"].split("-")[-1]): row
                  for row in summary.get("checks", []) if str(row.get("role", "")).startswith("landing-")}
        current = np.asarray(result["initial"]["atoms"]["positions"], dtype=float)
        if not np.allclose(current, x0, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"{arm} initial minimum is not in Hessian atom order")
        for record in result["records"]:
            idx = int(record["index"])
            climb = record.get("climb") or []
            if climb:
                first = climb[0]
                center0 = np.asarray(first["center"], dtype=float)
                recovered = first.get("recovered_direction") or {}
                proposal = recovered.get("proposal")
                cbd_direction = first.get("direction")
                center_matches = np.allclose(center0, x0, rtol=0.0, atol=1e-8)
                initial_matches = np.allclose(current, x0, rtol=0.0, atol=1e-8)
                prep_shift = center0 - current
                entry = {
                    "arm": arm, "outer_index": idx,
                    "current_matches_hessian_R0_in_same_atom_order": bool(initial_matches),
                    "post_prequench_center_matches_R0_in_same_atom_order": bool(center_matches),
                    "prequench_shift_norm_A": float(np.linalg.norm(prep_shift)),
                    "prequench_shift_projection": projected_cosine(prep_shift, target, basis),
                    "status": record.get("status"), "accepted": bool(record.get("accepted")),
                    "ih_graph_match": bool(checks.get(idx, {}).get("graphs", {}).get("1.64", {}).get("ih_graph_match", False)),
                    "comparison_note": ("same R0 and orientation/order" if initial_matches and center_matches else
                                        "softened or changed starting geometry; direction cosine is descriptive, not matched-state evidence"),
                }
                if proposal is not None:
                    entry["pre_CBD_proposal_vs_NEb_R1_minus_R0"] = projected_cosine(np.asarray(proposal), target, basis)
                if cbd_direction is not None:
                    entry["post_CBD_direction_vs_NEb_R1_minus_R0"] = projected_cosine(np.asarray(cbd_direction), target, basis)
                if len(climb) > 1:
                    next_center = np.asarray(climb[1]["center"], dtype=float)
                    quench_displacement = next_center - center0
                    entry["after_first_biased_quench_displacement"] = {
                        "norm_A": float(np.linalg.norm(quench_displacement)),
                        **projected_cosine(quench_displacement, target, basis),
                    }
                if record.get("landing") is not None:
                    landing = np.asarray(record["landing"]["atoms"]["positions"], dtype=float)
                    net = landing - center0
                    entry["true_landing_net_displacement"] = {
                        "norm_A": float(np.linalg.norm(net)), **projected_cosine(net, target, basis)
                    }
                cases.append(entry)
            if record.get("accepted") and record.get("landing") is not None:
                current = np.asarray(record["landing"]["atoms"]["positions"], dtype=float)

    return {
        "scope": "saved-trajectory coordinate comparison only; no PES or model calls",
        "reference": {
            "run": str(PATH_RUN),
            "vector": "CI-NEB image 1 minus image 0, projected into the saved 174D internal basis",
            "initial_position_max_abs_difference_A": float(np.max(np.abs(path0 - x0))),
            "internal_norm_A": reference_norm,
        },
        "cases": cases,
        "interpretation_limits": [
            "R1-R0 is a finite CI-NEB segment, not an exact reaction tangent or a known SSW target direction.",
            "Direction cosines are unsigned only when explicitly labeled; actual coordinate displacements retain sign.",
            "Post-LS CBD directions are evaluated after a shifted geometry and are not directly matched to the unsoftened Hessian state.",
            "The full-direction runs vary selection, local/global mixture, within-climb continuation, CBD rotation, and (in LS arms) softening together; these records do not isolate one mechanism.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="new JSON output path; source run artifacts remain read-only")
    args = parser.parse_args()
    result = analyze()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_count": len(result["cases"]),
                      "reference_internal_norm_A": result["reference"]["internal_norm_A"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
