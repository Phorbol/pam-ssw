#!/usr/bin/env python3
"""Zero-PES same-index edge-set trace for saved C60 climb centers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import read


HERE = Path(__file__).resolve().parent
PANEL = HERE.parent
RUN_ROOT = PANEL / "runs-1756515" / "c60"
METHODS = ("full", "local_only", "feedback_only")
SEEDS = (26101231, 26101232, 26101233, 26101234)
CUTOFFS_A = (1.64, 1.70, 1.80)


def edge_set(positions: np.ndarray, cutoff: float) -> set[tuple[int, int]]:
    delta = positions[:, None, :] - positions[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    i, j = np.where(np.triu((d2 < cutoff * cutoff) & (d2 > 0.0), 1))
    return set(zip(i.tolist(), j.tolist()))


def snapshot(label: str, positions: np.ndarray, initial_edges: dict[float, set]):
    rows = {}
    for cutoff in CUTOFFS_A:
        edges = edge_set(positions, cutoff)
        base = initial_edges[cutoff]
        rows[f"{cutoff:.2f}"] = {
            "edge_count": len(edges),
            "same_index_edges_as_initial": edges == base,
            "symmetric_difference_edge_count": len(edges ^ base),
        }
    return {"label": label, "cutoffs": rows}


def analyze_slot(method: str, seed: int):
    slot = RUN_ROOT / method / f"seed-{seed}"
    result_path = slot / "result.json"
    if not result_path.is_file():
        return {"method": method, "seed": seed, "status": "missing_result"}
    result = json.loads(result_path.read_text())
    structure_path = slot / "structures.extxyz"
    if not structure_path.is_file():
        return {"method": method, "seed": seed, "status": result.get("status"),
                "error": "missing structures.extxyz"}
    frames = read(structure_path, index=":")
    initial = next((a for a in frames if a.info.get("role") == "initial"), None)
    landing = next((a for a in frames if a.info.get("role") == "landing"), None)
    if initial is None:
        return {"method": method, "seed": seed, "status": result.get("status"),
                "error": "missing initial role"}
    if len(initial) != 60 or set(initial.get_chemical_symbols()) != {"C"}:
        return {"method": method, "seed": seed, "status": result.get("status"),
                "error": "unexpected initial composition"}
    initial_positions = np.asarray(initial.positions, dtype=float)
    base_edges = {c: edge_set(initial_positions, c) for c in CUTOFFS_A}
    timeline = [snapshot("initial", initial_positions, base_edges)]
    event = next((ev for ev in result.get("outer_events", []) if ev.get("kind") == "outer_step"), None)
    climbs = [] if event is None else event.get("climb") or []
    for row in climbs:
        center = row.get("center")
        if center is None:
            continue
        positions = np.asarray(center, dtype=float)
        if positions.shape != initial_positions.shape:
            continue
        timeline.append(snapshot(f"gaussian_center_{row.get('index')}", positions, base_edges))
    if landing is not None:
        timeline.append(snapshot("saved_landing", np.asarray(landing.positions, dtype=float), base_edges))

    by_cutoff = {}
    for c in CUTOFFS_A:
        key = f"{c:.2f}"
        changed_centers = [
            row["label"] for row in timeline
            if row["label"].startswith("gaussian_center_")
            and not row["cutoffs"][key]["same_index_edges_as_initial"]
        ]
        by_cutoff[key] = {
            "changed_center_count": len(changed_centers),
            "first_changed_center": changed_centers[0] if changed_centers else None,
            "any_saved_center_changed_edges": bool(changed_centers),
            "saved_landing_same_index_edges_as_initial": (
                next((r["cutoffs"][key]["same_index_edges_as_initial"] for r in timeline
                      if r["label"] == "saved_landing"), None)),
        }
    return {
        "method": method, "seed": seed, "execution_status": result.get("status"),
        "result_error": result.get("error"),
        "search_requests": result.get("search_requests"),
        "saved_climb_center_count": sum(r["label"].startswith("gaussian_center_") for r in timeline),
        "timeline": timeline, "summary_by_cutoff": by_cutoff,
        "scope_note": "center geometry is a saved Gaussian center, not the unsaved terminal biased-quench endpoint",
    }


def main():
    rows = [analyze_slot(method, seed) for method in METHODS for seed in SEEDS]
    payload = {
        "analysis": "zero-PES same-index C-C edge-set comparison at saved climb centers",
        "input_run_root": str(RUN_ROOT.relative_to(PANEL)),
        "cutoffs_A": list(CUTOFFS_A),
        "edge_rule": "nonperiodic Euclidean C-C pair distance strictly below cutoff; i<j, no atom permutation",
        "additional_pes_calls": 0,
        "limitations": [
            "A saved Gaussian center is not the generally unsaved last biased endpoint.",
            "A return to initial edges at landing does not establish whether the biased endpoint left and the true quench reversed it.",
            "Same-index edge sets are diagnostics, not basin or topology proofs.",
        ],
        "slots": rows,
    }
    out = HERE / "classification.json"
    with out.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    changed = sum(any(v["any_saved_center_changed_edges"] for v in row.get("summary_by_cutoff", {}).values())
                  for row in rows)
    print(json.dumps({"output": str(out), "slots": len(rows),
                      "slots_with_any_changed_center_at_any_cutoff": changed,
                      "additional_pes_calls": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
