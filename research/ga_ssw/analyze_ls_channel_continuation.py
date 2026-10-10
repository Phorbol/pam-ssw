#!/usr/bin/env python3
"""Offline summary and fixed-W response estimates for an LS continuation run.

This reads saved JSON, structures, and Hessian arrays only. It never creates a
calculator or evaluates the physical potential.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from ase.io import read

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pamssw.standalone.softening import FrozenBondSoftening


def frozen_softening(atoms, saved):
    return FrozenBondSoftening(
        tuple(map(int, atoms.numbers)), tuple(map(tuple, atoms.cell.array)),
        tuple(map(bool, atoms.pbc)), tuple(map(tuple, saved["pairs"])),
        tuple(map(float, saved["reference_distances_A"])),
        tuple(map(float, saved["strengths_eV"])), xi=float(saved["xi"]))


def fixed_w_response(run_dir: Path, row: dict, gauche, soft, atom_count: int):
    amplitude = float(row["amplitude"])
    min_record = row.get("phases", {}).get("minimum", {})
    if not (min_record.get("qualified_stationarity") and min_record.get("qualified_hessian")):
        return {"status": "unavailable_minimum_not_qualified"}

    stem = f"a-{amplitude:g}-minimum"
    hessian_path = run_dir / f"{stem}-hessian.npz"
    geometry_path = run_dir / f"{stem}.extxyz"
    if not hessian_path.is_file() or not geometry_path.is_file():
        return {"status": "unavailable_saved_minimum_or_hessian_missing",
                "hessian_path": str(hessian_path), "geometry_path": str(geometry_path)}

    minimum = read(geometry_path, index=0)
    if not np.array_equal(minimum.numbers, gauche.numbers):
        return {"status": "unavailable_atom_order_mismatch"}
    with np.load(hessian_path) as data:
        qbasis = np.asarray(data["Q"], dtype=float)
        hessian = np.asarray(data["H_F_h0005"], dtype=float)
    if hessian.shape != (qbasis.shape[1], qbasis.shape[1]):
        return {"status": "unavailable_hessian_shape_mismatch",
                "Q_shape": list(qbasis.shape), "H_shape": list(hessian.shape)}

    _, force_w = soft.evaluate(minimum)
    gq = -qbasis.T @ force_w.ravel()
    try:
        chi = float(gq @ np.linalg.solve(hessian, gq))
    except np.linalg.LinAlgError:
        return {"status": "unavailable_hessian_solve_failed"}
    if not np.isfinite(chi) or chi <= 0:
        return {"status": "unavailable_nonpositive_or_nonfinite_chi",
                "chi_eV": chi}
    s0 = float(sum(soft.strengths))
    return {
        "status": "computed_conditional_fixed_W_estimate",
        "formula": "gq=-Q.T F_W; chi=gq.T solve(H_F_h0005,gq); Pprime=a*chi/N; eta_crit_fixed_W=2*S0/(a*chi)",
        "amplitude": amplitude, "atom_count": atom_count,
        "gq_eV_A": gq.tolist(), "chi_eV": chi,
        "Pprime_per_atom_eV_per_amplitude": amplitude * chi / atom_count,
        "S0_eV": s0,
        "eta_crit_fixed_W": 2.0 * s0 / (amplitude * chi),
        "hessian_path": str(hessian_path),
        "hessian_field": "H_F_h0005",
        "interpretation_limit": "conditional scalar update under fixed W and local harmonic response; not a rebuilt-controller stability test"}


def summarize_row(run_dir: Path, row: dict, gauche, soft, original_baseline: dict):
    phases = row.get("phases", {})
    phase_summary = {}
    for name, record in phases.items():
        phase_summary[name] = {
            key: record.get(key) for key in (
                "status", "qualified_stationarity", "qualified_hessian", "optimizer",
                "certificate", "hessian", "requests", "requests_optimization_and_certificate",
                "requests_including_hessian", "requests_total_start", "requests_total_end", "error")
            if key in record}
    min_record = phases.get("minimum", {})
    min_v = min_record.get("certificate", {}).get("V_eV")
    loading = None
    if min_v is not None and original_baseline.get("V_eV") is not None:
        loading = (float(min_v) - float(original_baseline["V_eV"])) / len(gauche)
    return {
        "amplitude": row.get("amplitude"), "status": row.get("status"),
        "error": row.get("error"), "requests": row.get("requests"),
        "phases": phase_summary,
        "stationary_values": row.get("stationary_values"),
        "physical_loading_per_atom_eV": row.get("physical_loading_per_atom_eV", loading),
        "biased_downhill": row.get("biased_downhill"),
        "physical_release_of_biased_minimum": row.get("physical_release_of_biased_minimum"),
        "fixed_W_response": fixed_w_response(run_dir, row, gauche, soft, len(gauche))}


def analyze(input_result: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    data = json.loads(input_result.read_text())
    run_dir = input_result.resolve().parent
    gauche_path = Path(data["inputs"]["gauche"])
    if not gauche_path.is_file():
        raise FileNotFoundError(gauche_path)
    gauche = read(gauche_path, index=0)
    saved_w = data["frozen_W"]
    soft = frozen_softening(gauche, saved_w)
    rows = [summarize_row(run_dir, row, gauche, soft, data.get("original_gauche_baseline", {}))
            for row in data.get("rows", [])]
    report = {
        "source_result": str(input_result.resolve()),
        "source_run_status": data.get("status"),
        "source_requests": data.get("requests"),
        "source_actual_calculate_calls": data.get("actual_calculate_calls"),
        "source_denials": data.get("denials"),
        "original_gauche_baseline": data.get("original_gauche_baseline"),
        "frozen_W": {"origin": saved_w.get("origin"), "xi": saved_w.get("xi"),
                     "initial_fraction": saved_w.get("initial_fraction"),
                     "bond_lengths_rule": saved_w.get("bond_lengths_rule"),
                     "strengths_eV": saved_w.get("strengths_eV"),
                     "S0_eV": float(sum(soft.strengths))},
        "connectivity_validated": False,
        "rows": rows,
        "interpretation_limit": "offline stationary-branch summary; endpoint distances, graphs and energies are reported as recorded, with no automatic basin identity threshold; fixed-W eta is not actual controller stability"}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = analyze(args.input_result, args.output)
    print(json.dumps({"source_run_status": report["source_run_status"],
                      "rows": len(report["rows"]), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
