#!/usr/bin/env python3
"""Physically relax five saved LS-biased C4H6 endpoints and compare references.

The script diagnoses endpoint geometry/energy correspondence. RMSD and graph
values are reported without an automatic basin-identity decision.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from ase import __version__ as ase_version
from ase.io import read, write
import scipy
from scipy.linalg import null_space
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = ROOT / "research/ga_ssw/qualify_ls_torsion_channel.py"
RING_TOOLS_PATH = ROOT / "research/ga_ssw/probe_ls_ring_channel.py"
ENDPOINTS = (
    (16.0, "easy_ts", "minus"),
    (16.0, "easy_ts", "plus"),
    (16.0, "ring_ts", "minus"),
    (16.0, "ring_ts", "plus"),
    (8.0, "easy_ts", "plus"),
)
REQUEST_CAP = 1000
WALL_SECONDS = 300
FORCE_TOL = 3e-4
GTOL = 1e-4
MAXITER = 200


def import_file(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_geometry(base: Path, filename: str) -> Path:
    candidates = (base / filename, base / "qualification" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"missing {filename} under {base}")


def find_result(base: Path) -> Path | None:
    for candidate in (base / "result.json", base / "qualification" / "result.json"):
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_reference(label: str, base: Path, filename: str, phase_name: str):
    geometry_path = find_geometry(base, filename)
    atoms = read(geometry_path, index=0)
    record_path = find_result(base)
    reference_energy = None
    energy_source = None
    if record_path is not None:
        data = json.loads(record_path.read_text())
        phase = data.get("phases", {}).get(phase_name, {})
        value = phase.get("minimum_energy_eV", phase.get("energy_eV"))
        if value is not None:
            reference_energy = float(value)
            energy_source = str(record_path)
    return {"label": label, "path": str(geometry_path), "atoms": atoms,
            "energy_eV": reference_energy, "energy_source": energy_source}


def validate_isolated_c4h6(atoms, label: str):
    expected = np.array([6] * 4 + [1] * 6)
    if not np.array_equal(atoms.numbers, expected):
        raise ValueError(f"{label}: expected fixed C0..C3,H0..H5 atom numbering")
    if atoms.constraints or np.any(atoms.pbc) or not np.isfinite(atoms.positions).all():
        raise ValueError(f"{label}: expected finite isolated unconstrained C4H6")


def graph_edges(atoms, bond_lengths, margin=0.1):
    edges = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            pair = tuple(sorted((int(atoms.numbers[i]), int(atoms.numbers[j]))))
            if atoms.get_distance(i, j, mic=False) <= float(bond_lengths[pair]) + margin:
                edges.append([i, j])
    return edges


def objective_factory(start, frame, qbasis, surface):
    def geometry(q):
        atoms = start.copy()
        atoms.positions = frame.positions(
            frame.reference + (qbasis @ np.asarray(q, dtype=float)).reshape((-1, 3)))
        return atoms

    def objective(q):
        atoms = geometry(q)
        energy, forces = surface.evaluate(atoms)
        return energy, -qbasis.T @ forces.ravel()

    return geometry, objective


def main_run(finite_run: Path, torsion_ref: Path, ring_ref: Path, out: Path):
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    finite_run = finite_run.resolve()
    if not (finite_run / "result.json").is_file():
        raise FileNotFoundError(finite_run / "result.json")
    if not torsion_ref.is_dir() or not ring_ref.is_dir():
        raise NotADirectoryError("reference inputs must be directories")

    finite_data = json.loads((finite_run / "result.json").read_text())
    if finite_data.get("status") != "completed_stationary_branch_diagnostics":
        raise ValueError(f"finite run is not complete: {finite_data.get('status')}")
    finite_rows = {float(row["amplitude"]): row for row in finite_data.get("rows", [])}
    for amplitude in (8.0, 16.0):
        row = finite_rows.get(amplitude)
        if row is None or row.get("status") != "qualified_stationary_branches_and_downhill_diagnostics":
            raise ValueError(f"a={amplitude:g} row missing or not qualified")

    gauche = load_reference("torsion_gauche", torsion_ref, "plus-0.050.extxyz", "plus-0.050")
    trans = load_reference("torsion_trans", torsion_ref, "minus-0.050.extxyz", "minus-0.050")
    cyclobutene = load_reference("cyclobutene", ring_ref, "minus-0.050.extxyz", "minus-0.050")
    refs = (gauche, trans, cyclobutene)
    for ref in refs:
        validate_isolated_c4h6(ref["atoms"], ref["label"])

    out.mkdir(parents=True, exist_ok=False)
    utilities = import_file("release_ls_qualifier", QUALIFIER_PATH).load_ledger()
    ring_tools = import_file("release_ls_ring_tools", RING_TOOLS_PATH)
    result = {
        "status": "started", "finite_run": str(finite_run),
        "finite_run_status": finite_data.get("status"),
        "references": [{key: value for key, value in ref.items() if key != "atoms"}
                       for ref in refs],
        "model": {}, "limits": {"request_cap_total": REQUEST_CAP,
                                  "wall_seconds_from_surface_creation": WALL_SECONDS,
                                  "force_tolerance_eV_A": FORCE_TOL,
                                  "optimizer": "projected physical-V BFGS", "gtol": GTOL,
                                  "maxiter_per_endpoint": MAXITER},
        "graph_rule": "same-index HC_BOND_LENGTHS + 0.1 A; includes all C-C and C-H adjacency",
        "basin_identity_decided": False,
        "endpoints": [{"label": f"a-{amplitude:g}-{branch}-{sign}",
                       "amplitude": amplitude, "branch": branch, "sign": sign,
                       "start_path": str(finite_run / f"a-{amplitude:g}-{branch}-{sign}-landing.extxyz"),
                       "status": "not_started"} for amplitude, branch, sign in ENDPOINTS]}
    utilities.dump(out / "result.json", result)
    source_script = Path(__file__).resolve()
    shutil.copy2(source_script, out / "runner.py")
    actual = {"calls": 0}
    surface = None
    start_wall = time.monotonic()
    try:
        qualifier = import_file("release_ls_qualifier_constants", QUALIFIER_PATH)
        observed_hash = qualifier.sha256(qualifier.MODEL)
        if observed_hash != qualifier.MODEL_SHA256:
            raise ValueError(f"MH-1 hash mismatch: {observed_hash}")
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from mace.calculators import MACECalculator
        calculator = MACECalculator(model_paths=str(qualifier.MODEL), head="omol", device="cuda",
                                    default_dtype="float64", enable_cueq=False, enable_oeq=False)
        actual = utilities.instrument_calculate(calculator)
        surface = utilities.CountedSurface(calculator, out / "requests.jsonl",
                                           cap=REQUEST_CAP, wall=WALL_SECONDS)
        result["model"] = {"path": str(qualifier.MODEL), "sha256": observed_hash,
                           "expected_sha256": qualifier.MODEL_SHA256, "head": "omol",
                           "device": "cuda", "dtype": "float64", "torch": torch.__version__,
                           "numpy": np.__version__, "python": sys.version,
                           "platform": platform.platform(), "torch_num_threads": torch.get_num_threads(),
                           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                           "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cuda_device_name": torch.cuda.get_device_name(0)}
        result["model"].update({"ase": ase_version, "scipy": scipy.__version__})
        result["source_hashes"] = {"runner": qualifier.sha256(source_script),
                                   "qualifier": qualifier.sha256(QUALIFIER_PATH),
                                   "ring_tools": qualifier.sha256(RING_TOOLS_PATH),
                                   "finite_run_result": qualifier.sha256(finite_run / "result.json"),
                                   "torsion_reference_geometry": qualifier.sha256(Path(gauche["path"])),
                                   "trans_reference_geometry": qualifier.sha256(Path(trans["path"])),
                                   "cyclobutene_reference_geometry": qualifier.sha256(Path(cyclobutene["path"]))}
        for ref in refs:
            if ref["energy_source"]:
                result["source_hashes"][f"{ref['label']}_energy_record"] = qualifier.sha256(Path(ref["energy_source"]))
        utilities.dump(out / "result.json", result)

        for endpoint_index, (amplitude, branch, sign) in enumerate(ENDPOINTS):
            label = f"a-{amplitude:g}-{branch}-{sign}"
            entry = result["endpoints"][endpoint_index]
            start_path = Path(entry["start_path"])
            entry.update({"status": "started", "requests_start": int(surface.requests)})
            utilities.dump(out / "result.json", result)
            try:
                start = read(start_path, index=0)
                entry["start_sha256"] = qualifier.sha256(start_path)
                validate_isolated_c4h6(start, label)
                write(out / f"{label}-start.extxyz", start)
                frame = qualifier.ClusterFrame(start)
                qbasis = null_space(frame.basis.T)
                if qbasis.shape != (3 * len(start), 3 * len(start) - 6):
                    raise RuntimeError(f"unexpected Eckart basis shape {qbasis.shape}")
                geometry, objective = objective_factory(start, frame, qbasis, surface)
                opt = minimize(objective, np.zeros(qbasis.shape[1]), jac=True, method="BFGS",
                               options={"gtol": GTOL, "maxiter": MAXITER})
                minimum = geometry(opt.x)
                energy, forces = surface.evaluate(minimum)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                write(out / f"{label}-physical-minimum.extxyz", minimum)
                end_graph = graph_edges(minimum, qualifier.HC_BOND_LENGTHS)
                comparisons = {}
                for ref in refs:
                    aligned, _, _, rmsd = ring_tools.proper_kabsch(minimum.positions,
                                                                   ref["atoms"].positions)
                    ref_graph = graph_edges(ref["atoms"], qualifier.HC_BOND_LENGTHS)
                    comparisons[ref["label"]] = {
                        "reference_path": ref["path"],
                        "reference_energy_eV": ref["energy_eV"],
                        "reference_energy_source": ref["energy_source"],
                        "physical_energy_difference_to_reference_eV": (
                            float(energy) - ref["energy_eV"] if ref["energy_eV"] is not None else None),
                        "same_index_proper_kabsch_rmsd_A": float(rmsd),
                        "same_index_graph_equal": end_graph == ref_graph,
                        "endpoint_graph_edges": end_graph,
                        "reference_graph_edges": ref_graph,
                        "alignment_is_diagnostic_only": True}
                entry.update({"status": "completed_diagnostics", "optimizer": {
                                  "success_flag": bool(opt.success), "message": str(opt.message),
                                  "iterations": int(opt.nit), "function_calls": int(opt.nfev)},
                              "certificate": {"energy_eV": float(energy), "fmax_eV_A": fmax,
                                              "force_qualified": bool(fmax <= FORCE_TOL)},
                              "basis_shape": list(qbasis.shape), "graph_edges": end_graph,
                              "reference_comparisons": comparisons})
                if fmax > FORCE_TOL:
                    entry["status"] = "completed_force_unqualified"
            except Exception as error:
                entry.update({"status": "failed", "error": repr(error)})
                result["status"] = "stopped_at_endpoint_error"
                result["stop_reason"] = repr(error)
                utilities.dump(out / "result.json", result)
            finally:
                entry["requests_end"] = int(surface.requests)
                entry["requests"] = int(surface.requests - entry["requests_start"])
                entry["counted_wall_seconds"] = float(time.monotonic() - surface.started)
                entry["actual_calculate_calls_total"] = int(actual["calls"])
                utilities.dump(out / "result.json", result)
            if surface.denials:
                result["status"] = "stopped_at_budget_boundary"
                result["request_boundary"] = surface.boundary
                for pending in result["endpoints"][endpoint_index + 1:]:
                    pending["status"] = "not_attempted_after_budget_stop"
                    pending["requests_start"] = pending["requests_end"] = int(surface.requests)
                    pending["requests"] = 0
                break
        else:
            result["status"] = ("completed_five_endpoint_diagnostics" if all(
                entry.get("status") == "completed_diagnostics" for entry in result["endpoints"])
                else "completed_with_endpoint_errors_or_force_failures")
    except Exception as error:
        result["status"] = "stopped"
        result["error"] = repr(error)
    finally:
        if surface is None:
            for entry in result["endpoints"]:
                if entry.get("status") == "not_started":
                    entry["status"] = "not_attempted_setup_failure"
        result.update(requests=int(surface.requests) if surface is not None else 0,
                      actual_calculate_calls=int(actual["calls"]),
                      denials=int(surface.denials) if surface is not None else 0,
                      request_boundary=surface.boundary if surface is not None else None,
                      counted_wall_seconds=(time.monotonic() - surface.started) if surface is not None else 0.0,
                      total_wall_seconds=time.monotonic() - start_wall)
        utilities.dump(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("finite_run", type=Path)
    parser.add_argument("torsion_reference_run", type=Path)
    parser.add_argument("ring_qualification_run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = main_run(args.finite_run, args.torsion_reference_run,
                      args.ring_qualification_run, args.out)
    print(json.dumps({"status": result.get("status"), "requests": result.get("requests"),
                      "out": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
