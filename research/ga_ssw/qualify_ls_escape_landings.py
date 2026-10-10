#!/usr/bin/env python3
"""Qualify saved T3 C4H6 landings as stationary points and diagnose identity.

This is a post-search geometry qualification only. It does not evaluate SSW
performance or decide basin identity from graph/RMSD diagnostics.
"""
from __future__ import annotations

import argparse
import importlib.metadata
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
from scipy.linalg import null_space
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = ROOT / "research/ga_ssw/qualify_ls_torsion_channel.py"
RING_TOOLS_PATH = ROOT / "research/ga_ssw/probe_ls_ring_channel.py"
RELEASE_TOOLS_PATH = ROOT / "research/ga_ssw/release_ls_channel_endpoints.py"
REQUEST_CAP = 6000
WALL_SECONDS = 600
FORCE_TOL = 3e-4
GTOL = 1e-4
MAXITER = 150
HESSIAN_STEPS = (0.001, 0.0005)
HESSIAN_MARGIN = 5.0


def import_file(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_run(run_dir: Path):
    result_path = run_dir / "summary.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    data = json.loads(result_path.read_text())
    if data.get("status") != "all_24_arms_recorded" or len(data.get("runs", [])) != 24:
        raise ValueError("T3 input must contain all 24 recorded arms")
    refs = {row["label"]: row for row in data["references"]}
    required = {"gauche", "trans", "cyclobutene"}
    if not required.issubset(refs):
        raise ValueError(f"T3 result lacks reference records: {required - set(refs)}")
    return data, refs


def mapped_reference_diagnostics(candidate, references, ring_tools, qualifier):
    candidate_graph = ring_tools.graph(candidate, qualifier.HC_BOND_LENGTHS)
    comparisons = {}
    for label, reference in references.items():
        ref_atoms = reference["atoms"]
        ref_graph = ring_tools.graph(ref_atoms, qualifier.HC_BOND_LENGTHS)
        try:
            matches = ring_tools.best_graph_mapping(
                ref_atoms, candidate, ref_graph, candidate_graph)
            rmsd, mapping, inverse, aligned, rotation, translation = matches[0]
            comparisons[label] = {
                "reference_path": reference["path"],
                "reference_energy_eV": reference.get("energy_eV"),
                "graph_isomorphic_with_element_and_C_H_adjacency": True,
                "best_proper_kabsch_rmsd_A": float(rmsd),
                "candidate_to_reference_mapping": [int(x) for x in mapping],
                "candidate_reorder_to_reference_indices": [int(x) for x in inverse],
                "alignment_rotation": np.asarray(rotation).tolist(),
                "alignment_translation_A": np.asarray(translation).tolist(),
            }
        except ValueError as error:
            comparisons[label] = {
                "reference_path": reference["path"],
                "reference_energy_eV": reference.get("energy_eV"),
                "graph_isomorphic_with_element_and_C_H_adjacency": False,
                "best_proper_kabsch_rmsd_A": None,
                "mapping_error": str(error),
            }
    edges = [[int(i), int(j)] for i, j in zip(*np.where(np.triu(candidate_graph, 1)))]
    return {"graph_edges": edges, "degree_sequence": sorted(candidate_graph.sum(axis=0).astype(int).tolist()),
            "reference_comparisons": comparisons,
            "identity_limit": "graph mapping and proper-Kabsch RMSD are diagnostics; no identity threshold"}


def hessian_at(geometry, q, qbasis, surface):
    n_q = qbasis.shape[1]
    matrices = []
    asymmetry = []
    for step in HESSIAN_STEPS:
        columns = []
        for axis in np.eye(n_q):
            _, f_plus = surface.evaluate(geometry(q + step * axis))
            _, f_minus = surface.evaluate(geometry(q - step * axis))
            columns.append(-qbasis.T @ (f_plus - f_minus).ravel() / (2.0 * step))
        raw = np.column_stack(columns)
        asymmetry.append(float(np.linalg.norm(raw - raw.T, 2)))
        matrices.append((raw + raw.T) / 2.0)
    spread = float(np.linalg.norm(matrices[0] - matrices[1], 2))
    return matrices, {"steps_A": list(HESSIAN_STEPS), "stencil_spread_eV_A2": spread,
                      "antisymmetric_norms_eV_A2": asymmetry}


def run(run_dir: Path, out: Path):
    run_dir = run_dir.resolve()
    out = out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output: {out}")
    if not out.parent.is_dir():
        raise FileNotFoundError(out.parent)
    t3, reference_records = load_run(run_dir)
    qualifier = import_file("ls_escape_qualifier", QUALIFIER_PATH)
    ring_tools = import_file("ls_escape_ring_tools", RING_TOOLS_PATH)
    release_tools = import_file("ls_escape_release_tools", RELEASE_TOOLS_PATH)
    utilities = qualifier.load_ledger()

    references = {}
    for label, record in reference_records.items():
        path = Path(record["path"])
        atoms = read(path, index=0)
        release_tools.validate_isolated_c4h6(atoms, label)
        references[label] = {**record, "atoms": atoms}

    endpoints = []
    for row in t3["runs"]:
        candidate_path = run_dir / row["run_id"] / "candidate.extxyz"
        fresh = row.get("candidate_fresh_check") or {}
        endpoints.append({
            "run_id": row["run_id"], "start": row["start"], "seed": row["seed"],
            "arm": row["arm"], "candidate_path": str(candidate_path),
            "candidate_sha256": qualifier.sha256(candidate_path) if candidate_path.is_file() else None,
            "t3_reported_fresh_energy_eV": fresh.get("energy_eV"),
            "t3_reported_fresh_fmax_eV_A": fresh.get("fmax_eV_A"),
            "t3_reported_force_qualified": fresh.get("force_qualified"),
            "status": "not_started" if candidate_path.is_file() else "missing_candidate_geometry",
        })

    out.mkdir(parents=False, exist_ok=False)
    shutil.copy2(Path(__file__).resolve(), out / "qualify_ls_escape_landings.py")
    result = {
        "status": "started", "t3_run": str(run_dir),
        "t3_result_sha256": qualifier.sha256(run_dir / "summary.json"),
        "t3_status": t3["status"],
        "references": [{key: value for key, value in ref.items() if key != "atoms"}
                       for ref in references.values()],
        "method": {"coordinate_chart": "fixed affine Eckart internal-coordinate section",
                   "optimizer": "physical-V BFGS in Q coordinates", "gtol": GTOL,
                   "maxiter_per_endpoint": MAXITER, "force_tolerance_eV_A": FORCE_TOL,
                   "hessian": "central physical-force differences, symmetrized per stencil",
                   "hessian_steps_A": list(HESSIAN_STEPS),
                   "positive_gate": "lambda_min > 5 * spectral_norm(H_0.001 - H_0.0005)"},
        "limits": {"request_cap_total": REQUEST_CAP,
                   "wall_seconds_from_surface_creation": WALL_SECONDS,
                   "endpoints": len(endpoints)},
        "identity_decided": False,
        "endpoints": endpoints,
        "model": {},
        "source_hashes": {"runner": qualifier.sha256(Path(__file__).resolve()),
                           "qualifier": qualifier.sha256(QUALIFIER_PATH),
                           "ring_tools": qualifier.sha256(RING_TOOLS_PATH),
                           "release_tools": qualifier.sha256(RELEASE_TOOLS_PATH)},
    }
    for label, ref in references.items():
        result["source_hashes"][f"{label}_geometry"] = qualifier.sha256(Path(ref["path"]))
        source_result = ref.get("result_path")
        if source_result:
            result["source_hashes"][f"{label}_result"] = qualifier.sha256(Path(source_result))
    utilities.dump(out / "result.json", result)

    started = time.monotonic()
    surface = None
    actual = {"calls": 0}
    calc = None
    try:
        if not qualifier.MODEL.is_file():
            raise FileNotFoundError(qualifier.MODEL)
        model_hash = qualifier.sha256(qualifier.MODEL)
        if model_hash != qualifier.MODEL_SHA256:
            raise ValueError(f"MH-1 model hash mismatch: {model_hash}")
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from mace.calculators import MACECalculator
        calc = MACECalculator(model_paths=str(qualifier.MODEL), head="omol", device="cuda",
                              default_dtype="float64", enable_cueq=False, enable_oeq=False)
        actual = utilities.instrument_calculate(calc)
        surface = utilities.CountedSurface(calc, out / "requests.jsonl", REQUEST_CAP, WALL_SECONDS)
        result["model"] = {"path": str(qualifier.MODEL), "sha256": model_hash,
                           "head": "omol", "device": "cuda", "dtype": "float64",
                           "torch": torch.__version__, "torch_num_threads": torch.get_num_threads(),
                           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                           "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cuda_device_name": torch.cuda.get_device_name(0),
                           "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
                           "ase": ase_version, "mace_torch": importlib.metadata.version("mace-torch"),
                           "python": sys.version, "platform": platform.platform()}
        result["source_hashes"]["model"] = model_hash
        utilities.dump(out / "result.json", result)

        for index, entry in enumerate(endpoints):
            candidate_path = Path(entry["candidate_path"])
            if not candidate_path.is_file():
                continue
            if surface.denials:
                entry["status"] = "not_attempted_after_budget_stop"
                continue
            entry["requests_start"] = int(surface.requests)
            entry["calculate_calls_start"] = int(actual["calls"])
            try:
                start = read(candidate_path, index=0)
                release_tools.validate_isolated_c4h6(start, entry["run_id"])
                write(out / f"{entry['run_id']}-start.extxyz", start)
                frame, qbasis = qualifier.chart_for(start)
                if qbasis.shape != (30, 24):
                    raise RuntimeError(f"unexpected Q dimensions {qbasis.shape}")
                geometry, objective = release_tools.objective_factory(start, frame, qbasis, surface)
                opt = minimize(objective, np.zeros(qbasis.shape[1]), jac=True, method="BFGS",
                               options={"gtol": GTOL, "maxiter": MAXITER})
                final = geometry(opt.x)
                energy, forces = surface.evaluate(final)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                force_ok = fmax <= FORCE_TOL
                write(out / f"{entry['run_id']}-physical.extxyz", final)
                entry.update({"status": "completed_force_unqualified" if not force_ok else "force_qualified",
                              "optimizer": {"success_flag": bool(opt.success), "message": str(opt.message),
                                            "iterations": int(opt.nit), "function_calls": int(opt.nfev)},
                              "certificate": {"energy_eV": float(energy), "fmax_eV_A": fmax,
                                              "force_qualified": bool(force_ok)},
                              "Q_shape": list(qbasis.shape),
                              "graph_and_reference_diagnostics": mapped_reference_diagnostics(
                                  final, references, ring_tools, qualifier)})
                for comparison in entry["graph_and_reference_diagnostics"]["reference_comparisons"].values():
                    reference_energy = comparison.get("reference_energy_eV")
                    comparison["energy_difference_to_reference_eV"] = (
                        float(energy) - reference_energy if reference_energy is not None else None)
                entry["refinement_proper_rmsd_A"] = float(
                    ring_tools.proper_kabsch(final.positions, start.positions)[3])
                if force_ok:
                    hessians, hinfo = hessian_at(geometry, np.asarray(opt.x, float), qbasis, surface)
                    eigens = [np.linalg.eigvalsh(hessian) for hessian in hessians]
                    spread = hinfo["stencil_spread_eV_A2"]
                    positive = bool(float(eigens[1][0]) > HESSIAN_MARGIN * spread)
                    entry["hessian"] = {**hinfo,
                                         "eigenvalues_eV_A2_by_step": [values.tolist() for values in eigens],
                                         "smallest_eigenvalue_eV_A2_by_step": [float(values[0]) for values in eigens],
                                         "stencil_spread_eV_A2": spread,
                                         "positive_definite_qualified": positive,
                                         "negative_curvature_observed": bool(float(eigens[1][0]) < -HESSIAN_MARGIN * spread),
                                         "interpretation": "negative curvature remains unqualified as a minimum; unresolved signs fail the positive gate"}
                    entry["hessian_qualified"] = positive
                    np.savez_compressed(out / f"{entry['run_id']}-hessian.npz",
                                        Q=qbasis, H_h001=hessians[0], H_h0005=hessians[1],
                                        eig_h001=eigens[0], eig_h0005=eigens[1],
                                        hessian_steps_A=np.asarray(HESSIAN_STEPS),
                                        stencil_spread_eV_A2=np.asarray(spread))
                    entry["qualification_status"] = ("positive_hessian_qualified_minimum" if positive
                                                      else "force_qualified_but_hessian_not_positive_qualified")
                else:
                    entry["qualification_status"] = "force_unqualified_no_hessian"
                entry["t3_recheck"] = {
                    "prior_fresh_energy_eV": entry.get("t3_reported_fresh_energy_eV"),
                    "physical_relaxed_energy_minus_t3_fresh_eV": (
                        float(energy) - entry["t3_reported_fresh_energy_eV"]
                        if entry.get("t3_reported_fresh_energy_eV") is not None else None),
                    "prior_fresh_fmax_eV_A": entry.get("t3_reported_fresh_fmax_eV_A")}
            except Exception as error:
                entry.update(status="endpoint_error", error=repr(error))
            finally:
                entry["requests_end"] = int(surface.requests)
                entry["requests"] = int(surface.requests - entry["requests_start"])
                entry["actual_calculate_calls"] = int(actual["calls"] - entry["calculate_calls_start"])
                entry["actual_calculate_calls_total"] = int(actual["calls"])
                entry["counted_wall_seconds"] = float(time.monotonic() - surface.started)
                utilities.dump(out / "result.json", result)
            if surface.denials:
                result["status"] = "stopped_at_budget_boundary"
                result["budget_boundary"] = surface.boundary
                for pending in endpoints[index + 1:]:
                    pending["status"] = "not_attempted_after_budget_stop"
                    pending["requests_start"] = pending["requests_end"] = int(surface.requests)
                    pending["requests"] = 0
                break
        else:
            diagnostic_statuses = {"force_qualified", "completed_force_unqualified", "endpoint_error"}
            if all(entry["status"] in diagnostic_statuses for entry in endpoints):
                result["status"] = ("completed_all_endpoint_diagnostics" if all(
                    entry.get("qualification_status") == "positive_hessian_qualified_minimum"
                    for entry in endpoints) else "completed_with_unqualified_or_failed_endpoints")
            else:
                result["status"] = "completed_with_missing_endpoints"
    except Exception as error:
        result["status"] = "setup_failed_or_stopped"
        result["error"] = repr(error)
    finally:
        result.update(requests=int(surface.requests) if surface is not None else 0,
                      actual_calculate_calls=int(actual["calls"]),
                      denials=int(surface.denials) if surface is not None else 0,
                      budget_boundary=surface.boundary if surface is not None else None,
                      counted_wall_seconds=(time.monotonic() - surface.started) if surface is not None else 0.0,
                      total_elapsed_seconds=time.monotonic() - started)
        utilities.dump(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("t3_run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = run(args.t3_run, args.out)
    print(json.dumps({"status": result.get("status"), "endpoints": len(result.get("endpoints", [])),
                      "requests": result.get("requests"), "out": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
