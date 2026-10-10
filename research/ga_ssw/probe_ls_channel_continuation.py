#!/usr/bin/env python
"""Continue one fixed LS minimum and two qualified C4H6 saddles in amplitude.

This is a bounded stationary-branch diagnostic on MH-1/omol. It is not a
search controller and does not certify endpoint connectivity or a reaction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.linalg import null_space
from scipy.optimize import minimize, root
from pamssw.standalone.cluster_frame import ClusterFrame


ROOT = Path(__file__).resolve().parents[2]
QUALIFIER_PATH = ROOT / "research/ga_ssw/qualify_ls_torsion_channel.py"
RING_RUNNER_PATH = ROOT / "research/ga_ssw/probe_ls_ring_channel.py"
RESPONSE_PATH = ROOT / "research/ga_ssw/probe_ls_response.py"
AMPLITUDES = (1.0, 2.0, 4.0, 8.0, 16.0)
REQUEST_CAP = 7000
WALL_SECONDS = 900
FORCE_TOL = 3e-4
HESSIAN_STEPS = (0.001, 0.0005)
HESSIAN_MARGIN = 5.0
DOWNHILL_STEP = 0.05


def import_file(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_file(base: Path, filename: str) -> Path:
    candidates = (base / filename, base / "qualification" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not find {filename} in {base} or {base / 'qualification'}")


def q_chart(reference, frame, qbasis, q):
    atoms = reference.copy()
    atoms.positions = frame.positions(frame.reference + (qbasis @ np.asarray(q)).reshape((-1, 3)))
    return atoms


def align_seed(seed, reference, ring_tools, frame, qbasis):
    if not np.array_equal(seed.numbers, reference.numbers):
        raise ValueError("branch seed atom numbers/order differ from gauche reference")
    if seed.constraints or np.any(seed.pbc):
        raise ValueError("branch seeds must be isolated and unconstrained")
    aligned, rotation, translation, rmsd = ring_tools.proper_kabsch(
        seed.positions, reference.positions)
    displacement = aligned - frame.reference
    q = qbasis.T @ displacement.ravel()
    reconstructed = q_chart(reference, frame, qbasis, q)
    reconstruction_max_A = float(np.max(np.linalg.norm(reconstructed.positions - aligned, axis=1)))
    if reconstruction_max_A > 1e-8:
        raise ValueError(f"Q chart reconstruction changed aligned geometry: max={reconstruction_max_A:.3e} A")
    aligned_atoms = seed.copy()
    aligned_atoms.positions = aligned
    aligned_atoms.cell = reference.cell
    aligned_atoms.pbc = reference.pbc
    return q, aligned_atoms, {"proper_rotation": rotation, "translation_A": translation,
                              "aligned_rmsd_to_gauche_A": rmsd,
                              "Q_reconstruction_max_change_A": reconstruction_max_A}


def hc_graph(atoms, lengths, margin=0.1):
    edges = []
    for i in range(len(atoms) - 1):
        for j in range(i + 1, len(atoms)):
            key = tuple(sorted((int(atoms.numbers[i]), int(atoms.numbers[j]))))
            if atoms.get_distance(i, j, mic=False) <= float(lengths[key]) + margin:
                edges.append([i, j, key[0], key[1]])
    return edges


def optimize_minimum(q0, amplitude, geometry, evaluate, qbasis, soft):
    def objective(q):
        atoms = geometry(q)
        energy, forces = evaluate(atoms)
        bias, bias_forces = soft.evaluate(atoms)
        total_force = forces + amplitude * bias_forces
        return energy + amplitude * bias, -qbasis.T @ total_force.ravel()

    result = minimize(objective, q0, jac=True, method="BFGS",
                      options={"gtol": 1e-4, "maxiter": 300})
    return np.asarray(result.x), result


def main_run(torsion_run: Path, ring_run: Path, out: Path, *,
             amplitudes=AMPLITUDES, bias_geometry="pair",
             request_cap=REQUEST_CAP, wall_seconds=WALL_SECONDS):
    if bias_geometry not in ("pair", "shape"):
        raise ValueError("unknown research bias geometry")
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    qualifier = import_file("ls_continuation_qualifier", QUALIFIER_PATH)
    ring_tools = import_file("ls_continuation_ring_tools", RING_RUNNER_PATH)
    response_tools = import_file("ls_continuation_response_tools", RESPONSE_PATH)
    utilities = qualifier.load_ledger()

    gauche_path = source_file(torsion_run, "plus-0.050.extxyz")
    easy_ts_path = source_file(torsion_run, "ts-candidate.extxyz")
    ring_ts_path = source_file(ring_run, "ts-candidate.extxyz")
    ring_qual_dir = ring_ts_path.parent
    ring_gauche_path = ring_qual_dir / "plus-0.050.extxyz"
    if not ring_gauche_path.is_file():
        raise FileNotFoundError(ring_gauche_path)

    result = {"status": "started", "inputs": {
        "torsion_run": str(torsion_run.resolve()), "ring_qualification_run": str(ring_run.resolve()),
        "gauche": str(gauche_path), "easy_ts": str(easy_ts_path), "ring_ts": str(ring_ts_path),
        "ring_gauche_release_for_provenance": str(ring_gauche_path)},
        "bias_geometry": bias_geometry,
        "limits": {"requests": request_cap, "counted_wall_seconds": wall_seconds,
                   "force_tolerance_eV_A": FORCE_TOL, "amplitudes": amplitudes,
                   "hessian_steps_A": HESSIAN_STEPS, "hessian_margin_factor": HESSIAN_MARGIN,
                   "negative_mode_displacement_A": DOWNHILL_STEP},
        "connectivity_validated": False, "rows": [],
        "interpretation_limit": "fixed-W stationary-branch and local downhill diagnostics only; no connection, basin identity, finite-temperature behavior, or LS controller effectiveness is certified"}
    utilities.dump(out / "result.json", result)
    start_wall = time.monotonic()
    surface = None
    actual = {"calls": 0}
    try:
        gauche = read(gauche_path, index=0)
        easy_ts = read(easy_ts_path, index=0)
        ring_ts = read(ring_ts_path, index=0)
        if len(gauche) != 10 or not np.array_equal(gauche.numbers, np.array([6] * 4 + [1] * 6)):
            raise ValueError("gauche reference must use the archived C4H6 C0..C3,H0..H5 order")
        if gauche.constraints or np.any(gauche.pbc):
            raise ValueError("gauche reference must be isolated and unconstrained")

        frame = ClusterFrame(gauche)
        qbasis = null_space(frame.basis.T)
        q_min = np.zeros(qbasis.shape[1])
        q_easy, easy_aligned, easy_alignment = align_seed(easy_ts, gauche, ring_tools, frame, qbasis)
        q_ring, ring_aligned, ring_alignment = align_seed(ring_ts, gauche, ring_tools, frame, qbasis)
        write(out / "gauche-reference.extxyz", gauche)
        write(out / "easy-ts-aligned.extxyz", easy_aligned)
        write(out / "ring-ts-aligned.extxyz", ring_aligned)
        result["chart"] = {"kind": "one fixed gauche-anchored Eckart chart",
                            "Q_shape": qbasis.shape,
                            "Q_orthonormal_error": float(np.linalg.norm(qbasis.T @ qbasis - np.eye(qbasis.shape[1]), 2)),
                            "easy_ts_alignment": easy_alignment, "ring_ts_alignment": ring_alignment,
                            "reconstruction_tolerance_A": 1e-8}
        result["source_hashes"] = {name: qualifier.sha256(path) for name, path in
                                   (("gauche", gauche_path), ("easy_ts", easy_ts_path), ("ring_ts", ring_ts_path),
                                    ("qualifier", QUALIFIER_PATH), ("ring_helper", RING_RUNNER_PATH),
                                    ("response_helper", RESPONSE_PATH))}

        soft = qualifier.FrozenBondSoftening.from_atoms(
            gauche, bond_energies=qualifier.HC_BOND_ENERGIES,
            bond_lengths={key: value + 0.1 for key, value in qualifier.HC_BOND_LENGTHS.items()},
            initial_fraction=0.03, xi=0.2)
        result["frozen_W"] = {"origin": str(gauche_path), "pairs": soft.pairs,
                              "reference_distances_A": soft.reference_distances,
                              "strengths_eV": soft.strengths, "xi": soft.xi,
                              "initial_fraction": 0.03, "bond_lengths_rule": "HC table + 0.1 A"}
        if bias_geometry == "shape":
            shape_module_path = ROOT / "research/ga_ssw/shape_ls_probe.py"
            shape_tools = import_file("shape_ls_probe", shape_module_path)
            soft = shape_tools.FrozenShapeBias(soft, gauche)
            result["source_hashes"]["shape_bias"] = qualifier.sha256(shape_module_path)
            result["frozen_W"]["shape_reference_radius_A"] = soft.radius

        def cartesian_bias_hessian(atoms):
            if bias_geometry == "shape":
                return soft.hessian(atoms)
            radial, transverse = response_tools.stiffness(soft, atoms)
            return radial + transverse

        model_hash = qualifier.sha256(qualifier.MODEL)
        if model_hash != qualifier.MODEL_SHA256:
            raise ValueError(f"MH-1 model hash mismatch: {model_hash}")
        import torch
        from ase import __version__ as ase_version
        from mace.calculators import MACECalculator
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        calculator = MACECalculator(model_paths=str(qualifier.MODEL), head="omol", device="cuda",
                                    default_dtype="float64", enable_cueq=False, enable_oeq=False)
        actual = utilities.instrument_calculate(calculator)
        surface = utilities.CountedSurface(calculator, out / "requests.jsonl",
                                           cap=request_cap, wall=wall_seconds)
        result["model"] = {"path": str(qualifier.MODEL), "sha256": model_hash,
                           "head": "omol", "device": "cuda", "dtype": "float64",
                           "torch": torch.__version__, "ase": ase_version,
                           "numpy": np.__version__, "scipy": qualifier.importlib.metadata.version("scipy"),
                           "python": sys.version, "platform": platform.platform(),
                           "torch_num_threads": torch.get_num_threads(),
                           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                           "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                           "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                           "cuda_device_name": torch.cuda.get_device_name(0)}
        result["chart"].update({"coordinates": "q=Q.T(x-x_gauche); all points share fixed Q",
                                "dimension": int(qbasis.shape[1])})
        baseline_request_start = surface.requests
        gauche_v, gauche_force = surface.evaluate(gauche)
        gauche_w, _ = soft.evaluate(gauche)
        original_gauche_baseline = {
            "V_eV": float(gauche_v), "W_eV": float(gauche_w),
            "F_at_a0_eV": float(gauche_v),
            "physical_fmax_eV_A": float(np.linalg.norm(gauche_force, axis=1).max()),
            "requests_total_start": int(baseline_request_start),
            "requests_total_end": int(surface.requests),
            "requests": int(surface.requests - baseline_request_start)}
        result["original_gauche_baseline"] = original_gauche_baseline
        utilities.dump(out / "result.json", result)

        def geometry(q):
            return q_chart(gauche, frame, qbasis, q)

        def evaluate(atoms):
            return surface.evaluate(atoms)

        def total_force(atoms, amplitude):
            _, force_v = evaluate(atoms)
            _, force_w = soft.evaluate(atoms)
            return force_v + amplitude * force_w

        def certificate(q, amplitude):
            atoms = geometry(q)
            energy_v, force_v = evaluate(atoms)
            energy_w, force_w = soft.evaluate(atoms)
            force_f = force_v + amplitude * force_w
            return atoms, {"V_eV": float(energy_v), "W_eV": float(energy_w),
                           "F_eV": float(energy_v + amplitude * energy_w),
                           "physical_fmax_eV_A": float(np.linalg.norm(force_v, axis=1).max()),
                           "total_fmax_eV_A": float(np.linalg.norm(force_f, axis=1).max()),
                           "force_qualified": bool(np.linalg.norm(force_f, axis=1).max() <= FORCE_TOL)}

        def total_hessian(q, amplitude):
            physical_mats, antisymmetry = [], []
            for step in HESSIAN_STEPS:
                columns = []
                for axis in np.eye(qbasis.shape[1]):
                    _, fplus = evaluate(geometry(q + step * axis))
                    _, fminus = evaluate(geometry(q - step * axis))
                    columns.append(-qbasis.T @ (fplus - fminus).ravel() / (2 * step))
                raw = np.column_stack(columns)
                antisymmetry.append(float(np.linalg.norm(raw - raw.T, 2)))
                physical_mats.append((raw + raw.T) / 2)
            atoms = geometry(q)
            bias_cart = cartesian_bias_hessian(atoms)
            bias_q = qbasis.T @ bias_cart @ qbasis
            total_mats = [matrix + amplitude * bias_q for matrix in physical_mats]
            spread = float(np.linalg.norm(total_mats[0] - total_mats[1], 2))
            values, vectors = np.linalg.eigh(total_mats[-1])
            return {"physical_matrices": physical_mats, "bias_cartesian": bias_cart,
                    "bias_q": bias_q, "total_matrices": total_mats,
                    "spread": spread, "antisymmetry": antisymmetry,
                    "eigenvalues": values, "eigenvectors": vectors}

        def bias_hessian_q(q):
            atoms = geometry(q)
            return qbasis.T @ cartesian_bias_hessian(atoms) @ qbasis

        def save_hessian(path, hs):
            np.savez(path, H_V_h001=hs["physical_matrices"][0],
                     H_V_h0005=hs["physical_matrices"][1], K_W_cartesian=hs["bias_cartesian"],
                     K_W_q=hs["bias_q"], H_F_h001=hs["total_matrices"][0],
                     H_F_h0005=hs["total_matrices"][1], Q=qbasis,
                     eigenvalues=hs["eigenvalues"], eigenvectors=hs["eigenvectors"])

        def stationary_record(name, q, amplitude, kind):
            phase_start = surface.requests
            if kind == "minimum":
                q_opt, opt = optimize_minimum(q, amplitude, geometry, evaluate, qbasis, soft)
                optimizer = {"success_flag": bool(opt.success), "message": str(opt.message),
                             "iterations": int(opt.nit), "function_calls": int(opt.nfev)}
            else:
                def gradient(x):
                    return -qbasis.T @ total_force(geometry(x), amplitude).ravel()
                def jacobian(x):
                    columns = []
                    step = HESSIAN_STEPS[0]
                    for axis in np.eye(qbasis.shape[1]):
                        _, fplus = evaluate(geometry(x + step * axis))
                        _, fminus = evaluate(geometry(x - step * axis))
                        columns.append(-qbasis.T @ (fplus - fminus).ravel() / (2 * step))
                    return np.column_stack(columns) + amplitude * bias_hessian_q(x)
                opt = root(gradient, q, jac=jacobian, method="hybr", options={"maxfev": 200})
                q_opt = np.asarray(opt.x)
                optimizer = {"success_flag": bool(opt.success), "status_code": int(opt.status),
                             "message": str(opt.message), "function_calls": int(opt.nfev),
                             "jacobian": "physical central force differences h=0.001 A + exact a*K_W",
                             "jacobian_calls": int(getattr(opt, "njev", -1)),
                             "root_residual_inf_eV_A": float(np.max(np.abs(opt.fun))),
                             "root_residual_qualified": bool(np.max(np.abs(opt.fun)) <= FORCE_TOL)}
            atoms = geometry(q_opt)
            try:
                atoms, cert = certificate(q_opt, amplitude)
            except Exception as error:
                write(out / f"a-{amplitude:g}-{name}.extxyz", atoms)
                return q_opt, atoms, {"name": name, "kind": kind, "optimizer": optimizer,
                                      "q": q_opt.tolist(), "status": "certificate_failed",
                                      "error": repr(error), "qualified_stationarity": False,
                                      "requests_total_start": int(phase_start),
                                      "requests_total_end": int(surface.requests),
                                      "requests": int(surface.requests - phase_start)}
            write(out / f"a-{amplitude:g}-{name}.extxyz", atoms)
            record = {"name": name, "kind": kind, "optimizer": optimizer,
                      "certificate": cert, "requests_optimization_and_certificate": int(surface.requests - phase_start),
                      "requests_total_start": int(phase_start),
                      "requests_total_end": int(surface.requests),
                      "q": q_opt.tolist(), "hessian": None}
            if not cert["force_qualified"] or (kind == "saddle" and not optimizer["root_residual_qualified"]):
                record["qualified_stationarity"] = False
                return q_opt, atoms, record
            try:
                hs = total_hessian(q_opt, amplitude)
            except Exception as error:
                record.update({"qualified_stationarity": True, "qualified_hessian": False,
                               "hessian": {"status": "failed", "error": repr(error)},
                               "requests_total_end": int(surface.requests),
                               "requests_including_hessian": int(surface.requests - phase_start)})
                return q_opt, atoms, record
            save_hessian(out / f"a-{amplitude:g}-{name}-hessian.npz", hs)
            values = hs["eigenvalues"]
            if kind == "minimum":
                hessian_ok = bool(values[0] > HESSIAN_MARGIN * hs["spread"])
            else:
                hessian_ok = bool(values[0] < -HESSIAN_MARGIN * hs["spread"] and
                                  values[1] > HESSIAN_MARGIN * hs["spread"])
            record["hessian"] = {"steps_A": HESSIAN_STEPS,
                                 "stencil_spread_eV_A2": hs["spread"],
                                 "antisymmetric_norms_eV_A2": hs["antisymmetry"],
                                 "eigenvalues_eV_A2": values.tolist(),
                                 "negative_mode_count": int(np.count_nonzero(values < -HESSIAN_MARGIN * hs["spread"])),
                                 "positive_mode_count": int(np.count_nonzero(values > HESSIAN_MARGIN * hs["spread"])),
                                 "near_zero_or_unresolved_count": int(np.count_nonzero(
                                     np.abs(values) <= HESSIAN_MARGIN * hs["spread"])),
                                 "qualified": hessian_ok,
                                 "criterion": ("minimum: lambda_min > 5*spread" if kind == "minimum" else
                                               "saddle: lambda0 < -5*spread and lambda1 > 5*spread")}
            record["qualified_stationarity"] = True
            record["qualified_hessian"] = hessian_ok
            record["negative_mode_q"] = hs["eigenvectors"][:, 0].tolist() if kind == "saddle" else None
            record["requests_including_hessian"] = int(surface.requests - phase_start)
            record["requests_total_end"] = int(surface.requests)
            return q_opt, atoms, record

        def record_energy_and_slope(point_record, minimum_record):
            point = point_record["certificate"]
            minimum = minimum_record["certificate"]
            return {"V_eV": point["V_eV"], "W_eV": point["W_eV"], "F_eV": point["F_eV"],
                    "barrier_F_eV": point["F_eV"] - minimum["F_eV"],
                    "physical_energy_difference_between_biased_stationary_points_eV": point["V_eV"] - minimum["V_eV"],
                    "exact_dB_da_W_TS_minus_W_min_eV": point["W_eV"] - minimum["W_eV"]}

        previous = {"minimum": q_min, "easy_ts": q_easy, "ring_ts": q_ring}
        for amplitude in amplitudes:
            request_start = surface.requests
            row = {"amplitude": amplitude, "status": "started", "phases": {}}
            result["rows"].append(row)
            utilities.dump(out / "result.json", result)
            try:
                points = {}
                for name, kind in (("minimum", "minimum"), ("easy_ts", "saddle"), ("ring_ts", "saddle")):
                    phase_start = surface.requests
                    q0 = previous[name]
                    try:
                        q_new, atoms, record = stationary_record(name, q0, amplitude, kind)
                    except Exception as error:
                        row["phases"][name] = {"name": name, "kind": kind, "status": "failed",
                                               "error": repr(error),
                                               "requests_total_start": int(phase_start),
                                               "requests_total_end": int(surface.requests),
                                               "requests": int(surface.requests - phase_start)}
                        utilities.dump(out / "result.json", result)
                        raise
                    row["phases"][name] = record
                    points[name] = (q_new, atoms, record)
                    write(out / f"a-{amplitude:g}-{name}.extxyz", atoms)
                    utilities.dump(out / "result.json", result)
                    if not record.get("qualified_stationarity") or not record.get("qualified_hessian"):
                        raise RuntimeError(f"{name} qualification failed at a={amplitude:g}")

                minrec = points["minimum"][2]
                easyrec, ringrec = points["easy_ts"][2], points["ring_ts"][2]
                easy_barrier = record_energy_and_slope(easyrec, minrec)
                ring_barrier = record_energy_and_slope(ringrec, minrec)
                row["stationary_values"] = {"minimum": minrec["certificate"],
                                             "easy_ts": easy_barrier, "ring_ts": ring_barrier,
                                             "barrier_gap_ring_minus_easy_F_eV": ring_barrier["barrier_F_eV"] - easy_barrier["barrier_F_eV"],
                                             "exact_gap_slope_W_eV": ring_barrier["exact_dB_da_W_TS_minus_W_min_eV"] - easy_barrier["exact_dB_da_W_TS_minus_W_min_eV"]}
                row["physical_loading_per_atom_eV"] = (
                    minrec["certificate"]["V_eV"] - original_gauche_baseline["V_eV"]
                ) / len(gauche)
                row["biased_downhill"] = {}
                for ts_name in ("easy_ts", "ring_ts"):
                    q_ts, _, ts_record = points[ts_name]
                    mode = np.asarray(ts_record["negative_mode_q"], dtype=float)
                    side_rows = {}
                    for sign_name, sign in (("minus", -1.0), ("plus", 1.0)):
                        phase_start = surface.requests
                        try:
                            q_start = q_ts + sign * DOWNHILL_STEP * mode
                            q_end, opt = optimize_minimum(q_start, amplitude, geometry, evaluate, qbasis, soft)
                            landing, cert = certificate(q_end, amplitude)
                            write(out / f"a-{amplitude:g}-{ts_name}-{sign_name}-landing.extxyz", landing)
                            _, _, _, rmsd = ring_tools.proper_kabsch(
                                landing.positions, points["minimum"][1].positions)
                            min_graph = hc_graph(points["minimum"][1], qualifier.HC_BOND_LENGTHS)
                            landing_graph = hc_graph(landing, qualifier.HC_BOND_LENGTHS)
                            side = {"optimizer_success_flag": bool(opt.success), "optimizer_message": str(opt.message),
                                    "optimizer_iterations": int(opt.nit), "certificate": cert,
                                    "downhill_force_qualified": bool(cert["force_qualified"]),
                                    "same_numbering_proper_Kabsch_rmsd_to_current_min_A": rmsd,
                                    "biased_total_energy_difference_to_current_min_eV": cert["F_eV"] - minrec["certificate"]["F_eV"],
                                    "physical_energy_difference_to_current_min_eV": cert["V_eV"] - minrec["certificate"]["V_eV"],
                                    "HC_graph_edges": landing_graph,
                                    "HC_graph_same_as_current_min": landing_graph == min_graph,
                                    "requests_total_start": int(phase_start),
                                    "requests_total_end": int(surface.requests),
                                    "requests": int(surface.requests - phase_start)}
                        except Exception as error:
                            side_rows[sign_name] = {"status": "failed", "error": repr(error),
                                                    "requests_total_start": int(phase_start),
                                                    "requests_total_end": int(surface.requests),
                                                    "requests": int(surface.requests - phase_start)}
                            row["biased_downhill"][ts_name] = side_rows
                            utilities.dump(out / "result.json", result)
                            raise
                        side_rows[sign_name] = side
                        row["biased_downhill"][ts_name] = side_rows
                        utilities.dump(out / "result.json", result)
                        if not cert["force_qualified"]:
                            raise RuntimeError(f"{ts_name} biased downhill force gate failed ({sign_name}, a={amplitude:g})")
                    row["biased_downhill"][ts_name] = side_rows

                phase_start = surface.requests
                try:
                    q_debiased, opt_debiased = optimize_minimum(
                        points["minimum"][0], 0.0, geometry, evaluate, qbasis, soft)
                    released, released_cert = certificate(q_debiased, 0.0)
                    write(out / f"a-{amplitude:g}-minimum-physical-release.extxyz", released)
                    _, _, _, release_rmsd = ring_tools.proper_kabsch(released.positions, gauche.positions)
                    row["physical_release_of_biased_minimum"] = {
                        "optimizer_success_flag": bool(opt_debiased.success),
                        "optimizer_message": str(opt_debiased.message),
                        "optimizer_iterations": int(opt_debiased.nit), "certificate": released_cert,
                        "same_numbering_proper_Kabsch_rmsd_to_original_gauche_A": release_rmsd,
                        "physical_energy_difference_to_original_gauche_eV": released_cert["V_eV"] - original_gauche_baseline["V_eV"],
                        "HC_graph_edges": hc_graph(released, qualifier.HC_BOND_LENGTHS),
                        "requests_total_start": int(phase_start), "requests_total_end": int(surface.requests),
                        "requests": int(surface.requests - phase_start)}
                except Exception as error:
                    row["physical_release_of_biased_minimum"] = {
                        "status": "failed", "error": repr(error),
                        "requests_total_start": int(phase_start), "requests_total_end": int(surface.requests),
                        "requests": int(surface.requests - phase_start)}
                    utilities.dump(out / "result.json", result)
                    raise
                if not released_cert["force_qualified"]:
                    raise RuntimeError(f"physical minimum release force gate failed at a={amplitude:g}")
                row["requests"] = int(surface.requests - request_start)
                row["status"] = "qualified_stationary_branches_and_downhill_diagnostics"
                previous = {name: points[name][0] for name in ("minimum", "easy_ts", "ring_ts")}
                utilities.dump(out / "result.json", result)
            except Exception as error:
                row["status"] = "stopped"
                row["error"] = repr(error)
                row["requests"] = int(surface.requests - request_start)
                result["status"] = "stopped_at_amplitude"
                utilities.dump(out / "result.json", result)
                break
        else:
            result["status"] = "completed_stationary_branch_diagnostics"
    except Exception as error:
        result["status"] = "stopped"
        result["error"] = repr(error)
    finally:
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
    parser.add_argument("torsion_run", type=Path)
    parser.add_argument("ring_qualification_run", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = main_run(args.torsion_run, args.ring_qualification_run, args.out)
    print(json.dumps({"status": result.get("status"), "out": str(args.out.resolve()),
                      "requests": result.get("requests"), "error": result.get("error")}, indent=2))


if __name__ == "__main__":
    main()
