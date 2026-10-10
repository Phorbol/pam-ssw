"""Qualify a saved C4H6 negative-curvature frame as a TS candidate.

This bounded MH-1/omol diagnostic finds a stationary-point candidate in one
fixed Eckart chart, checks its projected Hessian, and tests both signs of its
negative mode for true-PES minimum endpoints. It does not continue a biased
saddle branch or claim that matching endpoint coordinates prove basin identity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write
from scipy.linalg import null_space
from scipy.optimize import minimize, root

from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
from pamssw.standalone.softening import FrozenBondSoftening


ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
INPUT_DEFAULT = Path(
    "/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/"
    "research/ga_ssw/evidence/c4h6-mh1-curvature-20260924/"
    "native_ls-min03/input.json"
)
FORCE_TOL = 3e-4
ROOT_RESIDUAL_TOL = 3e-4
HESSIAN_STEPS = (0.001, 0.0005)
HESSIAN_MARGIN = 5.0


def load_ledger():
    spec = importlib.util.spec_from_file_location("ls_channel_ledger", LEDGER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ledger from {LEDGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atoms_from_input(path: Path):
    data = json.loads(path.read_text())
    required = ("numbers", "positions_A", "cell_A", "pbc")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"input JSON missing required fields: {missing}")
    atoms = Atoms(numbers=data["numbers"], positions=data["positions_A"],
                  cell=data["cell_A"], pbc=data["pbc"])
    if len(atoms) != 10 or sorted(atoms.numbers.tolist()) != [1] * 6 + [6] * 4:
        raise ValueError("expected one C4H6 geometry")
    if np.any(atoms.pbc) or atoms.constraints:
        raise ValueError("fixed isolated-cluster chart requires nonperiodic unconstrained atoms")
    if not np.isfinite(atoms.positions).all():
        raise ValueError("input coordinates must be finite")
    return data, atoms


def cccc_torsion(atoms):
    """Return an ASE dihedral only when the 1.64 A C graph is a four-C path."""
    carbons = [int(i) for i, z in enumerate(atoms.numbers) if int(z) == 6]
    adjacency = {i: set() for i in carbons}
    cutoff = 1.64
    for pos, i in enumerate(carbons):
        for j in carbons[pos + 1:]:
            if np.linalg.norm(atoms.positions[i] - atoms.positions[j]) <= cutoff:
                adjacency[i].add(j)
                adjacency[j].add(i)
    if len(carbons) != 4 or sum(map(len, adjacency.values())) != 6:
        return None
    endpoints = [i for i in carbons if len(adjacency[i]) == 1]
    middles = [i for i in carbons if len(adjacency[i]) == 2]
    if len(endpoints) != 2 or len(middles) != 2:
        return None
    order = [min(endpoints)]
    previous = None
    while len(order) < 4:
        choices = sorted(adjacency[order[-1]] - ({previous} if previous is not None else set()))
        if len(choices) != 1 or choices[0] in order:
            return None
        previous = order[-1]
        order.append(choices[0])
    if order[-1] not in endpoints or len(set(order)) != 4:
        return None
    try:
        return float(atoms.get_dihedral(*order, mic=False)), order
    except (ValueError, ZeroDivisionError):
        return None


def chart_for(atoms):
    frame = ClusterFrame(atoms)
    qbasis = null_space(frame.basis.T)
    if qbasis.shape != (3 * len(atoms), 3 * len(atoms) - 6):
        raise RuntimeError(f"unexpected internal basis shape {qbasis.shape}")
    return frame, qbasis


def run(input_path: Path, out: Path):
    ledger = load_ledger()
    source = input_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    input_data, start_atoms = atoms_from_input(source)
    if out.exists():
        raise FileExistsError(f"output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    write(out / "input.extxyz", start_atoms)

    setup_start = time.monotonic()
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    observed_hash = sha256(MODEL)
    if observed_hash != MODEL_SHA256:
        raise ValueError(f"MH-1 model hash mismatch: {observed_hash}")
    import torch
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calculator = MACECalculator(model_paths=str(MODEL), head="omol", device="cuda",
                                default_dtype="float64", enable_cueq=False, enable_oeq=False)
    setup_seconds = time.monotonic() - setup_start
    calculate_counter = ledger.instrument_calculate(calculator)
    surface = ledger.CountedSurface(calculator, out / "requests.jsonl", cap=1500, wall=600)
    frame, qbasis = chart_for(start_atoms)
    n_q = qbasis.shape[1]

    def geometry(q):
        q = np.asarray(q, dtype=float)
        work = start_atoms.copy()
        work.positions = frame.positions(frame.reference + (qbasis @ q).reshape((-1, 3)))
        return work

    def evaluate(q):
        return surface.evaluate(geometry(q))

    def projected_gradient(q):
        _, forces = evaluate(q)
        return -qbasis.T @ forces.ravel()

    def hessian(q, steps=HESSIAN_STEPS):
        matrices, asymmetry = [], []
        eye = np.eye(n_q)
        for step in steps:
            columns = []
            for axis in eye:
                _, f_plus = evaluate(q + step * axis)
                _, f_minus = evaluate(q - step * axis)
                columns.append(-qbasis.T @ (f_plus - f_minus).ravel() / (2 * step))
            raw = np.column_stack(columns)
            asymmetry.append(float(np.linalg.norm(raw - raw.T, 2)))
            matrices.append((raw + raw.T) / 2)
        spread = float(np.linalg.norm(matrices[0] - matrices[1], 2))
        return matrices, {"steps_A": list(steps), "stencil_spread_eV_A2": spread,
                          "antisymmetric_norms_eV_A2": asymmetry}

    def force_certificate(q):
        energy, forces = evaluate(q)
        atoms = geometry(q)
        fmax = float(np.linalg.norm(forces, axis=1).max())
        qgrad = -qbasis.T @ forces.ravel()
        return atoms, {"energy_eV": energy, "fmax_eV_A": fmax,
                       "projected_gradient_inf_eV_A": float(np.max(np.abs(qgrad))),
                       "force_qualified": bool(fmax <= FORCE_TOL)}

    result = {
        "status": "started",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_provenance": input_data.get("provenance", {}),
        "model": {"path": str(MODEL), "sha256": observed_hash, "expected_sha256": MODEL_SHA256,
                  "head": "omol", "device": "cuda", "dtype": "float64",
                  "enable_cueq": False, "enable_oeq": False,
                  "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                  "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                  "torch_num_threads": int(torch.get_num_threads()),
                  "cuda_device_name": torch.cuda.get_device_name(0)},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
                        "ase": importlib.metadata.version("ase"),
                        "torch": torch.__version__,
                        "mace_torch": importlib.metadata.version("mace-torch")},
        "setup_seconds_excluding_counted_wall": setup_seconds,
        "chart": {"kind": "fixed affine Eckart section", "Q_shape": qbasis.shape,
                  "Q_orthonormal_error": float(np.linalg.norm(qbasis.T @ qbasis - np.eye(n_q), 2))},
        "limits": {"requests": 1500, "counted_wall_seconds": 600,
                   "force_tolerance_eV_A": FORCE_TOL,
                   "projected_root_residual_tolerance_eV_A": ROOT_RESIDUAL_TOL,
                   "hessian_margin_factor": HESSIAN_MARGIN,
                   "hessian_steps_A": list(HESSIAN_STEPS)},
        "phases": {},
    }
    ledger.dump(out / "result.json", result)
    total_start = time.monotonic()

    def phase_start():
        return surface.requests

    def phase_end(name, begin, record):
        record["requests"] = int(surface.requests - begin)
        record["request_total"] = int(surface.requests)
        result["phases"][name] = record
        ledger.dump(out / "result.json", result)

    q_ts = np.zeros(n_q)
    try:
        root_begin = phase_start()

        def root_jac(q):
            columns = []
            eye = np.eye(n_q)
            for axis in eye:
                _, f_plus = evaluate(q + 0.001 * axis)
                _, f_minus = evaluate(q - 0.001 * axis)
                columns.append(-qbasis.T @ (f_plus - f_minus).ravel() / 0.002)
            return np.column_stack(columns)

        try:
            root_result = root(projected_gradient, q_ts, jac=root_jac, method="hybr",
                               options={"maxfev": 200})
        except Exception as error:
            phase_end("ts_root", root_begin, {"status": "failed", "error": repr(error),
                                              "method": "scipy.root hybr", "maxfev": 200})
            raise
        q_ts = np.asarray(root_result.x, dtype=float)
        phase_end("ts_root", root_begin, {
            "method": "scipy.root hybr", "jacobian": "central force differences, h=0.001 A",
            "maxfev": 200, "success_flag": bool(root_result.success),
            "status_code": int(root_result.status), "message": str(root_result.message),
            "nfev": int(root_result.nfev), "njev": int(getattr(root_result, "njev", -1)),
            "q_residual_inf_eV_A": float(np.max(np.abs(root_result.fun))),
            "q_residual_norm_eV_A": float(np.linalg.norm(root_result.fun)),
            "residual_qualified": bool(np.max(np.abs(root_result.fun)) <= ROOT_RESIDUAL_TOL),
        })
        cert_begin = phase_start()
        try:
            ts_atoms, ts_force = force_certificate(q_ts)
        except Exception as error:
            phase_end("ts_force_certificate", cert_begin,
                      {"status": "failed", "error": repr(error)})
            raise
        phase_end("ts_force_certificate", cert_begin, ts_force)
        write(out / "ts-candidate.extxyz", ts_atoms)
        result["ts_certificate"] = ts_force
        result["ts_certificate"]["stationary_qualified"] = bool(
            result["phases"]["ts_root"]["residual_qualified"] and ts_force["force_qualified"])
        result["ts_certificate"]["torsion_deg_and_order"] = cccc_torsion(ts_atoms)
        if not result["ts_certificate"]["stationary_qualified"]:
            raise RuntimeError("TS stationarity qualification failed")

        h_begin = phase_start()
        try:
            ts_matrices, ts_hinfo = hessian(q_ts)
        except Exception as error:
            phase_end("ts_hessian", h_begin, {"status": "failed", "error": repr(error)})
            raise
        ts_values, ts_vectors = np.linalg.eigh(ts_matrices[-1])
        ts_spread = ts_hinfo["stencil_spread_eV_A2"]
        ts_hessian_qualified = bool(ts_values[0] < -HESSIAN_MARGIN * ts_spread and
                                    ts_values[1] > HESSIAN_MARGIN * ts_spread)
        np.savez(out / "ts-hessian.npz", H_h001=ts_matrices[0], H_h0005=ts_matrices[1],
                 Q=qbasis, eigenvalues=ts_values, eigenvectors=ts_vectors,
                 negative_mode_q=ts_vectors[:, 0], negative_mode_cartesian=qbasis @ ts_vectors[:, 0])
        phase_end("ts_hessian", h_begin, {**ts_hinfo,
            "eigenvalues_eV_A2": ts_values, "negative_eigenvalue_count": int(np.count_nonzero(ts_values < 0)),
            "single_resolved_negative_mode": ts_hessian_qualified,
            "criterion": "lambda0 < -5*stencil spread and lambda1 > 5*stencil spread"})
        result["ts_certificate"]["hessian_qualified"] = ts_hessian_qualified
        result["ts_certificate"]["negative_eigenvalue_eV_A2"] = float(ts_values[0])
        result["ts_certificate"]["next_eigenvalue_eV_A2"] = float(ts_values[1])
        if not ts_hessian_qualified:
            raise RuntimeError("TS Hessian is not a resolved first-order saddle")
        ledger.dump(out / "result.json", result)

        mode = ts_vectors[:, 0]
        endpoint = {}
        for side, sign in (("minus", -1.0), ("plus", 1.0)):
            endpoint[side] = {}
            for displacement in (0.025, 0.05):
                name = f"{side}-{displacement:.3f}"
                begin = phase_start()
                q_start = q_ts + sign * displacement * mode

                def objective(q):
                    energy, forces = evaluate(q)
                    return energy, -qbasis.T @ forces.ravel()

                try:
                    opt = minimize(objective, q_start, jac=True, method="BFGS",
                                   options={"gtol": 1e-4, "maxiter": 300})
                except Exception as error:
                    phase_end(name, begin, {"status": "failed", "error": repr(error),
                                            "side": side, "start_displacement_A": displacement})
                    raise
                q_end = np.asarray(opt.x, dtype=float)
                try:
                    end_atoms, certificate = force_certificate(q_end)
                except Exception as error:
                    phase_end(name, begin, {"status": "failed", "error": repr(error),
                                            "side": side, "start_displacement_A": displacement,
                                            "optimizer_success_flag": bool(opt.success),
                                            "optimizer_steps": int(opt.nit)})
                    raise
                write(out / f"{name}.extxyz", end_atoms)
                torsion = cccc_torsion(end_atoms)
                row = {**certificate, "optimizer_success_flag": bool(opt.success),
                       "optimizer_message": str(opt.message), "optimizer_steps": int(opt.nit),
                       "torsion_deg_and_order": torsion,
                       "start_displacement_A": float(displacement), "side": side}
                phase_end(name, begin, row)
                endpoint[side][displacement] = {"q": q_end, "atoms": end_atoms,
                                                "record": row, "energy": certificate["energy_eV"]}
                ledger.dump(out / "result.json", result)
                if not certificate["force_qualified"]:
                    raise RuntimeError(f"endpoint force qualification failed: {name}")

                if displacement == 0.05:
                    h_begin = phase_start()
                    try:
                        end_matrices, end_hinfo = hessian(q_end)
                    except Exception as error:
                        phase_end(f"{name}_hessian", h_begin,
                                  {"status": "failed", "error": repr(error)})
                        raise
                    end_values, end_vectors = np.linalg.eigh(end_matrices[-1])
                    spread = end_hinfo["stencil_spread_eV_A2"]
                    positive = bool(end_values[0] > HESSIAN_MARGIN * spread)
                    np.savez(out / f"{name}-hessian.npz", H_h001=end_matrices[0],
                             H_h0005=end_matrices[1], Q=qbasis, eigenvalues=end_values,
                             eigenvectors=end_vectors)
                    hrec = {**end_hinfo, "eigenvalues_eV_A2": end_values,
                            "positive_resolved_hessian": positive,
                            "criterion": "lambda0 > 5*stencil spread"}
                    phase_end(f"{name}_hessian", h_begin, hrec)
                    endpoint[side][displacement]["record"]["hessian_qualified"] = positive
                    endpoint[side][displacement]["record"]["lowest_eigenvalue_eV_A2"] = float(end_values[0])
                    if not certificate["force_qualified"]:
                        endpoint[side][displacement]["record"]["stable_minimum_qualified"] = False
                    else:
                        endpoint[side][displacement]["record"]["stable_minimum_qualified"] = positive
                    endpoint[side][displacement]["record"]["minimum_energy_eV"] = certificate["energy_eV"]
                    endpoint[side][displacement]["record"]["hessian_stencil_spread_eV_A2"] = spread
                    if not positive:
                        raise RuntimeError(f"endpoint Hessian is not a resolved minimum: {name}")
                    ledger.dump(out / "result.json", result)

        comparisons = {}
        for side in ("minus", "plus"):
            near, far = endpoint[side][0.025], endpoint[side][0.05]
            tors_near, tors_far = near["record"]["torsion_deg_and_order"], far["record"]["torsion_deg_and_order"]
            delta_torsion = None
            if tors_near is not None and tors_far is not None:
                delta_torsion = (tors_near[0] - tors_far[0] + 180.0) % 360.0 - 180.0
            comparisons[side] = {
                "same_order_rmsd_A": float(np.sqrt(np.mean(np.sum(
                    (near["atoms"].positions - far["atoms"].positions) ** 2, axis=1)))),
                "energy_difference_025_minus_050_eV": float(near["energy"] - far["energy"]),
                "torsion_025_deg_and_order": tors_near,
                "torsion_050_deg_and_order": tors_far,
                "wrapped_torsion_difference_025_minus_050_deg": delta_torsion,
                "interpretation": "geometry comparison only; no basin identity inferred",
            }
        result["same_side_endpoint_comparisons"] = comparisons

        w_comparisons = {}
        for side in ("minus", "plus"):
            min_atoms = endpoint[side][0.05]["atoms"]
            soft = FrozenBondSoftening.from_atoms(
                min_atoms, bond_energies=HC_BOND_ENERGIES,
                bond_lengths={key: value + 0.1 for key, value in HC_BOND_LENGTHS.items()},
                initial_fraction=0.03, xi=0.2)
            w_ts, _ = soft.evaluate(ts_atoms)
            w_min, _ = soft.evaluate(min_atoms)
            w_comparisons[side] = {
                "frozen_at_endpoint": f"{side}-0.050",
                "pair_count": len(soft.pairs), "pairs": soft.pairs,
                "reference_distances_A": soft.reference_distances,
                "strengths_eV": soft.strengths, "xi": soft.xi,
                "W_TS_eV": w_ts, "W_min_eV": w_min,
                "baseline_barrier_slope_W_TS_minus_W_min_eV": w_ts - w_min,
                "interpretation": "a=0 envelope-slope prediction for this frozen W; no finite-a biased branch verified",
            }
        result["baseline_W_differences"] = w_comparisons
        result["status"] = "completed_stationary_and_downhill_diagnostics"
    except Exception as error:
        result["status"] = "stopped"
        result["error"] = repr(error)
    finally:
        result.update(requests=int(surface.requests), denials=int(surface.denials),
                      request_boundary=surface.boundary,
                      actual_calculate_calls=int(calculate_counter["calls"]),
                      counted_wall_seconds=time.monotonic() - surface.started,
                      total_seconds_after_model_setup=time.monotonic() - total_start)
        ledger.dump(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = run(args.input, args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out.resolve()),
                      "requests": result.get("requests"), "error": result.get("error")}, indent=2))


if __name__ == "__main__":
    main()
