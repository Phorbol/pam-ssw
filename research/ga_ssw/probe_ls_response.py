"""Bounded LS stationary-branch diagnostic; no search/default changes.

See evidence/ls-theory-20261010/mechanism-protocol.md. Coordinates are the
existing fixed Eckart chart; all Hessians share its Euclidean basis.
"""
import argparse
import importlib.util
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from scipy.linalg import null_space
from scipy.optimize import minimize
from pamssw.standalone.cluster_frame import ClusterFrame
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("ledger", ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py")
ledger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ledger)


def stiffness(soft, atoms):
    """Exact Cartesian pair Hessian, separated radial/transverse terms."""
    radial = np.zeros((3 * len(atoms), 3 * len(atoms)))
    transverse = radial.copy()
    for (i, j), r0, strength in zip(soft.pairs, soft.reference_distances, soft.strengths):
        delta = atoms.positions[j] - atoms.positions[i]
        r = np.linalg.norm(delta)
        u = delta / r
        length = soft.xi * r0
        value = strength * np.exp(-(r - r0) / length)
        rr = value / length**2 * np.outer(u, u)
        tt = -value / (length * r) * (np.eye(3) - np.outer(u, u))
        for target, block in ((radial, rr), (transverse, tt)):
            for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
                target[3*a:3*a+3, 3*b:3*b+3] += sign * block
    return radial, transverse


def run_case(source, out, calculator, *, amplitudes=(.125, .25, .5, 1.),
             bond_tables=None, request_cap=4000, wall_seconds=450):
    out.mkdir(parents=True, exist_ok=False)
    atoms = read(source)
    write(out / "input.extxyz", atoms)
    surface = ledger.CountedSurface(calculator, out / "requests.jsonl", request_cap, wall_seconds)
    actual = ledger.instrument_calculate(calculator)
    start = time.monotonic()
    result = {"source": str(source), "status": "started", "rows": []}
    try:
        frame = ClusterFrame(atoms)
        qbasis = null_space(frame.basis.T)

        def geometry(q):
            work = atoms.copy()
            work.positions = frame.positions(frame.reference + (qbasis @ q).reshape((-1, 3)))
            return work

        def optimize(q, soft=None, amplitude=0):
            before = surface.requests
            def objective(x):
                work = geometry(x)
                energy, force = surface.evaluate(work)
                if soft is not None:
                    bias, fbias = soft.evaluate(work)
                    energy += amplitude * bias
                    force = force + amplitude * fbias
                return energy, -qbasis.T @ force.ravel()
            opt = minimize(objective, q, jac=True, method="BFGS",
                           options={"gtol": 1e-4, "maxiter": 300})
            work = geometry(opt.x)
            energy, force = surface.evaluate(work)
            bias, fbias = (0., np.zeros_like(force)) if soft is None else soft.evaluate(work)
            fmax = float(np.linalg.norm(force + amplitude * fbias, axis=1).max())
            record = {"success_flag": bool(opt.success), "message": str(opt.message),
                      "steps": opt.nit, "requests": surface.requests-before,
                      "energy": energy, "bias": bias, "fmax": fmax}
            if fmax > 3e-4:
                raise RuntimeError("stationarity gate failed: " + json.dumps(record))
            return opt.x, work, record

        def hessian(q):
            matrices, asymmetries = [], []
            for h in (.001, .0005):
                columns = []
                for axis in np.eye(len(q)):
                    _, plus = surface.evaluate(geometry(q + h*axis))
                    _, minus = surface.evaluate(geometry(q - h*axis))
                    columns.append(-qbasis.T @ (plus-minus).ravel() / (2*h))
                raw = np.column_stack(columns)
                asymmetries.append(float(np.linalg.norm(raw-raw.T, 2)))
                matrices.append((raw+raw.T)/2)
            return matrices[-1], {"stencil_spread": float(np.linalg.norm(matrices[0]-matrices[1], 2)),
                                  "antisymmetric_norms": asymmetries}

        _, minimum, base_opt = optimize(np.zeros(qbasis.shape[1]))
        atoms = minimum.copy()
        frame = ClusterFrame(atoms)
        qbasis = null_space(frame.basis.T)
        q0 = np.zeros(qbasis.shape[1])
        if bond_tables is not None:
            energies, lengths = bond_tables
        elif atoms.numbers[0] == 29:
            energies, lengths = {(29, 29): 1.0}, {(29, 29): 3.0}
        else:
            energies = HC_BOND_ENERGIES
            lengths = {key: value + .1 for key, value in HC_BOND_LENGTHS.items()}
        soft = FrozenBondSoftening.from_atoms(atoms, bond_energies=energies, bond_lengths=lengths)
        result["frozen_ls"] = {"pairs": soft.pairs, "r0": soft.reference_distances,
                               "strengths": soft.strengths, "xi": soft.xi}
        radial0, trans0 = stiffness(soft, atoms)
        k0 = qbasis.T @ (radial0+trans0) @ qbasis
        # Independent finite differences of existing LS forces: no oracle cost.
        fd = []
        for axis in np.eye(len(q0)):
            _, fp = soft.evaluate(geometry(q0 + 1e-5*axis))
            _, fm = soft.evaluate(geometry(q0 - 1e-5*axis))
            fd.append(-qbasis.T @ (fp-fm).ravel() / 2e-5)
        error = float(np.linalg.norm(np.column_stack(fd)-k0, 2))
        result["ls_hessian_absolute_error"] = error
        if error > 1e-6 * max(1., np.linalg.norm(k0, 2)):
            raise RuntimeError("analytic LS Hessian disagrees with force differences")
        h0, hinfo = hessian(q0)
        values0, modes0 = np.linalg.eigh(h0)
        result["baseline"] = {**base_opt, **hinfo, "eigenvalues": values0}
        write(out / "minimum.extxyz", atoms)
        if values0[0] <= 5*hinfo["stencil_spread"]:
            raise RuntimeError("baseline not a numerically resolved strict minimum")
        _, fsoft = soft.evaluate(atoms)
        g = -qbasis.T @ fsoft.ravel()
        response = np.linalg.solve(h0, g)
        chi = float(g @ response)
        dilation = qbasis.T @ frame.relative.ravel()
        dilation_fraction = float((dilation@g)**2 / ((dilation@h0@dilation)*chi))
        result["linear_response"] = {"chi_eV": chi, "dilation_elastic_fraction": dilation_fraction,
                                     "dilation_drive_eV": float(dilation@g),
                                     "expected_dilation_drive_eV": -sum(soft.strengths)/soft.xi}
        np.savez(out / "baseline.npz", H=h0, Q=qbasis, radial=radial0, transverse=trans0, gradient=g)
        q = q0.copy()
        for amplitude in amplitudes:
            q, point, opt = optimize(q, soft, amplitude)
            write(out / f"soft-{amplitude}.extxyz", point)
            hv, info = hessian(q)
            radial, trans = stiffness(soft, point)
            k = qbasis.T @ (radial+trans) @ qbasis
            ht = hv+amplitude*k
            eigen, modes = np.linalg.eigh(ht)
            row = {"amplitude": amplitude, **opt, **info, "total_eigenvalues": eigen,
                   "response_per_atom": (opt["energy"]-base_opt["energy"])/len(atoms),
                   "predicted_response_per_atom": .5*amplitude**2*chi/len(atoms),
                   "linear_displacement_relative_error": float(np.linalg.norm(q+amplitude*response)/(amplitude*np.linalg.norm(response))),
                   "rms_displacement": float(np.linalg.norm(q)/np.sqrt(len(atoms))),
                   "radius_ratio": float(np.linalg.norm(point.positions-point.positions.mean(0))/np.linalg.norm(frame.relative)),
                   "baseline_soft_mode_direct_shift": float(amplitude*modes0[:,0]@k0@modes0[:,0]),
                   "current_soft_mode_radial": float(amplitude*modes[:,0]@(qbasis.T@radial@qbasis)@modes[:,0]),
                   "current_soft_mode_transverse": float(amplitude*modes[:,0]@(qbasis.T@trans@qbasis)@modes[:,0]),
                   "current_soft_mode_geometry": float(modes[:,0]@(hv-h0)@modes[:,0]),
                   "resolved_stable": bool(eigen[0] > 5*info["stencil_spread"])}
            np.savez(out / f"hessian-{amplitude}.npz", physical=hv, bias=k, total=ht, q=q)
            result["rows"].append(row)
            ledger.dump(out / "result.json", result)
            if not row["resolved_stable"]:
                raise RuntimeError("soft branch stability unresolved; stop continuation")
            _, released, release = optimize(q)
            write(out / f"released-{amplitude}.extxyz", released)
            row["release"] = {**release, "same_order_rms_displacement": float(np.linalg.norm(released.positions-atoms.positions)/np.sqrt(len(atoms)))}
        result["status"] = "completed"
    except Exception as error:
        result["status"] = "stopped"
        result["error"] = repr(error)
    finally:
        result.update(requests=surface.requests, actual_calculations=actual["calls"], seconds=time.monotonic()-start)
        ledger.dump(out / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=("emt", "mh1"))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import ase
    import scipy
    ledger.dump(args.out / "environment.json", {"python": platform.python_version(), "ase": ase.__version__,
        "scipy": scipy.__version__, "numpy": np.__version__, "job": os.environ.get("SLURM_JOB_ID"),
        "backend": args.backend, "argv": __import__("sys").argv})
    if args.backend == "emt":
        from ase.calculators.emt import EMT
        def factory():
            return EMT()
        inputs = [ROOT / f"research/ga_ssw/evidence/population-comparison-20260923/inputs/cu13-{i}.extxyz" for i in (0, 1)]
    else:
        import torch
        from mace.calculators import MACECalculator
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        model = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
        model_hash = ledger.sha256(model)
        if model_hash != "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47":
            raise ValueError("MH-1 model changed")
        ledger.dump(args.out / "model.json", {"path": str(model), "sha256": model_hash, "head": "omol",
            "torch": torch.__version__, "device": torch.cuda.get_device_name(), "dtype": "float64"})
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        def factory():
            return MACECalculator(model_paths="/home/gengjianrui/.cache/mace/mace-mh-1.model", head="omol", device="cuda", default_dtype="float64", enable_cueq=False, enable_oeq=False)
        inputs = [ROOT / f"research/ga_ssw/evidence/c4h6-ls-isomer-transfer-20261007/qualified-1662853/representative-{i}/refined.extxyz" for i in (1, 2)]
    # Separate calculators avoid nested instrumentation/cached state between cases.
    rows = [run_case(source, args.out / f"case-{i}", factory()) for i, source in enumerate(inputs)]
    ledger.dump(args.out / "summary.json", rows)


if __name__ == "__main__":
    main()
