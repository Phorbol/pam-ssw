"""Qualify the frozen 8-atom AFLOW CuO cell for a bounded joint-VC run.

This script performs only E/F/stress prechecks, representation comparison, one
strict 64-atom cell quench, and a two-step finite-cell Hessian.  It never starts
SSW.  The input directory must contain ``source-input.json`` and ``plan.json``;
the latter supplies the local MACE model path/hash and ``caps``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import signal
import time
import traceback
from pathlib import Path

import numpy as np
from ase import Atoms
from scipy.linalg import null_space

from pamssw.standalone.cell_relax import cell_quench
from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atoms(data):
    if not isinstance(data, dict):
        raise ValueError("source-input.json must be an object")
    return Atoms(symbols=data["symbols"], positions=data["positions"],
                 cell=data["cell"], pbc=data["pbc"])


class BudgetSurface(ASEStressSurface):
    def __init__(self, calculator, *, cap, deadline, output):
        super().__init__(calculator)
        self.cap = int(cap)
        self.deadline = deadline
        self.output = output
        self.label = "unlabeled"

    def replace_calculator(self, calculator):
        self.calculator = calculator

    def evaluate(self, atoms):
        if self.requests >= self.cap:
            raise RuntimeError(f"EFS request cap exhausted ({self.cap})")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("600 second qualification wall cap reached")
        label = self.label
        e, f, s = super().evaluate(atoms)
        row = {"request": self.requests, "label": label, "natoms": len(atoms),
               "energy": float(e), "energy_per_atom": float(e / len(atoms)),
               "forces": np.asarray(f).tolist(), "numbers": atoms.numbers.tolist(), "pbc": atoms.pbc.tolist(),
               "fmax": float(np.linalg.norm(f, axis=1).max()),
               "stress": np.asarray(s).tolist(),
               "positions": np.asarray(atoms.positions).tolist(),
               "cell": np.asarray(atoms.cell.array).tolist()}
        with (self.output / "evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return e, f, s


def _evaluate(surface, atoms, label):
    old = surface.label
    surface.label = label
    try:
        return surface.evaluate(atoms)
    finally:
        surface.label = old


def _runtime(model, device):
    import torch
    package_versions = {}
    for name in ("ase", "numpy", "scipy", "torch", "mace-torch"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    return {
        "python": __import__("sys").version,
        "packages": package_versions,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "model": str(model),
        "model_sha256": _sha256(model),
    }


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    output = root / "qualification"
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "root": str(root), "requests": 0,
              "scope": "input qualification only; no SSW walk"}
    _save_json(output / "progress.json", report)
    (output / "run-source.py").write_text(Path(__file__).read_text())
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError("600 second qualification wall cap reached")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(600)
    try:
        source_path, plan_path = root / "source-input.json", root / "plan.json"
        source = json.loads(source_path.read_text())
        plan = json.loads(plan_path.read_text())
        _save_json(output / "source-input.json", source)
        _save_json(output / "plan-used.json", plan)
        source_atoms = _atoms(source)
        model_value = plan.get("model", plan.get("model_path"))
        if isinstance(model_value, dict):
            model_value = model_value.get("path")
        model = Path(model_value)
        expected_sha = plan.get("model_sha256")
        if expected_sha is None and isinstance(plan.get("model"), dict):
            expected_sha = plan["model"].get("sha256")
        if not model.is_file():
            raise FileNotFoundError(f"MACE model not found: {model}")
        actual_sha = _sha256(model)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(f"model SHA256 mismatch: expected {expected_sha}, got {actual_sha}")
        caps = plan.get("caps", plan.get("budget", {}))
        cap = caps.get("max_EFS", caps.get("EFS", caps.get("requests", 1500)))
        seconds = float(caps.get("seconds", caps.get("wall_seconds", 600)))
        if int(cap) != 1500 or seconds != 600:
            raise ValueError("plan caps must be exactly max_EFS=1500 and seconds=600")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from mace.calculators import MACECalculator
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this qualification requires MACE-OMAT CUDA")
        device = "cuda"
        runtime = _runtime(model, device)
        _save_json(output / "runtime.json", runtime)
        calc = MACECalculator(model_paths=str(model), device=device,
                              default_dtype="float64", enable_cueq=False)
        surface = BudgetSurface(calc, cap=int(cap), deadline=started + seconds, output=output)

        source_info = {"natoms": len(source_atoms), "symbols": source_atoms.get_chemical_symbols(),
                       "pbc": source_atoms.pbc.tolist(), "cell": source_atoms.cell.array.tolist(),
                       "volume_A3": float(source_atoms.get_volume())}
        _save_json(output / "source-readback.json", source_info)
        e0, f0, s0 = _evaluate(surface, source_atoms, "source-initial")
        basis = source_atoms.copy()
        # Explicit row-cell operation: c' = c + a; positions are unchanged.
        basis.set_cell(source_atoms.cell.array + np.array([[0., 0., 0.], [0., 0., 0.], source_atoms.cell.array[0]]), scale_atoms=False)
        eb, fb, sb = _evaluate(surface, basis, "basis-c-plus-a")
        repeated = source_atoms.repeat((2, 2, 2))
        er, fr, sr = _evaluate(surface, repeated, "repeat-2x2x2")
        n = len(source_atoms)
        blocks = np.asarray(fr).reshape((-1, n, 3))
        equivalence = {"basis_det_U": 1.0,
                       "basis_energy_error_eV": float(eb - e0),
                       "basis_force_error_eV_A": float(np.max(np.abs(fb - f0))),
                       "basis_stress_error_eV_A3": float(np.max(np.abs(sb - s0))),
                       "repeat_natoms": len(repeated),
                       "repeat_energy_per_atom_error_eV": float(er / len(repeated) - e0 / n),
                       "repeat_force_error_eV_A": float(np.max(np.abs(blocks - f0))),
                       "repeat_stress_error_eV_A3": float(np.max(np.abs(sr - s0)))}
        _save_json(output / "equivalence.json", equivalence)

        strain_length = float(repeated.get_volume() ** (1.0 / 3.0))
        quench = cell_quench(repeated, surface, strain_length=strain_length,
            pressure=0.0, fmax=1e-4, stress_tol=1e-5, max_step=0.2,
            maxiter=500, lbfgs_memory=10)
        endpoint = quench.evaluation
        report.update({"runtime": runtime, "source": source_info,
                       "initial": {"energy": float(e0), "volume_A3": float(source_atoms.get_volume()),
                                   "fmax": float(np.linalg.norm(f0, axis=1).max()),
                                   "stress": np.asarray(s0).tolist(),
                                   "optimizer": {"status": "not_run", "error": None}},
                       "equivalence": equivalence,
                       "quench": {"converged": bool(quench.converged),
                                   "optimizer_converged": bool(quench.optimizer.converged),
                                   "optimizer_status": quench.optimizer.status,
                                   "optimizer_error": quench.optimizer.error,
                                   "certificate": quench.certificate,
                                   "requests": quench.requests,
                                   "strain_length_A": strain_length},
                       "requests": surface.requests})
        _save_json(output / "progress.json", report)
        if endpoint is None or not quench.converged:
            report["status"] = "quench_failed"
            _save_json(output / "result.json", report)
            return 0

        del calc
        torch.cuda.empty_cache()
        fresh_calc = MACECalculator(model_paths=str(model), device=device,
                                    default_dtype="float64", enable_cueq=False)
        surface.replace_calculator(fresh_calc)
        endpoint_json = {"symbols": endpoint.atoms.get_chemical_symbols(),
                         "positions": endpoint.atoms.positions.tolist(),
                         "cell": endpoint.atoms.cell.array.tolist(),
                         "pbc": endpoint.atoms.pbc.tolist()}
        _save_json(output / "quench-endpoint.json", endpoint_json)
        ef, ff, sf = _evaluate(surface, endpoint.atoms, "endpoint-fresh")
        report["requests"] = surface.requests
        fresh_check = {"energy_error_eV": float(ef - endpoint.energy),
                       "fmax": float(np.linalg.norm(ff, axis=1).max()),
                       "stress_max": float(np.abs(sf).max()),
                       "force_error_eV_A": float(np.max(np.abs(ff - endpoint.forces))),
                       "stress_error_eV_A3": float(np.max(np.abs(sf - endpoint.stress)))}
        report["fresh_endpoint"] = fresh_check
        if fresh_check["fmax"] > 1e-4 or fresh_check["stress_max"] > 1e-5:
            report["status"] = "fresh_certificate_failed"
            _save_json(output / "result.json", report)
            return 0

        # The replacement calculator is used for the Hessian too, and every gradient is
        # obtained from chart.evaluate, so all 3N+6 derivatives are charged.
        chart = SymmetricLogStrainChart(endpoint.atoms, strain_length=strain_length)
        q = chart.pack(endpoint.atoms)
        ndof = q.size
        translations = np.zeros((ndof, 3))
        for axis in range(3):
            translations[axis:3 * len(endpoint.atoms):3, axis] = 1.0 / np.sqrt(len(endpoint.atoms))
        basis_q = null_space(translations.T)
        if basis_q.shape != (ndof, ndof - 3):
            raise RuntimeError(f"unexpected nontranslation basis shape {basis_q.shape}")
        _save_json(output / "hessian-basis.json", basis_q.tolist())
        hessians, eigenvalues, skew_norms = {}, {}, {}
        for h in (1e-4, 5e-5):
            key = f"{h:.0e}"
            raw_path = output / f"hessian-{key}-raw.dat"
            H = np.memmap(raw_path, dtype="float64", mode="w+",
                          shape=(ndof - 3, ndof - 3), order="C")
            H[:] = np.nan
            for i in range(ndof - 3):
                plus = chart.evaluate(q + h * basis_q[:, i], surface.evaluate, pressure=0.0)
                minus = chart.evaluate(q - h * basis_q[:, i], surface.evaluate, pressure=0.0)
                H[:, i] = basis_q.T @ (plus.gradient - minus.gradient) / (2.0 * h)
                if (i + 1) % 25 == 0:
                    H.flush()
                    _save_json(output / "progress.json", {**report, "requests": surface.requests,
                               "hessian_step": key, "hessian_columns": i + 1})
            H.flush()
            raw = np.asarray(H)
            sym = (raw + raw.T) / 2.0
            vals = np.linalg.eigvalsh(sym)
            np.save(output / f"hessian-{key}-sym.npy", sym)
            np.save(output / f"eigenvalues-{key}.npy", vals)
            hessians[key] = sym.tolist()
            eigenvalues[key] = vals.tolist()
            skew_norms[key] = float(np.linalg.norm(raw - raw.T))
            _save_json(output / "progress.json", {**report, "requests": surface.requests,
                       "hessian_completed": key, "hessian_columns": ndof - 3})
        diff = np.asarray(hessians["1e-04"]) - np.asarray(hessians["5e-05"])
        report.update({"requests": surface.requests, "hessian": hessians,
                       "eigenvalues": eigenvalues,
                       "hessian_skew_norm": skew_norms,
                       "hessian_step_difference_spectral": float(np.linalg.norm(diff, ord=2)),
                       "hessian_step_difference_frobenius": float(np.linalg.norm(diff)),
                       "hessian_basis_shape": list(basis_q.shape),
                       "interpretation": "qualification evidence only; positive curvature is finite-cell local evidence, not phase stability or search success"})
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                      requests=(locals().get("surface").requests if "surface" in locals() else 0))
    finally:
        signal.alarm(0)
        report["seconds"] = time.monotonic() - started
        _save_json(output / "result.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
