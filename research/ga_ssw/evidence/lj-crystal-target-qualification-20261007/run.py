#!/usr/bin/env python3
"""Qualify tiny periodic fcc/hcp LJ references; this is not a crystal search."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import ase
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
import numpy as np
import scipy
from scipy.optimize import minimize

# Permit the explicitly allowed login-node dummy preflight without relying on
# a scheduler-only PYTHONPATH setting.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pamssw.standalone.vc_geometry import ASEStressSurface


EPSILON = 1.0  # eV
SIGMA = 1.0  # Angstrom
RC_OVER_SIGMA = (2.7, 3.0, 6.0, 10.0)
MAX_REQUESTS = 100
PROCESS_DEADLINE_S = 480.0
FD_STEP = 1.0e-4
FD_ATOL = 1.0e-7  # eV/atom per log-scale coordinate
FD_RTOL = 1.0e-4
FORCE_TOL = 0.05  # eV/Angstrom
STRESS_TOL = 1.0e-5  # eV/Angstrom^3; reference qualification only
OPT_GTOL = 1.0e-7  # eV/atom per log-scale coordinate
OPT_MAXITER = 80


class SlotStop(RuntimeError):
    pass


class CountingCalculator(Calculator):
    """Count ASE's actual calls to the wrapped calculator's calculate method."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.implemented_properties = list(inner.implemented_properties)
        self.calculate_calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        self.calculate_calls += 1
        super().calculate(atoms, properties, system_changes)
        self.inner.calculate(atoms, properties, system_changes)
        self.results = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in self.inner.results.items()
        }


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _append_jsonl(path: Path, item):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_jsonable(item), allow_nan=False) + "\n")
        f.flush()


def _write_json(path: Path, item):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(item), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_atoms(phase: str) -> Atoms:
    """One-atom primitive fcc or two-atom primitive hcp at LJ r_min."""
    r_min = 2.0 ** (1.0 / 6.0) * SIGMA
    if phase == "fcc":
        return bulk("Ar", "fcc", a=np.sqrt(2.0) * r_min)
    if phase == "hcp":
        return bulk("Ar", "hcp", a=r_min, c=np.sqrt(8.0 / 3.0) * r_min)
    raise ValueError(f"unknown phase {phase!r}")


def scaled_atoms(reference: Atoms, phase: str, x) -> Atoms:
    x = np.asarray(x, dtype=float)
    expected = 1 if phase == "fcc" else 2
    if x.shape != (expected,) or not np.isfinite(x).all():
        raise ValueError("invalid log-scale vector")
    atoms = reference.copy()
    frac = reference.get_scaled_positions(wrap=False)
    if phase == "fcc":
        transform = np.eye(3) * np.exp(float(x[0]))
    else:
        transform = np.diag([np.exp(float(x[0])), np.exp(float(x[0])), np.exp(float(x[1]))])
    atoms.set_cell(reference.cell.array @ transform, scale_atoms=False)
    atoms.positions = frac @ atoms.cell.array
    return atoms


def log_gradient(phase: str, atoms: Atoms, stress: np.ndarray) -> np.ndarray:
    volume = atoms.get_volume()
    if phase == "fcc":
        return np.array([volume * np.trace(stress) / len(atoms)])
    return np.array([
        volume * (stress[0, 0] + stress[1, 1]) / len(atoms),
        volume * stress[2, 2] / len(atoms),
    ])


class LoggedSurface(ASEStressSurface):
    def __init__(self, calculator, *, cap, deadline, ledger_path, stage):
        super().__init__(calculator if isinstance(calculator, CountingCalculator)
                         else CountingCalculator(calculator))
        self.cap = int(cap)
        self.deadline = deadline
        self.ledger_path = Path(ledger_path)
        self.stage = stage
        self.failures = 0

    def evaluate(self, atoms):
        if time.monotonic() >= self.deadline:
            raise SlotStop("global 480-second process deadline reached")
        if self.requests >= self.cap:
            raise SlotStop(f"slot {self.cap}-request cap reached")
        before = self.requests
        calculate_before = self.calculator.calculate_calls
        row = {
            "request_before": before,
            "stage": self.stage,
            "wall_elapsed_s": time.monotonic() - (self.deadline - PROCESS_DEADLINE_S),
            "cell_A": atoms.cell.array.copy(),
            "positions_A": atoms.positions.copy(),
            "numbers": atoms.numbers.copy(),
            "pbc": atoms.pbc.copy(),
        }
        try:
            energy, forces, stress = super().evaluate(atoms)
            row.update(
                status="ok", charged=True, request=self.requests,
                calculator_calculate_calls_before=calculate_before,
                calculator_calculate_calls_after=self.calculator.calculate_calls,
                calculator_calculate_calls_delta=self.calculator.calculate_calls - calculate_before,
                energy_eV=energy, forces_eV_A=forces, stress_eV_A3=stress,
            )
            return energy, forces, stress
        except Exception as exc:
            self.failures += 1
            row.update(
                status="failed", charged=(self.requests > before), request=self.requests,
                calculator_calculate_calls_before=calculate_before,
                calculator_calculate_calls_after=self.calculator.calculate_calls,
                calculator_calculate_calls_delta=self.calculator.calculate_calls - calculate_before,
                error=repr(exc), traceback=traceback.format_exc(limit=4),
            )
            raise
        finally:
            _append_jsonl(self.ledger_path, row)
            _write_json(self.ledger_path.with_name(self.ledger_path.stem + "-progress.json"), {
                "stage": self.stage, "requests": self.requests, "failed_requests": self.failures,
                "calculator_calculate_calls": self.calculator.calculate_calls,
                "last_request_status": row["status"], "last_request_wall_s": row["wall_elapsed_s"],
            })


def _certificate(atoms, energy, forces, stress):
    fmax = float(np.linalg.norm(forces, axis=1).max())
    stress_max = float(np.abs(stress).max())
    finite = bool(
        np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(stress).all()
    )
    return {
        "finite": finite,
        "fmax_eV_A": fmax,
        "stress_max_eV_A3": stress_max,
        "force_pass": finite and fmax <= FORCE_TOL,
        "stress_pass": finite and stress_max <= STRESS_TOL,
        "qualified": finite and fmax <= FORCE_TOL and stress_max <= STRESS_TOL,
    }


def _slot_metadata(phase, rc, out, command):
    source_paths = {}
    for name in ("pamssw.standalone.vc_geometry", "scipy.optimize"):
        mod = __import__(name, fromlist=["__file__"])
        source_paths[name] = str(Path(mod.__file__).resolve())
    lj_module = sys.modules.get("ase.calculators.lj")
    if lj_module is not None:
        source_paths["ase.calculators.lj"] = str(Path(lj_module.__file__).resolve())
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        head = f"unavailable: {exc!r}"
    return {
        "phase": phase,
        "rc_over_sigma": rc,
        "epsilon_eV": EPSILON,
        "sigma_A": SIGMA,
        "smooth": False,
        "potential": "ASE LennardJones, shifted energy, unsmoothed force at cutoff",
        "initial_structure": "primitive fcc (1 atom) or hcp (2 atoms), nearest-neighbor r_min",
        "initial_geometry": out / "initial.extxyz",
        "final_geometry": out / "final.extxyz",
        "protocol": Path(__file__).with_name("protocol.md").resolve(),
        "executed_script": Path(__file__).resolve(),
        "command": command,
        "git_head": head,
        "python": sys.executable,
        "python_version": sys.version,
        "ase_version": ase.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_paths": source_paths,
        "source_snapshot_paths": {
            "runner": "source-snapshot/run.py",
            "vc_geometry": "source-snapshot/pamssw/standalone/vc_geometry.py",
        },
        "environment": {k: os.environ.get(k) for k in (
            "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONNOUSERSITE", "PYTHONPATH",
        )},
    }


def _lj_calculator(rc):
    from ase.calculators.lj import LennardJones
    return LennardJones(epsilon=EPSILON, sigma=SIGMA, rc=float(rc) * SIGMA, smooth=False)


def run_slot(phase, rc, out, deadline, command, *, calculator_factory=None, synthetic=False):
    if calculator_factory is None and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("actual LJ E/F/stress evaluations require a Slurm job")
    out.mkdir(parents=True, exist_ok=True)
    factory = calculator_factory or _lj_calculator
    calc = factory(rc)
    row = _slot_metadata(phase, rc, out, command)
    row.update({
        "status": "running",
        "synthetic_preflight": bool(synthetic),
        "calculator": "DummyCalculator (synthetic path only)" if synthetic else "ASE LennardJones",
        "potential": "synthetic zero E/F/stress" if synthetic else "ASE LennardJones, shifted energy, unsmoothed force at cutoff",
        "max_requests": MAX_REQUESTS,
        "fd_step_log_scale": FD_STEP,
        "fd_atol_eV_atom": FD_ATOL,
        "fd_rtol": FD_RTOL,
        "optimizer": "SciPy BFGS with analytic stress-derived log-scale gradient",
        "optimizer_gtol_eV_atom": OPT_GTOL,
        "optimizer_maxiter": OPT_MAXITER,
        "qualification": {"fmax_max_eV_A": FORCE_TOL, "stress_max_eV_A3": STRESS_TOL},
    })
    _write_json(out / "slot.json", row)
    Path(out / "requests.jsonl").write_text("", encoding="utf-8")
    Path(out / "fresh.jsonl").write_text("", encoding="utf-8")
    atoms0 = make_atoms(phase)
    atoms0.calc = None
    write(out / "initial.extxyz", atoms0)
    source_copy = out / "source-snapshot"
    source_copy.mkdir(exist_ok=True)
    shutil.copy2(__file__, source_copy / "run.py")
    vc_source = Path(sys.modules[ASEStressSurface.__module__].__file__).resolve()
    vc_snapshot = source_copy / "pamssw/standalone/vc_geometry.py"
    vc_snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vc_source, vc_snapshot)
    protocol = Path(__file__).with_name("protocol.md")
    if protocol.exists():
        shutil.copy2(protocol, source_copy / "protocol.md")
    surface = LoggedSurface(calc, cap=MAX_REQUESTS, deadline=deadline,
                            ledger_path=out / "requests.jsonl", stage="initial")
    nscale = 1 if phase == "fcc" else 2
    x0 = np.zeros(nscale, dtype=float)
    def evaluate(x, stage):
        x = np.asarray(x, dtype=float)
        atoms = scaled_atoms(atoms0, phase, x)
        surface.stage = stage
        energy, forces, stress = surface.evaluate(atoms)
        per_atom = float(energy / len(atoms))
        grad = log_gradient(phase, atoms, stress)
        return per_atom, grad, atoms, energy, forces, stress

    try:
        # Paid central differences at the declared initial structure.
        _, analytic, *_ = evaluate(x0, "initial_and_fd_center")
        fd = np.empty(nscale, dtype=float)
        for j in range(nscale):
            dx = np.zeros(nscale)
            dx[j] = FD_STEP
            ep = evaluate(x0 + dx, "initial_fd_plus")[0]
            em = evaluate(x0 - dx, "initial_fd_minus")[0]
            fd[j] = (ep - em) / (2.0 * FD_STEP)
        abs_error = np.abs(fd - analytic)
        allowed = FD_ATOL + FD_RTOL * np.maximum(np.abs(fd), np.abs(analytic))
        fd_pass = bool(np.all(abs_error <= allowed))
        row["initial_gradient_check"] = {
            "analytic_eV_atom": analytic,
            "finite_difference_eV_atom": fd,
            "absolute_error_eV_atom": abs_error,
            "allowed_error_eV_atom": allowed,
            "pass": fd_pass,
            "note": "initial-point implementation check; not efficacy evidence",
        }
        if not fd_pass:
            row["status"] = "gradient_check_failed"
            row["requests"] = surface.requests
            row["calculator_calculate_calls"] = surface.calculator.calculate_calls
            row["failed_requests"] = surface.failures
            row["total_elapsed_s"] = time.monotonic() - (deadline - PROCESS_DEADLINE_S)
            _write_json(out / "slot.json", row)
            return row

        result = minimize(
            lambda x: evaluate(x, "bfgs")[:2], x0.copy(), method="BFGS", jac=True,
            options={"gtol": OPT_GTOL, "maxiter": OPT_MAXITER, "disp": False},
        )
        xfinal = np.asarray(result.x, dtype=float)
        energy_per_atom, grad, final_atoms, energy, forces, stress = evaluate(xfinal, "terminal")
        final_atoms.calc = None
        write(out / "final.extxyz", final_atoms)
        terminal_cert = _certificate(final_atoms, energy, forces, stress)
        row.update({
            "log_scales": xfinal,
            "lattice_parameters_A": {
                "a": float(final_atoms.cell.lengths()[0]),
                "b": float(final_atoms.cell.lengths()[1]),
                "c": float(final_atoms.cell.lengths()[2]),
                "angles_deg": final_atoms.cell.angles(),
            },
            "energy_eV": energy,
            "energy_eV_atom": energy_per_atom,
            "terminal_gradient_eV_atom": grad,
            "terminal_EFS": {"energy_eV": energy, "forces_eV_A": forces, "stress_eV_A3": stress},
            "terminal_certificate": terminal_cert,
            "optimizer_exit": {
                "success": bool(result.success), "status": int(result.status),
                "message": str(result.message), "nit": int(result.nit), "nfev": int(result.nfev),
            },
        })
        message = str(result.message).lower()
        row["optimizer_classification"] = (
            "converged" if result.success else
            "precision_loss" if result.status == 2 or "precision loss" in message else
            "iteration_limit" if result.status == 1 else "stopped_without_convergence"
        )
        if terminal_cert["qualified"]:
            fresh = LoggedSurface(
                factory(rc),
                cap=1, deadline=deadline, ledger_path=out / "fresh.jsonl", stage="independent_cold",
            )
            try:
                e2, f2, s2 = fresh.evaluate(final_atoms)
                row["fresh_EFS"] = {"energy_eV": e2, "forces_eV_A": f2, "stress_eV_A3": s2}
                row["fresh_certificate"] = _certificate(final_atoms, e2, f2, s2)
            finally:
                row["fresh_requests"] = fresh.requests
                row["fresh_failures"] = fresh.failures
                row["fresh_calculator_calculate_calls"] = fresh.calculator.calculate_calls
        row["status"] = "qualified" if row.get("fresh_certificate", {}).get("qualified") else "not_qualified"
        row["requests"] = surface.requests
        row["calculator_calculate_calls"] = surface.calculator.calculate_calls
        row["failed_requests"] = surface.failures
        row["total_calculator_calculate_calls"] = (
            surface.calculator.calculate_calls + row.get("fresh_calculator_calculate_calls", 0)
        )
        row["total_elapsed_s"] = time.monotonic() - (deadline - PROCESS_DEADLINE_S)
        _write_json(out / "slot.json", row)
        return row
    except SlotStop as exc:
        row.update({"status": "deadline" if "deadline" in str(exc) else "request_cap", "stop_reason": str(exc)})
    except Exception as exc:
        row.update({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
    row["requests"] = surface.requests
    row["calculator_calculate_calls"] = surface.calculator.calculate_calls
    row["failed_requests"] = surface.failures
    row["total_calculator_calculate_calls"] = (
        surface.calculator.calculate_calls + row.get("fresh_calculator_calculate_calls", 0)
    )
    row["total_elapsed_s"] = time.monotonic() - (deadline - PROCESS_DEADLINE_S)
    _write_json(out / "slot.json", row)
    return row


class DummyCalculator(Calculator):
    """Zero E/F/stress calculator used only by --preflight."""
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros(6),
        }


def preflight(output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to reuse preflight output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    slots = []
    for phase in ("fcc", "hcp"):
        for rc in RC_OVER_SIGMA:
            slot = output / f"{phase}-rc{rc:g}"
            slot.mkdir(parents=True, exist_ok=True)
            slots.append(str(slot))
    checks = []
    deadline = time.monotonic() + PROCESS_DEADLINE_S
    for phase in ("fcc", "hcp"):
        rc = RC_OVER_SIGMA[0]
        slot = output / f"{phase}-rc{rc:g}"
        result = run_slot(phase, rc, slot, deadline,
                          [sys.executable, str(Path(__file__).resolve()), "--preflight"],
                          calculator_factory=lambda _rc: DummyCalculator(), synthetic=True)
        checks.append({"phase": phase, "status": result.get("status"),
                       "synthetic_preflight": result.get("synthetic_preflight"),
                       "requests": result.get("requests", 0),
                       "calculator_calculate_calls": result.get("calculator_calculate_calls", 0),
                       "fresh_requests": result.get("fresh_requests", 0),
                       "fresh_calculator_calculate_calls": result.get("fresh_calculator_calculate_calls", 0),
                       "optimizer_exit": result.get("optimizer_exit"),
                       "terminal_certificate": result.get("terminal_certificate"),
                       "fresh_certificate": result.get("fresh_certificate")})
    protocol = Path(__file__).with_name("protocol.md")
    lj_imported = "ase.calculators.lj" in sys.modules
    report = {
        "status": "preflight_ok" if not lj_imported and all(x["status"] == "qualified" for x in checks) else "preflight_failed",
        "calculator": "DummyCalculator only; both phases ran synthetic full-slot path",
        "lennard_jones_constructed": False, "lennard_jones_module_imported": lj_imported,
        "slots_precreated": slots,
        "synthetic_slot_checks": checks, "script": Path(__file__).resolve(),
        "protocol": protocol.resolve(), "protocol_exists": protocol.exists(),
        "python": sys.executable, "ase": ase.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__,
        "vc_surface_source": str(Path(sys.modules[ASEStressSurface.__module__].__file__).resolve()),
    }
    _write_json(output / "preflight.json", report)
    print(json.dumps(_jsonable(report), indent=2))
    return 0 if report["status"] == "preflight_ok" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="dummy calculator only; no LJ construction")
    args = parser.parse_args(argv)
    if args.preflight:
        return preflight(args.output.resolve())

    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("actual LJ E/F/stress evaluations require a Slurm job; use --preflight for dummy-only checks")

    if not args.output.is_absolute():
        parser.error("--output must resolve to an absolute directory")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"refusing to reuse output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output)]
    # Create all slot directories before the first model calculation so failures
    # in one slot cannot hide later phase/cutoff slots.
    slots = [(phase, rc, output / f"{phase}-rc{rc:g}")
             for phase in ("fcc", "hcp") for rc in RC_OVER_SIGMA]
    for _, _, slot in slots:
        slot.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + PROCESS_DEADLINE_S
    results = []
    for phase, rc, slot in slots:
        if time.monotonic() >= deadline:
            slot.mkdir(parents=True, exist_ok=True)
            skipped = {"phase": phase, "rc_over_sigma": rc,
                       "status": "not_started_global_deadline", "requests": 0,
                       "fresh_requests": 0, "process_deadline_s": PROCESS_DEADLINE_S,
                       "protocol": str(Path(__file__).with_name("protocol.md").resolve()),
                       "executed_script": str(Path(__file__).resolve())}
            _write_json(slot / "slot.json", skipped)
            results.append(skipped)
            continue
        try:
            result = run_slot(phase, rc, slot, deadline, command)
        except Exception as exc:
            result = {"phase": phase, "rc_over_sigma": rc, "status": "setup_failed",
                      "requests": 0, "fresh_requests": 0, "error": repr(exc),
                      "traceback": traceback.format_exc(),
                      "protocol": str(Path(__file__).with_name("protocol.md").resolve()),
                      "executed_script": str(Path(__file__).resolve())}
            _write_json(slot / "slot.json", result)
        results.append(result)
        _write_json(output / "summary.json", {
            "status": "running", "results": results,
            "aggregate_requests": sum(int(x.get("requests", 0)) + int(x.get("fresh_requests", 0)) for x in results),
            "aggregate_calculator_calculate_calls": sum(
                int(x.get("calculator_calculate_calls", 0)) + int(x.get("fresh_calculator_calculate_calls", 0))
                for x in results),
            "global_deadline_s": PROCESS_DEADLINE_S,
        })
    delta = {}
    for rc in RC_OVER_SIGMA:
        fcc = next((r for r in results if r.get("phase") == "fcc" and r.get("rc_over_sigma") == rc), None)
        hcp = next((r for r in results if r.get("phase") == "hcp" and r.get("rc_over_sigma") == rc), None)
        if fcc and hcp and fcc.get("fresh_certificate", {}).get("qualified") and hcp.get("fresh_certificate", {}).get("qualified"):
            delta[str(rc)] = float(hcp["fresh_EFS"]["energy_eV"] / 2.0 - fcc["fresh_EFS"]["energy_eV"])
    summary = {
        "status": "completed" if len(results) == 8 and all(r.get("status") not in ("not_started_global_deadline", "deadline") for r in results) else "incomplete",
        "all_eight_slots_attempted": len(results) == 8 and all(r.get("status") != "not_started_global_deadline" for r in results),
        "results": results,
        "delta_e_hcp_minus_fcc_eV_atom": delta,
        "aggregate_requests": sum(int(x.get("requests", 0)) + int(x.get("fresh_requests", 0)) for x in results),
        "aggregate_calculator_calculate_calls": sum(
            int(x.get("calculator_calculate_calls", 0)) + int(x.get("fresh_calculator_calculate_calls", 0))
            for x in results),
        "aggregate_cap": 808,
        "interpretation": "Finite primitive-cell phase references on each declared ASE model only; not a global-minimum or paper-rate result.",
        "total_elapsed_s": time.monotonic() - (deadline - PROCESS_DEADLINE_S),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
