#!/usr/bin/env python3
"""Bounded Safe-total versus SciPy comparison on frozen biased VC objectives."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from ase import Atoms

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
PARITY = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity")
EVID = PARITY / "research/ga_ssw/evidence"
PHASE87 = EVID / "tio2-phase87-vc-pqc-single-step"
PHASE139 = EVID / "tio2-phase139-joint-memory-compare/memory400/result.json"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-omat-0-small.model")
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
MAX_TOTAL = 6000
INNER_CAP = 1000
QUENCH_CAP = 400
WALL_SECONDS = 1680  # keep two minutes inside the Slurm wall limit for output flush
CASES = ("phase87", "phase139")
METHODS = ("safe_total", "scipy_lbfgsb")
CORE_FILES = (
    "pamssw/standalone/vc_geometry.py",
    "pamssw/standalone/generalized_numerics.py",
    "pamssw/standalone/cell_relax.py",
    "pamssw/relax.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def atom_from_record(record: dict) -> Atoms:
    return Atoms(numbers=record["numbers"], positions=record["positions"],
                 cell=record["cell"], pbc=record["pbc"])


def _term(center, direction, weight, width):
    c = np.asarray(center, dtype=float)
    d = np.asarray(direction, dtype=float)
    w = float(width)
    h = float(weight)
    if c.ndim != 1 or d.shape != c.shape or not np.isfinite(c).all() or not np.isfinite(d).all():
        raise ValueError("Gaussian center/direction must be finite equal-length vectors")
    if not np.isfinite(h) or h <= 0 or not np.isfinite(w) or w <= 0:
        raise ValueError("Gaussian weight and width must be finite and positive")
    return {"center": c, "direction": d, "weight": h, "width": w}


def load_frozen_cases():
    """Load only archived records; perform no calculator or PES calls."""
    frozen_path = PHASE87 / "joint-stage8-memory-compare-v1/frozen-input.json"
    plan87_path = PHASE87 / "joint-stage8-memory-compare-v1/plan.json"
    source_result_path = PHASE87 / "joint/result.json"
    phase87_frozen = json.loads(frozen_path.read_text())
    plan87 = json.loads(plan87_path.read_text())
    width87 = float(plan87["objective_contract"]["width"])
    if sha256(frozen_path) != plan87["frozen_sha256"] or width87 != .6:
        raise ValueError("phase87 frozen input hash or archived Gaussian width differs from its plan")
    phase87_source = json.loads(source_result_path.read_text())
    phase87_diag_path = PHASE87 / "joint-stage8-diagnostic-v1/result.json"
    phase87_diag = json.loads(phase87_diag_path.read_text())
    if phase87_frozen.get("term_count") != 9 or len(phase87_frozen["previous_terms"]) != 9:
        raise ValueError("phase87 frozen objective must contain the archived nine terms")
    if phase87_frozen.get("saved_stage8_weight_reused") is not True:
        raise ValueError("phase87 provenance must retain the reused saved stage-8 weight")
    last87 = phase87_frozen["previous_terms"][-1]
    atoms87 = atom_from_record(phase87_source["records"][1]["chart_reference"])
    terms87 = [_term(t["center"], t["direction"], t["weight"], t.get("width", width87))
               for t in phase87_frozen["previous_terms"]]
    q87 = np.asarray(last87["center"], dtype=float) + width87 * np.asarray(last87["direction"], dtype=float)
    if not np.array_equal(q87, np.asarray(phase87_frozen["center_q"], dtype=float) +
                          width87 * np.asarray(last87["direction"], dtype=float)):
        raise ValueError("phase87 lifecycle start disagrees with the archived center_q construction")

    phase139_result = json.loads(PHASE139.read_text())
    phase139_record = phase139_result["records"][1]
    climb8 = next((x for x in phase139_record["climb"] if int(x["index"]) == 8), None)
    if climb8 is None:
        raise ValueError("phase139 saved record has no Gaussian index-8 entry")
    saved_terms = phase139_record["frozen_gaussians"]
    if len(saved_terms) != 9:
        raise ValueError("phase139 newly constructed objective must use all nine stored terms")
    terms139 = [_term(t["center"], t["direction"], t["weight"], t["width"])
                for t in saved_terms]
    last139 = terms139[-1]
    q139 = last139["center"] + last139["width"] * last139["direction"]
    atoms139 = atom_from_record(phase139_record["chart_reference"])
    source_config139 = phase139_result["joint_config"]
    return {
        "phase87": {
            "atoms": atoms87, "q0": q87, "terms": terms87, "pressure": 0.0,
            "strain_length": 5.0, "source_label": "constructed phase87 Gaussian-8 objective",
            "objective_provenance": "stage-8 direction reconstructed; archived weight reused",
            "saved_terminal_q": np.asarray(phase87_frozen["saved_stage8_failed_q"], dtype=float),
            "files": [frozen_path, source_result_path, phase87_diag_path,
                      plan87_path,
                      PHASE87 / "joint-stage8-diagnostic-v1/provenance.json"],
            "diagnostic_mode_residual": float(phase87_diag["mode"]["residual"]),
        },
        "phase139": {
            "atoms": atoms139, "q0": q139, "terms": terms139,
            "pressure": float(phase139_result["pressure"]),
            "strain_length": float(source_config139["strain_length"]),
            "source_label": "constructed phase139 Gaussian-8 objective",
            "objective_provenance": "all nine archived frozen terms; stage-8 q is terminal-candidate provenance only",
            "saved_terminal_q": np.asarray(climb8["q"], dtype=float),
            "files": [PHASE139],
            "saved_terminal_stage_status": climb8["status"],
            "saved_terminal_rotation_residual": climb8.get("rotation_residual"),
        },
    }


def qkey(q) -> bytes:
    return np.ascontiguousarray(np.asarray(q, dtype=np.float64)).tobytes()


def block_norm(gradient, natoms: int) -> float:
    g = np.asarray(gradient, dtype=float)
    if g.shape != (3 * natoms + 6,) or not np.isfinite(g).all():
        raise ValueError("gradient is not a finite generalized VC vector")
    atom = g[:-6].reshape(natoms, 3)
    return max(float(np.linalg.norm(atom, axis=1).max()), float(np.linalg.norm(g[-6:])))


class EvaluationCache:
    """Accepted-point callback data; lookups never call the PES."""
    def __init__(self):
        self.rows = {}

    def put(self, q, row):
        self.rows[qkey(q)] = row

    def exact(self, q):
        key = qkey(q)
        row = self.rows.get(key)
        if row is None or not np.array_equal(row["q"], np.asarray(q, dtype=float)):
            raise RuntimeError("SciPy callback q has no exact cached objective/gradient")
        return row


class PaidCallLimit(RuntimeError):
    pass


class CampaignLimit(RuntimeError):
    pass


class PaidOracle:
    """Counted E/F/stress surface with global, stage, wall, and failure records."""
    def __init__(self, calculator, cap_total=MAX_TOTAL, deadline=None, wall_seconds=WALL_SECONDS):
        from pamssw.standalone.vc_geometry import ASEStressSurface
        self.surface = ASEStressSurface(calculator)
        self.cap_total = cap_total
        self.deadline = time.monotonic() + wall_seconds if deadline is None else float(deadline)
        self.ledger = []
        self.total_paid = 0
        self.total_failed = 0
        self.total_denied = 0
        self.stage_paid = 0
        self.stage_attempts = 0
        self.stage_cap = None
        self.stage_name = "unset"
        self.calculator = calculator

    @property
    def requests(self):
        return self.surface.requests

    def begin_stage(self, name, cap):
        self.stage_name = name
        self.stage_cap = int(cap)
        self.stage_paid = 0
        self.stage_attempts = 0

    def evaluate(self, atoms, *, q=None, objective=None, role=None):
        self.stage_attempts += 1
        if time.monotonic() >= self.deadline:
            self.total_denied += 1
            self.ledger.append({"stage": self.stage_name, "role": role,
                                "paid": False, "denial": "28-minute_work_deadline",
                                "q_sha256": None if q is None else hashlib.sha256(qkey(q)).hexdigest()})
            raise CampaignLimit("28-minute paid-work deadline reached")
        if self.total_paid >= self.cap_total:
            self.total_denied += 1
            self.ledger.append({"stage": self.stage_name, "role": role,
                                "paid": False, "denial": "global_6000_efs_cap",
                                "q_sha256": None if q is None else hashlib.sha256(qkey(q)).hexdigest()})
            raise CampaignLimit("global 6000 EFS cap denied next request")
        if self.stage_cap is not None and self.stage_paid >= self.stage_cap:
            self.total_denied += 1
            self.ledger.append({"stage": self.stage_name, "role": role,
                                "paid": False, "denial": f"stage_cap_{self.stage_cap}",
                                "q_sha256": None if q is None else hashlib.sha256(qkey(q)).hexdigest()})
            raise PaidCallLimit(f"{self.stage_name} hard paid-request cap")
        self.stage_paid += 1
        before = self.surface.requests
        actual_before = getattr(self.calculator, "actual_calculate_calls", None)
        try:
            e, f, stress = self.surface.evaluate(atoms)
        except Exception as exc:
            actually_requested = self.surface.requests > before
            actual_calls = (getattr(self.calculator, "actual_calculate_calls", actual_before) - actual_before
                            if actual_before is not None else None)
            if actually_requested:
                self.total_paid += 1
                self.total_failed += 1
            self.ledger.append({"stage": self.stage_name, "role": role,
                                "paid": bool(actually_requested), "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "actual_calculator_calls": actual_calls,
                                "q_sha256": None if q is None else hashlib.sha256(qkey(q)).hexdigest()})
            raise
        self.total_paid += 1
        max_force = float(np.linalg.norm(f, axis=1).max())
        max_stress = float(np.abs(stress).max())
        actual_calls = (getattr(self.calculator, "actual_calculate_calls", actual_before) - actual_before
                        if actual_before is not None else None)
        self.ledger.append({"stage": self.stage_name, "role": role, "paid": True,
                            "request": self.surface.requests,
                            "actual_calculator_calls": actual_calls,
                            "q_sha256": None if q is None else hashlib.sha256(qkey(q)).hexdigest(),
                            "energy_eV": float(e), "max_force_eV_A": max_force,
                            "max_abs_stress_eV_A3": max_stress,
                            "objective_eV": None if objective is None else float(objective)})
        return e, f, stress


def gaussian_objective(chart, oracle, case, terms, pressure, cache=None, *, role="inner"):
    def evaluate(q):
        q = np.asarray(q, dtype=float).copy()
        ev = chart.evaluate(q, lambda atoms: oracle.evaluate(atoms, q=q, role=role),
                            pressure=pressure)
        energy = ev.objective
        gradient = chart.project(ev.gradient)
        for term in terms:
            direction = term["direction"]
            z = float((q - term["center"]) @ direction)
            bias = term["weight"] * math.exp(-.5 * (z / term["width"]) ** 2)
            energy += bias
            gradient -= bias * z / term["width"] ** 2 * direction
        if oracle.ledger and oracle.ledger[-1].get("q_sha256") == hashlib.sha256(qkey(q)).hexdigest():
            oracle.ledger[-1]["biased_objective_eV"] = float(energy)
            oracle.ledger[-1]["biased_block_norm_eV_A"] = block_norm(gradient, chart.natoms)
        row = {"q": q.copy(), "objective": float(energy), "gradient": gradient.copy(),
               "block_norm": block_norm(gradient, chart.natoms),
               "physical_energy": float(ev.energy), "max_force": float(np.linalg.norm(ev.forces, axis=1).max()),
               "max_stress": float(np.abs(ev.stress).max()), "atoms": ev.atoms.copy()}
        if cache is not None:
            cache.put(q, row)
        return row
    return evaluate


def validate_artifacts():
    if sha256(MODEL) != MODEL_SHA256:
        raise ValueError("OMAT-small model SHA256 mismatch")
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    cases = load_frozen_cases()
    rows = {}
    for name, case in cases.items():
        atoms, q, terms = case["atoms"], case["q0"], case["terms"]
        if len(atoms) != 48 or not atoms.pbc.all() or atoms.constraints:
            raise ValueError(f"{name}: expected unconstrained 48-atom periodic cell")
        if sorted(atoms.get_chemical_symbols()).count("O") != 32 or sorted(atoms.get_chemical_symbols()).count("Ti") != 16:
            raise ValueError(f"{name}: expected Ti16O32 composition")
        chart = SymmetricLogStrainChart(atoms, strain_length=case["strain_length"])
        if q.shape != (150,) or len(terms) != 9:
            raise ValueError(f"{name}: expected 150 coordinates and nine Gaussian terms")
        if any(t["center"].shape != (150,) or t["direction"].shape != (150,) for t in terms):
            raise ValueError(f"{name}: Gaussian vectors do not match 3N+6 chart")
        if case["saved_terminal_q"].shape != (150,) or not np.isfinite(case["saved_terminal_q"]).all():
            raise ValueError(f"{name}: saved terminal provenance q must be finite 150-vector")
        if not np.isfinite(q).all() or any(not np.isfinite(t["center"]).all() or
                not np.isfinite(t["direction"]).all() for t in terms):
            raise ValueError(f"{name}: start or Gaussian data is nonfinite")
        unpacked = chart.unpack(q)
        if not np.isfinite(unpacked.positions).all() or not np.isfinite(unpacked.cell.array).all():
            raise ValueError(f"{name}: reconstructed start is nonfinite")
        rows[name] = {"natoms": len(atoms), "formula": atoms.get_chemical_formula(),
                      "ndof": chart.ndof, "term_count": len(terms),
                      "q_sha256": hashlib.sha256(qkey(q)).hexdigest(),
                      "geometry": {"volume_A3": float(unpacked.get_volume()),
                                   "cell_determinant_A3": float(np.linalg.det(unpacked.cell.array))},
                      "source_files": {str(p): sha256(p) for p in case["files"]},
                      "objective_provenance": case["objective_provenance"],
                      "saved_terminal_stage_status": case.get("saved_terminal_stage_status")}
    import scipy
    from scipy.optimize import minimize
    if not callable(minimize):
        raise RuntimeError("SciPy L-BFGS-B entry point is unavailable")
    core = {p: sha256(REPO / p) for p in CORE_FILES}
    dirty_core = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", "pamssw"],
                                check=False).returncode
    if dirty_core:
        raise RuntimeError("core PAM-SSW files are dirty; source provenance is ambiguous")
    import pamssw
    if not Path(pamssw.__file__).resolve().is_relative_to(REPO.resolve()):
        raise RuntimeError(f"pamssw import is outside intended source root: {pamssw.__file__}")
    from mace.calculators import MACECalculator
    inspect.signature(MACECalculator).bind(model_paths=str(MODEL), device="cuda", default_dtype="float64")
    from pamssw.standalone.cell_relax import relax_cell_coordinates
    inspect.signature(relax_cell_coordinates).bind(
        object(), np.zeros(150), object(), pressure=0.0, fmax=.05,
        stress_tol=.001, max_step=.2, maxiter=300, lbfgs_memory=500)
    return {"status": "zero_pes_preflight_ok", "pes_calls": 0,
            "model": str(MODEL), "model_sha256": MODEL_SHA256,
            "cases": rows, "scipy_version": scipy.__version__,
            "scipy_options": {"maxcor": 500, "maxiter": 300, "maxfun": INNER_CAP,
                              "ftol": 2.220446049250313e-9, "gtol": 0.0, "maxls": 20},
            "runner_sha256": sha256(Path(__file__).resolve()),
            "source": {"git_head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
                       "pamssw_path": str(Path(pamssw.__file__).resolve()), "core_sha256": core}}


def run_safe_inner(q0, evaluate, natoms):
    from pamssw.standalone.generalized_numerics import safe_lbfgs
    cache = EvaluationCache()
    initial = evaluate(q0)
    cache.put(q0, initial)
    if initial["block_norm"] <= .1:
        return {"status": "initial_target_reached", "q": np.asarray(q0).copy(),
                "initial": initial, "accepted": [initial], "iterations": 0,
                "requests": 1, "shared_target_reached": True}
    used_initial = False
    def fn(q):
        nonlocal used_initial
        if not used_initial and np.array_equal(q, q0):
            used_initial = True
            return initial["objective"], initial["gradient"].copy()
        row = evaluate(q)
        cache.put(q, row)
        return row["objective"], row["gradient"]
    def norm(g):
        return block_norm(g, natoms)
    result = safe_lbfgs(q0, fn, gradient_norm=norm, step_norm=norm, gtol=.1,
                        max_step=.2, maxiter=300, max_requests=INNER_CAP,
                        lbfgs_memory=500)
    accepted = []
    for item in result.trace:
        row = cache.exact(item["q"])
        accepted.append({"q": row["q"].copy(), "objective": row["objective"],
                         "block_norm": row["block_norm"]})
    state = {"status": result.status, "error": result.error, "q": result.q.copy(),
             "initial": initial, "accepted": accepted, "iterations": result.steps,
             "requests": result.requests, "accepted_secants": result.accepted_secants,
             "rejected_secants": result.rejected_secants,
             "rejected_trials": result.rejected_trials,
             "shared_target_reached": result.status == "converged"}
    if isinstance(result.error, str) and "PaidCallLimit" in result.error:
        state["status"] = "request_limit"
    elif isinstance(result.error, str) and "CampaignLimit" in result.error:
        state["status"] = "campaign_limit"
    if result.energy is None:
        state["q"] = None
    return state


def run_scipy_inner(q0, evaluate, natoms):
    from scipy.optimize import minimize
    cache = EvaluationCache()
    initial = evaluate(q0)
    cache.put(q0, initial)
    if initial["block_norm"] <= .1:
        return {"status": "initial_target_reached", "q": np.asarray(q0).copy(),
                "initial": initial, "accepted": [initial], "iterations": 0,
                "requests": 1, "shared_target_reached": True}
    accepted = [{"q": initial["q"].copy(), "objective": initial["objective"],
                 "block_norm": initial["block_norm"]}]
    stop = {"shared_target": False}
    used_initial = False
    def fun(q):
        nonlocal used_initial
        if not used_initial and np.array_equal(q, q0):
            used_initial = True
            row = initial
        else:
            row = evaluate(q)
        cache.put(q, row)
        return row["objective"], row["gradient"].copy()
    def callback(xk):
        row = cache.exact(xk)
        accepted.append({"q": row["q"].copy(), "objective": row["objective"],
                         "block_norm": row["block_norm"]})
        if row["block_norm"] <= .1:
            stop["shared_target"] = True
            raise StopIteration
    try:
        result = minimize(fun, np.asarray(q0, dtype=float), jac=True, method="L-BFGS-B",
                          callback=callback,
                          options={"maxcor": 500, "maxiter": 300, "maxfun": INNER_CAP,
                                   "ftol": 2.220446049250313e-9, "gtol": 0.0, "maxls": 20})
        status, message = classify_scipy_result(result, stop["shared_target"])
        candidate = np.asarray(result.x, dtype=float)
        row = cache.exact(candidate)
        qlast = candidate.copy()
        if not np.array_equal(accepted[-1]["q"], qlast):
            accepted.append({"q": qlast.copy(), "objective": row["objective"],
                             "block_norm": row["block_norm"]})
        return {"status": status, "native_status": int(result.status),
                "message": str(result.message), "q": qlast, "initial": initial,
                "accepted": accepted, "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "shared_target_reached": bool(stop["shared_target"] or row["block_norm"] <= .1)}
    except PaidCallLimit as exc:
        return {"status": "request_limit", "message": str(exc), "q": accepted[-1]["q"].copy(),
                "initial": initial, "accepted": accepted,
                "iterations": len(accepted) - 1, "shared_target_reached": False}
    except CampaignLimit as exc:
        return {"status": "campaign_limit", "message": str(exc), "q": accepted[-1]["q"].copy(),
                "initial": initial, "accepted": accepted,
                "iterations": len(accepted) - 1, "shared_target_reached": False}
    except Exception:
        raise


def classify_scipy_result(result, callback_target):
    message = str(result.message).lower()
    if callback_target or "callback function requested stop" in message:
        return "shared_block_target", str(result.message)
    if "relative reduction of f" in message or "factr*epsmch" in message:
        return "relative_energy_stop", str(result.message)
    if ("maximum number of function evaluations" in message or
            "total no. of f,g evaluations exceeds limit" in message):
        return "request_limit", str(result.message)
    if "maximum number of iterations" in message or "total no. of iterations reached limit" in message:
        return "maxiter", str(result.message)
    if "line search" in message:
        return "line_search_stop", str(result.message)
    if bool(result.success):
        return "scipy_success_other", str(result.message)
    return "scipy_stop_other", str(result.message)


def atom_record(atoms):
    return {"numbers": atoms.numbers.tolist(), "positions": atoms.positions.tolist(),
            "cell": atoms.cell.array.tolist(), "pbc": atoms.pbc.tolist()}


def run_campaign(out: Path):
    if out.exists():
        raise FileExistsError(f"output exists; preserving it: {out}")
    out.mkdir(parents=True)
    total_start = time.monotonic()
    preflight = validate_artifacts()
    dump(out / "preflight.json", preflight)
    # Pin actual MACE constructor/import before first PES request.
    import torch
    import mace
    from mace.calculators import MACECalculator
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    from pamssw.standalone.cell_relax import relax_cell_coordinates
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this protocol requires its single V100 allocation")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("launch requires PYTHONNOUSERSITE=1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    class CountingMACECalculator(MACECalculator):
        def __init__(self, **kwargs):
            self.actual_calculate_calls = 0
            super().__init__(**kwargs)
        def calculate(self, *args, **kwargs):
            self.actual_calculate_calls += 1
            return super().calculate(*args, **kwargs)

    calc = CountingMACECalculator(model_paths=str(MODEL), device="cuda", default_dtype="float64")
    if sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("model hash changed between preflight and calculator construction")
    cases = load_frozen_cases()
    oracle = PaidOracle(calc, deadline=total_start + WALL_SECONDS)
    manifest = {"status": "running", "protocol": "frozen-biased-vc-local-optimizer-v1",
                "model_sha256": MODEL_SHA256, "method_order": list(METHODS),
                "case_order": list(CASES), "preflight": preflight,
                "runtime": {"python": sys.executable, "mace": mace.__version__,
                            "torch": torch.__version__, "numpy": np.__version__,
                            "scipy": importlib.metadata.version("scipy"),
                            "device": "cuda", "dtype": "float64",
                            "threads": torch.get_num_threads(),
                            "tf32": False, "core_sha256": preflight["source"]["core_sha256"]},
                "arms": [], "total_paid_efs": 0, "actual_calculator_calls": 0}
    dump(out / "manifest.json", manifest)
    campaign_error = None
    stop_campaign = False
    for case_name in CASES:
        case = cases[case_name]
        chart = SymmetricLogStrainChart(case["atoms"], strain_length=case["strain_length"])
        # One fresh qualification evaluation is shared as a case-level start check,
        # but is not cached into either optimizer's independent arm.
        oracle.begin_stage(f"{case_name}:start_qualification", 2)
        calc.reset()
        start_paid_before = oracle.total_paid
        start_actual_before = calc.actual_calculate_calls
        try:
            start_ev = chart.evaluate(case["q0"], lambda a: oracle.evaluate(a, q=case["q0"], role="start_qualification"),
                                      pressure=case["pressure"])
        except Exception as exc:
            campaign_error = {"case": case_name, "stage": "start_qualification",
                              "error": f"{type(exc).__name__}: {exc}"}
            stop_campaign = True
            break
        if not all(np.isfinite(x).all() for x in (start_ev.forces, start_ev.stress, start_ev.gradient)):
            campaign_error = {"case": case_name, "stage": "start_qualification", "error": "nonfinite fresh E/F/stress"}
            stop_campaign = True
            break
        start_meta = {"energy_eV": float(start_ev.energy), "objective_eV": float(start_ev.objective),
                      "max_force_eV_A": float(np.linalg.norm(start_ev.forces, axis=1).max()),
                      "max_abs_stress_eV_A3": float(np.abs(start_ev.stress).max()),
                      "q": case["q0"].tolist()}
        from ase.io import write
        case_dir = out / case_name
        case_dir.mkdir()
        write(case_dir / "start.extxyz", start_ev.atoms, format="extxyz")
        dump(case_dir / "frozen-case.json", {
            "case": case_name, "atoms": atom_record(case["atoms"]),
            "start_q": case["q0"].tolist(), "pressure": case["pressure"],
            "strain_length_A": case["strain_length"],
            "gaussians": [{"source_index": i, "center": t["center"].tolist(),
                           "direction": t["direction"].tolist(), "weight_eV": t["weight"],
                           "width_A": t["width"]} for i, t in enumerate(case["terms"])],
            "objective_provenance": case["objective_provenance"]})
        start_meta["paid_efs"] = oracle.total_paid - start_paid_before
        start_meta["actual_calculator_calls"] = calc.actual_calculate_calls - start_actual_before
        case_result = {"case": case_name, "objective_provenance": case["objective_provenance"],
                       "start_qualification": start_meta, "arms": []}
        for method in METHODS:
            arm_name = f"{case_name}-{method}"
            arm_dir = out / arm_name
            arm_dir.mkdir()
            oracle.begin_stage(f"{arm_name}:inner", INNER_CAP)
            calc.reset()
            cache = EvaluationCache()
            evaluate_row = gaussian_objective(chart, oracle, case_name, case["terms"],
                                               case["pressure"], cache, role="inner_biased")
            result = {"case": case_name, "method": method, "start": case["q0"].tolist(),
                      "objective_provenance": case["objective_provenance"],
                      "status": "running", "inner": None, "quench": None}
            ledger_start = len(oracle.ledger)
            paid_start = oracle.total_paid
            actual_start = calc.actual_calculate_calls
            try:
                # The arm's first fresh EFS is the inner objective at q0.
                inner_start = time.monotonic()
                if method == "safe_total":
                    inner = run_safe_inner(case["q0"], evaluate_row, chart.natoms)
                else:
                    inner = run_scipy_inner(case["q0"], evaluate_row, chart.natoms)
                result["inner"] = {k: v for k, v in inner.items() if k not in ("initial", "accepted", "q")}
                result["inner"]["logical_attempts"] = oracle.stage_attempts
                result["inner"]["elapsed_seconds"] = time.monotonic() - inner_start
                result["inner"]["initial"] = {"objective_eV": inner["initial"]["objective"],
                    "biased_block_norm_eV_A": inner["initial"]["block_norm"],
                    "physical_energy_eV": inner["initial"]["physical_energy"],
                    "max_force_eV_A": inner["initial"]["max_force"],
                    "max_abs_stress_eV_A3": inner["initial"]["max_stress"]}
                result["inner"]["accepted_trace"] = [
                    {"q": np.asarray(x["q"]).tolist(), "objective_eV": float(x["objective"]),
                     "biased_block_norm_eV_A": float(x["block_norm"])} for x in inner["accepted"]]
                inner_q = inner.get("q")
                if inner_q is None:
                    result["status"] = "inner_no_valid_state"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method,
                                      "error": "inner produced no valid accepted state"}
                    stop_campaign = True
                    break
                inner_q = np.asarray(inner_q, dtype=float)
                if not np.isfinite(inner_q).all():
                    result["status"] = "nonfinite_inner_state"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method, "error": "nonfinite inner state"}
                    stop_campaign = True
                    break
                from ase.io import write
                write(arm_dir / "inner-endpoint.extxyz", chart.unpack(inner_q), format="extxyz")
                # A calculator/backend error is fatal to the campaign. Ordinary
                # optimizer stops, including line-search/ftol/caps, hand off.
                if inner.get("status") == "campaign_limit":
                    result["status"] = "inner_campaign_truncated"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method,
                                      "error": inner.get("message") or inner.get("error")}
                    stop_campaign = True
                    break
                if inner.get("status") == "evaluation_failed":
                    result["status"] = "inner_evaluation_failed"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method,
                                      "error": inner.get("error") or inner.get("message")}
                    stop_campaign = True
                    break
                inner_paid_efs = oracle.total_paid - paid_start
                inner_start_seconds = result["inner"]["elapsed_seconds"]
                oracle.begin_stage(f"{arm_name}:unbiased_quench", QUENCH_CAP)
                calc.reset()
                quench_start = time.monotonic()
                quench = relax_cell_coordinates(chart, inner_q, oracle,
                    pressure=case["pressure"], fmax=.05, stress_tol=.001,
                    max_step=.2, maxiter=300, lbfgs_memory=500)
                qend = np.asarray(quench.q, dtype=float) if quench.q is not None else None
                qstatus = quench.status
                qerror = quench.error
                if isinstance(qerror, str) and "PaidCallLimit" in qerror:
                    qstatus = "request_limit"
                elif isinstance(qerror, str) and "CampaignLimit" in qerror:
                    qstatus = "campaign_limit"
                elif qstatus == "evaluation_failed":
                    result["status"] = "quench_evaluation_failed"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method, "error": qerror}
                    stop_campaign = True
                    break
                if qend is None or not np.isfinite(qend).all():
                    result["status"] = "quench_no_valid_state"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method, "error": "no valid quench state"}
                    stop_campaign = True
                    break
                write(arm_dir / "quench-endpoint.extxyz", chart.unpack(qend), format="extxyz")
                result["quench"] = {"status": qstatus, "error": qerror,
                    "optimizer_steps": quench.steps, "logical_requests": oracle.stage_attempts,
                    "endpoint_q": qend.tolist(), "energy_eV": quench.energy,
                    "biased_gradient_expected": False}
                quench_paid_efs = oracle.total_paid - paid_start - inner_paid_efs
                result["quench"]["paid_efs"] = quench_paid_efs
                result["quench"]["elapsed_seconds"] = time.monotonic() - quench_start
                if qstatus == "campaign_limit":
                    result["status"] = "quench_campaign_truncated"
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    campaign_error = {"case": case_name, "method": method, "error": qerror}
                    stop_campaign = True
                    break
                # One separately reset physical certificate, even for a capped
                # ordinary optimizer/quench exit with a valid state.
                oracle.begin_stage(f"{arm_name}:fresh_certificate", 1)
                calc.reset()
                try:
                    fresh = chart.evaluate(qend,
                        lambda a: oracle.evaluate(a, q=qend, role="fresh_final_certificate"),
                        pressure=case["pressure"])
                except Exception as exc:
                    result["status"] = ("final_certificate_truncated" if isinstance(exc, CampaignLimit)
                                         else "final_certificate_failed")
                    campaign_error = {"case": case_name, "method": method,
                                      "error": f"{type(exc).__name__}: {exc}"}
                    case_result["arms"].append(result)
                    dump(arm_dir / "result.json", result)
                    stop_campaign = True
                    break
                cert = {"energy_eV": float(fresh.energy), "enthalpy_eV": float(fresh.objective),
                        "max_force_eV_A": float(np.linalg.norm(fresh.forces, axis=1).max()),
                        "max_abs_stress_eV_A3": float(np.abs(fresh.stress).max())}
                cert["force_qualified"] = cert["max_force_eV_A"] <= .05
                cert["stress_qualified"] = cert["max_abs_stress_eV_A3"] <= .001
                cert["jointly_qualified"] = cert["force_qualified"] and cert["stress_qualified"]
                result["fresh_certificate"] = cert
                result["cost"] = {"inner_logical_requests": inner.get("logical_attempts"),
                    "inner_elapsed_seconds": inner_start_seconds,
                    "inner_paid_efs": inner_paid_efs,
                    "quench_paid_efs": result["quench"]["paid_efs"],
                    "certificate_paid_efs": 1,
                    "arm_paid_efs": oracle.total_paid - paid_start,
                    "arm_actual_calculator_calls": calc.actual_calculate_calls - actual_start}
                result["status"] = "completed"
                dump(arm_dir / "result.json", result)
                case_result["arms"].append(result)
            except CampaignLimit as exc:
                result["status"] = "campaign_limit"
                result["error"] = str(exc)
                case_result["arms"].append(result)
                dump(arm_dir / "result.json", result)
                campaign_error = {"case": case_name, "method": method, "error": str(exc)}
                stop_campaign = True
                break
            except Exception as exc:
                result["status"] = "backend_or_runner_error"
                result["error"] = f"{type(exc).__name__}: {exc}"
                result["traceback"] = traceback.format_exc()
                case_result["arms"].append(result)
                dump(arm_dir / "result.json", result)
                campaign_error = {"case": case_name, "method": method,
                                  "error": result["error"]}
                stop_campaign = True
                break
            finally:
                dump(arm_dir / "efs-ledger.json", oracle.ledger[ledger_start:])
        manifest["arms"].append(case_result)
        dump(out / "manifest.json", manifest)
        if stop_campaign:
            break
    error_text = "" if campaign_error is None else str(campaign_error.get("error", ""))
    is_censored = "CampaignLimit" in error_text or "deadline reached" in error_text or "global 6000 EFS cap" in error_text
    manifest["status"] = "censored" if is_censored else ("stopped_on_error" if stop_campaign else "completed")
    manifest["campaign_error"] = campaign_error
    manifest["total_paid_efs"] = oracle.total_paid
    manifest["failed_paid_efs"] = oracle.total_failed
    manifest["denied_unpaid_requests"] = oracle.total_denied
    manifest["actual_calculator_calls"] = calc.actual_calculate_calls
    manifest["elapsed_seconds"] = time.monotonic() - total_start
    recorded = {f"{case['case']}-{arm['method']}"
                for case in manifest["arms"] for arm in case["arms"]}
    manifest["not_run_arms"] = [f"{case}-{method}" for case in CASES for method in METHODS
                                 if f"{case}-{method}" not in recorded]
    dump(out / "manifest.json", manifest)
    dump(out / "campaign-ledger.json", oracle.ledger)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(validate_artifacts(), indent=2, allow_nan=False))
        return
    if args.out is None:
        parser.error("--run requires --out")
    run_campaign(args.out.resolve())


if __name__ == "__main__":
    main()
