#!/usr/bin/env python3
"""Bounded saved-state comparison of the existing joint-VC rotation solvers."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN_PATH = HERE / "plan.json"
PANEL_RUN = ROOT / "research/ga_ssw/evidence/tio2-vc-target-panel-20261007/run.py"
QUAL_RUN = ROOT / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run.py"
TOTAL_SEARCH = 320
TOTAL_FRESH = 20
TOTAL_REFERENCE = 84
TOTAL_SECONDS = 240.0
MODE_CAP = 40
METHODS = ("plane_dimer", "central_ritz")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_plan():
    plan = json.loads(PLAN_PATH.read_text())
    if (len(plan.get("cases", [])) != 2 or len(plan.get("methods", [])) != 2
            or tuple(plan["methods"]) != METHODS):
        raise ValueError("plan must define the two saved cases and existing solver pair")
    rot, budget = plan["rotation"], plan["budget"]
    expected_rotation = {"strain_length": 5.0, "pressure": 0.0,
        "rotation_bias": 1.0, "fd_step": 0.001, "tol": 0.02,
        "dimer_max_hvp": 39, "ritz_max_force_calls": 40}
    if rot != expected_rotation:
        raise ValueError(f"rotation contract differs from frozen plan: {rot!r}")
    if budget != {"total_search_requests": TOTAL_SEARCH,
                  "total_fresh_requests": TOTAL_FRESH, "total_seconds": TOTAL_SECONDS,
                  "total_reference_requests": TOTAL_REFERENCE}:
        raise ValueError(f"budget differs from frozen plan: {budget!r}")
    if (len(plan["cases"]) * 2 * len(METHODS) * MODE_CAP != TOTAL_SEARCH
            or plan.get("reference") != {"case": "phase87_12", "coordinate_columns": 42,
                "finite_difference": "central", "fd_step": 0.001,
                "project_translation": True,
                "interpretation": "local rotation reference, not Hessian minimum certificate"}):
        raise ValueError("mode caps do not close to the plan search budget")
    return plan


def validate_inputs(plan):
    from ase.io import read
    sources = []
    for case in plan["cases"]:
        for field, expected_hash in (("input", case["input_sha256"]),):
            if sha256(case[field]) != expected_hash:
                raise RuntimeError(f"{case['id']} {field} hash mismatch")
        atoms = read(case["input"])
        if (len(atoms) != case["natoms"] or not atoms.pbc.all()
                or atoms.constraints or atoms.get_chemical_formula() !=
                ("O8Ti4" if case["natoms"] == 12 else "O32Ti16")):
            raise ValueError(f"{case['id']} endpoint violates saved-state contract")
        result = json.loads(Path(case["source_result"]).read_text())
        provenance = json.loads(Path(case["source_provenance"]).read_text())
        if (result.get("case") != case["id"] or result.get("arm") != "joint_vc"
                or result.get("status") != "completed"
                or result.get("seed") != case["seed"]
                or not result.get("numerically_qualified_initial")):
            raise ValueError(f"{case['id']} source run does not qualify as the joint endpoint")
        if (Path(provenance["model"]) != Path(plan["model"])
                or provenance["model_sha256"] != plan["model_sha256"]):
            raise ValueError(f"{case['id']} source model differs from probe model")
        for name in ("vc_reference", "generalized_numerics", "vc_geometry",
                     "runner", "qualification_helper"):
            source_path = Path(provenance["imports"][name])
            if sha256(source_path) != provenance["source_sha256"][name]:
                raise RuntimeError(f"saved panel source hash no longer matches {name}")
        sources.append((atoms, result, provenance))
    if sha256(plan["model"]) != plan["model_sha256"]:
        raise RuntimeError("OMAT-small model hash mismatch")
    return sources


def provenance(plan, helper, sources):
    imports = {name: str(Path(importlib.import_module(name).__file__).resolve())
               for name in ("ase", "mace", "torch", "pamssw.standalone.vc_reference",
                            "pamssw.standalone.generalized_numerics",
                            "pamssw.standalone.generalized_ritz",
                            "pamssw.standalone.direction",
                            "pamssw.standalone.vc_geometry")}
    inputs = []
    for case, (_, result, prior) in zip(plan["cases"], sources):
        inputs.append({"id": case["id"], "endpoint": case["input"],
            "endpoint_sha256": sha256(case["input"]), "source_result": case["source_result"],
            "source_result_sha256": sha256(case["source_result"]),
            "source_provenance": case["source_provenance"],
            "source_provenance_sha256": sha256(case["source_provenance"]),
            "source_panel_head": prior.get("git_head"),
            "source_panel_core_sha256": prior.get("source_sha256", {}),
            "prior_status": result.get("status"), "prior_attempts": result.get("outer_attempts_recorded")})
    return {"checkout": str(ROOT),
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.executable, "python_version": platform.python_version(),
        "cwd": os.getcwd(), "imports": imports,
        "source_sha256": {name: sha256(path) for name, path in {
            "probe_runner": Path(__file__), "panel_runner": PANEL_RUN,
            "qualification_helper": QUAL_RUN,
            "vc_reference": imports["pamssw.standalone.vc_reference"],
            "generalized_numerics": imports["pamssw.standalone.generalized_numerics"],
            "generalized_ritz": imports["pamssw.standalone.generalized_ritz"],
            "direction": imports["pamssw.standalone.direction"],
            "vc_geometry": imports["pamssw.standalone.vc_geometry"]}.items()},
        "plan_sha256": sha256(PLAN_PATH), "model": plan["model"],
        "model_sha256": sha256(plan["model"]), "cases": inputs,
        "rotation": plan["rotation"], "budget": plan["budget"],
        "methods": list(plan["methods"])}


def _mode_call(method, q0, anchor, evaluate, rotation):
    from pamssw.standalone.generalized_numerics import generalized_central_ritz, generalized_dimer
    kwargs = {"rotation_bias": rotation["rotation_bias"],
              "fd_step": rotation["fd_step"], "tol": rotation["tol"],
              "evaluate": evaluate}
    if method == "plane_dimer":
        return generalized_dimer(q0, anchor, max_hvp=rotation["dimer_max_hvp"], **kwargs)
    return generalized_central_ritz(q0, anchor,
        max_force_calls=rotation["ritz_max_force_calls"], **kwargs)


def _call_count(budget):
    return {"requests": budget.requests, "calculator_calls": budget.calculator_calls,
            "failures": budget.failures, "denials": budget.denials,
            "boundary": budget.boundary}


def _ledger_closure_from_file(ledger):
    events = [json.loads(line) for line in Path(ledger).read_text().splitlines() if line.strip()]
    paid = [row for row in events if row.get("event") in {"evaluation", "failure"}]
    recognized = len(paid) + sum(row.get("event") == "denial" for row in events)
    return {"closed": recognized == len(events),
            "paid_events": len(paid),
            "calculator_calls": sum(int(row.get("calculator_calls", 0) or 0) for row in paid),
            "failures": sum(row.get("event") == "failure" for row in events),
            "denials": sum(row.get("event") == "denial" for row in events),
            "events": len(events)}


def _ledger_closure(ledger, budget):
    observed = _ledger_closure_from_file(ledger)
    observed.update(requests=budget.requests, budget_calculator_calls=budget.calculator_calls,
        closed=(observed["paid_events"] == budget.requests and
                observed["calculator_calls"] == budget.calculator_calls and
                observed["failures"] == budget.failures and
                observed["denials"] == budget.denials))
    return observed


class BudgetRouter:
    """Let the shared helper instrument one calculator wrapper exactly once."""
    def __init__(self):
        self.active = None

    @property
    def calculator_calls(self):
        return 0 if self.active is None else self.active.calculator_calls

    @calculator_calls.setter
    def calculator_calls(self, value):
        if self.active is None:
            raise RuntimeError("calculator called without an active counted budget")
        self.active.calculator_calls = value


def _translation_activity_basis(natoms):
    import numpy as np
    from scipy.linalg import null_space
    dim = 3 * natoms + 6
    translations = np.zeros((dim, 3))
    for axis in range(3):
        translations[axis:3 * natoms:3, axis] = 1.0 / math.sqrt(natoms)
    return null_space(translations.T)


def _full_hessian_reference(q0, evaluate, rotation, basis):
    """42 physical-Hessian columns, then restrict to the translation-free space."""
    import numpy as np
    h = float(rotation["fd_step"])
    center = np.asarray(q0, dtype=float)
    dim = center.size
    matrix = np.empty((dim, dim), dtype=float)
    for column in range(dim):
        axis = np.zeros(dim); axis[column] = 1.0
        gradients = []
        for sign in (1.0, -1.0):
            q = center + sign * h * axis
            _energy, gradient = evaluate(q)
            gradients.append(np.asarray(gradient))
        matrix[:, column] = (gradients[0] - gradients[1]) / (2.0 * h)
    sym = 0.5 * (matrix + matrix.T)
    restricted_raw = basis.T @ matrix @ basis
    restricted = basis.T @ sym @ basis
    return {"hessian_raw": matrix, "hessian_symmetric": sym,
        "projected_restricted_raw": restricted_raw,
        "projected_restricted_symmetric": restricted,
        "full_asymmetry_frobenius_eV_A2": float(np.linalg.norm(matrix - matrix.T)),
        "projected_asymmetry_frobenius_eV_A2": float(np.linalg.norm(restricted_raw - restricted_raw.T))}


def _reference_mode(hessian, basis, anchor, rotation):
    import numpy as np
    n0 = np.asarray(anchor, dtype=float) / np.linalg.norm(anchor)
    biased = hessian - float(rotation["rotation_bias"]) * np.outer(n0, n0)
    restricted = basis.T @ (0.5 * (biased + biased.T)) @ basis
    eigenvalues, eigenvectors = np.linalg.eigh(restricted)
    direction = basis @ eigenvectors[:, 0]
    direction /= np.linalg.norm(direction)
    if np.dot(direction, n0) < 0:
        direction = -direction
    return direction, float(eigenvalues[0]), eigenvalues, restricted


def _cold_residual(q0, anchor, direction, evaluate, rotation):
    import numpy as np
    h = float(rotation["fd_step"])
    n0 = np.asarray(anchor, dtype=float) / np.linalg.norm(anchor)
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    gradients = []
    for sign in (1.0, -1.0):
        q = q0 + sign * h * direction
        _energy, gradient = evaluate(q)
        projection = float(np.dot(q - q0, n0))
        gradients.append(np.asarray(gradient) - rotation["rotation_bias"] * projection * n0)
    hv = (gradients[0] - gradients[1]) / (2.0 * h)
    curvature = float(direction @ hv)
    residual = hv - curvature * direction
    residual_norm = float(np.linalg.norm(residual))
    return {"curvature_eV_A2": curvature,
            "residual_eV_A2": residual_norm,
            "residual_angle_deg": math.degrees(math.atan2(residual_norm, abs(curvature)))}


def _preflight(helper, plan):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart

    class Dummy(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if self.atoms.info.get("fail", False):
                raise RuntimeError("intentional counted dummy failure")
            self.results = {"energy": 0.0, "forces": np.zeros_like(self.atoms.positions),
                            "stress": np.zeros(6)}

    import tempfile
    from pathlib import Path
    started = time.monotonic()
    deadline = started + 30.0
    dummy_atoms = Atoms("Ti4O8", positions=np.array([[i, 0., 0.] for i in range(12)]),
                        cell=np.eye(3) * 12, pbc=True)
    chart = SymmetricLogStrainChart(dummy_atoms, strain_length=plan["rotation"]["strain_length"])
    q0 = chart.pack(dummy_atoms)
    anchor = chart.project(np.arange(1, q0.size + 1, dtype=float)); anchor /= np.linalg.norm(anchor)
    checks = []
    with tempfile.TemporaryDirectory(prefix="tio2-vc-rotation-preflight-") as td:
        for method in METHODS:
            budget = helper.CaseBudget(method, Path(td) / f"{method}.jsonl", 40, 30., deadline)
            calc = Dummy(); helper.instrument_calculate(calc, budget)
            surface = helper.CountedStressSurface(calc, budget, "dummy_mode")
            def evaluate(q):
                ev = chart.evaluate(q, surface.evaluate, pressure=0.)
                return ev.objective, chart.project(ev.gradient)
            mode = _mode_call(method, q0, anchor.copy(), evaluate, plan["rotation"])
            before = budget.requests
            direction = np.asarray(mode.direction, dtype=float)
            for sign in (1., -1.):
                chart.evaluate(q0 + sign * plan["rotation"]["fd_step"] * direction,
                               surface.evaluate, pressure=0.)
            if budget.requests - before != 2 or budget.requests > 42:
                raise RuntimeError(f"{method} preflight central residual cost mismatch")
            if not _ledger_closure(Path(td) / f"{method}.jsonl", budget)["closed"]:
                raise RuntimeError(f"{method} preflight request/call ledger did not close")
            checks.append({"method": method, "mode": _call_count(budget),
                           "hvp_calls": int(mode.hvp_calls), "converged": bool(mode.converged),
                           "cold_residual_requests": 2})
        ref_budget = helper.CaseBudget("hessian_reference", Path(td) / "reference.jsonl",
                                      TOTAL_REFERENCE, 30., deadline)
        ref_calc = Dummy(); helper.instrument_calculate(ref_calc, ref_budget)
        ref_surface = helper.CountedStressSurface(ref_calc, ref_budget, "dummy_reference")
        def ref_evaluate(q):
            ev = chart.evaluate(q, ref_surface.evaluate, pressure=0.)
            return ev.objective, chart.project(ev.gradient)
        basis = _translation_activity_basis(len(dummy_atoms))
        reference = _full_hessian_reference(q0, ref_evaluate, plan["rotation"], basis)
        if ref_budget.requests != TOTAL_REFERENCE or basis.shape != (q0.size, q0.size - 3):
            raise RuntimeError("preflight full-Hessian dimension/cost mismatch")
        if not _ledger_closure(Path(td) / "reference.jsonl", ref_budget)["closed"]:
            raise RuntimeError("preflight reference ledger did not close")
        ref_direction, ref_curvature, _, _ = _reference_mode(
            reference["hessian_symmetric"], basis, anchor, plan["rotation"])
        cold_ref_budget = helper.CaseBudget("reference_cold", Path(td) / "reference-cold.jsonl",
                                            2, 30., deadline)
        cold_ref_calc = Dummy(); helper.instrument_calculate(cold_ref_calc, cold_ref_budget)
        cold_ref_surface = helper.CountedStressSurface(cold_ref_calc, cold_ref_budget, "dummy_reference_cold")
        def cold_ref_evaluate(q):
            ev = chart.evaluate(q, cold_ref_surface.evaluate, pressure=0.)
            return ev.objective, chart.project(ev.gradient)
        # Use a separate counted oracle for the independently charged two-point check.
        cold = _cold_residual(q0, anchor, ref_direction, cold_ref_evaluate, plan["rotation"])
        if (cold_ref_budget.requests != 2 or
                not np.isfinite(ref_curvature + cold["residual_eV_A2"])):
            raise RuntimeError("preflight reference-mode cold residual mismatch")
        if not _ledger_closure(Path(td) / "reference-cold.jsonl", cold_ref_budget)["closed"]:
            raise RuntimeError("preflight reference cold ledger did not close")
        fail_budget = helper.CaseBudget("failure", Path(td) / "failure.jsonl", 1, 30., deadline)
        fail_calc = Dummy(); helper.instrument_calculate(fail_calc, fail_budget)
        fail_surface = helper.CountedStressSurface(fail_calc, fail_budget, "dummy_failure")
        bad = dummy_atoms.copy(); bad.info["fail"] = True
        try:
            fail_surface.evaluate(bad)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("preflight dummy EFS failure was swallowed")
        if fail_budget.requests != 1 or fail_budget.failures != 1:
            raise RuntimeError("preflight failed-call accounting mismatch")
        denied = False
        try:
            fail_surface.evaluate(dummy_atoms)
        except RuntimeError:
            denied = True
        if not denied or fail_budget.denials != 1 or fail_budget.requests != 1:
            raise RuntimeError("preflight budget-denial accounting mismatch")
        if not _ledger_closure(Path(td) / "failure.jsonl", fail_budget)["closed"]:
            raise RuntimeError("preflight failure/denial ledger did not close")
        router_calc = Dummy(); router = BudgetRouter()
        helper.instrument_calculate(router_calc, router)
        for index in range(2):
            routed = helper.CaseBudget(f"routed-{index}", Path(td) / f"routed-{index}.jsonl",
                                       1, 30., deadline)
            router.active = routed
            router_calc.reset()
            helper.CountedStressSurface(router_calc, routed, "router").evaluate(dummy_atoms.copy())
            if routed.requests != 1 or routed.calculator_calls != 1:
                raise RuntimeError("budget-router preflight attribution failed")
        router.active = None
    return {"real_pes_requests": 0, "mode_checks": checks,
            "hessian_reference": {"requests": ref_budget.requests,
                "calculator_calls": ref_budget.calculator_calls,
                "columns": q0.size, "activity_basis_columns": basis.shape[1],
                "cold_residual_requests": cold_ref_budget.requests},
            "failure_case": _call_count(fail_budget), "budget_router_cases": 2,
            "cap_denial_checked": True}


def execute(plan, output):
    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart

    sys.path.insert(0, str(ROOT))
    helper = load_module(QUAL_RUN, "tio2_vc_rotation_qualification_helpers")
    sources = validate_inputs(plan)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-snapshot").mkdir()
    prov = provenance(plan, helper, sources)
    for name, source in (("runner.py", Path(__file__)), ("panel-runner.py", PANEL_RUN),
                         ("qualification-helper.py", QUAL_RUN),
                         ("plan.json", PLAN_PATH), ("protocol.md", HERE / "protocol.md")):
        shutil.copy2(source, output / "source-snapshot" / name)
    for name in ("vc_reference", "generalized_numerics", "generalized_ritz", "direction", "vc_geometry"):
        path = Path(prov["imports"][f"pamssw.standalone.{name}"])
        shutil.copy2(path, output / "source-snapshot" / path.name)
    (output / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    (output / "effective-plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calc_kw = {"model_paths": plan["model"], "head": plan["calculator"]["head"],
        "device": plan["calculator"]["device"], "default_dtype": plan["calculator"]["dtype"],
        "enable_cueq": plan["calculator"]["enable_cueq"],
        "enable_oeq": plan["calculator"]["enable_oeq"]}
    started = time.monotonic()
    deadline = started + TOTAL_SECONDS
    rows, arrays, contexts = [], {}, {}
    search_paid = fresh_paid = reference_paid = 0
    calc = MACECalculator(**calc_kw)
    search_router = BudgetRouter()
    helper.instrument_calculate(calc, search_router)
    for case, (atoms, _source_result, _source_prov) in zip(plan["cases"], sources):
        atoms.calc = None
        chart = SymmetricLogStrainChart(atoms, strain_length=plan["rotation"]["strain_length"])
        q0 = chart.pack(atoms)
        rng = np.random.default_rng(int(case["seed"]))
        for anchor_index in case["outer_anchor_indices"]:
            anchor = chart.project(rng.normal(size=q0.size))
            anchor /= np.linalg.norm(anchor)
            prefix = f"{case['id']}_anchor{anchor_index}"
            contexts[prefix] = {"case": case, "chart": chart, "q0": q0.copy(), "anchor": anchor.copy()}
            arrays[prefix + "_q0"] = q0.copy()
            arrays[prefix + "_anchor"] = anchor.copy()
            for method in METHODS:
                mode_id = f"{prefix}_{method}"
                ledger = output / f"requests-{mode_id}.jsonl"
                budget = helper.CaseBudget(mode_id, ledger, MODE_CAP, TOTAL_SECONDS, deadline)
                search_router.active = budget
                calc.reset()
                surface = helper.CountedStressSurface(calc, budget, "rotation_search")
                def evaluate(q):
                    ev = chart.evaluate(q, surface.evaluate, pressure=plan["rotation"]["pressure"])
                    return ev.objective, chart.project(ev.gradient)
                row = {"case": case["id"], "natoms": case["natoms"],
                    "seed": case["seed"], "anchor_index": anchor_index, "method": method,
                    "q0_array_key": prefix + "_q0", "anchor_array_key": prefix + "_anchor",
                    "direction_array_key": mode_id + "_direction",
                    "search_ledger": ledger.name, "search_status": "running"}
                try:
                    mode = _mode_call(method, q0, anchor.copy(), evaluate, plan["rotation"])
                    direction = np.asarray(mode.direction, dtype=float).copy()
                    arrays[mode_id + "_direction"] = direction
                    atomic, strain = direction[:-6], direction[-6:]
                    cosine = float(np.clip(np.dot(direction, anchor) /
                        (np.linalg.norm(direction) * np.linalg.norm(anchor)), -1., 1.))
                    row.update(search_status="completed", converged=bool(mode.converged),
                        stop_reason=getattr(mode, "stop_reason", None),
                        curvature_eV_A2=float(mode.curvature),
                        residual_eV_A2=float(mode.residual_norm),
                        projected_symmetry_error_eV_A2=float(mode.projected_symmetry_error),
                        solver_hvp_calls=int(mode.hvp_calls), solver_evaluate_calls=int(mode.force_calls),
                        direction_anchor_angle_deg=math.degrees(math.acos(cosine)),
                        direction_atomic_norm=float(np.linalg.norm(atomic)),
                        direction_strain_coordinate_norm=float(np.linalg.norm(strain)),
                        direction_atomic_fraction=float(np.linalg.norm(atomic) / np.linalg.norm(direction)),
                        direction_strain_fraction=float(np.linalg.norm(strain) / np.linalg.norm(direction)))
                except Exception as error:
                    row.update(search_status="failed", error=repr(error), traceback=traceback.format_exc())
                row["search_cost"] = _call_count(budget)
                row["search_ledger_sha256"] = sha256(ledger)
                row["search_ledger_closure"] = _ledger_closure(ledger, budget)
                rows.append(row)
                search_paid += budget.requests
                search_router.active = None
    del calc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # One common 42-column physical Hessian at the 12-atom center; each anchor
    # adds its own analytic rank-one term before the translation-free eigensolve.
    reference_case = next(c for c in plan["cases"] if c["id"] == "phase87_12")
    reference_atoms = read(reference_case["input"]); reference_atoms.calc = None
    reference_chart = SymmetricLogStrainChart(reference_atoms,
        strain_length=plan["rotation"]["strain_length"])
    reference_q0 = reference_chart.pack(reference_atoms)
    activity_basis = _translation_activity_basis(len(reference_atoms))
    ref_calc = MACECalculator(**calc_kw)
    ref_router = BudgetRouter(); reference_id = "phase87_12_full_hessian"
    ref_ledger = output / "requests-phase87_12_full_hessian.jsonl"
    ref_budget = helper.CaseBudget(reference_id, ref_ledger, TOTAL_REFERENCE, TOTAL_SECONDS, deadline)
    ref_router.active = ref_budget
    helper.instrument_calculate(ref_calc, ref_router)
    reference_surface = helper.CountedStressSurface(ref_calc, ref_budget, "central_full_hessian")
    def reference_evaluate(q):
        ev = reference_chart.evaluate(q, reference_surface.evaluate,
                                      pressure=plan["rotation"]["pressure"])
        return ev.objective, reference_chart.project(ev.gradient)
    reference_row = {"case": "phase87_12", "natoms": 12,
        "coordinate_dimension": int(reference_q0.size),
        "activity_dimension_after_translation_projection": int(activity_basis.shape[1]),
        "finite_difference": "central", "fd_step_A": plan["rotation"]["fd_step"],
        "request_cap": TOTAL_REFERENCE, "ledger": ref_ledger.name, "status": "running"}
    reference_data = None
    try:
        reference_data = _full_hessian_reference(reference_q0, reference_evaluate,
                                                  plan["rotation"], activity_basis)
        reference_row.update(status="completed", columns_computed=int(reference_data["hessian_raw"].shape[1]),
            full_asymmetry_frobenius_eV_A2=reference_data["full_asymmetry_frobenius_eV_A2"],
            projected_asymmetry_frobenius_eV_A2=reference_data["projected_asymmetry_frobenius_eV_A2"],
            raw_hessian_array_key="full_reference_hessian_raw",
            symmetric_hessian_array_key="full_reference_hessian_symmetric",
            projected_raw_array_key="full_reference_projected_raw",
            projected_symmetric_array_key="full_reference_projected_symmetric",
            activity_basis_array_key="full_reference_activity_basis",
            request_cost=_call_count(ref_budget))
        arrays["full_reference_hessian_raw"] = reference_data["hessian_raw"]
        arrays["full_reference_hessian_symmetric"] = reference_data["hessian_symmetric"]
        arrays["full_reference_projected_raw"] = reference_data["projected_restricted_raw"]
        arrays["full_reference_projected_symmetric"] = reference_data["projected_restricted_symmetric"]
        arrays["full_reference_activity_basis"] = activity_basis
        reference_rows = []
        for anchor_index in reference_case["outer_anchor_indices"]:
            prefix = f"{reference_case['id']}_anchor{anchor_index}"
            anchor = arrays[prefix + "_anchor"]
            direction, eigenvalue, eigenvalues, biased_projected = _reference_mode(
                reference_data["hessian_symmetric"], activity_basis,
                anchor, plan["rotation"])
            mode_id = f"{prefix}_full_hessian_reference"
            arrays[mode_id + "_direction"] = direction
            arrays[mode_id + "_eigenvalues"] = eigenvalues
            arrays[mode_id + "_projected_biased_matrix"] = biased_projected
            reference_rows.append({"case": reference_case["id"], "anchor_index": anchor_index,
                "method": "full_hessian_reference", "curvature_eV_A2": eigenvalue,
                "q0_array_key": f"{prefix}_q0",
                "anchor_array_key": f"{prefix}_anchor",
                "direction_array_key": mode_id + "_direction",
                "eigenvalues_array_key": mode_id + "_eigenvalues",
                "projected_biased_matrix_key": mode_id + "_projected_biased_matrix"})
        reference_row["modes"] = reference_rows
    except Exception as error:
        reference_row.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                             request_cost=_call_count(ref_budget))
    reference_row["ledger_closure"] = _ledger_closure(ref_ledger, ref_budget)
    reference_row["ledger_sha256"] = sha256(ref_ledger)
    reference_paid = ref_budget.requests
    ref_router.active = None
    del ref_calc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # All direct directional checks use a distinct calculator, budget and ledger.
    fresh_calc = MACECalculator(**calc_kw)
    fresh_router = BudgetRouter(); helper.instrument_calculate(fresh_calc, fresh_router)
    context_by_id = {key: value for key, value in contexts.items()}
    cold_rows = []
    modes_to_check = []
    for row in rows:
        if row["search_status"] == "completed":
            key = f"{row['case']}_anchor{row['anchor_index']}"
            modes_to_check.append((row, key, row["method"],
                arrays[f"{key}_{row['method']}_direction"], "iterative"))
    if reference_data is not None:
        for ref_mode in reference_row.get("modes", []):
            key = f"{ref_mode['case']}_anchor{ref_mode['anchor_index']}"
            modes_to_check.append((ref_mode, key, "full_hessian_reference",
                arrays[f"{key}_full_hessian_reference_direction"], "reference"))
    for row, key, method, direction, source_kind in modes_to_check:
        context = context_by_id[key]
        chart, q0, anchor = context["chart"], context["q0"], context["anchor"]
        ledger = output / f"fresh-{key}_{method}.jsonl"
        budget = helper.CaseBudget(f"fresh-{key}_{method}", ledger, 2, TOTAL_SECONDS, deadline)
        fresh_router.active = budget
        fresh_calc.reset()
        surface = helper.CountedStressSurface(fresh_calc, budget, "central_directional_residual")
        def cold_evaluate(q):
            ev = chart.evaluate(q, surface.evaluate, pressure=plan["rotation"]["pressure"])
            return ev.objective, chart.project(ev.gradient)
        record = {"case_anchor": key, "method": method, "direction_source": source_kind,
                  "ledger": ledger.name, "status": "running"}
        try:
            record.update(_cold_residual(q0, anchor, direction, cold_evaluate, plan["rotation"]))
            record["status"] = "completed"
        except Exception as error:
            record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        record["cost"] = _call_count(budget)
        record["ledger_closure"] = _ledger_closure(ledger, budget)
        record["ledger_sha256"] = sha256(ledger)
        fresh_paid += budget.requests
        fresh_router.active = None
        if method == "full_hessian_reference":
            row["cold_check"] = record
        else:
            row.update(cold_status=record["status"], cold_check=record)
        cold_rows.append(record)
    del fresh_calc

    arrays_path = output / "mode-arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    rows_path = output / "mode-records.jsonl"
    rows_path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    (output / "reference.json").write_text(json.dumps(reference_row, indent=2, allow_nan=False) + "\n")
    (output / "fresh-records.jsonl").write_text("".join(
        json.dumps(row, allow_nan=False) + "\n" for row in cold_rows))
    ledger_paths = sorted(output.glob("requests-*.jsonl")) + sorted(output.glob("fresh-*.jsonl"))
    ledger_closures = {path.name: _ledger_closure_from_file(path) for path in ledger_paths}
    closure_checks = ([row.get("search_ledger_closure", {}).get("closed", False) for row in rows]
        + [reference_row.get("ledger_closure", {}).get("closed", False)]
        + [record.get("ledger_closure", {}).get("closed", False) for record in cold_rows])
    summary = {"status": "completed", "claim_scope": "saved-state local rotation diagnosis only",
        "search_paid_requests": search_paid, "search_request_cap": TOTAL_SEARCH,
        "search_requests_within_cap": search_paid <= TOTAL_SEARCH,
        "reference_paid_requests": reference_paid, "reference_request_cap": TOTAL_REFERENCE,
        "reference_requests_within_cap": reference_paid <= TOTAL_REFERENCE,
        "fresh_paid_requests": fresh_paid, "fresh_request_cap": TOTAL_FRESH,
        "fresh_requests_within_cap": fresh_paid <= TOTAL_FRESH,
        "total_paid_requests": search_paid + reference_paid + fresh_paid,
        "elapsed_seconds": time.monotonic() - started,
        "total_wall_cap_seconds": TOTAL_SECONDS,
        "mode_record_count": len(rows), "fresh_record_count": len(cold_rows),
        "reference_status": reference_row.get("status"),
        "ledger_closures": ledger_closures,
        "all_ledgers_closed": all(closure_checks),
        "outputs": {"mode_records": rows_path.name, "fresh_records": "fresh-records.jsonl",
            "reference": "reference.json", "arrays": arrays_path.name},
        "arrays_sha256": sha256(arrays_path)}
    (output / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    plan = read_plan()
    helper = load_module(QUAL_RUN, "tio2_vc_rotation_qualification_helpers")
    if args.preflight:
        validate_inputs(plan)
        outcome = _preflight(helper, plan)
        print(json.dumps(outcome, indent=2, allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required with --execute")
    validate_inputs(plan)
    summary = execute(plan, args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "search_paid_requests": summary["search_paid_requests"],
                      "reference_paid_requests": summary["reference_paid_requests"],
                      "fresh_paid_requests": summary["fresh_paid_requests"],
                      "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
