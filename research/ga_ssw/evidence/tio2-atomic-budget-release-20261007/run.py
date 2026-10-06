#!/usr/bin/env python3
"""One bounded atomic-climb policy arm for the TiO2 saved-state probe.

Preflight uses only a dummy calculator.  --execute runs exactly one plan slot.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN_PATH = HERE / "plan.json"
QUAL_RUN = ROOT / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run.py"
SLOT_ORDER = ((0, "force"), (0, "force_or_budget"),
              (1, "force"), (1, "force_or_budget"))
SNAPSHOT_MODULES = (
    "pamssw.standalone.atomic_climb", "pamssw.standalone.paper_reference",
    "pamssw.standalone.cell_relax", "pamssw.standalone.vc_geometry",
    "pamssw.standalone.generalized_numerics", "pamssw.standalone.surface",
    "pamssw.standalone.dimer", "pamssw.standalone.direction",
    "pamssw.standalone.gaussian", "pamssw.standalone.periodic_geometry",
    "pamssw.standalone.periodic_ga_reference", "pamssw.standalone.cluster_frame",
    "pamssw.standalone.ls_cycle", "pamssw.standalone.ls_prequench",
    "pamssw.standalone.softening", "pamssw.standalone.cluster_reconnection",
    "pamssw.standalone.native_mc", "pamssw.standalone.starter_selection",
    "pamssw.relax", "pamssw.result",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def append(path, obj):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(obj, sort_keys=True, allow_nan=False) + "\n")


def ledger_closure(path, budget):
    rows = ([json.loads(line) for line in Path(path).read_text().splitlines()]
            if Path(path).exists() else [])
    paid = [r for r in rows if r.get("event") in ("evaluation", "failure")]
    denied = [r for r in rows if r.get("event") == "denial"]
    raw_calls = sum(int(r.get("calculator_calls", 0)) for r in paid)
    return {"paid_ledger_rows": len(paid), "denial_rows": len(denied),
            "raw_calculator_calls": raw_calls,
            "closed": len(paid) == budget.requests and len(denied) == budget.denials and
                      raw_calls == budget.calculator_calls}


def jsonable(value):
    if dataclasses.is_dataclass(value):
        return {k: jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_helper():
    spec = importlib.util.spec_from_file_location("tio2_qualification_runner", QUAL_RUN)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT))
    spec.loader.exec_module(module)
    return module


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def validate_plan(plan):
    required = {"model", "model_sha256", "calculator", "cases", "atomic_config",
                "true_quench", "target", "identity_tolerances", "policies", "budget"}
    if required - set(plan):
        raise ValueError(f"plan missing fields: {sorted(required - set(plan))}")
    expected_budget = {"per_arm_requests": 3000, "per_arm_seconds": 120,
                       "total_seconds": 540, "fresh_per_arm_requests": 2,
                       "total_search_request_cap": 12000,
                       "total_fresh_request_cap": 8}
    if plan["budget"] != expected_budget:
        raise ValueError("budget no longer matches the bounded four-slot protocol")
    if plan["policies"] != ["force", "force_or_budget"]:
        raise ValueError("policy comparison differs from the approved pair")
    if len(plan["cases"]) != 2 or [x["id"] for x in plan["cases"]] != ["phase87_12", "phase87_48"]:
        raise ValueError("unexpected case order or count")
    if plan["calculator"] != {"head": "omat_pbe", "device": "cuda", "dtype": "float64",
                              "enable_cueq": False, "enable_oeq": False}:
        raise ValueError("calculator settings differ from approved OMAT-small model")
    model = Path(plan["model"])
    if not model.is_file() or sha256(model) != plan["model_sha256"]:
        raise RuntimeError("model missing or hash differs from the plan")
    target = plan["target"]
    if sha256(target["path"]) != target["sha256"]:
        raise RuntimeError("target endpoint hash differs from plan")

    from ase.io import read
    for case in plan["cases"]:
        source = Path(case["input"])
        if not source.is_file() or sha256(source) != case["input_sha256"]:
            raise RuntimeError(f"saved-state hash mismatch: {source}")
        atoms = read(source)
        if len(atoms) not in (12, 48) or not atoms.pbc.all() or atoms.constraints:
            raise ValueError(f"invalid saved state contract for {case['id']}")
        if case["id"].endswith("12") != (len(atoms) == 12):
            raise ValueError(f"saved-state atom count mismatch for {case['id']}")
        verify_source_record(case)
    return model


def verify_source_record(case):
    source = Path(case["source_run"])
    result = json.loads((source / "result.json").read_text())
    rows = [json.loads(line) for line in (source / "outer-records.jsonl").read_text().splitlines()]
    minima = [json.loads(line) for line in (source / "minima.jsonl").read_text().splitlines()]
    idx = case["source_outer_index"]
    if result.get("case") != case["id"] or result.get("arm") != "block":
        raise ValueError(f"unexpected source run identity: {source}")
    if len(rows) <= idx or rows[idx].get("status") != "atomic_rotation_failed":
        raise ValueError(f"saved state is not after the declared atomic rotation failure: {source}")
    atomic = rows[idx].get("atomic") or {}
    if atomic.get("status") != "rotation_failed":
        raise ValueError(f"source outer record has no atomic rotation failure: {source}")
    if case["id"] == "phase87_12":
        ref_rows = [r for r in minima if r.get("outer_index") is None]
    else:
        ref_rows = [r for r in minima if r.get("outer_index") == 0 and r.get("accepted") is True]
    if len(ref_rows) != 1 or not math.isclose(float(ref_rows[0]["energy_eV"]),
                                               float(case["reference_energy_eV"]),
                                               rel_tol=0., abs_tol=1e-10):
        raise ValueError(f"plan reference energy does not match the source current state: {source}")


def provenance(plan):
    sys.path.insert(0, str(ROOT))
    imports = {}
    sources = {}
    for name in SNAPSHOT_MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        imports[name] = str(path)
        sources[name] = sha256(path)
    versions = {}
    for package in ("ase", "mace-torch", "torch", "numpy", "scipy", "pymatgen"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "head": git("rev-parse", "HEAD"),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "imports": imports,
        "source_sha256": sources,
        "runner_sha256": sha256(__file__),
        "qualification_helper": str(QUAL_RUN),
        "qualification_helper_sha256": sha256(QUAL_RUN),
        "plan_sha256": sha256(PLAN_PATH),
        "model": plan["model"],
        "model_sha256": sha256(plan["model"]),
        "inputs": {c["id"]: {"path": c["input"], "sha256": sha256(c["input"]),
                               "source_run": c["source_run"]} for c in plan["cases"]},
        "target": {"path": plan["target"]["path"], "sha256": sha256(plan["target"]["path"])},
    }


def build_config(plan, policy):
    from pamssw.standalone.paper_reference import SSWConfig
    values = dict(plan["atomic_config"])
    values["rotation_exit_policy"] = policy
    config = SSWConfig(**values)
    if config.rotation_exit_policy != policy:
        raise RuntimeError("atomic config failed to preserve the selected policy")
    return config


class AtomicEnergyForceSurface:
    """Fixed-cell API view over one shared, stress-counted surface."""
    def __init__(self, counted_surface):
        self.counted_surface = counted_surface

    @property
    def requests(self):
        return self.counted_surface.requests

    def evaluate(self, atoms):
        energy, forces, _stress = self.counted_surface.evaluate(atoms)
        return energy, forces


def preflight(plan):
    validate_plan(plan)
    sys.path.insert(0, str(ROOT))
    helper = load_helper()
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from pamssw.standalone.atomic_climb import atomic_climb
    from pamssw.standalone.cell_relax import cell_quench
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.vc_geometry import ASEStressSurface

    class Dummy(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0., "forces": np.zeros_like(atoms.positions),
                            "stress": np.zeros(6)}

    configs = {policy: build_config(plan, policy) for policy in plan["policies"]}
    if any(not isinstance(config, SSWConfig) for config in configs.values()):
        raise TypeError("wrong SSWConfig implementation")
    with __import__("tempfile").TemporaryDirectory(prefix="tio2-budget-preflight-") as tmp:
        budget = helper.CaseBudget("dummy", Path(tmp) / "requests.jsonl", 2, 20,
                                   time.monotonic() + 20)
        calc = Dummy()
        helper.instrument_calculate(calc, budget)
        surface = helper.CountedStressSurface(calc, budget, "dummy")
        dummy = Atoms("TiO2", positions=np.zeros((3, 3)), cell=np.eye(3) * 8, pbc=True)
        surface.evaluate(dummy)
        surface.evaluate(dummy.copy())
        try:
            surface.evaluate(dummy.copy())
        except RuntimeError:
            pass
        else:
            raise RuntimeError("dummy budget cap did not deny an over-cap request")
        ledger_rows = [json.loads(line) for line in (Path(tmp) / "requests.jsonl").read_text().splitlines()]
        if (budget.requests, budget.calculator_calls, budget.denials) != (2, 1, 1):
            raise RuntimeError("dummy budget/cache accounting did not close")
        dummy_state = Atoms("H3", positions=[[1., 1., 1.], [2.2, 1., 1.],
                           [1., 2.3, 1.]], cell=np.eye(3) * 8., pbc=True)
        atomic_budget = helper.CaseBudget("dummy-atomic", Path(tmp) / "atomic.jsonl",
            100, 20, time.monotonic() + 20)
        atomic_calc = Dummy()
        helper.instrument_calculate(atomic_calc, atomic_budget)
        atomic_surface = helper.CountedStressSurface(atomic_calc, atomic_budget, "dummy_atomic")
        smoke_config = dataclasses.replace(configs["force"], max_gaussians=1,
                                           rotation_hvp=4, relax_steps=3)
        smoke_climb = atomic_climb(dummy_state, AtomicEnergyForceSurface(atomic_surface),
            reference_energy=-1., config=smoke_config, rng=np.random.default_rng(1))
        if (smoke_climb.status != "gaussian_limit" or
                smoke_climb.requests != atomic_budget.requests or
                not smoke_climb.climb or "direction" not in smoke_climb.climb[0]):
            raise RuntimeError("dummy atomic-climb API/state/cost check failed")
        quench_budget = helper.CaseBudget("dummy-cell-quench", Path(tmp) / "cell.jsonl",
            100, 20, time.monotonic() + 20)
        quench_calc = Dummy()
        helper.instrument_calculate(quench_calc, quench_budget)
        quench_surface = helper.CountedStressSurface(quench_calc, quench_budget, "dummy_cell_quench")
        smoke_quench = cell_quench(dummy_state, quench_surface,
            strain_length=float(plan["true_quench"]["strain_length"]),
            pressure=float(plan["true_quench"]["pressure"]), fmax=float(plan["true_quench"]["fmax"]),
            stress_tol=float(plan["true_quench"]["stress_tol"]),
            max_step=float(plan["true_quench"]["max_step"]), maxiter=2,
            lbfgs_memory=int(plan["true_quench"]["lbfgs_memory"]))
        if (not smoke_quench.converged or smoke_quench.requests != quench_budget.requests):
            raise RuntimeError("dummy cell-quench API/cost check failed")
    return {
        "status": "preflight_passed", "real_ef_requests": 0,
        "runner_api": {"atomic_climb": str(__import__("inspect").signature(atomic_climb)),
                       "cell_quench": str(__import__("inspect").signature(cell_quench)),
                       "counted_surface": str(__import__("inspect").signature(helper.CountedStressSurface)),
                       "ase_surface": str(__import__("inspect").signature(ASEStressSurface))},
        "configs": {k: dataclasses.asdict(v) for k, v in configs.items()},
        "dummy_budget": {"requests": budget.requests, "calculator_calls": budget.calculator_calls,
                         "denials": budget.denials, "ledger_rows": len(ledger_rows)},
        "dummy_atomic_climb": {"status": smoke_climb.status,
            "requests": smoke_climb.requests, "calculator_calls": atomic_budget.calculator_calls,
            "events": len(smoke_climb.climb)},
        "dummy_cell_quench": {"converged": smoke_quench.converged,
            "requests": smoke_quench.requests, "calculator_calls": quench_budget.calculator_calls},
    }


def scalar_event(event):
    telemetry = event.get("optimizer_telemetry")
    telemetry_row = None
    if telemetry is not None:
        telemetry_row = {name: jsonable(getattr(telemetry, name, None)) for name in
                         ("status", "termination_reason", "steps", "evaluations", "gradient_norm",
                          "step_norm", "convergence_norm", "message")}
    keep = ("index", "status", "rotation_solver", "rotation_residual", "residual",
            "rotation_stop_reason", "rotation_converged", "rotation_budget_released",
            "rotation_force_requests", "force_requests", "quench_requests", "weight", "width",
            "actual_rotation_bias", "biased_energy", "force_certificate", "max_force",
            "termination_reason", "stage_stop_reason", "background_forward_force", "true_energy",
            "actual_anchor", "center", "direction")
    row = {key: jsonable(event[key]) for key in keep if key in event}
    row["optimizer_telemetry"] = telemetry_row
    return row


def evaluate_target(atoms, energy, plan):
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    target = __import__("ase.io", fromlist=["read"]).read(plan["target"]["path"])
    out = {"energy_eV": float(energy), "energy_per_atom_eV": float(energy) / len(atoms),
           "energy_window_passed": bool(float(energy) / len(atoms) <=
               float(plan["target"]["energy_eV"]) / int(plan["target"]["natoms"]) +
               float(plan["target"]["energy_tolerance_eV_per_atom"])),
           "identity_matches": {}}
    for name, tol in plan["identity_tolerances"].items():
        try:
            matcher = pymatgen_identity(**tol)
            out["identity_matches"][name] = {"matches": bool(matcher(atoms, target)),
                                             "status": "completed"}
        except Exception as exc:
            out["identity_matches"][name] = {"matches": None, "status": "error",
                                             "error": repr(exc)}
    out["both_identity_matchers_passed"] = all(
        out["identity_matches"].get(k, {}).get("matches") is True for k in ("tight", "broad"))
    out["target_phase_energy_gate"] = bool(out["both_identity_matchers_passed"] and
                                            out["energy_window_passed"])
    return out


def strip_and_write(path, atoms, **info):
    from ase.io import write
    out = atoms.copy()
    out.calc = None
    out.info.update({k: jsonable(v) for k, v in info.items()})
    write(path, out, format="extxyz")


def module_snapshot(out):
    root = out / "source-snapshot"
    root.mkdir()
    shutil.copy2(__file__, root / "run.py")
    shutil.copy2(QUAL_RUN, root / "qualification-run.py")
    for name in SNAPSHOT_MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        shutil.copy2(path, root / path.name)


def run_slot(plan, prov, slot, out, global_deadline):
    if not 0 <= slot < len(SLOT_ORDER):
        raise ValueError(f"slot must be in [0,{len(SLOT_ORDER)-1}]")
    case_index, policy = SLOT_ORDER[slot]
    case = plan["cases"][case_index]
    arm = f"{case['id']}-{policy}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(PLAN_PATH, out / "effective-plan.json")
    shutil.copy2(case["input"], out / "input.extxyz")
    source = Path(case["source_run"])
    shutil.copy2(source / "result.json", out / "source-result.json")
    shutil.copy2(source / "provenance.json", out / "source-provenance.json")
    source_rows = [json.loads(line) for line in (source / "outer-records.jsonl").read_text().splitlines()]
    source_minima = [json.loads(line) for line in (source / "minima.jsonl").read_text().splitlines()]
    ref_row = next(r for r in source_minima if
                   ((case["id"] == "phase87_12" and r.get("outer_index") is None) or
                    (case["id"] == "phase87_48" and r.get("outer_index") == 0 and
                     r.get("accepted") is True)))
    dump(out / "source-outer-record.json", source_rows[case["source_outer_index"]])
    dump(out / "source-reference-minimum.json", ref_row)
    shutil.copy2(plan["target"]["path"], out / "anatase-target.extxyz")
    dump(out / "provenance.json", {**prov, "slot": slot, "arm": arm,
                                    "case_seed": case["seed"], "policy": policy})
    module_snapshot(out)
    config = build_config(plan, policy)
    quench_spec = dict(plan["true_quench"])
    effective = {"atomic_config": dataclasses.asdict(config),
                 "true_quench": quench_spec,
                 "policy_factor": "rotation_exit_policy",
                 "slot": slot, "case": case["id"], "seed": case["seed"]}
    dump(out / "effective-config.json", effective)

    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone.atomic_climb import atomic_climb
    from pamssw.standalone.cell_relax import cell_quench
    helper = load_helper()
    atoms = read(out / "input.extxyz")
    atoms.calc = None
    calc_spec = plan["calculator"]
    calc_kw = {"model_paths": plan["model"], **calc_spec}
    calc_kw["default_dtype"] = calc_kw.pop("dtype")
    started = time.monotonic()
    search_budget = helper.CaseBudget(arm, out / "requests.jsonl",
        int(plan["budget"]["per_arm_requests"]), int(plan["budget"]["per_arm_seconds"]),
        global_deadline)
    row = {"status": "running", "slot": slot, "arm": arm, "case": case["id"],
           "natoms": len(atoms), "seed": int(case["seed"]), "rotation_exit_policy": policy,
           "reference_energy_eV": float(case["reference_energy_eV"]),
           "search_ledger": "requests.jsonl", "fresh_ledger": "fresh-requests.jsonl",
           "atomic_climb": {}, "true_quench": None, "cold_checks": [],
           "target_candidate": False}
    calculator = None
    fresh_budget = None
    fresh_calc = None
    fresh_surface = None
    true_result = None
    try:
        if time.monotonic() >= global_deadline:
            raise TimeoutError("shared absolute experiment deadline reached before arm setup")
        calculator = MACECalculator(**calc_kw)
        param_types = sorted({str(p.dtype) for model in calculator.models for p in model.parameters()})
        if param_types != ["torch.float64"]:
            raise RuntimeError(f"loaded model parameter dtypes differ: {param_types}")
        row["model_parameter_dtypes"] = param_types
        helper.instrument_calculate(calculator, search_budget)
        calculator.reset()
        surface = helper.CountedStressSurface(calculator, search_budget, "atomic_and_true_quench")
        atomic_started_requests = search_budget.requests
        climb = atomic_climb(atoms, AtomicEnergyForceSurface(surface),
                            reference_energy=float(case["reference_energy_eV"]),
                            config=config, rng=np.random.default_rng(int(case["seed"])))
        atomic_end_requests = search_budget.requests
        strip_and_write(out / "atomic-climb-endpoint.extxyz", climb.atoms,
                        diagnostic_status=climb.status, role="atomic_climb_endpoint")
        event_rows = []
        for event in climb.climb:
            event_row = scalar_event(event)
            event_row["arm"] = arm
            if "center" in event:
                center_path = out / f"gaussian-{int(event.get('index', 0)):03d}-center.extxyz"
                center_atoms = atoms.copy()
                center_atoms.positions = np.asarray(event["center"], dtype=float)
                strip_and_write(center_path, center_atoms, role="gaussian_center",
                                gaussian_index=int(event.get("index", 0)))
                event_row["center_structure"] = center_path.name
            append(out / "gaussians.jsonl", event_row)
            event_rows.append(event_row)
        rotation_cost = sum(int(e.get("rotation_force_requests", e.get("force_requests", 0)) or 0)
                            for e in climb.climb)
        biased_quench_cost = sum(int(e.get("quench_requests", 0) or 0) for e in climb.climb)
        row["atomic_climb"] = {
            "status": climb.status, "error": climb.error,
            "reported_requests": int(climb.requests),
            "ledger_request_delta": atomic_end_requests - atomic_started_requests,
            "ledger_request_range_1based_inclusive": [atomic_started_requests + 1,
                                                        atomic_end_requests],
            "events": event_rows,
            "rotation_force_requests": rotation_cost,
            "biased_quench_requests": biased_quench_cost,
            "other_climber_requests": int(climb.requests) - rotation_cost - biased_quench_cost,
            "rotation_converged": [e.get("rotation_converged") for e in event_rows],
            "rotation_budget_released": [e.get("rotation_budget_released") for e in event_rows],
            "rotation_stop_reason": [e.get("rotation_stop_reason") for e in event_rows],
        }
        if climb.requests != atomic_end_requests - atomic_started_requests:
            raise RuntimeError("atomic_climb result request count does not close against raw ledger")

        if climb.status in ("lower_true_energy", "gaussian_limit"):
            before = search_budget.requests
            try:
                true_result = cell_quench(climb.atoms, surface,
                    strain_length=float(quench_spec["strain_length"]),
                    pressure=float(quench_spec["pressure"]), fmax=float(quench_spec["fmax"]),
                    stress_tol=float(quench_spec["stress_tol"]),
                    max_step=float(quench_spec["max_step"]), maxiter=int(quench_spec["maxiter"]),
                    lbfgs_memory=int(quench_spec["lbfgs_memory"]))
                ev = true_result.evaluation
                candidate = bool(true_result.converged and ev is not None and
                    true_result.certificate.get("certified", False) and
                    float(true_result.certificate.get("fmax", math.inf)) <= float(quench_spec["fmax"]) and
                    float(true_result.certificate.get("stress_max", math.inf)) <= float(quench_spec["stress_tol"]))
                quench_row = {"status": "certified_minimum" if candidate else "uncertified_endpoint",
                    "optimizer_converged": bool(true_result.optimizer.converged),
                    "optimizer_status": str(true_result.optimizer.status),
                    "certificate": jsonable(true_result.certificate),
                    "reported_requests": int(true_result.requests),
                    "ledger_request_delta": search_budget.requests - before,
                    "ledger_request_range_1based_inclusive": [before + 1, search_budget.requests],
                    "candidate_minimum": candidate}
                if ev is not None:
                    metrics = {"energy_eV": float(ev.energy),
                               "fmax_eV_A": float(np.linalg.norm(ev.forces, axis=1).max()),
                               "stress_max_eV_A3": float(np.abs(ev.stress).max())}
                    quench_row.update(metrics)
                    quench_row.update(evaluate_target(ev.atoms, ev.energy, plan))
                    strip_and_write(out / ("certified-true-quench-minimum.extxyz" if candidate
                                           else "uncertified-true-quench-endpoint.extxyz"),
                        ev.atoms, role=("certified_true_quench_minimum" if candidate
                                        else "uncertified_true_quench_endpoint"), **metrics)
                else:
                    quench_row["evaluation_unavailable"] = True
                row["true_quench"] = quench_row
                row["target_candidate"] = bool(candidate and quench_row.get("target_phase_energy_gate"))
            except Exception as exc:
                row["true_quench"] = {"status": "exception", "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "ledger_request_delta": search_budget.requests - before}
            if true_result is not None and true_result.requests != search_budget.requests - before:
                raise RuntimeError("true-quench result request count does not close against raw ledger")
        else:
            row["true_quench"] = {"status": "not_run_for_atomic_status",
                                   "atomic_status": climb.status, "reported_requests": 0}

        # Independent fresh checks are limited to input measurement and, only if
        # certified, one true-quench minimum.  They do not share search accounting.
        candidates = [("input_measurement", atoms, None, False)]
        if row.get("true_quench", {}).get("candidate_minimum"):
            candidates.append(("certified_true_quench_candidate",
                true_result.evaluation.atoms, float(true_result.evaluation.energy), True))
        del surface, calculator
        calculator = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if time.monotonic() >= global_deadline:
            raise TimeoutError("shared absolute experiment deadline reached before cold checks")
        fresh_calc = MACECalculator(**calc_kw)
        fresh_budget = helper.CaseBudget(arm, out / "fresh-requests.jsonl",
            int(plan["budget"]["fresh_per_arm_requests"]),
            max(1., global_deadline-time.monotonic()), global_deadline)
        helper.instrument_calculate(fresh_calc, fresh_budget)
        fresh_surface = helper.CountedStressSurface(fresh_calc, fresh_budget, "cold_check")
        for role, structure, expected_energy, is_candidate in candidates:
            try:
                fresh_calc.reset()
                energy, forces, stress = fresh_surface.evaluate(structure)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                stress_max = float(np.abs(stress).max())
                target_info = evaluate_target(structure, energy, plan)
                physical = bool(np.isfinite([energy, fmax, stress_max]).all() and
                                fmax <= float(plan["target"]["fmax_eV_A"]) and
                                stress_max <= float(plan["target"]["stress_tol_eV_A3"]))
                cold_matches = bool(target_info["both_identity_matchers_passed"])
                cold_target = bool(is_candidate and physical and cold_matches and
                                   target_info["energy_window_passed"])
                check = {"role": role, "status": "completed", "energy_eV": float(energy),
                         "energy_per_atom_eV": float(energy)/len(structure),
                         "fmax_eV_A": fmax, "stress_max_eV_A3": stress_max,
                         "physical_force_stress_passed": physical,
                         "energy_difference_from_search_eV": (None if expected_energy is None
                                                               else float(energy-expected_energy)),
                         "energy_reproduced_within_1e-6_eV": (None if expected_energy is None
                             else bool(abs(float(energy-expected_energy)) <= 1e-6)),
                         "target": target_info,
                         "cold_confirmed_target_candidate": cold_target}
                row["cold_checks"].append(check)
            except Exception as exc:
                row["cold_checks"].append({"role": role, "status": "error", "error": repr(exc),
                    "traceback": traceback.format_exc(), "cold_confirmed_target_candidate": False})
        row["cold_confirmed_target_candidate"] = any(
            c.get("cold_confirmed_target_candidate", False) for c in row["cold_checks"])
        row["status"] = "complete"
    except Exception as exc:
        row.update(status="exception", error=repr(exc), traceback=traceback.format_exc())
    finally:
        if calculator is not None:
            del calculator
        if fresh_surface is not None:
            del fresh_surface
        if fresh_calc is not None:
            del fresh_calc
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        row["search_cost"] = {"requests": int(search_budget.requests),
            "calculator_calls": int(search_budget.calculator_calls),
            "failures": int(search_budget.failures), "denials": int(search_budget.denials),
            "boundary": search_budget.boundary,
            "elapsed_seconds": time.monotonic()-search_budget.started}
        row["fresh_cost"] = {"requests": 0 if fresh_budget is None else int(fresh_budget.requests),
            "calculator_calls": 0 if fresh_budget is None else int(fresh_budget.calculator_calls),
            "failures": 0 if fresh_budget is None else int(fresh_budget.failures),
            "denials": 0 if fresh_budget is None else int(fresh_budget.denials),
            "boundary": None if fresh_budget is None else fresh_budget.boundary,
            "elapsed_seconds": 0 if fresh_budget is None else time.monotonic()-fresh_budget.started}
        row["search_ledger_closure"] = ledger_closure(out / "requests.jsonl", search_budget)
        row["fresh_ledger_closure"] = (ledger_closure(out / "fresh-requests.jsonl", fresh_budget)
            if fresh_budget is not None else {"paid_ledger_rows": 0, "denial_rows": 0,
                                              "raw_calculator_calls": 0, "closed": True})
        row["true_quench_cost_requests"] = int((row.get("true_quench") or {}).get("ledger_request_delta", 0))
        row["atomic_plus_true_quench_closure"] = {
            "reported_atomic": row.get("atomic_climb", {}).get("reported_requests", 0),
            "reported_true_quench": (row.get("true_quench") or {}).get("reported_requests", 0),
            "ledger_search_total": search_budget.requests,
            "closed": bool(row.get("atomic_climb", {}).get("ledger_request_delta", 0) +
                (row.get("true_quench") or {}).get("ledger_request_delta", 0) == search_budget.requests)}
        row["fresh_closure"] = {"ledger_requests": row["fresh_cost"]["requests"],
            "cold_check_attempts": len(row["cold_checks"]),
            "closed": row["fresh_cost"]["requests"] == len(row["cold_checks"])}
        row["elapsed_seconds"] = time.monotonic() - started
        dump(out / "result.json", row)
    return 0 if row["status"] == "complete" else 1


def execute_all(plan, prov, out_root, run_id, global_deadline):
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in run_id):
        raise ValueError("run_id must be a nonempty filename-safe identifier")
    out_root.mkdir(parents=True, exist_ok=True)
    expected_paths = [out_root / f"run-{run_id}-{slot}" for slot in range(len(SLOT_ORDER))]
    summary_path = out_root / f"run-{run_id}-summary.json"
    collisions = [p for p in [*expected_paths, summary_path] if p.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite existing run outputs: {collisions}")
    global_started = global_deadline - float(plan["budget"]["total_seconds"])
    rows = []
    for slot in range(len(SLOT_ORDER)):
        slot_out = expected_paths[slot]
        row = {"slot": slot, "output": str(slot_out), "return_code": None,
               "runner_status": "not_started"}
        try:
            code = run_slot(plan, prov, slot, slot_out, global_deadline)
            row["return_code"] = code
            result_path = slot_out / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                row.update(runner_status=result.get("status"), arm=result.get("arm"),
                    case=result.get("case"), policy=result.get("rotation_exit_policy"),
                    atomic_status=(result.get("atomic_climb") or {}).get("status"),
                    search_requests=(result.get("search_cost") or {}).get("requests"),
                    fresh_requests=(result.get("fresh_cost") or {}).get("requests"),
                    search_ledger_closed=(result.get("search_ledger_closure") or {}).get("closed"),
                    fresh_ledger_closed=(result.get("fresh_ledger_closure") or {}).get("closed"))
            else:
                row["runner_status"] = "result_missing"
        except Exception as exc:
            row.update(return_code=1, runner_status="runner_exception", error=repr(exc),
                        traceback=traceback.format_exc())
            if slot_out.is_dir() and not (slot_out / "runner-failure.json").exists():
                dump(slot_out / "runner-failure.json", row)
        rows.append(row)
    complete = all(r["runner_status"] == "complete" for r in rows)
    summary = {"status": "all_slots_recorded" if len(rows) == len(SLOT_ORDER)
               else "incomplete", "run_id": run_id, "started_monotonic": global_started,
               "deadline_monotonic": global_deadline,
               "elapsed_seconds": time.monotonic() - global_started,
               "configured_total_seconds": int(plan["budget"]["total_seconds"]),
               "all_runner_slots_complete": complete, "slots": rows}
    dump(summary_path, summary)
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true", help="run one diagnostic slot")
    mode.add_argument("--execute-all", action="store_true",
                      help="run all four slots sequentially under one shared deadline")
    parser.add_argument("--slot", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    global_deadline = time.monotonic() + 540.
    plan = json.loads(PLAN_PATH.read_text())
    if args.preflight:
        print(json.dumps(preflight(plan), indent=2, allow_nan=False))
        return 0
    validate_plan(plan)
    prov = provenance(plan)
    if args.execute_all:
        if args.out_root is None or args.run_id is None:
            parser.error("--execute-all requires --out-root PATH and --run-id IDENTIFIER")
        return execute_all(plan, prov, args.out_root.resolve(), args.run_id, global_deadline)
    if args.slot is None or args.out is None:
        parser.error("--execute requires --slot 0..3 and --out PATH")
    return run_slot(plan, prov, args.slot, args.out.resolve(), global_deadline)


if __name__ == "__main__":
    raise SystemExit(main())
