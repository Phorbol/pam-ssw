#!/usr/bin/env python3
"""Paired, bounded phase-87 block-VC contribution experiment.

The search kernel is the existing run_block_ssw implementation. This file only
freezes the two configs, accounts for calls, and records physical endpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.util
import json
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
PANEL_PATH = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-vc-target-panel-20261007/run.py")
QUAL_PATH = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/tio2-phase-qualification-20261007/run.py")
PLAN_PATH = PANEL_PATH.with_name("plan.json")
CASE_ID = "phase87_48"
SEEDS = (26101241, 26101242)
OUTER_STEPS = 20
SEARCH_CAP = 4000
FRESH_CAP = 3
INTERNAL_WALL_SECONDS = 28 * 60
ARM_WALL_SECONDS = 10 * 60
FRESH_WALL_SECONDS = 60
SEARCH_CAP_TOTAL = 4 * SEARCH_CAP
FRESH_CAP_TOTAL = 4 * FRESH_CAP + 1
PAIR_TRACE_ATOL = 1e-8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def panel_module():
    return load_module(PANEL_PATH, "tio2_vc_target_panel_helpers_atomic_contribution")


def load_plan():
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(PLAN_PATH)
    return json.loads(PLAN_PATH.read_text())


def treatment_configurations(plan=None):
    """Build constructor-valid arms; only BlockSSWConfig.atomic_period varies."""
    sys.path.insert(0, str(ROOT))
    panel = panel_module()
    atomic, block, _joint = panel.target_config(load_plan() if plan is None else plan)
    atomic = replace(atomic, rotation_exit_policy="force_or_budget")
    block = replace(block, atomic=atomic)
    return {
        "cell_only": {"atomic": atomic, "block": replace(block, atomic_period=21)},
        "cell_plus_atomic": {"atomic": atomic, "block": replace(block, atomic_period=2)},
    }


def classify_outer_records(records, ledger):
    """Preserve actual paid failures ahead of a coincident cap boundary."""
    paid_failures = [row for row in ledger if row.get("event") == "failure"]
    denials = [row for row in ledger if row.get("event") == "denial"]
    outcomes = []
    cumulative = 0
    failed_at = min((int(row.get("request", 0)) for row in paid_failures), default=None)
    boundary_at = min((int(row.get("request", 0)) for row in denials), default=None)
    for record in records:
        start = int(record.get("request_start", cumulative))
        cost = int(record.get("requests", 0) or 0)
        end = start + cost
        index = record.get("index")
        if failed_at is not None and start < failed_at <= end:
            outcome = "paid_failure"
        elif failed_at is not None and start >= failed_at:
            outcome = "not_run_after_paid_failure"
        elif record.get("status") == "valid_landing":
            outcome = "valid_landing"
        elif boundary_at is not None and start < boundary_at <= end:
            outcome = "budget_censored"
        elif boundary_at is not None and cost == 0 and start >= boundary_at:
            outcome = "not_started_no_budget"
        else:
            outcome = str(record.get("status", "unknown"))
        outcomes.append({"index": index, "outcome": outcome,
                         "request_interval": [start, end]})
        cumulative = end
    return outcomes


def atomic_path_diagnostic(record):
    if not record.get("atomic_scheduled"):
        return "not_scheduled"
    cycles = record.get("cell_cycles") or []
    if any(cycle.get("status") not in ("completed",) for cycle in cycles):
        return "atomic_not_reached_after_cell_failure"
    atomic = record.get("atomic")
    if atomic is None:
        return "atomic_not_reached"
    climbs = getattr(atomic, "climb", []) or []
    released = any(bool(item.get("rotation_budget_released")) for item in climbs)
    completed_gaussians = sum("true_energy" in item for item in climbs)
    atomic_status = getattr(atomic, "status", None)
    landing = record.get("landing")
    if landing is not None and getattr(landing, "converged", False):
        return "atomic_landing_valid"
    if atomic_status == "biased_quench_failed":
        return f"biased_quench_failed_after_{completed_gaussians}_completed_gaussians"
    if atomic_status == "rotation_failed":
        return "atomic_rotation_failed"
    if completed_gaussians and landing is not None:
        return "completed_gaussians_then_landing_failed"
    if released:
        return f"rotation_budget_released_{completed_gaussians}_completed_gaussians"
    return "atomic_attempted_without_budget_release"


def _config_row(config):
    return {arm: {key: asdict(value) for key, value in values.items()}
            for arm, values in config.items()}


def validate_inputs(plan, case_id=CASE_ID):
    sys.path.insert(0, str(ROOT))
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone.block_ssw import BlockSSWConfig
    from pamssw.standalone.paper_reference import SSWConfig
    case = next((item for item in plan["cases"] if item.get("id") == case_id), None)
    if case is None or case.get("natoms") != 48:
        raise ValueError(f"frozen {case_id} case absent or wrong atom count")
    model = Path(plan["model"])
    if not model.is_file() or sha256(model) != plan["model_sha256"]:
        raise RuntimeError("frozen OMAT-small model missing or hash mismatch")
    for field, expected in (("path", True), ("sha256", True)):
        if not case.get(field):
            raise ValueError(f"phase87_48 missing {field}")
    if not Path(case["path"]).is_file() or sha256(case["path"]) != case["sha256"]:
        raise RuntimeError(f"{case_id} source missing or hash mismatch")
    atoms = read(case["path"])
    if (len(atoms) != 48 or atoms.get_chemical_formula() != "O32Ti16" or
            not atoms.pbc.all() or atoms.constraints):
        raise ValueError(f"frozen {case_id} geometry contract mismatch")
    target = plan["target"]
    if not Path(target["path"]).is_file() or sha256(target["path"]) != target["sha256"]:
        raise RuntimeError("frozen anatase target missing or hash mismatch")
    if (target["natoms"] != 12 or target["fmax_eV_A"] != .05 or
            target["stress_tol_eV_A3"] != .001 or target["energy_tolerance_eV_per_atom"] != .001):
        raise ValueError("approved target qualification thresholds changed")
    pair = treatment_configurations(plan)
    for arm in pair.values():
        if not isinstance(arm["atomic"], SSWConfig) or not isinstance(arm["block"], BlockSSWConfig):
            raise TypeError("existing SSW/block constructors did not produce expected configs")
    c_only, c_atomic = pair["cell_only"], pair["cell_plus_atomic"]
    if {k for k in asdict(c_only["block"]) if asdict(c_only["block"])[k] != asdict(c_atomic["block"])[k]} != {"atomic_period"}:
        raise ValueError("treatments differ outside atomic_period")
    if asdict(c_only["atomic"]) != asdict(c_atomic["atomic"]):
        raise ValueError("atomic SSW configs differ between treatments")
    if c_only["block"].atomic_period != 21 or c_atomic["block"].atomic_period != 2:
        raise ValueError("unexpected treatment schedule")
    return {"case": case, "atoms": atoms, "pair": pair,
            "mace_constructor": str(__import__("inspect").signature(MACECalculator))}


def preflight(plan=None, case_id=CASE_ID):
    sys.path.insert(0, str(ROOT))
    plan = load_plan() if plan is None else plan
    validated = validate_inputs(plan, case_id)
    from ase.io import read
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    target = read(plan["target"]["path"])
    matches = {}
    for label, settings in plan["identity_tolerances"].items():
        matcher = pymatgen_identity(**settings)
        matches[label] = bool(matcher(target, target.copy()))
    if not all(matches.values()):
        raise RuntimeError(f"target identity self-check failed: {matches}")
    imports = {name: str(Path(importlib.import_module(name).__file__).resolve())
               for name in ("pamssw.standalone.block_ssw", "pamssw.standalone.paper_reference",
                            "pamssw.standalone.cell_relax", "pamssw.standalone.vc_geometry")}
    if any(not Path(path).is_relative_to(ROOT) for path in imports.values()):
        raise RuntimeError(f"core import escaped selected worktree: {imports}")
    return {"status": "preflight_passed", "real_pes_requests": 0,
            "case": {"id": case_id, "path": validated["case"]["path"],
                     "sha256": sha256(validated["case"]["path"]), "natoms": len(validated["atoms"])},
            "model_sha256": sha256(plan["model"]), "treatment_configs": _config_row(validated["pair"]),
            "target_identity_self_match": matches, "mace_constructor": validated["mace_constructor"],
            "core_imports": imports}


class StopAfterFailureSurface:
    """Latch true paid calculator failure or budget denial across walker catches."""
    def __init__(self, surface):
        self.surface = surface
        self.paid_error = None
        self.budget_boundary = None

    @property
    def requests(self):
        return self.surface.requests

    def evaluate(self, atoms):
        budget = self.surface.budget
        failures0, denials0 = budget.failures, budget.denials
        if self.paid_error or self.budget_boundary:
            raise RuntimeError("campaign stopped after latched boundary/failure")
        try:
            return self.surface.evaluate(atoms)
        except Exception as error:
            if budget.failures > failures0:
                self.paid_error = {"error": repr(error), "request": budget.requests}
            elif budget.denials > denials0:
                self.budget_boundary = budget.boundary
            raise


def _geometry_metrics(atoms, energy, forces, stress):
    import numpy as np
    return {"energy_eV": float(energy), "energy_per_atom_eV": float(energy / len(atoms)),
            "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
            "stress_max_eV_A3": float(np.abs(stress).max()), "natoms": len(atoms),
            "volume_A3": float(atoms.get_volume())}


def _target_identity(atoms, target, plan):
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    results = {}
    for label, settings in plan["identity_tolerances"].items():
        try:
            results[label] = bool(pymatgen_identity(**settings)(atoms, target))
        except Exception as error:
            results[label] = {"error": repr(error)}
    return results


def _target_gates(atoms, energy, forces, stress, plan, target, reference_energy=None):
    metric = _geometry_metrics(atoms, energy, forces, stress)
    identity = _target_identity(atoms, target, plan)
    identity_pass = all(value is True for value in identity.values())
    reference = float((plan["target"]["energy_eV"] if reference_energy is None
                       else reference_energy) / plan["target"]["natoms"])
    energy_pass = metric["energy_per_atom_eV"] <= reference + plan["target"]["energy_tolerance_eV_per_atom"]
    physical_pass = (metric["fmax_eV_A"] <= plan["target"]["fmax_eV_A"] and
                     metric["stress_max_eV_A3"] <= plan["target"]["stress_tol_eV_A3"])
    return {**metric, "identity_matches": identity, "identity_qualified": identity_pass,
            "energy_window_passed": energy_pass, "physical_gate_passed": physical_pass,
            "target_candidate": bool(identity_pass and energy_pass and physical_pass)}


def _calc_kwargs(plan):
    spec = plan["calculator"]
    return {"model_paths": plan["model"], "head": spec["head"], "device": spec["device"],
            "default_dtype": spec["dtype"], "enable_cueq": spec["enable_cueq"],
            "enable_oeq": spec["enable_oeq"]}


def _core_provenance():
    names = ("pamssw.standalone.block_ssw", "pamssw.standalone.paper_reference",
             "pamssw.standalone.atomic_climb", "pamssw.standalone.cell_relax",
             "pamssw.standalone.vc_geometry")
    modules = {name: Path(importlib.import_module(name).__file__).resolve() for name in names}
    if any(not path.is_relative_to(ROOT) for path in modules.values()):
        raise RuntimeError(f"PAM-SSW source import does not come from selected worktree: {modules}")
    return {str(name): {"path": str(path), "sha256": sha256(path)} for name, path in modules.items()}


def _new_budget(helper, label, ledger, cap, wall, deadline):
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.touch(exist_ok=True)
    return helper.CaseBudget(label, ledger, cap, wall, deadline)


def _fresh_check(calc, helper, atoms, budget, label, target, plan, output, reference_energy):
    import numpy as np
    from pamssw.standalone.vc_geometry import ASEStressSurface
    calc.reset()
    surface = helper.CountedStressSurface(calc, budget, label)
    energy, forces, stress = surface.evaluate(atoms.copy())
    row = _target_gates(atoms, energy, forces, stress, plan, target, reference_energy)
    row.update(role=label, calculator_calls=budget.calculator_calls,
               paid_requests=budget.requests)
    return row


def _outer_scalar(record, start_cost):
    panel = panel_module()
    atomic = record.get("atomic")
    climbs = [] if atomic is None else list(getattr(atomic, "climb", []) or [])
    completed = sum("true_energy" in event for event in climbs)
    stage = panel._record_stage_costs(record)
    cell_failure = any(cycle.get("status") not in ("completed",) for cycle in record.get("cell_cycles", []))
    cycle_trace = []
    for cycle in record.get("cell_cycles", []) or []:
        mode = cycle.get("mode")
        direction = getattr(mode, "direction", None)
        cycle_trace.append({"index": cycle.get("index"), "status": cycle.get("status"),
            "requests": int(cycle.get("requests", 0) or 0),
            "partial_status": cycle.get("partial_status"), "partial_steps": cycle.get("partial_steps"),
            "distance": cycle.get("distance"),
            "cell_before": None if cycle.get("cell_before") is None else cycle["cell_before"].tolist(),
            "cell_after": None if cycle.get("cell_after") is None else cycle["cell_after"].tolist(),
            "direction": None if direction is None else direction.tolist(),
            "mode_converged": getattr(mode, "converged", None),
            "mode_stop_reason": getattr(mode, "stop_reason", None),
            "mode_force_calls": getattr(mode, "force_calls", None)})
    return {"index": record.get("index"), "request_start": start_cost,
            "requests": int(record.get("requests", 0) or 0), "status": record.get("status"),
            "accepted": record.get("accepted"), "atomic_scheduled": record.get("atomic_scheduled"),
            "atomic_path_diagnostic": atomic_path_diagnostic(record),
            "cell_failed_before_atomic": bool(record.get("atomic_scheduled") and cell_failure and atomic is None),
            "completed_gaussians": completed,
            "rotation_budget_released": sum(bool(event.get("rotation_budget_released")) for event in climbs),
            "stage_diagnostic_costs": stage,
            "stage_cost_unassigned_difference": int(record.get("requests", 0) or 0) - sum(stage.values()),
            "cell_cycles": cycle_trace,
            "atomic": panel.jsonable(atomic), "error": record.get("error")}


def _tree_difference(left, right):
    """Return maximum numeric absolute difference and categorical equality."""
    import numpy as np
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return None, False
        rows = [_tree_difference(left[key], right[key]) for key in left]
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return None, False
        rows = [_tree_difference(a, b) for a, b in zip(left, right)]
    elif isinstance(left, (float, int)) and not isinstance(left, bool) and isinstance(right, (float, int)) and not isinstance(right, bool):
        return abs(float(left) - float(right)), True
    else:
        return 0.0, left == right
    if any(not equal for _difference, equal in rows):
        return None, False
    differences = [difference for difference, _equal in rows if difference is not None]
    return (max(differences) if differences else 0.0), True


def paired_trace_check(cell_only, cell_plus_atomic):
    """Check shared initial/first-outer and pre-atomic stages at outer 1."""
    left = cell_only.get("outer_trace", [])
    right = cell_plus_atomic.get("outer_trace", [])
    by_left = {row["index"]: row for row in left}
    by_right = {row["index"]: row for row in right}
    checks = {}
    for index, label in ((0, "first_outer"), (1, "pre_atomic_cell_stage")):
        a, b = by_left.get(index), by_right.get(index)
        available = a is not None and b is not None
        difference, categories_match = (None, False) if not available else _tree_difference(
            a["cell_cycles"], b["cell_cycles"])
        checks[label] = {"available": available,
                         "max_abs_numeric_difference": difference,
                         "categorical_fields_match": categories_match,
                         "matches_within_trace_tolerance": bool(available and categories_match and
                             difference is not None and difference <= PAIR_TRACE_ATOL)}
    energy_difference, energies_comparable = _tree_difference(
        cell_only.get("initial_energy_eV"), cell_plus_atomic.get("initial_energy_eV"))
    geometry_difference, geometry_categories_match = _tree_difference(
        cell_only.get("initial_structure_signature"), cell_plus_atomic.get("initial_structure_signature"))
    checks["shared_initial"] = {
        "energy_abs_difference_eV": energy_difference if energies_comparable else None,
        "geometry_max_abs_difference": geometry_difference if geometry_categories_match else None,
        "geometry_categories_match": geometry_categories_match,
        "matches_within_trace_tolerance": bool(energies_comparable and geometry_categories_match and
            energy_difference is not None and geometry_difference is not None and
            max(energy_difference, geometry_difference) <= PAIR_TRACE_ATOL)}
    checks["diagnostic"] = "pre-treatment trace comparison only; not a physical-search qualification"
    return checks


def _arm(plan, case, arm_name, seed, out, deadline, started, reference_energy):
    import numpy as np
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone.block_ssw import run_block_ssw
    panel = panel_module()
    helper = load_module(QUAL_PATH, "tio2_phase_qualification_helpers_atomic_campaign")
    selected = treatment_configurations(plan)[arm_name]
    out.mkdir(parents=True, exist_ok=False)
    atoms = read(case["path"]); atoms.calc = None
    write(out / "input.extxyz", atoms)
    target = read(plan["target"]["path"]); target.calc = None
    dump(out / "effective-config.json", _config_row({arm_name: selected}))
    budget = _new_budget(helper, f"{seed}-{arm_name}-search", out / "search-ledger.jsonl",
                         SEARCH_CAP, ARM_WALL_SECONDS, deadline)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    calc = None
    result = None
    walker_exception = None
    surface = None
    arm_start = time.monotonic()
    try:
        if time.monotonic() >= deadline:
            raise TimeoutError("campaign wall deadline reached before arm startup")
        calc = MACECalculator(**_calc_kwargs(plan))
        helper.instrument_calculate(calc, budget)
        calc.reset()
        counted = helper.CountedStressSurface(calc, budget, "search")
        surface = StopAfterFailureSurface(counted)
        result = run_block_ssw(atoms, surface, steps=OUTER_STEPS,
                               config=selected["block"], rng=np.random.default_rng(seed))
    except Exception as error:
        walker_exception = {"error": repr(error), "traceback": traceback.format_exc()}
    arm_elapsed = time.monotonic() - arm_start
    records = [] if result is None else result.records
    minima = [] if result is None else list(result.minima)
    # The initial endpoint and every valid landing are recorded independently.
    initial_record = records[0] if records and records[0].get("stage") == "initial" else {}
    initial_quench = initial_record.get("quench")
    initial_eval = getattr(initial_quench, "evaluation", None)
    saved = []
    candidates = []
    if initial_eval is not None:
        candidates.append(("initial", initial_eval))
    outer_rows = []
    cumulative = int(initial_record.get("requests", 0) or 0)
    target_gates = None if initial_eval is None else _target_gates(
        initial_eval.atoms, initial_eval.energy, initial_eval.forces, initial_eval.stress,
        plan, target, reference_energy)
    first_target = None
    best_qualified = None
    if target_gates is not None and target_gates["physical_gate_passed"]:
        best_qualified = {"role": "initial", "evaluation": initial_eval,
                          "qualification": target_gates}
    initial_target_candidate = None
    if target_gates is not None and target_gates["target_candidate"]:
        initial_target_candidate = {**target_gates,
                                    "paid_requests_through_initial_quench": int(initial_record.get("requests", 0) or 0)}
        first_target = {"role": "initial", "evaluation": initial_eval,
                        "qualification": target_gates,
                        "paid_requests_through_outer": int(initial_record.get("requests", 0) or 0)}
    observed = []
    for record in records:
        if not isinstance(record, dict) or record.get("stage") == "initial":
            continue
        row = _outer_scalar(record, cumulative)
        landing = record.get("landing")
        landing_eval = getattr(landing, "evaluation", None)
        if landing_eval is not None and any(landing_eval is item for item in minima):
            # Evaluation dataclasses can include ndarray fields, so identify by object identity.
            if not any(landing_eval is prior for _role, prior in candidates):
                candidates.append((f"outer_{record.get('index')}", landing_eval))
            gates = _target_gates(landing_eval.atoms, landing_eval.energy,
                                  landing_eval.forces, landing_eval.stress, plan, target, reference_energy)
            row["landing_qualification"] = gates
            if best_qualified is None or (gates["physical_gate_passed"] and
                    gates["energy_eV"] < best_qualified["qualification"]["energy_eV"]):
                if gates["physical_gate_passed"]:
                    best_qualified = {"role": f"outer_{record.get('index')}",
                                      "evaluation": landing_eval, "qualification": gates}
            if gates["target_candidate"] and first_target is None:
                first_target = {"role": f"outer_{record.get('index')}",
                                "evaluation": landing_eval, "qualification": gates,
                                "paid_requests_through_outer": cumulative + row["requests"]}
        outer_rows.append(row)
        cumulative += row["requests"]
    # Save scalar stage trace and geometries without altering the live result object.
    for idx, (role, ev) in enumerate(candidates):
        path = out / f"endpoint-{idx:02d}-{role}.extxyz"
        write(path, ev.atoms)
        observed.append({"role": role, "path": path.name, "energy_eV": float(ev.energy),
                         "fmax_eV_A": float(np.linalg.norm(ev.forces, axis=1).max()),
                         "stress_max_eV_A3": float(np.abs(ev.stress).max())})
    (out / "outer-records.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in outer_rows))
    ledger = [json.loads(line) for line in (out / "search-ledger.jsonl").read_text().splitlines()]
    outcomes = classify_outer_records(outer_rows, ledger)
    for row, outcome in zip(outer_rows, outcomes):
        row["execution_outcome"] = outcome["outcome"]
    (out / "outer-records.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in outer_rows))
    paid = sum(item.get("event") in ("evaluation", "failure") for item in ledger)
    calls = sum(int(item.get("calculator_calls", 0) or 0) for item in ledger)
    failures = [item for item in ledger if item.get("event") == "failure"]
    denials = [item for item in ledger if item.get("event") == "denial"]
    status = ("paid_backend_failure" if failures else
              "budget_censored" if denials else
              "runner_error" if walker_exception else
              "completed" if result is not None and result.status == "completed" else
              "incomplete")
    if surface is not None and surface.paid_error:
        status = "paid_backend_failure"
    if surface is not None and surface.budget_boundary and not failures:
        status = "budget_censored"
    fresh_rows = []
    fresh_budget = _new_budget(helper, f"{seed}-{arm_name}-fresh", out / "fresh-ledger.jsonl",
                              FRESH_CAP, FRESH_WALL_SECONDS, deadline)
    if status not in ("paid_backend_failure", "runner_error") and calc is not None:
        unique = []
        seen = set()
        choices = []
        if initial_eval is not None: choices.append(("initial", initial_eval))
        if best_qualified is not None: choices.append(("best_physical", best_qualified["evaluation"]))
        if first_target is not None: choices.append(("first_target", first_target["evaluation"]))
        elif candidates: choices.append(("last_distinct", candidates[-1][1]))
        for role, ev in choices:
            digest = hashlib.sha256(ev.atoms.numbers.tobytes() + ev.atoms.positions.astype("<f8").tobytes() + ev.atoms.cell.array.astype("<f8").tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest); unique.append((role, ev))
        try:
            fresh_calc = MACECalculator(**_calc_kwargs(plan))
            helper.instrument_calculate(fresh_calc, fresh_budget)
            for role, ev in unique[:FRESH_CAP]:
                row = _fresh_check(fresh_calc, helper, ev.atoms, fresh_budget, role, target, plan, out,
                                   reference_energy)
                row["search_energy_eV"] = float(ev.energy)
                row["cold_energy_delta_eV"] = row["energy_eV"] - float(ev.energy)
                row["cold_energy_consistent_1e-6_eV"] = abs(row["cold_energy_delta_eV"]) <= 1e-6
                fresh_rows.append(row)
        except Exception as error:
            fresh_rows.append({"status": "fresh_paid_backend_failure" if fresh_budget.failures else
                               "fresh_budget_censored" if fresh_budget.denials else "fresh_runner_error",
                               "error": repr(error), "paid_requests": fresh_budget.requests})
    fresh_ledger = [json.loads(line) for line in (out / "fresh-ledger.jsonl").read_text().splitlines()]
    fresh_paid = sum(item.get("event") in ("evaluation", "failure") for item in fresh_ledger)
    fresh_calls = sum(int(item.get("calculator_calls", 0) or 0) for item in fresh_ledger)
    if any(row.get("status") in ("fresh_paid_backend_failure", "fresh_runner_error") for row in fresh_rows):
        status = ("fresh_backend_failure" if any(row.get("status") == "fresh_paid_backend_failure"
                 for row in fresh_rows) else "fresh_runner_error")
    payload = {"seed": seed, "arm": arm_name, "status": status,
               "walker_status": None if result is None else result.status,
               "walker_exception": walker_exception,
               "paid_error": None if surface is None else surface.paid_error,
               "budget_boundary_latched": None if surface is None else surface.budget_boundary,
               "search_requests": budget.requests, "search_ledger_paid": paid,
               "search_calculator_calls": calls, "search_failures": budget.failures,
               "search_denials": budget.denials, "search_boundary": budget.boundary,
               "search_elapsed_seconds": arm_elapsed, "outer_count": len(outer_rows),
               "outer_outcomes": outcomes, "outer_trace": outer_rows,
               "initial_energy_eV": None if initial_eval is None else float(initial_eval.energy),
               "initial_structure_signature": None if initial_eval is None else {
                   "numbers": initial_eval.atoms.numbers.tolist(),
                   "positions_A": np.round(initial_eval.atoms.positions, 12).tolist(),
                   "cell_A": np.round(initial_eval.atoms.cell.array, 12).tolist()},
               "initial_quench_qualification": target_gates,
               "initial_target_candidate_not_search_arrival": initial_target_candidate,
               "observed_minima": observed,
               "best_physical_search_endpoint": None if best_qualified is None else best_qualified["qualification"],
               "first_target_candidate": None if first_target is None else {
                   **first_target["qualification"], "paid_requests_through_outer": first_target["paid_requests_through_outer"]},
               "fresh_checks": fresh_rows, "fresh_requests": fresh_budget.requests,
               "fresh_ledger_paid": fresh_paid, "fresh_calculator_calls": fresh_calls,
               "elapsed_since_campaign_start_seconds": time.monotonic() - started,
               "budget_caps": {"search_requests": SEARCH_CAP, "fresh_requests": FRESH_CAP,
                               "outer_steps": OUTER_STEPS}}
    dump(out / "result.json", payload)
    return payload


def execute(out, *, plan=None, case_id=CASE_ID, seeds=SEEDS,
            protocol_path=None, entrypoint_path=None):
    started = time.monotonic()
    deadline = started + INTERNAL_WALL_SECONDS
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(ROOT))
    plan = load_plan() if plan is None else plan
    validated = validate_inputs(plan, case_id)
    case = validated["case"]
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    protocol_path = Path(protocol_path) if protocol_path is not None else HERE / "protocol.md"
    entrypoint_path = Path(entrypoint_path) if entrypoint_path is not None else Path(__file__).resolve()
    source_snapshot = out / "source-snapshot"
    source_snapshot.mkdir()
    snapshot_inputs = {"shared_runner": Path(__file__).resolve(), "entrypoint": entrypoint_path,
                       "protocol": protocol_path, "target_panel_helper": PANEL_PATH,
                       "qualification_helper": QUAL_PATH, "parent_plan": PLAN_PATH}
    for label, source in snapshot_inputs.items():
        if source.is_file():
            shutil.copy2(source, source_snapshot / f"{label}__{source.name}")
    meta = {"status": "started", "case_id": case_id,
            "scientific_scope": f"paired 48-atom {case_id} block-VC component transfer",
            "started_unix": time.time(), "internal_deadline_seconds": INTERNAL_WALL_SECONDS,
            "source_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_core": _core_provenance(), "runner_sha256": sha256(__file__),
            "entrypoint_path": str(entrypoint_path), "entrypoint_sha256": sha256(entrypoint_path),
            "protocol_path": str(protocol_path), "protocol_sha256": sha256(protocol_path),
            "source_snapshot": {label: {"path": str(source), "sha256": sha256(source)}
                                for label, source in snapshot_inputs.items() if source.is_file()},
            "input": {"path": case["path"], "sha256": sha256(case["path"])},
            "target": {"path": plan["target"]["path"], "sha256": sha256(plan["target"]["path"])},
            "model": {"path": plan["model"], "sha256": sha256(plan["model"])},
            "reused_helpers": {"target_panel": {"path": str(PANEL_PATH), "sha256": sha256(PANEL_PATH)},
                               "qualification": {"path": str(QUAL_PATH), "sha256": sha256(QUAL_PATH)},
                               "plan": {"path": str(PLAN_PATH), "sha256": sha256(PLAN_PATH)}},
            "seeds": list(seeds), "limits": {"search_requests_total": SEARCH_CAP_TOTAL,
                "fresh_requests_total": FRESH_CAP_TOTAL, "outer_per_arm": OUTER_STEPS,
                "per_arm_search_requests": SEARCH_CAP, "per_arm_fresh_requests": FRESH_CAP}}
    dump(out / "manifest.json", meta)
    # Snapshot frozen source inputs and the exact effective pair before EFS work.
    dump(out / "frozen-plan.json", plan)
    dump(out / "frozen-configs.json", _config_row(validated["pair"]))
    from ase.io import read, write
    frozen_target = read(plan["target"]["path"]); frozen_target.calc = None
    write(out / "target.extxyz", frozen_target)
    write(out / "input.extxyz", validated["atoms"])
    helper = load_module(QUAL_PATH, "tio2_phase_qualification_helpers_atomic_reference")
    try:
        # One independent model evaluation anchors target energy to this exact backend.
        from mace.calculators import MACECalculator
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        ref_dir = out / "shared-target"
        ref_dir.mkdir()
        reference_budget = _new_budget(helper, "shared-target", ref_dir / "ledger.jsonl", 1,
                                       FRESH_WALL_SECONDS, deadline)
        calc = MACECalculator(**_calc_kwargs(plan)); helper.instrument_calculate(calc, reference_budget)
        calc.reset()
        energy, forces, stress = helper.CountedStressSurface(
            calc, reference_budget, "shared_target").evaluate(frozen_target)
        reference = _geometry_metrics(frozen_target, energy, forces, stress)
        reference["physical_gate_passed"] = (reference["fmax_eV_A"] <= plan["target"]["fmax_eV_A"] and
                                              reference["stress_max_eV_A3"] <= plan["target"]["stress_tol_eV_A3"])
        if not reference["physical_gate_passed"]:
            raise RuntimeError(f"fresh target reference failed force/stress gate: {reference}")
        reference.update(calculator_calls=reference_budget.calculator_calls, paid_requests=reference_budget.requests)
        dump(ref_dir / "result.json", reference)
        meta["reference_energy_eV_per_atom"] = reference["energy_per_atom_eV"]
        meta["shared_reference_requests"] = reference_budget.requests
        dump(out / "manifest.json", meta)
        arm_rows = []
        stop_all = False
        for seed in seeds:
            for arm_name in ("cell_only", "cell_plus_atomic"):
                arm_out = out / f"seed-{seed}-{arm_name}"
                if stop_all or time.monotonic() >= deadline:
                    row = {"seed": seed, "arm": arm_name,
                           "status": "not_started_no_budget" if not stop_all else "not_run_after_campaign_failure"}
                    dump(arm_out.with_suffix(".json"), row)
                    arm_rows.append(row)
                    continue
                row = _arm(plan, case, arm_name, seed, arm_out, deadline, started,
                           reference["energy_eV"])
                row["reference_energy_eV_per_atom"] = reference["energy_per_atom_eV"]
                dump(arm_out / "result.json", row)
                arm_rows.append(row)
                if row["status"] in ("paid_backend_failure", "runner_error", "fresh_backend_failure", "fresh_runner_error"):
                    stop_all = True
        pairs = {}
        for seed in seeds:
            cell = next((row for row in arm_rows if row.get("seed") == seed and row.get("arm") == "cell_only"), {})
            both = next((row for row in arm_rows if row.get("seed") == seed and row.get("arm") == "cell_plus_atomic"), {})
            pairs[str(seed)] = paired_trace_check(cell, both)
        dump(out / "paired-trace-checks.json", pairs)
        final_status = ("stopped_after_failure" if stop_all else
                        "campaign_wall_censored" if time.monotonic() >= deadline else
                        "completed" if len(arm_rows) == 4 and all(row["status"] in ("completed", "budget_censored") for row in arm_rows) else
                        "incomplete")
        meta.update(status=final_status, elapsed_seconds=time.monotonic() - started,
                    total_search_paid=sum(int(row.get("search_ledger_paid", 0)) for row in arm_rows),
                    total_search_calculator_calls=sum(int(row.get("search_calculator_calls", 0)) for row in arm_rows),
                    total_fresh_paid=sum(int(row.get("fresh_ledger_paid", 0)) for row in arm_rows),
                    total_fresh_and_reference_paid=(int(reference_budget.requests) +
                        sum(int(row.get("fresh_ledger_paid", 0)) for row in arm_rows)),
                    total_fresh_and_reference_calculator_calls=(int(reference_budget.calculator_calls) +
                        sum(int(row.get("fresh_calculator_calls", 0)) for row in arm_rows)),
                    arm_results=[{"seed": row.get("seed"), "arm": row.get("arm"), "status": row.get("status")} for row in arm_rows],
                    paired_trace_checks=pairs)
        dump(out / "manifest.json", meta)
    except Exception as error:
        meta.update(status="runner_error", error=repr(error), traceback=traceback.format_exc(),
                    elapsed_seconds=time.monotonic() - started)
        dump(out / "manifest.json", meta)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="zero-PES source/config/input checks")
    parser.add_argument("--execute", action="store_true", help="run the frozen bounded campaign")
    parser.add_argument("--out", type=Path, help="new, nonexistent output directory")
    args = parser.parse_args()
    if args.execute == args.preflight:
        parser.error("choose exactly one of --preflight or --execute")
    if args.preflight:
        print(json.dumps(preflight(), indent=2, allow_nan=False))
    else:
        if args.out is None:
            parser.error("--execute requires --out")
        execute(args.out.resolve())


if __name__ == "__main__":
    main()
