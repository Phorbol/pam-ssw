#!/usr/bin/env python3
"""Bounded paired TiO2 variable-cell target-discovery panel.

Each slot runs one existing whole walker once for 40 outer steps. Preflight
uses a dummy calculator only; real model evaluations require --slot.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN_PATH = HERE / "plan.json"
QUAL_HELPER_PATH = ROOT / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run.py"
QUAL_DIR = QUAL_HELPER_PATH.parent
QUAL_RUN_PATH = QUAL_DIR / "run.py"
PAIR_SEEDS = (26100751, 26100752)
# Slot order is part of the submitted array contract: size first, joint then block.
SLOTS = ((0, "joint_vc"), (0, "block"), (1, "joint_vc"), (1, "block"))
OUTER_STEPS = 40
SEARCH_CAP = 20_000
ARM_SECONDS = 600.0
FRESH_CAP = 3
FRESH_SECONDS = 30.0


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def jsonable(value):
    """Small scalar/dataclass summary; omit geometry and array payloads."""
    import numpy as np
    from ase import Atoms
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Atoms) or isinstance(value, np.ndarray):
        return None
    if hasattr(value, "__dataclass_fields__"):
        return {key: jsonable(getattr(value, key)) for key in value.__dataclass_fields__
                if key not in {"atoms", "forces", "stress", "gradient", "positions", "cell",
                               "checkpoint", "pending", "q"}}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()
                if key not in {"atoms", "last_work", "forces", "stress", "gradient",
                               "center", "direction", "initial_direction", "anchor", "q",
                               "checkpoint", "pending"}}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return jsonable(value.item())
    return str(value)


def plan_data():
    if not PLAN_PATH.is_file():
        raise FileNotFoundError(f"plan not present: {PLAN_PATH}")
    return json.loads(PLAN_PATH.read_text())


def target_config(plan=None):
    """Build the frozen settings recorded in the approved plan."""
    from pamssw.standalone.paper_reference import SSWConfig
    from pamssw.standalone.block_ssw import BlockSSWConfig
    from pamssw.standalone.vc_reference import VCSSWConfig
    plan = plan_data() if plan is None else plan
    atomic = SSWConfig(**plan["atomic_config"])
    block = BlockSSWConfig(atomic=atomic, **plan["block_config"])
    joint = VCSSWConfig(**plan["joint_config"])
    return atomic, block, joint


def validate_plan(plan):
    required = {"model", "model_sha256", "calculator", "cases", "target",
                "identity_tolerances", "budget"}
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"plan missing required fields: {missing}")
    if plan["calculator"] != {"head": "omat_pbe", "device": "cuda",
            "dtype": "float64", "enable_cueq": False, "enable_oeq": False}:
        raise ValueError("calculator settings differ from approved OMAT-small protocol")
    budget = plan["budget"]
    expected = {"per_arm_requests": SEARCH_CAP, "outer_steps": OUTER_STEPS,
                "per_arm_seconds": ARM_SECONDS, "fresh_requests_per_arm": FRESH_CAP,
                "shared_reference_requests": 1, "max_concurrent": 2}
    for key, value in expected.items():
        if budget.get(key) != value:
            raise ValueError(f"budget {key}={budget.get(key)!r}; expected {value!r}")
    if len(plan["cases"]) != 2 or [c.get("id") for c in plan["cases"]] != ["phase87_12", "phase87_48"]:
        raise ValueError("cases must be ordered phase87_12, phase87_48")
    model = Path(plan["model"])
    if not model.is_file() or sha256(model) != plan["model_sha256"]:
        raise RuntimeError(f"OMAT-small missing or SHA256 mismatch: {model}")
    from ase.io import read
    expected_n = {"phase87_12": 12, "phase87_48": 48}
    for case in plan["cases"]:
        source = Path(case["path"])
        if not source.is_file() or sha256(source) != case["sha256"]:
            raise RuntimeError(f"input missing or SHA256 mismatch: {source}")
        atoms = read(source)
        if (len(atoms) != expected_n[case["id"]] or case.get("natoms") != expected_n[case["id"]]
                or not atoms.pbc.all() or atoms.constraints):
            raise ValueError(f"{case['id']} violates atom-count/PBC/constraint contract")
        if atoms.get_chemical_formula() != ("O8Ti4" if len(atoms) == 12 else "O32Ti16"):
            raise ValueError(f"unexpected composition: {atoms.get_chemical_formula()}")
    target = plan["target"]
    if not Path(target["path"]).is_file() or sha256(target["path"]) != target["sha256"]:
        raise RuntimeError("qualified anatase target missing/hash mismatch")
    target_atoms = read(target["path"])
    if (len(target_atoms) != target.get("natoms") or target.get("natoms") != 12
            or target_atoms.get_chemical_formula() != "O8Ti4"
            or not target_atoms.pbc.all() or target_atoms.constraints):
        raise ValueError("qualified anatase target violates atom-count/composition/PBC contract")
    if (target.get("fmax_eV_A") != 0.05 or target.get("stress_tol_eV_A3") != 0.001
            or target.get("energy_tolerance_eV_per_atom") != 0.001):
        raise ValueError("target gate differs from the approved force/stress/energy window")
    if set(plan["identity_tolerances"]) != {"tight", "broad"}:
        raise ValueError("plan must provide tight and broad identity tolerances")
    target_config(plan)
    return model


def config_record():
    plan = plan_data()
    return {"purpose": "actual anatase-phase discovery from phase87 inputs on existing OMAT-small",
            "claim_scope": "paired whole-walker comparison; not component causality or DFT paper-rate replication",
            "walkers": {"atomic": plan["atomic_config"], "block": plan["block_config"],
                        "joint_vc": plan["joint_config"]},
            "paired_rng_seeds": list(PAIR_SEEDS), "outer_steps": OUTER_STEPS,
            "budget": {"per_arm_search_requests_including_initial_and_failures": SEARCH_CAP,
                       "per_arm_outer_attempts": OUTER_STEPS, "per_arm_wall_seconds": ARM_SECONDS,
                       "per_arm_fresh_requests": FRESH_CAP, "per_arm_fresh_seconds": FRESH_SECONDS,
                       "panel_search_request_cap": 80_000, "panel_fresh_request_cap": 13,
                       "first_target_cost": "all paid requests through its completed outer step"},
            "target_gate": "physical fmax/stress + tight and broad pymatgen identity + energy per atom window"}


def provenance(plan):
    module_names = {"atomic_climb": "pamssw.standalone.atomic_climb",
        "block_ssw": "pamssw.standalone.block_ssw", "cbd_cell": "pamssw.standalone.cbd_cell",
        "direction": "pamssw.standalone.direction", "dimer": "pamssw.standalone.dimer",
        "gaussian": "pamssw.standalone.gaussian", "paper_reference": "pamssw.standalone.paper_reference",
        "periodic_identity": "pamssw.standalone.periodic_ga_reference",
        "periodic_geometry": "pamssw.standalone.periodic_geometry",
        "vc_reference": "pamssw.standalone.vc_reference", "vc_geometry": "pamssw.standalone.vc_geometry",
        "cell_relax": "pamssw.standalone.cell_relax",
        "generalized_numerics": "pamssw.standalone.generalized_numerics",
        "surface": "pamssw.standalone.surface", "ase": "ase", "mace": "mace", "torch": "torch"}
    paths = {key: Path(importlib.import_module(module_name).__file__).resolve()
             for key, module_name in module_names.items()}
    paths["atomic_config"] = paths["paper_reference"]
    paths.update(qualification_helper=QUAL_RUN_PATH.resolve(), runner=Path(__file__).resolve())
    return {"checkout": str(ROOT), "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "head_pamssw_tree": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD:pamssw"], text=True).strip(),
            "python": sys.executable, "python_version": platform.python_version(),
            "imports": {name: str(path) for name, path in paths.items()},
            "source_sha256": {name: sha256(path) for name, path in paths.items()},
            "plan_sha256": sha256(PLAN_PATH),
            "protocol_sha256": sha256(HERE / "protocol.md"),
            "model": plan["model"], "model_sha256": sha256(plan["model"]),
            "inputs": {row["id"]: {"path": row["path"], "sha256": sha256(row["path"])} for row in plan["cases"]},
            "target": {"path": plan["target"]["path"], "sha256": sha256(plan["target"]["path"])}}


def preflight(plan):
    validate_plan(plan)
    sys.path.insert(0, str(ROOT))
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from mace.calculators import MACECalculator
    import numpy as np
    from pamssw.standalone.block_ssw import run_block_ssw
    from pamssw.standalone.vc_reference import run_vc_ssw
    from tempfile import TemporaryDirectory
    helper = load_module(QUAL_RUN_PATH, "tio2_phase_qualification_helpers")
    helper_plan = json.loads((QUAL_DIR / "plan.json").read_text())
    helper_budget_checks = helper.preflight(helper_plan)
    from ase.io import read
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    target_atoms = read(plan["target"]["path"])
    identity_checks = {}
    for label, settings in plan["identity_tolerances"].items():
        matcher = pymatgen_identity(**settings)
        self_match = bool(matcher(target_atoms, target_atoms.copy()))
        supercell_match = bool(matcher(target_atoms, target_atoms.repeat((2, 2, 1))))
        identity_checks[label] = {"target_self_match": self_match,
                                  "target_48_atom_supercell_match": supercell_match}
        if not self_match or not supercell_match:
            raise RuntimeError(f"anatase identity preflight failed for {label}: {identity_checks[label]}")

    class Dummy(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0.0, "forces": np.zeros_like(self.atoms.positions),
                            "stress": np.zeros(6)}

    atomic, block, joint = target_config(plan)
    api = {"run_block_ssw": str(__import__(run_block_ssw.__module__, fromlist=["x"]).__file__),
           "run_vc_ssw": str(__import__(run_vc_ssw.__module__, fromlist=["x"]).__file__),
           "mace_constructor": str(__import__("inspect").signature(MACECalculator)),
           "block_signature": str(__import__("inspect").signature(run_block_ssw)),
           "joint_signature": str(__import__("inspect").signature(run_vc_ssw)),
           "qualification_helpers": str(QUAL_RUN_PATH)}
    # Invoke each complete one-step API once on dummy PES to exercise result
    # schema and counted surface behavior without real force evaluations.
    with TemporaryDirectory(prefix="tio2-vc-panel-preflight-") as temp:
        for label, walker, cfg in (("block", run_block_ssw, block), ("joint_vc", run_vc_ssw, joint)):
            ledger = Path(temp) / f"{label}.jsonl"
            budget = helper.CaseBudget(label, ledger, 2000, 120.0, time.monotonic() + 120.0)
            calc = Dummy(); helper.instrument_calculate(calc, budget)
            surface = helper.CountedStressSurface(calc, budget, "dummy")
            dummy = Atoms("Ti4O8", positions=np.array([[i % 4, (i // 4) % 2, i // 8] for i in range(12)], float),
                          cell=np.eye(3) * 10., pbc=True)
            if label == "block":
                result = walker(dummy, surface, steps=1, config=cfg, rng=np.random.default_rng(7))
            else:
                result = walker(dummy, surface, steps=1, config=cfg, rng=np.random.default_rng(7))
            if result.status not in ("completed", "initial_quench_failed") or budget.requests != surface.requests:
                raise RuntimeError(f"dummy walker preflight failed for {label}: {result.status}")
            api[label + "_dummy"] = {"status": result.status, "requests": budget.requests,
                                     "records": len(result.records), "calculator_calls": budget.calculator_calls}
    return {"status": "preflight_passed", "real_pes_requests": 0,
            "configuration": config_record(), "api_checks": api,
            "qualification_helper_budget_cache_preflight": helper_budget_checks,
            "identity_preflight": identity_checks,
            "provenance": provenance(plan)}


def _clean_atoms(atoms):
    result = atoms.copy()
    result.calc = None
    result.constraints = []
    return result


def _identity_setup(plan):
    from ase.io import read
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    matchers = {key: pymatgen_identity(**settings) for key, settings in plan["identity_tolerances"].items()}
    return read(plan["target"]["path"]), matchers


def _metrics(evaluation):
    import numpy as np
    return {"energy_eV": float(evaluation.energy),
            "energy_per_atom_eV": float(evaluation.energy / len(evaluation.atoms)),
            "fmax_eV_A": float(np.linalg.norm(evaluation.forces, axis=1).max()),
            "stress_max_eV_A3": float(np.abs(evaluation.stress).max()),
            "volume_A3": float(evaluation.volume), "natoms": len(evaluation.atoms)}


def _target_check(evaluation, target, matchers, threshold):
    metric = _metrics(evaluation)
    matches = {}
    for name, matcher in matchers.items():
        try:
            matches[name] = {"matches": bool(matcher(evaluation.atoms, target)), "status": "completed"}
        except Exception as error:
            matches[name] = {"matches": None, "status": "error", "error": repr(error)}
    identity = all(row["matches"] is True for row in matches.values())
    spec = plan_data()["target"]
    energy_tolerance = spec["energy_tolerance_eV_per_atom"]
    gate = (metric["fmax_eV_A"] <= spec["fmax_eV_A"] and
            metric["stress_max_eV_A3"] <= spec["stress_tol_eV_A3"] and identity
            and metric["energy_per_atom_eV"] <= threshold + energy_tolerance)
    return {**metric, "anatase_geometry_match": identity,
            "identity_matches": matches, "identity_qualified": identity,
            "energy_window_threshold_eV_atom": threshold,
            "energy_window_tolerance_eV_atom": energy_tolerance,
            "energy_window_passed": metric["energy_per_atom_eV"] <= threshold + energy_tolerance,
            "physical_gate_passed": metric["fmax_eV_A"] <= spec["fmax_eV_A"] and metric["stress_max_eV_A3"] <= spec["stress_tol_eV_A3"],
            "target_candidate": bool(gate)}


def _append_geometry(path: Path, atoms, info):
    from ase.io import write
    clean = _clean_atoms(atoms)
    clean.info.update(info)
    write(path, clean, format="extxyz")


def _save_evaluation(path: Path, evaluation, info):
    metric = _metrics(evaluation)
    _append_geometry(path, evaluation.atoms, {**info, **metric})
    return metric


def _as_evaluation(value):
    """Unwrap block-quench records; joint walker stores VCEvaluation directly."""
    if value is None:
        return None
    return getattr(value, "evaluation", value)


def _record_stage_costs(record):
    costs = {"cell_rotation_and_partial": 0, "atomic_rotation": 0,
             "gaussian_height": 0, "biased_quench": 0,
             "true_checks_and_landing_quench": 0}
    for cycle in record.get("cell_cycles", []) or []:
        costs["cell_rotation_and_partial"] += int(cycle.get("requests", 0) or 0)
    climb_result = record.get("atomic")
    if climb_result is not None:
        for event in getattr(climb_result, "climb", ()):
            costs["atomic_rotation"] += int(event.get("rotation_force_requests", 0) or 0)
            costs["biased_quench"] += int(event.get("quench_requests", 0) or 0)
            costs["true_checks_and_landing_quench"] += int("true_energy" in event)
    for event in record.get("climb", []) or []:
        costs["atomic_rotation"] += int(event.get("rotation_requests", 0) or 0)
        costs["gaussian_height"] += int(event.get("height_requests", 0) or 0)
        costs["biased_quench"] += int(event.get("biased_quench_requests", 0) or 0)
        costs["true_checks_and_landing_quench"] += int(event.get("true_check_requests", 0) or 0)
    optimizer = record.get("landing_optimizer")
    if isinstance(optimizer, dict):
        costs["true_checks_and_landing_quench"] += int(optimizer.get("requests", 0) or 0)
    landing_record = record.get("landing")
    if landing_record is not None and hasattr(landing_record, "requests"):
        costs["true_checks_and_landing_quench"] += int(landing_record.requests or 0)
    return costs


def execute_slot(plan, slot: int, out: Path):
    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone.block_ssw import run_block_ssw
    from pamssw.standalone.vc_reference import run_vc_ssw
    sys.path.insert(0, str(ROOT))
    helper = load_module(QUAL_RUN_PATH, "tio2_phase_qualification_helpers_exec")
    targets, matchers = _identity_setup(plan)
    case_index, arm = SLOTS[slot]
    case = plan["cases"][case_index]
    seed = int(case["paired_seed"])
    target = targets
    ref_energy = float(plan["target"]["energy_eV"] / plan["target"]["natoms"])
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "effective-plan.json", plan)
    (out / "source-snapshot").mkdir()
    shutil.copy2(Path(__file__).resolve(), out / "source-snapshot" / "runner__run.py")
    shutil.copy2(QUAL_RUN_PATH.resolve(), out / "source-snapshot" / "qualification_helper__run.py")
    protocol = HERE / "protocol.md"
    if protocol.is_file():
        shutil.copy2(protocol, out / "source-snapshot" / "protocol.md")
    # Snapshot exact Python dependencies used by the existing walkers.
    provenance_row = provenance(plan)
    for name, source in provenance_row["imports"].items():
        p = Path(source)
        if p.is_file() and p.suffix == ".py":
            shutil.copy2(p, out / "source-snapshot" / (name.replace("/", "_") + "__" + p.name))
    dump(out / "provenance.json", provenance_row)
    input_atoms = read(case["path"]); input_atoms.calc = None
    shutil.copy2(case["path"], out / "input-source.extxyz")
    shutil.copy2(plan["target"]["path"], out / "anatase-target-source.extxyz")
    calc_spec = plan["calculator"]
    calc_kw = {"model_paths": plan["model"], "head": calc_spec["head"],
               "device": calc_spec["device"], "default_dtype": calc_spec["dtype"],
               "enable_cueq": calc_spec["enable_cueq"], "enable_oeq": calc_spec["enable_oeq"]}
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    budget = helper.CaseBudget(f"{case['id']}-{arm}", out / "requests.jsonl", SEARCH_CAP,
                               ARM_SECONDS, started + ARM_SECONDS)
    rng = np.random.default_rng(seed)
    atomic, block_cfg, joint_cfg = target_config(plan)
    try:
        calc = MACECalculator(**calc_kw)
        helper.instrument_calculate(calc, budget)
        calc.reset()
        surface = helper.CountedStressSurface(calc, budget, "search")
        # The only search call. In particular, no repeated steps=1 calls.
        if arm == "block":
            result = run_block_ssw(input_atoms, surface, steps=OUTER_STEPS,
                                   config=block_cfg, rng=rng)
        else:
            result = run_vc_ssw(input_atoms, surface, steps=OUTER_STEPS,
                                config=joint_cfg, rng=rng)
        walker_exception = None
    except Exception as error:
        result = None
        walker_exception = {"error": repr(error), "traceback": traceback.format_exc()}
    elapsed = time.monotonic() - started
    rows = []
    records = [] if result is None else result.records
    minima = [] if result is None else list(result.minima)
    valid_minimum_ids = {id(item) for item in minima}
    initial_record = records[0] if records and records[0].get("stage") == "initial" else {}
    initial_value = None if result is None else result.initial
    initial_eval = _as_evaluation(initial_value)
    initial_valid = initial_eval is not None and id(initial_eval) in valid_minimum_ids
    initial_paid = int(initial_record.get("requests", 0) or 0)
    current_atoms = None if initial_eval is None else _clean_atoms(initial_eval.atoms)
    if initial_eval is not None:
        _save_evaluation(out / "initial-endpoint.extxyz", initial_eval,
                         {"role": "initial_all_dof_endpoint", "arm": arm, "case": case["id"]})
    outer_rows = []
    cumulative_paid = initial_paid
    first_target = None
    first_geometry_arrival = None
    target_evaluation = None
    minimum_index = 0
    if initial_eval is not None and id(initial_eval) in valid_minimum_ids:
        initial_check = _target_check(initial_eval, target, matchers, ref_energy)
        initial_check.update(minimum_index=minimum_index, outer_index=None,
                             paid_requests_through_outer=initial_paid,
                             observed_at="initial_all_dof_quench", accepted=None)
        initial_check["structure_path"] = f"minimum-{minimum_index:03d}.extxyz"
        rows.append(initial_check)
        _save_evaluation(out / initial_check["structure_path"], initial_eval,
                         {"role": "observed_minimum", "minimum_index": minimum_index,
                          "paid_requests_through_outer": initial_paid, **_metrics(initial_eval)})
        if initial_check["anatase_geometry_match"]:
            first_geometry_arrival = initial_check
        if initial_check["target_candidate"]:
            first_target = initial_check
            target_evaluation = initial_eval
        minimum_index += 1
    for record in records:
        if not isinstance(record, dict) or record.get("stage") == "initial":
            continue
        i = int(record.get("index", len(outer_rows)))
        if current_atoms is not None:
            _append_geometry(out / f"outer-{i:03d}-start.extxyz", current_atoms,
                             {"role": "outer_start", "outer_index": i})
        landing_record = record.get("landing")
        landing_eval = _as_evaluation(landing_record)
        if landing_eval is not None and hasattr(landing_eval, "atoms"):
            if id(landing_eval) in valid_minimum_ids:
                candidate = _target_check(landing_eval, target, matchers, ref_energy)
                candidate.update(outer_index=i,
                    paid_requests_through_outer=cumulative_paid + int(record.get("requests", 0) or 0),
                    minimum_index=minimum_index, observed_at="completed_landing",
                    outer_status=record.get("status"), accepted=bool(record.get("accepted", False)))
                candidate["structure_path"] = f"minimum-{minimum_index:03d}.extxyz"
                rows.append(candidate)
                _save_evaluation(out / candidate["structure_path"], landing_eval,
                    {"role": "observed_minimum", "minimum_index": minimum_index,
                     "outer_index": i, "accepted": bool(record.get("accepted", False)),
                     "paid_requests_through_outer": candidate["paid_requests_through_outer"],
                     **_metrics(landing_eval)})
                minimum_index += 1
                if candidate["anatase_geometry_match"] and first_geometry_arrival is None:
                    first_geometry_arrival = candidate
                if candidate["target_candidate"] and first_target is None:
                    first_target = candidate
                    target_evaluation = landing_eval
                if record.get("accepted"):
                    current_atoms = _clean_atoms(landing_eval.atoms)
                _append_geometry(out / f"outer-{i:03d}-landing.extxyz", landing_eval.atoms,
                    {"role": "completed_landing_minimum", "outer_index": i,
                     "accepted": bool(record.get("accepted", False)), **_metrics(landing_eval)})
            else:
                partial_metrics = _metrics(landing_eval)
                _append_geometry(out / f"outer-{i:03d}-partial-landing.extxyz", landing_eval.atoms,
                    {"role": "uncertified_partial_landing", "outer_index": i,
                     "status": str(record.get("status", "unknown")), **partial_metrics})
                record["partial_landing_metrics"] = partial_metrics
        work = record.get("last_work")
        if work is not None:
            _append_geometry(out / f"outer-{i:03d}-last-work.extxyz", work,
                {"role": "last_work", "outer_index": i,
                 "status": str(record.get("status", "unknown"))})
        if current_atoms is not None:
            _append_geometry(out / f"outer-{i:03d}-current.extxyz", current_atoms,
                {"role": "current_after_outer", "outer_index": i,
                 "accepted": bool(record.get("accepted", False))})
        cost = int(record.get("requests", 0) or 0)
        stage_costs = _record_stage_costs(record)
        outer_rows.append({"outer_index": i, "status": record.get("status"),
            "accepted": record.get("accepted"), "request_cost": cost,
            "cumulative_paid_requests": cumulative_paid + cost,
            "stage_costs": stage_costs,
            "stage_cost_unassigned_difference": cost - sum(stage_costs.values()),
            "cell_cycles": jsonable(record.get("cell_cycles", [])),
            "atomic": jsonable(record.get("atomic")),
            "climb": jsonable(record.get("climb", [])),
            "landing_optimizer": jsonable(record.get("landing_optimizer")),
            "partial_landing_metrics": record.get("partial_landing_metrics"),
            "error": record.get("error")})
        cumulative_paid += cost
    (out / "requests.jsonl").touch(exist_ok=True)
    paid = budget.requests
    ledger_events = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
    raw_paid = sum(1 for item in ledger_events if item.get("event") in {"evaluation", "failure"})
    raw_calls = sum(int(item.get("calculator_calls", 0) or 0) for item in ledger_events)
    recorded_paid = initial_paid + sum(row["request_cost"] for row in outer_rows)
    result_requests = None if result is None else int(result.requests)
    cost_closure = {"budget_paid_requests": paid,
        "raw_ledger_paid_requests": raw_paid,
        "walker_reported_requests": result_requests,
        "initial_plus_outer_records": recorded_paid,
        "walker_minima_count": len(minima), "saved_minima_scalar_rows": len(rows),
        "raw_ledger_calculator_calls": raw_calls,
        "ledger_matches_budget": raw_paid == paid,
        "walker_matches_budget": None if result_requests is None else result_requests == paid,
        "records_match_budget": None if result is None else recorded_paid == paid,
        "minima_rows_match_walker": None if result is None else len(rows) == len(minima)}
    first_target_paid = None if first_target is None else first_target["paid_requests_through_outer"]
    first_geometry_paid = None if first_geometry_arrival is None else first_geometry_arrival["paid_requests_through_outer"]
    with (out / "outer-records.jsonl").open("w") as stream:
        for row in outer_rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with (out / "minima.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    request_cost_by_stage = {"initial_all_dof_quench": initial_paid,
                             "outer_attempts_total": sum(row["request_cost"] for row in outer_rows),
                             "outer_stages": {key: sum(row["stage_costs"][key] for row in outer_rows)
                                              for key in (outer_rows[0]["stage_costs"] if outer_rows else [])},
                             "outer_stage_unassigned_difference": sum(
                                 row["stage_cost_unassigned_difference"] for row in outer_rows)}
    result_payload = {"case": case["id"], "arm": arm, "seed": seed,
        "status": "exception" if result is None else result.status,
        "walker_exception": walker_exception, "numerically_qualified_initial": bool(initial_valid),
        "initial_force_stress_passed": bool(initial_eval is not None and
            _target_check(initial_eval, target, matchers, ref_energy)["physical_gate_passed"]),
        "initial_record": jsonable(initial_record),
        "search_paid_requests": paid, "search_calculator_calls": budget.calculator_calls,
        "search_failures": budget.failures, "search_denials": budget.denials,
        "search_boundary": budget.boundary, "search_elapsed_seconds": elapsed,
        "recorded_minima": len(rows), "outer_attempts_recorded": len(outer_rows),
        "first_anatase_geometry_arrival": first_geometry_arrival,
        "first_anatase_geometry_paid_requests": first_geometry_paid,
        "first_target_candidate": first_target, "first_target_paid_requests": first_target_paid,
        "final_paid_requests": paid, "stage_request_counts": request_cost_by_stage,
        "cost_closure": cost_closure,
        "request_closure": None if result is None else bool(
            cost_closure["ledger_matches_budget"] and cost_closure["walker_matches_budget"] and
            cost_closure["records_match_budget"]),
        "fresh_checks": [], "config": config_record()}
    dump(out / "search-result.json", result_payload)
    best_eval = None if result is None else _as_evaluation(result.best)
    if best_eval is not None and id(best_eval) not in valid_minimum_ids:
        best_eval = None
    last_eval = None if not minima else _as_evaluation(minima[-1])
    try:
        fresh_result = run_fresh_checks(plan, case, arm, out, initial_eval,
                                        best_eval, target_evaluation, last_eval, result_payload)
    except Exception as error:
        fresh_result = {"fresh_checks": [], "fresh_requests": 0,
                        "fresh_calculator_calls": 0, "fresh_failures": 0,
                        "fresh_denials": 0, "fresh_elapsed_seconds": 0.0,
                        "fresh_exception": repr(error),
                        "first_target_cold_confirmed": False,
                        "first_target_discovery_cost": first_target_paid}
    if slot == 0:
        result_payload["shared_reference_cold_check"] = shared_reference_check(plan, out, helper,
                                                                                  MACECalculator)
    result_payload.update(fresh_result)
    dump(out / "result.json", result_payload)
    return result_payload


def _geometry_key(atoms):
    import numpy as np
    clean = _clean_atoms(atoms)
    h = hashlib.sha256()
    h.update(clean.numbers.tobytes()); h.update(np.asarray(clean.cell.array, dtype="<f8").tobytes())
    h.update(np.asarray(clean.positions, dtype="<f8").tobytes()); h.update(np.asarray(clean.pbc, dtype=np.uint8).tobytes())
    return h.hexdigest()


def run_fresh_checks(plan, case, arm, out, initial_eval, best_eval,
                     first_candidate_eval, last_eval, result_payload):
    helper = load_module(QUAL_RUN_PATH, "tio2_phase_qualification_helpers_fresh")
    targets, matchers = _identity_setup(plan)
    target = targets
    threshold = float(plan["target"]["energy_eV"] / plan["target"]["natoms"])
    candidates = []
    if initial_eval is not None: candidates.append(("initial", initial_eval.atoms, initial_eval.energy))
    if best_eval is not None: candidates.append(("best", best_eval.atoms, best_eval.energy))
    if first_candidate_eval is not None:
        candidates.append(("first_target_candidate", first_candidate_eval.atoms,
                           first_candidate_eval.energy))
    elif last_eval is not None:
        candidates.append(("last_available_minimum", last_eval.atoms, last_eval.energy))
    unique, seen = [], set()
    for role, atoms, energy in candidates:
        key = _geometry_key(atoms)
        if key not in seen:
            unique.append((role, _clean_atoms(atoms), float(energy), key)); seen.add(key)
    unique = unique[:FRESH_CAP]
    fresh = {"model_paths": plan["model"], "head": plan["calculator"]["head"],
             "device": "cuda", "default_dtype": "float64", "enable_cueq": False, "enable_oeq": False}
    fresh_dir = out / "fresh"
    fresh_dir.mkdir(exist_ok=True)
    fresh_started = time.monotonic()
    fresh_budget = helper.CaseBudget(f"{case['id']}-{arm}-fresh", fresh_dir / "requests.jsonl",
        FRESH_CAP, FRESH_SECONDS, fresh_started + FRESH_SECONDS)
    checked = []
    fresh_exception = None
    try:
        from mace.calculators import MACECalculator
        calc = MACECalculator(**fresh)
        helper.instrument_calculate(calc, fresh_budget)
        for index, (role, atoms, old_energy, key) in enumerate(unique):
            row = {"role": role, "geometry_sha256": key}
            try:
                calc.reset()
                surface = helper.CountedStressSurface(calc, fresh_budget, "fresh_" + role)
                energy, forces, stress = surface.evaluate(atoms)
                from types import SimpleNamespace
                evaluation = SimpleNamespace(energy=energy, atoms=atoms, forces=forces,
                                             stress=stress, volume=float(atoms.get_volume()))
                checked_target = _target_check(evaluation, target, matchers, threshold)
                energy_consistent = abs(float(energy-old_energy)) <= 1e-6
                numerical_cold = bool(energy_consistent and checked_target["physical_gate_passed"])
                target_cold = bool(numerical_cold and checked_target["identity_qualified"] and
                                   checked_target["energy_window_passed"])
                row.update(checked_target, energy_error_eV=float(energy-old_energy),
                           cold_energy_consistent=energy_consistent,
                           numerical_cold_qualified=numerical_cold,
                           target_cold_qualified=target_cold)
                _append_geometry(fresh_dir / f"{index:02d}-{role}.extxyz", atoms,
                                 {"fresh_role": role, **{k:v for k,v in checked_target.items()
                                  if isinstance(v, (str, bool, int, float))}})
            except Exception as error:
                row.update(status="error", error=repr(error))
            checked.append(row)
    except Exception as error:
        fresh_exception = repr(error)
    (fresh_dir / "requests.jsonl").touch(exist_ok=True)
    target_hash = None if first_candidate_eval is None else _geometry_key(first_candidate_eval.atoms)
    fresh_target = next((row for row in checked if row["geometry_sha256"] == target_hash), None)
    return {"fresh_checks": checked, "fresh_requests": fresh_budget.requests,
            "fresh_calculator_calls": fresh_budget.calculator_calls,
            "fresh_failures": fresh_budget.failures, "fresh_denials": fresh_budget.denials,
            "fresh_elapsed_seconds": time.monotonic() - fresh_started,
            "fresh_exception": fresh_exception,
            "first_target_cold_confirmed": bool(fresh_target and fresh_target.get("target_cold_qualified")),
            "first_target_discovery_cost": result_payload["first_target_paid_requests"]}


def shared_reference_check(plan, out, helper, calculator_class):
    """One independent cold E/F/stress check of the common qualified target."""
    import numpy as np
    from ase.io import read
    target_spec = plan["target"]
    atoms = read(target_spec["path"]); atoms.calc = None
    target_dir = out / "shared-reference"
    target_dir.mkdir(exist_ok=True)
    budget = helper.CaseBudget("shared-anatase-reference", target_dir / "requests.jsonl",
        int(plan["budget"]["shared_reference_requests"]), FRESH_SECONDS,
        time.monotonic() + FRESH_SECONDS)
    try:
        calc_spec = plan["calculator"]
        calculator = calculator_class(model_paths=plan["model"], head=calc_spec["head"],
            device=calc_spec["device"], default_dtype=calc_spec["dtype"],
            enable_cueq=calc_spec["enable_cueq"], enable_oeq=calc_spec["enable_oeq"])
        helper.instrument_calculate(calculator, budget)
        surface = helper.CountedStressSurface(calculator, budget, "shared_reference_cold")
        energy, forces, stress = surface.evaluate(atoms)
        row = {"status": "completed", "energy_eV": float(energy),
               "energy_error_eV": float(energy-target_spec["energy_eV"]),
               "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
               "stress_max_eV_A3": float(np.abs(stress).max()),
               "qualified": bool(abs(float(energy-target_spec["energy_eV"])) <= 1e-6 and
                    float(np.linalg.norm(forces, axis=1).max()) <= target_spec["fmax_eV_A"] and
                    float(np.abs(stress).max()) <= target_spec["stress_tol_eV_A3"])}
    except Exception as error:
        row = {"status": "error", "qualified": False, "error": repr(error)}
    (target_dir / "requests.jsonl").touch(exist_ok=True)
    row.update(requests=budget.requests, calculator_calls=budget.calculator_calls,
               failures=budget.failures, denials=budget.denials,
               elapsed_seconds=time.monotonic()-budget.started,
               source_path=target_spec["path"], source_sha256=target_spec["sha256"])
    dump(target_dir / "result.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--slot", type=int, choices=range(4))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    plan = plan_data()
    if args.preflight:
        print(json.dumps(preflight(plan), indent=2, allow_nan=False))
        return 0
    if args.output is None:
        parser.error("--slot requires --output PATH")
    validate_plan(plan)
    result = execute_slot(plan, args.slot, args.output.resolve())
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
