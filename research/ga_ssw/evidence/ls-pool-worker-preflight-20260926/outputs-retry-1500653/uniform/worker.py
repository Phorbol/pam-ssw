#!/usr/bin/env python3
"""Single-arm Native-LS pool-routing research worker.

Run with PYTHONPATH pointing at the separately qualified mainline checkout.
This script does not provide restart/resume orchestration for production arms.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import shutil
import time
import tempfile
import traceback
import sys

import numpy as np


SEARCH_CAP = 80_000
FRESH_CAP = 101
SEARCH_WALL = 1700.0
PROCESS_WALL = 1800.0


def serial(value):
    from ase import Atoms
    if isinstance(value, Atoms):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist(),
                "constraints": [serial(c.todict()) for c in value.constraints]}
    if is_dataclass(value):
        return {k: serial(v) for k, v in vars(value).items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def append_jsonl(path: Path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(serial(value), allow_nan=False) + "\n")


def git(*args):
    return subprocess.run(["git", *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def key_pairs(mapping):
    result = {}
    for key, value in mapping.items():
        pair = ast.literal_eval(key if key.startswith("(") else f"({key})")
        if not isinstance(pair, tuple):
            pair = (pair, pair)
        result[tuple(int(x) for x in pair)] = float(value)
    return result


def package_imports():
    from ase.io import read
    from pamssw.standalone.paper_reference import SSWConfig, run_ssw
    from pamssw.standalone.surface import ASESurface
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.native_mc import NativeMCSettings
    return locals()


def mainline_root():
    configured = os.environ.get("PAMSSW_MAINLINE")
    if configured:
        root = Path(configured).resolve()
    else:
        roots = [Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        if not roots:
            raise RuntimeError("PYTHONPATH must identify the qualified PAM-SSW mainline")
        root = roots[0]
    if not (root / "pamssw" / "standalone" / "paper_reference.py").is_file():
        raise RuntimeError(f"not a PAM-SSW checkout: {root}")
    sys.path.insert(0, str(root))
    return root


class CappedSurface:
    """Count paid surface requests, retaining starts, denials and exceptions."""
    def __init__(self, base, *, cap, deadline, ledger, stage):
        self.base = base
        self.cap = int(cap)
        self.deadline = float(deadline)
        self.ledger = Path(ledger)
        self.stage = stage
        self.started = time.monotonic()
        self.requests = 0
        self.attempts = 0
        self.censored = False
        self.censor_reason = None

    def evaluate(self, atoms):
        self.attempts += 1
        attempt = self.attempts
        started = time.monotonic()
        before = self.requests
        append_jsonl(self.ledger, {"event": "attempt_started", "stage": self.stage,
                                   "attempt": attempt, "request_candidate": before + 1,
                                   "elapsed_seconds": started - self.started})
        reason = None
        if before >= self.cap:
            reason = "request_cap"
        elif started >= self.deadline:
            reason = "wall_deadline"
        if reason:
            self.censored = True
            self.censor_reason = self.censor_reason or reason
            append_jsonl(self.ledger, {"event": "budget_censor", "stage": self.stage,
                "attempt": attempt, "request": None, "charged": False, "reason": reason})
            raise RuntimeError(f"ls_pool_budget_{self.stage}_{reason}")
        try:
            energy, forces = self.base.evaluate(atoms)
            self.requests = self.base.requests
        except Exception as exc:
            self.requests = self.base.requests
            append_jsonl(self.ledger, {"event": "attempt_error", "stage": self.stage,
                "attempt": attempt, "request": self.requests if self.requests > before else None,
                "charged": self.requests > before, "error": f"{type(exc).__name__}: {exc}"})
            raise
        append_jsonl(self.ledger, {"event": "attempt_completed", "stage": self.stage,
            "attempt": attempt, "request": self.requests, "charged": self.requests > before,
            "energy_eV": energy, "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
            "elapsed_seconds": time.monotonic() - started})
        return energy, forces


def make_calculator(spec, *, device_override=None):
    if spec.get("kind") == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    from mace.calculators import MACECalculator
    model = Path(spec["path"])
    if not model.is_file():
        raise FileNotFoundError(model)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if spec.get("sha256") and digest != spec["sha256"]:
        raise ValueError(f"model SHA256 mismatch: {digest}")
    return MACECalculator(model_paths=str(model), device=device_override or spec["device"],
                          default_dtype=spec["dtype"], head=spec["head"],
                          enable_cueq=bool(spec["enable_cueq"]),
                          enable_oeq=bool(spec["enable_oeq"]))


def mc_archive_summary(result, *, energy_tol, rmsd_tol):
    from pamssw.state import State
    from research.ga_ssw.pool_molecular_archive import ASEPermutationArchive
    archive = ASEPermutationArchive(energy_tol=energy_tol, rmsd_tol=rmsd_tol)
    mapping = []
    for minimum in result.minima:
        atoms = minimum.atoms
        state = State(atoms.numbers.copy(), atoms.positions.copy(), atoms.cell.array.copy(),
                      tuple(atoms.pbc))
        entry = archive.add(state, float(minimum.energy), None)
        mapping.append(entry.entry_id)
    return {"identity_matcher": "ase_permute_v1", "runtime_selector": False,
            "energy_tol_eV": energy_tol, "rmsd_tol_A": rmsd_tol,
            "mapping_minimum_index_to_entry": mapping,
            "entries": [{"entry_id": e.entry_id, "energy_eV": e.energy,
                         "visits": e.visits, "parent_id": e.parent_id}
                        for e in archive.entries]}


def candidate_rows(result):
    rows = [{"candidate_index": 0, "source": "initial", "record_index": -1,
             "minimum_index": 0, "converged": bool(result.initial.converged),
             "accepted": True, "evaluation": result.initial}]
    minimum_index = 1
    for record_index, record in enumerate(result.records):
        if record.landing is None:
            continue
        converged = bool(record.landing.converged)
        rows.append({"candidate_index": len(rows), "source": "landing",
                     "record_index": record_index,
                     "minimum_index": minimum_index if converged else None,
                     "converged": converged, "accepted": bool(record.accepted),
                     "evaluation": record.landing})
        if converged:
            minimum_index += 1
    if minimum_index != len(result.minima):
        raise ValueError("core minima do not map to initial plus converged landing records")
    return rows


def run_arm(args):
    # All project imports occur after the caller's PYTHONPATH has been checked.
    expected_root = mainline_root()
    api = package_imports()
    from ase.io import read, write
    from pamssw.standalone.surface import ASESurface
    from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter

    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    if args.case_index < 0 or args.case_index >= len(plan["cases"]):
        raise ValueError("case-index outside plan")
    case = plan["cases"][args.case_index]
    if args.mode not in plan["methods"]:
        raise ValueError("mode is not listed in plan methods")
    budget = plan["budget"]
    limits = (int(budget["search_requests_per_arm"]), int(budget["fresh_requests_per_arm"]))
    if limits != (SEARCH_CAP, FRESH_CAP):
        raise ValueError("plan caps differ from frozen worker limits")
    if int(budget["wall_seconds_per_arm"]) != 1700 or int(budget["process_seconds_per_arm"]) != 1800:
        raise ValueError("plan wall limits differ from frozen worker limits")
    if int(plan["steps"]) > 100:
        raise ValueError("step count exceeds protocol")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    input_path = Path(case["input"]).resolve()
    initial = read(input_path)
    plan["cases"][args.case_index]["input"] = str(out / "initial.extxyz")
    write_json(out / "effective-plan.json", plan)
    module_path = Path(api["run_ssw"].__code__.co_filename).resolve()
    if not module_path.is_relative_to(expected_root):
        raise RuntimeError(f"loaded PAM-SSW source is outside expected mainline: {module_path}")
    model = dict(plan["model"])
    config_data = dict(case["config"])
    cfg = api["SSWConfig"](**config_data)
    lsdata = dict(case["native_ls"])
    preq = lsdata.get("prequench")
    lsdata["bond_energies"] = key_pairs(lsdata["bond_energies"])
    lsdata["bond_lengths"] = key_pairs(lsdata["bond_lengths"])
    if preq is not None:
        lsdata["prequench"] = api["LSPrequenchSettings"](**preq)
    ls = api["NativeLSSettings"](**lsdata)
    rotation = (api["RecoveredRotationSettings"](**case["recovered_rotation"])
                if case.get("recovered_rotation") else None)
    direction = (api["RecoveredDirectionSettings"](**case["recovered_direction"])
                 if case.get("recovered_direction") else None)
    mcdata = case.get("native_mc")
    mc = (api["NativeMCSettings"](energy_tol=float(mcdata["energy_tol_eV"]),
                                   maxtrap=int(mcdata["maxtrap"])) if mcdata else None)
    pool_cfg = plan["pool"]
    adapter = None
    if args.mode != "mc":
        adapter = PoolStarterAdapter(mode=args.mode,
            energy_tol=float(pool_cfg["energy_tol"]), rmsd_tol=float(pool_cfg["rmsd_tol"]),
            identity_matcher=pool_cfg["identity_matcher"])

    write(out / "initial.extxyz", initial)
    run_started = time.monotonic()
    search_deadline = run_started + SEARCH_WALL
    search_ledger = out / "search-ledger.jsonl"
    search_ledger.touch(exist_ok=False)
    progress_path = out / "progress.jsonl"
    progress_path.touch(exist_ok=False)

    worker_copy = out / "worker.py"
    shutil.copy2(Path(__file__).resolve(), worker_copy)
    worker_repo = Path(__file__).resolve().parents[2]
    actual_config = {"case": case["name"], "mode": args.mode, "steps": int(plan["steps"]),
        "seed": int(plan["seed"]), "selector_seed": int(plan["selector_seed"]) if adapter else None,
        "config": config_data, "native_ls": serial(ls), "mc": serial(mc),
        "recovered_rotation": serial(rotation), "recovered_direction": serial(direction),
        "model": model, "plan": str(plan_path), "protocol": plan.get("protocol"),
        "source_plan": str(getattr(args, "source_plan", plan_path)),
        "effective_plan": "effective-plan.json",
        "search_cap": SEARCH_CAP, "fresh_cap": FRESH_CAP,
        "search_wall_seconds": SEARCH_WALL, "process_wall_seconds": PROCESS_WALL,
        "loaded_module": str(module_path), "mainline_root": str(expected_root),
        "root_git_head": git("-C", str(expected_root), "rev-parse", "HEAD"),
        "root_git_status": git("-C", str(expected_root), "status", "--short"),
        "worker_git_head": git("-C", str(worker_repo), "rev-parse", "HEAD"),
        "worker_git_status": git("-C", str(worker_repo), "status", "--short"),
        "worker_copy": str(worker_copy),
        "worker_sha256": hashlib.sha256(worker_copy.read_bytes()).hexdigest()}
    write_json(out / "actual-config.json", actual_config)

    def progress(event):
        step = event.step
        append_jsonl(progress_path, {"kind": event.kind, "next_index": event.next_index,
            "evaluation_requests": event.evaluation_requests,
            "current_energy_eV": event.current_energy,
            "best_energy_eV": event.best.energy,
            "step_status": None if step is None else step.status,
            "accepted": None if step is None else step.accepted,
            "has_landing": bool(step is not None and step.landing is not None),
            "starter_selection": None if step is None else step.starter_selection})
        return False

    selector_rng = np.random.default_rng(int(plan["selector_seed"])) if adapter else None
    if adapter:
        write_json(out / "selector-contract.json", {
            "contract": adapter.checkpoint_contract(), "mode": args.mode,
            "selector_seed": int(plan["selector_seed"]),
            "runtime_rng_separate_from_ssw_rng": True,
            "identity_matcher": pool_cfg["identity_matcher"],
            "energy_tol_eV": pool_cfg["energy_tol"], "rmsd_tol_A": pool_cfg["rmsd_tol"]})
    else:
        write_json(out / "selector-contract.json", {"mode": "mc", "runtime_selector": False,
            "offline_identity_only": "ASE permutation archive applied after the run"})

    status, error = "not_started", None
    result = None
    search_surface = CappedSurface(ASESurface(make_calculator(model)), cap=SEARCH_CAP,
                                   deadline=search_deadline, ledger=search_ledger, stage="search")
    try:
        result = api["run_ssw"](initial.copy(), search_surface,
            steps=int(plan["steps"]), config=cfg, rng=np.random.default_rng(int(plan["seed"])),
            ls=ls, mc=mc, recovered_rotation=rotation, recovered_direction=direction,
            starter_selector=adapter, selector_rng=selector_rng,
            progress_callback=progress)
        status = result.status
        write_json(out / "search-result.json", {"status": status,
            "search_requests": search_surface.requests, "adapter_calls": None,
            "result": result})
    except Exception as exc:
        status = "exception"
        error = f"{type(exc).__name__}: {exc}"
        (out / "search-error.txt").write_text(traceback.format_exc())
    search_requests = search_surface.requests
    if result is not None and adapter is not None:
        try:
            write_json(out / "pool-report.json", adapter.finalize(result))
        except Exception as exc:
            error = error or f"pool_finalize:{type(exc).__name__}: {exc}"
            status = "analysis_error"
            (out / "pool-finalize-error.txt").write_text(traceback.format_exc())

    fresh_checks = []
    fresh_requests = 0
    fresh_errors = []
    fresh_surface = None
    fresh_candidate_total = None
    if result is not None:
        rows = candidate_rows(result)
        fresh_candidate_total = len(rows)
        if len(rows) > FRESH_CAP:
            error = error or "fresh candidate count exceeds cap"
            status = "fresh_cap_violation"
            raise ValueError("initial plus returned landings exceed protocol fresh cap")
        fresh_deadline = run_started + PROCESS_WALL
        fresh_ledger = out / "fresh-ledger.jsonl"
        fresh_ledger.touch(exist_ok=False)
        fresh_surface = CappedSurface(ASESurface(make_calculator(model)), cap=FRESH_CAP,
                                      deadline=fresh_deadline, ledger=fresh_ledger, stage="fresh")
        for row in rows:
            fresh_censored = False
            evaluation = row["evaluation"]
            check = {k: row[k] for k in ("candidate_index", "source", "record_index",
                                         "minimum_index", "converged", "accepted")}
            check["atoms"] = evaluation.atoms
            check["algorithm_energy_eV"] = float(evaluation.energy)
            check["algorithm_max_force_eV_A"] = float(evaluation.max_force)
            try:
                fresh_surface.base.calculator.reset()
                energy, forces = fresh_surface.evaluate(evaluation.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                check.update(status="fresh_completed", fresh_requests=fresh_surface.requests,
                    fresh_energy_eV=float(energy), fresh_fmax_eV_A=fmax,
                    energy_delta_eV=float(energy - evaluation.energy),
                    force_qualified=bool(fmax <= cfg.fmax),
                    certified=bool(fmax <= cfg.fmax and math.isfinite(energy)))
            except Exception as exc:
                check.update(status="fresh_error", fresh_requests=fresh_surface.requests,
                             error=f"{type(exc).__name__}: {exc}")
                fresh_errors.append(check["error"])
                if fresh_surface.censored:
                    status = "fresh_censored"
                    error = error or fresh_surface.censor_reason
                    fresh_censored = True
            fresh_checks.append(check)
            write_json(out / "fresh-checks.json", {"checks": fresh_checks,
                "candidate_count": len(rows), "qualified_count": sum(bool(x.get("certified")) for x in fresh_checks),
                "fresh_requests": fresh_surface.requests})
            if fresh_censored:
                break
        fresh_requests = fresh_surface.requests
        write_json(out / "fresh-checks.json", {"checks": fresh_checks,
            "candidate_count": len(rows), "qualified_count": sum(bool(x.get("certified")) for x in fresh_checks),
            "fresh_requests": fresh_requests, "errors": fresh_errors})
        if args.mode == "mc":
            write_json(out / "offline-identity.json", mc_archive_summary(result,
                energy_tol=float(pool_cfg["energy_tol"]), rmsd_tol=float(pool_cfg["rmsd_tol"])))
        write(out / "minima.extxyz", [minimum.atoms for minimum in result.minima])
        write_json(out / "search-result.json", {"status": result.status,
            "postprocess_status": status, "search_requests": search_requests,
            "adapter_calls": len(adapter.decisions) if adapter else None,
            "result": result})
    elif not (out / "fresh-checks.json").exists():
        write_json(out / "fresh-checks.json", {"checks": [], "candidate_count": None,
            "qualified_count": 0, "fresh_requests": fresh_requests,
            "note": "search produced no serializable result; fresh cost is not implied to be zero"})

    summary = {"case": case["name"], "mode": args.mode, "seed": int(plan["seed"]),
        "status": status, "algorithm_status": None if result is None else result.status,
        "budget_censor": bool(search_surface.censored or (fresh_surface and fresh_surface.censored)),
        "censor_reason": search_surface.censor_reason or
                         (fresh_surface.censor_reason if fresh_surface else None),
        "search_budget_censor": bool(search_surface.censored),
        "fresh_budget_censor": bool(fresh_surface and fresh_surface.censored),
        "search_requests": search_requests, "fresh_requests": fresh_requests,
        "total_requests": search_requests + fresh_requests,
        "outer_attempt_records": None if result is None else len(result.records),
        "returned_landings": None if result is None else sum(r.landing is not None for r in result.records),
        "successful_minima_in_core_result": None if result is None else len(result.minima),
        "fresh_candidate_count": fresh_candidate_total,
        "fresh_checks_recorded": len(fresh_checks),
        "fresh_qualified_count": sum(bool(x.get("certified")) for x in fresh_checks),
        "adapter_call_count": None if adapter is None else len(adapter.decisions),
        "wall_seconds": time.monotonic() - run_started, "error": error,
        "loaded_module": str(module_path), "actual_config": "actual-config.json",
        "search_ledger": "search-ledger.jsonl",
        "fresh_ledger": "fresh-ledger.jsonl" if result is not None else None}
    write_json(out / "summary.json", summary)
    return 0 if result is not None else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--mode", choices=("mc", "uniform", "pam"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--emt-preflight", action="store_true")
    args = parser.parse_args()
    if args.plan is None or args.mode is None:
        parser.error("--plan and --mode are required")
    if args.emt_preflight:
        return run_emt_preflight(args)
    return run_arm(args)


def run_emt_preflight(args):
    """Run the ordinary worker path on a two-step relaxed Cu4/EMT plan."""
    from ase import Atoms
    from ase.io import write
    original_plan = args.plan.resolve()
    plan = json.loads(original_plan.read_text())
    case = dict(plan["cases"][args.case_index])
    with tempfile.TemporaryDirectory(prefix="ls-pool-emt-") as temp:
        temp_path = Path(temp)
        atoms = Atoms("Cu4", positions=[[0, 0, 0], [2.56, 0, 0],
                                         [1.28, 2.217, 0], [1.28, .739, 2.091]])
        input_path = temp_path / "cu4.extxyz"
        write(input_path, atoms)
        case.update(input=str(input_path), config={"width": .1, "rotation_bias": 100.,
            "max_gaussians": 1, "temperature_K": 150., "fmax": .05, "relax_steps": 200,
            "fd_step": .001, "rotation_hvp": 4, "rotation_tol": .2,
            "forward_force": .1, "direction_sampling": "global", "rotation_solver": "dimer",
            "cluster_frame": "direction_only", "quench_optimizer": "safe-lbfgs-total",
            "lbfgs_memory": None, "bias_stage_steps": None, "bias_fmax": None,
            "pre_rotation_hvp": None, "rotation_exit_policy": "force"},
            native_ls={"bond_energies": {"(29, 29)": 3.6298000812530518},
                "bond_lengths": {"(29, 29)": 1.875}, "scale": 5., "amp_c": 2.,
                "length_tolerance": .1, "target_mev_per_atom": 20., "eta": .005,
                "max_change": .01, "frequency": 10, "presteps": 100, "cycle": 100,
                "ratio": 1.100000023841858, "lselfadapt": True,
                "bond_geometry": "native-mic",
                "prequench": {"fmax": .1, "steps": 50,
                              "exit_policy": "force_or_step_limit"}},
            recovered_rotation=None, recovered_direction=None, native_mc=None)
        plan["cases"] = [case]
        plan["methods"] = [args.mode]
        plan["steps"] = 2
        plan["seed"] = 181
        plan["selector_seed"] = 193
        plan["model"] = {"kind": "emt"}
        temp_plan = temp_path / "emt-plan.json"
        temp_plan.write_text(json.dumps(plan))
        args.source_plan = original_plan
        args.plan = temp_plan
        args.case_index = 0
        rc = run_arm(args)
    if rc == 0:
        write_json(args.out.resolve() / "preflight.json", {
            "status": "completed", "mode": args.mode, "steps": 2,
            "backend": "ASE EMT", "path": "same run_arm, ledgers, fresh checks, result serializer",
            "scientific_scope": "API/accounting smoke only; no MH1 or search-quality claim"})
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
