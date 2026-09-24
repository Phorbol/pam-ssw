#!/usr/bin/env python3
"""Explicit-only, one-outer-step bias_fmax sensitivity probe."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
LEDGER = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def load_json(path):
    return json.loads(Path(path).read_text())


def source_paths(spec):
    base = ROOT / "research/ga_ssw/evidence" / spec["source_case"]
    if spec["system"] == "C4H6":
        return base / spec["input"], ROOT / spec["effective_config"]
    return ROOT / spec["input"], ROOT / spec["effective_config"]


def preflight(plan, *, require_empty_output=True):
    if plan["status"] != "PREPARED_NOT_SUBMITTED":
        raise RuntimeError("prepared protocol status changed")
    if git("rev-parse", "HEAD:pamssw") != plan["core_tree"] or git("diff", "HEAD", "--", "pamssw"):
        raise RuntimeError("pamssw core tree changed since protocol preparation")
    if sha256(MODEL) != plan["model_sha256"]:
        raise RuntimeError("model hash mismatch")
    for name, spec in plan["cases"].items():
        input_path, effective_path = source_paths(spec)
        if sha256(input_path) != spec["input_sha256"]:
            raise RuntimeError(f"input hash mismatch: {name}")
        if sha256(effective_path) != spec["effective_config_sha256"]:
            raise RuntimeError(f"effective config hash mismatch: {name}")
        effective = load_json(effective_path)
        cfg = effective.get("config", effective.get("ssw_config"))
        if cfg is None or cfg.get("bias_fmax") != 0.1:
            raise RuntimeError(f"source bias_fmax is not the frozen 0.1: {name}")
        if cfg.get("fmax") != 0.03 or cfg.get("max_gaussians") != (25 if spec["system"] == "C4H6" else 12):
            raise RuntimeError(f"source full-depth/true-force settings mismatch: {name}")
    if require_empty_output and (HERE / "runs").exists():
        raise FileExistsError("runs/ already exists; no overwrite, resume, or retry")
    return {"head": git("rev-parse", "HEAD"), "core_tree": git("rev-parse", "HEAD:pamssw"),
            "plan_sha256": sha256(HERE / "plan.json"), "runner_sha256": sha256(__file__),
            "model_sha256": sha256(MODEL)}


def parse_pair_map(values):
    pairs = {}
    for key, value in values.items():
        parts = key.strip().strip("()").split(",")
        pairs[tuple(int(part.strip()) for part in parts)] = value
    return pairs


def make_settings(effective, bias_fmax):
    from pamssw.standalone import (LSPrequenchSettings, NativeLSSettings,
                                   NativeMCSettings, RecoveredRotationSettings,
                                   SSWConfig)

    c4 = "config" in effective
    cfg_data = dict(effective["config"] if c4 else effective["ssw_config"])
    cfg_data["bias_fmax"] = bias_fmax
    cfg = SSWConfig(**cfg_data)
    native = effective["ls_settings"] if c4 else effective["native_ls"]
    ls = NativeLSSettings(
        bond_energies=parse_pair_map(native["bond_energies"]),
        bond_lengths=parse_pair_map(native["bond_lengths"]),
        **{key: value for key, value in native.items()
           if key not in ("bond_energies", "bond_lengths", "prequench", "parameter_source")},
        prequench=LSPrequenchSettings(**native["prequench"]),
    )
    rotation_data = effective.get("rotation_settings", effective.get("recovered_rotation"))
    rotation = RecoveredRotationSettings(**rotation_data)
    mc_data = effective.get("native_mc")
    mc = None if mc_data is None else NativeMCSettings(
        energy_tol=mc_data["energy_tol_eV"], maxtrap=mc_data["maxtrap"])
    return cfg, ls, rotation, mc


def check_fresh(atoms, reference, system, calculator, surface, label, folder, fmax):
    import numpy as np
    from ase.io import write
    row = {"label": label, "status": "started"}
    write(folder / f"{label}.traj", atoms)
    try:
        calculator.reset()
        energy, forces = surface.evaluate(atoms)
        actual_fmax = float(np.linalg.norm(forces, axis=1).max())
        row.update(status="completed", energy_eV=energy, fmax_eV_A=actual_fmax,
                   force_qualified=bool(actual_fmax <= fmax),
                   same_numbers=bool(np.array_equal(atoms.numbers, reference.numbers)),
                   same_cell=bool(np.array_equal(atoms.cell.array, reference.cell.array)),
                   same_pbc=bool(np.array_equal(atoms.pbc, reference.pbc)))
    except Exception as error:
        row.update(status="error", error=repr(error), force_qualified=False)
    return row


def make_calculators(plan, system, ledger):
    import torch
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    torch.manual_seed(plan["runtime"]["torch_manual_seed"])
    if system == "C60":
        torch.use_deterministic_algorithms(plan["runtime"]["torch_deterministic_algorithms"])
        torch.backends.cuda.matmul.allow_tf32 = plan["runtime"]["tf32"]
        torch.backends.cudnn.allow_tf32 = plan["runtime"]["tf32"]
    kwargs = dict(model_paths=str(MODEL), head=plan["head"], device=plan["device"],
                  default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False)
    calculator, fresh_calculator = MACECalculator(**kwargs), MACECalculator(**kwargs)
    counter = ledger.instrument_calculate(calculator)
    fresh_counter = ledger.instrument_calculate(fresh_calculator)
    return calculator, fresh_calculator, counter, fresh_counter


def run_arm(plan, name, spec, bias_fmax, deadline, ledger, calculators):
    import numpy as np
    import torch
    from ase.io import read, write
    from pamssw.standalone import run_ssw

    folder = HERE / "runs" / f"{name}-bias{int(round(100*bias_fmax)):03d}"
    folder.mkdir(parents=True, exist_ok=False)
    input_path, effective_path = source_paths(spec)
    effective = load_json(effective_path)
    original = read(input_path)
    write(folder / "input.traj", original)
    config, ls, rotation, mc = make_settings(effective, bias_fmax)
    effective_row = {"case": name, "seed": spec["seed"], "bias_fmax": bias_fmax,
                     "config": config.__dict__, "native_ls": ls.__dict__,
                     "rotation_settings": rotation.__dict__,
                     "native_mc": None if mc is None else mc.__dict__,
                     "source_effective_config": str(effective_path),
                     "input": str(input_path)}
    ledger.dump(folder / "effective-config.json", effective_row)

    calculator, fresh_calculator, counter, fresh_counter = calculators
    calls_before, fresh_calls_before = counter["calls"], fresh_counter["calls"]
    calculator.reset()
    started = time.monotonic()
    surface = ledger.CountedSurface(calculator, folder / "requests.jsonl",
                                    plan["search_cap_per_arm"], max(0.0, deadline - started))
    fresh_surface = ledger.CountedSurface(fresh_calculator, folder / "fresh-requests.jsonl",
                                          plan["fresh_cap_per_arm"], max(0.0, deadline - started))
    row = {"case": name, "system": spec["system"], "seed": spec["seed"],
           "bias_fmax": bias_fmax, "status": "started", "outer_steps_requested": 1,
           "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                       "torch_seed": plan["runtime"]["torch_manual_seed"],
                       "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
           "provenance": {"git_head": git("rev-parse", "HEAD"),
                          "core_tree": git("rev-parse", "HEAD:pamssw"),
                          "input_sha256": sha256(input_path),
                          "effective_config_sha256": sha256(effective_path)}}
    result = None
    try:
        result = run_ssw(original.copy(), surface, steps=1, config=config,
                         rng=np.random.default_rng(spec["seed"]), ls=ls,
                         recovered_rotation=rotation, mc=mc,
                         checkpoint_path=folder / "checkpoint.pkl")
        ledger.dump(folder / "result.json", result)
        row.update(status=result.status, result_records=len(result.records),
                   record_statuses=[record.status for record in result.records],
                   result_evaluation_requests=result.evaluation_requests,
                   request_count_matches_result=(result.evaluation_requests == surface.requests))
    except Exception as error:
        row.update(status="exception", error=repr(error))
        if (folder / "checkpoint.pkl").is_file():
            row["checkpoint_retained"] = True

    initial = None if result is None else getattr(result.initial, "atoms", None)
    record = None if result is None or not result.records else result.records[0]
    landing_result = None if record is None else record.landing
    landing = None if landing_result is None else getattr(landing_result, "atoms", None)
    checks = []
    if initial is not None and time.monotonic() < deadline:
        checks.append(check_fresh(initial, original, spec["system"], fresh_calculator,
                                  fresh_surface, "post-initial-quench", folder, config.fmax))
    if landing is not None and time.monotonic() < deadline and fresh_surface.requests < plan["fresh_cap_per_arm"]:
        checks.append(check_fresh(landing, initial if initial is not None else original,
                                  spec["system"], fresh_calculator, fresh_surface,
                                  "outer-landing", folder, config.fmax))
    if initial is not None:
        write(folder / "post-initial-quench.traj", initial)
    if landing is not None:
        write(folder / "outer-landing.traj", landing)
    row.update(search_requests=surface.requests, search_calculator_calls=counter["calls"] - calls_before,
               search_denials=surface.denials, search_boundary=surface.boundary,
               fresh_checks=checks, fresh_requests=fresh_surface.requests,
               fresh_calculator_calls=fresh_counter["calls"] - fresh_calls_before,
               completed_gaussian_stages=(None if record is None else len(record.climb)),
               biased_stage_requests=([] if record is None else
                   [stage["requests"] for stage in record.climb]),
               initial_quench_requests=(None if result is None else result.initial.evaluation_requests),
               landing_quench_requests=(None if landing_result is None else landing_result.evaluation_requests),
               landing_reported_converged=(None if landing_result is None else landing_result.converged),
               landing_reported_fmax_eV_A=(None if landing_result is None else landing_result.max_force),
               elapsed_seconds=time.monotonic() - started,
               budget_censored=surface.boundary in ("request_cap", "wall_cap"))
    row["landing_fresh_qualified"] = bool(
        landing is not None and landing_result.converged and len(checks) >= 2
        and checks[-1].get("label") == "outer-landing"
        and checks[-1].get("force_qualified") and checks[-1].get("same_numbers")
        and checks[-1].get("same_cell") and checks[-1].get("same_pbc"))
    row["total_requests_including_fresh"] = row["search_requests"] + row["fresh_requests"]
    ledger.dump(folder / "summary.json", row)
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.execute and args.check:
        parser.error("choose --execute or --check")
    plan = load_json(HERE / "plan.json")
    if not args.execute:
        if not args.check:
            print("prepared_not_executed")
            return
        preflight(plan, require_empty_output=False)
        from dataclasses import asdict
        from ase.io import read
        for name, spec in plan["cases"].items():
            input_path, effective_path = source_paths(spec)
            structures = []
            settings = []
            for value in plan["arms"]:
                structures.append(read(input_path))
                settings.append(make_settings(load_json(effective_path), value))
            if (structures[0].get_chemical_formula() != structures[1].get_chemical_formula()
                    or structures[0].get_positions().tolist() != structures[1].get_positions().tolist()
                    or structures[0].get_cell().tolist() != structures[1].get_cell().tolist()
                    or structures[0].get_pbc().tolist() != structures[1].get_pbc().tolist()):
                raise RuntimeError(f"paired starting input mismatch: {name}")
            left, right = settings
            if {**asdict(left[0]), "bias_fmax": 0.2} != asdict(right[0]):
                raise RuntimeError(f"paired config differs beyond bias_fmax: {name}")
            if asdict(left[1]) != asdict(right[1]) or asdict(left[2]) != asdict(right[2]) or left[3] != right[3]:
                raise RuntimeError(f"paired LS, rotation, or MC settings differ: {name}")
        print("check_passed: 8 arm configs and inputs checked; paired configs differ only by bias_fmax; steps=1 is one additional outer attempt; no MACE/PES call")
        return
    provenance = preflight(plan)
    (HERE / "runs").mkdir()
    ledger_spec = __import__("importlib.util").util.spec_from_file_location("bias_probe_ledger", LEDGER)
    ledger = __import__("importlib.util").util.module_from_spec(ledger_spec)
    ledger_spec.loader.exec_module(ledger)
    started = time.monotonic()
    deadline = started + plan["resource_ceiling"]["cooperative_seconds"]
    ledger.dump(HERE / "execution.json", {"status": "started", **provenance,
                                           "plan_sha256": sha256(HERE / "plan.json"),
                                           "started_unix": time.time()})
    rows = []
    calculators = {}
    for name, spec in plan["cases"].items():
        if spec["system"] not in calculators and time.monotonic() < deadline:
            calculators[spec["system"]] = make_calculators(plan, spec["system"], ledger)
        for value in plan["arms"]:
            if time.monotonic() >= deadline:
                rows.append({"case": name, "bias_fmax": value, "status": "not_started_cooperative_deadline"})
                continue
            try:
                rows.append(run_arm(plan, name, spec, value, deadline, ledger,
                                    calculators[spec["system"]]))
            except Exception as error:
                folder = HERE / "runs" / f"{name}-bias{int(round(100*value)):03d}"
                rows.append({"case": name, "bias_fmax": value, "status": "runner_exception",
                             "error": repr(error),
                             "search_requests": ledger_charge(folder / "requests.jsonl"),
                             "fresh_requests": ledger_charge(folder / "fresh-requests.jsonl"),
                             "total_requests_including_fresh": (
                                 ledger_charge(folder / "requests.jsonl") +
                                 ledger_charge(folder / "fresh-requests.jsonl"))})
            ledger.dump(HERE / "runs-summary.json", {"status": "running", "rows": rows,
                                                       "elapsed_seconds": time.monotonic() - started})
    ledger.dump(HERE / "runs-summary.json", {"status": "complete" if all(
        row.get("status") in ("completed", "failed", "exception") for row in rows) else "censored",
        "rows": rows, "elapsed_seconds": time.monotonic() - started,
        "total_search_requests": sum(row.get("search_requests", 0) for row in rows),
        "total_fresh_requests": sum(row.get("fresh_requests", 0) for row in rows)})


def ledger_charge(path):
    if not path.is_file():
        return 0
    count = 0
    with path.open() as stream:
        for line in stream:
            try:
                if json.loads(line).get("kind") in ("search", "search_failure"):
                    count += 1
            except (ValueError, TypeError):
                continue
    return count


if __name__ == "__main__":
    main()
