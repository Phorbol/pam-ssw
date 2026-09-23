"""Bounded MH-1 C4H6 SSW/LS lifecycle and request-cost pilot.

Execution is opt-in. This single-seed pilot does not qualify search efficacy or
reactive accuracy; all minima are observations, not Hessian-certified states.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
QUAL = HERE.parent / "c4h6-model-qualification-20260924-r2"
PLAN_PATH = HERE / "plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def preflight(plan):
    if plan.get("status") != "ready_for_pilot":
        raise RuntimeError("pilot plan must be ready_for_pilot")
    qualification = json.loads((QUAL / "qualification.json").read_text())
    if (qualification.get("status") != "complete" or not qualification.get("all_qualified")
            or len(qualification.get("rows", [])) != 3
            or not all(row.get("qualified") for row in qualification["rows"])):
        raise RuntimeError("three-case MH-1 qualification is not complete and qualified")
    qplan = json.loads((QUAL / "plan.json").read_text())
    for key in ("model", "model_sha256", "head", "device", "dtype", "source_commit", "source_core_tree",
                "shared_ledger", "shared_ledger_sha256"):
        if plan.get(key) != qplan.get(key):
            raise RuntimeError(f"pilot/qualification provenance mismatch: {key}")
    if plan.get("config") != qplan.get("config"):
        raise RuntimeError("pilot SSW config differs from qualified frozen config")
    if sha256(plan["model"]) != plan["model_sha256"]:
        raise RuntimeError("model SHA256 mismatch")
    ledger_path = (HERE / plan["shared_ledger"]).resolve()
    if sha256(ledger_path) != plan["shared_ledger_sha256"]:
        raise RuntimeError("shared ledger SHA256 mismatch")
    if git("rev-parse", "HEAD:pamssw") != plan["source_core_tree"]:
        raise RuntimeError("frozen core tree mismatch")
    if git("diff", "HEAD", "--", "pamssw"):
        raise RuntimeError("tracked pamssw core is dirty")
    if plan.get("arms") != ["ssw", "paper_ls", "native_ls"] or plan.get("seed") != 59:
        raise RuntimeError("pilot arms/seed differ from frozen protocol")
    if (plan.get("steps") != 12 or plan.get("request_cap_per_arm") != 12000
            or plan.get("fresh_cap_per_arm") != 13 or plan.get("wall_seconds_per_arm") != 840):
        raise RuntimeError("pilot budget differs from frozen protocol")
    if (plan.get("head"), plan.get("device"), plan.get("dtype")) != ("omol", "cuda", "float64"):
        raise RuntimeError("unexpected model execution settings")
    input_path = (HERE / plan["input"]).resolve()
    if input_path != (QUAL / "butadiene" / "input.extxyz").resolve():
        raise RuntimeError("pilot input differs from qualified butadiene input")
    if not input_path.is_file() or sha256(input_path) != plan.get("input_sha256"):
        raise RuntimeError("qualified input is missing or changed")
    if (plan.get("recovered_rotation") != dict(pre_rotmax=5, rotmax=15, pre_ftol=.2,
            ftol=.02, metric="euclidean", max_force_calls=40)
            or plan.get("prequench") != dict(fmax=.1, steps=50, exit_policy="force_or_step_limit")
            or plan.get("paper_target_eV_per_atom") != .7
            or plan.get("native_target_meV_per_atom") != 700):
        raise RuntimeError("pilot rotation/LS settings differ from frozen protocol")
    if (any((HERE / arm).exists() for arm in plan["arms"])
            or (HERE / "summary.json").exists() or (HERE / "effective_config.json").exists()):
        raise FileExistsError("pilot output already exists; refusing overwrite or implicit retry")
    return {"git_head": git("rev-parse", "HEAD"), "core_tree": plan["source_core_tree"],
            "model_sha256": plan["model_sha256"], "ledger_sha256": plan["shared_ledger_sha256"],
            "plan_sha256": sha256(PLAN_PATH), "qualification_sha256": sha256(QUAL / "qualification.json"),
            "input_sha256": sha256(input_path), "ledger_path": str(ledger_path)}


def execute(plan, provenance):
    import numpy as np
    import torch
    from ase.io import read
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    torch.manual_seed(0)
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone import SSWConfig, LSSettings, NativeLSSettings, run_ssw
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    from pamssw.standalone.paper_reference import load_ssw_checkpoint

    spec = importlib.util.spec_from_file_location("pilot_shared_ledger", provenance["ledger_path"])
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    cfg = SSWConfig(**plan["config"])
    rotation = RecoveredRotationSettings(**plan["recovered_rotation"])
    prequench = LSPrequenchSettings(**plan["prequench"])
    ls_by_arm = {
        "ssw": None,
        "paper_ls": LSSettings(HC_BOND_ENERGIES, {k: v + .1 for k, v in HC_BOND_LENGTHS.items()},
                               target_per_atom=plan["paper_target_eV_per_atom"], prequench=prequench),
        "native_ls": NativeLSSettings(HC_BOND_ENERGIES, HC_BOND_LENGTHS,
                                      target_mev_per_atom=plan["native_target_meV_per_atom"], prequench=prequench),
    }
    effective = {"config": asdict(cfg), "rotation_settings": asdict(rotation),
                 "ls_settings": {arm: None if ls is None else asdict(ls) for arm, ls in ls_by_arm.items()},
                 "seed": plan["seed"], "outer_attempts": plan["steps"],
                 "request_cap_per_arm": plan["request_cap_per_arm"],
                 "fresh_cap_per_arm": plan["fresh_cap_per_arm"],
                 "wall_seconds_per_arm": plan["wall_seconds_per_arm"]}
    ledger.dump(HERE / "effective_config.json", effective)
    atoms = read(QUAL / "butadiene" / "input.extxyz")
    pilot_started = time.monotonic()
    args = dict(model_paths=plan["model"], head=plan["head"], device=plan["device"],
                default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False)
    search_calc, fresh_calc = MACECalculator(**args), MACECalculator(**args)
    search_calls, fresh_calls = ledger.instrument_calculate(search_calc), ledger.instrument_calculate(fresh_calc)
    rows = []
    for arm in plan["arms"]:
        folder = HERE / arm
        folder.mkdir(exist_ok=False)
        checkpoint_path = folder / "checkpoint.pkl"
        row = {"arm": arm, "seed": plan["seed"], "status": "started",
               "search_requests": 0, "fresh_requests": 0, "minima_total": None}
        surface = fresh_surface = result = None
        before_search, before_fresh = search_calls["calls"], fresh_calls["calls"]
        started = time.monotonic()
        try:
            search_calc.reset()
            remaining = 2700 - (time.monotonic() - pilot_started)
            if remaining <= 0:
                raise RuntimeError("pilot_wall_cap")
            surface = ledger.CountedSurface(search_calc, folder / "requests.jsonl",
                plan["request_cap_per_arm"], min(plan["wall_seconds_per_arm"], remaining))
            result = run_ssw(atoms.copy(), surface, steps=plan["steps"], config=cfg,
                rng=np.random.default_rng(plan["seed"]), ls=ls_by_arm[arm],
                recovered_rotation=rotation, checkpoint_path=checkpoint_path)
            ledger.dump(folder / "result.json", result)
            row.update(status=result.status, outer_records=len(result.records),
                       converged_landings=sum(bool(r.landing is not None and r.landing.converged)
                                              for r in result.records),
                       outer_statuses=[r.status for r in result.records],
                       run_evaluation_requests=result.evaluation_requests,
                       request_count_matches_result=surface.requests == result.evaluation_requests)
        except Exception as error:
            row.update(status="exception", error=repr(error))
            partial = getattr(error, "result", None)
            if partial is not None:
                ledger.dump(folder / "partial-exception-result.json", partial)
            if checkpoint_path.is_file():
                try:
                    saved = load_ssw_checkpoint(checkpoint_path)
                    row.update(checkpoint_outer_completed=len(saved.records),
                               checkpoint_status=saved.status,
                               checkpoint_requests=saved.evaluation_requests)
                    ledger.dump(folder / "checkpoint.json", saved)
                except Exception as checkpoint_error:
                    row["checkpoint_read_error"] = repr(checkpoint_error)
        row["search_elapsed_seconds"] = time.monotonic() - started
        row["search_requests"] = 0 if surface is None else surface.requests
        row["search_calculator_calls"] = search_calls["calls"] - before_search
        row["search_denials"] = 0 if surface is None else surface.denials
        row["search_boundary"] = None if surface is None else surface.boundary
        minima = () if result is None else result.minima
        row["minima_total"] = len(minima) if result is not None else None
        checks = []
        if result is not None:
            if len(minima) > plan["fresh_cap_per_arm"]:
                row["fresh_cap_exceeded"] = len(minima) - plan["fresh_cap_per_arm"]
                row["status"] = "fresh_cap_exceeded"
            else:
                fresh_surface = ledger.CountedSurface(fresh_calc, folder / "fresh-requests.jsonl",
                    plan["fresh_cap_per_arm"], max(0., 2700 - (time.monotonic() - pilot_started)))
                for index, minimum in enumerate(minima):
                    check = {"index": index, "reported_energy_eV": minimum.energy,
                             "reported_fmax_eV_A": minimum.max_force,
                             "reported_converged": minimum.converged}
                    try:
                        fresh_calc.reset()
                        energy, forces = fresh_surface.evaluate(minimum.atoms)
                        fmax = float(np.linalg.norm(forces, axis=1).max())
                        same_numbers = bool(np.array_equal(minimum.atoms.numbers, atoms.numbers))
                        same_cell = bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array))
                        same_pbc = bool(np.array_equal(minimum.atoms.pbc, atoms.pbc))
                        check.update(energy_eV=energy, energy_error_eV=energy-minimum.energy,
                                     fmax_eV_A=fmax, same_numbers=same_numbers,
                                     same_cell=same_cell, same_pbc=same_pbc,
                                     numbers=minimum.atoms.numbers.tolist(),
                                     cell_A=minimum.atoms.cell.array.tolist(),
                                     pbc=minimum.atoms.pbc.tolist(),
                                     numerical_qualified=bool(minimum.converged and same_numbers
                                         and same_cell and same_pbc and abs(energy-minimum.energy) <= 1e-6
                                         and fmax <= cfg.fmax))
                    except Exception as error:
                        check.update(numerical_qualified=False, error=repr(error))
                    checks.append(check)
                    ledger.dump(folder / "fresh-checks.json", checks)
        row["fresh_requests"] = 0 if fresh_surface is None else fresh_surface.requests
        row["fresh_calculator_calls"] = fresh_calls["calls"] - before_fresh
        row["fresh_checked"] = len(checks)
        row["fresh_all_qualified"] = bool(result is not None and len(checks) == len(minima)
                                           and all(c["numerical_qualified"] for c in checks))
        row["pilot_integrity_ok"] = bool(result is not None
            and row.get("request_count_matches_result", False)
            and row["fresh_all_qualified"] and not row.get("fresh_cap_exceeded"))
        row["elapsed_seconds"] = time.monotonic() - started
        ledger.dump(folder / "summary.json", row)
        rows.append(row)
        ledger.dump(HERE / "summary.json", {"scope": "cost_and_lifecycle_pilot_only",
            "provenance": provenance, "rows": rows, "completed_arms": len(rows),
            "search_requests_total": sum(r["search_requests"] for r in rows),
            "fresh_requests_total": sum(r["fresh_requests"] for r in rows),
            "pilot_integrity_ok": len(rows) == len(plan["arms"])
                and all(r["pilot_integrity_ok"] for r in rows)})
        if row.get("fresh_cap_exceeded"):
            raise RuntimeError(f"{arm}: {len(minima)} minima exceed frozen fresh cap of 13")
    return 0 if all(row["pilot_integrity_ok"] for row in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("prepared_not_executed: pass --execute for the bounded MH-1 pilot")
        return
    plan = json.loads(PLAN_PATH.read_text())
    provenance = preflight(plan)
    raise SystemExit(execute(plan, provenance))


if __name__ == "__main__":
    main()
