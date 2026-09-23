"""One fixed-protocol C4H6/MH-1 comparison arm; explicit execution only."""
import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PILOT = HERE.parent / "c4h6-mh1-lifecycle-20260924"
PLAN_PATH = HERE / "plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def preflight(plan, arm, seed):
    if plan.get("status") != "ready_for_controlled_comparison":
        raise RuntimeError("comparison plan is not ready")
    pilot = json.loads((PILOT / "plan.json").read_text())
    pilot_summary = json.loads((PILOT / "summary.json").read_text())
    if (not pilot_summary.get("pilot_integrity_ok") or len(pilot_summary.get("rows", [])) != 3
            or not all(row.get("pilot_integrity_ok") for row in pilot_summary["rows"])):
        raise RuntimeError("completed pilot integrity evidence is missing")
    if not (PILOT / "effective_config.json").is_file():
        raise RuntimeError("pilot effective settings artifact is missing")
    if arm not in plan.get("arms", ()) or seed not in plan.get("seeds", ()):
        raise ValueError("arm or seed is outside the frozen protocol")
    if plan.get("arms") != ["ssw", "paper_ls", "native_ls"] or plan.get("seeds") != [61, 67]:
        raise RuntimeError("arms/seeds differ from the fixed comparison")
    if (plan.get("steps") != 400 or plan.get("request_cap_per_arm") != 320000
            or plan.get("wall_seconds_per_arm") != 6600 or plan.get("fresh_cap_per_arm") != 401):
        raise RuntimeError("comparison budget differs from the fixed protocol")
    if plan.get("source_commit") != "57e5097c2a8546ccb6cf60a4396af662f19c58a1":
        raise RuntimeError("unexpected source commit")
    git("cat-file", "-e", plan["source_commit"] + "^{commit}")
    if git("rev-parse", plan["source_commit"] + ":pamssw") != plan["source_core_tree"]:
        raise RuntimeError("source commit core differs from frozen core")
    if git("rev-parse", "HEAD:pamssw") != plan.get("source_core_tree"):
        raise RuntimeError("frozen core tree mismatch")
    if git("diff", "HEAD", "--", "pamssw"):
        raise RuntimeError("tracked pamssw core differs from HEAD")
    common = ("source_core_tree", "shared_ledger_sha256", "model", "model_sha256",
              "head", "device", "dtype", "input_sha256", "config", "recovered_rotation",
              "prequench", "paper_target_eV_per_atom", "native_target_meV_per_atom")
    for key in common:
        if plan.get(key) != pilot.get(key):
            raise RuntimeError(f"comparison differs from pilot settings: {key}")
    qualification_path = (HERE / plan["qualification"]).resolve()
    qualification = json.loads(qualification_path.read_text())
    if (qualification.get("status") != "complete" or not qualification.get("all_qualified")
            or len(qualification.get("rows", [])) != 3
            or not all(row.get("qualified") for row in qualification["rows"])):
        raise RuntimeError("MH-1 three-case qualification is incomplete")
    if plan.get("slurm_wall_seconds_per_arm") != 7200:
        raise RuntimeError("single-arm wall ceiling differs from fixed protocol")
    if (plan.get("head"), plan.get("device"), plan.get("dtype")) != ("omol", "cuda", "float64"):
        raise RuntimeError("unexpected calculator settings")
    model = Path(plan["model"])
    if sha256(model) != plan["model_sha256"]:
        raise RuntimeError("model SHA256 mismatch")
    input_path = (HERE / plan["input"]).resolve()
    if input_path != (PILOT / pilot["input"]).resolve() or sha256(input_path) != plan["input_sha256"]:
        raise RuntimeError("qualified input path or SHA256 mismatch")
    ledger_path = (HERE / plan["shared_ledger"]).resolve()
    if sha256(ledger_path) != plan["shared_ledger_sha256"]:
        raise RuntimeError("shared ledger SHA256 mismatch")
    folder = HERE / f"{arm}-seed{seed}"
    if folder.exists():
        raise FileExistsError(f"output exists: {folder}; no overwrite or implicit retry")
    return folder, {"git_head": git("rev-parse", "HEAD"), "core_tree": plan["source_core_tree"],
                    "plan_sha256": sha256(PLAN_PATH), "runner_sha256": sha256(__file__),
                    "pilot_plan_sha256": sha256(PILOT / "plan.json"),
                    "pilot_summary_sha256": sha256(PILOT / "summary.json"),
                    "pilot_effective_sha256": sha256(PILOT / "effective_config.json"),
                    "qualification_sha256": sha256(qualification_path),
                    "model_sha256": plan["model_sha256"], "input_sha256": plan["input_sha256"],
                    "ledger_sha256": plan["shared_ledger_sha256"], "ledger_path": str(ledger_path)}


def execute(plan, arm, seed, folder, provenance):
    import numpy as np
    import ase
    import mace
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

    spec = importlib.util.spec_from_file_location("comparison_shared_ledger", provenance["ledger_path"])
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    cfg = SSWConfig(**plan["config"])
    rotation = RecoveredRotationSettings(**plan["recovered_rotation"])
    prequench = LSPrequenchSettings(**plan["prequench"])
    ls = {"ssw": None,
          "paper_ls": LSSettings(HC_BOND_ENERGIES, {k: v + .1 for k, v in HC_BOND_LENGTHS.items()},
                                 target_per_atom=plan["paper_target_eV_per_atom"], prequench=prequench),
          "native_ls": NativeLSSettings(HC_BOND_ENERGIES, HC_BOND_LENGTHS,
                                        target_mev_per_atom=plan["native_target_meV_per_atom"],
                                        prequench=prequench)}[arm]
    effective = {"config": asdict(cfg), "rotation_settings": asdict(rotation),
                 "ls_settings": None if ls is None else asdict(ls), "arm": arm, "seed": seed,
                 "steps": plan["steps"], "request_cap": plan["request_cap_per_arm"],
                 "fresh_cap": plan["fresh_cap_per_arm"], "search_wall_seconds": plan["wall_seconds_per_arm"]}
    prior = json.loads((PILOT / "effective_config.json").read_text())
    expected = {"config": prior["config"], "rotation_settings": prior["rotation_settings"],
                "ls_settings": prior["ls_settings"][arm]}
    actual = ledger._jsonable(effective)
    if {key: actual[key] for key in expected} != expected:
        raise RuntimeError("effective settings differ from executed pilot")
    folder.mkdir(exist_ok=False)
    ledger.dump(folder / "effective_config.json", effective)
    atoms = read((HERE / plan["input"]).resolve())
    started = time.monotonic()
    calc_args = dict(model_paths=plan["model"], head=plan["head"], device=plan["device"],
                     default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False)
    row = {"arm": arm, "seed": seed, "status": "started", "minima_total": None,
           "runtime": {"python": platform.python_version(), "ase": ase.__version__,
                       "mace": mace.__version__, "torch": torch.__version__,
                       "torch_manual_seed": 0, "torch_num_threads": torch.get_num_threads(),
                       "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                       "cudnn_deterministic": torch.backends.cudnn.deterministic},
           "provenance": provenance}
    search = fresh = surface = fresh_surface = result = None
    search_calls = fresh_calls = None
    checkpoint = folder / "checkpoint.pkl"
    try:
        search, fresh = MACECalculator(**calc_args), MACECalculator(**calc_args)
        search_calls, fresh_calls = ledger.instrument_calculate(search), ledger.instrument_calculate(fresh)
        search.reset()
        remaining = plan["slurm_wall_seconds_per_arm"] - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeError("single_arm_wall_cap")
        surface = ledger.CountedSurface(search, folder / "requests.jsonl",
            plan["request_cap_per_arm"], min(plan["wall_seconds_per_arm"], remaining))
        result = run_ssw(atoms.copy(), surface, steps=plan["steps"], config=cfg,
            rng=np.random.default_rng(seed), ls=ls, recovered_rotation=rotation,
            checkpoint_path=checkpoint)
        ledger.dump(folder / "result.json", result)
        row.update(status=result.status, run_evaluation_requests=result.evaluation_requests,
                   request_count_matches_result=result.evaluation_requests == surface.requests,
                   outer_records=len(result.records),
                   converged_landings=sum(bool(r.landing is not None and r.landing.converged)
                                          for r in result.records),
                   accepted_landings=sum(bool(r.landing is not None and r.accepted)
                                         for r in result.records),
                   outer_statuses=[r.status for r in result.records])
    except Exception as error:
        row.update(status="exception", error=repr(error))
        partial = getattr(error, "result", None)
        if partial is not None:
            ledger.dump(folder / "partial-exception-result.json", partial)
        if checkpoint.is_file():
            try:
                saved = load_ssw_checkpoint(checkpoint)
                row.update(checkpoint_outer_records=len(saved.records), checkpoint_status=saved.status,
                           checkpoint_requests=saved.evaluation_requests)
                ledger.dump(folder / "checkpoint.json", saved)
            except Exception as checkpoint_error:
                row["checkpoint_read_error"] = repr(checkpoint_error)
    row["search_elapsed_seconds"] = time.monotonic() - started
    row["search_requests"] = 0 if surface is None else surface.requests
    row["search_calculator_calls"] = 0 if search_calls is None else search_calls["calls"]
    row["search_denials"] = 0 if surface is None else surface.denials
    row["search_boundary"] = None if surface is None else surface.boundary
    minima = () if result is None else result.minima
    row["minima_total"] = None if result is None else len(minima)
    checks = []
    if result is not None and len(minima) <= plan["fresh_cap_per_arm"]:
        remaining = max(0., plan["slurm_wall_seconds_per_arm"] - (time.monotonic() - started))
        fresh_surface = ledger.CountedSurface(fresh, folder / "fresh-requests.jsonl",
            plan["fresh_cap_per_arm"], remaining)
        for index, minimum in enumerate(minima):
            check = {"index": index, "reported_energy_eV": minimum.energy,
                     "reported_fmax_eV_A": minimum.max_force,
                     "reported_converged": minimum.converged}
            try:
                fresh.reset()
                energy, forces = fresh_surface.evaluate(minimum.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                same_numbers = bool(np.array_equal(minimum.atoms.numbers, atoms.numbers))
                same_cell = bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array))
                same_pbc = bool(np.array_equal(minimum.atoms.pbc, atoms.pbc))
                check.update(energy_eV=energy, energy_error_eV=energy-minimum.energy,
                             fmax_eV_A=fmax, same_numbers=same_numbers, same_cell=same_cell,
                             same_pbc=same_pbc, numbers=minimum.atoms.numbers.tolist(),
                             cell_A=minimum.atoms.cell.array.tolist(), pbc=minimum.atoms.pbc.tolist(),
                             numerical_qualified=bool(minimum.converged and same_numbers and same_cell
                                 and same_pbc and abs(energy-minimum.energy) <= 1e-6 and fmax <= cfg.fmax))
            except Exception as error:
                check.update(numerical_qualified=False, error=repr(error))
            checks.append(check)
            ledger.append(folder / "fresh-checks.jsonl", check)
            if "error" in check:
                for missing in range(index + 1, len(minima)):
                    skipped = {"index": missing, "numerical_qualified": False,
                               "status": "not_run_after_fresh_failure"}
                    checks.append(skipped)
                    ledger.append(folder / "fresh-checks.jsonl", skipped)
                break
        ledger.dump(folder / "fresh-checks.json", checks)
    elif result is not None:
        row["fresh_cap_exceeded"] = len(minima) - plan["fresh_cap_per_arm"]
    row["fresh_requests"] = 0 if fresh_surface is None else fresh_surface.requests
    row["fresh_calculator_calls"] = 0 if fresh_calls is None else fresh_calls["calls"]
    row["fresh_checked"] = sum("energy_eV" in c or "error" in c for c in checks)
    row["fresh_unrun"] = sum(c.get("status") == "not_run_after_fresh_failure" for c in checks)
    row["fresh_all_qualified"] = bool(result is not None and len(minima) > 0
                                      and len(checks) == len(minima)
                                      and all(c["numerical_qualified"] for c in checks))
    row["integrity_ok"] = bool(result is not None and row.get("request_count_matches_result", False)
                               and row["fresh_all_qualified"] and not row.get("fresh_cap_exceeded"))
    row["protocol_completed"] = bool(result is not None and result.status == "completed"
                                      and len(result.records) == plan["steps"])
    row["budget_censored"] = row["search_boundary"] in ("request_cap", "wall_cap")
    row["execution_ok"] = bool(row["integrity_ok"]
                                and (row["protocol_completed"] or row["budget_censored"]))
    row["elapsed_seconds"] = time.monotonic() - started
    ledger.dump(folder / "summary.json", row)
    if row.get("fresh_cap_exceeded"):
        raise RuntimeError(f"{len(minima)} minima exceed frozen fresh cap of 401")
    return 0 if row["execution_ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("ssw", "paper_ls", "native_ls"))
    parser.add_argument("--seed", required=True, type=int, choices=(61, 67))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("prepared_not_executed: pass --execute for one fixed comparison arm")
        return
    plan = json.loads(PLAN_PATH.read_text())
    folder, provenance = preflight(plan, args.arm, args.seed)
    raise SystemExit(execute(plan, args.arm, args.seed, folder, provenance))


if __name__ == "__main__":
    main()
