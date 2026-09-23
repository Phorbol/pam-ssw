"""Prepare or execute the explicitly authorized three-structure C4H6 qualification."""
import argparse, hashlib, importlib.util, json, platform, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT, PLAN_PATH = HERE.parents[3], HERE / "plan.json"


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def preflight(plan):
    if plan.get("status") != "ready_for_qualification":
        raise RuntimeError(f"plan status is {plan.get('status')!r}; waiting for model-scope approval")
    required = ("model", "model_sha256", "head", "device", "dtype", "source_commit",
                "source_core_tree", "shared_ledger", "shared_ledger_sha256")
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"plan missing provenance fields: {missing}")
    git("cat-file", "-e", plan["source_commit"] + "^{commit}")
    actual_head, core_tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD:pamssw")
    if core_tree != plan["source_core_tree"]:
        raise RuntimeError(f"core tree mismatch: {core_tree} != plan {plan['source_core_tree']}")
    if git("diff", "HEAD", "--", "pamssw"):
        raise RuntimeError("tracked pamssw files differ from HEAD")
    model = Path(plan["model"])
    model_hash = sha256(model) if model.is_file() else None
    if model_hash != plan["model_sha256"]:
        raise RuntimeError(f"model missing or SHA256 mismatch: {model}")
    ledger = (HERE / plan["shared_ledger"]).resolve()
    ledger_hash = sha256(ledger) if ledger.is_file() else None
    if ledger_hash != plan["shared_ledger_sha256"]:
        raise RuntimeError(f"shared ledger missing or SHA256 mismatch: {ledger}")
    if plan.get("cases") != ["butadiene", "cyclobutene", "bicyclobutane"]:
        raise ValueError("plan cases must be the fixed ASE G2 three-structure set")
    if plan.get("request_cap_per_case") != 1500 or plan.get("wall_seconds_per_case") != 120:
        raise ValueError("plan request/wall caps differ from the fixed qualification protocol")
    cfg = plan.get("config", {})
    if (cfg.get("quench_optimizer") != "safe-lbfgs-total" or cfg.get("lbfgs_memory") != 500
            or cfg.get("fmax") != 0.03 or cfg.get("relax_steps") != 400):
        raise ValueError("plan SSW config differs from Safe-total/history500/fmax.03/steps400")
    return {"actual_head": actual_head, "actual_core_tree": core_tree,
            "source_commit": plan["source_commit"], "model_sha256": model_hash,
            "shared_ledger_sha256": ledger_hash, "shared_ledger": str(ledger)}


def execute(plan, provenance):
    if (HERE / "qualification.json").exists():
        raise FileExistsError("refusing to overwrite qualification.json")
    import numpy as np
    import ase
    import torch
    import mace
    from ase.collections import g2
    from mace.calculators import MACECalculator
    sys.path.insert(0, str(ROOT))
    from ase.io import write
    from pamssw.standalone import SSWConfig, run_ssw
    from pamssw.standalone.paper_reference import InitialQuenchError

    ledger_path = Path(provenance["shared_ledger"])
    spec = importlib.util.spec_from_file_location("shared_ledger", ledger_path)
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    sys.path.insert(0, str(ROOT / "research" / "ga_ssw"))
    from analyze_c4h6_ls_reaction_coverage import graph
    import networkx as nx

    runtime = {"python": platform.python_version(), "ase": ase.__version__,
               "mace": mace.__version__, "torch": torch.__version__}
    torch.set_num_threads(1)
    torch.manual_seed(0)
    calc_args = dict(model_paths=plan["model"], head=plan["head"], device=plan["device"],
                     default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False)
    calculator, fresh_calculator = MACECalculator(**calc_args), MACECalculator(**calc_args)
    search_counter = ledger.instrument_calculate(calculator)
    fresh_counter = ledger.instrument_calculate(fresh_calculator)
    rows = []
    for case in plan["cases"]:
        folder = HERE / case
        folder.mkdir(exist_ok=False)
        atoms = g2[case].copy()
        atoms.calc = None
        write(folder / "input.extxyz", atoms)
        ledger.dump(folder / "input.json", atoms)
        row = {"case": case, "status": "started", "search_requests": 0,
               "search_calculator_calls": 0, "fresh_requests": 0, "fresh_calculator_calls": 0,
               "numerical_qualified": False, "graph_retained": False, "qualified": False}
        result = terminal = None
        surface = fresh_surface = None
        started = time.monotonic()
        search_calls_before = search_counter["calls"]
        fresh_calls_before = fresh_counter["calls"]
        try:
            calculator.reset()
            surface = ledger.CountedSurface(calculator, folder / "requests.jsonl",
                plan["request_cap_per_case"], plan["wall_seconds_per_case"])
            config = SSWConfig(**plan["config"])
            result = run_ssw(atoms.copy(), surface, steps=0, config=config, rng=np.random.default_rng(plan["seed"]))
            ledger.dump(folder / "result.json", result)
            terminal = result.initial.atoms
            row.update(status=result.status, run_evaluation_requests=result.evaluation_requests,
                       initial_converged=result.initial.converged,
                       initial_energy_eV=result.initial.energy, initial_fmax_eV_A=result.initial.max_force)
        except Exception as error:
            row.update(status="exception", error=repr(error))
            failed = getattr(error, "result", None)
            if isinstance(error, InitialQuenchError) and failed is not None:
                result = failed
                terminal = failed.atoms
                ledger.dump(folder / "result.json", failed)
                row.update(initial_converged=failed.converged, initial_energy_eV=failed.energy, initial_fmax_eV_A=failed.max_force)
        row["search_requests"] = 0 if surface is None else surface.requests
        row["search_denials"] = 0 if surface is None else surface.denials
        row["search_boundary"] = None if surface is None else surface.boundary
        row["search_calculator_calls"] = search_counter["calls"] - search_calls_before
        if result is not None and surface is not None:
            row["request_count_matches_result"] = surface.requests == result.evaluation_requests
        if terminal is not None:
            write(folder / "terminal.extxyz", terminal)
            ledger.dump(folder / "terminal.json", terminal)
            try:
                fresh_calculator.reset()
                fresh_surface = ledger.CountedSurface(fresh_calculator, folder / "fresh-requests.jsonl", 1, 120)
                energy, forces = fresh_surface.evaluate(terminal)
                fresh_fmax = float(np.linalg.norm(forces, axis=1).max())
                retained = nx.is_isomorphic(graph(atoms), graph(terminal),
                    node_match=nx.algorithms.isomorphism.categorical_node_match("number", None))
                same_numbers = bool(np.array_equal(atoms.numbers, terminal.numbers))
                same_cell = bool(np.array_equal(atoms.cell.array, terminal.cell.array))
                same_pbc = bool(np.array_equal(atoms.pbc, terminal.pbc))
                initial = result.initial if hasattr(result, "initial") else result
                energy_error = float(energy - initial.energy)
                request_match = bool(row.get("request_count_matches_result", False))
                numerical = bool(np.isfinite([energy, fresh_fmax, energy_error]).all()
                    and abs(energy_error) <= 1e-6 and fresh_fmax <= plan["config"]["fmax"]
                    and row.get("initial_converged", False) and request_match
                    and same_numbers and same_cell and same_pbc and not bool(np.any(terminal.pbc)))
                row.update(fresh_energy_eV=energy, fresh_energy_error_eV=energy_error, fresh_fmax_eV_A=fresh_fmax,
                    same_numbers=same_numbers, same_cell=same_cell, same_pbc=same_pbc,
                    graph_retained=bool(retained), numerical_qualified=numerical, qualified=bool(numerical and retained))
            except Exception as error:
                row["fresh_error"] = repr(error)
        row["fresh_requests"] = 0 if fresh_surface is None else fresh_surface.requests
        row["fresh_calculator_calls"] = (0 if fresh_surface is None else
            fresh_counter["calls"] - fresh_calls_before)
        row["elapsed_seconds"] = time.monotonic() - started
        row["relative_energy_to_butadiene_eV"] = None
        rows.append(row)
        by_case = {r["case"]: r for r in rows}
        base = by_case.get("butadiene", {}).get("initial_energy_eV")
        if base is not None:
            for item in rows:
                e = item.get("initial_energy_eV")
                item["relative_energy_to_butadiene_eV"] = None if e is None else float(e - base)
        ledger.dump(HERE / "qualification.json", {"status": "running", "runtime": runtime,
            "provenance": provenance, "rows": rows, "all_qualified": False})
    totals = {key: sum(row[key] for row in rows) for key in
              ("search_requests", "search_calculator_calls", "fresh_requests", "fresh_calculator_calls")}
    all_qualified = len(rows) == 3 and all(r["qualified"] for r in rows)
    ledger.dump(HERE / "qualification.json", {"status": "complete", "runtime": runtime,
        "provenance": provenance, "totals": totals, "rows": rows, "all_qualified": all_qualified})
    return 0 if all_qualified else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute only after explicit model-scope approval")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    if plan.get("status") != "ready_for_qualification":
        print(f"prepared_not_executed: plan status is {plan.get('status')!r}")
        return 0
    if not args.execute:
        print("prepared_not_executed: pass --execute only after plan review")
        return 0
    provenance = preflight(plan)
    raise SystemExit(execute(plan, provenance))


if __name__ == "__main__":
    main()
