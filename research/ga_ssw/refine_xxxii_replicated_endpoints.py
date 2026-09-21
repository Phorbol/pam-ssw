"""Prepare or run independent strict full-cell quenches of two saved endpoints."""

import argparse
import hashlib
import json
import shutil
import signal
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research/ga_ssw/evidence/xxxii-rc-vc-replicated-completion"
OUT = ROOT / "research/ga_ssw/evidence/xxxii-replicated-endpoint-quench"
FIX = ROOT / "tests/standalone/fixtures/type2_xxxii.extxyz"
TOPO = SOURCE
MAX_SEARCH = 1000
MAX_FRESH = 1
WALL = 90


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_fresh_endpoints():
    rows = []
    with (SOURCE / "ledger.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("role") == "fresh" and row.get("status") == "completed":
                rows.append(row)
    if len(rows) < 2:
        raise RuntimeError("two completed full-precision fresh ledger rows are required")
    return rows[-2:]


def plan():
    endpoints = saved_fresh_endpoints()
    return {
        "status": "prepared",
        "source_run": str(SOURCE),
        "source_result_sha256": sha(SOURCE / "result.json"),
        "endpoint_source": "last two completed fresh rows in replicated-completion/ledger.jsonl; full precision positions/cell",
        "endpoints": [{"source_index": row["index"], "energy": row["energy"], "positions_shape": [len(row["positions"]), 3]} for row in endpoints],
        "public_atoms": 172,
        "repetitions": [1, 1, 2],
        "internal_atoms_per_engine_evaluation": 344,
        "quench": {"mode": "full_cell", "strain_length_A": 5.0, "pressure": 0.0,
                   "max_step": 0.2, "lbfgs_memory": 400, "maxiter": 1000,
                   "fmax": 0.001, "stress_tol": 0.0001},
        "budget": {"per_endpoint_search_api": MAX_SEARCH, "per_endpoint_fresh_api": MAX_FRESH,
                   "per_endpoint_total_api": MAX_SEARCH + MAX_FRESH, "wall_seconds": WALL,
                   "total_api_cap": 2 * (MAX_SEARCH + MAX_FRESH)},
        "qualification": "independent strict stationarity check; 10x force and stress tolerances relative to search; not search retuning",
        "limits": ["fixed GAFF topology", "known ERFC/derivative numerical floor", "no Hessian or chemical-stability claim"],
    }, endpoints


def freeze(plan_data, endpoints):
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "plan.json").write_text(json.dumps(plan_data, indent=2) + "\n")
    (OUT / "endpoints.json").write_text(json.dumps(endpoints, indent=2) + "\n")
    shutil.copy2(__file__, OUT / "runner-prepared.py")
    for name in ("xxxii_replicated_calculator.py", "convert_xxxii_amber.py"):
        shutil.copy2(ROOT / "research/ga_ssw" / name, OUT / name)
    for name in ("cell_relax.py", "vc_geometry.py", "generalized_numerics.py"):
        shutil.copy2(ROOT / "pamssw/standalone" / name, OUT / name)
    shutil.copytree(ROOT / "pamssw", OUT / "source/pamssw",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source, target in ((FIX, OUT / FIX.name), (SOURCE / "lmp.data", OUT / "lmp.data"),
                           (SOURCE / "in.simple", OUT / "in.simple"),
                           (SOURCE / "manifest.json", OUT / "manifest.json"),
                           (SOURCE / "rigidbody", OUT / "rigidbody"),
                           (SOURCE / "blist", OUT / "blist")):
        shutil.copy2(source, target)
    files = [OUT / "runner-prepared.py", OUT / "xxxii_replicated_calculator.py",
             OUT / "cell_relax.py", OUT / "vc_geometry.py", OUT / "generalized_numerics.py",
             OUT / "lmp.data", OUT / "in.simple", OUT / "manifest.json"]
    (OUT / "source-digests.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files) + "\n")


def execute():
    from ase import Atoms
    from pamssw.standalone.cell_relax import relax_cell_coordinates
    from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
    from pamssw.standalone.rc_topology import read_rigid_topology
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    from pamssw.standalone.vc_geometry import ASEStressSurface
    from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator

    if (OUT / "result.json").exists() or (OUT / "ledger.jsonl").exists():
        raise RuntimeError("refuse overwrite an executed endpoint refinement")
    plan_data, endpoint_rows = plan()
    if not OUT.exists():
        freeze(plan_data, endpoint_rows)
    started = time.monotonic()
    shutil.copytree(ROOT / 'pamssw', OUT / 'source-executed/pamssw',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Refresh the execution snapshot after review, before any PES call.
    shutil.copy2(__file__, OUT / "runner-executed.py")
    shutil.copy2(ROOT / "research/ga_ssw/xxxii_replicated_calculator.py", OUT / "xxxii_replicated_calculator.py")
    shutil.copy2(ROOT / "pamssw/standalone/generalized_numerics.py", OUT / "generalized_numerics.py")
    (OUT / "source-execution-refresh.json").write_text(json.dumps({
        "status": "refreshed_before_execution", "runner_sha256": sha(OUT / "runner-executed.py"),
        "generalized_numerics_sha256": sha(OUT / "generalized_numerics.py"),
        "replicated_calculator_sha256": sha(OUT / "xxxii_replicated_calculator.py")}, indent=2) + "\n")
    topology = read_rigid_topology(OUT / "rigidbody", OUT / "blist", natoms=172)
    all_results = []

    def endpoint_atoms(row):
        return Atoms(numbers=row["numbers"], positions=row["positions"],
                    cell=row["cell"], pbc=row["pbc"])

    def one(index, row):
        endpoint_started = time.monotonic()
        attempts = []
        counts = {"search": 0, "fresh": 0}
        engines = []
        endpoint = endpoint_atoms(row)
        chart = SymmetricLogStrainChart(endpoint, strain_length=5.0)

        def new_surface(role):
            calc = XXXIIReplicatedCalculator(
                data_path=OUT / "lmp.data", input_path=OUT / "in.simple",
                model_manifest=OUT / "manifest.json", reference_atoms=endpoint,
                repetitions=(1, 1, 2))
            engines.append(calc)

            class Counted(ASEStressSurface):
                def evaluate(self, atoms):
                    row_log = {"index": len(attempts), "role": role, "status": "started",
                               "endpoint": index, "api_calls": 0,
                               "positions": atoms.positions.tolist(), "cell": atoms.cell.array.tolist()}
                    attempts.append(row_log)
                    engine_before = self.calculator.engine_calls
                    atoms_before = self.calculator.atoms_evaluated
                    try:
                        if counts[role] >= (MAX_SEARCH if role == "search" else MAX_FRESH):
                            raise RuntimeError("declared endpoint budget exhausted")
                        if time.monotonic() - endpoint_started >= WALL:
                            raise RuntimeError("declared endpoint wall budget exhausted")
                        counts[role] += 1
                        row_log["api_calls"] = 1
                        energy, force, stress = super().evaluate(atoms)
                        row_log.update(status="completed", energy=energy, forces=force.tolist(), stress=stress.tolist())
                        return energy, force, stress
                    except BaseException as error:
                        row_log.update(status="failed", error=repr(error))
                        raise
                    finally:
                        row_log["elapsed_seconds"] = time.monotonic() - started
                        row_log["calculator_api_calls"] = self.calculator.api_calls
                        row_log["engine_calls"] = self.calculator.engine_calls - engine_before
                        row_log["atoms_evaluated"] = self.calculator.atoms_evaluated - atoms_before
                        with (OUT / "ledger.jsonl").open("a") as stream:
                            stream.write(json.dumps(row_log) + "\n")

            return Counted(calc), calc

        result = {"endpoint": index, "status": "running", "source_energy": row["energy"]}
        search_surface, search_calc = new_surface("search")
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(RuntimeError("declared endpoint wall budget exhausted")))
        signal.setitimer(signal.ITIMER_REAL, WALL)
        try:
            relaxed = relax_cell_coordinates(
                chart, chart.pack(endpoint), search_surface, pressure=0.0,
                fmax=0.001, stress_tol=0.0001, max_step=0.2, maxiter=1000,
                lbfgs_memory=400)
            result["optimizer"] = {"status": relaxed.status, "steps": relaxed.steps,
                                    "requests": relaxed.requests, "error": relaxed.error,
                                    "energy": relaxed.energy}
            final_atoms = chart.unpack(relaxed.q)
            search_calc.close()
            fresh_surface, fresh_calc = new_surface("fresh")
            evaluation = chart.evaluate(relaxed.q, fresh_surface.evaluate, pressure=0.0)
            result["final"] = {"energy": evaluation.energy, "fmax": float(np.linalg.norm(evaluation.forces, axis=1).max()),
                                "stress_max": float(np.abs(evaluation.stress).max()),
                                "volume": final_atoms.get_volume(), "positions": final_atoms.positions.tolist(),
                                "cell": final_atoms.cell.array.tolist(), "numbers": final_atoms.numbers.tolist(),
                                "pbc": final_atoms.pbc.tolist(), "forces": evaluation.forces.tolist(),
                                "stress": evaluation.stress.tolist()}
            fresh_calc.close()
            result["status"] = "completed" if relaxed.converged and result["final"]["fmax"] <= 0.001 and result["final"]["stress_max"] <= 0.0001 else "completed_with_failures"
        except BaseException as error:
            result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for calc in engines:
                calc.close()
        result.update(search_api=counts["search"], fresh_api=counts["fresh"],
                      engine_calls=sum(c.engine_calls for c in engines),
                      atoms_evaluated=sum(c.atoms_evaluated for c in engines),
                      attempts=len(attempts), wall_seconds=time.monotonic() - endpoint_started)
        (OUT / f"endpoint-{index}-result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    for index, row in enumerate(endpoint_rows):
        all_results.append(one(index, row))
    report = dict(status='completed' if all(r['status']=='completed' for r in all_results) else 'completed_with_failures', endpoints=all_results,
                  total_API=sum(r['search_api']+r['fresh_api'] for r in all_results), engine_calls=sum(r['engine_calls'] for r in all_results),
                  atoms_evaluated=sum(r['atoms_evaluated'] for r in all_results), wall_seconds=time.monotonic()-started)
    (OUT / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print({k:v for k,v in report.items() if k!='endpoints'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        plan_data, endpoint_rows = plan()
        if OUT.exists():
            raise RuntimeError("output directory already exists")
        freeze(plan_data, endpoint_rows)
        print(json.dumps({"status": "prepared", "pes_calls": 0,
                          "endpoints": len(endpoint_rows), "per_endpoint_cap": MAX_SEARCH + MAX_FRESH}))


if __name__ == "__main__":
    main()
