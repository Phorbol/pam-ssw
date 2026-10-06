"""Bounded MACE qualification of three periodic Ti4O8 phase inputs.

--preflight validates provenance and APIs using a zero-force dummy oracle.
--execute is reserved for the authorized single Slurm allocation.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN_PATH = HERE / "plan.json"
PYTHON = Path("/home/gengjianrui/.conda/envs/mace_env/bin/python3.12")
CORE_FILES = ("pamssw/standalone/__init__.py", "pamssw/standalone/vc_geometry.py", "pamssw/standalone/cell_relax.py",
              "pamssw/standalone/generalized_numerics.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def validate_plan(plan):
    required = {"model", "model_sha256", "calculator", "quench", "budget", "cases"}
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"plan missing required fields: {missing}")
    calc = plan["calculator"]
    if calc != {"head": "omat_pbe", "device": "cuda", "dtype": "float64",
                "enable_cueq": False, "enable_oeq": False}:
        raise ValueError("calculator settings differ from the approved OMAT-small configuration")
    if plan["quench"] != {"strain_length_A": 5.0, "pressure": 0.0, "fmax": 0.05,
                          "stress_tol": 0.001, "max_step": 0.2, "maxiter": 300,
                          "lbfgs_memory": 500}:
        raise ValueError("quench settings differ from the approved protocol")
    if plan["budget"] != {"per_case_requests": 1000, "per_case_seconds": 180,
                          "total_seconds": 600, "independent_fresh_per_case": 2}:
        raise ValueError("budget differs from the approved protocol")
    ids = [case.get("id") for case in plan["cases"]]
    if ids != ["phase87_is", "anatase_fs", "rutile"]:
        raise ValueError(f"unexpected case order/IDs: {ids}")
    if len(plan["cases"]) != 3:
        raise ValueError("exactly three cases are required")
    model = Path(plan["model"])
    if not model.is_file() or sha256(model) != plan["model_sha256"]:
        raise RuntimeError(f"model missing or SHA256 mismatch: {model}")
    from ase.io import read
    for case in plan["cases"]:
        source = Path(case["path"])
        if not source.is_file() or sha256(source) != case["sha256"]:
            raise RuntimeError(f"case source missing or SHA256 mismatch: {source}")
        atoms = read(source)
        if len(atoms) != 12 or not atoms.pbc.all() or atoms.constraints:
            raise ValueError(f"{case['id']} must be an unconstrained, fully periodic 12-atom input")
        if atoms.get_chemical_formula() != "O8Ti4":
            raise ValueError(f"{case['id']} composition is {atoms.get_chemical_formula()}, expected Ti4O8")
    return model


def provenance(plan):
    sys.path.insert(0, str(ROOT))
    from ase.io import read
    inputs = {case["id"]: {"path": case["path"], "sha256": sha256(case["path"]),
                           "atoms": len(read(case["path"]))} for case in plan["cases"]}
    core = {name: sha256(ROOT / name) for name in CORE_FILES}
    return {"head": git("rev-parse", "HEAD"), "head_pamssw_tree": git("rev-parse", "HEAD:pamssw"),
            "python": sys.executable, "python_version": platform.python_version(),
            "imports": {name: str(Path(importlib.import_module(name).__file__).resolve())
                        for name in ("ase", "mace", "torch", "pamssw.standalone.vc_geometry",
                                     "pamssw.standalone.cell_relax")},
            "model": plan["model"], "model_sha256": sha256(plan["model"]),
            "source_hashes": {"plan.json": sha256(PLAN_PATH), "runner.py": sha256(__file__),
                              **core}, "inputs": inputs}


def preflight(plan):
    validate_plan(plan)
    sys.path.insert(0, str(ROOT))
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from mace.calculators import MACECalculator
    from pamssw.standalone.vc_geometry import ASEStressSurface
    from pamssw.standalone.cell_relax import cell_quench
    import torch

    import numpy as np
    class ZeroEFStressCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if atoms.info.get("dummy_fail", False):
                raise RuntimeError("intentional zero-PES preflight failure")
            self.results = {"energy": 0.0, "forces": np.zeros_like(atoms.positions),
                            "stress": np.zeros(6)}

    dummy = Atoms("Ti4O8", positions=np.zeros((12, 3)), cell=np.eye(3) * 8, pbc=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="tio2-preflight-") as temp:
        cache_budget = CaseBudget("preflight-cache", Path(temp) / "cache.jsonl", 2, 30,
                                  time.monotonic() + 30)
        cache_calc = ZeroEFStressCalculator()
        instrument_calculate(cache_calc, cache_budget)
        cache_surface = CountedStressSurface(cache_calc, cache_budget, "dummy")
        cache_surface.evaluate(dummy)
        cache_surface.evaluate(dummy.copy())
        try:
            cache_surface.evaluate(dummy.copy())
        except RuntimeError:
            pass
        else:
            raise RuntimeError("counted surface did not enforce dummy request cap")
        if (cache_budget.requests != 2 or cache_budget.calculator_calls != 1 or
                cache_budget.denials != 1 or cache_budget.failures != 0):
            raise RuntimeError("counted surface cache/cap accounting mismatch")

        fail_budget = CaseBudget("preflight-failure", Path(temp) / "failure.jsonl", 1, 30,
                                 time.monotonic() + 30)
        fail_calc = ZeroEFStressCalculator()
        instrument_calculate(fail_calc, fail_budget)
        fail_surface = CountedStressSurface(fail_calc, fail_budget, "dummy_failure")
        failing_dummy = dummy.copy()
        failing_dummy.info["dummy_fail"] = True
        try:
            fail_surface.evaluate(failing_dummy)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("dummy calculator failure was not propagated")
        if (fail_budget.requests != 1 or fail_budget.calculator_calls != 1 or
                fail_budget.failures != 1):
            raise RuntimeError("counted surface failure accounting mismatch")

        quench_budget = CaseBudget('preflight-quench', Path(temp) / 'quench.jsonl', 10, 30,
                                  time.monotonic() + 30)
        quench_calc = ZeroEFStressCalculator()
        instrument_calculate(quench_calc, quench_budget)
        quench_surface = CountedStressSurface(quench_calc, quench_budget, 'dummy_quench')
        quench = cell_quench(dummy, quench_surface, strain_length=5., pressure=0.,
                            fmax=.05, stress_tol=.001, max_step=.2, maxiter=300,
                            lbfgs_memory=500)
        if not quench.converged or quench.requests != quench_budget.requests:
            raise RuntimeError('counted cell-quench API/cost check failed')

    constructor_signature = inspect.signature(MACECalculator)
    calculate_signature = inspect.signature(MACECalculator.calculate)
    if "default_dtype" not in constructor_signature.parameters:
        raise RuntimeError("installed MACECalculator lacks the default_dtype constructor argument")
    return {"status": "preflight_passed", "real_ef_requests": 0,
            "model_constructor": str(constructor_signature),
            "calculate_method": str(calculate_signature),
            "runtime_dtype": {"requested": plan["calculator"]["dtype"],
                              "torch_float64": str(torch.float64),
                              "mace_argument": "default_dtype"},
            "surface_api": str(inspect.signature(ASEStressSurface)),
            "cell_quench_api": str(inspect.signature(cell_quench)),
            "dummy_cache_case": {"requests": cache_budget.requests,
                                 "calculator_calls": cache_budget.calculator_calls,
                                 "denials": cache_budget.denials},
            "dummy_failure_case": {"requests": fail_budget.requests,
                                   "calculator_calls": fail_budget.calculator_calls,
                                   "failures": fail_budget.failures},
            "dummy_quench": {"converged": quench.converged, 'requests': quench.requests},
            "provenance": provenance(plan)}


class CaseBudget:
    def __init__(self, case_id, ledger, request_cap, wall_seconds, total_deadline):
        self.case_id, self.ledger = case_id, ledger
        self.cap, self.wall = request_cap, wall_seconds
        self.started = time.monotonic()
        self.total_deadline = total_deadline
        self.requests = self.calculator_calls = self.denials = self.failures = 0
        self.boundary = None

    def check(self, stage):
        if self.requests >= self.cap:
            self.boundary = "per_case_request_cap"
        elif time.monotonic() - self.started >= self.wall:
            self.boundary = "per_case_wall_cap"
        elif time.monotonic() >= self.total_deadline:
            self.boundary = "total_wall_cap"
        if self.boundary:
            self.denials += 1
            append(self.ledger, {"event": "denial", "stage": stage,
                                 "request": self.requests, "calculator_calls": 0,
                                 "reason": self.boundary})
            raise RuntimeError(self.boundary)
        self.requests += 1


class CountedStressSurface:
    def __init__(self, calculator, budget, stage):
        from pamssw.standalone.vc_geometry import ASEStressSurface
        self.calculator, self.budget, self.stage = calculator, budget, stage
        self.core = ASEStressSurface(calculator)

    @property
    def requests(self):
        return self.budget.requests

    def evaluate(self, atoms):
        self.budget.check(self.stage)
        calls_before = self.budget.calculator_calls
        try:
            energy, forces, stress = self.core.evaluate(atoms)
            import numpy as np
            forces, stress = np.asarray(forces, float), np.asarray(stress, float)
            row = {"event": "evaluation", "stage": self.stage,
                   "request": self.budget.requests, "energy_eV": energy,
                   "calculator_calls": self.budget.calculator_calls - calls_before,
                   "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                   "stress_max_eV_A3": float(np.abs(stress).max()),
                   "volume_A3": float(atoms.get_volume()), "natoms": len(atoms)}
            append(self.budget.ledger, row)
            return energy, forces.copy(), stress.copy()
        except Exception as error:
            self.budget.failures += 1
            append(self.budget.ledger, {"event": "failure", "stage": self.stage,
                                        "request": self.budget.requests,
                                        "calculator_calls": self.budget.calculator_calls - calls_before,
                                        "error": repr(error)})
            raise


def instrument_calculate(calculator, budget):
    original = calculator.calculate
    def counted(*args, **kwargs):
        budget.calculator_calls += 1
        return original(*args, **kwargs)
    calculator.calculate = counted


def execute(plan, prov, out):
    total_deadline = time.monotonic() + plan["budget"]["total_seconds"]
    import numpy as np
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone.cell_relax import cell_quench

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    calc_spec = plan["calculator"]
    calc_kw = {"model_paths": plan["model"], "head": calc_spec["head"],
               "device": calc_spec["device"], "default_dtype": calc_spec["dtype"],
               "enable_cueq": calc_spec["enable_cueq"], "enable_oeq": calc_spec["enable_oeq"]}
    rows = []
    for case in plan["cases"]:
        case_dir = out / case["id"]
        case_dir.mkdir(exist_ok=False)
        atoms = read(case["path"])
        atoms.calc = None
        write(case_dir / "input.extxyz", atoms)
        row = {"case": case["id"], "status": "started", "fresh_checks": [],
               "numerical_qualified": False}
        search_budget = CaseBudget(case["id"], case_dir / "requests.jsonl",
                            plan["budget"]["per_case_requests"],
                            plan["budget"]["per_case_seconds"], total_deadline)
        result = None
        fresh_budget = None
        search_elapsed = None
        try:
            calculator = MACECalculator(**calc_kw)
            row['model_parameter_dtypes'] = sorted({str(p.dtype) for model in calculator.models
                                                    for p in model.parameters()})
            if row['model_parameter_dtypes'] != ['torch.float64']:
                raise RuntimeError('loaded model parameters do not use requested float64')
            instrument_calculate(calculator, search_budget)
            calculator.reset()
            initial_surface = CountedStressSurface(calculator, search_budget, "initial")
            e0, f0, s0 = initial_surface.evaluate(atoms)
            row.update(initial_energy_eV=e0, initial_fmax_eV_A=float(np.linalg.norm(f0, axis=1).max()),
                       initial_stress_max_eV_A3=float(np.abs(s0).max()))
            quench_surface = CountedStressSurface(calculator, search_budget, "cell_quench")
            result = cell_quench(atoms, quench_surface, **{
                "strain_length": plan["quench"]["strain_length_A"],
                **{key: value for key, value in plan["quench"].items() if key != "strain_length_A"}})
            if result.evaluation is not None:
                terminal = result.evaluation.atoms.copy()
                write(case_dir / "endpoint.extxyz", terminal)
                row.update(endpoint_energy_eV=float(result.evaluation.energy),
                           endpoint_fmax_eV_A=float(np.linalg.norm(result.evaluation.forces, axis=1).max()),
                           endpoint_stress_max_eV_A3=float(np.abs(result.evaluation.stress).max()),
                           certificate=result.certificate,
                           optimizer_converged=bool(result.optimizer.converged),
                           optimizer_status=str(result.optimizer.status),
                           quench_requests=int(result.requests))
            else:
                terminal = None
                row.update(certificate=result.certificate,
                           optimizer_converged=bool(result.optimizer.converged),
                           optimizer_status=str(result.optimizer.status),
                           quench_requests=int(result.requests))
            search_elapsed = time.monotonic() - search_budget.started
            # Fresh cold checks have their own two-request, 30-second ledger.
            fresh_deadline = time.monotonic() + 30.
            fresh_budget = CaseBudget(case["id"], case_dir / "fresh-requests.jsonl", 2, 30.,
                                      fresh_deadline)
            fresh_calc = MACECalculator(**calc_kw)
            instrument_calculate(fresh_calc, fresh_budget)
            targets = [("initial", atoms, e0),
                       ("endpoint", terminal, None if terminal is None else row.get("endpoint_energy_eV"))]
            for name, target, stored_energy in targets:
                if target is None:
                    row["fresh_checks"].append({"target": name, "qualified": False,
                                                  "error": "endpoint unavailable"})
                    continue
                try:
                    fresh_calc.reset()
                    fresh_surface = CountedStressSurface(fresh_calc, fresh_budget, "fresh_" + name)
                    energy, forces, stress = fresh_surface.evaluate(target)
                    ferr = float(np.linalg.norm(forces, axis=1).max())
                    energy_error = float(energy - stored_energy)
                    qualified = bool(np.isfinite([energy, ferr, energy_error]).all() and
                                     abs(energy_error) <= 1e-6)
                    if name == "endpoint":
                        qualified = bool(qualified and ferr <= plan["quench"]["fmax"] and
                                         float(np.abs(stress).max()) <= plan["quench"]["stress_tol"])
                    row["fresh_checks"].append({"target": name, "energy_eV": energy,
                        "energy_error_eV": energy_error, "fmax_eV_A": ferr,
                        "stress_max_eV_A3": float(np.abs(stress).max()), "qualified": qualified})
                except Exception as error:
                    row["fresh_checks"].append({"target": name, "qualified": False,
                                                  "error": repr(error)})
            numerical = bool(result is not None and result.converged and
                             len(row["fresh_checks"]) == 2 and
                             all(check.get("qualified", False) for check in row["fresh_checks"]))
            row["numerical_qualified"] = numerical
            row["status"] = "complete" if numerical else "numerically_unqualified"
        except Exception as error:
            row.update(status="exception", error=repr(error), traceback=traceback.format_exc())
        if search_elapsed is None:
            search_elapsed = time.monotonic() - search_budget.started
        row.update(search_requests=search_budget.requests,
                   search_calculator_calls=search_budget.calculator_calls,
                   search_failures=search_budget.failures,
                   search_denials=search_budget.denials,
                   search_boundary=search_budget.boundary,
                   search_elapsed_seconds=search_elapsed,
                   fresh_requests=0 if fresh_budget is None else fresh_budget.requests,
                   fresh_calculator_calls=0 if fresh_budget is None else fresh_budget.calculator_calls,
                   fresh_failures=0 if fresh_budget is None else fresh_budget.failures,
                   fresh_denials=0 if fresh_budget is None else fresh_budget.denials,
                   fresh_boundary=None if fresh_budget is None else fresh_budget.boundary,
                   fresh_elapsed_seconds=0. if fresh_budget is None else time.monotonic() - fresh_budget.started)
        row["total_requests"] = row["search_requests"] + row["fresh_requests"]
        row["total_calculator_calls"] = row["search_calculator_calls"] + row["fresh_calculator_calls"]
        dump(case_dir / "result.json", row)
        rows.append(row)
        dump(out / "qualification.json", {"status": "running", "provenance": prov,
              "rows": rows, "completed_cases": len(rows)})
    all_qualified = len(rows) == 3 and all(row["numerical_qualified"] for row in rows)
    dump(out / "qualification.json", {"status": "complete", "provenance": prov,
          "rows": rows, "all_numerically_qualified": all_qualified,
          "totals": {key: sum(row[key] for row in rows) for key in
                     ("search_requests", "search_calculator_calls", "fresh_requests",
                      "fresh_calculator_calls", "total_requests", "total_calculator_calls")}})
    return 0 if all_qualified else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    if args.preflight:
        print(json.dumps(preflight(plan), indent=2))
        return 0
    if args.out is None:
        parser.error("--execute requires --out PATH")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    validate_plan(plan)
    prov = provenance(plan)
    shutil.copy2(PLAN_PATH, out / "effective-plan.json")
    (out / "source-snapshot").mkdir()
    shutil.copy2(__file__, out / "source-snapshot" / "run.py")
    for name in CORE_FILES:
        shutil.copy2(ROOT / name, out / "source-snapshot" / Path(name).name)
    dump(out / "provenance.json", prov)
    return execute(plan, prov, out)


if __name__ == "__main__":
    raise SystemExit(main())
