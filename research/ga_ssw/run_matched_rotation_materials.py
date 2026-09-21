"""Matched fixed-cell OMAT-small SSW comparison for periodic materials.

This is a research runner around the existing public ``run_ssw`` entry point.
It does not invoke the native LASP program.  Use ``--execute`` only after the
prepared ``plan.json`` has been reviewed.
"""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ROOT / "research/ga_ssw/prospective/complex-vc-feasibility/inputs"
DEFAULT_CASES = ("aloh26", "brookite48")
SOLVERS = ("ritz", "dimer", "broyden-euclidean")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numbers") and hasattr(value, "positions"):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist(),
                "symbols": value.get_chemical_symbols()}
    if is_dataclass(value):
        return {field: serial(item) for field, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(item) for item in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


class BudgetSurface:
    """Count and persist every search E/F request, including failures/denials."""
    def __init__(self, surface, path, cap):
        self.surface, self.path, self.cap = surface, Path(path), int(cap)
        self.requests = 0
        self.denials = 0
        self.path.touch()

    def _log(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps(serial(row), allow_nan=False) + "\n")

    def evaluate(self, atoms):
        if self.requests >= self.cap:
            self.denials += 1
            self._log({"kind": "search_denial", "request": self.requests,
                       "reason": "search_cap"})
            raise RuntimeError("declared search E/F budget exhausted")
        self.requests += 1
        try:
            energy, forces = self.surface.evaluate(atoms)
            forces = np.asarray(forces, dtype=float)
            if not np.isfinite(energy) or forces.shape != atoms.positions.shape or not np.isfinite(forces).all():
                raise ValueError("nonfinite or shape-invalid E/F")
            self._log({"kind": "search", "request": self.requests, "atoms": atoms,
                       "energy": energy, "forces": forces,
                       "fmax": float(np.linalg.norm(forces, axis=1).max())})
            return energy, forces
        except Exception as error:
            self._log({"kind": "search_failure", "request": self.requests,
                       "atoms": atoms, "error": repr(error)})
            raise


def make_config(solver, rotation_exit_policy='force'):
    from pamssw.standalone import SSWConfig
    return SSWConfig(width=.1, rotation_bias=None, max_gaussians=25,
        temperature_K=150., fmax=.03, bias_fmax=.1, relax_steps=400,
        fd_step=.001, rotation_hvp=100, rotation_tol=.02,
        direction_sampling="global", rotation_solver=solver,
        cluster_frame="translation_only", quench_optimizer="safe-lbfgs-total",
        pre_rotation_hvp=5,rotation_exit_policy=rotation_exit_policy)


def run_one(out, case, seed, solver, model, device, steps, search_cap, validation_cap,
            input_path, search_calc, validation_calc, rotation_exit_policy='force'):
    from ase.io import read, write
    from pamssw.standalone import ASESurface, run_ssw, load_ssw_checkpoint
    from mace.calculators import MACECalculator

    folder = out / f"{case}-{solver}-seed{seed}"
    folder.mkdir()
    atoms = read(input_path)
    write(folder / "input.extxyz", atoms)
    cfg = make_config(solver,rotation_exit_policy)
    started = time.monotonic()
    surface = BudgetSurface(ASESurface(search_calc), folder / "evaluations.jsonl", search_cap)
    checkpoint = folder / "checkpoint.pkl"
    row = {"case": case, "seed": seed, "solver": solver, "status": "started",
           "config": asdict(cfg), "search_cap": search_cap, "validation_cap": validation_cap}
    result = None
    try:
        result = run_ssw(atoms, surface, steps=steps, config=cfg,
                         rng=np.random.default_rng(seed), checkpoint_path=checkpoint)
        row["status"] = result.status
    except Exception as error:
        row.update(status="exception", error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, 'result'):
            write_json(folder / 'initial-quench-failure.json', error.result)
        if checkpoint.exists():
            try:
                result = load_ssw_checkpoint(checkpoint)
                row["checkpoint_status"] = result.status
            except Exception as checkpoint_error:
                row["checkpoint_error"] = repr(checkpoint_error)
    if result is not None:
        write_json(folder / "result.json", result)
        minima = getattr(result, "minima", ())
        write(folder / "minima.extxyz", [minimum.atoms for minimum in minima])
        row.update(steps_completed=len(getattr(result, "records", ())),
                   statuses={status: sum(r.status == status for r in result.records)
                             for status in sorted({r.status for r in result.records})},
                   accepted=sum(bool(r.accepted) for r in result.records),
                   minima=len(minima))
        checks = []
        validation = BudgetSurface(ASESurface(validation_calc), folder / "validation.jsonl", validation_cap)
        for index, minimum in enumerate(minima):
            try:
                energy, forces = validation.evaluate(minimum.atoms)
                checks.append({"index": index, "energy": energy,
                    "energy_error": energy - minimum.energy,
                    "fmax": float(np.linalg.norm(forces, axis=1).max()),
                    "force_qualified": bool(np.linalg.norm(forces, axis=1).max() <= cfg.fmax),
                    "cell_unchanged": bool(np.array_equal(atoms.cell.array, minimum.atoms.cell.array)),
                    "numbers_unchanged": bool(np.array_equal(atoms.numbers, minimum.atoms.numbers))})
            except Exception as error:
                checks.append({"index": index, "error": repr(error)})
        write_json(folder / "validation.json", checks)
        row["validation_requests"] = validation.requests
        row["validation"] = checks
    row.update(search_requests=surface.requests, search_denials=surface.denials,
               seconds=time.monotonic() - started,
               rotation_failed=sum(r.status == "rotation_failed" for r in getattr(result, "records", ()))
               if result is not None else None)
    write_json(folder / "summary.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29])
    parser.add_argument('--solvers',nargs='+',choices=SOLVERS,default=list(SOLVERS))
    parser.add_argument('--rotation-exit-policy',choices=['force','force_or_budget'],default='force')
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--search-cap", type=int, default=6000)
    parser.add_argument("--validation-cap", type=int, default=101)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.search_cap < 1 or args.validation_cap < 1:
        parser.error("steps must be nonnegative; caps must be positive")
    if args.model is None and args.execute:
        model = Path(json.loads((args.outdir.expanduser().resolve() / "plan.json").read_text())["model"])
    elif args.model is not None:
        model = args.model.expanduser().resolve()
    else:
        parser.error("--model is required when preparing an outdir")
    if not model.exists():
        parser.error(f"model does not exist: {model}")
    out = args.outdir.expanduser().resolve()
    if args.execute:
        if not out.is_dir() or not (out / "plan.json").exists():
            parser.error("--execute requires an existing prepared outdir/plan.json")
        plan = json.loads((out / "plan.json").read_text())
        args.cases = plan["cases"]
        args.seeds = plan["seeds"]
        if plan.get("model_sha256") != hashlib.sha256(model.read_bytes()).hexdigest():
            parser.error("model hash differs from prepared plan")
        source = out / "source"
        expected_manifest = plan.get("source_manifest", {})
        actual_manifest = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(source.rglob("*.py"))}
        if actual_manifest != expected_manifest:
            parser.error("frozen source manifest differs from prepared plan")
        for case in args.cases:
            if not (out / "inputs" / f"{case}.extxyz").exists():
                parser.error(f"missing frozen input for case {case}: prepare output is incomplete")
        sys.path.insert(0, str(source))
        import pamssw
        assert Path(pamssw.__file__).resolve().is_relative_to(source)
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        write_json(out / 'execution.json', dict(device=(torch.cuda.get_device_name()
            if plan['device'].startswith('cuda') else plan['device']),
            scope='frozen source, inputs and model; matched numerical solver comparison'))
        from mace.calculators import MACECalculator
        search_calc = MACECalculator(model_paths=str(model), device=plan["device"], default_dtype="float64",
                                     enable_cueq=False, enable_oeq=False)
        validation_calc = MACECalculator(model_paths=str(model), device=plan["device"], default_dtype="float64",
                                          enable_cueq=False, enable_oeq=False)
    else:
        out.mkdir(parents=True, exist_ok=False)
        source = out / "source"
        shutil.copytree(ROOT / "pamssw", source / "pamssw",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        frozen_inputs = {}
        for case in args.cases:
            target = out / "inputs" / f"{case}.extxyz"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(INPUTS / f"{case}.extxyz", target)
            frozen_inputs[case] = str(target)
        plan = {"cases": args.cases, "seeds": args.seeds, "solvers": args.solvers,
                "steps": args.steps, "search_cap": args.search_cap,
                "validation_cap": args.validation_cap, "model": str(model),
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "device": args.device, "config": asdict(make_config("ritz",args.rotation_exit_policy)),
                "purpose": "matched fixed-cell independent public ASE SSW comparison; no native LASP parity",
                "budget_total_search_max": len(args.cases) * len(args.seeds) * len(args.solvers) * args.search_cap,
                "inputs": frozen_inputs}
        manifest = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(source.rglob("*.py"))}
        plan["source_manifest"] = manifest
        write_json(out / "plan.json", plan)
        shutil.copy2(Path(__file__), out / "runner.py")
        return
    rows = []
    for case in args.cases:
        for seed in args.seeds:
            for solver in plan['solvers']:
                rows.append(run_one(out, case, seed, solver, model, plan["device"],
                                    plan["steps"], plan["search_cap"], plan["validation_cap"],
                                    out / "inputs" / f"{case}.extxyz", search_calc, validation_calc,
                                    plan['config'].get('rotation_exit_policy','force')))
                write_json(out / "summary.json", rows)


if __name__ == "__main__":
    main()
