#!/usr/bin/env python3
"""Prepare or execute the bounded native-MC material harness.

Execution is deliberately separate from preparation.  The runner copies the
current PAM source only during ``--prepare``, prepends
that frozen source before PAM imports, and writes every ledger/result
exclusively into a new evidence directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PARENT_C60 = ROOT / "research/ga_ssw/evidence/ssw-allocation-factorial-20260918/c60"
PARENT_CONTROLS = ROOT / "research/ga_ssw/evidence/ssw-allocation-factorial-20260918/controls"
DEFAULT_OUT = ROOT / "research/ga_ssw/evidence/native-mc-materials-20260918"
MODEL = "/home/gengjianrui/.cache/mace/mace-omat-0-small.model"
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"


def serial(value):
    import numpy as np
    from ase import Atoms

    if isinstance(value, Atoms):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if hasattr(value, "__dict__"):
        return serial(vars(value))
    return value


def dump(path: Path, value, *, exclusive: bool = False):
    mode = "x" if exclusive else "w"
    with path.open(mode) as handle:
        json.dump(serial(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: Path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(serial(value), allow_nan=False) + "\n")


def plan_from_parent() -> dict:
    c60 = json.loads((PARENT_C60 / "plan.json").read_text())
    controls = json.loads((PARENT_CONTROLS / "plan.json").read_text())
    cases = ["c60_17093", "c60_17094", "cu55", "water15"]
    configs = {case: (c60["configs"][case] if case.startswith("c60_") else controls["configs"][case]) for case in cases}
    seeds = {case: c60["seeds"][case] for case in cases}
    inputs = {
        case: {"original_path": str((PARENT_C60 if case.startswith("c60_") else PARENT_CONTROLS) / "inputs" / f"{case}.traj"),
               "prepared_path": f"inputs/{case}.traj"}
        for case in cases
    }
    return {
        "cases": cases,
        "seeds": seeds,
        "inputs": inputs,
        "configs": configs,
        "steps": 4,
        "request_cap": 6000,
        "wall_seconds": 600,
        "fresh_minimum_limit": 5,
        "fresh_energy_tolerance_eV": 1e-6,
        "backend": {"name": "MACE-OMAT-0-small", "model": MODEL, "model_sha256": MODEL_SHA256,
                     "device": "cuda", "dtype": "float64"},
        "native_mc": {"energy_tol_eV": 0.1, "maxtrap": 99999,
                       "source_basis": "actual C60 allkeys maxtrap; inherited T=150 K"},
        "runtime": {"torch_manual_seed": 0, "torch_deterministic_algorithms": True,
                    "torch_num_threads": 1, "tf32": False,
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "scope": "wiring harness only; no ranking or native end-to-end claim",
    }


def prepare(out: Path):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    plan = plan_from_parent()
    source = out / "source"
    shutil.copytree(ROOT / "pamssw", source / "pamssw",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, out / "runner.py")
    shutil.copy2(PARENT_C60 / "ledger_helpers.py", out / "ledger_helpers.py")
    (out / "inputs").mkdir()
    input_hashes = {}
    for case, paths in plan["inputs"].items():
        original = Path(paths["original_path"])
        prepared = out / paths["prepared_path"]
        shutil.copy2(original, prepared)
        input_hashes[case] = {"original_path": str(original), "prepared_path": str(prepared),
                              "sha256": hashlib.sha256(prepared.read_bytes()).hexdigest()}
    plan["source_snapshot"] = "source"
    plan["helper_snapshot"] = "ledger_helpers.py"
    dump(out / "plan.json", plan, exclusive=True)
    dump(out / "parent-manifest.json", {
        "c60_plan": str(PARENT_C60 / "plan.json"),
        "controls_plan": str(PARENT_CONTROLS / "plan.json"),
        "input_sha256": input_hashes,
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.rglob("*.py")},
        "helper_sha256": hashlib.sha256((out / "ledger_helpers.py").read_bytes()).hexdigest(),
    }, exclusive=True)
    print(json.dumps({"prepared": str(out), "cases": plan["cases"]}, indent=2))


def execute(out: Path):
    if not out.is_dir() or not (out / "plan.json").exists():
        raise FileNotFoundError("run --prepare first in a new evidence directory")
    source = out / "source"
    helper = out / "ledger_helpers.py"
    if not source.is_dir() or not helper.is_file():
        raise FileNotFoundError("prepared source/helper snapshot is missing")
    sys.path[:0] = [str(source), str(out)]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import ASESurface, NativeMCSettings, SSWConfig, run_ssw
    from ledger_helpers import CountedSurface

    plan = json.loads((out / "plan.json").read_text())
    all_rows = []
    for case in plan["cases"]:
        case_dir = out / case
        case_dir.mkdir()
        atoms = read(out / plan["inputs"][case]["prepared_path"])
        base = SSWConfig(**plan["configs"][case])
        mc = NativeMCSettings(energy_tol=plan["native_mc"]["energy_tol_eV"],
                              maxtrap=plan["native_mc"]["maxtrap"])

        def calculator():
            return MACECalculator(model_paths=plan["backend"]["model"], device="cuda",
                                  default_dtype="float64", enable_cueq=False, enable_oeq=False)

        ledger = case_dir / "requests.jsonl"
        ledger.touch()
        search = CountedSurface(calculator(), ledger, cap=plan["request_cap"], wall=plan["wall_seconds"])
        fresh = ASESurface(calculator())
        row = {"case": case, "seed": plan["seeds"][case], "status": "started"}
        started = time.monotonic()
        try:
            result = run_ssw(atoms.copy(), search, steps=plan["steps"], config=base,
                             rng=np.random.default_rng(plan["seeds"][case]), mc=mc)
            dump(case_dir / "result.json", result, exclusive=True)
            decisions = []
            for index, record in enumerate(result.records):
                if record.mc_telemetry is not None:
                    decisions.append({"record": index, "mc_telemetry": record.mc_telemetry})
            dump(case_dir / "mc-decisions.json", decisions, exclusive=True)
            checks = []
            for index, minimum in enumerate(result.minima[:plan["fresh_minimum_limit"]]):
                try:
                    energy, forces = fresh.evaluate(minimum.atoms)
                    fmax = float(np.linalg.norm(forces, axis=1).max())
                    composition_match = bool(np.array_equal(minimum.atoms.numbers, atoms.numbers))
                    pbc_match = bool(np.array_equal(minimum.atoms.pbc, atoms.pbc))
                    cell_unchanged = bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array))
                    energy_finite = bool(np.isfinite(energy))
                    energy_error_finite = bool(np.isfinite(energy - minimum.energy))
                    energy_agrees = abs(energy - minimum.energy) <= plan['fresh_energy_tolerance_eV']
                    checks.append({"index": index, "energy_eV": energy,
                                   "energy_error_eV": energy - minimum.energy,
                                   "fmax_eV_per_A": fmax,
                                   "qualified": bool(fmax <= base.fmax and energy_finite and energy_error_finite and energy_agrees
                                                      and composition_match and pbc_match and cell_unchanged),
                                   "energy_finite": energy_finite,
                                   "energy_error_finite": energy_error_finite,
                                   "energy_agrees": bool(energy_agrees),
                                   "composition_match": composition_match,
                                   "pbc_match": pbc_match,
                                   "cell_unchanged": cell_unchanged})
                except Exception as error:
                    checks.append({"index": index, "qualified": False, "error": repr(error)})
            dump(case_dir / "fresh-qualification.json", checks, exclusive=True)
            row.update(status=result.status, minima=len(result.minima),
                       records=len(result.records), record_statuses=dict(Counter(r.status for r in result.records)),
                       mc_decisions=len(decisions), fresh_checks=checks)
        except Exception as error:
            row.update(status="exception", error=repr(error))
        row.update(search_requests=search.requests, denials=search.denials,
                   boundary=search.boundary, fresh_requests=fresh.requests,
                   elapsed_seconds=time.monotonic() - started)
        dump(case_dir / "summary.json", row, exclusive=True)
        all_rows.append(row)
    dump(out / "summary.json", all_rows, exclusive=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.prepare == args.execute:
        parser.error("choose exactly one of --prepare or --execute")
    (prepare if args.prepare else execute)(args.output.resolve())


if __name__ == "__main__":
    main()
