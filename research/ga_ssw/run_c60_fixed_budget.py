#!/usr/bin/env python3
"""Prepare/execute the bounded random-C60 native-MC development runner.

Preparation freezes the local PAM source and inputs.  Execution imports only
that frozen source, writes a fresh case directory, and records paid requests
plus independent fresh-force qualification.  This is a development budget
curve, not an independent success-rate or native-trajectory claim.
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
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "research/ga_ssw/evidence/c60-random-inputs-development-20260917"
STRICT_PLAN = ROOT / "research/ga_ssw/evidence/ssw-allocation-factorial-20260918/c60/plan.json"
MH1_PLAN = ROOT / "research/ga_ssw/evidence/c60-mh1-qualification-20260919/plan.json"
DEFAULT_OUT = ROOT / "research/ga_ssw/evidence/c60-native-mc-fixed-budget-20260918"

BACKEND_PLAN_SOURCES = {
    "omat-small": STRICT_PLAN,
    "mh1-omol": MH1_PLAN,
}


def serial(value):
    import numpy as np
    from ase import Atoms
    if isinstance(value, Atoms):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if hasattr(value, "__dict__"): return serial(vars(value))
    return value


def dump(path: Path, value, *, exclusive=False):
    with path.open("x" if exclusive else "w") as handle:
        json.dump(serial(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: Path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(serial(value), allow_nan=False) + "\n")


def backend_from_frozen_plan(name):
    source = BACKEND_PLAN_SOURCES[name]
    frozen = json.loads(source.read_text())
    raw = frozen.get("backend") or {}
    model = frozen.get("model") or raw.get("model")
    expected_sha = frozen.get("model_sha256") or raw.get("model_sha256")
    head = raw.get("head", frozen.get("head"))
    device = raw.get("device") or frozen.get("device") or "cuda"
    dtype = raw.get("dtype") or raw.get("default_dtype") or frozen.get("dtype") or "float64"
    if not model or not expected_sha:
        raise ValueError(f"frozen backend plan lacks model/hash: {source}")
    if name == "mh1-omol" and head != "omol":
        raise ValueError(f"frozen MH1 backend plan must select head='omol': {source}")
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"backend model is missing: {model_path}")
    actual_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"backend model hash mismatch for {name}: expected {expected_sha}, got {actual_sha}")
    return {"name": name, "requested": raw.get("requested", raw.get("name", name)),
            "model": str(model_path), "model_sha256": actual_sha, "head": head,
            "device": device, "dtype": dtype, "source_plan": str(source)}


def plan_from_sources(variant="ssw", backend_name="omat-small"):
    if variant not in ("ssw", "native-ls", "native-height"):
        raise ValueError(f"unknown variant {variant!r}")
    strict = json.loads(STRICT_PLAN.read_text())
    plan = {
        "cases": ["c60_17093", "c60_17094"],
        "seeds": {"c60_17093": 17093, "c60_17094": 17094},
        "inputs": {case: f"inputs/{case}.traj" for case in ("c60_17093", "c60_17094")},
        "config": strict["configs"]["c60_17093"],
        "steps": 1000,
        "request_cap": 60000,
        "wall_seconds": 3600,
        "fresh_minimum_limit": 1001,
        "fresh_energy_tolerance_eV": 1e-6,
        "backend": backend_from_frozen_plan(backend_name),
        "native_mc": {"energy_tol_eV": 0.1, "maxtrap": 99999},
        "variant": variant,
        "runtime": {"torch_manual_seed": 0, "torch_deterministic_algorithms": True,
                    "torch_num_threads": 1, "tf32": False,
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "scope": "development budget curve; no independent success-rate or parameter-alignment claim",
    }
    if variant == "native-ls":
        plan["native_ls"] = {
            "bond_energies": {"6,6": 3.4468400478363037},
            "bond_lengths": {"6,6": 1.5399999618530273},
            "scale": 5.0,
            "amp_c": 2.0,
            "length_tolerance": 0.1,
            "target_mev_per_atom": 20.0,
            "eta": 0.005,
            "max_change": 0.01,
            "frequency": 10,
            "presteps": 100,
            "cycle": 100,
            "ratio": 1.100000023841858,
            "lselfadapt": True,
            "bond_geometry": "native-mic",
            "prequench": {"fmax": 0.1, "steps": 50,
                          "exit_policy": "force_or_step_limit"},
            "parameter_source": (
                "C-C release lookup in pamssw/standalone/native_ls.py; "
                "recovered-materials development LS protocol .1/50; "
                "not C60 fitting and not native stage-exit parity"
            ),
        }
    else:
        plan["native_ls"] = None
    if variant == "native-height":
        plan["native_height"] = {
            "initial_weight": 0.5,
            "negative_weight": 0.2,
            "level": 1,
            "max_weight": 10.0,
            "growth_step": 1.0,
            "growth_scale": 2.0,
            "parameter_source": (
                "current native allkeys SSW settings; stage-frozen and "
                "single-count Python repair, not complete native addgaussian parity"
            ),
        }
    else:
        plan["native_height"] = None
    return plan


def prepare(out: Path, variant="ssw", backend_name="omat-small"):
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True)
    source = out / "source"
    shutil.copytree(ROOT / "pamssw", source / "pamssw",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(__file__, out / "runner.py")
    shutil.copy2(ROOT / "research/ga_ssw/run_public_broyden_ssw.py", out / "ledger_helpers.py")
    (out / "inputs").mkdir()
    plan = plan_from_sources(variant, backend_name)
    hashes = {}
    for case, relative in plan["inputs"].items():
        from ase.io import read, write
        original = ROOT / "research/ga_ssw/evidence/c60-random-native-development-20260917" / f"seed{case.split('_')[1]}" / "input.arc"
        target = out / relative
        atoms = read(original, format="dmol-arc")
        atoms.pbc = False  # external MACE callback uses isolated clusters
        write(target, atoms)
        hashes[case] = {"source": str(original), "prepared": str(target),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    plan["source_snapshot"] = "source"
    plan["runner_snapshot"] = "runner.py"
    dump(out / "plan.json", plan, exclusive=True)
    dump(out / "manifest.json", {
        "strict_plan": str(STRICT_PLAN), "input_hashes": hashes,
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.rglob("*.py")},
    }, exclusive=True)
    print(json.dumps({"prepared": str(out), "cases": plan["cases"]}, indent=2))


def execute(out: Path, selected_case: str | None):
    if not out.is_dir() or not (out / "plan.json").is_file():
        raise FileNotFoundError("run --prepare first in a new evidence directory")
    source = out / "source"
    if not source.is_dir():
        raise FileNotFoundError("prepared frozen source is missing")
    plan = json.loads((out / "plan.json").read_text())
    cases = plan["cases"] if selected_case is None else [selected_case]
    if selected_case is not None and selected_case not in plan["cases"]:
        raise ValueError(f"unknown case {selected_case!r}")
    sys.path[:0] = [str(source), str(out)]
    from ledger_helpers import CountedSurface
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", plan["runtime"]["CUBLAS_WORKSPACE_CONFIG"])
    import numpy as np
    import torch
    torch.set_num_threads(plan["runtime"]["torch_num_threads"])
    torch.manual_seed(plan["runtime"]["torch_manual_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import (ASESurface, ConservativeNativeHeightPolicy,
                                   NativeMCSettings, SSWConfig, run_ssw)
    from pamssw.standalone import LSPrequenchSettings, NativeLSSettings

    backend = plan.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("prepared plan lacks backend metadata")
    model = Path(backend["model"])
    if not model.is_file():
        raise FileNotFoundError(f"backend model is missing: {model}")
    actual_sha = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual_sha != backend["model_sha256"]:
        raise RuntimeError(
            f"backend model hash mismatch: expected {backend['model_sha256']}, got {actual_sha}")
    config = SSWConfig(**plan["config"])
    mc = NativeMCSettings(plan["native_mc"]["energy_tol_eV"], plan["native_mc"]["maxtrap"])
    ls = None
    if plan.get("native_ls") is not None:
        spec = plan["native_ls"]
        def pair_table(values):
            return {tuple(int(part) for part in key.split(",")): value
                    for key, value in values.items()}
        pq = spec["prequench"]
        ls = NativeLSSettings(
            bond_energies=pair_table(spec["bond_energies"]),
            bond_lengths=pair_table(spec["bond_lengths"]),
            scale=spec["scale"], amp_c=spec["amp_c"],
            length_tolerance=spec["length_tolerance"],
            target_mev_per_atom=spec["target_mev_per_atom"],
            eta=spec["eta"], max_change=spec["max_change"],
            frequency=spec["frequency"], presteps=spec["presteps"],
            cycle=spec["cycle"], ratio=spec["ratio"],
            lselfadapt=spec["lselfadapt"],
            bond_geometry=spec["bond_geometry"],
            prequench=LSPrequenchSettings(**pq),
        )
    height_policy = None
    if plan.get("native_height") is not None:
        height_policy = ConservativeNativeHeightPolicy(**{
            key: value for key, value in plan["native_height"].items()
            if key != "parameter_source"
        })
    calculator_kwargs = {
        "model_paths": str(model), "device": backend["device"],
        "default_dtype": backend["dtype"], "enable_cueq": False, "enable_oeq": False,
    }
    if backend.get("head") is not None:
        calculator_kwargs["head"] = backend["head"]

    def calculator():
        return MACECalculator(**calculator_kwargs)

    rows = []
    for case in cases:
        case_dir = out / case
        if case_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing case directory: {case_dir}")
        case_dir.mkdir()
        ledger = case_dir / "requests.jsonl"
        ledger.touch()
        atoms = read(out / plan["inputs"][case])
        search = CountedSurface(calculator(), ledger, cap=plan["request_cap"], wall=plan["wall_seconds"])
        fresh = ASESurface(calculator())
        checkpoint_path = case_dir / "checkpoint.pkl"
        row = {"case": case, "seed": plan["seeds"][case], "status": "started"}
        started = time.monotonic()
        try:
            result = run_ssw(atoms.copy(), search, steps=plan["steps"], config=config,
                             rng=np.random.default_rng(plan["seeds"][case]), ls=ls,
                             height_policy=height_policy, mc=mc,
                             checkpoint_path=checkpoint_path)
            dump(case_dir / "result.json", result, exclusive=True)
            decisions = [{"record": i, "mc_telemetry": r.mc_telemetry}
                         for i, r in enumerate(result.records) if r.mc_telemetry is not None]
            dump(case_dir / "mc-decisions.json", decisions, exclusive=True)
            checks = []
            for i, minimum in enumerate(result.minima[:plan["fresh_minimum_limit"]]):
                try:
                    energy, forces = fresh.evaluate(minimum.atoms)
                    fmax = float(np.linalg.norm(forces, axis=1).max())
                    composition = bool(np.array_equal(minimum.atoms.numbers, atoms.numbers))
                    pbc = bool(np.array_equal(minimum.atoms.pbc, atoms.pbc))
                    cell = bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array))
                    agreement = abs(energy - minimum.energy) <= plan["fresh_energy_tolerance_eV"]
                    checks.append({"index": i, "energy_eV": energy, "energy_error_eV": energy - minimum.energy,
                                   "fmax_eV_per_A": fmax,
                                   "qualified": bool(np.isfinite(energy) and np.isfinite(fmax) and fmax <= config.fmax
                                                      and agreement and composition and pbc and cell),
                                   "energy_agrees": bool(agreement), "composition_match": composition,
                                   "pbc_match": pbc, "cell_unchanged": cell})
                except Exception as error:
                    checks.append({"index": i, "qualified": False, "error": repr(error)})
            dump(case_dir / "fresh-qualification.json", checks, exclusive=True)
            row.update(status=result.status, minima=len(result.minima), records=len(result.records),
                       record_statuses=dict(Counter(r.status for r in result.records)),
                       mc_decisions=len(decisions), fresh_checks=checks)
        except Exception as error:
            row.update(status="exception", error=repr(error))
        row.update(search_requests=search.requests, denials=search.denials,
                   boundary=search.boundary, fresh_requests=fresh.requests,
                   elapsed_seconds=time.monotonic() - started)
        dump(case_dir / "summary.json", row, exclusive=True)
        rows.append(row)
        completed = [json.loads(path.read_text()) for path in sorted(out.glob("c60_*/summary.json"))]
        dump(out / "summary.json", completed)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--case", choices=("c60_17093", "c60_17094"))
    parser.add_argument("--variant", choices=("ssw", "native-ls", "native-height"),
                        help="preparation variant; execution uses the prepared plan")
    parser.add_argument("--backend", choices=tuple(BACKEND_PLAN_SOURCES),
                        help="backend selected while preparing; execution uses the prepared plan")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        if args.case is not None:
            parser.error("--case is valid only with --execute")
        prepare(output, args.variant or "ssw", args.backend or "omat-small")
    else:
        if args.backend is not None:
            parser.error("--backend is valid only with --prepare")
        execute(output, args.case)


if __name__ == "__main__":
    main()
