#!/usr/bin/env python3
"""Prepared C60 recovered-rotation SSW runner; execution is opt-in."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "source"))
from ledger import CountedSurface, dump, instrument_calculate, sha256, _jsonable  # noqa: E402
from ase.io import read, write  # noqa: E402
from pamssw.standalone import (  # noqa: E402
    LSPrequenchSettings, NativeLSSettings, NativeMCSettings,
    RecoveredRotationSettings, SSWConfig,
    load_ssw_checkpoint, run_ssw,
)
from validator import conditional_candidate  # noqa: E402


def check_frozen(plan):
    for rel, expected in plan["source_sha256"].items():
        assert sha256(HERE / rel) == expected, rel
    for case, expected in plan["input_sha256"].items():
        assert sha256(HERE / "inputs" / f"{case}.traj") == expected, case
    for rel, expected in plan["harness_sha256"].items():
        assert sha256(HERE / rel) == expected, rel
    assert sha256(plan["model"]) == plan["model_sha256"], "model changed"


def runtime_versions():
    from importlib import metadata
    versions = {}
    for name in ("numpy", "scipy", "ase", "mace-torch", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable, "packages": versions}


def save_failed_initial(result, folder):
    """Retain a failed initial quench without letting nonfinite fields hide it."""
    saved = {"json": False, "traj": False, "serialization_fallback": False}
    try:
        dump(folder / "failed-initial.json", result)
        saved["json"] = True
    except (TypeError, ValueError) as error:
        def sanitize(value):
            if isinstance(value,float):return value if np.isfinite(value) else None
            if isinstance(value,dict):return {str(k):sanitize(v) for k,v in value.items()}
            if isinstance(value,list):return [sanitize(v) for v in value]
            return value
        fallback = sanitize(_jsonable(result))
        fallback["serialization_fallback_reason"] = repr(error)
        (folder / "failed-initial.json").write_text(json.dumps(fallback, indent=2) + "\n")
        saved.update(json=True, serialization_fallback=True)
    try:
        write(folder / "failed-initial.traj", result.atoms)
        saved["traj"] = True
    except Exception as error:
        saved["traj_error"] = repr(error)
    return saved


def fresh_check(minimum, original, config, calculator, folder, label):
    row = {"label": label, "status": "started"}
    write(folder / f"{label}.traj", minimum.atoms)
    try:
        calculator.reset()
        work = minimum.atoms.copy()
        work.calc = calculator
        energy = float(work.get_potential_energy())
        forces = np.asarray(work.get_forces(), float)
        fmax = float(np.linalg.norm(forces, axis=1).max())
        cell_ok = bool(np.array_equal(work.cell.array, original.cell.array))
        pbc_ok = bool(np.array_equal(work.pbc, original.pbc))
        numbers_ok = bool(np.array_equal(work.numbers, original.numbers))
        error = energy - minimum.energy
        row.update(status="completed", energy_eV=energy, fmax_eV_A=fmax,
                   energy_error_eV=error, cell_unchanged=cell_ok,
                   pbc_unchanged=pbc_ok, numbers_unchanged=numbers_ok,
                   numerical_qualified=bool(
                       np.isfinite(energy) and np.isfinite(forces).all()
                       and fmax <= config.fmax and abs(error) <= 1e-6
                       and cell_ok and pbc_ok and numbers_ok and minimum.converged))
    except Exception as error:
        row.update(status="error", error=repr(error), numerical_qualified=False)
    (folder / f"{label}-fresh.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def _pair_map(values):
    return {tuple(int(part) for part in key.split(",")): value
            for key, value in values.items()}


def make_ls(plan):
    spec = plan["native_ls"]
    return NativeLSSettings(
        bond_energies=_pair_map(spec["bond_energies"]),
        bond_lengths=_pair_map(spec["bond_lengths"]),
        scale=spec["scale"], amp_c=spec["amp_c"],
        length_tolerance=spec["length_tolerance"],
        target_mev_per_atom=spec["target_mev_per_atom"], eta=spec["eta"],
        max_change=spec["max_change"], frequency=spec["frequency"],
        presteps=spec["presteps"], cycle=spec["cycle"], ratio=spec["ratio"],
        lselfadapt=spec["lselfadapt"], bond_geometry=spec["bond_geometry"],
        prequench=LSPrequenchSettings(**spec["prequench"]),
    )


def run_case(plan, case):
    seed = plan["seeds"][case]
    folder = HERE / f"{case}-seed{seed}"
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    ssw_config = dict(plan["ssw_config"])
    config = SSWConfig(**ssw_config)
    rotation = RecoveredRotationSettings(**plan["recovered_rotation"])
    ls = make_ls(plan)
    atoms = read(HERE / "inputs" / f"{case}.traj")
    write(folder / "input.traj", atoms)
    (folder / "effective-config.json").write_text(json.dumps({
        "arm": "recovered_rotation_native_ls", "case": case, "seed": seed,
        "ssw_config": ssw_config, "recovered_rotation": plan["recovered_rotation"],
        "native_ls": plan["native_ls"],
        "native_mc": plan["native_mc"],
        "runtime_versions": runtime_versions(),
    }, indent=2) + "\n")

    import torch
    from mace.calculators import MACECalculator
    torch.set_num_threads(1)
    torch.manual_seed(plan["runtime"]["torch_manual_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = dict(model_paths=plan["model"], head=plan["head"],
                  device=plan["device"], default_dtype=plan["dtype"],
                  enable_cueq=False, enable_oeq=False)
    calculator = MACECalculator(**kwargs)
    fresh_calculator = MACECalculator(**kwargs)
    counter = instrument_calculate(calculator)
    fresh_counter = instrument_calculate(fresh_calculator)
    calculator.reset()
    before = counter["calls"]
    surface = CountedSurface(calculator, folder / "requests.jsonl",
                             plan["search_cap"], plan["wall_seconds"])
    row = {"arm": "recovered_rotation_native_ls", "case": case, "seed": seed,
           "status": "started",
           "runtime_versions": runtime_versions()}
    checkpoint = folder / "checkpoint.pkl"
    result = None
    started = time.monotonic()
    try:
        result = run_ssw(
            atoms.copy(), surface, steps=plan["outer_steps"], config=config,
            rng=np.random.default_rng(seed),
            mc=NativeMCSettings(plan["native_mc"]["energy_tol_eV"],
                                plan["native_mc"]["maxtrap"]),
            checkpoint_path=checkpoint, recovered_rotation=rotation, ls=ls)
        assert result.evaluation_requests == surface.requests
        row.update(status=result.status, records=len(result.records),
                   record_statuses=dict(Counter(r.status for r in result.records)))
    except Exception as error:
        row.update(status="exception", error=repr(error),
                   traceback=traceback.format_exc())
        failed_initial = getattr(error, "result", None)
        if failed_initial is not None:
            row["failed_initial_saved"] = save_failed_initial(failed_initial, folder)
        if checkpoint.exists():
            try:
                result = load_ssw_checkpoint(checkpoint)
                row["recovered_checkpoint"] = True
            except Exception as checkpoint_error:
                row["checkpoint_error"] = repr(checkpoint_error)

    if result is not None:
        dump(folder / "result.json", result)
        minima = list(result.minima)
        write(folder / "minima.traj", [m.atoms for m in minima])
    else:
        minima = []
    checks = []
    initial_best_checks = 0
    conditional_checks = 0
    cage_diagnostics = []
    if minima:
        checks.append(fresh_check(minima[0], atoms, config, fresh_calculator,
                                  folder, "initial"))
        best = min(minima, key=lambda value: value.energy)
        checks.append(fresh_check(best, atoms, config, fresh_calculator,
                                  folder, "best"))
        initial_best_checks = 2
        candidate_index, cage_diagnostics = conditional_candidate(minima)
        if candidate_index is not None:
            checks.append(fresh_check(minima[candidate_index], atoms, config, fresh_calculator,
                                      folder, "first_cage"))
            conditional_checks = 1
    row.update(search_requests=surface.requests,
               search_calculator_calls=counter["calls"] - before,
               denials=surface.denials, boundary=surface.boundary,
               fresh_requests=len(checks),
               fresh_initial_best_requests=initial_best_checks,
               fresh_conditional_requests=conditional_checks,
               conditional_cage_index=candidate_index if minima else None,
               fresh_calculator_calls=fresh_counter["calls"], minima=len(minima),
               fresh=checks, cage_diagnostics=cage_diagnostics,
               elapsed_seconds=time.monotonic() - started,
               execution_class=("budget_censored" if surface.boundary else row["status"]))
    assert row["search_requests"] <= plan["search_cap"]
    assert row["fresh_initial_best_requests"] <= plan["fresh_per_case"]
    assert row["fresh_conditional_requests"] <= plan["conditional_first_cage_extra_per_case"]
    assert row["fresh_requests"] <= plan["fresh_per_case"] + plan["conditional_first_cage_extra_per_case"]
    (folder / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case", choices=("c60_17093", "c60_17094"))
    args = parser.parse_args()
    if not args.execute:
        print("prepared_not_executed")
        return
    if args.case is None:
        parser.error("--case is required; each job owns one case directory")
    plan = json.loads((HERE / "plan.json").read_text())
    check_frozen(plan)
    run_case(plan, args.case)


if __name__ == "__main__":
    main()
