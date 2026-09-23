#!/usr/bin/env python3
"""Matched periodic TiO2 SSW / NativeLS runner; this file owns no plan or inputs."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
LEDGER_PATH = HERE.parent / "periodic-rotation-priority-20260923" / "ledger.py"


def _ledger():
    spec = importlib.util.spec_from_file_location("tio2_pair_ledger", LEDGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime():
    sys.path.insert(0, str(SOURCE))
    from pamssw.standalone import SSWConfig, run_ssw
    from pamssw.standalone.ls_native_reference import NativeLSSettings
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.native_ls import (TIO_BOND_ENERGIES, TIO_BOND_LENGTHS,
                                             initialize_native_ls)
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    return (SSWConfig, run_ssw, NativeLSSettings, LSPrequenchSettings,
            TIO_BOND_ENERGIES, TIO_BOND_LENGTHS, initialize_native_ls,
            RecoveredRotationSettings)


def _settings(plan, runtime):
    SSWConfig, _, NativeLSSettings, LSPrequenchSettings, bond_e, bond_l, _, Rotation = runtime
    config = SSWConfig(**plan["config"])
    rotation = Rotation(**plan["recovered_rotation"])
    values = dict(plan["native_ls"])
    pre = values.pop("prequench", None)
    if pre is not None:
        values["prequench"] = LSPrequenchSettings(**pre)
    # The release tables come from the frozen implementation, never user-tuned plan data.
    for key, expected in (("bond_energies", bond_e), ("bond_lengths", bond_l)):
        supplied = values.pop(key, None)
        if supplied is not None and supplied != expected:
            raise ValueError(f"plan {key} differs from frozen TIO table")
    ls = NativeLSSettings(bond_energies=bond_e, bond_lengths=bond_l, **values)
    return config, rotation, ls


def _verify_plan(plan, runtime, *, inputs=True):
    ledger = _ledger()
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"frozen source not found: {SOURCE}")
    planned_ledger = Path(plan["shared_ledger"]).resolve()
    if planned_ledger != LEDGER_PATH.resolve():
        raise ValueError("plan shared ledger path differs from the ledger actually imported")
    for rel, expected in plan["source_sha256"].items():
        relpath = Path(rel)
        path = HERE / relpath if relpath.parts[0] == "source" else SOURCE / relpath
        actual = ledger.sha256(path)
        if actual != expected:
            raise ValueError(f"frozen source hash mismatch: {rel}")
    if ledger.sha256(plan["shared_ledger"]) != plan["shared_ledger_sha256"]:
        raise ValueError("shared request-ledger source hash mismatch")
    config, rotation, ls = _settings(plan, runtime)
    if not isinstance(plan["seeds"], list) or not plan["seeds"]:
        raise ValueError("nonempty plan seeds list required")
    if plan["methods"] != ["ssw", "native_ls"]:
        raise ValueError("paired methods must be ssw and native_ls")
    if inputs:
        from ase.io import read
        for case in ("phase87", "phase139"):
            path = HERE / "inputs" / f"{case}.extxyz"
            if ledger.sha256(path) != plan["input_sha256"][case]:
                raise ValueError(f"input hash mismatch: {case}")
            atoms = read(path)
            if len(atoms) != 48 or not atoms.pbc.all() or not np.isfinite(atoms.positions).all():
                raise ValueError(f"invalid full-periodic TiO2 input: {case}")
    for key in ("search_cap", "outer_steps"):
        if isinstance(plan[key], bool) or int(plan[key]) <= 0:
            raise ValueError(f"positive integer {key} required")
    if float(plan["wall_per_arm_seconds"]) <= 0:
        raise ValueError("positive wall_per_arm_seconds required")
    return ledger, config, rotation, ls


def _preflight(plan, runtime):
    ledger, config, rotation, ls = _verify_plan(plan, runtime)
    from ase.io import read
    _, run_ssw, _, _, bond_e, bond_l, initialize_native_ls, _ = runtime
    rows = []

    class SentinelSurface:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0.0, np.zeros_like(atoms.positions)

    for case in ("phase87", "phase139"):
        atoms = read(HERE / "inputs" / f"{case}.extxyz")
        geometry = {}
        for mode in ("native-mic", "periodic-images"):
            initialized = initialize_native_ls(atoms, bond_energies=bond_e,
                bond_lengths=bond_l, bond_geometry=mode)
            geometry[mode] = int(initialized.bond_count)
            if geometry[mode] <= 0:
                raise ValueError(f"zero LS neighbors for {case}/{mode}")
        for method in plan["methods"]:
            surface = SentinelSurface()
            result = run_ssw(atoms.copy(), surface, steps=0, config=config,
                rng=np.random.default_rng(plan["seeds"][0]),
                recovered_rotation=rotation,
                ls=ls if method == "native_ls" else None)
            if result.status != "completed" or surface.requests < 1:
                raise RuntimeError(f"public run_ssw sentinel boundary failed: {case}/{method}")
            rows.append(dict(case=case, method=method, sentinel_evaluations=surface.requests,
                             native_ls_neighbor_counts=geometry,
                             result_status=result.status, model_loaded=False,
                             physical_pes_evaluated=False))
    ledger.dump(HERE / "preflight.json", dict(status="passed", checks=rows,
        source_commit=plan["source_commit"], input_sha256=plan["input_sha256"],
        warning="Sentinel force surface only; no MACE load, physical E/F or search."))


def _qualification_gate(plan):
    path = HERE / "qualification.json"
    gate = json.loads(path.read_text())
    for case in ("phase87", "phase139"):
        row = gate[case]
        if row.get("initial_converged") is not True or row.get("fresh_qualified") is not True:
            raise ValueError(f"qualification gate not passed for {case}")


def _execute(plan, runtime):
    if plan.get("status") != "ready_for_search":
        raise RuntimeError(f"search deferred by frozen plan status: {plan.get('status')!r}")
    ledger, config, rotation, ls = _verify_plan(plan, runtime)
    _qualification_gate(plan)
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator
    _, run_ssw, _, _, _, _, _, _ = runtime
    model = Path(plan["model"])
    if ledger.sha256(model) != plan["model_sha256"]:
        raise ValueError("model hash mismatch")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summary = []
    for case in ("phase87", "phase139"):
        input_atoms = read(HERE / "inputs" / f"{case}.extxyz")
        for seed in plan["seeds"]:
            for method in plan["methods"]:
                folder = HERE / f"{case}-seed{seed}-{method}"
                folder.mkdir(parents=True, exist_ok=False)
                calc = MACECalculator(model_paths=str(model), head=plan["head"],
                    device=plan["device"], default_dtype=plan["dtype"],
                    enable_cueq=False, enable_oeq=False)
                calc_counter = ledger.instrument_calculate(calc)
                surface = ledger.CountedSurface(calc, folder / "requests.jsonl",
                    int(plan["search_cap"]), float(plan["wall_per_arm_seconds"]))
                active_ls = ls if method == "native_ls" else None
                ledger.dump(folder / "effective-config.json", dict(case=case, seed=seed,
                    method=method, model=str(model), model_sha256=plan["model_sha256"],
                    source_commit=plan["source_commit"], input_sha256=plan["input_sha256"][case],
                    config=config, recovered_rotation=rotation, native_ls=active_ls,
                    search_cap=plan["search_cap"], wall_per_arm_seconds=plan["wall_per_arm_seconds"],
                    outer_steps=plan["outer_steps"]))
                started = time.monotonic()
                status, driver_status, error, result, checkpoint_requests = "exception", None, None, None, None
                try:
                    result = run_ssw(input_atoms.copy(), surface, steps=int(plan["outer_steps"]),
                        config=config, rng=np.random.default_rng(int(seed)),
                        recovered_rotation=rotation, ls=active_ls,
                        checkpoint_path=str(folder / "checkpoint.pkl"))
                    ledger.dump(folder / "result.json", result)
                    for record in result.records:
                        ledger.append(folder / "steps.jsonl", record)
                    expected = result.initial.evaluation_requests + sum(r.evaluation_requests for r in result.records)
                    if result.evaluation_requests != surface.requests or expected != surface.requests:
                        raise AssertionError("result, initial/step, and counted request totals disagree")
                    driver_status = result.status
                    status = ("budget_censored_" + surface.boundary if surface.boundary
                              else "numerical_failure" if driver_status == "evaluation_failed"
                              else driver_status)
                    frames = [m.atoms.copy() for m in result.minima]
                    for i, (frame, minimum) in enumerate(zip(frames, result.minima)):
                        frame.info["minimum_index"] = i
                        frame.info["energy_eV"] = float(minimum.energy)
                        frame.info["max_force_eV_A"] = float(minimum.max_force)
                    if frames:
                        write(folder / "minima.extxyz", frames)
                    best = min(result.minima, key=lambda m: m.energy)
                    initial_frame, best_frame = result.initial.atoms.copy(), best.atoms.copy()
                    initial_frame.info.update(energy_eV=float(result.initial.energy),
                                              fmax_eV_A=float(result.initial.max_force))
                    best_frame.info.update(energy_eV=float(best.energy), fmax_eV_A=float(best.max_force))
                    write(folder / "initial.extxyz", initial_frame)
                    write(folder / "best.extxyz", best_frame)
                except Exception as exc:
                    error = repr(exc)
                    failed_quench = getattr(exc, "result", None)
                    if failed_quench is not None:
                        ledger.dump(folder / "exception-result.json", failed_quench)
                        if getattr(failed_quench, "atoms", None) is not None:
                            from ase.io import write
                            write(folder / "failed-endpoint.extxyz", failed_quench.atoms)
                    checkpoint_path = folder / "checkpoint.pkl"
                    if checkpoint_path.is_file():
                        try:
                            from pamssw.standalone.paper_reference import load_ssw_checkpoint
                            saved = load_ssw_checkpoint(checkpoint_path)
                            ledger.dump(folder / "checkpoint-result.json", saved)
                            checkpoint_requests = int(saved.evaluation_requests)
                        except Exception as checkpoint_error:
                            checkpoint_requests = None
                            error += f"; checkpoint_read_error={checkpoint_error!r}"
                    else:
                        checkpoint_requests = None
                    if surface.boundary is not None:
                        status = "budget_censored_" + surface.boundary
                    elif failed_quench is not None or isinstance(exc, (ValueError, FloatingPointError, np.linalg.LinAlgError)):
                        status = "numerical_failure"
                checks, fresh_calls = _fresh_checks(result, model, plan, input_atoms, config, ledger) if result else ([], 0)
                if checks:
                    ledger.dump(folder / "fresh-checks.json", checks)
                elapsed = time.monotonic() - started
                row = dict(case=case, seed=int(seed), method=method, status=status,
                    driver_status=driver_status, error=error,
                    search_requests=surface.requests, expected_request_cap=int(plan["search_cap"]),
                    calculator_calls=calc_counter["calls"], budget_denials=surface.denials,
                    fresh_calculator_calls=fresh_calls, boundary=surface.boundary,
                    elapsed_seconds=elapsed, checkpoint_saved=(folder / "checkpoint.pkl").is_file(),
                    checkpoint_path=str(folder / "checkpoint.pkl"),
                    checkpoint_requests=checkpoint_requests, fresh_checks=checks)
                ledger.dump(folder / "summary.json", row)
                summary.append(row)
                ledger.dump(HERE / "summary.json", summary)
    return summary


def _fresh_checks(result, model, plan, input_atoms, config, ledger):
    checks = []
    # Independent calculator instance prevents reuse of search ASE results.
    from mace.calculators import MACECalculator
    fresh_calc = MACECalculator(model_paths=str(model), head=plan["head"],
        device=plan["device"], default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False)
    fresh_counter = ledger.instrument_calculate(fresh_calc)
    best = min(result.minima, key=lambda m: m.energy)
    candidates = [("initial", result.initial), ("best", best)]
    for label, minimum in candidates:
        atoms, expected_energy = minimum.atoms, minimum.energy
        try:
            work = atoms.copy(); work.calc = fresh_calc
            energy = float(work.get_potential_energy()); forces = np.asarray(work.get_forces(), float)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            finite = bool(np.isfinite(energy) and forces.shape == work.positions.shape
                          and np.isfinite(forces).all())
            cell_ok = bool(np.array_equal(work.cell.array, input_atoms.cell.array))
            pbc_ok = bool(np.array_equal(work.pbc, input_atoms.pbc))
            numbers_ok = bool(np.array_equal(work.numbers, input_atoms.numbers))
            energy_error = None if expected_energy is None else abs(energy-float(expected_energy))
            checks.append(dict(label=label, energy_eV=energy, expected_energy_eV=expected_energy,
                energy_error_eV=energy_error, fmax_eV_A=fmax, finite=finite,
                optimizer_converged=bool(minimum.converged), force_qualified=fmax <= config.fmax,
                cell_ok=cell_ok, pbc_ok=pbc_ok, numbers_ok=numbers_ok,
                qualified=bool(finite and minimum.converged and fmax <= config.fmax and cell_ok and pbc_ok and numbers_ok
                    and energy_error <= 1e-6)))
        except Exception as exc:
            checks.append(dict(label=label, qualified=False, error=repr(exc)))
        fresh_calc.reset()
    return checks, fresh_counter["calls"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="CPU-only structural/sentinel checks")
    mode.add_argument("--execute", action="store_true", help="run only after qualification.json gate")
    args = parser.parse_args()
    plan = json.loads((HERE / "plan.json").read_text())
    if args.execute and plan.get("status") != "ready_for_search":
        parser.error(f"search is deferred by plan status {plan.get('status')!r}")
    runtime = _load_runtime()
    if args.preflight:
        _preflight(plan, runtime)
    else:
        _execute(plan, runtime)


if __name__ == "__main__":
    main()
