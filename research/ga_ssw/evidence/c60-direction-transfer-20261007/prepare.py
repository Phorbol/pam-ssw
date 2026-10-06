#!/usr/bin/env python3
"""Freeze two prospective C60 starts and qualify them under the frozen MH-1 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PROTOCOL = HERE / "protocol.md"
REFERENCE = ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-1/final.extxyz"
VALIDATOR = ROOT / "research/ga_ssw/evidence/c60-long-budget-20260924/ssw-17101/validator.py"
RUNNER = ROOT / "research/ga_ssw/c60_long_budget.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
SEEDS = (26100781, 26100782)
REFERENCE_ENERGY = -62215.393370790625
INPUT_CAP, INPUT_SECONDS, PROCESS_SECONDS = 3000, 100.0, 240.0
SEARCH_CAP, SEARCH_SECONDS, FRESH_CAP = 60000, 4800, 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def protocol_config():
    return {
        "width": .6, "rotation_bias": 1., "max_gaussians": 12,
        "temperature_K": 150., "fmax": .03, "relax_steps": 1000,
        "fd_step": .001, "rotation_hvp": 39, "rotation_tol": .02,
        "forward_force": .1, "direction_sampling": "global",
        "rotation_solver": "broyden-euclidean", "cluster_frame": "direction_only",
        "quench_optimizer": "safe-lbfgs-total", "lbfgs_memory": 500,
        "bias_stage_steps": None, "bias_fmax": .1, "pre_rotation_hvp": None,
        "rotation_exit_policy": "force_or_budget",
    }


def rotation_settings():
    return {"pre_rotmax": 5, "rotmax": 15, "pre_ftol": .2,
            "ftol": .02, "metric": "euclidean", "max_force_calls": 40}


def direction_settings():
    return {"ratio_local": 50, "local_probability": .5,
            "group_threshold": .5, "pre_rotmax": 5, "rotmax": 15,
            "pre_ftol": .2, "ftol": .02, "metric": "euclidean",
            "max_force_calls": 40, "c1_radius_policy": "per_atom",
            "startup_order": "legacy", "geometry": "nonperiodic"}


def generate_input(seed: int):
    import numpy as np
    from ase import Atoms

    rng = np.random.default_rng(seed)
    accepted = []
    trials = 0
    while len(accepted) < 60:
        trials += 1
        if trials > 100000:
            raise RuntimeError(f"seed {seed} exceeded the 100000-trial initialization cap")
        point = rng.uniform(-5., 5., 3)
        if np.linalg.norm(point) > 5.:
            continue
        if accepted and np.min(np.linalg.norm(np.asarray(accepted) - point, axis=1)) < 1.:
            continue
        accepted.append(point)
    positions = np.round(np.asarray(accepted) + 25., 10)
    atoms = Atoms("C60", positions=positions, cell=[50., 50., 50.], pbc=False)
    return atoms, trials


def core_sources():
    return sorted((ROOT / "pamssw").rglob("*.py"))


def write_inputs(out: Path):
    from ase.io import read, write
    (out / "inputs").mkdir(parents=True, exist_ok=False)
    generated = []
    for seed in SEEDS:
        atoms, trials = generate_input(seed)
        path = out / "inputs" / f"seed-{seed}.extxyz"
        write(path, atoms)
        atoms = read(path)
        item = {"seed": seed, "trials": trials, "path": str(path.relative_to(out)),
                "sha256": sha256(path), "radius_A": 5., "minimum_separation_A": 1.,
                "coordinate_rounding_decimals": 10, "translation_A": [25., 25., 25.],
                "cell_A": [50., 50., 50.], "pbc": [False, False, False],
                "selection_by_relaxed_quality": False}
        dump(out / "inputs" / f"seed-{seed}.json", item)
        generated.append((seed, atoms, item))
    return generated


def snapshot(out: Path):
    source = out / "source"
    copied = []
    for live_path in core_sources():
        relative = live_path.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(live_path, target)
        copied.append({"path": str(relative), "sha256": sha256(target),
                       "live_path": str(live_path.resolve())})
    shutil.copy2(RUNNER, out / "runner.py")
    shutil.copy2(VALIDATOR, out / "validator.py")
    shutil.copy2(Path(__file__).resolve(), out / "prepare.py")
    shutil.copy2(PROTOCOL, out / "protocol.md")
    tracked = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short",
                                      "--", "pamssw", "research/ga_ssw/c60_long_budget.py"], text=True)
    manifest = {
        "checkout": str(ROOT),
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "git_branch": subprocess.check_output(["git", "-C", str(ROOT), "branch", "--show-current"], text=True).strip(),
        "git_status_short": tracked.splitlines(),
        "snapshot_scope": "Python files under pamssw plus this runner and the existing C60 graph validator; no repository clone",
        "core_files": copied,
        "runner": {"path": "runner.py", "sha256": sha256(out / "runner.py"), "live_path": str(RUNNER.resolve())},
        "validator": {"path": "validator.py", "sha256": sha256(out / "validator.py"), "live_path": str(VALIDATOR.resolve())},
        "prepare": {"path": "prepare.py", "sha256": sha256(out / "prepare.py"), "live_path": str(Path(__file__).resolve())},
        "protocol_sha256": sha256(out / "protocol.md"),
    }
    dump(out / "source-manifest.json", manifest)
    return manifest


def verify_snapshot(out: Path):
    manifest = json.loads((out / "source-manifest.json").read_text())
    for row in manifest["core_files"]:
        path = out / "source" / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"frozen core source changed: {path}")
    for key in ("runner", "validator", "prepare"):
        row = manifest[key]
        path = out / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"frozen {key} changed: {path}")
    if sha256(out / "protocol.md") != manifest["protocol_sha256"]:
        raise ValueError("frozen protocol changed")
    return manifest


def common_plan(case: str, seed: int, input_path: Path, out: Path, ref_energy: float):
    relative_input = "input.extxyz"
    plan_base = {
        "model": str(MODEL), "model_sha256": MODEL_SHA, "head": "omol",
        "device": "cuda", "dtype": "float64",
        "runtime": {"torch_manual_seed": 0, "torch_deterministic_algorithms": True,
                    "torch_num_threads": 1, "tf32": False,
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        "ssw_config": protocol_config(),
        "native_mc": {"energy_tol_eV": .1, "maxtrap": 99999},
        "seed": seed, "input": relative_input,
        "input_sha256": sha256(input_path), "search_cap": SEARCH_CAP,
        "fresh_cap": FRESH_CAP, "wall_seconds": SEARCH_SECONDS,
        "reference_energy_eV": ref_energy,
        "reference_input": "../../qualification/reference-source.extxyz",
        "reference_sha256": sha256(out / "qualification" / "reference-source.extxyz"),
        "frozen_files": {
            "input.extxyz": sha256(input_path),
            "../../validator.py": sha256(out / "validator.py"),
            "../../runner.py": sha256(out / "runner.py"),
            "../../source-manifest.json": sha256(out / "source-manifest.json"),
            "../../protocol.md": sha256(out / "protocol.md"),
            "../../qualification/qualification-summary.json":
                sha256(out / "qualification" / "qualification-summary.json"),
        },
        "provenance": {"prepared_git_head": json.loads((out / "source-manifest.json").read_text())["git_head"],
                       "source_manifest": "../../source-manifest.json",
                       "experiment": HERE.name, "case": case},
    }
    return plan_base


def arm_plans(out: Path, inputs, reference_energy: float):
    for seed, _, _ in inputs:
        case = f"seed-{seed}"
        for arm in ("rotation", "direction"):
            folder = out / case / arm
            folder.mkdir(parents=True, exist_ok=False)
            input_path = folder / "input.extxyz"
            shutil.copy2(out / "inputs" / f"seed-{seed}.extxyz", input_path)
            plan = common_plan(case, seed, input_path, out, reference_energy)
            plan["case"] = case
            plan["arm"] = arm
            if arm == "rotation":
                plan["recovered_rotation"] = rotation_settings()
            else:
                plan["recovered_direction"] = direction_settings()
            dump(folder / "plan.json", plan)
            dump(folder / "qualification-link.json", {"case": case, "seed": seed,
                  "qualification": f"../../qualification/inputs/{case}/qualification.json"})
    plan_rows = []
    arm_map = []
    for path in sorted(out.glob("seed-*/[!q]*/plan.json")):
        relative = path.relative_to(out)
        plan = json.loads(path.read_text())
        plan_rows.append({"path": str(relative), "sha256": sha256(path)})
        arm_map.append({"array_task_id": SEEDS.index(plan["seed"]) * 2 +
                            (0 if plan["arm"] == "rotation" else 1),
                        "case": plan["case"], "seed": plan["seed"], "arm": plan["arm"],
                        "run_dir": str(relative.parent), "plan": str(relative),
                        "input": str(relative.parent / "input.extxyz"),
                        "input_sha256": plan["input_sha256"]})
    if len(plan_rows) != 4:
        raise RuntimeError(f"expected four frozen arm plans, got {len(plan_rows)}")
    dump(out / "search-plan-manifest.json", {
        "plans": plan_rows,
        "arm_map": arm_map,
        "qualification_summary": "qualification/qualification-summary.json",
        "source_snapshot_paths": {"core_root": "source/pamssw",
                                   "source_manifest": "source-manifest.json",
                                   "runner": "runner.py", "validator": "validator.py",
                                   "prepare": "prepare.py", "protocol": "protocol.md"},
        "qualification_summary_sha256": sha256(out / "qualification" / "qualification-summary.json"),
        "source_manifest_sha256": sha256(out / "source-manifest.json")})


def instrument_calculate(calculator):
    original = calculator.calculate
    calls = {"count": 0}
    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)
    calculator.calculate = counted
    return calls


class CountedSurface:
    """Count every E/F request, actual ASE calculate calls, denials, and failures."""
    def __init__(self, calculator, ledger: Path, cap: int, deadline: float, process_deadline: float):
        self.calculator, self.ledger = calculator, ledger
        self.cap, self.deadline, self.process_deadline = int(cap), deadline, process_deadline
        self.requests = self.denials = 0

    def append(self, row):
        with self.ledger.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def evaluate(self, atoms):
        import numpy as np
        reason = ("request_cap" if self.requests >= self.cap else
                  "process_deadline" if time.monotonic() >= self.process_deadline else
                  "case_deadline" if time.monotonic() >= self.deadline else None)
        if reason:
            self.denials += 1
            self.append({"kind": "denial", "request": self.requests + 1,
                         "charged": False, "reason": reason})
            raise RuntimeError(f"qualification_{reason}")
        self.requests += 1
        calls_before = self.calculator._c60_call_counter["count"]
        self.append({"kind": "started", "request": self.requests, "charged": True,
                     "actual_calculator_calls_at_start": calls_before})
        work = atoms.copy()
        work.calc = self.calculator
        try:
            energy = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), dtype=float)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            if not np.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(fmax):
                raise ValueError("nonfinite energy or forces")
            self.append({"kind": "evaluation", "request": self.requests,
                         "charged": True, "energy_eV": energy, "fmax_eV_A": fmax,
                         "actual_calculator_calls": self.calculator._c60_call_counter["count"] - calls_before})
            return energy, forces
        except Exception as error:
            self.append({"kind": "failure", "request": self.requests,
                         "charged": True, "error": repr(error),
                         "actual_calculator_calls": self.calculator._c60_call_counter["count"] - calls_before})
            raise


def input_diagnostics(atoms, validator):
    import numpy as np
    if not np.isfinite(atoms.positions).all():
        return {"finite_positions": False, "minimum_distance_A": None,
                "principal_rms_extents_A": None, "graphs": None}
    centered = atoms.positions - atoms.positions.mean(axis=0)
    distances = np.linalg.norm(centered[:, None] - centered[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    extents = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(centered))
    return {
        "finite_positions": bool(np.isfinite(atoms.positions).all()),
        "minimum_distance_A": float(distances.min()),
        "principal_rms_extents_A": extents.tolist(),
        "graphs": {str(cutoff): validator.graph_row(atoms.numbers, atoms.positions, cutoff)
                   for cutoff in (1.64, 1.70, 1.80)},
    }


def run_preflight():
    """No real calculator/model: source syntax plus a dummy counted-surface contract."""
    compile(Path(__file__).read_text(), str(__file__), "exec")
    compile(RUNNER.read_text(), str(RUNNER), "exec")
    compile(VALIDATOR.read_text(), str(VALIDATOR), "exec")
    for source in core_sources():
        compile(source.read_text(), str(source), "exec")
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    class Dummy(Calculator):
        implemented_properties = ["energy", "forces"]
        def __init__(self):
            super().__init__(); self.calls = 0
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {"energy": 0., "forces": np.zeros((len(self.atoms), 3))}

    with tempfile.TemporaryDirectory(prefix="c60-direction-preflight-") as temp:
        calculator = Dummy(); calls = instrument_calculate(calculator)
        calculator._c60_call_counter = calls
        surface = CountedSurface(calculator, Path(temp) / "ledger.jsonl", 1,
                                 time.monotonic() + 60, time.monotonic() + 60)
        atom = Atoms("C", positions=[[0., 0., 0.]])
        surface.evaluate(atom)
        try:
            surface.evaluate(atom)
        except RuntimeError as error:
            denied = "request_cap" in str(error)
        else:
            denied = False
        rows = [json.loads(line) for line in (Path(temp) / "ledger.jsonl").read_text().splitlines()]
        if not (surface.requests == 1 and surface.denials == 1 and calls["count"] == 1
                and rows[0]["kind"] == "started" and rows[0]["charged"]
                and rows[1]["kind"] == "evaluation" and rows[1]["charged"]
                and rows[2]["kind"] == "denial" and rows[2]["charged"] is False and denied):
            raise RuntimeError("dummy counted-surface preflight failed")
        direct_dummy_calls = calls["count"]
        from ase import Atoms
        sys.path.insert(0, str(ROOT))
        from pamssw.standalone import NativeMCSettings, SSWConfig, run_ssw
        calculator.reset()
        integration_before = calls["count"]
        integration_surface = CountedSurface(calculator, Path(temp) / "ssw-integration.jsonl", 8,
            time.monotonic() + 60, time.monotonic() + 60)
        result = run_ssw(Atoms("C2", positions=[[0., 0., 0.], [1.4, 0., 0.]]),
            integration_surface, steps=0, config=SSWConfig(**protocol_config()),
            rng=np.random.default_rng(SEEDS[0]), mc=NativeMCSettings(.1, 99999))
        if not (result.status == "completed" and result.initial.converged
                and len(result.records) == 0
                and result.evaluation_requests == integration_surface.requests
                and integration_surface.requests == 1
                and calls["count"] - integration_before == 1):
            raise RuntimeError("zero-force run_ssw initialization integration failed")
        scaffold = Path(temp) / "prepared"
        scaffold.mkdir()
        inputs = write_inputs(scaffold)
        snapshot(scaffold)
        (scaffold / "qualification").mkdir()
        shutil.copy2(REFERENCE, scaffold / "qualification" / "reference-source.extxyz")
        dump(scaffold / "qualification" / "qualification-summary.json", {
            "status": "qualified", "search_eligible": True,
            "reference": {"qualified": True},
            "inputs": [{"label": f"seed-{seed}", "qualified": True} for seed in SEEDS],
            "preflight_stub_only": True})
        arm_plans(scaffold, inputs, REFERENCE_ENERGY)
        verify_output(scaffold)
    generated = []
    for seed in SEEDS:
        atoms, trials = generate_input(seed)
        if (len(atoms) != 60 or not np.all(atoms.numbers == 6) or atoms.pbc.any()
                or not np.isfinite(atoms.positions).all()):
            raise RuntimeError(f"seed {seed} failed composition/finite/PBC input check")
        relative = atoms.positions - 25.
        pair = np.linalg.norm(relative[:, None] - relative[None, :], axis=2)
        np.fill_diagonal(pair, np.inf)
        if np.max(np.linalg.norm(relative, axis=1)) > 5. + 1e-10 or pair.min() < 1. - 1e-10:
            raise RuntimeError(f"seed {seed} failed sphere/minimum-separation check")
        generated.append({"seed": seed, "trials": trials})
    print(json.dumps({"status": "preflight_passed", "real_pes_requests": 0,
                      "real_model_initialized": False, "dummy_accounting": "passed",
                      "input_snapshot_and_plan_scaffolding": "passed_with_stub_qualification",
                      "dummy_surface_requests": surface.requests,
                      "dummy_calculator_calls": direct_dummy_calls,
                      "dummy_cap_denials": surface.denials,
                      "zero_force_run_ssw_requests": integration_surface.requests,
                      "zero_force_run_ssw_calculator_calls": calls["count"] - integration_before,
                      "input_generation_trials": generated}, indent=2))


def qualify_one(label: str, atoms, out: Path, process_deadline: float,
                search_calculator, cold_calculator, config, mc, run_ssw, validator):
    import numpy as np
    from ase.io import write
    folder = (out / "qualification" / "reference" if label == "reference" else
              out / "qualification" / "inputs" / label)
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "source.extxyz", atoms)
    started = time.monotonic()
    before_calls = search_calculator._c60_call_counter["count"]
    surface = CountedSurface(search_calculator, folder / "initial-ef-ledger.jsonl",
        INPUT_CAP, min(started + INPUT_SECONDS, process_deadline), process_deadline)
    row = {"label": label, "status": "started", "input_sha256": sha256(folder / "source.extxyz")}
    initial = None
    try:
        result = run_ssw(atoms.copy(), surface, steps=0, config=config,
                         rng=np.random.default_rng(SEEDS[0] if label == "seed-26100781" else
                             SEEDS[1] if label == "seed-26100782" else 0), mc=mc)
        initial = result.initial
        row["ssw_status"] = result.status
    except Exception as error:
        initial = getattr(error, "result", None)
        row.update(status="initial_quench_failed", error=repr(error))
    row.update(initial_requests=surface.requests, initial_denials=surface.denials,
        initial_calculator_calls=search_calculator._c60_call_counter["count"] - before_calls,
        initial_elapsed_seconds=time.monotonic() - started)
    qualified_minimum = False
    if initial is not None:
        write(folder / "initial.extxyz", initial.atoms)
        initial_diag = input_diagnostics(initial.atoms, validator)
        energy = float(initial.energy) if initial.energy is not None else float("nan")
        initial_fmax = float(initial.max_force)
        initial_finite = bool(np.isfinite(energy) and np.isfinite(initial.atoms.positions).all()
                              and np.isfinite(initial_fmax))
        row.update(initial_converged=bool(initial.converged),
                   initial_energy_eV=energy if np.isfinite(energy) else None,
                   initial_fmax_eV_A=initial_fmax if np.isfinite(initial_fmax) else None,
                   initial_finite=initial_finite,
                   initial_diagnostics=initial_diag)
        qualified_minimum = bool(initial_finite and initial.converged and initial_fmax <= config.fmax
                                 and surface.denials == 0 and time.monotonic() < process_deadline)
    if qualified_minimum:
        try:
            if time.monotonic() >= process_deadline:
                raise RuntimeError("process_deadline_before_cold_check")
            cold_calculator.reset()
            cold_before = cold_calculator._c60_call_counter["count"]
            cold_surface = CountedSurface(cold_calculator, folder / "cold-ef-ledger.jsonl", 1,
                process_deadline, process_deadline)
            write(folder / "cold.extxyz", initial.atoms)
            cold_energy, cold_forces = cold_surface.evaluate(initial.atoms)
            cold_fmax = float(np.linalg.norm(cold_forces, axis=1).max())
            cold_diag = input_diagnostics(initial.atoms, validator)
            repeat = abs(cold_energy - initial.energy) <= 1e-6
            row.update(cold_status="completed", cold_energy_eV=cold_energy,
                cold_energy_error_eV=float(cold_energy - initial.energy), cold_fmax_eV_A=cold_fmax,
                cold_calculator_calls=cold_calculator._c60_call_counter["count"] - cold_before,
                cold_requests=cold_surface.requests, cold_denials=cold_surface.denials,
                cold_energy_repeat=bool(repeat),
                cold_diagnostics=cold_diag)
            row["qualified"] = bool(repeat and np.isfinite(cold_fmax)
                and cold_fmax <= config.fmax and row["initial_finite"]
                and row["initial_converged"] and row["initial_fmax_eV_A"] <= config.fmax)
        except Exception as error:
            row.update(cold_status="failed", cold_error=repr(error), qualified=False)
    else:
        row["qualified"] = False
        row["cold_status"] = "not_run_initial_not_qualified"
    row["status"] = "qualified" if row["qualified"] else row.get("status", "not_qualified")
    dump(folder / "qualification.json", row)
    return row


def execute(out: Path):
    import numpy as np
    from ase.io import read, write

    started = time.monotonic()
    process_deadline = started + PROCESS_SECONDS
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "preparation-status.json", {"status": "preparing", "started_unix": time.time(),
                                             "process_cap_seconds": PROCESS_SECONDS})
    try:
        generated = write_inputs(out)
        manifest = snapshot(out)
        (out / "qualification").mkdir(parents=True, exist_ok=True)
        shutil.copy2(REFERENCE, out / "qualification" / "reference-source.extxyz")
        reference = read(out / "qualification" / "reference-source.extxyz")
        if len(reference) != 60 or not np.all(reference.numbers == 6) or not np.isfinite(reference.positions).all():
            raise ValueError("reference source is not a finite C60 structure")
        if reference.pbc.any():
            raise ValueError("reference source is unexpectedly periodic")
        if sha256(REFERENCE) != sha256(out / "qualification" / "reference-source.extxyz"):
            raise ValueError("reference source copy changed")
        if sha256(MODEL) != MODEL_SHA:
            raise ValueError("MH-1 model SHA-256 differs from frozen protocol")

        validator_spec = __import__("importlib.util", fromlist=["spec_from_file_location"])
        spec = validator_spec.spec_from_file_location("c60_direction_validator", out / "validator.py")
        validator = validator_spec.module_from_spec(spec); spec.loader.exec_module(validator)
        for seed, atoms, item in generated:
            item["raw_geometry_diagnostics"] = input_diagnostics(atoms, validator)
            dump(out / "inputs" / f"seed-{seed}.json", item)

        # Snapshot source is installed first so preparation and workers use this tree.
        sys.path.insert(0, str(out / "source"))
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from mace.calculators import MACECalculator
        from pamssw.standalone import NativeMCSettings, SSWConfig, run_ssw
        from pamssw.standalone.surface import ASESurface  # imported source path is recorded below
        import pamssw.standalone.paper_reference as paper_reference
        import pamssw.standalone.recovered_direction as recovered_direction
        import pamssw.standalone.recovered_rotation as recovered_rotation
        config = SSWConfig(**protocol_config())
        mc = NativeMCSettings(.1, 99999)
        recovered_direction.RecoveredDirectionSettings(**direction_settings())
        recovered_rotation.RecoveredRotationSettings(**rotation_settings())
        source_imports = {name: str(Path(module.__file__).resolve()) for name, module in {
            "paper_reference": paper_reference, "surface": sys.modules[ASESurface.__module__],
            "recovered_direction": recovered_direction, "recovered_rotation": recovered_rotation}.items()}
        for import_path in source_imports.values():
            if not Path(import_path).is_relative_to(out / "source"):
                raise RuntimeError(f"qualification imported non-snapshot core source: {import_path}")
        if time.monotonic() >= process_deadline:
            raise RuntimeError("process deadline reached during imports/model startup")
        model_kwargs = dict(model_paths=str(MODEL), head="omol", device="cuda",
                            default_dtype="float64", enable_cueq=False, enable_oeq=False)
        search_calc = MACECalculator(**model_kwargs)
        search_calc._c60_call_counter = instrument_calculate(search_calc)
        cold_calc = MACECalculator(**model_kwargs)
        cold_calc._c60_call_counter = instrument_calculate(cold_calc)
        source = {"manifest_sha256": sha256(out / "source-manifest.json"),
                  "core_imports": source_imports,
                  "validator": {"path": str((out / "validator.py").resolve()),
                                "sha256": sha256(out / "validator.py")},
                  "runner": {"path": str((out / "runner.py").resolve()),
                             "sha256": sha256(out / "runner.py")},
                  "model": str(MODEL), "model_sha256": MODEL_SHA,
                  "model_head": "omol", "dtype": "float64", "device": "cuda",
                  "enable_cueq": False, "enable_oeq": False,
                  "python": sys.version, "python_executable": sys.executable,
                  "numpy": np.__version__, "ase": __import__("ase").__version__,
                  "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                  "cublas_workspace_config": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                  "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                  "slurm_job_id": os.getenv("SLURM_JOB_ID")}
        dump(out / "qualification" / "source-runtime.json", source)

        reference_start = time.monotonic()
        cold_calc.reset()
        reference_counter_before = cold_calc._c60_call_counter["count"]
        reference_surface = CountedSurface(cold_calc,
            out / "qualification" / "reference-cold-ef-ledger.jsonl", 1,
            process_deadline, process_deadline)
        reference_row = {"label": "reference", "status": "started",
                         "source_path": str(REFERENCE.resolve()),
                         "source_sha256": sha256(out / "qualification" / "reference-source.extxyz")}
        try:
            reference_energy, reference_forces = reference_surface.evaluate(reference)
            reference_fmax = float(np.linalg.norm(reference_forces, axis=1).max())
            reference_graphs = {str(cutoff): validator.graph_row(reference.numbers, reference.positions, cutoff)
                                for cutoff in (1.64, 1.70, 1.80)}
            reference_diag = input_diagnostics(reference, validator)
            graph_qualified = all(row["graph_cage_candidate"] for row in reference_graphs.values())
            repeat_qualified = abs(reference_energy - REFERENCE_ENERGY) <= 1e-6
            reference_row.update(status="qualified" if graph_qualified and repeat_qualified
                and np.isfinite(reference_fmax) and reference_fmax <= config.fmax else "not_qualified",
                cold_status="completed", cold_energy_eV=reference_energy,
                cold_energy_error_eV=float(reference_energy - REFERENCE_ENERGY),
                cold_fmax_eV_A=reference_fmax, cold_calculator_calls=
                cold_calc._c60_call_counter["count"] - reference_counter_before,
                cold_requests=reference_surface.requests, cold_denials=reference_surface.denials,
                energy_repeat=bool(repeat_qualified), fmax_qualified=bool(np.isfinite(reference_fmax)
                    and reference_fmax <= config.fmax), graph_qualified=graph_qualified,
                graphs=reference_graphs, cold_diagnostics=reference_diag,
                elapsed_seconds=time.monotonic() - reference_start,
                qualified=bool(graph_qualified and repeat_qualified and np.isfinite(reference_fmax)
                    and reference_fmax <= config.fmax))
        except Exception as error:
            reference_row.update(status="cold_check_failed", qualified=False, error=repr(error),
                cold_requests=reference_surface.requests, cold_denials=reference_surface.denials,
                cold_calculator_calls=cold_calc._c60_call_counter["count"] - reference_counter_before,
                elapsed_seconds=time.monotonic() - reference_start)
        dump(out / "qualification" / "reference" / "qualification.json", reference_row)
        dump(out / "qualification" / "reference-qualification.json", reference_row)
        rows = []
        for seed, atoms, item in generated:
            case = f"seed-{seed}"
            search_calc.reset()
            if time.monotonic() >= process_deadline:
                rows.append({"label": case, "status": "not_started_process_deadline",
                             "qualified": False, "initial_requests": 0,
                             "initial_calculator_calls": 0, "initial_denials": 0})
                break
            row = qualify_one(case, atoms, out, process_deadline, search_calc, cold_calc,
                              config, mc, run_ssw, validator)
            rows.append(row)
            dump(out / "qualification" / "qualification-summary.json", {
                "status": "running", "reference": reference_row, "inputs": rows,
                "elapsed_seconds": time.monotonic() - started,
                "search_eligible": False})
        ready = bool(reference_row.get("qualified") and len(rows) == 2 and all(x.get("qualified") for x in rows)
                     and time.monotonic() < process_deadline)
        ref_energy = reference_row.get("cold_energy_eV", REFERENCE_ENERGY)
        summary = {"status": "qualified" if ready else "qualification_failed",
                   "reference": reference_row, "inputs": rows,
                   "elapsed_seconds": time.monotonic() - started,
                   "process_cap_seconds": PROCESS_SECONDS,
                   "maximum_initial_requests_per_input": INPUT_CAP,
                   "maximum_initial_seconds_per_input": INPUT_SECONDS,
                   "maximum_independent_cold_checks": 3,
                   "search_eligible": ready,
                   "search_block_reason": None if ready else "reference_or_initial_quench_qualification_failed",
                   "source_manifest_sha256": sha256(out / "source-manifest.json"),
                   "search_arms": [f"seed-{seed}/{arm}" for seed in SEEDS
                                   for arm in ("rotation", "direction")]}
        dump(out / "qualification" / "qualification-summary.json", summary)
        arm_plans(out, generated, ref_energy)
        dump(out / "preparation-status.json", {"status": "complete", "elapsed_seconds": summary["elapsed_seconds"],
                                                   "search_eligible": ready})
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0 if ready else 2
    except Exception as error:
        dump(out / "preparation-status.json", {"status": "failed", "error": repr(error),
            "elapsed_seconds": time.monotonic() - started,
            "qualification_summary": str(out / "qualification" / "qualification-summary.json")})
        raise


def verify_output(out: Path):
    out = out.resolve()
    verify_snapshot(out)
    summary_path = out / "qualification" / "qualification-summary.json"
    if not summary_path.is_file():
        raise ValueError("qualification summary is missing")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "qualified" or not summary.get("search_eligible"):
        raise ValueError("initial/reference qualification did not pass; automatic search is prohibited")
    if (not summary.get("reference", {}).get("qualified") or
            {row.get("label") for row in summary.get("inputs", []) if row.get("qualified")} !=
            {f"seed-{seed}" for seed in SEEDS}):
        raise ValueError("reference or both fixed input qualifications are not present")
    plan_manifest = json.loads((out / "search-plan-manifest.json").read_text())
    if len(plan_manifest.get("plans", [])) != 4:
        raise ValueError("prepared search plan manifest does not contain four arms")
    if plan_manifest.get("qualification_summary_sha256") != sha256(summary_path):
        raise ValueError("qualification summary changed after the search plans were frozen")
    if plan_manifest.get("source_manifest_sha256") != sha256(out / "source-manifest.json"):
        raise ValueError("source manifest changed after the search plans were frozen")
    for row in plan_manifest["plans"]:
        path = out / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError(f"frozen search plan changed: {path}")
    expected_plans = {f"seed-{seed}/{arm}/plan.json" for seed in SEEDS
                      for arm in ("rotation", "direction")}
    if {row["path"] for row in plan_manifest["plans"]} != expected_plans:
        raise ValueError("prepared plan paths do not match the four fixed arms")
    for seed in SEEDS:
        pair_plans = {}
        for arm in ("rotation", "direction"):
            folder = out / f"seed-{seed}" / arm
            plan = json.loads((folder / "plan.json").read_text())
            pair_plans[arm] = plan
            if (plan["search_cap"] != SEARCH_CAP or plan["fresh_cap"] != FRESH_CAP
                    or plan["wall_seconds"] != SEARCH_SECONDS or plan["seed"] != seed
                    or plan["ssw_config"] != protocol_config()):
                raise ValueError(f"search plan differs from frozen protocol: {folder}")
            for rel, expected in plan["frozen_files"].items():
                path = folder / rel
                if not path.is_file() or sha256(path) != expected:
                    raise ValueError(f"frozen worker input changed: {path}")
        rotation, direction = pair_plans["rotation"], pair_plans["direction"]
        if (rotation["input_sha256"] != direction["input_sha256"] or
                rotation.get("recovered_rotation") != rotation_settings() or
                direction.get("recovered_direction") != direction_settings() or
                "recovered_direction" in rotation or "recovered_rotation" in direction):
            raise ValueError(f"paired arms differ outside the frozen direction bundle: seed-{seed}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.output is not None:
            parser.error("--preflight does not write an output directory")
        run_preflight()
        return
    if args.execute:
        if args.output is None:
            parser.error("--execute requires --output NEW_DIRECTORY")
        if args.output.exists():
            parser.error("--execute output must be a new directory")
        raise SystemExit(execute(args.output))
    if args.output is not None:
        parser.error("--verify-output and --output cannot be combined")
    verify_output(args.verify_output)


if __name__ == "__main__":
    main()
