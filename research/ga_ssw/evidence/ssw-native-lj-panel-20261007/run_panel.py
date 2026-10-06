#!/usr/bin/env python3
"""Bounded preparation and execution runner for the native-LJ panel.

PES work is opt-in through --prepare, --execute, or --native-probe.  This
script does not alter PAM-SSW and keeps all run products under a new directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import time
import traceback

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
SEEDS = {55: 26100711, 38: 26100712}
RADIUS = 5.5 * 2.7
CELL = np.eye(3) * 100.0
EPSILON, SIGMA = 1.0, 2.7
FMAX = 0.05
REFERENCE = {38: -173.928427, 55: -279.248470}
HIT_TOL = 0.001
NATIVE_BINARY = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")
NATIVE_BINARY_SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
MPI_LIB = "/opt/devtools/intel/oneapi/mpi/2021.13/lib"
NATIVE_PACKAGE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-vacuum-geometry/research/ga_ssw/evidence/c60-native-long-20260920/seed17093")


def json_write(path: Path, data) -> None:
    path.write_text(json.dumps(load_serializer()._jsonable(data), indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze_runner(out: Path) -> str:
    runner = out / "run_panel.py"
    shutil.copy2(Path(__file__).resolve(), runner)
    shutil.copy2(HERE / "protocol.md", out / "protocol.md")
    return sha256(runner)


def load_serializer():
    path = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
    spec = importlib.util.spec_from_file_location("ssw_lj_panel_ledger", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import existing serializer at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sources():
    from research.ga_ssw.full_pair_lj import FullPairLJ
    import pamssw.standalone.paper_reference as paper
    import pamssw.standalone.recovered_rotation as rotation
    import pamssw.standalone.recovered_direction as direction
    import pamssw.standalone.native_height_policy as height
    import pamssw.standalone.native_mc as mc
    import research.ga_ssw.full_pair_lj as lj
    import research.ga_ssw.lasp_external_ase as external
    return {"checkout": str(ROOT), "git_head": subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip(),
        "head_pamssw_tree": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:pamssw"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "tracked_pamssw_changes": subprocess.run(["git", "-C", str(ROOT), "status", "--short", "--", "pamssw"],
            check=True, capture_output=True, text=True).stdout.splitlines(),
        "FullPairLJ": str(Path(lj.__file__).resolve()),
        "run_ssw": str(Path(paper.__file__).resolve()),
        "RecoveredRotationSettings": str(Path(rotation.__file__).resolve()),
        "RecoveredDirectionSettings": str(Path(direction.__file__).resolve()),
        "ConservativeNativeHeightPolicy": str(Path(height.__file__).resolve()),
        "NativeMCSettings": str(Path(mc.__file__).resolve()),
        "lasp_external_ase": str(Path(external.__file__).resolve()),
        "FullPairLJ_class": f"{FullPairLJ.__module__}.{FullPairLJ.__name__}"}


def settings():
    from pamssw.standalone import (SSWConfig, RecoveredRotationSettings,
        ConservativeNativeHeightPolicy, NativeMCSettings)
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    # Native MC's recovered exponent gives exp(-dE * 96485/(20*8.314*T));
    # this T makes its scale correspond to the existing LJ kBT=0.8 eV recipe.
    temperature = 0.8 * 96485.0 / (20.0 * 8.314)
    config = SSWConfig(width=0.6, rotation_bias=1.0, max_gaussians=14,
        temperature_K=temperature, fmax=FMAX, bias_fmax=0.1,
        relax_steps=1000, fd_step=0.001, rotation_hvp=39, rotation_tol=0.02,
        direction_sampling="global", rotation_solver="broyden-euclidean",
        cluster_frame="direction_only", quench_optimizer="safe-lbfgs-total",
        lbfgs_memory=500, rotation_exit_policy="force_or_budget")
    rotation = RecoveredRotationSettings(5, 15, 0.2, 0.02, "euclidean", 40)
    direction = RecoveredDirectionSettings(50, 0.5, 0.5, 5, 15, 0.2, 0.02,
        "euclidean", 40, c1_radius_policy="restricted", startup_order="legacy",
        geometry="nonperiodic")
    height = ConservativeNativeHeightPolicy(0.5, 0.2, 1, 10.0, 1.0, 2.0)
    mc = NativeMCSettings(0.1, 99999)
    return config, rotation, direction, height, mc, temperature


def uniform_volume_cluster(n: int, seed: int, initialization="dilute"):
    # fcc density at pair r_min=2**(1/6)*sigma is 1/sigma**3.
    radius_A = RADIUS if initialization == "dilute" else SIGMA * (3*n/(4*math.pi))**(1/3)
    from ase import Atoms
    children = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(children[0])
    direction = rng.normal(size=(n, 3))
    norm = np.linalg.norm(direction, axis=1)
    if np.any(norm == 0):
        raise FloatingPointError("zero random direction; preserve failure, do not resample")
    direction /= norm[:, None]
    radius = rng.random(n) ** (1.0 / 3.0) * radius_A
    positions = direction * radius[:, None]
    positions += 50.0 - positions.mean(axis=0)
    return Atoms(f"Ar{n}", positions=positions, cell=CELL, pbc=False), children[1]


def connected_components(atoms):
    # Geometric qualification only; not a bond/Hessian stability assertion.
    adjacent = (atoms.get_all_distances(mic=False) < 1.3 * SIGMA)
    todo = set(range(len(atoms)))
    sizes = []
    while todo:
        stack = [todo.pop()]
        size = 0
        while stack:
            i = stack.pop()
            size += 1
            new = set(np.flatnonzero(adjacent[i])) & todo
            todo -= new
            stack.extend(new)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def geometry_gate(atoms):
    positions = np.asarray(atoms.positions, dtype=float)
    if positions.shape != (len(atoms), 3) or not np.isfinite(positions).all():
        return {"eligible": False, "reason": "nonfinite_or_malformed_positions"}
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    span = hi - lo
    inside = bool(np.all(lo >= 0.0) and np.all(hi < 100.0))
    return {"eligible": inside and bool(np.all(span < 50.0)),
            "reason": "ok" if inside and np.all(span < 50.0) else
                      ("outside_storage_cell" if not inside else "axis_span_not_below_half_cell"),
            "min_A": lo.tolist(), "max_A": hi.tolist(), "axis_span_A": span.tolist(),
            "storage_cell_A": 100.0, "max_span_A": 50.0}


class CountedSurface:
    """Count every attempted E/F, persist its geometry, and enforce bounds."""
    def __init__(self, calculator, path, *, cap, deadline=None):
        from pamssw.standalone import ASESurface
        self.inner = ASESurface(calculator)
        self.path, self.cap, self.deadline = Path(path), int(cap), deadline
        self.records = []
        self.paid = 0
        self.actual_calculator_calls = 0

    @property
    def requests(self):
        return self.paid

    def evaluate(self, atoms):
        if self.paid >= self.cap:
            self._denial(atoms, "request_cap_denied_before_evaluation")
            raise RuntimeError("request_cap_denied_before_evaluation")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self._denial(atoms, "wall_cap_denied_before_evaluation")
            raise RuntimeError("wall_cap_denied_before_evaluation")
        self.paid += 1
        row = {"request": self.paid, "charged": True,
               "atoms": {"symbols": atoms.get_chemical_symbols(),
                         "positions_A": np.asarray(atoms.positions).tolist(),
                         "cell_A": np.asarray(atoms.cell.array).tolist(),
                         "pbc": np.asarray(atoms.pbc).tolist()},
               "energy_eV": None, "forces_eV_A": None,
               "actual_calculator_called": False}
        gate = geometry_gate(atoms)
        row["geometry_gate"] = gate
        self.records.append(row)
        try:
            if not gate["eligible"]:
                raise RuntimeError("geometry_gate_failed:" + gate["reason"])
            row["actual_calculator_called"] = True
            self.actual_calculator_calls += 1
            energy, forces = self.inner.evaluate(atoms)
            row.update(energy_eV=float(energy), forces_eV_A=np.asarray(forces).tolist(), status="ok")
            return energy, forces
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            with self.path.open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _denial(self, atoms, reason):
        row = {"request": self.paid + 1, "charged": False, "status": "denied",
               "reason": reason, "atoms": {"symbols": atoms.get_chemical_symbols(),
                   "positions_A": np.asarray(atoms.positions).tolist()}}
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def campaign_config():
    config, rotation, direction, height, mc, temperature = settings()
    return {"potential": {"class": "FullPairLJ", "epsilon_eV": EPSILON,
                           "sigma_A": SIGMA, "cutoff_A": None, "periodic": False},
            "ssw_config": asdict(config), "recovered_rotation": asdict(rotation),
            "recovered_direction": asdict(direction), "native_height": asdict(height),
            "native_mc": asdict(mc), "native_equivalent_temperature_K": temperature,
            "outer_steps": 6, "request_cap_per_arm": 4000,
            "wall_cap_seconds_per_arm": 600, "geometry_gate": "all positions in [0,100), axis ptp<50 A"}


def preflight(initialization="dilute"):
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import run_ssw, ASESurface
    from research.ga_ssw.lasp_external_ase import run_lasp
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    serializer = load_serializer()
    cfg = campaign_config()
    calculator = FullPairLJ(epsilon=EPSILON, sigma=SIGMA)
    generated = {}
    for n, seed in SEEDS.items():
        atoms, _ = uniform_volume_cluster(n, seed, initialization)
        generated[str(n)] = {"seed": seed, "natoms": len(atoms), "initialization": initialization,
            "geometry_gate": geometry_gate(atoms),
            "raw_positions_sha256": hashlib.sha256(atoms.positions.tobytes()).hexdigest()}
    result = {"status": "preflight_ok", "pes_requests": 0, "protocol": cfg,
        "generated_inputs": generated, "dependencies": {"FullPairLJ": True,
            "FullPairLJ_properties": list(calculator.implemented_properties),
            "run_ssw": callable(run_ssw), "ASESurface": callable(ASESurface),
            "run_lasp": callable(run_lasp), "save_ssw_checkpoint": callable(save_ssw_checkpoint)},
        "serializer": callable(getattr(serializer, "_jsonable", None)),
        "imports": sources(), "native_binary_exists": NATIVE_BINARY.is_file(),
        "native_binary_sha256": sha256(NATIVE_BINARY) if NATIVE_BINARY.is_file() else None,
        "native_binary_hash_matches_audited": (NATIVE_BINARY.is_file() and
            sha256(NATIVE_BINARY) == NATIVE_BINARY_SHA256),
        "mpi_library_exists": Path(MPI_LIB).is_dir(),
        "native_package_exists": NATIVE_PACKAGE.is_dir(),
        "native_client_exists": (NATIVE_PACKAGE / "client.py").is_file(),
        "bounded_supervisor_exists": (NATIVE_PACKAGE / "bounded_process.py").is_file()}
    print(json.dumps(result, indent=2, allow_nan=False))
    dependencies_ok = all(result[key] for key in (
        "native_binary_exists", "native_binary_hash_matches_audited", "mpi_library_exists",
        "native_package_exists", "native_client_exists", "bounded_supervisor_exists", "serializer"))
    if not dependencies_ok or not all(row["geometry_gate"]["eligible"] for row in generated.values()):
        raise SystemExit("preflight geometry gate failed")


def prepare(output: Path, initialization="dilute"):
    from ase.io import write
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import ASESurface, run_ssw
    from pamssw.standalone.paper_reference import InitialQuenchError, save_ssw_checkpoint
    output.mkdir(parents=True, exist_ok=False)
    runner_hash = freeze_runner(output)
    json_write(output / "provenance.json", {"mode": "prepare", "imports": sources(),
        "runner_sha256": runner_hash,
        "protocol": campaign_config(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    cfg, _rot, _direction, _height, _mc, _temp = settings()
    for n, seed in SEEDS.items():
        case = output / f"lj{n}-seed{seed}"
        case.mkdir()
        surface = None
        try:
            raw, search_seed = uniform_volume_cluster(n, seed, initialization)
            write(case / "random-raw.extxyz", raw)
            surface = CountedSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
                                     case / "prepare-ef.jsonl", cap=1000)
            rng = np.random.default_rng(search_seed)
            result = run_ssw(raw.copy(), surface, steps=0, config=cfg, rng=rng,
                             checkpoint_callback=lambda _cp: False)
            checkpoint = result.checkpoint
            if checkpoint is not None:
                save_ssw_checkpoint(case / "prepare-checkpoint.pkl", checkpoint)
            initial = checkpoint.initial if checkpoint is not None else result.initial
            candidate = checkpoint.best if checkpoint is not None else None
            qualified = bool(initial.converged and np.isfinite(initial.energy) and
                             initial.max_force <= FMAX and candidate is not None and
                             candidate.converged and candidate.max_force <= FMAX and
                             geometry_gate(candidate.atoms)["eligible"])
            components = connected_components(candidate.atoms) if candidate is not None else []
            numerical_qualified = qualified
            qualified = qualified and components == [n]
            row = {"status": "qualified" if qualified else "not_qualified",
                "initialization": initialization, "raw_radius_A": RADIUS if initialization == "dilute" else SIGMA*(3*n/(4*math.pi))**(1/3),
                "numerical_qualified": numerical_qualified, "component_sizes_1p3sigma": components,
                "n": n, "seed": seed, "prepare_requests": surface.requests,
                "initial": {"energy_eV": float(initial.energy),
                    "fmax_eV_A": float(initial.max_force), "converged": bool(initial.converged)},
                "qualified": qualified,
                "ssw_rng_child_state": search_seed.generate_state(4).tolist()}
            if qualified:
                fresh = CountedSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
                                       case / "fresh-qualification.jsonl", cap=1)
                energy, forces = fresh.evaluate(candidate.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                row["fresh"] = {"energy_eV": float(energy), "fmax_eV_A": fmax,
                    "energy_error_eV": float(energy - candidate.energy),
                    "qualified": bool(np.isfinite(energy) and fmax <= FMAX)}
                row["qualified"] = bool(row["fresh"]["qualified"])
                if row["qualified"]:
                    common = candidate.atoms.copy()
                    common.info.update(seed=seed, natoms=n, preparation_qualified=True,
                        preparation_fmax_eV_A=float(candidate.max_force),
                        preparation_energy_eV=float(candidate.energy))
                    write(case / "prepared.extxyz", common)
                    write(output / f"lj{n}.extxyz", common)
            json_write(case / "result.json", row)
            if checkpoint is not None:
                json_write(case / "checkpoint-summary.json", {
                    "evaluation_requests": int(checkpoint.evaluation_requests),
                    "minimum_count": len(checkpoint.minima), "status": checkpoint.status})
        except Exception as exc:
            json_write(case / "failure.json", {"n": n, "seed": seed,
                "error": repr(exc), "traceback": traceback.format_exc(),
                "requests": None if surface is None else surface.requests})
    json_write(output / "prepare-summary.json", {"cases": [str(p) for p in sorted(output.glob("lj*/result.json"))],
        "failures": [str(p) for p in sorted(output.glob("lj*/failure.json"))],
        "status": "complete_with_failures_preserved"})


def native_input_text(n, steps):
    t = 0.8 * 96485.0 / (20.0 * 8.314)
    # Keys below are present in the recovered native allkeys log.  Do not add
    # undocumented aliases: effective values are captured from allkeys.log.
    lines = ["potential external", "explore_type ssw", "Ewaldflag 0", "Run_type 5",
        f"ranseed {SEEDS[n]}",
        f"SSW.SSWsteps {steps}", f"SSW.ftol {FMAX / math.sqrt(3):.15g}",
        "SSW.MaxOptstep 1000", "SSW.NG 14", f"SSW.Temp {t:.15g}",
        "SSW.ds_atom 0.6", "SSW.internal_LJ F", "SSW.Lmode_Q F",
        "SSW.RotMaxStep_preRot 5", "SSW.RotMaxStep 15",
        "SSW.Rotftol_preRot 0.2", "SSW.Rotftol 0.02", "SSW.DimerdR 0.001",
        "SSW.gausW_level 1", "SSW.MaxW 10", "SSW.W_initial 0.5",
        "SSW.W_neg 0.2", "SSW.W_step 1", "SSW.W_scalefact 2",
        "SSW.Bfgs_history 500", "SSW.energy_tol 0.1", "SSW.maxtrap 99999",
        "SSW.output T", "SSW.printevery T"]
    return "\n".join(lines) + "\n"


def make_arc(atoms):
    rows = ["!BIOSYM archive 2", "PBC=ON", "Energy 0 0.0 0.0", "!DATE",
            "PBC 100 100 100 90 90 90"]
    for i, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions), 1):
        rows.append(f"{symbol} {pos[0]:.14f} {pos[1]:.14f} {pos[2]:.14f} CORE {i} {symbol} {symbol} 0.0 {i}")
    rows.extend(["end", "end", ""])
    return "\n".join(rows)


def execute(args):
    from ase.io import read, write
    from research.ga_ssw.full_pair_lj import FullPairLJ
    # The supervisor changes cwd; retain absolute child script/status paths.
    args.output = args.output.resolve()
    args.input = args.input.resolve()
    if args.output.exists():
        raise FileExistsError(f"output must be new: {args.output}")
    if not (0 < args.cap <= 4000 and 0 < args.wall <= 600 and 0 < args.steps <= 6):
        raise ValueError("per-arm cap, wall, and steps must be within protocol maxima 4000/600/6")
    if args.native_probe and args.arm != "native":
        raise ValueError("--native-probe requires --arm native")
    atoms = read(args.input)
    if len(atoms) != args.n or atoms.get_chemical_symbols() != ["Ar"] * args.n:
        raise ValueError("prepared input composition does not match --n")
    if atoms.info.get("preparation_qualified") is not True or int(atoms.info.get("seed", -1)) != SEEDS[args.n]:
        raise ValueError("input metadata does not certify the requested common prepared seed")
    seed = SEEDS[args.n]
    case_result = args.input.parent / f"lj{args.n}-seed{seed}" / "result.json"
    if not case_result.is_file() and args.input.parent.name == f"lj{args.n}-seed{seed}":
        case_result = args.input.parent / "result.json"
    if not case_result.is_file():
        raise FileNotFoundError(f"prepared-case qualification record is required: {case_result}")
    qualification = json.loads(case_result.read_text())
    if (qualification.get("qualified") is not True or
            int(qualification.get("n", -1)) != args.n or
            int(qualification.get("seed", -1)) != seed):
        raise ValueError("prepared-case result does not qualify this requested seed")
    if connected_components(atoms) != [args.n]:
        raise ValueError("prepared input is fragmented at 1.3sigma; not an intact-cluster escape control")
    atoms.set_cell(CELL)
    atoms.set_pbc(False)
    if not geometry_gate(atoms)["eligible"]:
        raise ValueError("prepared common input fails 100 A geometry gate")
    args.output.mkdir(parents=True, exist_ok=False)
    runner_hash = freeze_runner(args.output)
    nseed = SEEDS[args.n]
    arm = args.arm
    cap = 120 if args.native_probe else args.cap
    wall = args.wall
    steps = 1 if args.native_probe else args.steps
    if args.native_probe and arm != "native":
        raise ValueError("--native-probe requires --arm native")
    write(args.output / "input.extxyz", atoms)
    common = {"mode": "native_probe" if args.native_probe else "execute", "arm": arm,
        "n": args.n, "seed": nseed, "input": str(args.input),
        "input_sha256": sha256(args.input), "preparation_result": str(case_result),
        "preparation_qualification": qualification, "cap": cap,
        "wall_seconds": wall, "steps": steps, "settings": campaign_config(),
        "imports": sources(), "binary": str(NATIVE_BINARY), "mpi_lib": MPI_LIB,
        "binary_sha256": sha256(NATIVE_BINARY) if NATIVE_BINARY.is_file() else None,
        "runner_sha256": runner_hash,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    common["settings"]["recovered_direction"]["c1_radius_policy"] = args.c1_radius_policy
    json_write(args.output / "provenance.json", common)
    try:
        if arm == "native":
            execute_native(args.output, atoms, args.n, steps, cap, wall)
        else:
            execute_python(args.output, atoms, args.n, nseed, arm, steps, cap, wall, args.c1_radius_policy)
    except Exception as exc:
        json_write(args.output / "failure.json", {"error": repr(exc),
            "traceback": traceback.format_exc(), "arm": arm,
            "surface_requests": getattr(locals().get("surface", None), "requests", None)})
        raise


def execute_python(out, atoms, n, seed, arm, steps, cap, wall, c1_radius_policy="restricted"):
    from ase.io import write
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import (run_ssw, RecoveredRotationSettings,
        ConservativeNativeHeightPolicy, NativeMCSettings)
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    config, rotation, direction, height, mc, _temp = settings()
    direction = replace(direction, c1_radius_policy=c1_radius_policy)
    surface = CountedSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
                             out / "ef-ledger.jsonl", cap=cap,
                             deadline=time.monotonic() + wall)
    options = {"recovered_rotation": rotation if arm == "rotation" else None,
               "recovered_direction": direction if arm == "full" else None,
               "height_policy": height, "mc": mc}
    events, minima = [], []
    def progress(p):
        row = {"kind": p.kind, "next_index": p.next_index,
            "evaluation_requests": p.evaluation_requests,
            "current_energy_eV": float(p.current_energy),
            "best_energy_eV": float(p.best.energy)}
        if p.step is not None:
            rec = p.step
            row.update(status=rec.status, accepted=bool(rec.accepted),
                outer_index=int(rec.index),
                step_requests=int(rec.evaluation_requests),
                landing_energy_eV=None if rec.landing is None else float(rec.landing.energy),
                climb=[dict(x) for x in (rec.climb or [])])
        if p.new_minimum is not None:
            m = p.new_minimum
            idx = len(minima)
            write(out / f"minimum-{idx:04d}.extxyz", m.atoms)
            minima.append({"index": idx, "energy_eV": float(m.energy),
                "fmax_eV_A": float(m.max_force), "converged": bool(m.converged),
                "geometry_gate": geometry_gate(m.atoms),
                "target_energy_candidate": bool(m.converged and m.max_force <= FMAX and
                    m.energy <= REFERENCE[n] + HIT_TOL)})
            row["new_minimum_index"] = idx
        events.append(row)
        with (out / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(load_serializer()._jsonable(row), allow_nan=False) + "\n")
        return False
    result = run_ssw(atoms.copy(), surface, steps=steps, config=config,
        rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1]),
        progress_callback=progress, **options)
    if result.checkpoint is not None:
        save_ssw_checkpoint(out / "checkpoint.pkl", result.checkpoint)
    best_quench = result.checkpoint.best
    write(out / "best.extxyz", result.best)
    fresh = CountedSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
                           out / "fresh-ef.jsonl", cap=2)
    fresh_rows = []
    for label, quench in (("initial", result.initial), ("best", best_quench)):
        state = quench.atoms
        if (not quench.converged or not np.isfinite(quench.energy) or
                not np.isfinite(quench.max_force) or quench.max_force > FMAX):
            fresh_rows.append({"label": label, "status": "skipped_not_qualified",
                               "energy_eV": float(quench.energy), "fmax_eV_A": float(quench.max_force)})
            continue
        try:
            energy, forces = fresh.evaluate(state)
            fresh_rows.append({"label": label, "energy_eV": float(energy),
                "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                "calls": fresh.requests})
        except Exception as exc:
            fresh_rows.append({"label": label, "error": repr(exc), "calls": fresh.requests})
    json_write(out / "result.json", {"status": result.status,
        "evaluation_requests": int(result.evaluation_requests),
        "surface_requests": surface.requests,
        "initial": {"energy_eV": float(result.initial.energy),
            "fmax_eV_A": float(result.initial.max_force),
            "converged": bool(result.initial.converged)},
        "best_energy_eV": float(best_quench.energy),
        "fresh": fresh_rows, "fresh_requests": fresh.requests, "minima": minima,
        "outer_events": events, "best_geometry_gate": geometry_gate(result.best),
        "configuration": {"ssw": asdict(config),
            "recovered_rotation": asdict(rotation) if arm == "rotation" else None,
            "recovered_direction": asdict(direction) if arm == "full" else None,
            "height": asdict(height), "mc": asdict(mc), "ls": None}})


def execute_native(out, atoms, n, steps, cap, wall):
    from ase import Atoms
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from research.ga_ssw.lasp_external_ase import run_lasp
    if not NATIVE_BINARY.is_file():
        raise FileNotFoundError(NATIVE_BINARY)
    case = out
    (case / "input.arc").write_text(make_arc(atoms))
    (case / "lasp.in").write_text(native_input_text(n, steps))
    (case / "client.py").write_bytes((NATIVE_PACKAGE / "client.py").read_bytes())
    (case / "bounded_process.py").write_bytes((NATIVE_PACKAGE / "bounded_process.py").read_bytes())
    (case / "lasp.external.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n" +
        "if [ ! -f socket-first.txt ]; then\n"
        "  { printf '%s\\n' \"$LASP_MACE_SOCKET\"; ls -l \"$LASP_MACE_SOCKET\"; } > socket-first.txt 2>&1\n"
        "fi\n" + f"{sys.executable} {case / 'client.py'}\n")
    (case / "lasp.external.sh").chmod(0o755)
    template = atoms.copy()
    template.set_cell(CELL); template.set_pbc(False)
    calc = FullPairLJ(epsilon=EPSILON, sigma=SIGMA)
    checked = GeometryCheckedCalculator(calc)
    env = os.environ.copy()
    env.update({"LD_LIBRARY_PATH": MPI_LIB + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
                "I_MPI_FABRICS": "shm", "I_MPI_PIN": "0"})
    command = [sys.executable, str(case / "bounded_process.py"), "--cwd", str(case),
        "--timeout", str(wall), "--log", str(case / "stdout.txt"),
        "--status", str(case / "process.json"), "--", "/lib64/ld-linux-x86-64.so.2",
        str(NATIVE_BINARY)]
    # One observed callback could not find its /tmp Unix socket. Use a short,
    # user-owned shared filesystem directory; retain first-client visibility.
    socket_base = Path("/home/gengjianrui/.cache/pam-ssw-external")
    socket_base.mkdir(parents=True, exist_ok=True)
    previous_tempdir = tempfile.tempdir
    try:
        tempfile.tempdir = str(socket_base)
        result = run_lasp(command, cwd=case, atoms=template, calculator=checked,
                          pbc=(False, False, False), max_requests=cap, env=env)
    finally:
        tempfile.tempdir = previous_tempdir
    json_write(case / "external-ef.json", result)
    if (case / "lasp.out").is_file():
        shutil.copy2(case / "lasp.out", case / "lasp.out.raw")
    if (case / "allkeys.log").is_file():
        shutil.copy2(case / "allkeys.log", case / "allkeys.log.raw")
    process = json.loads((case / "process.json").read_text()) if (case / "process.json").exists() else None
    json_write(case / "native-result.json", {"supervisor_state": result["state"],
        "returncode": result["returncode"], "wall_seconds": result["wall_seconds"],
        "process_json": process,
        "native_returncode": None if process is None else process.get("returncode"),
        "cleanup_survivors": None if process is None else process.get("cleanup_survivors"),
        "successful_requests": len(result["requests"]),
        "external_attempts": len(result["requests"]) + len(result["errors"]),
        "calculator_evaluation_attempts": checked.evaluation_attempts,
        "actual_calculator_calls": checked.actual_calculator_calls,
        "calculator_failures": checked.failures, "errors": result["errors"],
        "note": "raw LASP/allkeys output retained; event matching is offline"})


class GeometryCheckedCalculator(Calculator):
    """Apply the protocol geometry domain to each native callback request."""
    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calculator):
        super().__init__()
        self.calculator = calculator
        self.evaluation_attempts = 0
        self.actual_calculator_calls = 0
        self.failures = []

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.evaluation_attempts += 1
        gate = geometry_gate(atoms)
        if not gate["eligible"]:
            self.failures.append({"status": "geometry_gate_failed", "gate": gate})
            raise RuntimeError("geometry_gate_failed:" + gate["reason"])
        work = atoms.copy()
        work.calc = self.calculator
        self.actual_calculator_calls += 1
        try:
            energy = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), dtype=float)
        except Exception as exc:
            self.failures.append({"status": "calculator_failed", "error": repr(exc), "gate": gate})
            raise
        self.results = {"energy": energy, "free_energy": energy, "forces": forces}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--native-probe", action="store_true")
    p.add_argument("--initialization", choices=("dilute", "bulk-density"), default="dilute")
    p.add_argument("--output", type=Path)
    p.add_argument("--arm", choices=("rotation", "full", "native"))
    p.add_argument("--c1-radius-policy", choices=("restricted", "per_atom"), default="restricted")
    p.add_argument("--n", type=int, choices=(38, 55))
    p.add_argument("--input", type=Path)
    p.add_argument("--cap", type=int, default=4000)
    p.add_argument("--wall", type=int, default=600)
    p.add_argument("--steps", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    if args.preflight:
        preflight(args.initialization); return
    if args.prepare:
        if args.output is None: raise SystemExit("--prepare requires --output")
        prepare(args.output, args.initialization); return
    if args.execute or args.native_probe:
        missing = [name for name in ("output", "arm", "n", "input") if getattr(args, name) is None]
        if missing: raise SystemExit("execution requires --output --arm --n --input")
        execute(args)


if __name__ == "__main__":
    main()
