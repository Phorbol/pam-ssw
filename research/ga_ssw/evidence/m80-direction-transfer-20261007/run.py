#!/usr/bin/env python3
"""Bounded, research-only M80 direction-transfer panel.

No PES evaluation occurs during --preflight. Actual searches require an
explicit --slot invocation and are capped per arm and for the full panel.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PANEL_DIR = ROOT / "research/ga_ssw/evidence/ssw-native-lj-panel-20261007"
TARGET_RUN_PATH = ROOT / "research/ga_ssw/evidence/lj55-direction-target-20261007/run.py"
INPUT_DIR = ROOT / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/morse-compact-initials"
INPUT_SUMMARY = INPUT_DIR / "summary.json"
REFERENCE_PATH = ROOT / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/references/morse-80G-rho14.extxyz"
N = 80
R0_A, RHO0, EPSILON = 2.7, 14.0, 1.0
TARGET_E, HIT_TOL, FMAX = -340.811371, 0.001, 0.05
CONNECT_CUTOFF = 1.3 * R0_A
PAIR_SEEDS = (26100731, 26100732)
INPUT_SEEDS = (25092501, 25092502)
SEARCH_CAP, OUTER_CAP, WALL_CAP = 80_000, 100, 600.0
FRESH_CAP, PANEL_FRESH_CAP = 2, 9


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PANEL = load_module(PANEL_DIR / "run_panel.py", "m80_direction_panel_source")
ANALYSIS = load_module(PANEL_DIR / "analyze_panel.py", "m80_direction_panel_analysis")
TARGET_HELPER = load_module(TARGET_RUN_PATH, "m80_direction_lj55_target_helpers")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def imports():
    import ase
    from ase.calculators.morse import MorsePotential
    from pamssw.standalone import run_ssw, ASESurface
    import pamssw.standalone.paper_reference as paper
    import pamssw.standalone.recovered_rotation as rotation
    import pamssw.standalone.recovered_direction as direction
    import pamssw.standalone.native_height_policy as height
    import pamssw.standalone.native_mc as mc
    import research.ga_ssw.lasp_external_ase as external
    serializer_path = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
    return {
        "python": sys.executable, "ase_version": ase.__version__,
        "MorsePotential": str(Path(__import__("ase.calculators.morse", fromlist=["__file__"]).__file__).resolve()),
        "run_ssw": str(Path(paper.__file__).resolve()),
        "standalone_init": str(Path(__import__("pamssw.standalone", fromlist=["__file__"]).__file__).resolve()),
        "RecoveredRotationSettings": str(Path(rotation.__file__).resolve()),
        "RecoveredDirectionSettings": str(Path(direction.__file__).resolve()),
        "ConservativeNativeHeightPolicy": str(Path(height.__file__).resolve()),
        "NativeMCSettings": str(Path(mc.__file__).resolve()),
        "lasp_external_ase": str(Path(external.__file__).resolve()),
        "panel_runner": str(Path(PANEL.__file__).resolve()),
        "panel_analysis": str(Path(ANALYSIS.__file__).resolve()),
        "compact_surface_helper": str(Path(TARGET_HELPER.__file__).resolve()),
        "serializer": str(serializer_path.resolve()),
        "run_ssw_callable": callable(run_ssw), "ASESurface_callable": callable(ASESurface),
        "MorsePotential_callable": callable(MorsePotential),
    }


def config_record():
    config, rotation, direction, height, mc, temp = PANEL.settings()
    direction = replace(direction, c1_radius_policy="per_atom")
    return {
        "purpose": "development functional/cost transfer on short-range Morse; not GM-rate replication",
        "potential": {"class": "ase.calculators.morse.MorsePotential", "epsilon_eV": EPSILON,
                      "rho0": RHO0, "r0_A": R0_A,
                      "rcut1_reduced_r0": 100.0, "rcut2_reduced_r0": 101.0,
                      "rcut1_A": 270.0, "rcut2_A": 272.7,
                      "pbc": False, "constraints": [], "confinement": None,
                      "cutoff_note": "ASE finite neighbor cutoff; settings match prior rho14 qualification. No claim is made for arbitrarily extreme configurations outside the observed compact-cluster domain."},
        "ssw": asdict(config), "rotation": asdict(rotation),
        "full_direction_per_atom": asdict(direction), "height": asdict(height), "mc": asdict(mc),
        "operational_temperature_K": temp,
        "target": {"energy_eV": TARGET_E, "energy_tolerance_eV": HIT_TOL,
                   "force_max_eV_A": FMAX, "connected_cutoff_A": CONNECT_CUTOFF,
                   "reference": str(REFERENCE_PATH), "geometry_method": "analyze_panel.compare_geometry; graph permutation + proper-rotation RMS"},
        "arms": ["rotation", "full_per_atom"], "input_seeds": list(INPUT_SEEDS),
        "paired_search_rng_seeds": list(PAIR_SEEDS),
        "budget_per_arm": {"requests_including_initial_and_failed_work": SEARCH_CAP,
                           "outer_attempts": OUTER_CAP, "wall_seconds": WALL_CAP,
                           "fresh_requests": FRESH_CAP},
        "panel_total": {"search_requests_max": 4 * SEARCH_CAP, "fresh_requests_max": PANEL_FRESH_CAP,
                        "cpu_tasks_max": 2, "arm_runs": 4, "no_repeats_or_continuations": True},
        "inputs": [str(INPUT_DIR / f"m80-{seed:08d}" / "final.extxyz") for seed in INPUT_SEEDS],
        "initial_structures": "use the saved force-qualified final.extxyz from prior random-structure qualification byte-for-byte; no resampling, repair, recentering, or extra preparation",
        "evaluation_domain": "finite coordinates and nonperiodic only; bounding span recorded as diagnostic, never a reject condition",
    }


def input_paths():
    return [INPUT_DIR / f"m80-{seed:08d}" / "final.extxyz" for seed in INPUT_SEEDS]


def target_atoms():
    from ase.io import read
    atoms = read(REFERENCE_PATH)
    if len(atoms) != N or not np.isfinite(atoms.positions).all() or atoms.pbc.any():
        raise ValueError("Morse80 reference violates the finite, 80-atom, nonperiodic contract")
    return atoms


def finite_nonperiodic(atoms):
    x = np.asarray(atoms.positions, dtype=float)
    finite = bool(x.shape == (len(atoms), 3) and np.isfinite(x).all())
    nonperiodic = not bool(np.asarray(atoms.pbc).any())
    span = np.ptp(x, axis=0).tolist() if finite and len(x) else None
    return {"eligible": finite and nonperiodic,
            "reason": "ok" if finite and nonperiodic else "nonfinite_coordinates" if not finite else "periodic_input",
            "span_A_diagnostic_only": span, "rejects_by_span": False}


def connected_components(atoms):
    d = np.asarray(atoms.get_all_distances(mic=False), dtype=float)
    adjacency = (d < CONNECT_CUTOFF) & (d > 0.0)
    todo, sizes = set(range(len(atoms))), []
    while todo:
        stack, size = [todo.pop()], 0
        while stack:
            i = stack.pop()
            size += 1
            new = set(np.flatnonzero(adjacency[i])) & todo
            todo -= new
            stack.extend(new)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def preflight():
    from ase.io import read
    from ase.calculators.calculator import Calculator, all_changes
    cfg, rotation, direction, height, mc, _ = PANEL.settings()
    direction = replace(direction, c1_radius_policy="per_atom")
    inputs = []
    prior = {int(row["seed"]): row for row in json.loads(INPUT_SUMMARY.read_text()) if row.get("n") == N}
    for seed, path in zip(INPUT_SEEDS, input_paths()):
        atoms = read(path)
        gate = finite_nonperiodic(atoms)
        inputs.append({"seed": seed, "path": str(path), "sha256": sha256(path),
                       "atom_count": len(atoms), "components_1p3r0": connected_components(atoms),
                       "pbc": np.asarray(atoms.pbc).tolist(), "span_A_diagnostic_only": gate["span_A_diagnostic_only"],
                       "domain_gate": gate,
                       "prior_random_qualification": prior.get(seed),
                       "prior_qualification_summary": str(INPUT_SUMMARY)})
    ref = target_atoms()
    class Dummy(Calculator):
        implemented_properties = ["energy", "forces"]
        def __init__(self):
            super().__init__(); self.calls = 0
        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.calls += 1
            self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}
    import tempfile
    with tempfile.TemporaryDirectory(prefix="m80-direction-preflight-") as tmp:
        dummy = Dummy()
        surface = TARGET_HELPER.CompactSurface(dummy, Path(tmp) / "ledger.jsonl", cap=2)
        from ase import Atoms
        one = Atoms("Ar", positions=[[0.0, 0.0, 0.0]], pbc=False)
        surface.evaluate(one); surface.evaluate(one.copy())
        denied = False
        try:
            surface.evaluate(one)
        except RuntimeError as exc:
            denied = "request_cap_denied" in str(exc)
        rows = [json.loads(s) for s in (Path(tmp) / "ledger.jsonl").read_text().splitlines()]
        dummy_contract = bool(denied and surface.requests == 2 and surface.denied == 1 and
                              dummy.calls == 1 and len(rows) == 3 and rows[-1]["charged"] is False)
    checks = {"run_ssw": callable(__import__("pamssw.standalone", fromlist=["run_ssw"]).run_ssw),
              "reference_atom_count": len(ref), "dummy_surface_contract": dummy_contract,
              "panel_settings_importable": True, "real_pes_requests": 0,
              "morse_calculator_constructed_or_evaluated": False}
    print(json.dumps({"status": "preflight_ok", "real_pes_requests": 0,
                      "config": config_record(), "imports": imports(), "inputs": inputs,
                      "reference_path": str(REFERENCE_PATH), "reference_sha256": sha256(REFERENCE_PATH),
                      "reference_atom_count": len(ref), "panel_settings": {
                          "ssw": asdict(cfg), "rotation": asdict(rotation),
                          "full_direction_per_atom": asdict(direction), "height": asdict(height),
                          "mc": asdict(mc)}, "checks": checks}, indent=2, allow_nan=False))
    if (not dummy_contract or any(x["atom_count"] != N or not x["domain_gate"]["eligible"] for x in inputs)):
        raise SystemExit("preflight contract failed")


def snapshot(out: Path, source):
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__).resolve(), out / "run.py")
    shutil.copy2(HERE / "protocol.md", out / "protocol.md")
    depdir = out / "source-snapshot"
    depdir.mkdir()
    dependencies = {}
    for label, source_path in source.items():
        p = Path(source_path)
        if p.is_file():
            dest = depdir / (label.replace(".", "_") + "__" + p.name)
            shutil.copy2(p, dest)
            dependencies[label] = {"import_path": str(p), "sha256": sha256(p),
                                   "snapshot": str(dest.relative_to(out))}
    return dependencies


def source_info():
    from ase.calculators.morse import MorsePotential
    import ase.calculators.morse as morse
    import pamssw.standalone as standalone
    import pamssw.standalone.paper_reference as paper
    import pamssw.standalone.recovered_rotation as rotation
    import pamssw.standalone.recovered_direction as direction
    import pamssw.standalone.native_height_policy as height
    import pamssw.standalone.native_mc as mc
    import research.ga_ssw.evidence
    ledger = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
    paths = {"morse_calculator": morse.__file__, "standalone_init": standalone.__file__,
             "paper_reference": paper.__file__, "recovered_rotation": rotation.__file__,
             "recovered_direction": direction.__file__, "native_height_policy": height.__file__,
             "native_mc": mc.__file__, "panel_runner": PANEL.__file__,
             "panel_analysis": ANALYSIS.__file__, "compact_surface_helper": TARGET_HELPER.__file__,
             "serializer": ledger}
    return {"checkout": str(ROOT), "git_head": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "head_pamssw_tree": subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD:pamssw"], text=True).strip(),
        "python": sys.executable, "imports": {k: str(Path(v).resolve()) for k, v in paths.items()},
        "calculator_class": f"{MorsePotential.__module__}.{MorsePotential.__name__}",
        "dirty_pamssw_status": subprocess.run(["git", "-C", str(ROOT), "status", "--short", "--", "pamssw"],
            capture_output=True, text=True, check=True).stdout.splitlines()}, paths


def qualified_morse(atoms, energy, fmax, converged, reference):
    energy_ok = bool(np.isfinite(energy) and energy <= TARGET_E + HIT_TOL)
    force_ok = bool(np.isfinite(fmax) and fmax <= FMAX)
    connected = connected_components(atoms) if force_ok and converged else None
    geom = (ANALYSIS.compare_geometry(reference, atoms)
            if energy_ok and connected == [N] else
            {"classification": "inconclusive", "reason": "prefilter_not_passed"})
    return {"energy_eV": float(energy), "energy_gate": energy_ok,
            "fmax_eV_A": float(fmax), "force_gate_0p05": force_ok,
            "converged": bool(converged), "connected_components_1p3r0": connected,
            "connected_gate": connected == [N],
            "connected_force_qualified": bool(converged and force_ok and connected == [N]),
            "geometry_to_reference": geom,
            "qualified_candidate": bool(energy_ok and force_ok and converged and
                                        connected == [N] and geom.get("classification") == "same")}


def run_arm(out: Path, slot: int, input_seed: int, rng_seed: int, arm: str, input_path: Path,
            reference, cfg, rotation, direction, height, mc):
    from ase.io import read, write
    from ase.calculators.morse import MorsePotential
    from pamssw.standalone import run_ssw
    from pamssw.standalone.paper_reference import save_ssw_checkpoint
    source, dep_paths = source_info()
    dependencies = snapshot(out, dep_paths)
    atoms = read(input_path)
    domain = finite_nonperiodic(atoms)
    if len(atoms) != N or not domain["eligible"]:
        raise ValueError(f"input contract failed: {input_path}: {domain}")
    shutil.copy2(input_path, out / "input.extxyz")
    prior_rows = {int(row["seed"]): row for row in json.loads(INPUT_SUMMARY.read_text()) if row.get("n") == N}
    prior = prior_rows[input_seed]
    provenance = {"mode": "single_arm", "slot": slot, "input_seed": input_seed,
        "arm": arm, "paired_rng_seed": rng_seed, "started_unix_time": time.time(),
        "input_path": str(input_path), "input_sha256": sha256(input_path),
        "prior_random_qualification_summary": str(INPUT_SUMMARY),
        "prior_random_qualification": prior, "reference_path": str(REFERENCE_PATH),
        "reference_sha256": sha256(REFERENCE_PATH), "imports": source,
        "dependency_snapshot": dependencies, "runner_sha256": sha256(out / "run.py"),
        "protocol_sha256": sha256(out / "protocol.md"), "effective_config": config_record(),
        "git_head": source["git_head"], "head_pamssw_tree": source["head_pamssw_tree"]}
    dump(out / "provenance.json", provenance)
    policy = rotation if arm == "rotation" else None
    direction_policy = direction if arm == "full_per_atom" else None
    shared_reference = {"status": "not_repeated_for_this_slot",
        "existing_independent_qualification": str(INPUT_DIR.parent / "morse-reference-qualification.json")}
    shared_reference_requests = 0
    if slot == 0:
        ref_fresh = TARGET_HELPER.CompactSurface(
            MorsePotential(epsilon=EPSILON, rho0=RHO0, r0=R0_A, rcut1=100., rcut2=101.),
            out / "shared-reference-fresh-ledger.jsonl", cap=1)
        try:
            er, fr = ref_fresh.evaluate(reference)
            fmr = float(np.linalg.norm(fr, axis=1).max())
            shared_reference = {"status": "fresh_qualified" if abs(er - TARGET_E) < 1e-5 and fmr <= FMAX else "fresh_failed",
                "energy_eV": float(er), "expected_energy_eV": TARGET_E,
                "energy_delta_eV": float(er - TARGET_E), "fmax_eV_A": fmr,
                "passed": bool(abs(er - TARGET_E) < 1e-5 and fmr <= FMAX),
                "requests": ref_fresh.requests, "actual_calculator_calls": ref_fresh.actual_calls}
        except Exception as exc:
            shared_reference = {"status": "fresh_failed", "error": repr(exc),
                                "requests": ref_fresh.requests}
        shared_reference_requests = ref_fresh.requests
        dump(out / "shared-reference-qualification.json", shared_reference)
    else:
        ref_qual_path = INPUT_DIR.parent / "morse-reference-qualification.json"
        reference_prior = next((row for row in json.loads(ref_qual_path.read_text())
                               if row.get("label") == "80G"), None)
        shared_reference["existing_independent_qualification_sha256"] = sha256(ref_qual_path)
        shared_reference["existing_independent_qualification"] = reference_prior
        dump(out / "shared-reference-qualification.json", shared_reference)
    surface = TARGET_HELPER.CompactSurface(
        MorsePotential(epsilon=EPSILON, rho0=RHO0, r0=R0_A, rcut1=100., rcut2=101.),
        out / "search-ef-ledger.jsonl", cap=SEARCH_CAP, deadline=time.monotonic() + WALL_CAP,
        failures_dir=out / "failed-geometries")
    events, seen_steps, candidate = [], set(), None
    qualified_minima = []
    seen_minima = 0
    previous = atoms.copy()
    progress_path = out / "progress.jsonl"
    trajectory = out / "outer-structures.extxyz"
    def frame(role, index, structure):
        if structure is None:
            return None
        q = structure.copy()
        q.info.update(event_role=role, event_index=int(index))
        write(trajectory, q, format="extxyz", append=trajectory.exists())
        return f"outer-structures.extxyz#{role}:{index}"
    def emit(row):
        events.append(row)
        with progress_path.open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    def progress(p):
        nonlocal previous, candidate, seen_minima
        if p.kind == "initial":
            minimum_ref, minimum_gate = None, None
            if p.new_minimum is not None:
                seen_minima += 1
                minimum_ref = frame("minimum", -1, p.new_minimum.atoms)
                minimum_gate = qualified_morse(p.new_minimum.atoms, p.new_minimum.energy,
                    p.new_minimum.max_force, p.new_minimum.converged, reference)
                if minimum_gate["connected_force_qualified"]:
                    qualified_minima.append({"outer_index": -1,
                        "cumulative_paid_requests": int(p.evaluation_requests),
                        "energy_eV": float(p.new_minimum.energy),
                        "fmax_eV_A": float(p.new_minimum.max_force), "structure": minimum_ref,
                        "reference_geometry": minimum_gate["geometry_to_reference"]})
                if minimum_gate["qualified_candidate"]:
                    candidate = {"event_index": -1, "qualification": minimum_gate,
                                 "atoms": p.new_minimum.atoms.copy()}
            emit({"kind": "initial", "evaluation_requests": int(p.evaluation_requests),
                  "energy_eV": float(p.best.energy), "fmax_eV_A": float(p.best.max_force),
                  "converged": bool(p.best.converged), "geometry": frame("initial", -1, p.best.atoms),
                  "minimum_structure": minimum_ref, "minimum_gate": minimum_gate})
            previous = p.current.copy()
            return candidate is not None
        step = p.step
        idx = int(step.index)
        seen_steps.add(idx)
        start_ref = frame("start", idx, previous)
        landing = step.landing
        landing_atoms = None if landing is None else landing.atoms
        landing_ref = frame("landing", idx, landing_atoms)
        landing_fmax = None if landing is None else float(landing.max_force)
        landing_connected = (connected_components(landing_atoms) if landing is not None and
                             landing.converged and np.isfinite(landing_fmax) and landing_fmax <= FMAX else None)
        new_min = p.new_minimum
        minimum_ref = None if new_min is None else frame("minimum", idx, new_min.atoms)
        minimum_gate = None
        if new_min is not None:
            seen_minima += 1
            minimum_gate = qualified_morse(new_min.atoms, new_min.energy,
                new_min.max_force, new_min.converged, reference)
            if minimum_gate["connected_force_qualified"]:
                qualified_minima.append({"outer_index": idx,
                    "cumulative_paid_requests": int(p.evaluation_requests),
                    "energy_eV": float(new_min.energy), "fmax_eV_A": float(new_min.max_force),
                    "structure": minimum_ref, "reference_geometry": minimum_gate["geometry_to_reference"]})
            if minimum_gate["qualified_candidate"]:
                candidate = {"event_index": idx, "qualification": minimum_gate,
                             "atoms": new_min.atoms.copy()}
        current_ref = frame("current", idx, p.current)
        emit({"kind": "outer_step", "index": idx, "status": step.status,
              "accepted": bool(step.accepted), "step_requests": int(step.evaluation_requests),
              "cumulative_requests": int(p.evaluation_requests), "start_structure": start_ref,
              "landing_structure": landing_ref, "landing_energy_eV": None if landing is None else float(landing.energy),
              "landing_fmax_eV_A": landing_fmax, "landing_converged": None if landing is None else bool(landing.converged),
              "landing_components_1p3r0": landing_connected,
              "landing_qualified_fmax_and_connected": bool(landing is not None and landing.converged and
                  landing_fmax <= FMAX and landing_connected == [N]),
              "minimum_structure": minimum_ref, "minimum_gate": minimum_gate,
              "current_structure": current_ref,
              "climb_summary": [{k: x.get(k) for k in ("index", "status", "requests", "rotation_force_requests", "quench_requests", "error") if k in x}
                                for x in (step.climb or [])]})
        previous = p.current.copy()
        return candidate is not None
    started = time.monotonic()
    error = None
    result = None
    try:
        result = run_ssw(atoms.copy(), surface, steps=OUTER_CAP, config=cfg,
            rng=np.random.default_rng(rng_seed), progress_callback=progress,
            recovered_rotation=policy, recovered_direction=direction_policy,
            height_policy=height, mc=mc)
    except Exception as exc:
        error = {"error": repr(exc), "traceback": traceback.format_exc()}
    # Some terminal failures finish after paid work but before the safe callback.
    if result is not None:
        tail_paid = int(result.initial.evaluation_requests)
        for rec in result.records:
            tail_paid += int(rec.evaluation_requests)
            if rec.index in seen_steps:
                continue
            landing = rec.landing
            latoms = None if landing is None else landing.atoms
            lref = frame("terminal_landing", rec.index, latoms)
            lforce = None if landing is None else float(landing.max_force)
            lcomp = (connected_components(latoms) if landing is not None and landing.converged and
                     lforce is not None and lforce <= FMAX else None)
            minimum_ref, gate = None, None
            if seen_minima < len(result.minima):
                minimum = result.minima[seen_minima]; seen_minima += 1
                minimum_ref = frame("terminal_minimum", rec.index, minimum.atoms)
                gate = qualified_morse(minimum.atoms, minimum.energy, minimum.max_force,
                                       minimum.converged, reference)
                if gate["connected_force_qualified"]:
                    qualified_minima.append({"outer_index": int(rec.index),
                        "cumulative_paid_requests": tail_paid, "energy_eV": float(minimum.energy),
                        "fmax_eV_A": float(minimum.max_force), "structure": minimum_ref,
                        "reference_geometry": gate["geometry_to_reference"]})
            emit({"kind": "outer_step", "index": int(rec.index), "status": rec.status,
                  "accepted": bool(rec.accepted), "step_requests": int(rec.evaluation_requests),
                  "cumulative_requests": tail_paid, "progress_callback_observed": False,
                  "start_structure": frame("terminal_start", rec.index, previous),
                  "landing_structure": lref, "landing_energy_eV": None if landing is None else float(landing.energy),
                  "landing_fmax_eV_A": lforce, "landing_converged": None if landing is None else bool(landing.converged),
                  "landing_components_1p3r0": lcomp,
                  "landing_qualified_fmax_and_connected": bool(landing is not None and landing.converged and
                      lforce is not None and lforce <= FMAX and lcomp == [N]),
                  "minimum_structure": minimum_ref, "minimum_gate": gate,
                  "current_structure": frame("terminal_current", rec.index, result.current)})
            seen_steps.add(rec.index)
    if result is not None and result.checkpoint is not None:
        save_ssw_checkpoint(out / "last-checkpoint.pkl", result.checkpoint)
    if result is not None:
        write(out / "best.extxyz", result.best)
    search_elapsed = time.monotonic() - started
    fresh = TARGET_HELPER.CompactSurface(
        MorsePotential(epsilon=EPSILON, rho0=RHO0, r0=R0_A, rcut1=100., rcut2=101.),
        out / "fresh-ef-ledger.jsonl", cap=FRESH_CAP, failures_dir=out / "failed-geometries")
    fresh_rows = []
    if result is not None:
        states = [("initial", result.initial)]
        best_q = result.checkpoint.best if result.checkpoint is not None else result.initial
        states.append(("best", best_q))
        for label, q in states:
            try:
                e, f = fresh.evaluate(q.atoms)
                fm = float(np.linalg.norm(f, axis=1).max())
                gate = qualified_morse(q.atoms, e, fm, q.converged, reference)
                fresh_rows.append({"role": label, "energy_eV": float(e), "fmax_eV_A": fm,
                                   "requests": fresh.requests, "qualification": gate})
            except Exception as exc:
                fresh_rows.append({"role": label, "error": repr(exc), "requests": fresh.requests})
    elapsed = time.monotonic() - started
    payload = {"slot": slot, "input_seed": input_seed, "input": str(input_path),
        "input_sha256": sha256(input_path), "arm": arm, "paired_rng_seed": rng_seed,
        "prior_random_qualification": prior,
        "prior_random_qualification_summary": str(INPUT_SUMMARY),
        "shared_reference_qualification": shared_reference,
        "shared_reference_requests": shared_reference_requests,
        "git_head": source["git_head"], "head_pamssw_tree": source["head_pamssw_tree"],
        "imports": source["imports"], "dependency_snapshot": dependencies,
        "runner_sha256": sha256(out / "run.py"), "protocol_sha256": sha256(out / "protocol.md"),
        "settings": config_record(), "search_requests": surface.requests,
        "search_actual_calculator_calls": surface.actual_calls, "search_denied_requests": surface.denied,
        "search_wall_seconds": search_elapsed, "run_plus_fresh_seconds": elapsed,
        "initial_quench_requests": None if result is None else int(result.initial.evaluation_requests),
        "outer_requests": sum(int(r.evaluation_requests) for r in result.records if r.index >= 0) if result is not None else None,
        "request_closure": None if result is None else bool(surface.requests == result.initial.evaluation_requests +
            sum(r.evaluation_requests for r in result.records if r.index >= 0)),
        "candidate_pause_confirmed": bool(candidate is not None and result is not None and result.status == 'paused'),
        "outer_callbacks": sum(e.get("kind") == "outer_step" for e in events),
        "outer_status_counts": {s: sum(e.get("kind") == "outer_step" and e.get("status") == s for e in events)
                                for s in sorted({e.get("status") for e in events if e.get("kind") == "outer_step"})},
        "result_status": None if result is None else result.status,
        "best_search_energy_eV": None if result is None else float(
            result.checkpoint.best.energy if result.checkpoint is not None else result.initial.energy),
        "initial_search": None if result is None else {"energy_eV": float(result.initial.energy),
                           "fmax_eV_A": float(result.initial.max_force), "converged": bool(result.initial.converged)},
        "candidate": None if candidate is None else {"event_index": candidate["event_index"],
                                                        "qualification": candidate["qualification"]},
        "connected_force_qualified_minima": qualified_minima,
        "best_connected_force_qualified_minimum": (min(qualified_minima, key=lambda x: x["energy_eV"])
                                                    if qualified_minima else None),
        "fresh": fresh_rows, "fresh_requests": fresh.requests, "error": error}
    dump(out / "result.json", payload)
    if error is not None:
        dump(out / "failure.json", error)
    return {k: payload[k] for k in ("slot", "input_seed", "arm", "paired_rng_seed", "search_requests",
             "search_actual_calculator_calls", "search_wall_seconds", "outer_callbacks",
             "result_status", "candidate", "fresh_requests", "error")}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--slot", type=int, choices=range(4))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.output is not None:
            parser.error("--preflight takes no output path")
        preflight()
    else:
        if args.output is None:
            parser.error("--slot requires --output")
        slot = int(args.slot)
        input_index = slot // 2
        arm = "rotation" if slot % 2 == 0 else "full_per_atom"
        source_seed = INPUT_SEEDS[input_index]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cfg, rotation, direction, height, mc, _ = PANEL.settings()
        direction = replace(direction, c1_radius_policy="per_atom")
        run_arm(args.output, slot, source_seed, PAIR_SEEDS[input_index], arm,
                input_paths()[input_index], target_atoms(), cfg, rotation,
                direction, height, mc)


if __name__ == "__main__":
    main()
