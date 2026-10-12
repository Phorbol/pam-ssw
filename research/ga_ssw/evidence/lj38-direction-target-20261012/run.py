#!/usr/bin/env python3
"""Paired full-budget LJ38 direction transfer; PES work is explicit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
sys.path.insert(0, str(ROOT))
ARCHIVE_BASE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
SOURCE_RUNNER = ROOT / "research/ga_ssw/evidence/lj55-direction-target-20261007/run.py"
REFERENCE_PATH = ROOT / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/references/lj38.points"
OLD_RUNS = ARCHIVE_BASE / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/compact-global-runs"
SEEDS = (25092501, 25092502)
N, SIGMA, EPSILON = 38, 2.7, 1.0
FMAX, TARGET_E, HIT_TOL = 0.05, -173.928427, 0.001
SEARCH_CAP, OUTER_CAP, WALL_CAP = 800_000, 2_000, 1_800.0
FRESH_CAP = 2


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_module(SOURCE_RUNNER, "lj55_target_reusable_runner")
BASE.N = N
BASE.SEEDS = SEEDS
BASE.REFERENCE_PATH = REFERENCE_PATH
BASE.SIGMA, BASE.EPSILON, BASE.FMAX = SIGMA, EPSILON, FMAX
BASE.TARGET_E, BASE.HIT_TOL = TARGET_E, HIT_TOL
BASE.SEARCH_CAP, BASE.OUTER_CAP, BASE.WALL_CAP = SEARCH_CAP, OUTER_CAP, WALL_CAP
BASE.FRESH_CAP = FRESH_CAP
PANEL, ANALYSIS = BASE.PANEL, BASE.ANALYSIS


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_path(seed: int) -> Path:
    return OLD_RUNS / f"lj38-seed{seed}" / "initial.extxyz"


def protocol_config():
    record = BASE.config_record()
    record["initialization"] = (
        "archived LJ38 raw uniform-volume input; each arm performs and pays its own initial quench"
    )
    record["budgets"] = {
        "search_surface_requests_including_initial_quench": SEARCH_CAP,
        "completed_outer_steps": OUTER_CAP,
        "wall_seconds": WALL_CAP,
        "fresh_requests_per_arm": FRESH_CAP,
    }
    record["paired_design"] = {
        "seeds": list(SEEDS),
        "arms": ["rotation", "full_per_atom"],
        "only_changed_setting": "recovered rotation versus recovered direction per_atom",
        "same_raw_input_sha256_within_seed": True,
    }
    return record


def imports_record():
    record = BASE.source_info()
    source_paths = {key: value for key, value in record.items()
                    if key in {"FullPairLJ", "run_ssw", "RecoveredRotationSettings",
                               "RecoveredDirectionSettings", "ConservativeNativeHeightPolicy",
                               "NativeMCSettings", "panel_runner", "panel_analysis"}}
    record.update({
        "source_checkout": str(ROOT),
        "archived_input_checkout": str(ARCHIVE_BASE),
        "source_run_py": str(SOURCE_RUNNER),
        "source_run_py_sha256": sha256(SOURCE_RUNNER),
        "lj38_reference_sha256": sha256(REFERENCE_PATH),
        "raw_input_paths": {str(seed): str(input_path(seed)) for seed in SEEDS},
        "raw_input_sha256": {str(seed): sha256(input_path(seed)) for seed in SEEDS},
        "imported_source_sha256": {key: sha256(Path(path)) for key, path in source_paths.items()},
        "campaign_source_sha256": {path: sha256(ROOT / path) for path in CAMPAIGN_SOURCE_PATHS
                                    if (ROOT / path).is_file()},
    })
    return record


def source_status(paths):
    return subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", *paths],
                                   text=True).strip()


CORE_SOURCE_PATHS = [
    "pamssw/standalone/paper_reference.py",
    "pamssw/standalone/recovered_rotation.py",
    "pamssw/standalone/recovered_direction.py",
    "pamssw/standalone/native_height_policy.py",
    "pamssw/standalone/native_mc.py",
    "research/ga_ssw/full_pair_lj.py",
    "research/ga_ssw/evidence/ssw-native-lj-panel-20261007/run_panel.py",
    "research/ga_ssw/evidence/ssw-native-lj-panel-20261007/analyze_panel.py",
    "research/ga_ssw/evidence/lj55-direction-target-20261007/run.py",
]
CAMPAIGN_SOURCE_PATHS = [
    "research/ga_ssw/evidence/lj38-direction-target-20261012/run.py",
    "research/ga_ssw/evidence/lj38-direction-target-20261012/protocol.md",
    "research/ga_ssw/evidence/lj38-direction-target-20261012/panel.sbatch",
    "research/ga_ssw/evidence/lj38-direction-target-20261012/reference.sbatch",
    "research/ga_ssw/evidence/lj38-direction-target-20261012/smoke.sbatch",
    "research/ga_ssw/evidence/lj38-direction-target-20261012/analyze.py",
]


def require_committed_sources():
    dirty_core = source_status(CORE_SOURCE_PATHS)
    dirty_campaign = source_status(CAMPAIGN_SOURCE_PATHS)
    if dirty_core or dirty_campaign:
        raise RuntimeError("commit campaign and imported sources before numerical qualification/search; "
                           f"dirty_core={dirty_core!r}, dirty_campaign={dirty_campaign!r}")


def target_atoms():
    from ase import Atoms
    xyz = np.loadtxt(REFERENCE_PATH, dtype=float)
    if xyz.shape != (N, 3) or not np.isfinite(xyz).all():
        raise ValueError(f"invalid LJ38 reference shape/content: {xyz.shape}")
    atoms = Atoms(f"Ar{N}", positions=SIGMA * xyz, cell=np.eye(3) * 100., pbc=False)
    atoms.positions += 50. - atoms.positions.mean(axis=0)
    return atoms


def arm_for_slot(slot: int):
    if slot not in range(4):
        raise ValueError("slot must be 0..3")
    return SEEDS[slot // 2], "rotation" if slot % 2 == 0 else "full_per_atom"


def snapshot(out: Path):
    shutil.copy2(Path(__file__).resolve(), out / "run.py")
    shutil.copy2(HERE / "protocol.md", out / "protocol.md")


def preflight():
    from ase.io import read
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import run_ssw, ASESurface
    target = target_atoms()
    cfg, rotation, direction, height, mc, temperature = PANEL.settings()
    from dataclasses import replace
    direction = replace(direction, c1_radius_policy="per_atom")
    inputs = []
    for seed in SEEDS:
        path = input_path(seed)
        atoms = read(path)
        inputs.append({"seed": seed, "path": str(path), "sha256": sha256(path),
                       "natoms": len(atoms), "pbc": atoms.pbc.tolist(),
                       "finite": bool(np.isfinite(atoms.positions).all()),
                       "positions_sha256": hashlib.sha256(atoms.positions.tobytes()).hexdigest()})
    directions = [arm_for_slot(i) for i in range(4)]
    checks = {
        "runner_imports": callable(run_ssw) and callable(ASESurface),
        "calculator_type": issubclass(FullPairLJ, __import__("ase.calculators.calculator", fromlist=["Calculator"]).Calculator),
        "reference_shape": target.get_positions().shape == (N, 3),
        "reference_finite_nonperiodic": bool(np.isfinite(target.positions).all() and not target.pbc.any()),
        "reference_graph_connected": PANEL.connected_components(target) == [N],
        "reference_matches_itself": ANALYSIS.compare_geometry(target, target)["classification"] == "same",
        "per_atom_is_frozen": direction.c1_radius_policy == "per_atom",
        "target_contract": {"epsilon_eV": EPSILON, "sigma_A": SIGMA,
                            "energy_eV": TARGET_E, "energy_tolerance_eV": HIT_TOL,
                            "search_fmax_eV_A": FMAX},
        "settings": {"ssw": asdict(cfg), "rotation": asdict(rotation),
                     "full_direction_per_atom": asdict(direction), "height": asdict(height),
                     "mc": asdict(mc), "equivalent_temperature_K": temperature},
        "slot_map": [{"slot": slot, "seed": seed, "arm": arm}
                     for slot, (seed, arm) in enumerate(directions)],
        "no_pes_requests": True,
    }
    for item in inputs:
        checks[f"seed_{item['seed']}_valid"] = item["natoms"] == N and item["finite"] and item["pbc"] == [False] * 3
    for seed in SEEDS:
        global_path = input_path(seed)
        paired_paper = ARCHIVE_BASE / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/compact-paper-runs" / f"lj38-seed{seed}" / "initial.extxyz"
        checks[f"seed_{seed}_archived_pair_identical"] = sha256(global_path) == sha256(paired_paper)
    result = {"status": "preflight_ok" if all(v for k, v in checks.items() if isinstance(v, bool)) else "preflight_failed",
              "pes_requests": 0, "inputs": inputs, "checks": checks,
              "imports": imports_record(), "protocol": protocol_config()}
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["status"] != "preflight_ok":
        raise SystemExit("zero-PES preflight failed")


def qualify_reference(output: Path):
    """Independently evaluate the published structure on the campaign LJ calculator."""
    require_committed_sources()
    from research.ga_ssw.full_pair_lj import FullPairLJ
    atoms = target_atoms()
    calc = FullPairLJ(epsilon=EPSILON, sigma=SIGMA)
    atoms.calc = calc
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max())
    max_component = float(np.abs(forces).max())
    payload = {
        "status": "qualified" if abs(energy - TARGET_E) < 1.0e-5 and
                  max_component < 0.04 / SIGMA else "failed",
        "requests": 1,
        "input": str(REFERENCE_PATH),
        "input_sha256": sha256(REFERENCE_PATH),
        "potential": {"class": "FullPairLJ", "epsilon_eV": EPSILON,
                      "sigma_A": SIGMA, "cutoff_A": None, "periodic": False},
        "reference_energy_eV": TARGET_E, "energy_eV": energy,
        "energy_tolerance_eV": 1.0e-5, "energy_passed": abs(energy - TARGET_E) < 1.0e-5,
        "fmax_eV_A": fmax, "max_force_component_eV_A": max_component,
        "component_force_limit_eV_A": 0.04 / SIGMA,
        "force_criterion_source": "pre-existing LJ reference qualification protocol; same check as lj-reference-qualification.json",
        "force_passed": max_component < 0.04 / SIGMA,
        "passed": abs(energy - TARGET_E) < 1.0e-5 and max_component < 0.04 / SIGMA,
        "committed_sources": True,
        "imports": imports_record(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    BASE.dump(output, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))
    if not payload["passed"]:
        raise SystemExit("independent reference E/F qualification failed")


def execute(slot: int, output: Path, reference_qualification: Path, *, smoke=False):
    from ase.io import read, write
    from research.ga_ssw.full_pair_lj import FullPairLJ
    from pamssw.standalone import run_ssw
    from pamssw.standalone.paper_reference import save_ssw_checkpoint

    require_committed_sources()
    seed, arm = arm_for_slot(slot)
    source_input = input_path(seed)
    ref_qual = json.loads(reference_qualification.read_text())
    current_imports = imports_record()
    if (not ref_qual.get("passed") or ref_qual.get("input_sha256") != sha256(REFERENCE_PATH) or
            ref_qual.get("potential") != {"class": "FullPairLJ", "epsilon_eV": EPSILON,
                "sigma_A": SIGMA, "cutoff_A": None, "periodic": False} or
            ref_qual.get("imports", {}).get("git_head") != current_imports.get("git_head") or
            ref_qual.get("imports", {}).get("head_pamssw_tree") != current_imports.get("head_pamssw_tree") or
            ref_qual.get("imports", {}).get("imported_source_sha256") != current_imports.get("imported_source_sha256")):
        raise RuntimeError("independent reference qualification missing, failed, or mismatched")
    raw = read(source_input)
    if len(raw) != N or raw.pbc.any() or not np.isfinite(raw.positions).all():
        raise RuntimeError("archived raw input fails finite, 38-atom, nonperiodic contract")
    pair_path = ARCHIVE_BASE / "research/ga_ssw/evidence/cluster-paper-reproduction-20260925/compact-paper-runs" / f"lj38-seed{seed}" / "initial.extxyz"
    if sha256(pair_path) != sha256(source_input):
        raise RuntimeError("archived paired starting inputs differ; refusing search")

    output.mkdir(parents=True, exist_ok=False)
    snapshot(output)
    config = protocol_config()
    run_cap, run_steps, run_wall = ((2_000, 1, 120.0) if smoke else
                                    (SEARCH_CAP, OUTER_CAP, WALL_CAP))
    seed_child = np.random.SeedSequence(seed).spawn(2)[1]
    imports = current_imports
    (output / "input.extxyz").write_bytes(source_input.read_bytes())
    BASE.dump(output / "provenance.json", {
        "mode": "execute", "slot": slot, "seed": seed, "arm": arm,
        "input_path": str(source_input.resolve()), "input_sha256": sha256(source_input),
        "paired_paper_input_path": str(pair_path.resolve()), "paired_input_sha256": sha256(pair_path),
        "reference_qualification_path": str(reference_qualification.resolve()),
        "reference_qualification": ref_qual,
        "imports": imports, "runner_sha256": sha256(output / "run.py"),
        "protocol": config, "started_unix_time": time.time(),
        "run_mode": "one_outer_step_smoke" if smoke else "full_search",
        "effective_run_limits": {"search_surface_requests_including_initial_quench": run_cap,
                                  "completed_outer_steps": run_steps, "wall_seconds": run_wall},
        "rng_child_state": seed_child.generate_state(4).tolist(),
    })

    reference = target_atoms()
    surface = BASE.CompactSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
        output / "search-ef-ledger.jsonl", cap=run_cap,
        deadline=time.monotonic() + run_wall, failures_dir=output / "failed-geometries")
    cfg, rotation, direction, height, mc, _temperature = PANEL.settings()
    from dataclasses import replace
    if arm == "full_per_atom":
        direction = replace(direction, c1_radius_policy="per_atom")

    events, candidate = [], None
    seen_steps, seen_minima = set(), 0
    previous_current = raw.copy()
    trajectory = output / "outer-structures.extxyz"

    def record_frame(role, index, structure):
        if structure is None:
            return None
        frame = structure.copy()
        frame.info.update(event_role=role, event_index=int(index))
        write(trajectory, frame, format="extxyz", append=trajectory.exists())
        return f"outer-structures.extxyz#{role}:{index}"

    def record_event(row):
        events.append(row)
        with (output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def progress(p):
        nonlocal candidate, previous_current, seen_minima
        if p.kind == "initial":
            gate = None
            if p.new_minimum is not None:
                seen_minima += 1
                gate = BASE.common_qualification(p.new_minimum.atoms, p.new_minimum.energy,
                    p.new_minimum.max_force, p.new_minimum.converged, reference)
                if gate["qualified_candidate"]:
                    candidate = {"event_index": -1, "energy_eV": float(p.new_minimum.energy),
                        "fmax_eV_A": float(p.new_minimum.max_force),
                        "evaluation_requests": int(p.evaluation_requests), "geometry": gate,
                        "atoms": p.new_minimum.atoms.copy()}
                    write(output / "target-candidate.extxyz", candidate["atoms"])
            record_event({"kind": "initial", "evaluation_requests": int(p.evaluation_requests),
                "energy_eV": float(p.best.energy), "fmax_eV_A": float(p.best.max_force),
                "converged": bool(p.best.converged), "minimum_gate": gate,
                "connected_component_sizes_1p3sigma": PANEL.connected_components(p.best.atoms),
                "structure": record_frame("initial", -1, p.best.atoms)})
            previous_current = p.current.copy()
            return candidate is not None
        step = p.step
        idx = int(step.index)
        seen_steps.add(idx)
        start_ref = record_frame("start", idx, previous_current)
        landing = None if step.landing is None else step.landing.atoms
        landing_ref = record_frame("landing", idx, landing)
        minimum_ref, candidate_gate = None, None
        if p.new_minimum is not None:
            seen_minima += 1
            minimum_ref = record_frame("minimum", idx, p.new_minimum.atoms)
            minimum_components = PANEL.connected_components(p.new_minimum.atoms)
            candidate_gate = BASE.common_qualification(p.new_minimum.atoms, p.new_minimum.energy,
                p.new_minimum.max_force, p.new_minimum.converged, reference)
            if candidate_gate["qualified_candidate"]:
                candidate = {"event_index": idx, "energy_eV": float(p.new_minimum.energy),
                    "fmax_eV_A": float(p.new_minimum.max_force),
                    "evaluation_requests": int(p.evaluation_requests), "geometry": candidate_gate,
                    "atoms": p.new_minimum.atoms.copy()}
        current_ref = record_frame("current", idx, p.current)
        record_event({"kind": "outer_step", "index": idx, "status": step.status,
            "accepted": bool(step.accepted), "step_requests": int(step.evaluation_requests),
            "cumulative_requests": int(p.evaluation_requests), "start_structure": start_ref,
            "landing_structure": landing_ref, "minimum_structure": minimum_ref,
            "current_structure": current_ref,
            "landing_energy_eV": None if step.landing is None else float(step.landing.energy),
            "landing_fmax_eV_A": None if step.landing is None else float(step.landing.max_force),
            "minimum_gate": candidate_gate,
            "minimum_connected_component_sizes_1p3sigma": (
                minimum_components if p.new_minimum is not None else None),
            "climb_summary": [{k: item.get(k) for k in
                ("index", "status", "requests", "rotation_force_requests", "quench_requests", "error")
                if k in item} for item in (step.climb or [])]})
        previous_current = p.current.copy()
        if candidate is not None:
            write(output / "target-candidate.extxyz", candidate["atoms"])
            return True
        return False

    run_started = time.monotonic()
    result, error = None, None
    try:
        result = run_ssw(raw.copy(), surface, steps=run_steps, config=cfg,
            rng=np.random.default_rng(seed_child), progress_callback=progress,
            recovered_rotation=rotation if arm == "rotation" else None,
            recovered_direction=direction if arm == "full_per_atom" else None,
            height_policy=height, mc=mc)
    except Exception as exc:
        error = {"error": repr(exc), "traceback": traceback.format_exc()}

    # Keep a terminal partial attempt even if the safe-boundary callback did not observe it.
    if result is not None:
        terminal_cumulative = int(result.initial.evaluation_requests)
        for rec in result.records:
            terminal_cumulative += int(rec.evaluation_requests)
            if rec.index in seen_steps:
                continue
            idx = int(rec.index)
            landing = None if rec.landing is None else rec.landing.atoms
            landing_ref = record_frame("landing", idx, landing)
            minimum_ref, gate = None, None
            if seen_minima < len(result.minima):
                minimum = result.minima[seen_minima]
                seen_minima += 1
                minimum_ref = record_frame("minimum", idx, minimum.atoms)
                minimum_components = PANEL.connected_components(minimum.atoms)
                gate = BASE.common_qualification(minimum.atoms, minimum.energy,
                    minimum.max_force, minimum.converged, reference)
                if gate["qualified_candidate"] and candidate is None:
                    candidate = {"event_index": idx, "energy_eV": float(minimum.energy),
                        "fmax_eV_A": float(minimum.max_force),
                        "evaluation_requests": terminal_cumulative, "geometry": gate,
                        "atoms": minimum.atoms.copy()}
            record_event({"kind": "outer_step", "index": idx, "status": rec.status,
                "accepted": bool(rec.accepted), "step_requests": int(rec.evaluation_requests),
                "cumulative_requests": terminal_cumulative, "landing_structure": landing_ref,
                "minimum_structure": minimum_ref, "minimum_gate": gate,
                "minimum_connected_component_sizes_1p3sigma": (
                    minimum_components if minimum_ref is not None else None),
                "progress_callback_observed": False})
            seen_steps.add(idx)
    if result is not None and result.checkpoint is not None:
        save_ssw_checkpoint(output / "last-checkpoint.pkl", result.checkpoint)
        write(output / "best.extxyz", result.best)
    elif result is not None:
        write(output / "best.extxyz", result.best)

    fresh_rows = []
    fresh = BASE.CompactSurface(FullPairLJ(epsilon=EPSILON, sigma=SIGMA),
        output / "fresh-ef-ledger.jsonl", cap=FRESH_CAP, failures_dir=output / "failed-geometries")
    if candidate is not None:
        try:
            energy, forces = fresh.evaluate(candidate["atoms"])
            fmax = float(np.linalg.norm(forces, axis=1).max())
            qualified = BASE.common_qualification(candidate["atoms"], energy, fmax,
                fmax <= FMAX, reference)
            fresh_rows.append({"role": "target_candidate", "energy_eV": float(energy),
                "fmax_eV_A": fmax, "passed": bool(qualified["qualified_candidate"]),
                "force_gate_0p01_informational": bool(fmax <= 0.01), "qualification": qualified})
        except Exception as exc:
            fresh_rows.append({"role": "target_candidate", "passed": False, "error": repr(exc)})
    elif result is not None:
        best = result.checkpoint.best if result.checkpoint else result.initial
        for role, quench in (("initial", result.initial), ("best", best)):
            try:
                energy, forces = fresh.evaluate(quench.atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                fresh_rows.append({"role": role, "energy_eV": float(energy), "fmax_eV_A": fmax,
                    "force_gate_0p05": bool(fmax <= FMAX),
                    "force_gate_0p01_informational": bool(fmax <= 0.01), "requests": fresh.requests})
            except Exception as exc:
                fresh_rows.append({"role": role, "error": repr(exc), "requests": fresh.requests})
    candidate_fresh = next((row for row in fresh_rows if row.get("role") == "target_candidate"), None)
    pause_confirmed = bool(candidate is not None and result is not None and result.status == "paused")
    first_hit = bool(pause_confirmed and candidate_fresh and candidate_fresh.get("passed"))
    status = ("first_hit" if first_hit else "fresh_failed" if pause_confirmed else
              "candidate_unconfirmed" if candidate is not None else
              result.status if result is not None else "failed")
    initial_requests = None if result is None else int(result.initial.evaluation_requests)
    outer_requests = sum(int(event.get("step_requests", 0)) for event in events
                         if event.get("kind") == "outer_step")
    BASE.dump(output / "result.json", {
        "status": status, "slot": slot, "seed": seed, "arm": arm, "error": error,
        "run_mode": "one_outer_step_smoke" if smoke else "full_search",
        "effective_run_limits": {"search_surface_requests_including_initial_quench": run_cap,
                                  "completed_outer_steps": run_steps, "wall_seconds": run_wall},
        "input_sha256": sha256(source_input), "search_requests_total": surface.requests,
        "initial_quench_requests": initial_requests, "outer_requests": outer_requests,
        "request_closure": None if initial_requests is None or result is None else
            surface.requests == initial_requests + outer_requests,
        "denied_requests": surface.denied, "actual_calculator_calls": surface.actual_calls,
        "search_wall_seconds": time.monotonic() - run_started,
        "outer_callbacks": sum(event.get("kind") == "outer_step" for event in events),
        "first_candidate": None if candidate is None else
            {k: value for k, value in candidate.items() if k != "atoms"},
        "fresh": fresh_rows, "fresh_requests": fresh.requests,
        "candidate_pause_confirmed": pause_confirmed,
        "best_energy_eV": None if result is None else float(
            result.checkpoint.best.energy if result.checkpoint else result.initial.energy),
        "result_status": None if result is None else result.status,
        "protocol": config, "settings": config["ssw"],
        "source_head": imports["git_head"],
        "reference_qualification_path": str(reference_qualification.resolve()),
    })
    if error is not None:
        BASE.dump(output / "failure.json", error)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--qualify-reference", action="store_true")
    group.add_argument("--execute", action="store_true")
    group.add_argument("--smoke", action="store_true")
    parser.add_argument("--slot", type=int, choices=range(4))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-qualification", type=Path)
    args = parser.parse_args()
    if args.preflight:
        if args.slot is not None or args.output is not None or args.reference_qualification is not None:
            parser.error("--preflight accepts no run arguments")
        preflight()
    elif args.qualify_reference:
        if args.slot is not None or args.output is None or args.reference_qualification is not None:
            parser.error("--qualify-reference requires --output only")
        qualify_reference(args.output.resolve())
    else:
        if args.slot is None or args.output is None or args.reference_qualification is None:
            parser.error("--execute requires --slot, --output and --reference-qualification")
        if args.smoke and args.slot not in (0, 1):
            parser.error("--smoke accepts only slots 0 or 1 (the seed 25092501 pair)")
        execute(args.slot, args.output.resolve(), args.reference_qualification.resolve(), smoke=args.smoke)


if __name__ == "__main__":
    main()
