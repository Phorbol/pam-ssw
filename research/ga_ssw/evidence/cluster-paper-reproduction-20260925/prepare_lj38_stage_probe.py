#!/usr/bin/env python3
"""Prepare a bounded, opt-in LJ38 early-stage capture replay; PES requires --execute."""
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
sys.path.insert(0, str(ROOT))
SOURCE_RUNNER = HERE / "run_lj_pilot.py"
PLAN = HERE / "stage-probe-plan.md"
ORACLE = ROOT / "research/ga_ssw/full_pair_lj.py"
SEEDS = (25092501, 25092502)
ARMS = ("global", "paper")
OUTER_ATTEMPTS = 3
REQUEST_CAP_PER_ARM = 6000
WALL_CAP_TOTAL_SECONDS = 660
DEFAULT_OUTPUT = HERE / "stage-probe-runs"


def load_source_runner():
    spec = importlib.util.spec_from_file_location("lj38_stage_probe_source_runner", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load existing LJ pilot runner: {SOURCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def arm_settings(source, arm):
    config, rotation = source.make_settings()
    expected = "global" if arm == "global" else "paper"
    if config.direction_sampling != expected:
        config = replace(config, direction_sampling=expected)
    return config, rotation


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(output):
    """Check existing provenance, settings and deterministic inputs; no PES calls."""
    from research.ga_ssw.full_pair_lj import FullPairLJ

    if output.exists():
        raise FileExistsError(output)
    for path in (SOURCE_RUNNER, PLAN, ORACLE):
        if not path.is_file():
            raise FileNotFoundError(path)
    source = load_source_runner()
    # Calculator construction checks the import/API only; no atoms are evaluated.
    calculator = FullPairLJ(epsilon=source.EPSILON_EV, sigma=source.SIGMA_A)
    del calculator
    inputs = []
    for arm in ARMS:
        config, rotation = arm_settings(source, arm)
        old_arm = "runs" if arm == "global" else "paper-direction-runs"
        previous = HERE / old_arm
        old_spec = importlib.util.spec_from_file_location(
            f"lj38_stage_probe_frozen_{arm}", previous / "run_lj_pilot.py")
        if old_spec is None or old_spec.loader is None:
            raise RuntimeError(f"cannot load archived {arm} generator: {previous / 'run_lj_pilot.py'}")
        frozen = importlib.util.module_from_spec(old_spec)
        sys.modules[old_spec.name] = frozen
        old_spec.loader.exec_module(frozen)
        for seed in SEEDS:
            old_case = previous / f"lj38-seed{seed}"
            old_input = old_case / "initial.extxyz"
            old_steps = old_case / "outer-steps.jsonl"
            old_summary_path = old_case / "summary.json"
            if not old_input.is_file():
                raise FileNotFoundError(old_input)
            if not old_steps.is_file():
                raise FileNotFoundError(old_steps)
            old_summary = json.loads(old_summary_path.read_text())
            if (old_summary["settings"]["ssw_config"] != asdict(config) or
                    old_summary["settings"]["recovered_rotation"] != asdict(rotation)):
                raise ValueError(f"effective settings differ from archived {arm} run: {old_summary_path}")
            atoms, search_seed = source.uniform_volume_cluster(38, seed)
            old_atoms, old_search_seed = frozen.uniform_volume_cluster(38, seed)
            if not np.array_equal(atoms.positions, old_atoms.positions):
                raise ValueError(f"current and archived generators differ in positions for {arm} seed {seed}")
            if not np.array_equal(search_seed.generate_state(4), old_search_seed.generate_state(4)):
                raise ValueError(f"current and archived generators differ in child RNG for {arm} seed {seed}")
            inputs.append({"arm": arm, "seed": seed, "input": str(old_input),
                           "atoms": len(atoms), "positions_exact_between_generators": True,
                           "search_child_rng_exact": True,
                           "direction_sampling": config.direction_sampling})
    ledger = source.load_ledger()
    return {
        "status": "preflight_passed_zero_PES",
        "output_available": str(output),
        "source_runner": str(SOURCE_RUNNER),
        "ledger_serializer": str(ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"),
        "full_pair_lj": str(ORACLE),
        "source_runner_sha256": sha256(SOURCE_RUNNER),
        "ledger_serializer_loadable": ledger is not None,
        "arms": list(ARMS), "seeds": list(SEEDS),
        "outer_attempts_per_arm": OUTER_ATTEMPTS,
        "requests_per_arm_max": REQUEST_CAP_PER_ARM,
        "campaign_requests_max": REQUEST_CAP_PER_ARM * len(ARMS) * len(SEEDS),
        "campaign_wall_cap_seconds": WALL_CAP_TOTAL_SECONDS,
        "seeded_inputs": inputs,
        "PES_requests": 0,
    }


def compact_trace(event):
    rotation = event.get("recovered_rotation") or {}
    trace = rotation.get("trace") or ()
    keep = ("event", "stage", "rotnum", "force_calls", "curvature", "real_curvature",
            "residual_norm", "retries", "force_converged", "rotation_limit")
    return [{key: entry[key] for key in keep if key in entry} for entry in trace]


def record_row(record):
    stages = []
    for event in record.climb:
        rotation = event.get("recovered_rotation") or {}
        stages.append({
            "gaussian_index": event.get("index"),
            "status": event.get("status"),
            "center_A": event.get("center"),
            "mode_direction": event.get("direction"),
            "rotation_trace": compact_trace(event),
            "rotation_force_cost": event.get("rotation_force_requests"),
            "rotation_trace_cost": (None if not rotation.get("trace") else
                                    rotation["trace"][-1].get("force_calls")),
            "stage_requests": event.get("requests"),
            "width_A": event.get("width"),
            "height_eV": event.get("weight"),
            "quench_requests": event.get("quench_requests"),
            "true_energy_eV": event.get("true_energy"),
            "rotation_residual": event.get("rotation_residual"),
            "rotation_converged": event.get("rotation_converged"),
            "rotation_stop_reason": event.get("rotation_stop_reason"),
        })
    return {
        "outer_index": int(record.index), "status": str(record.status),
        "accepted": bool(record.accepted), "outer_requests": int(record.evaluation_requests),
        "stages": stages,
    }


def execute(output, validation):
    from ase.io import write
    from pamssw.standalone.paper_reference import BiasStageQuenchOutcome, quench, run_ssw
    from research.ga_ssw.full_pair_lj import FullPairLJ

    source = load_source_runner()
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), output / Path(__file__).name)
    shutil.copy2(SOURCE_RUNNER, output / "source_run_lj_pilot.py")
    shutil.copy2(PLAN, output / PLAN.name)
    shutil.copy2(ORACLE, output / "full_pair_lj.py")
    ledger = source.load_ledger()
    git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    provenance = {
        "git_head": git("rev-parse", "HEAD"),
        "source_runner": str(SOURCE_RUNNER), "source_runner_sha256": sha256(SOURCE_RUNNER),
        "probe_runner_sha256": sha256(Path(__file__).resolve()),
        "plan_sha256": sha256(PLAN), "oracle_sha256": sha256(ORACLE),
        "arms": list(ARMS), "seeds": list(SEEDS), "outer_attempts_per_arm": OUTER_ATTEMPTS,
        "request_cap_per_arm": REQUEST_CAP_PER_ARM,
        "wall_cap_total_seconds": WALL_CAP_TOTAL_SECONDS,
        "observer": None, "checkpointing": False,
        "adapter": "same quench invocation; capture-only BiasStageQuenchOutcome",
        "zero_pes_preflight": validation,
    }
    ledger.dump(output / "execution.json", provenance)
    started = time.monotonic()
    total_deadline = started + WALL_CAP_TOTAL_SECONDS
    summaries = []
    for arm in ARMS:
        for seed in SEEDS:
            case_name = f"lj38-{arm}-seed{seed}"
            folder = output / case_name
            if time.monotonic() >= total_deadline:
                row = {"arm": arm, "seed": seed, "status": "not_run_campaign_wall_cap",
                       "search_requests": 0, "outer_attempts": 0}
                ledger.dump(folder.with_suffix(".json"), row)
                summaries.append(row)
                continue
            folder.mkdir()
            config, rotation = arm_settings(source, arm)
            atoms, search_seed = source.uniform_volume_cluster(38, seed)
            input_path = folder / "initial.extxyz"
            write(input_path, atoms)
            surface = source.BoundedSurface(
                FullPairLJ(epsilon=source.EPSILON_EV, sigma=source.SIGMA_A),
                arm_cap=REQUEST_CAP_PER_ARM,
                arm_deadline=total_deadline,
                total_deadline=total_deadline,
            )
            rng = np.random.default_rng(search_seed)
            biased_dir = folder / "biased-endpoints"
            biased_dir.mkdir()
            biased_endpoints = []

            def capture_biased_quench(displaced, surface, *, fmax, steps, terms,
                                      optimizer, frame, lbfgs_memory, context):
                # This is exactly the driver's ordinary biased-quench call.
                relaxed = quench(displaced, surface, fmax=fmax, steps=steps, terms=terms,
                                 optimizer=optimizer, frame=frame, lbfgs_memory=lbfgs_memory)
                outer = int(context["outer_index"])
                gaussian = int(context["gaussian_index"])
                endpoint = biased_dir / f"outer-{outer:02d}-gaussian-{gaussian:02d}.extxyz"
                write(endpoint, relaxed.atoms)
                biased_endpoints.append({"outer_index": outer, "gaussian_index": gaussian,
                                         "path": str(endpoint.relative_to(folder))})
                return BiasStageQuenchOutcome(relaxed=relaxed, stage_stopped=False,
                                              release_all=False, diagnostics={})

            row = {"arm": arm, "seed": seed, "status": "started",
                   "direction_sampling": config.direction_sampling,
                   "settings": {"ssw_config": asdict(config),
                                "recovered_rotation": asdict(rotation)},
                   "outer_attempts_requested": OUTER_ATTEMPTS,
                   "request_cap": REQUEST_CAP_PER_ARM,
                   "initialization": "same uniform_volume_cluster and RNG child streams as archived pilot",
                   "initial_input": str(input_path)}
            try:
                result = run_ssw(atoms.copy(), surface, steps=OUTER_ATTEMPTS, config=config,
                                 rng=rng, recovered_rotation=rotation,
                                 bias_quench_adapter=capture_biased_quench)
                current = result.initial.atoms.copy()
                current_energy = float(result.initial.energy)
                best_energy = current_energy
                write(folder / "initial-minimum.extxyz", current)
                outer_rows = []
                for record in result.records:
                    index = int(record.index)
                    write(folder / f"outer-{index:02d}-start.extxyz", current)
                    if record.landing is not None:
                        write(folder / f"outer-{index:02d}-landing.extxyz", record.landing.atoms)
                    outer_row = record_row(record)
                    outer_row["current_energy_before_eV"] = current_energy
                    outer_row["best_energy_before_eV"] = best_energy
                    outer_row["landing_energy_eV"] = (None if record.landing is None else
                                                       float(record.landing.energy))
                    if record.accepted and record.landing is not None:
                        current = record.landing.atoms.copy()
                        current_energy = float(record.landing.energy)
                    if record.landing is not None and record.landing.converged:
                        best_energy = min(best_energy, float(record.landing.energy))
                    outer_row["current_energy_after_eV"] = current_energy
                    outer_row["best_energy_after_eV"] = best_energy
                    outer_rows.append(outer_row)
                row.update(status=result.status, outer_attempts=len(result.records),
                           search_requests=int(surface.requests), denials=int(surface.denials),
                           boundary=surface.boundary, initial_requests=int(result.initial.evaluation_requests),
                           final_best_energy_eV=float(result.best.energy),
                           outer_steps=outer_rows, biased_endpoints=biased_endpoints)
                if result.evaluation_requests != surface.requests:
                    row["request_accounting_warning"] = {
                        "result_requests": int(result.evaluation_requests),
                        "surface_requests": int(surface.requests),
                    }
            except Exception as error:
                row.update(status="exception", error=repr(error), traceback=traceback.format_exc(),
                           search_requests=int(surface.requests), denials=int(surface.denials),
                           boundary=surface.boundary, biased_endpoints=biased_endpoints)
            row["wall_seconds"] = time.monotonic() - started
            ledger.dump(folder / "summary.json", row)
            summaries.append(row)
            ledger.dump(output / "summary.json", {
                "status": "running", "trajectories": summaries,
                "search_requests_total": sum(int(item.get("search_requests", 0)) for item in summaries),
                "wall_seconds_total": time.monotonic() - started,
            })
    ledger.dump(output / "summary.json", {
        "status": "complete_or_censored", "trajectories": summaries,
        "search_requests_total": sum(int(item.get("search_requests", 0)) for item in summaries),
        "wall_seconds_total": time.monotonic() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="zero-PES path/settings/input checks")
    modes.add_argument("--execute", action="store_true", help="run four bounded 3-attempt replays")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.preflight:
        print(json.dumps(preflight(output), indent=2))
    elif args.execute:
        validation = preflight(output)
        print(json.dumps(validation, indent=2), flush=True)
        execute(output, validation)
    else:
        print(f"Prepared; use --preflight for zero-PES checks or --execute for PES work. Output: {output}")


if __name__ == "__main__":
    main()
