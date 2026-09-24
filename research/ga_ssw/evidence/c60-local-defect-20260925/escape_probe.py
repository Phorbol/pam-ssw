#!/usr/bin/env python3
"""Prepared paired C60 SSW / native-LS trajectories; execution is opt-in."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN = HERE / "escape-plan.json"
PREFLIGHT = HERE / "preflight.json"
LEDGER_PATH = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
SETTINGS_RUNNER = ROOT / "research/ga_ssw/evidence/bias-tolerance-paired-20260925/run.py"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def cutoff_graph(atoms, cutoff, nx, np):
    delta = atoms.positions[:, None, :] - atoms.positions[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from((int(i), int(j)) for i, j in zip(*np.where(
        np.triu((distances < cutoff) & (distances > 0), 1))))
    return graph


def runtime_support(plan):
    """Load only settings conversion and the existing experiment ledger."""
    effective = plan["source_effective_config"]["snapshot"]
    settings_module = load_module("c60_escape_settings_source", SETTINGS_RUNNER)
    config, native_ls, rotation, mc = settings_module.make_settings(
        effective, effective["ssw_config"]["bias_fmax"])
    ledger = load_module("c60_escape_ledger", LEDGER_PATH)
    return settings_module, ledger, config, native_ls, rotation, mc


def preflight(*, require_preflight=False):
    """CPU-only input/configuration gate; this function never creates a calculator."""
    import numpy as np
    import networkx as nx
    from ase.io import read

    plan = json.loads(PLAN.read_text())
    if plan["status"] != "PREPARED_NOT_SUBMITTED":
        raise RuntimeError("plan is not in PREPARED_NOT_SUBMITTED state")
    if git("rev-parse", "HEAD:pamssw") != plan["checkout"]["core_tree"]:
        raise RuntimeError("pamssw core tree changed after protocol preparation")
    if git("status", "--porcelain", "--", "pamssw"):
        raise RuntimeError("pamssw core has uncommitted changes")

    required_artifacts = ("defect_input", "defect_qualification", "ih_graph_reference",
                          "ih_energy_reference", "curvature_results")
    for name in required_artifacts:
        artifact = plan["source_artifacts"][name]
        path = Path(artifact["path"])
        if not path.is_file() or sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"input/provenance file missing or changed: {name}")
    model_path = Path(plan["model"]["path"])
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"model is missing or empty: {model_path}")
    runs = Path(plan["output"]["runs_dir"])
    if runs.exists():
        raise FileExistsError("runs/ already exists; no overwrite, resume, or retry")

    defect_result = json.loads(Path(plan["source_artifacts"]["defect_qualification"]["path"]).read_text())
    if not defect_result.get("force_qualified"):
        raise RuntimeError("source isomer-2 was not force qualified")
    if not all(defect_result.get("labeled_edges_preserved", {}).values()):
        raise RuntimeError("source isomer-2 did not preserve its labeled cage edges")
    if not all(row.get("graph_cage_candidate") for row in defect_result.get("final_graphs", [])):
        raise RuntimeError("source isomer-2 did not retain its fullerene cage")

    defect = read(plan["input"]["path"])
    ih = read(plan["references"]["ih_graph_input"])
    if (len(defect) != 60 or not np.all(defect.numbers == 6)
            or not np.isfinite(defect.positions).all() or defect.pbc.any()
            or defect.constraints):
        raise RuntimeError("source input must be finite, unconstrained, nonperiodic C60")
    if (len(ih) != 60 or not np.all(ih.numbers == 6)
            or not np.isfinite(ih.positions).all() or ih.pbc.any()):
        raise RuntimeError("Ih graph reference must be finite nonperiodic C60")
    graph_module = load_module("c60_escape_graph_helper",
                               ROOT / "research/ga_ssw/analyze_c60_random_development.py")
    ref_graphs = {cutoff: cutoff_graph(ih, cutoff, nx, np)
                  for cutoff in plan["references"]["graph_cutoffs_A"]}
    source_rows = {}
    for cutoff in plan["references"]["graph_cutoffs_A"]:
        row = graph_module.graph_row(defect.numbers, defect.positions, cutoff, ref_graphs[cutoff])
        if not row["graph_cage_candidate"] or row["ih_graph_match"]:
            raise RuntimeError(f"source isomer-2 graph failed frozen identity check at {cutoff} A")
        source_rows[str(cutoff)] = row

    curvature_path = Path(plan["source_artifacts"]["curvature_results"]["path"])
    curvature = json.loads(curvature_path.read_text())
    if len(curvature.get("cases", [])) != 2:
        raise RuntimeError("curvature gate needs both qualified source endpoints")
    curvature_rows = []
    for case in curvature["cases"]:
        values = np.asarray(case.get("eigenvalues_eV_A2", []), float)
        direct = [row.get("direct_curvature_eV_A2")
                  for row in case.get("lowest_modes_sensitivity", [])]
        if (values.shape != (174,) or not np.isfinite(values).all() or np.any(values <= 0)
                or len(direct) != 2 or not np.isfinite(direct).all()
                or any(value <= 0 for value in direct)):
            raise RuntimeError("curvature gate failed: all 174 modes and direct checks must be positive")
        curvature_rows.append({"source": case["source"], "minimum_internal_eigenvalue_eV_A2": float(values.min()),
                               "lowest_direct_curvatures_eV_A2": direct})

    _, ledger, config, ls, rotation, mc = runtime_support(plan)
    snapshot = plan["source_effective_config"]["snapshot"]
    if (config.max_gaussians != 12 or config.bias_fmax != 0.1 or config.fmax != 0.03
            or config.quench_optimizer != "safe-lbfgs-total" or config.lbfgs_memory != 500
            or config.cluster_frame != "direction_only" or ls is None or mc is None):
        raise RuntimeError("constructed run settings differ from the frozen C60 protocol")
    runner_hash = sha256(__file__)
    plan_hash = sha256(PLAN)
    if require_preflight:
        if not PREFLIGHT.is_file():
            raise RuntimeError("run requires a successful --preflight record")
        prior = json.loads(PREFLIGHT.read_text())
        if (prior.get("status") != "preflight_passed"
                or prior.get("runner_sha256") != runner_hash
                or prior.get("plan_sha256") != plan_hash
                or prior.get("core_tree") != git("rev-parse", "HEAD:pamssw")):
            raise RuntimeError("saved preflight is stale or did not pass")

    return {
        "status": "preflight_passed", "head": git("rev-parse", "HEAD"),
        "core_tree": git("rev-parse", "HEAD:pamssw"),
        "runner_sha256": runner_hash, "plan_sha256": plan_hash,
        "settings": {"ssw_config": config, "native_ls": ls,
                     "recovered_rotation": rotation, "native_mc": mc},
        "source_graph_checks": source_rows, "curvature_gate": curvature_rows,
        "resource_contract": {
            "arms": 4, "search_request_ceiling": 48000,
            "fresh_request_ceiling": 44,
            "cap_enforcement": "declared for runtime; not exercised by zero-PES preflight",
            "calculator_or_pes_requests": 0,
        },
    }, ledger


def endpoint_graphs(atoms, ih_graphs, defect_graphs, graph_module, cutoffs, nx, np):
    rows = {}
    for cutoff in cutoffs:
        ih_row = graph_module.graph_row(atoms.numbers, atoms.positions, cutoff, ih_graphs[cutoff])
        defect_row = graph_module.graph_row(atoms.numbers, atoms.positions, cutoff, defect_graphs[cutoff])
        current_graph = cutoff_graph(atoms, cutoff, nx, np)
        ih_row["source_defect_graph_match"] = defect_row["ih_graph_match"]
        ih_row["source_defect_labeled_edges_preserved"] = (
            set(tuple(sorted(edge)) for edge in current_graph.edges())
            == set(tuple(sorted(edge)) for edge in defect_graphs[cutoff].edges()))
        rows[str(cutoff)] = ih_row
    return rows


def execute():
    started = time.monotonic()
    checked, ledger = preflight(require_preflight=True)
    plan = json.loads(PLAN.read_text())
    deadline = started + plan['resources']['wall_seconds_total']
    runs = Path(plan['output']['runs_dir'])
    runs.mkdir(exist_ok=False)
    ledger.dump(runs / 'preflight-used.json', checked)
    (runs / 'runner.py').write_bytes(Path(__file__).read_bytes())
    (runs / 'plan.json').write_bytes(PLAN.read_bytes())
    import traceback
    import numpy as np
    import networkx as nx
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator
    from pamssw.standalone import run_ssw

    torch.set_num_threads(1)
    torch.manual_seed(plan['runtime']['torch_manual_seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = plan['model']
    kwargs = dict(model_paths=model['path'], head=model['head'], device=model['device'],
                  default_dtype=model['dtype'], enable_cueq=False, enable_oeq=False)
    calculator, fresh_calculator = MACECalculator(**kwargs), MACECalculator(**kwargs)
    counter = ledger.instrument_calculate(calculator)
    fresh_counter = ledger.instrument_calculate(fresh_calculator)
    initial = read(plan['input']['path'])
    initial.calc = None
    ih = read(plan['references']['ih_graph_input'])
    cutoffs = plan['references']['graph_cutoffs_A']
    ih_graphs = {c: cutoff_graph(ih, c, nx, np) for c in cutoffs}
    defect_graphs = {c: cutoff_graph(initial, c, nx, np) for c in cutoffs}
    graph_module = load_module('escape_graphs', ROOT / 'research/ga_ssw/analyze_c60_random_development.py')
    rows = []
    for seed in plan['arms']['seeds']:
        for method in plan['methods']:
            row = dict(seed=seed, method=method, status='not_started')
            if time.monotonic() >= deadline:
                row['status'] = 'not_run_deadline'
                rows.append(row)
                ledger.dump(runs / 'summary.json', rows)
                continue
            folder = runs / f'{method}-{seed}'
            folder.mkdir()
            _, _, config, ls, rotation, mc = runtime_support(plan)
            selected_ls = ls if method == 'native_ls' else None
            ledger.dump(folder / 'effective-config.json', dict(config=config, ls=selected_ls,
                        rotation=rotation, mc=mc, seed=seed))
            before, fresh_before = counter['calls'], fresh_counter['calls']
            calculator.reset()
            surface = ledger.CountedSurface(calculator, folder / 'requests.jsonl',
                plan['arms']['search_requests_per_arm'], max(0, deadline-time.monotonic()))
            fresh = ledger.CountedSurface(fresh_calculator, folder / 'fresh.jsonl',
                plan['arms']['fresh_requests_per_arm'], max(0, deadline-time.monotonic()))
            result = None
            arm_start = time.monotonic()
            try:
                result = run_ssw(initial.copy(), surface, steps=plan['arms']['outer_attempts'],
                    config=config, rng=np.random.default_rng(seed), ls=selected_ls,
                    recovered_rotation=rotation, mc=mc, checkpoint_path=folder / 'checkpoint.pkl')
                ledger.dump(folder / 'result.json', result)
                row.update(status=result.status, result_requests=result.evaluation_requests,
                           requests_match_result=result.evaluation_requests == surface.requests)
            except Exception as error:
                row.update(status='exception', error=repr(error), traceback=traceback.format_exc())
                # Preserve available completed records for diagnostics only;
                # this does not resume the run or turn an exception into success.
                if result is None and (folder / 'checkpoint.pkl').exists():
                    try:
                        from pamssw.standalone import load_ssw_checkpoint
                        result = load_ssw_checkpoint(folder / 'checkpoint.pkl')
                        ledger.dump(folder / 'checkpoint-prefix.json', result)
                        row['diagnostic_source'] = 'saved_checkpoint_prefix'
                    except Exception as checkpoint_error:
                        row['checkpoint_read_error'] = repr(checkpoint_error)
            checks, costs = [], []
            if result is not None:
                cumulative = result.initial.evaluation_requests
                endpoints = [('initial', result.initial, cumulative)]
                for record in result.records:
                    cumulative += record.evaluation_requests
                    costs.append(dict(index=record.index, status=record.status, accepted=record.accepted,
                        requests=record.evaluation_requests, cumulative_requests=cumulative,
                        gaussian_count=len(record.climb), ls_update=record.ls_update,
                        ls_preparation=record.ls_preparation))
                    if record.landing is not None:
                        endpoints.append((f'landing-{record.index}', record.landing, cumulative))
                row['record_cost_sum_matches_requests'] = cumulative == surface.requests
                row['requests_outside_saved_records'] = surface.requests - cumulative
                for role, endpoint, cost in endpoints:
                    atoms = endpoint.atoms.copy()
                    atoms.calc = None
                    write(folder / f'{role}.traj', atoms)
                    check = dict(role=role, search_cost=cost, reported_converged=endpoint.converged,
                                 status='not_checked_deadline')
                    if time.monotonic() < deadline:
                        try:
                            fresh_calculator.reset()
                            energy, forces = fresh.evaluate(atoms)
                            fmax = float(np.linalg.norm(forces, axis=1).max())
                            check.update(status='checked', energy_eV=energy, fmax_eV_A=fmax,
                                force_qualified=fmax <= config.fmax,
                                composition_preserved=bool(np.array_equal(atoms.numbers, initial.numbers)),
                                cell_preserved=bool(np.array_equal(atoms.cell.array, initial.cell.array)),
                                pbc_preserved=bool(np.array_equal(atoms.pbc, initial.pbc)),
                                delta_ih_eV=energy-plan['references']['ih_energy_eV'],
                                energy_window_met=energy <= plan['references']['ih_energy_eV']+plan['references']['energy_window_eV'],
                                graphs=endpoint_graphs(atoms, ih_graphs, defect_graphs, graph_module, cutoffs, nx, np))
                        except Exception as error:
                            check.update(status='fresh_failed', error=repr(error))
                    checks.append(check)
                    ledger.dump(folder / 'checks.json', checks)
            ledger.dump(folder / 'record-costs.json', costs)
            row.update(search_requests=surface.requests, fresh_requests=fresh.requests,
                search_calculations=counter['calls']-before,
                fresh_calculations=fresh_counter['calls']-fresh_before,
                search_boundary=surface.boundary, search_denials=surface.denials,
                fresh_boundary=fresh.boundary, fresh_denials=fresh.denials,
                elapsed_seconds=time.monotonic()-arm_start, checks=checks,
                checkpoint_exists=(folder / 'checkpoint.pkl').exists())
            ledger.dump(folder / 'summary.json', row)
            rows.append(row)
            ledger.dump(runs / 'summary.json', rows)
            print(method, seed, row['status'], surface.requests, 'search requests', flush=True)
    ledger.dump(runs / 'execution.json', dict(elapsed_seconds=time.monotonic()-started,
        search_requests=sum(r.get('search_requests', 0) for r in rows),
        fresh_requests=sum(r.get('fresh_requests', 0) for r in rows),
        search_calculations=counter['calls'], fresh_calculations=fresh_counter['calls'],
        planned_arms=4, reported_arms=len(rows), head=git('rev-parse', 'HEAD')))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        row, ledger = preflight()
        ledger.dump(PREFLIGHT, row)
        print("preflight_passed; no calculator/PES requests; no runs/ directory created")
        return
    execute()


if __name__ == "__main__":
    main()
