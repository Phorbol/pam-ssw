#!/usr/bin/env python3
"""Bounded MH-1 SSW versus paper-form LS transfer on two C4H6 isomers.

This is a development comparison from paper-listed D/E isomer inputs, not a
reproduction of the paper's trans-butadiene-start GGA-PBE experiment. It writes
one immutable, fresh output directory and never retries a failed arm.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
QUAL = ROOT / "research/ga_ssw/evidence/c4h6-model-qualification-20260924-r2"
LIFECYCLE = ROOT / "research/ga_ssw/evidence/c4h6-mh1-lifecycle-20260924"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
EXPECTED_CORE_TREE = "c572a1cc0766f8d3ff534a467ad64613aaa78ce0"
INPUT_SHA256 = {
    "cyclobutene": "87bb85eb4ec12b4aae64a305c21c9a693bed98c2e8c996dc18055672c3227a1b",
    "bicyclobutane": "e9700f800d495b53674064aea953116c32d32f4ecf701582c222a42a87db7571",
}
CASES = ("cyclobutene", "bicyclobutane")
ARMS = ("ssw", "paper_ls")
SEED = 59
STEPS = 12
SEARCH_CAP = 8000
WALL_PER_ARM = 240
CAMPAIGN_WALL = 1080
FRESH_CAP = 13


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def load_ledger(path: Path):
    spec = importlib.util.spec_from_file_location("c4h6_transfer_ledger", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import shared ledger: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight(output: Path | None = None, *, require_cuda=False) -> dict:
    """Check provenance and imports only; this function makes no PES calls."""
    import numpy as np
    import ase
    import torch
    import mace
    from ase.io import read
    from mace.calculators import MACECalculator  # noqa: F401

    if not MODEL.is_file() or sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError(f"MH-1 model missing or SHA256 mismatch: {MODEL}")
    if require_cuda and (not torch.cuda.is_available() or torch.cuda.device_count() < 1):
        raise RuntimeError("CUDA is required by the frozen MH-1 protocol")

    qplan = json.loads((QUAL / "plan.json").read_text())
    qualification = json.loads((QUAL / "qualification.json").read_text())
    if (qualification.get("status") != "complete" or not qualification.get("all_qualified")
            or qplan.get("model_sha256") != MODEL_SHA256 or qplan.get("head") != "omol"
            or qplan.get("device") != "cuda" or qplan.get("dtype") != "float64"):
        raise RuntimeError("frozen MH-1 qualification/model provenance is not valid")
    qualified_rows = {row.get("case"): row for row in qualification.get("rows", [])}
    if any(not qualified_rows.get(case, {}).get("qualified") for case in CASES):
        raise RuntimeError("both selected isomers must have passed frozen input qualification")

    inputs = {}
    for case in CASES:
        path = QUAL / case / "input.extxyz"
        if not path.is_file() or sha256(path) != INPUT_SHA256[case]:
            raise RuntimeError(f"qualified input missing or SHA256 mismatch: {path}")
        atoms = read(path)
        if len(atoms) != 10 or atoms.get_chemical_formula() != "C4H6" or atoms.pbc.any():
            raise RuntimeError(f"unexpected input composition/boundary: {path}")
        inputs[case] = {"path": str(path), "sha256": sha256(path),
                        "numbers": atoms.numbers.tolist(), "cell_A": atoms.cell.array.tolist(),
                        "pbc": atoms.pbc.tolist()}

    core_tree = git("rev-parse", "HEAD:pamssw")
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "pamssw")
    if core_tree != EXPECTED_CORE_TREE or dirty:
        raise RuntimeError(f"current frozen core mismatch/dirty: tree={core_tree}, status={dirty!r}")
    ledger_path = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    ledger = load_ledger(ledger_path)

    sys.path.insert(0, str(ROOT))
    from pamssw.standalone import LSSettings, SSWConfig, run_ssw
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    from pamssw.standalone.surface import ASESurface
    from research.ga_ssw.analyze_c4h6_ls_reaction_coverage import graph, component_formulas

    prior = json.loads((LIFECYCLE / "effective_config.json").read_text())
    config = SSWConfig(**prior["config"])
    rotation = RecoveredRotationSettings(**prior["rotation_settings"])
    prequench = LSPrequenchSettings(fmax=0.1, steps=50, exit_policy="force_or_step_limit")
    ls = LSSettings(dict(HC_BOND_ENERGIES), {k: v + 0.1 for k, v in HC_BOND_LENGTHS.items()},
                    target_per_atom=0.7, initial_fraction=0.03, xi=0.2,
                    learning_rate=1.8, prequench=prequench)
    if (config.width, config.max_gaussians, config.temperature_K, config.fmax,
            config.relax_steps, config.fd_step) != (0.1, 25, 150.0, 0.03, 400, 0.001):
        raise RuntimeError("lifecycle SSW settings differ from the frozen C4H6 protocol")
    if rotation.max_force_calls != 40 or ls.target_per_atom != 0.7:
        raise RuntimeError("rotation/LS settings differ from the frozen C4H6 protocol")
    if output is not None:
        resolved = output.expanduser().resolve()
        if resolved == HERE or resolved.exists():
            raise FileExistsError(f"output must be a new directory distinct from runner: {resolved}")
        if not resolved.parent.is_dir():
            raise FileNotFoundError(f"output parent must already exist: {resolved.parent}")

    modules = {"run_ssw": run_ssw, "ASESurface": ASESurface, "LSSettings": LSSettings,
               "RecoveredRotationSettings": RecoveredRotationSettings,
               "LSPrequenchSettings": LSPrequenchSettings, "ledger": ledger,
               "c4h6_graph": graph, "component_formulas": component_formulas,
               "MACECalculator": MACECalculator}
    paths = {name: str(Path(inspect.getfile(obj)).resolve()) for name, obj in modules.items()}
    return {
        "status": "preflight_ok_zero_PES", "git_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "core_tree": core_tree,
        "tracked_core_clean": True, "model": str(MODEL), "model_sha256": sha256(MODEL),
        "model_head": "omol", "device": "cuda", "dtype": "float64",
        "runtime": {"python": sys.version, "numpy": np.__version__, "ase": ase.__version__,
                    "torch": torch.__version__, "mace": mace.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count()},
        "inputs": inputs, "qualification_path": str(QUAL / "qualification.json"),
        "qualification_sha256": sha256(QUAL / "qualification.json"),
        "lifecycle_config_path": str(LIFECYCLE / "effective_config.json"),
        "lifecycle_config_sha256": sha256(LIFECYCLE / "effective_config.json"),
        "ledger_path": str(ledger_path), "ledger_sha256": sha256(ledger_path),
        "source_import_paths": paths,
        "paper_parameters": {"T_K": 150, "NG": 25, "width_A": 0.1,
                             "target_eV_atom": 0.7, "initial_fraction": 0.03,
                             "xi_fraction": 0.2, "learning_rate": 1.8},
        "implementation_parameter_boundary":
            "HC energy/length tables come from current native_ls.py; paper LS uses its energies and +0.1 A pair cutoffs, as in the prior C4H6 lifecycle. The paper formula/target is reused, but these tables/cutoffs are not specified in the paper SI.",
        "config": {"ssw": asdict(config), "rotation": asdict(rotation),
                   "paper_ls": asdict(ls)},
        "budgets": {"cases": list(CASES), "arms": list(ARMS), "seed": SEED,
                    "outer_steps": STEPS, "search_cap_per_arm": SEARCH_CAP,
                    "wall_seconds_per_arm": WALL_PER_ARM,
                    "campaign_wall_seconds": CAMPAIGN_WALL,
                    "fresh_cap_per_arm": FRESH_CAP,
                    "search_total_cap": len(CASES) * len(ARMS) * SEARCH_CAP,
                    "fresh_total_cap": len(CASES) * len(ARMS) * FRESH_CAP},
        "physics_calls": 0,
    }


class ResetCountedSurface:
    """Reuse the shared counted ledger, cold-resetting ASE cache per E/F request."""
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evaluate(self, atoms):
        self.base.calculator.reset()
        return self.base.evaluate(atoms)


def classify(atoms, graph_fn, formula_fn, refs):
    import networkx as nx
    g = graph_fn(atoms)
    matcher = nx.algorithms.isomorphism.categorical_node_match("number", None)
    matches = [name for name, ref in refs.items() if nx.is_isomorphic(g, ref, node_match=matcher)]
    return {"component_count": nx.number_connected_components(g),
            "component_formulas": formula_fn(atoms, g), "reference_isomers": matches,
            "degree_sequence": sorted(dict(g.degree()).values())}


def stage_counts(step) -> list[dict]:
    rows = []
    for stage in step.climb:
        row = stage if isinstance(stage, dict) else vars(stage)
        rows.append({key: row.get(key) for key in
                     ("index", "stage", "status", "stop_reason", "rotation_stop_reason", "rotation_force_requests", "force_requests", "quench_requests", "max_force")
                     if key in row})
    return rows


def execute(output: Path, provenance: dict) -> int:
    import numpy as np
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator

    sys.path.insert(0, str(ROOT))
    from pamssw.standalone import LSSettings, SSWConfig, run_ssw
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    from pamssw.standalone import paper_reference
    from research.ga_ssw.analyze_c4h6_ls_reaction_coverage import graph, component_formulas

    ledger_path = Path(provenance["ledger_path"])
    ledger = load_ledger(ledger_path)
    output.mkdir(parents=False, exist_ok=False)
    shutil.copy2(__file__, output / "run.py")
    shutil.copy2(HERE / "protocol.md", output / "protocol.md")
    ledger.dump(output / "provenance.json", provenance)
    prior = json.loads((LIFECYCLE / "effective_config.json").read_text())
    config = SSWConfig(**prior["config"])
    rotation = RecoveredRotationSettings(**prior["rotation_settings"])
    prequench = LSPrequenchSettings(fmax=0.1, steps=50, exit_policy="force_or_step_limit")
    paper_ls = LSSettings(dict(HC_BOND_ENERGIES),
                          {k: v + 0.1 for k, v in HC_BOND_LENGTHS.items()},
                          target_per_atom=0.7, initial_fraction=0.03, xi=0.2,
                          learning_rate=1.8, prequench=prequench)
    ledger.dump(output / "effective_config.json", {
        "config": asdict(config), "rotation_settings": asdict(rotation),
        "ls_settings": {"ssw": None, "paper_ls": asdict(paper_ls)},
        "seed": SEED, "outer_attempts": STEPS, "search_cap_per_arm": SEARCH_CAP,
        "wall_seconds_per_arm": WALL_PER_ARM, "campaign_wall_seconds": CAMPAIGN_WALL,
        "fresh_cap_per_arm": FRESH_CAP, "mc": "existing default; no new settings",
        "height_policy": None, "direction_override": None})

    torch.set_num_threads(1)
    torch.manual_seed(0)
    campaign_started = time.monotonic()
    overall = {"status": "started", "runs": [], "search_requests_total": 0,
               "fresh_requests_total": 0, "campaign_started_unix": time.time()}
    ledger.dump(output / "summary.json", overall)
    model_started = time.monotonic()
    try:
        calc = MACECalculator(model_paths=str(MODEL), head="omol", device="cuda",
                              default_dtype="float64", enable_cueq=False, enable_oeq=False)
    except Exception as exc:
        overall.update(status="model_load_failed", error=repr(exc),
                       model_load_seconds=time.monotonic() - model_started,
                       elapsed_seconds=time.monotonic() - campaign_started)
        ledger.dump(output / "summary.json", overall)
        return 1
    model_load_seconds = time.monotonic() - model_started
    calc_counter = ledger.instrument_calculate(calc)
    refs = {case: graph(read(QUAL / case / "input.extxyz"))
            for case in ("butadiene", *CASES)}
    overall["model_load_seconds"] = model_load_seconds
    overall["model_calculator_loads"] = 1
    ledger.dump(output / "summary.json", overall)

    for case in CASES:
        atoms = read(QUAL / case / "input.extxyz")
        for arm in ARMS:
            folder = output / f"{case}-{arm}"
            folder.mkdir()
            shutil.copy2(QUAL / case / "input.extxyz", folder / "input.extxyz")
            current = {"case": case, "arm": arm, "seed": SEED, "status": "started",
                       "search_requests": 0, "fresh_requests": 0}
            ledger.dump(folder / "summary.json", current)
            arm_started = time.monotonic()
            remaining_campaign = CAMPAIGN_WALL - (time.monotonic() - campaign_started)
            surface = None
            fresh = None
            result = None
            before_calls = calc_counter["calls"]
            calls_after_search = before_calls
            try:
                if remaining_campaign <= 0:
                    raise TimeoutError("campaign_wall_cap_before_arm")
                calc.reset()
                base = ledger.CountedSurface(calc, folder / "requests.jsonl", SEARCH_CAP,
                                             min(WALL_PER_ARM, remaining_campaign))
                surface = ResetCountedSurface(base)
                def progress(p):
                    step = p.step
                    row = {"kind": p.kind, "next_index": p.next_index,
                           "cumulative_requests": p.evaluation_requests,
                           "current_energy_eV": p.current_energy,
                           "best_energy_eV": getattr(p.best, "energy", None),
                           "new_minimum_energy_eV": getattr(p.new_minimum, "energy", None)}
                    if step is not None:
                        row.update(step_index=step.index, status=step.status,
                                   accepted=step.accepted, energy_response=step.energy_response,
                                   evaluation_requests=step.evaluation_requests,
                                   climb_stages=len(step.climb), stage_summary=stage_counts(step),
                                   ls_update=step.ls_update, ls_preparation=step.ls_preparation,
                                   landing=step.landing)
                    ledger.append(folder / "progress.jsonl", row)
                    return False

                result = run_ssw(atoms.copy(), surface, steps=STEPS, config=config,
                                 rng=np.random.default_rng(SEED),
                                 ls=paper_ls if arm == "paper_ls" else None,
                                 recovered_rotation=rotation,
                                 checkpoint_path=folder / "checkpoint.pkl",
                                 progress_callback=progress)
                ledger.dump(folder / "result.json", result)
                calls_after_search = calc_counter["calls"]
                write(folder / "initial.extxyz", result.initial.atoms)
                best_atoms = result.best.atoms if hasattr(result.best, "atoms") else result.best
                write(folder / "best.extxyz", best_atoms)
                for index, minimum in enumerate(result.minima):
                    write(folder / f"minimum-{index:02d}.extxyz", minimum.atoms)
                current.update(status=result.status, outer_records=len(result.records),
                               outer_statuses=[r.status for r in result.records],
                               evaluation_requests=result.evaluation_requests,
                               request_count_matches_result=(surface.requests == result.evaluation_requests),
                               minima_count=len(result.minima))
                minima = result.minima
                if len(minima) <= FRESH_CAP:
                    fresh_rows = []
                    remaining_campaign = CAMPAIGN_WALL - (time.monotonic() - campaign_started)
                    remaining_arm = WALL_PER_ARM - (time.monotonic() - arm_started)
                    fresh = ResetCountedSurface(ledger.CountedSurface(
                        calc, folder / "fresh-requests.jsonl", FRESH_CAP,
                        max(0.0, min(remaining_campaign, remaining_arm))))
                    for index, minimum in enumerate(minima):
                        item = {"index": index, "reported_energy_eV": minimum.energy,
                                "reported_fmax_eV_A": minimum.max_force,
                                "reported_converged": minimum.converged,
                                "candidate_status": "force_candidate_only"}
                        try:
                            energy, forces = fresh.evaluate(minimum.atoms)
                            fmax = float(np.linalg.norm(forces, axis=1).max())
                            same_numbers = bool(np.array_equal(minimum.atoms.numbers, atoms.numbers))
                            same_cell = bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array))
                            same_pbc = bool(np.array_equal(minimum.atoms.pbc, atoms.pbc))
                            item.update(energy_eV=energy, energy_error_eV=energy - minimum.energy,
                                        fmax_eV_A=fmax,
                                        numerical_force_qualified=bool(minimum.converged and
                                            abs(energy - minimum.energy) <= 1e-6 and fmax <= config.fmax
                                            and same_numbers and same_cell and same_pbc),
                                        same_numbers=same_numbers, same_cell=same_cell,
                                        same_pbc=same_pbc,
                                        geometry=classify(minimum.atoms, graph, component_formulas, refs))
                        except Exception as exc:
                            item.update(numerical_force_qualified=False, error=repr(exc))
                        fresh_rows.append(item)
                        ledger.dump(folder / "fresh-checks.json", fresh_rows)
                    current.update(fresh_requests=fresh.requests,
                                   fresh_calculator_calls=calc_counter["calls"] - calls_after_search,
                                   fresh_checks=len(fresh_rows), fresh_checks_all_force_qualified=
                                   len(fresh_rows) == len(minima) and all(
                                       x.get("numerical_force_qualified") for x in fresh_rows))
                else:
                    current.update(status="fresh_cap_exceeded", fresh_cap_exceeded=len(minima))
            except Exception as exc:
                calls_after_search = calc_counter["calls"]
                current.update(status="exception", error=repr(exc))
                partial = getattr(exc, "result", None)
                if partial is not None:
                    ledger.dump(folder / "partial-result.json", partial)
                checkpoint_path = folder / "checkpoint.pkl"
                if checkpoint_path.is_file() and not (folder / "checkpoint.json").exists():
                    try:
                        saved = paper_reference.load_ssw_checkpoint(checkpoint_path)
                        ledger.dump(folder / "checkpoint.json", saved)
                    except Exception as checkpoint_error:
                        current["checkpoint_read_error"] = repr(checkpoint_error)
            finally:
                if surface is not None:
                    current.update(search_requests=surface.requests,
                                   search_calculator_calls=calls_after_search - before_calls,
                                   search_denials=surface.denials, search_boundary=surface.boundary)
                if fresh is not None:
                    current.update(fresh_requests=fresh.requests,
                                   fresh_denials=fresh.denials, fresh_boundary=fresh.boundary,
                                   fresh_calculator_calls=calc_counter["calls"] - calls_after_search)
                current["elapsed_seconds"] = time.monotonic() - arm_started
                current["completed_outer_steps"] = len(result.records) if result is not None else 0
                current["result_returned"] = result is not None
                current["request_count_matches_result"] = bool(result is not None and surface is not None
                    and surface.requests == result.evaluation_requests)
                ledger.dump(folder / "summary.json", current)
                overall["runs"].append(current.copy())
                overall["search_requests_total"] = sum(r.get("search_requests", 0)
                                                        for r in overall["runs"])
                overall["fresh_requests_total"] = sum(r.get("fresh_requests", 0)
                                                       for r in overall["runs"])
                overall["elapsed_seconds"] = time.monotonic() - campaign_started
                ledger.dump(output / "summary.json", overall)
            if time.monotonic() - campaign_started >= CAMPAIGN_WALL:
                break

    overall["elapsed_seconds"] = time.monotonic() - campaign_started
    all_returned = (len(overall["runs"]) == len(CASES) * len(ARMS)
                    and all(row.get("result_returned") for row in overall["runs"]))
    overall["status"] = "all_arms_returned" if all_returned else "campaign_incomplete"
    overall["total_calculator_calls"] = calc_counter["calls"]
    overall["search_calculator_calls"] = sum(r.get("search_calculator_calls", 0) for r in overall["runs"])
    overall["fresh_calculator_calls"] = sum(r.get("fresh_calculator_calls", 0) for r in overall["runs"])
    ledger.dump(output / "summary.json", overall)
    return 0 if all_returned else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="zero-PES provenance/dependency check")
    mode.add_argument("--execute", action="store_true", help="execute the fixed bounded comparison")
    parser.add_argument("--output", type=Path, help="new, previously absent output directory")
    args = parser.parse_args()
    if args.execute and args.output is None:
        parser.error("--execute requires --output NEW_DIR")
    report = preflight(args.output, require_cuda=args.execute)
    if args.preflight:
        print(json.dumps(load_ledger(Path(report["ledger_path"]))._jsonable(report), indent=2, sort_keys=True, allow_nan=False))
        return 0
    return execute(args.output.expanduser().resolve(), report)


if __name__ == "__main__":
    raise SystemExit(main())
