"""Bounded same-population GA-SSW versus independent-SSW comparison.

Research runner only. It uses existing PAM entry points, records search E/F
requests at the shared surface boundary, and makes no scientific qualification
claim. Run directories are created exclusively; existing output is never reused.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import time
import traceback

import numpy as np


class SearchStopped(RuntimeError):
    """Raised before an E/F call that would exceed a declared search guard."""


class CountedSurface:
    """Surface wrapper with a hard pre-call cap and timestamped request ledger."""
    def __init__(self, surface, *, cap, wall_seconds, started, ledger_path):
        self.surface = surface
        self.cap = int(cap)
        self.wall_seconds = float(wall_seconds)
        self.started = started
        self.ledger_path = Path(ledger_path)
        self.phase = "setup"
        self.phase_started = started
        self.deadline_reason = None
        self.phase_stop_reason = None
        self.records = []
        self.phase_cap = self.cap
        self.calculator_calculate_calls = None
        calculator = getattr(surface, "calculator", None)
        if calculator is not None and callable(getattr(calculator, "calculate", None)):
            original_calculate = calculator.calculate
            try:
                self.calculator_calculate_calls = 0
                def counted_calculate(*args, **kwargs):
                    self.calculator_calculate_calls += 1
                    return original_calculate(*args, **kwargs)
                calculator.calculate = counted_calculate
            except (AttributeError, TypeError):
                self.calculator_calculate_calls = None

    @property
    def requests(self):
        return int(self.surface.requests)

    def set_phase(self, phase, *, phase_cap=None):
        self.phase = str(phase)
        self.phase_started = time.monotonic()
        self.phase_cap = self.cap if phase_cap is None else min(self.cap, int(phase_cap))
        self.phase_stop_reason = None

    def evaluate(self, atoms):
        elapsed = time.monotonic() - self.started
        reason = ("search_cap" if self.requests >= self.cap else
                  "allocation_exhausted" if self.requests >= self.phase_cap else
                  "wall_seconds" if elapsed >= self.wall_seconds else None)
        if reason:
            if reason == "allocation_exhausted":
                self.phase_stop_reason = reason
            else:
                self.deadline_reason = reason
            row = {"kind": "refused", "request_index": self.requests + 1,
                   "phase": self.phase, "elapsed_seconds": elapsed,
                   "phase_seconds": time.monotonic() - self.phase_started,
                   "reason": reason}
            self.records.append(row)
            self._append(row)
            raise SearchStopped(reason)
        index = self.requests + 1
        try:
            energy, forces = self.surface.evaluate(atoms)
            row = {"kind": "request", "request_index": index,
                   "phase": self.phase, "elapsed_seconds": time.monotonic() - self.started,
                   "phase_seconds": time.monotonic() - self.phase_started,
                   "energy": float(energy)}
            self.records.append(row)
            self._append(row)
            return energy, forces
        except Exception as error:
            row = {"kind": "failed_request", "request_index": index,
                   "phase": self.phase, "elapsed_seconds": time.monotonic() - self.started,
                   "phase_seconds": time.monotonic() - self.phase_started,
                   "error": f"{type(error).__name__}: {error}"}
            self.records.append(row)
            self._append(row)
            raise

    @property
    def calculator_calls(self):
        return self.calculator_calculate_calls

    def _append(self, row):
        with self.ledger_path.open("a") as stream:
            stream.write(json.dumps(to_jsonable(row), allow_nan=False) + "\n")


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numbers") and hasattr(value, "positions"):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def allocate_remaining(total, n):
    """Allocate integer remainder evenly, assigning extra units in input order."""
    if total < 0 or n < 0:
        raise ValueError("total and n must be nonnegative")
    if n == 0:
        return []
    quotient, remainder = divmod(int(total), int(n))
    return [quotient + (i < remainder) for i in range(n)]


def run_walk_allocations(qualified, total, run_one):
    """Run independent starts in original order, returning unused quota to later starts."""
    remaining = list(qualified)
    left = int(total)
    rows = []
    stopped = False
    while remaining and left > 0:
        allowance = allocate_remaining(left, len(remaining))[0]
        index, quench_result = remaining.pop(0)
        if allowance <= 0:
            break
        outcome = run_one(index, quench_result, allowance)
        used = int(outcome["used"])
        if used < 0 or used > allowance:
            raise ValueError("walk request count falls outside its allocation")
        rows.append({"input_index": index, "status": outcome["status"],
                     "allocation": allowance, "used": used,
                     "unused_carried": allowance-used,
                     "request_start": outcome["request_start"],
                     "request_end": outcome["request_end"],
                     "phase_stop_reason": outcome.get("phase_stop_reason"),
                     "result": outcome.get("result")})
        left -= used
        if outcome.get("stop", False):
            stopped = True
            break
    if stopped:
        last_request = rows[-1]["request_end"]
        for (index, _), allowance in zip(remaining, allocate_remaining(left, len(remaining))):
            rows.append({"input_index": index, "status": "not_run_due_global_guard",
                "allocation": allowance, "used": 0, "unused_carried": allowance,
                "request_start": last_request + 1, "request_end": last_request,
                "phase_stop_reason": "global_guard", "result": None})
    return rows, left


def build_backend(settings):
    """Import selected calculator only after CLI parsing and at execution."""
    kind = settings["kind"]
    if kind == "emt":
        from ase.calculators.emt import EMT
        return EMT()
    if kind == "mace":
        expected_hash = settings.get("model_sha256")
        if expected_hash:
            model_path = Path(settings["model"]).expanduser().resolve()
            digest = hashlib.sha256()
            with model_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"MACE model SHA-256 mismatch: expected {expected_hash}, got {actual_hash}")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=settings["model"], head=settings.get("head"),
                              device=settings.get("device", "cpu"),
                              default_dtype=settings.get("dtype", "float64"),
                              enable_cueq=False, enable_oeq=False)
    raise ValueError("backend.kind must be 'emt' or 'mace'")


def walker_options(plan):
    options = {}
    if "native_mc" in plan:
        from pamssw.standalone.native_mc import NativeMCSettings
        options["mc"] = NativeMCSettings(**plan["native_mc"])
    if "recovered_direction" in plan:
        from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
        options["recovered_direction"] = RecoveredDirectionSettings(**plan["recovered_direction"])
    if "recovered_rotation" in plan:
        from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
        options["recovered_rotation"] = RecoveredRotationSettings(**plan["recovered_rotation"])
    return options


def initial_quench_options(ssw_config):
    """Mirror paper_ga's existing true-quench optimizer selection."""
    from ase.optimize import BFGS
    optimizer = getattr(ssw_config, "quench_optimizer", None)
    if optimizer not in ("safe-lbfgs-total", "scipy-lbfgsb", "ase-lbfgs-linesearch"):
        optimizer = BFGS
    return {"optimizer": optimizer,
            "lbfgs_memory": getattr(ssw_config, "lbfgs_memory", None)}


def observation_eligible(result):
    return bool(result.converged and result.surface == "true")


def _prepare(plan, out):
    from ase.io import read, write
    from pamssw.standalone import ASESurface, SSWConfig, PaperGAConfig
    from pamssw.standalone.paper_ga import run_ga_ssw
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.legacy_descriptor import cluster_descriptor
    from pamssw.standalone.surface import quench

    inputs = [Path(path).expanduser().resolve() for path in plan["inputs"]]
    if not inputs:
        raise ValueError("plan.inputs must include at least one explicit structure")
    initial = [read(path) for path in inputs]
    for atoms in initial:
        if not len(atoms) or atoms.pbc.any() or atoms.constraints:
            raise ValueError("this TYPE0 comparison requires unconstrained nonperiodic clusters")
    if any(not np.array_equal(initial[0].numbers, item.numbers) for item in initial[1:]):
        raise ValueError("all explicit population members must have identical ordered composition")
    ssw_cfg = SSWConfig(**plan["ssw"])
    ga_options = dict(plan["ga"])
    ga_options["proposal_type"] = 0
    ga_cfg = PaperGAConfig(**ga_options)
    walk_options = walker_options(plan)
    backend = build_backend(plan["backend"])
    from ase.io import write as ase_write
    ase_write(out / "inputs.extxyz", initial)
    surface = CountedSurface(ASESurface(backend), cap=plan["search_cap"],
                             wall_seconds=plan["wall_seconds"], started=time.monotonic(),
                             ledger_path=out / "evaluations.jsonl")
    descriptor = plan["descriptor"]
    bonds = {tuple(sorted(map(int, pair[:2]))): float(pair[2])
             for pair in descriptor["bond_lengths"]}
    refs = tuple(cluster_descriptor(a.numbers, a.positions, bonds,
                                    descriptor["neighbor_range"]) for a in initial[:3])
    if len(refs) < 3:
        raise ValueError("GA requires at least three raw initial structures for frozen descriptors")
    return (initial, surface, ssw_cfg, ga_cfg, bonds, refs, run_ga_ssw, run_ssw,
            quench, write, walk_options)


def _provenance(out):
    root = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", "pamssw"], cwd=root, text=True)
    if patch:
        raise RuntimeError("tracked pamssw source is modified; freeze source before execution")
    (out / "provenance.json").write_text(json.dumps({
        "git_head": head, "tracked_pamssw_diff": patch,
        "runner": str(Path(__file__).resolve())}, indent=2) + "\n")


def _summary_minimum(row, index):
    atoms = row.atoms if hasattr(row, "atoms") else getattr(row, "result", row).atoms
    energy = getattr(row, "energy", getattr(getattr(row, "result", None), "energy", None))
    fmax = getattr(row, "max_force", getattr(getattr(row, "result", None), "max_force", None))
    converged = getattr(row, "converged", getattr(getattr(row, "result", None), "converged", None))
    return {"index": index, "energy": energy, "max_force": fmax,
            "converged": converged, "atoms": atoms}


def _persist_result(out, result, summary, all_geometries, write):
    with (out / "result.pkl").open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2,
                                                  allow_nan=False) + "\n")
    if all_geometries:
        write(out / "observed_minima.extxyz", all_geometries)


def fresh_validation(out, initial, geometries, counted_surface, fmax, write):
    """Independently recalculate selected stored minima outside the search budget."""
    from pamssw.standalone import ASESurface
    search_requests = counted_surface.requests
    search_calculator_calls = counted_surface.calculator_calls
    eligible = [a for a in geometries if a.info.get("force_surface_qualified")]
    selected = []
    if eligible:
        selected.append(("best_true_force_converged", min(eligible, key=lambda a: a.info["energy"])))
    if len(initial[0]) == 60 and np.all(initial[0].numbers == 6):
        from research.ga_ssw.analyze_c60_random_development import graph_row
        for candidate in geometries:
            if not candidate.info.get("force_surface_qualified"):
                continue
            if graph_row(candidate.numbers, candidate.positions, 1.8)["graph_cage_candidate"]:
                if not any(np.array_equal(candidate.positions, old.positions) for _, old in selected):
                    selected.append(("first_graph_cage_candidate", candidate))
                break
    checks = []
    calculator = counted_surface.surface.calculator
    for label, atoms in selected[:4]:
        calculator.reset()
        fresh = ASESurface(calculator)
        try:
            energy, forces = fresh.evaluate(atoms)
            max_force = float(np.linalg.norm(forces, axis=1).max())
            checks.append({"label": label, "geometry_index": atoms.info.get("geometry_index"),
                "stored_energy": float(atoms.info["energy"]), "fresh_energy": float(energy),
                "energy_difference": float(energy-atoms.info["energy"]),
                "max_force": max_force, "fmax_threshold": float(fmax),
                "force_qualified": bool(max_force <= fmax),
                "composition_match": bool(np.array_equal(atoms.numbers, initial[0].numbers)),
                "pbc_match": bool(np.array_equal(atoms.pbc, initial[0].pbc)),
                "cell_match": bool(np.array_equal(atoms.cell.array, initial[0].cell.array))})
        except Exception as error:
            checks.append({"label": label, "geometry_index": atoms.info.get("geometry_index"),
                           "error": f"{type(error).__name__}: {error}"})
    if selected:
        write(out / "fresh-best.extxyz", selected[0][1])
    (out / "fresh-checks.json").write_text(json.dumps(to_jsonable({
        "search_requests": search_requests,
        "search_calculator_calculate_calls": search_calculator_calls,
        "checks": checks},), indent=2, allow_nan=False) + "\n")
    return checks


def run(plan_path, arm, out_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    out = Path(out_path).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
    _provenance(out)
    started = time.monotonic()
    try:
        (initial, surface, ssw_cfg, ga_cfg, bonds, refs, run_ga, run_ssw, quench,
         write, walk_options) = _prepare(plan, out)
        # _prepare's clock must include setup as well as search; reset wall origin.
        surface.started = started
        search_result = None
        phases = []
        geometries = []
        if arm == "ga":
            surface.set_phase("ga")
            checkpoint_path = out / "ga-stage.pkl"
            def save_boundary(state):
                state.save(checkpoint_path)
                # Stop at the next existing GA boundary; refuse all PES calls
                # after the wall guard without inventing unspent request costs.
                return surface.deadline_reason == "wall_seconds"
            try:
                search_result = run_ga(initial, surface, groups=None, references=refs,
                descriptor_bonds=bonds, descriptor_weights=plan["descriptor"]["weights"],
                neighbor_range=plan["descriptor"]["neighbor_range"],
                descriptor_row_order="full_fingerprint",
                proposal_bond_limits={}, config=ga_cfg, ssw_config=ssw_cfg,
                rng=np.random.default_rng(plan["seed"]), max_evaluations=plan["search_cap"],
                checkpoint_callback=save_boundary,
                **walk_options)
            except SearchStopped:
                if checkpoint_path.exists():
                    with checkpoint_path.open("rb") as stream:
                        search_result = pickle.load(stream)
                    observations = search_result.observations
                else:
                    search_result = {"status": surface.deadline_reason or "wall_seconds",
                                     "observations": (), "search_requests": surface.requests}
                    observations = ()
                summary = {"arm": arm, "status": surface.deadline_reason or "wall_seconds",
                    "search_requests": surface.requests, "search_cap": plan["search_cap"],
                    "wall_seconds": time.monotonic() - started,
                    "boundary_reason": surface.deadline_reason,
                    "checkpoint_phase": getattr(search_result, "phase", None),
                    "checkpoint_evaluation_requests": getattr(search_result, "evaluation_requests", None),
                    "phases": [], "minima": []}
                for obs in observations:
                    atoms = obs.result.atoms.copy()
                    atoms.info.update(geometry_index=len(geometries), energy=float(obs.result.energy),
                        max_force=float(obs.result.max_force), converged=bool(obs.result.converged),
                        surface=str(obs.result.surface), phase=obs.phase,
                        force_surface_qualified=observation_eligible(obs.result),
                        eligible_for_archive=bool(obs.eligible_for_archive), observation_id=int(obs.id))
                    geometries.append(atoms)
                    summary["minima"].append({"id": obs.id, "phase": obs.phase,
                        "eligible_for_archive": obs.eligible_for_archive,
                        "energy": obs.result.energy, "max_force": obs.result.max_force,
                        "converged": obs.result.converged,
                        "geometry_index": atoms.info["geometry_index"]})
                summary["calculator_calculate_calls"] = surface.calculator_calls
                summary["search_calculator_calculate_calls"] = surface.calculator_calls
                summary["fresh_checks"] = fresh_validation(out, initial, geometries,
                    surface, ga_cfg.quench_fmax, write)
                _persist_result(out, search_result, summary, geometries, write)
                return 0
            cursor = 0
            compact_stages = []
            for stage in search_result.stages:
                start = cursor + 1
                cursor += int(stage.evaluation_requests)
                compact = {"phase": stage.phase, "generation": stage.generation,
                    "seed_id": stage.seed_id, "status": stage.status,
                    "evaluation_requests": stage.evaluation_requests,
                    "observations": stage.observations, "cycle": stage.cycle,
                    "request_start": start, "request_end": cursor}
                compact_stages.append(compact)
                phases.append(compact)
            observations = list(search_result.observations)
            for obs in observations:
                atoms = obs.result.atoms.copy()
                atoms.info.update(geometry_index=len(geometries), energy=float(obs.result.energy),
                    max_force=float(obs.result.max_force), converged=bool(obs.result.converged),
                    surface=str(obs.result.surface), phase=obs.phase,
                    force_surface_qualified=observation_eligible(obs.result),
                    eligible_for_archive=bool(obs.eligible_for_archive), observation_id=int(obs.id))
                geometries.append(atoms)
            summary = {"arm": arm, "status": search_result.status,
                "search_requests": surface.requests, "search_cap": plan["search_cap"],
                "wall_seconds": time.monotonic() - started,
                "boundary_reason": surface.deadline_reason,
                "phases": phases, "stages": compact_stages,
                "failures": [{"phase": f.phase, "generation": f.generation,
                    "seed_id": f.seed_id, "reason": f.reason,
                    "evaluation_requests": f.evaluation_requests}
                    for f in search_result.failures],
                "minima": [{"id": obs.id, "phase": obs.phase, "generation": obs.generation,
                    "eligible_for_archive": obs.eligible_for_archive,
                    "force_surface_qualified": observation_eligible(obs.result),
                    "energy": obs.result.energy, "max_force": obs.result.max_force,
                    "converged": obs.result.converged,
                    "geometry_index": index}
                    for index, obs in enumerate(observations)]}
        else:
            from pamssw.standalone.surface import QuenchResult
            initial_rows = []
            qualified = []
            for index, atoms in enumerate(initial):
                surface.set_phase(f"initial_quench:{index}")
                before = surface.requests
                try:
                    q = quench(atoms, surface, fmax=ga_cfg.quench_fmax,
                               steps=ga_cfg.quench_steps, **initial_quench_options(ssw_cfg))
                    status = "qualified" if q.converged else "unqualified"
                except SearchStopped as error:
                    initial_rows.append({"index": index, "status": f"censored:{error}",
                        "request_start": before + 1, "request_end": surface.requests})
                    initial_rows.extend({"index": later, "status": "not_run_due_budget",
                        "request_start": surface.requests + 1, "request_end": surface.requests}
                        for later in range(index + 1, len(initial)))
                    break
                except Exception as error:
                    q = getattr(error, "result", None)
                    status = f"failed:{type(error).__name__}"
                    if q is None:
                        initial_rows.append({"index": index, "status": status,
                            "error": repr(error), "request_start": before + 1,
                            "request_end": surface.requests})
                        continue
                row = {"index": index, "status": status, "result": q,
                       "request_start": before + 1, "request_end": surface.requests}
                initial_rows.append(row)
                initial_atoms = q.atoms.copy()
                initial_atoms.info.update(geometry_index=len(geometries), energy=float(q.energy),
                    max_force=float(q.max_force), converged=bool(q.converged), surface=str(q.surface),
                    phase="initial_quench", force_surface_qualified=observation_eligible(q),
                    eligible_for_archive=observation_eligible(q),
                    input_index=int(index))
                geometries.append(initial_atoms)
                if q.converged:
                    qualified.append((index, q))
            def run_one(index, q, allowance):
                surface.set_phase(f"ssw:{index}", phase_cap=surface.requests + allowance)
                before = surface.requests
                result = None
                try:
                    result = run_ssw(q.atoms.copy(), surface,
                                     steps=int(plan["outer_steps"]), config=ssw_cfg,
                                     rng=np.random.default_rng(plan["seed"] + index),
                                     checkpoint_path=out / f"ssw-{index}-checkpoint.pkl",
                                     **walk_options)
                    status = result.status
                except Exception as error:
                    result = getattr(error, "result", None)
                    status = f"failed:{type(error).__name__}"
                used = surface.requests - before
                if result is not None:
                    seen_landing_ids = set()
                    for minimum in result.minima:
                        atoms = minimum.atoms.copy()
                        atoms.info.update(geometry_index=len(geometries), energy=float(minimum.energy),
                            max_force=float(minimum.max_force), converged=bool(minimum.converged),
                            surface=str(minimum.surface), phase=f"ssw:{index}",
                            force_surface_qualified=observation_eligible(minimum),
                            eligible_for_archive=observation_eligible(minimum))
                        geometries.append(atoms)
                        seen_landing_ids.add(id(minimum))
                    for record in result.records:
                        landing = getattr(record, "landing", None)
                        if isinstance(landing, QuenchResult) and id(landing) not in seen_landing_ids:
                            atoms = landing.atoms.copy()
                            atoms.info.update(geometry_index=len(geometries), energy=float(landing.energy),
                                max_force=float(landing.max_force), converged=bool(landing.converged),
                                surface=str(landing.surface), phase=f"ssw:{index}",
                                force_surface_qualified=observation_eligible(landing),
                                eligible_for_archive=observation_eligible(landing),
                                step_status=str(record.status), accepted=bool(record.accepted))
                            geometries.append(atoms)
                            seen_landing_ids.add(id(landing))
                return {"status": status, "used": used, "result": result,
                        "request_start": before + 1, "request_end": surface.requests,
                        "phase_stop_reason": surface.phase_stop_reason,
                        "stop": surface.deadline_reason is not None}

            left = max(0, int(plan["search_cap"]) - surface.requests)
            walk_rows, left = run_walk_allocations(
                qualified, left,
                lambda index, q, allowance: run_one(index, q, allowance)
                if time.monotonic()-started < plan["wall_seconds"] else
                {"status": "not_run_due_wall", "used": 0, "result": None,
                 "request_start": surface.requests+1, "request_end": surface.requests,
                 "stop": True})
            phases.append({"phase": "initial_quenches", "count": len(initial_rows),
                           "request_start": 1, "request_end": sum(
                               max(0, r.get("request_end", 0)-r.get("request_start", 1)+1)
                               for r in initial_rows)})
            phases.extend({"phase": f"ssw:{row['input_index']}",
                           "start_request": row["request_start"],
                           "end_request": row["request_end"],
                           "allocation": row["allocation"], "used": row["used"],
                           "unused_carried": row["unused_carried"],
                           "phase_stop_reason": row["phase_stop_reason"],
                           "status": row["status"]}
                          for row in walk_rows)
            summary = {"arm": arm, "status": "completed", "search_requests": surface.requests,
                "search_cap": plan["search_cap"], "wall_seconds": time.monotonic()-started,
                "boundary_reason": surface.deadline_reason, "phases": phases,
                "allocation_rule": "At each start, floor(remaining requests / remaining qualified starts), extras to earlier input indices; unused requests are returned then remaining starts are reallocated in original order.",
                "initial_quenches": [{k: v for k, v in row.items() if k != "result"}
                                     for row in initial_rows],
                "walks": [{k: v for k, v in row.items() if k != "result"}
                          for row in walk_rows]}
            if time.monotonic() - started >= plan["wall_seconds"]:
                summary["status"] = "wall_censored"
            elif surface.deadline_reason == "search_cap":
                summary["status"] = "request_cap_censored"
            search_result = {"arm": arm, "initial_quenches": initial_rows, "walks": walk_rows,
                             "phases": phases, "search_requests": surface.requests}
        if surface.deadline_reason == "wall_seconds":
            summary["algorithm_status"] = summary["status"]
            summary["status"] = "wall_censored"
        summary["failed_search_requests"] = sum(r["kind"] == "failed_request" for r in surface.records)
        summary["refused_search_calls"] = sum(r["kind"] == "refused" for r in surface.records)
        summary["calculator_calculate_calls"] = surface.calculator_calls
        summary["fresh_checks"] = fresh_validation(out, initial, geometries,
            surface, ga_cfg.quench_fmax, write)
        summary["search_calculator_calculate_calls"] = summary["calculator_calculate_calls"]
        _persist_result(out, search_result, summary, geometries, write)
        return 0
    except Exception as error:
        failure = {"status": "exception", "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc(), "search_requests": locals().get("surface").requests if "surface" in locals() else 0,
                   "wall_seconds": time.monotonic()-started}
        (out / "failure.json").write_text(json.dumps(to_jsonable(failure), indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("ssw", "ga"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    return run(args.plan, args.arm, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
