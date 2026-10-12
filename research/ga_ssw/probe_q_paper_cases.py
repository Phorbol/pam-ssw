#!/usr/bin/env python3
"""Bounded Q-off/S1/S2 fixed-cell paper-case lifecycle probe.

Preparation is the default.  Execution is opt-in with ``--execute``; this
runner does not generate or relax input structures before the SSW starts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
DATA_ROOT = Path(os.environ.get("PAMSSW_RESEARCH_ROOT", DEFAULT_DATA_ROOT)).expanduser().resolve()
EVIDENCE = DATA_ROOT / "research/ga_ssw/evidence/q-paper-entry-20261012"
LEDGER = DATA_ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
MODEL_C60 = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_TIO2 = Path("/home/gengjianrui/.cache/mace/mace-omat-0-small.model")


def configure_data_root(data_root: Path):
    global DATA_ROOT, EVIDENCE, LEDGER, CASES
    DATA_ROOT = Path(data_root).expanduser().resolve()
    EVIDENCE = DATA_ROOT / "research/ga_ssw/evidence/q-paper-entry-20261012"
    LEDGER = DATA_ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
    c60 = DATA_ROOT / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-2"
    tio2 = DATA_ROOT / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run-1663408"
    CASES = {
        "c60-isomer2": {"input": c60 / "final.extxyz", "input_sha256": "ce9a47447535c400dc7d3c527ef891ed326563f357858916c2265232fbf980fb",
                        "qualification": c60 / "result.json", "qualification_sha256": "f3bcb0ad4d7cb924f1678bb5016c78fd9e9106ce22f1c995e58351a7d9c10353",
                        "model": MODEL_C60, "model_sha256": "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47",
                        "head": "omol", "cutoff_A": 3.0, "center": 0, "neighbor_species": (6,), "composition": {6: 60}, "periodic": False},
        "tio2-phase87": {"input": tio2 / "phase87_is/endpoint.extxyz", "input_sha256": "564aae0f5f7638b79edc6e23befe1e9ee12ecb6641fbf922e31a20ad8b23afc5",
                         "qualification": tio2 / "phase87_is/result.json", "qualification_sha256": None,
                         "manifest": tio2 / "effective-plan.json",
                         "model": MODEL_TIO2, "model_sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
                         "head": "omat_pbe", "cutoff_A": 3.0, "center": 8, "neighbor_species": (8,),
                         "composition": {8: 8, 22: 4}, "periodic": True},
    }


configure_data_root(DATA_ROOT)
ARMS = ("Qoff", "S1", "S2")
SEED = 20261012
SEARCH_CAP = 3000
FRESH_CAP = 3  # initial state and at most two outer landings
FRESH_WALL_SECONDS = 30
ARM_WALL_SECONDS = 90
CAMPAIGN_WALL_SECONDS = 30 * 60
OUTER_STEPS = 2
MAX_GAUSSIANS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def serial(value):
    from ase import Atoms
    if isinstance(value, Atoms):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return serial(asdict(value))
    return value


def dump(path: Path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def descriptor_mode(atoms, family, spec):
    from pamssw.standalone.descriptor_direction import DescriptorMode
    return DescriptorMode(family=family, center=spec["center"],
                          neighbor_species=spec["neighbor_species"],
                          cutoff=spec["cutoff_A"], exponent=0, epsilon=0.0,
                          degree=(2 if family == "S2" else None))


def validate_inputs():
    from ase.io import read
    prepared = {}
    for name, spec in CASES.items():
        for key in ("input", "model", "qualification"):
            path = spec[key]
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(spec["input"]) != spec["input_sha256"]:
            raise ValueError(f"frozen input hash changed: {spec['input']}")
        if sha256(spec["model"]) != spec["model_sha256"]:
            raise ValueError(f"frozen model hash changed: {spec['model']}")
        if spec["qualification_sha256"] and sha256(spec["qualification"]) != spec["qualification_sha256"]:
            raise ValueError("C60 qualification record changed")
        atoms = read(spec["input"], index=0)
        observed = {int(z): int(np.count_nonzero(atoms.numbers == z)) for z in np.unique(atoms.numbers)}
        if observed != spec["composition"] or not np.isfinite(atoms.positions).all():
            raise ValueError(f"unexpected composition/coordinates for {name}: {observed}")
        if bool(np.any(atoms.pbc)) != spec["periodic"] or atoms.constraints:
            raise ValueError(f"unexpected PBC/constraints for {name}")
        if name == "c60-isomer2":
            qualification = json.loads(spec["qualification"].read_text())
            if not qualification.get("force_qualified") or not all(qualification.get("labeled_edges_preserved", {}).values()):
                raise ValueError("saved C60 isomer-2 force/edge qualification did not pass")
        else:
            qualification = json.loads(spec["qualification"].read_text())
            if qualification.get("case") != "phase87_is" or not qualification.get("numerical_qualified"):
                raise ValueError("saved TiO2 phase87 model qualification did not pass")
        prepared[name] = atoms
    # Ensure the TiO2 entry is the saved SI phase87 structure and model/head
    # provenance remains tied to the published qualification manifest.
    plan = json.loads(CASES["tio2-phase87"]["manifest"].read_text())
    if (plan.get("model_sha256") != CASES["tio2-phase87"]["model_sha256"] or
            plan.get("calculator", {}).get("head") != "omat_pbe" or
            plan.get("paper", {}).get("doi") != "10.1039/c7sc01459g"):
        raise ValueError("TiO2 paper/model provenance mismatch")
    return prepared


def geometry_diagnostics(atoms):
    """Describe local connectivity/distances without assigning phase success."""
    numbers = np.asarray(atoms.numbers, dtype=int)
    distances = np.asarray(atoms.get_all_distances(mic=bool(np.any(atoms.pbc))), dtype=float)
    nonzero = distances[distances > 1e-8]
    output = {"minimum_nonself_distance_A": None if not len(nonzero) else float(nonzero.min()),
              "atom_count": len(atoms), "periodic": bool(np.any(atoms.pbc))}
    if not np.any(atoms.pbc):
        graph_rows = {}
        for cutoff in (1.64, 1.70, 1.80):
            adjacency = [set() for _ in numbers]
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    if distances[i, j] < cutoff:
                        adjacency[i].add(j); adjacency[j].add(i)
            unseen = set(range(len(numbers))); components = 0
            while unseen:
                components += 1
                stack = [unseen.pop()]
                while stack:
                    for j in adjacency[stack.pop()] & unseen:
                        unseen.remove(j); stack.append(j)
            degrees = [len(row) for row in adjacency]
            graph_rows[str(cutoff)] = {"edges": int(sum(degrees) // 2), "components": components,
                                       "degree_counts": {str(d): degrees.count(d) for d in sorted(set(degrees))}}
        output["distance_graphs"] = graph_rows
    else:
        cutoff = 3.0
        rows = {}
        for z in sorted(set(numbers.tolist())):
            counts = []
            for i in np.flatnonzero(numbers == z):
                counts.append(sum(1 for j, zj in enumerate(numbers) if i != j and distances[i, j] < cutoff))
            rows[str(int(z))] = {"cutoff_A": cutoff,
                                 "coordination_counts": {str(n): counts.count(n) for n in sorted(set(counts))}}
        output["coordination_by_center_species"] = rows
        output["interpretation"] = "distance and coordination diagnostic only; no phase assignment"
    return output


def prepare():
    """CPU-only entry gate, including descriptor value/gradient qualification."""
    import inspect
    from pamssw.standalone.descriptor_direction import descriptor_value_gradient
    from pamssw.standalone import SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    inputs = validate_inputs()
    rows = {}
    required_ssw_parameters = {"recovered_direction", "checkpoint_path", "checkpoint"}
    missing_ssw_parameters = required_ssw_parameters - set(inspect.signature(run_ssw).parameters)
    if missing_ssw_parameters:
        raise RuntimeError(f"run_ssw API is missing required fields: {sorted(missing_ssw_parameters)}")
    for name, atoms in inputs.items():
        spec = CASES[name]
        rows[name] = {"input": str(spec["input"]), "input_sha256": sha256(spec["input"]),
                      "qualification": str(spec["qualification"]),
                      "model": str(spec["model"]), "model_sha256": sha256(spec["model"]),
                      "head": spec["head"], "fixed_cell": True,
                      "geometry_diagnostics": geometry_diagnostics(atoms),
                      "descriptor_modes": {}}
        for family in ("S1", "S2"):
            mode = descriptor_mode(atoms, family, spec)
            value, gradient, image_count = descriptor_value_gradient(atoms, mode)
            norm = float(np.linalg.norm(gradient))
            valid = bool(np.isfinite(value) and np.isfinite(gradient).all() and value != 0.0 and norm != 0.0)
            rows[name]["descriptor_modes"][family] = {
                "family": family, "center": spec["center"], "neighbor_species": spec["neighbor_species"],
                "cutoff_A": spec["cutoff_A"], "exponent": 0, "epsilon_A": 0.0,
                "degree": 2 if family == "S2" else None, "value": float(value),
                "gradient_norm": norm, "selected_image_count": int(image_count), "qualified": valid}
            if not valid:
                raise ValueError(f"{name} {family} Q coordinate is zero or nonfinite; do not tune after observing it")
        config = SSWConfig(width=0.6, rotation_bias=1.0, max_gaussians=MAX_GAUSSIANS,
                           temperature_K=150.0, fmax=0.05, bias_fmax=0.1,
                           relax_steps=500, fd_step=0.001, rotation_hvp=39,
                           rotation_tol=0.02, forward_force=0.1,
                           direction_sampling="global", rotation_solver="broyden-euclidean",
                           cluster_frame=("translation_only" if spec["periodic"] else "direction_only"),
                           quench_optimizer="safe-lbfgs-total", lbfgs_memory=500,
                           rotation_exit_policy="force_or_budget")
        rows[name]["api_preflight"] = {"ssw_config_constructed": True,
                                       "run_ssw_contract_fields": sorted(required_ssw_parameters),
                                       "pes_calls": 0, "ssw_config": asdict(config),
                                       "direction_settings": {}}
        for arm in ARMS:
            family_modes = () if arm == "Qoff" else (descriptor_mode(atoms, arm, spec),)
            settings = RecoveredDirectionSettings(
                ratio_local=50, local_probability=0.5, group_threshold=0.5,
                pre_rotmax=5, rotmax=15, pre_ftol=0.2, ftol=0.02,
                metric="euclidean", max_force_calls=40,
                c1_radius_policy="restricted", geometry=("periodic_local" if spec["periodic"] else "nonperiodic"),
                descriptor_modes=family_modes,
                **({} if arm == "Qoff" else {"descriptor_probability": 1.0}))
            rows[name]["api_preflight"]["direction_settings"][arm] = serial(settings)
    return rows


def execute(out: Path):
    import ase
    import mace
    import scipy
    import torch
    from ase.io import write
    from mace.calculators import MACECalculator

    if out.exists():
        raise FileExistsError(f"output must be a new directory: {out}")
    inputs = validate_inputs()
    prepared_modes = prepare()
    from pamssw.standalone import (ASESurface, SSWConfig, load_ssw_checkpoint,
                                   run_ssw)
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings

    ledger = load_module("q_paper_entry_ledger", LEDGER)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve(), out / Path(__file__).name)
    started = time.monotonic()
    record = {"status": "running", "scope": "bounded fixed-cell paper-case lifecycle probe; one seed per structure; no superiority or phase-transition claim",
              "protocol": str(EVIDENCE / "protocol.md"), "seed": SEED,
              "limits": {"cases": list(CASES), "arms": ARMS, "outer_steps": OUTER_STEPS,
                         "max_gaussians": MAX_GAUSSIANS, "search_requests_per_arm": SEARCH_CAP,
                         "fresh_requests_per_arm": FRESH_CAP, "wall_seconds_per_arm": ARM_WALL_SECONDS,
                         "campaign_wall_seconds": CAMPAIGN_WALL_SECONDS, "gpu_count": 1},
              "sources": {}, "runs": [], "search_requests_total": 0,
              "fresh_requests_total": 0}
    for name, spec in CASES.items():
        record["sources"][name] = {k: str(spec[k]) for k in ("input", "qualification", "model")}
        record["sources"][name].update(input_sha256=sha256(spec["input"]), model_sha256=sha256(spec["model"]), head=spec["head"])
    record["runtime"] = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                         "mace": mace.__version__, "ase": ase.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    dump(out / "provenance.json", record)

    for case_name, atoms in inputs.items():
        spec = CASES[case_name]
        calc_args = dict(model_paths=str(spec["model"]), head=spec["head"], device="cuda",
                         default_dtype="float64", enable_cueq=False, enable_oeq=False)
        search_calc = MACECalculator(**calc_args)
        call_counter = ledger.instrument_calculate(search_calc)
        config = SSWConfig(width=0.6, rotation_bias=1.0, max_gaussians=MAX_GAUSSIANS,
                           temperature_K=150.0, fmax=0.05, bias_fmax=0.1,
                           relax_steps=500, fd_step=0.001, rotation_hvp=39,
                           rotation_tol=0.02, forward_force=0.1,
                           direction_sampling="global", rotation_solver="broyden-euclidean",
                           cluster_frame=("translation_only" if spec["periodic"] else "direction_only"),
                           quench_optimizer="safe-lbfgs-total",
                           lbfgs_memory=500, rotation_exit_policy="force_or_budget")
        modes = {family: descriptor_mode(atoms, family, spec) for family in ("S1", "S2")}
        descriptor_precheck = prepared_modes[case_name]["descriptor_modes"]
        record.setdefault("effective_case_configs", {})[case_name] = {"case": case_name, "input": str(spec["input"]),
                                            "input_sha256": sha256(spec["input"]),
                                            "qualification": str(spec["qualification"]),
                                            "qualification_manifest": str(spec.get("manifest", spec["qualification"])),
                                            "model": str(spec["model"]), "model_sha256": spec["model_sha256"],
                                            "head": spec["head"], "ssw_config": asdict(config),
                                            "descriptor_precheck": descriptor_precheck,
                                            "geometry_diagnostics": prepared_modes[case_name]["geometry_diagnostics"],
                                            "recovered_direction_common": {"ratio_local": 50, "local_probability": 0.5,
                                                "group_threshold": 0.5, "pre_rotmax": 5, "rotmax": 15,
                                                "pre_ftol": 0.2, "ftol": 0.02, "metric": "euclidean",
                                                "max_force_calls": 40, "geometry": "periodic_local" if spec["periodic"] else "nonperiodic"}}
        dump(out / "provenance.json", record)
        for arm in ARMS:
            folder = out / f"{case_name}-{arm}"
            folder.mkdir()
            write(folder / "source-input.extxyz", atoms)
            family_modes = () if arm == "Qoff" else (modes[arm],)
            setting = RecoveredDirectionSettings(
                ratio_local=50, local_probability=0.5, group_threshold=0.5,
                pre_rotmax=5, rotmax=15, pre_ftol=0.2, ftol=0.02,
                metric="euclidean", max_force_calls=40,
                c1_radius_policy="restricted", geometry=("periodic_local" if spec["periodic"] else "nonperiodic"),
                descriptor_modes=family_modes,
                **({} if arm == "Qoff" else {"descriptor_probability": 1.0}))
            row = {"case": case_name, "arm": arm, "seed": SEED, "status": "started",
                   "config": asdict(config), "recovered_direction": serial(setting),
                   "descriptor_precheck": descriptor_precheck, "cell_fixed": True}
            dump(folder / "summary.json", row)
            run_started = time.monotonic()
            calls_before = call_counter["calls"]
            surface = ledger.CountedSurface(search_calc, folder / "requests.jsonl",
                                             SEARCH_CAP, ARM_WALL_SECONDS)
            result = None
            try:
                if time.monotonic() - started >= CAMPAIGN_WALL_SECONDS:
                    raise TimeoutError("campaign wall cap")
                kwargs = {"recovered_direction": setting}
                # Deliberately checkpoint and explicitly resume at the outer
                # boundary; the second call adds exactly one outer attempt.
                first = run_ssw(atoms.copy(), surface, steps=1, config=config,
                                rng=np.random.default_rng(SEED), checkpoint_path=folder / "checkpoint.pkl", **kwargs)
                row["first_segment"] = {"status": first.status, "records": len(first.records),
                                        "requests": first.evaluation_requests, "checkpoint_written": True}
                dump(folder / "first-segment-result.json", first)
                if first.minima:
                    write(folder / "first-segment-minima.extxyz", [minimum.atoms for minimum in first.minima])
                if (folder / "checkpoint.pkl").is_file():
                    try:
                        checkpoint = load_ssw_checkpoint(folder / "checkpoint.pkl")
                        row["first_segment"].update(checkpoint_status=checkpoint.status,
                                                    checkpoint_next_index=checkpoint.next_index)
                    except Exception as checkpoint_error:
                        checkpoint = None
                        row["first_segment"].update(checkpoint_status="invalid_or_unreadable",
                                                    checkpoint_error=repr(checkpoint_error))
                else:
                    checkpoint = None
                    row["first_segment"].update(checkpoint_status="missing", checkpoint_next_index=None)
                if (checkpoint is not None and checkpoint.status == "completed" and
                        checkpoint.next_index == 1):
                    if time.monotonic() - run_started >= ARM_WALL_SECONDS:
                        result = first
                        row["status"] = "nonresumable_wall_limit_after_first_segment"
                        row["resume_skipped"] = "arm wall cap before resumed segment"
                    else:
                        result = run_ssw(atoms.copy(), surface, steps=1, config=config,
                                         rng=np.random.default_rng(SEED), checkpoint=checkpoint,
                                         checkpoint_path=folder / "checkpoint.pkl", **kwargs)
                        row["status"] = result.status
                        row["resume_status"] = "completed_second_segment"
                else:
                    result = first
                    row["status"] = "nonresumable_after_first_segment"
                    row["resume_skipped"] = "first segment did not produce a completed one-step boundary"
                dump(folder / "result.json", result)
                if result.minima:
                    write(folder / "minima.extxyz", [minimum.atoms for minimum in result.minima])
                if result.current is not None:
                    write(folder / "current.extxyz", result.current)
                trajectories = []
                for rec in result.records:
                    if rec.landing is not None:
                        trajectories.append(rec.landing.atoms)
                if trajectories:
                    write(folder / "landings.extxyz", trajectories)
                row.update(outer_records=len(result.records), search_requests_reported=result.evaluation_requests,
                           records=[{"index": r.index, "status": r.status, "accepted": r.accepted,
                                     "gaussians": len(r.climb), "requests": r.evaluation_requests,
                                     "landing_surface": None if r.landing is None else r.landing.surface,
                                     "landing_energy_eV": None if r.landing is None else r.landing.energy,
                                     "landing_fmax_eV_A": None if r.landing is None else r.landing.max_force}
                                    for r in result.records])
                # Fresh independent calculator checks source and each outer
                # landing; no archive or phase identity is inferred here.
                fresh_started = time.monotonic()
                fresh_calc = MACECalculator(**calc_args)
                fresh_setup_seconds = time.monotonic() - fresh_started
                fresh_counter = ledger.instrument_calculate(fresh_calc)
                fresh_surface = ledger.CountedSurface(fresh_calc, folder / "fresh-requests.jsonl",
                                                       FRESH_CAP, max(0.0, FRESH_WALL_SECONDS - fresh_setup_seconds))
                checks = []
                targets = [(f"minimum-{i}", minimum.atoms) for i, minimum in enumerate(result.minima)]
                if not targets:
                    initial_atoms = getattr(getattr(result, "initial", None), "atoms", None)
                    if initial_atoms is not None:
                        targets = [("initial", initial_atoms)]
                    else:
                        targets = [("source-input", atoms)]
                for label, target in targets[:FRESH_CAP]:
                    try:
                        fresh_calc.reset()
                        e, f = fresh_surface.evaluate(target)
                        checks.append({"target": label, "energy_eV": e,
                                       "reported_energy_eV": next((float(m.energy) for i, m in enumerate(result.minima)
                                                                    if label == f"minimum-{i}"), None),
                                       "fmax_eV_A": float(np.linalg.norm(f, axis=1).max()),
                                       "force_qualified": bool(np.linalg.norm(f, axis=1).max() <= config.fmax),
                                       "same_numbers": bool(np.array_equal(target.numbers, atoms.numbers)),
                                       "same_cell": bool(np.array_equal(target.cell.array, atoms.cell.array)),
                                       "same_pbc": bool(np.array_equal(target.pbc, atoms.pbc)),
                                       "cell_change_from_fixed_input_A": float(np.max(np.abs(target.cell.array - atoms.cell.array))),
                                       "geometry_diagnostics": geometry_diagnostics(target)})
                    except Exception as error:
                        checks.append({"target": label, "error": repr(error), "qualified": False})
                row["fresh_checks"] = checks
                row["fresh_requests"] = fresh_surface.requests
                row["fresh_calculate_calls"] = fresh_counter["calls"]
                row["fresh_setup_seconds"] = fresh_setup_seconds
                row["fresh_elapsed_seconds"] = time.monotonic() - fresh_started
                if any("error" in check for check in checks) and fresh_surface.boundary:
                    row["failure_class"] = "fresh_budget_censored"
                    row["fresh_boundary"] = fresh_surface.boundary
                elif any("error" in check for check in checks):
                    row["failure_class"] = "hard_calculator_failure_during_fresh_check"
            except Exception as error:
                row.update(status="exception", error=repr(error), traceback=traceback.format_exc())
                # RuntimeError carrying a partial SSW result or an explicit
                # budget boundary is a retained algorithmic censor.  A PES or
                # calculator exception with neither marker stops the panel.
                partial = getattr(error, "result", None)
                if partial is not None:
                    dump(folder / "partial-result.json", partial)
                    try:
                        write(folder / "partial-minima.extxyz", [minimum.atoms for minimum in partial.minima])
                    except Exception:
                        pass
                row["failure_class"] = ("algorithm_or_budget" if partial is not None or surface.boundary
                                        else "hard_calculator_or_setup_failure")
                row["checkpoint_exists"] = (folder / "checkpoint.pkl").is_file()
                if row["checkpoint_exists"]:
                    try:
                        cp = load_ssw_checkpoint(folder / "checkpoint.pkl")
                        row["checkpoint_status"] = cp.status
                        row["checkpoint_next_index"] = cp.next_index
                    except Exception as checkpoint_error:
                        row["checkpoint_load_error"] = repr(checkpoint_error)
            finally:
                row.update(search_requests=surface.requests, search_denials=surface.denials,
                           search_boundary=surface.boundary, elapsed_seconds=time.monotonic() - run_started,
                           search_calculate_calls=call_counter["calls"] - calls_before)
                with (folder / "requests.jsonl").open() as stream:
                    row["search_failures"] = sum(json.loads(line).get("kind") == "search_failure" for line in stream)
                if row["search_failures"]:
                    row["failure_class"] = "hard_calculator_failure_during_search"
                dump(folder / "summary.json", row)
                record["runs"].append(row)
                record["search_requests_total"] += surface.requests
                record["fresh_requests_total"] += int(row.get("fresh_requests", 0))
                record["actual_calculate_calls_total"] = sum(
                    r.get("search_calculate_calls", 0) + r.get("fresh_calculate_calls", 0)
                    for r in record["runs"])
                record["elapsed_seconds"] = time.monotonic() - started
                dump(out / "provenance.json", record)
            if row.get("failure_class", "").startswith("hard_calculator"):
                record["status"] = "aborted_hard_calculator_failure"
                record["abort_run_id"] = folder.name
                dump(out / "provenance.json", record)
                return record
            if time.monotonic() - started >= CAMPAIGN_WALL_SECONDS:
                break
        del search_calc
    record["status"] = "all_6_arms_recorded" if len(record["runs"]) == 6 else "campaign_incomplete"
    record["actual_calculate_calls_total"] = sum(r.get("search_calculate_calls", 0) + r.get("fresh_calculate_calls", 0) for r in record["runs"])
    record["elapsed_seconds"] = time.monotonic() - started
    dump(out / "provenance.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new output directory")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("PAMSSW_RESEARCH_ROOT", DEFAULT_DATA_ROOT)),
                        help="checkout holding frozen inputs, model manifests, ledger, and protocol")
    parser.add_argument("--execute", action="store_true", help="execute bounded GPU comparison")
    args = parser.parse_args()
    configure_data_root(args.data_root)
    out = args.out.expanduser().resolve()
    if not args.execute:
        prepared = prepare()
        print(json.dumps({"status": "prepared_not_executed", "cases": list(CASES),
                          "arms": ARMS, "max_search_EF": len(CASES) * len(ARMS) * SEARCH_CAP,
                          "max_fresh_EF": len(CASES) * len(ARMS) * FRESH_CAP,
                          "fresh_wall_seconds_per_arm": FRESH_WALL_SECONDS,
                          "preflight": prepared,
                          "out": str(out)}, indent=2))
        return
    result = execute(out)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                      "search_requests": result["search_requests_total"],
                      "fresh_requests": result["fresh_requests_total"], "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
