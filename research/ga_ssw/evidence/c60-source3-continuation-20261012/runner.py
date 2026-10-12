#!/usr/bin/env python3
"""Continue one frozen C60 source-3 SSW/LS arm from its saved checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

PROCESS_ENTRY = time.monotonic()

import numpy as np

from preflight import ARMS, PARENT_PREPARED, inspect_parent, sha


TASK_DIR = Path(__file__).resolve().parent
TOTAL_OUTER = 100
CUMULATIVE_SEARCH_CAP = 60000
CHILD_FRESH_CAP = 3
ALLOCATION_SECONDS = 1800
SEARCH_SECONDS = 1680
PARENT_SOURCE_HEAD = "4fec3e3f7d09009f0808dfc43cc1c2c2e78900d6"


class ContinuationStop(RuntimeError):
    """A non-paid boundary signal for request or wall limits."""


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def json_safe(value):
    from ase import Atoms

    if isinstance(value, Atoms):
        return {
            "numbers": value.numbers.tolist(),
            "positions_A": value.positions.tolist(),
            "cell_A": value.cell.array.tolist(),
            "pbc": value.pbc.tolist(),
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def sha_imports(source_root: Path):
    rows = {}
    for name, module in sorted(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if (name == "pamssw" or name.startswith("pamssw.")) and path:
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(source_root.resolve()):
                raise RuntimeError(f"imported core outside frozen parent source: {name}={resolved}")
            rows[name] = {"path": str(resolved), "sha256": sha(resolved)}
    return rows


def _record_landing(record, landing_dir: Path):
    if record.landing is None:
        return None
    from ase.io import write

    atoms = record.landing.atoms.copy()
    atoms.calc = None
    atoms.info.update({
        "outer_index": int(record.index),
        "outer_status": str(record.status),
        "mc_accepted": bool(record.accepted),
        "landing_energy_eV": float(record.landing.energy),
        "landing_fmax_eV_A": float(record.landing.max_force),
        "landing_converged": bool(record.landing.converged),
    })
    path = landing_dir / f"outer-{record.index:03d}-landing.extxyz"
    write(path, atoms, format="extxyz")
    return path.name


def _event_row(record, checkpoint, parent_search: int, landing_file):
    return {
        "outer_index": record.index,
        "cumulative_outer_count": record.index + 1,
        "status": record.status,
        "accepted": bool(record.accepted),
        "error": record.error,
        "paid_search_requests_this_outer": record.evaluation_requests,
        # The checkpoint counter already includes the inherited parent segment.
        "cumulative_paid_search_requests": checkpoint.evaluation_requests,
        "checkpoint_lineage_requests": checkpoint.evaluation_requests,
        "landing_file": landing_file,
        "landing": None if record.landing is None else {
            "energy_eV": record.landing.energy,
            "fmax_eV_A": record.landing.max_force,
            "converged": record.landing.converged,
        },
        "ls_response_eV_per_atom": json_safe(record.energy_response),
        "ls_update": json_safe(record.ls_update),
        "ls_preparation": json_safe(record.ls_preparation),
        "mc_telemetry": json_safe(record.mc_telemetry),
        "starter_selection": json_safe(record.starter_selection),
        "climb_events": json_safe(record.climb),
    }


def _load_frozen_runtime(parent_prepared: Path):
    source_root = parent_prepared / "source"
    sys.path.insert(0, str(source_root))
    from pamssw.standalone import (
        LSPrequenchSettings,
        LSSettings,
        NativeMCSettings,
        SSWConfig,
        load_ssw_checkpoint,
        run_ssw,
        save_ssw_checkpoint,
    )
    from pamssw.standalone.surface import ASESurface
    if not Path(sys.modules["pamssw"].__file__).resolve().is_relative_to(source_root.resolve()):
        raise RuntimeError("continuation imported a non-parent pamssw package")

    def calculator(plan):
        if sha(Path(plan["model"])) != plan["model_sha256"]:
            raise ValueError("model file changed after parent qualification")
        import torch
        from mace.calculators import MACECalculator
        torch.set_num_threads(plan["runtime"]["torch_num_threads"])
        torch.manual_seed(plan["runtime"]["torch_manual_seed"])
        torch.use_deterministic_algorithms(plan["runtime"]["torch_deterministic_algorithms"])
        torch.backends.cuda.matmul.allow_tf32 = plan["runtime"]["tf32"]
        torch.backends.cudnn.allow_tf32 = plan["runtime"]["tf32"]
        return MACECalculator(
            model_paths=plan["model"], head=plan["head"], device=plan["device"],
            default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False,
        )

    def ls_settings(plan):
        spec = plan.get("paper_ls")
        if spec is None:
            return None
        spec = dict(spec)
        spec.pop("parameter_source", None)
        for key in ("bond_energies", "bond_lengths"):
            spec[key] = {tuple(map(int, key_text.split(","))): value
                         for key_text, value in spec[key].items()}
        if spec.get("prequench") is not None:
            spec["prequench"] = LSPrequenchSettings(**spec["prequench"])
        return LSSettings(**spec)

    return {
        "ASESurface": ASESurface,
        "NativeMCSettings": NativeMCSettings,
        "SSWConfig": SSWConfig,
        "calculator": calculator,
        "load_ssw_checkpoint": load_ssw_checkpoint,
        "ls_settings": ls_settings,
        "run_ssw": run_ssw,
        "save_ssw_checkpoint": save_ssw_checkpoint,
        "source_root": source_root,
    }


def _graph_reference(plan, plan_path: Path, validator_path: Path):
    from ase.io import read
    import importlib.util
    import networkx as nx

    spec = importlib.util.spec_from_file_location("continuation_c60_validator", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference_path = (plan_path.parent / plan["reference_input"]).resolve()
    reference = read(reference_path)
    ref_graphs = {}
    for cutoff in (1.64, 1.70, 1.80):
        row = module.graph_row(reference.numbers, reference.positions, cutoff)
        graph = nx.Graph()
        graph.add_nodes_from(range(len(reference)))
        distances = reference.get_all_distances(mic=False)
        graph.add_edges_from(zip(*np.where(np.triu((distances < cutoff) &
                                                    (distances > 0), 1))))
        ref_graphs[cutoff] = graph
        if not row["graph_cage_candidate"]:
            raise ValueError("frozen Ih reference is not a C60 cage at the required cutoff")
    return module, reference, ref_graphs, nx


def _same_positions(left, right):
    return np.array_equal(left.positions, right.positions)


def _cumulative_readout(checkpoint, plan, validator, ref_graphs):
    import networkx as nx
    cumulative = checkpoint.initial.evaluation_requests
    observed = [(None, cumulative, checkpoint.initial, None)]
    ls_cost = 0
    for record in checkpoint.records:
        cumulative += record.evaluation_requests
        if (record.landing is not None and record.landing.converged
                and record.status != "ls_prequench_failed"):
            observed.append((record.index, cumulative, record.landing, record.accepted))
        if record.ls_preparation is not None:
            ls_cost += record.ls_preparation.get("evaluation_requests", 0)
    if cumulative != checkpoint.evaluation_requests:
        raise ValueError("full checkpoint records do not reproduce cumulative paid cost")
    rows = []
    connected_graph_classes = []
    for index, cost, minimum, accepted in observed:
        graphs = {str(c): validator.graph_row(minimum.atoms.numbers,
                                             minimum.atoms.positions, c, ref_graphs[c])
                  for c in (1.64, 1.70, 1.80)}
        force_ok = bool(minimum.converged and minimum.max_force <= plan["ssw_config"]["fmax"])
        connected = all(row["components"] == 1 for row in graphs.values())
        ih = all(row["graph_cage_candidate"] and row["ih_graph_match"]
                 for row in graphs.values())
        energy_ok = minimum.energy <= plan["reference_energy_eV"] + 0.01
        if force_ok and connected:
            graph = nx.Graph()
            graph.add_nodes_from(range(len(minimum.atoms)))
            distances = minimum.atoms.get_all_distances(mic=False)
            graph.add_edges_from(zip(*np.where(np.triu((distances < 1.70) &
                                                        (distances > 0), 1))))
            if not any(nx.is_isomorphic(graph, prior) for prior in connected_graph_classes):
                connected_graph_classes.append(graph)
        rows.append({"outer_index": index, "cumulative_paid_search_requests": cost,
                     "energy_eV": float(minimum.energy), "fmax_eV_A": float(minimum.max_force),
                     "force_qualified": force_ok, "connected_all_cutoffs": connected,
                     "ih_all_cutoffs": ih, "energy_target": energy_ok,
                     "target_candidate": bool(force_ok and ih and energy_ok),
                     "accepted": accepted, "graphs": graphs})
    prefixes = []
    for prefix in (5000, 10000, 20000, 40000, 60000):
        within = [row for row in rows if row["cumulative_paid_search_requests"] <= prefix]
        connected = [row for row in within if row["force_qualified"] and row["connected_all_cutoffs"]]
        targets = [row for row in within if row["target_candidate"]]
        prefixes.append({"requested_prefix": prefix,
                         "actual_lineage_cost": min(prefix, checkpoint.evaluation_requests),
                         "prefix_reached": checkpoint.evaluation_requests >= prefix,
                         "observation_count": len(within),
                         "best_connected_energy_eV": min(
                             (row["energy_eV"] for row in connected), default=None),
                         "first_target_cost": targets[0]["cumulative_paid_search_requests"]
                         if targets else None})
    return {"cumulative_paid_search_requests": checkpoint.evaluation_requests,
            "cumulative_outer_count": checkpoint.next_index, "observations": rows,
            "common_prefixes": prefixes, "ls_prequench_requests": ls_cost,
            "connected_graph_class_count_at_1.70A": len(connected_graph_classes),
            "outer_status_counts": {str(status): sum(r.status == status
                                                        for r in checkpoint.records)
                                    for status in sorted({r.status for r in checkpoint.records})},
            "mc_accepted_outer_count": sum(bool(r.accepted) for r in checkpoint.records),
            "mc_rejected_outer_count": sum(not bool(r.accepted) for r in checkpoint.records),
            "target_rule": "force gate + Ih graph match at 1.64/1.70/1.80 A + reference energy +0.01 eV",
            "physical_cage_review": "manual review remains required for qualifying candidates"}


def _write_paired_readout(output_root: Path):
    arms = {}
    for seed, arm in ARMS:
        folder = output_root / f"seed-{seed}-{arm}"
        summary_path, readout_path = folder / "summary.json", folder / "cumulative-readout.json"
        if not summary_path.is_file() or not readout_path.is_file():
            return False
        arms[(seed, arm)] = (json.loads(summary_path.read_text()),
                             json.loads(readout_path.read_text()))
    comparisons = []
    for seed in sorted({seed for seed, _ in ARMS}):
        ssw, ssw_readout = arms[(seed, "ssw")]
        ls, ls_readout = arms[(seed, "paper_ls")]
        common_cost = min(ssw_readout["cumulative_paid_search_requests"],
                          ls_readout["cumulative_paid_search_requests"])
        for prefix in (5000, 10000, 20000, 40000, 60000):
            horizon = min(prefix, common_cost)
            paired = {}
            for arm_name, readout in (("ssw", ssw_readout), ("paper_ls", ls_readout)):
                rows = [row for row in readout["observations"]
                        if row["cumulative_paid_search_requests"] <= horizon]
                connected = [row for row in rows if row["force_qualified"]
                             and row["connected_all_cutoffs"]]
                targets = [row for row in rows if row["target_candidate"]]
                paired[arm_name] = {
                    "qualified_observations": sum(row["force_qualified"] for row in rows),
                    "best_connected_energy_eV": min(
                        (row["energy_eV"] for row in connected), default=None),
                    "first_strict_Ih_target_cost": targets[0]["cumulative_paid_search_requests"]
                    if targets else None,
                    "fresh_checks": arms[(seed, arm_name)][0]["fresh_checks"],
                    "cumulative_outer_count": arms[(seed, arm_name)][0]["cumulative_outer_count"],
                    "run_status": arms[(seed, arm_name)][0]["status"],
                }
            comparisons.append({"seed": seed, "requested_prefix": prefix,
                                "common_paid_search_horizon": horizon,
                                "prefix_reached_by_both": common_cost >= prefix,
                                "arms": paired})
    atomic_json(output_root / "paired-cumulative-readout.json", {
        "status": "paired_available", "comparisons": comparisons,
        "scope": "paired cumulative parent+child fixed-horizon readout; not a barrier or PBE claim"})
    return True


def _fresh_qualify(output: Path, plan: dict, parent_folder: Path, checkpoint,
                   validator, ref_graphs, calc_factory, ase_surface,
                   started: float, hard_deadline: float, fresh_ledger):
    from ase.io import read, write

    old_positions = []
    for name in ("initial-fresh.traj", "best-fresh.traj"):
        path = parent_folder / name
        if path.exists():
            old_positions.append(read(path).positions)

    minima = list(checkpoint.minima)
    if not minima:
        minima = [checkpoint.best]
    joint_candidates = []
    for index, item in enumerate(minima):
        cage = validator.graph_row(item.atoms.numbers, item.atoms.positions, 1.8)
        if (item.converged and cage["graph_cage_candidate"]
                and item.energy <= plan["reference_energy_eV"] + 0.01):
            joint_candidates.append((index, item))
    candidates = []

    def add(label, minimum, index):
        if minimum is None or any(_same_positions(minimum.atoms, previous.atoms)
                                  for _, previous, _ in candidates):
            return
        if any(np.array_equal(minimum.atoms.positions, positions) for positions in old_positions):
            return
        candidates.append((label, minimum, index))

    if joint_candidates:
        target_index, target_minimum = joint_candidates[0]
        add("first_joint_candidate", target_minimum, target_index)
    converged_new = [(i, item) for i, item in enumerate(minima) if item.converged and
                     not any(np.array_equal(item.atoms.positions, positions) for positions in old_positions)]
    if converged_new:
        idx, minimum = min(converged_new, key=lambda pair: pair[1].energy)
        add("best_new_minimum", minimum, idx)
    cages = []
    for index, item in enumerate(minima):
        row = validator.graph_row(item.atoms.numbers, item.atoms.positions, 1.8)
        if item.converged and row["graph_cage_candidate"]:
            cages.append((index, item))
    if cages:
        idx, minimum = min(cages, key=lambda pair: pair[1].energy)
        add("best_new_cage", minimum, idx)

    checks = []
    if not candidates:
        return checks, 0
    calc = calc_factory(plan)
    actual_calculations = 0
    original_calculate = calc.calculate

    def counted_calculate(*args, **kwargs):
        nonlocal actual_calculations
        actual_calculations += 1
        return original_calculate(*args, **kwargs)

    calc.calculate = counted_calculate
    fresh_surface = ase_surface(calc)
    for label, minimum, search_index in candidates[:CHILD_FRESH_CAP]:
        if time.monotonic() >= hard_deadline:
            checks.append({"label": label, "search_index": search_index,
                           "status": "not_started_allocation_margin"})
            continue
        row = {"label": label, "search_index": search_index, "status": "failed"}
        fresh_id = len(fresh_ledger) + 1
        fresh_ledger.append({"id": fresh_id, "label": label, "search_index": search_index,
                             "status": "charged_unfinished"})
        atomic_json(output / "fresh-requests.json", fresh_ledger)
        try:
            calc.reset()
            energy, forces = fresh_surface.evaluate(minimum.atoms)
            fmax = float(np.linalg.norm(forces, axis=1).max())
            graphs = [validator.graph_row(minimum.atoms.numbers, minimum.atoms.positions, cutoff,
                                         ref_graphs[cutoff]) for cutoff in (1.64, 1.70, 1.80)]
            graph_ok = all(item["graph_cage_candidate"] and item["ih_graph_match"] for item in graphs)
            geometry_ok = (np.array_equal(minimum.atoms.numbers, checkpoint.initial.atoms.numbers)
                           and np.array_equal(minimum.atoms.cell.array, checkpoint.initial.atoms.cell.array)
                           and np.array_equal(minimum.atoms.pbc, checkpoint.initial.atoms.pbc))
            numerical = bool(minimum.converged and fmax <= plan["ssw_config"]["fmax"]
                             and abs(energy - minimum.energy) <= 1e-6 and geometry_ok)
            cage18 = bool(graphs[2]["graph_cage_candidate"])
            energy_target = energy <= plan["reference_energy_eV"] + 0.01
            parent_joint_target = bool(numerical and cage18 and energy_target)
            strict_ih_target = bool(numerical and graph_ok and energy_target)
            centered = minimum.atoms.positions - minimum.atoms.positions.mean(axis=0)
            extents = np.linalg.svd(centered, compute_uv=False) / np.sqrt(len(centered))
            distances = np.linalg.norm(centered[:, None, :] - centered[None, :, :], axis=2)
            np.fill_diagonal(distances, np.inf)
            row.update(status="completed", energy_eV=float(energy), fmax_eV_A=fmax,
                       energy_repeat_error_eV=float(energy - minimum.energy),
                       numerical_qualified=numerical, graph_match_all_cutoffs=graph_ok,
                       parent_joint_target=parent_joint_target,
                       strict_ih_target_candidate=strict_ih_target,
                       graphs=graphs,
                       principal_rms_extents_A=extents.tolist(),
                       minimum_distance_A=float(distances.min()),
                       physical_cage_review="required" if cage18
                       else "not_a_cage")
            atoms = minimum.atoms.copy()
            atoms.calc = None
            atoms.info.update({"continuation_fresh_label": label,
                               "numerical_qualified": numerical,
                               "parent_joint_target": parent_joint_target,
                               "strict_ih_target_candidate": strict_ih_target})
            write(output / f"fresh-{label}.extxyz", atoms, format="extxyz")
        except Exception as error:
            row["error"] = repr(error)
        fresh_ledger[-1].update(status=row["status"], error=row.get("error"),
                                energy_eV=row.get("energy_eV"),
                                fmax_eV_A=row.get("fmax_eV_A"))
        atomic_json(output / "fresh-requests.json", fresh_ledger)
        checks.append(row)
    del fresh_surface, calc
    return checks, actual_calculations


def run_arm(parent_prepared: Path, output_root: Path, arm_index: int,
            qualification_one_outer: bool = False):
    from ase.io import read

    if arm_index < 0 or arm_index >= len(ARMS):
        raise ValueError("arm index must be 0..3")
    seed, arm = ARMS[arm_index]
    parent_folder = parent_prepared / f"seed-{seed}" / arm
    parent_plan_path = parent_folder / "plan.json"
    plan = json.loads(parent_plan_path.read_text())
    parent_state = inspect_parent(parent_prepared)
    state = next(row for row in parent_state["arms"] if row["seed"] == seed and row["arm"] == arm)
    output = output_root / f"seed-{seed}-{arm}"
    if output.exists():
        raise FileExistsError(f"continuation output exists; no retry/overwrite: {output}")
    output.mkdir(parents=True)
    started = PROCESS_ENTRY
    target_outer = min(TOTAL_OUTER, state["next_index"] + 1) if qualification_one_outer else TOTAL_OUTER
    search_cap = (state["parent_paid_search_requests"] + 2000
                  if qualification_one_outer else CUMULATIVE_SEARCH_CAP)
    search_seconds = 360 if qualification_one_outer else SEARCH_SECONDS
    allocation_seconds = 480 if qualification_one_outer else ALLOCATION_SECONDS
    hard_deadline = started + allocation_seconds - 15
    search_deadline = started + search_seconds
    remaining = target_outer - state["next_index"]
    remaining_search = search_cap - state["parent_paid_search_requests"]
    if remaining <= 0 or remaining_search <= 0:
        raise ValueError("parent state has no remaining bounded continuation")

    # Create a new child manifest. The terminal parent budget and plan are never edited.
    parent_copy = {
        "parent_run": str(parent_folder),
        "parent_prepared": str(parent_prepared),
        "parent_plan_sha256": state["parent_plan_sha256"],
        "parent_budget_sha256": state["parent_budget_sha256"],
        "parent_request_ledger_sha256": state["parent_request_ledger_sha256"],
        "parent_checkpoint_sha256": state["parent_checkpoint_sha256"],
        "parent_paid_search_requests": state["parent_paid_search_requests"],
        "parent_fresh_requests": state["parent_fresh_requests"],
        "parent_reserved_seconds": state["parent_reserved_seconds"],
        "parent_checkpoint_next_index": state["next_index"],
        "additional_attempts": remaining,
        "cumulative_outer_limit": target_outer,
        "cumulative_search_cap": search_cap,
        "additional_search_allowance": remaining_search,
        "additional_fresh_cap": CHILD_FRESH_CAP,
        "allocation_seconds": allocation_seconds,
        "search_deadline_seconds_from_child_start": search_seconds,
        "qualification_only": qualification_one_outer,
        "child_runner": str(Path(__file__).resolve()),
        "child_runner_sha256": sha(Path(__file__).resolve()),
        "child_preflight_sha256": sha(TASK_DIR / "preflight.py"),
        "parent_frozen_source_head": PARENT_SOURCE_HEAD,
        "parent_source_manifest_sha256": parent_state["parent_source_manifest_sha256"],
        "parent_search_plan_manifest_sha256": parent_state["parent_search_plan_manifest_sha256"],
        "model_sha256": plan["model_sha256"],
        "seed": seed,
        "arm": arm,
        "input_sha256": plan["input_sha256"],
        "inherited_cost_not_replayed": True,
    }
    atomic_json(output / "continuation-provenance.json", parent_copy)
    shutil.copy2(parent_plan_path, output / "parent-plan.json")
    shutil.copy2(parent_folder / plan["input"], output / "input.extxyz")
    shutil.copy2(parent_folder / "validator.py", output / "validator.py")
    shutil.copy2(parent_prepared / "qualification" / "reference-source.extxyz",
                 output / "reference-source.extxyz")
    shutil.copy2(parent_folder / "checkpoint.pkl", output / "checkpoint-start.pkl")
    checkpoint_copy = output / "checkpoint.pkl"
    shutil.copy2(output / "checkpoint-start.pkl", checkpoint_copy)

    runtime = _load_frozen_runtime(parent_prepared)
    source_root = runtime["source_root"]
    imports_before = sha_imports(source_root)
    from pamssw.standalone.paper_reference import _policy_signature

    checkpoint = runtime["load_ssw_checkpoint"](checkpoint_copy)
    if sha(checkpoint_copy) != state["parent_checkpoint_sha256"]:
        raise ValueError("checkpoint copy differs from its parent")
    if checkpoint.next_index != state["next_index"]:
        raise ValueError("checkpoint outer index changed after preflight")
    if checkpoint.evaluation_requests != state["parent_paid_search_requests"]:
        raise ValueError("checkpoint lineage and inherited paid cost differ")
    config = runtime["SSWConfig"](**plan["ssw_config"])
    ls = runtime["ls_settings"](plan)
    mc = runtime["NativeMCSettings"](plan["native_mc"]["energy_tol_eV"],
                                      plan["native_mc"]["maxtrap"])
    if checkpoint.config != config or _policy_signature(checkpoint.ls) != _policy_signature(ls):
        raise ValueError("child settings do not match saved search state")
    if checkpoint.mc_settings != mc or checkpoint.rng_state.get("bit_generator") != "PCG64":
        raise ValueError("child MC or RNG type differs from saved search state")

    initial = read(output / "input.extxyz")
    calculator = runtime["calculator"](plan)
    actual_search_calculations = 0
    original_calculate = calculator.calculate

    def counted_calculate(*args, **kwargs):
        nonlocal actual_search_calculations
        actual_search_calculations += 1
        return original_calculate(*args, **kwargs)

    calculator.calculate = counted_calculate
    base_surface = runtime["ASESurface"](calculator)
    request_ledger = (output / "requests.jsonl").open("x", buffering=1)
    denials_ledger = (output / "denials.jsonl").open("x", buffering=1)
    events_ledger = (output / "outer-events.jsonl").open("x", buffering=1)
    landing_dir = output / "landings"
    landing_dir.mkdir()
    local_search_requests = 0
    latest_checkpoint = checkpoint

    class PaidSurface:
        requests = 0

        def evaluate(self, atoms):
            nonlocal local_search_requests
            reason = None
            if state["parent_paid_search_requests"] + local_search_requests >= search_cap:
                reason = "cumulative_search_cap"
            elif time.monotonic() >= search_deadline:
                reason = "search_wall_deadline"
            if reason:
                denials_ledger.write(json.dumps({
                    "kind": "search_denial", "reason": reason,
                    "parent_paid_search_requests": state["parent_paid_search_requests"],
                    "incremental_paid_search_requests": local_search_requests,
                    "cumulative_outer_index": latest_checkpoint.next_index,
                }) + "\n")
                raise ContinuationStop(reason)
            local_search_requests += 1
            self.requests += 1
            started_call = time.monotonic()
            row = {
                "incremental_id": local_search_requests,
                "parent_paid_search_requests": state["parent_paid_search_requests"],
                "cumulative_paid_id": state["parent_paid_search_requests"] + local_search_requests,
                "outer_index_at_call": latest_checkpoint.next_index,
            }
            try:
                energy, force = base_surface.evaluate(atoms)
                row.update(energy_eV=float(energy),
                           fmax_eV_A=float(np.linalg.norm(force, axis=1).max()))
                return energy, force
            except Exception as error:
                row["error"] = repr(error)
                raise
            finally:
                row["elapsed_seconds"] = time.monotonic() - started_call
                request_ledger.write(json.dumps(row, allow_nan=False) + "\n")

    surface = PaidSurface()

    def on_boundary(snapshot):
        nonlocal latest_checkpoint
        latest_checkpoint = snapshot
        runtime["save_ssw_checkpoint"](checkpoint_copy, snapshot)
        record = snapshot.records[-1]
        landing_file = _record_landing(record, landing_dir)
        events_ledger.write(json.dumps(
            _event_row(record, snapshot, state["parent_paid_search_requests"], landing_file),
            allow_nan=False,
        ) + "\n")
        return (state["parent_paid_search_requests"] + local_search_requests >= search_cap
                or time.monotonic() >= search_deadline)

    run_status = "not_started"
    run_error = None
    result = None
    try:
        result = runtime["run_ssw"](
            initial, surface, steps=remaining, config=config,
            rng=np.random.default_rng(plan["seed"]), checkpoint=checkpoint,
            checkpoint_callback=on_boundary, ls=ls, mc=mc,
        )
        run_status = result.status
        if result.checkpoint is not None:
            latest_checkpoint = result.checkpoint
            runtime["save_ssw_checkpoint"](checkpoint_copy, latest_checkpoint)
    except Exception as error:
        run_status = "failed"
        run_error = repr(error)
        (output / "exception.txt").write_text(traceback.format_exc())
    finally:
        request_ledger.close()
        denials_ledger.close()
        events_ledger.close()
        try:
            calculator.reset()
        except Exception:
            pass
        del base_surface, surface, calculator

    checkpoint = runtime["load_ssw_checkpoint"](checkpoint_copy)
    if checkpoint.evaluation_requests > state["parent_paid_search_requests"] + local_search_requests:
        raise RuntimeError("checkpoint lineage exceeds inherited plus charged child calls")
    uncommitted_paid_tail = (state["parent_paid_search_requests"] + local_search_requests
                             - checkpoint.evaluation_requests)
    if checkpoint.next_index > target_outer:
        raise RuntimeError("continuation exceeded the frozen 100-attempt horizon")

    # The frozen runner intentionally skips its callback for a terminal failed
    # outer record. Preserve that record from the returned safe checkpoint.
    event_path = output / "outer-events.jsonl"
    saved_indices = {json.loads(line)["outer_index"] for line in event_path.read_text().splitlines()
                     if line.strip()}
    for record in checkpoint.records[state["next_index"]:]:
        if record.index in saved_indices:
            continue
        landing_file = _record_landing(record, landing_dir)
        with event_path.open("a") as stream:
            stream.write(json.dumps(_event_row(record, checkpoint,
                                               state["parent_paid_search_requests"],
                                               landing_file), allow_nan=False) + "\n")
        saved_indices.add(record.index)
    if len(saved_indices) != checkpoint.next_index - state["next_index"]:
        raise RuntimeError("saved outer event indices do not cover the child checkpoint records")

    validator, reference, ref_graphs, _ = _graph_reference(plan, parent_plan_path,
                                                           output / "validator.py")
    search_finished = time.monotonic()
    search_elapsed = search_finished - started
    hard_deadline = started + allocation_seconds - 10
    search_complete = checkpoint.next_index == target_outer
    fresh_ledger = []
    if uncommitted_paid_tail:
        fresh_checks, fresh_calculations = [], 0
        atomic_json(output / "fresh-requests.json", fresh_ledger)
    else:
        fresh_checks, fresh_calculations = _fresh_qualify(
            output, plan, parent_folder, checkpoint, validator, ref_graphs,
            runtime["calculator"], runtime["ASESurface"], started, hard_deadline,
            fresh_ledger)
    imports_after = sha_imports(source_root)

    status = ("censored_uncheckpointed_paid_tail" if uncommitted_paid_tail else
              "qualification_one_outer" if qualification_one_outer and search_complete
              and run_status in ("completed", "paused") else
              "completed_100" if not qualification_one_outer and search_complete
              and run_status in ("completed", "paused") else
              "censored_search_cap" if state["parent_paid_search_requests"] + local_search_requests
              >= search_cap else
              "censored_wall_or_partial" if checkpoint.next_index < target_outer else
              "completed_with_terminal_status")
    summary = {
        "status": status,
        "algorithm_status": run_status,
        "error": run_error,
        "seed": seed,
        "arm": arm,
        "parent_next_index": state["next_index"],
        "additional_outer_attempts": checkpoint.next_index - state["next_index"],
        "cumulative_outer_count": checkpoint.next_index,
        "remaining_attempts": target_outer - checkpoint.next_index,
        "parent_paid_search_requests": state["parent_paid_search_requests"],
        "child_paid_search_requests": local_search_requests,
        "cumulative_paid_search_requests": state["parent_paid_search_requests"] + local_search_requests,
        "parent_fresh_requests": state["parent_fresh_requests"],
        "child_fresh_requests": len(fresh_ledger),
        "child_actual_search_calculate_calls": actual_search_calculations,
        "child_actual_fresh_calculate_calls": fresh_calculations,
        "checkpoint_lineage_requests": checkpoint.evaluation_requests,
        "uncommitted_paid_search_tail": uncommitted_paid_tail,
        "checkpoint_sha256": sha(checkpoint_copy),
        "parent_checkpoint_sha256": state["parent_checkpoint_sha256"],
        "parent_budget_not_modified": sha(parent_folder / "budget.json") == state["parent_budget_sha256"],
        "parent_plan_not_modified": sha(parent_plan_path) == state["parent_plan_sha256"],
        "parent_request_ledger_not_modified": sha(parent_folder / "requests.jsonl") == state["parent_request_ledger_sha256"],
        "search_elapsed_seconds": search_elapsed,
        "process_elapsed_seconds": time.monotonic() - started,
        "fresh_checks": fresh_checks,
        "source_imports_before": imports_before,
        "source_imports_after": imports_after,
        "parent_provenance": parent_copy,
        "landings_saved": len(list(landing_dir.glob("outer-*-landing.extxyz"))),
        "outer_events_saved": len((output / "outer-events.jsonl").read_text().splitlines()),
        "denials": [json.loads(line) for line in (output / "denials.jsonl").read_text().splitlines()],
        "reference_energy_eV": plan["reference_energy_eV"],
        "reference_sha256": plan["reference_sha256"],
        "scope": ("one-outer resume qualification only" if qualification_one_outer else
                  "paired continuation on the parent's fixed source, model, settings, seeds, and 100-attempt horizon"),
        "claim_ceiling": "does not establish a general LS advantage, a barrier explanation, or PBE reproduction",
    }
    if not all((summary["parent_budget_not_modified"], summary["parent_plan_not_modified"],
                summary["parent_request_ledger_not_modified"])):
        raise RuntimeError("historical parent artifacts changed during continuation")
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "cumulative-readout.json",
                _cumulative_readout(checkpoint, plan, validator, ref_graphs))
    _write_paired_readout(output_root)
    runtime_record = {
        "python": sys.version,
        "python_executable": sys.executable,
        "job_id": os.getenv("SLURM_JOB_ID"),
        "array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "child_runner": str(Path(__file__).resolve()),
        "child_runner_sha256": sha(Path(__file__).resolve()),
        "preflight_sha256": sha(TASK_DIR / "preflight.py"),
        "frozen_source_root": str(source_root.resolve()),
        "frozen_source_head": PARENT_SOURCE_HEAD,
        "imports_before": imports_before,
        "imports_after": imports_after,
        "status": status,
    }
    atomic_json(output / "runtime.json", runtime_record)
    print(json.dumps({k: summary[k] for k in (
        "status", "seed", "arm", "cumulative_outer_count", "parent_paid_search_requests",
        "child_paid_search_requests", "cumulative_paid_search_requests",
        "child_actual_search_calculate_calls", "process_elapsed_seconds")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-prepared", type=Path, default=PARENT_PREPARED)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm-index", type=int, required=True, choices=range(4))
    parser.add_argument("--qualification-one-outer", action="store_true",
                        help="run one real outer step as a separate, non-study resume qualification")
    parser.add_argument("--execute", action="store_true",
                        help="required to issue continuation E/F calls")
    args = parser.parse_args()
    if not args.execute:
        parser.error("no calculations are run without the explicit --execute flag")
    parent_prepared = args.parent_prepared.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root == parent_prepared or output_root.is_relative_to(parent_prepared):
        raise ValueError("child output must be outside the immutable parent run")
    output_root.mkdir(parents=True, exist_ok=True)
    run_arm(parent_prepared, output_root, args.arm_index,
            qualification_one_outer=args.qualification_one_outer)


if __name__ == "__main__":
    main()
