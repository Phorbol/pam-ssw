#!/usr/bin/env python3
"""One-step paired SSW/LS escape ablation on qualified C4H6 isomers.

Development evidence only: four paired seeds per input and arm. MC selection,
candidate generation, physical force checks, and basin diagnostics remain
separate observations; this script does not establish search effectiveness.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = ROOT / "research/ga_ssw/evidence/c4h6-mh1-lifecycle-20260924"
LEDGER_PATH = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
QUALIFIER_PATH = ROOT / "research/ga_ssw/qualify_ls_torsion_channel.py"
RING_TOOLS_PATH = ROOT / "research/ga_ssw/probe_ls_ring_channel.py"
GRAPH_TOOLS_PATH = ROOT / "research/ga_ssw/analyze_c4h6_ls_reaction_coverage.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
RING_REFERENCE = ROOT / "research/ga_ssw/evidence/ls-theory-20261010/ring-channel-v1/run-1728329/qualification/minus-0.050.extxyz"
SEEDS = (701, 702, 703, 704)
ARMS = (("none", None), ("ls-0.03", 0.03), ("ls-0.48", 0.48))
SEARCH_CAP_PER_ARM = 4000
WALL_SECONDS_PER_ARM = 100
FRESH_CAP_PER_ARM = 1
CAMPAIGN_WALL_SECONDS = 2700
FORCE_TOL = 0.03


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_geometry(base: Path, filename: str) -> Path:
    for candidate in (base / filename, base / "qualification" / filename):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not find {filename} in {base} or {base / 'qualification'}")


class ResetCountedSurface:
    """Preserve the existing C4H6 runner's cold calculator cache per request."""
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evaluate(self, atoms):
        self.base.calculator.reset()
        return self.base.evaluate(atoms)


def reference_record(label, geometry_path, result_path, phase_name, graph_fn):
    atoms = read(geometry_path, index=0)
    expected = np.array([6] * 4 + [1] * 6)
    if (not np.array_equal(atoms.numbers, expected) or atoms.constraints or atoms.pbc.any()
            or not np.isfinite(atoms.positions).all()):
        raise ValueError(f"invalid fixed-order isolated C4H6 reference: {geometry_path}")
    record = json.loads(result_path.read_text()) if result_path and result_path.is_file() else {}
    phase = record.get("phases", {}).get(phase_name, {})
    energy = phase.get("minimum_energy_eV", phase.get("energy_eV"))
    return {"label": label, "path": str(geometry_path), "sha256": sha256(geometry_path),
            "result_path": str(result_path) if result_path else None,
            "result_sha256": sha256(result_path) if result_path and result_path.is_file() else None,
            "phase": phase_name, "energy_eV": None if energy is None else float(energy),
            "atoms": atoms, "graph": graph_fn(atoms) if graph_fn is not None else None}


def initial_softening_record(atoms, ls_settings, softening_type, output_path):
    if ls_settings is None:
        return None
    frozen = softening_type.from_atoms(
        atoms, bond_energies=ls_settings.bond_energies,
        bond_lengths=ls_settings.bond_lengths,
        initial_fraction=ls_settings.initial_fraction, xi=ls_settings.xi,
        energy_filter=ls_settings.energy_filter)
    saved = {"origin": str(output_path), "numbers": list(frozen.numbers),
             "cell_A": [list(row) for row in frozen.cell], "pbc": list(frozen.pbc),
             "pairs": [list(pair) for pair in frozen.pairs],
             "reference_distances_A": list(frozen.reference_distances),
             "strengths_eV": list(frozen.strengths), "xi": frozen.xi,
             "initial_fraction": ls_settings.initial_fraction,
             "bond_energies_eV": {str(k): v for k, v in ls_settings.bond_energies.items()},
             "bond_lengths_A": {str(k): v for k, v in ls_settings.bond_lengths.items()},
             "energy_filter": frozen.energy_filter}
    output_path.write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n")
    return saved


def candidate_diagnostics(atoms, references, graph_fn, component_formulas, ring_tools):
    import networkx as nx
    candidate_graph = graph_fn(atoms)
    node_match = nx.algorithms.isomorphism.categorical_node_match("number", None)
    component_ids = nx.connected_components(candidate_graph)
    components = [sorted(map(int, component)) for component in component_ids]
    formulas = [component_formulas(atoms, candidate_graph)]
    comparisons = {}
    for label, ref in references.items():
        _, _, _, rmsd = ring_tools.proper_kabsch(atoms.positions, ref["atoms"].positions)
        isomorphic = nx.is_isomorphic(candidate_graph, ref["graph"], node_match=node_match)
        comparisons[label] = {"reference_path": ref["path"],
                              "reference_energy_eV": ref["energy_eV"],
                              "same_index_proper_kabsch_rmsd_A": rmsd,
                              "graph_isomorphic_with_element_labels": bool(isomorphic)}
    return {"component_count": len(components), "components_atom_indices": components,
            "component_formulas": formulas[0],
            "degree_sequence": sorted(dict(candidate_graph.degree()).values()),
            "reference_comparisons": comparisons,
            "identity_limit": "graph isomorphism and same-index RMSD are diagnostics; no basin identity threshold"}


def main_run(torsion_references: Path, output: Path):
    torsion_references = torsion_references.resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    if not torsion_references.is_dir():
        raise NotADirectoryError(torsion_references)
    if not RING_REFERENCE.is_file():
        raise FileNotFoundError(RING_REFERENCE)

    plus_path = resolve_geometry(torsion_references, "plus-0.050.extxyz")
    minus_path = resolve_geometry(torsion_references, "minus-0.050.extxyz")
    torsion_result = next((p for p in (torsion_references / "result.json",
                                       torsion_references / "qualification" / "result.json") if p.is_file()), None)
    torsion_gauche = reference_record("gauche", plus_path, torsion_result, "plus-0.050", None)
    torsion_trans = reference_record("trans", minus_path, torsion_result, "minus-0.050", None)
    ring_result_path = RING_REFERENCE.parent / "result.json"

    import scipy
    import ase
    import torch
    import mace
    from mace.calculators import MACECalculator  # import-only preflight; no instance here
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone import LSSettings, SSWConfig, run_ssw
    from pamssw.standalone import paper_reference
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    from pamssw.standalone.softening import FrozenBondSoftening

    utilities = import_file("ls_single_escape_ledger", LEDGER_PATH)
    qualifier = import_file("ls_single_escape_qualifier", QUALIFIER_PATH)
    ring_tools = import_file("ls_single_escape_ring_tools", RING_TOOLS_PATH)
    graph_tools = import_file("ls_single_escape_graph_tools", GRAPH_TOOLS_PATH)

    model_hash = sha256(MODEL)
    if model_hash != MODEL_SHA256 or model_hash != qualifier.MODEL_SHA256:
        raise ValueError(f"MH-1 model hash mismatch: {model_hash}")
    prior_path = LIFECYCLE / "effective_config.json"
    prior = json.loads(prior_path.read_text())
    config = SSWConfig(**prior["config"])
    rotation = RecoveredRotationSettings(**prior["rotation_settings"])
    prequench = LSPrequenchSettings(fmax=0.1, steps=50,
                                    exit_policy="force_or_step_limit")
    expected = (config.width, config.max_gaussians, config.temperature_K, config.fmax,
                config.relax_steps, config.fd_step, config.bias_fmax, config.lbfgs_memory,
                config.quench_optimizer)
    if expected != (0.1, 25, 150.0, FORCE_TOL, 400, 0.001, 0.1, 500, "safe-lbfgs-total"):
        raise ValueError(f"lifecycle effective SSW settings differ from protocol: {expected}")
    if asdict(rotation) != prior["rotation_settings"]:
        raise ValueError("lifecycle recovered-rotation settings did not round-trip")

    cyclo = reference_record("cyclobutene", RING_REFERENCE, ring_result_path,
                             "minus-0.050", graph_tools.graph)
    torsion_gauche["graph"] = graph_tools.graph(torsion_gauche["atoms"])
    torsion_trans["graph"] = graph_tools.graph(torsion_trans["atoms"])
    references = {ref["label"]: ref for ref in (torsion_gauche, torsion_trans, cyclo)}
    ref_public = [{key: val for key, val in ref.items() if key not in ("atoms", "graph")}
                  for ref in references.values()]

    campaign_started = time.monotonic()
    output.mkdir(parents=False, exist_ok=False)
    shutil.copy2(Path(__file__).resolve(), output / "probe_ls_single_escape.py")
    result = {"status": "started", "protocol": str(ROOT / "research/ga_ssw/evidence/ls-theory-20261010/escape-protocol.md"),
              "inputs": {"torsion_references_dir": str(torsion_references), "gauche": torsion_gauche["path"],
                         "trans": torsion_trans["path"], "cyclobutene": cyclo["path"]},
              "references": ref_public,
              "model": {"path": str(MODEL), "sha256": model_hash, "head": "omol",
                        "device": "cuda", "dtype": "float64", "torch": torch.__version__,
                        "mace": mace.__version__, "ase": ase.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__, "python": sys.version, "platform": platform.platform()},
              "source_hashes": {"runner": sha256(Path(__file__).resolve()),
                                "protocol": sha256(ROOT / "research/ga_ssw/evidence/ls-theory-20261010/escape-protocol.md"),
                                "lifecycle_config": sha256(prior_path), "ledger": sha256(LEDGER_PATH),
                                "qualifier": sha256(QUALIFIER_PATH), "ring_tools": sha256(RING_TOOLS_PATH),
                                "graph_tools": sha256(GRAPH_TOOLS_PATH),
                                "torsion_result": sha256(torsion_result) if torsion_result else None,
                                "ring_result": sha256(ring_result_path)},
              "limits": {"starts": ["gauche", "trans"], "seeds": SEEDS,
                         "arms": [name for name, _ in ARMS], "run_ssw_steps": 1,
                         "search_requests_per_arm": SEARCH_CAP_PER_ARM,
                         "wall_seconds_per_arm": WALL_SECONDS_PER_ARM,
                         "fresh_requests_per_arm": FRESH_CAP_PER_ARM,
                         "total_search_request_cap": 96000, "total_fresh_request_cap": 24,
                         "campaign_wall_seconds": CAMPAIGN_WALL_SECONDS},
              "configuration": {"ssw": asdict(config), "recovered_rotation": asdict(rotation),
                                "ls_common": {"bond_energies_eV": {str(k): v for k, v in HC_BOND_ENERGIES.items()},
                                              "bond_lengths_A": {str(k): v + 0.1 for k, v in HC_BOND_LENGTHS.items()},
                                              "target_per_atom_eV": 0.7, "xi": 0.2,
                                              "learning_rate": 1.8, "prequench": asdict(prequench)},
                                "ls_arms": {name: (None if fraction is None else
                                                   {"initial_fraction": fraction,
                                                    "bond_energies_eV": {str(k): v for k, v in HC_BOND_ENERGIES.items()},
                                                    "bond_lengths_A": {str(k): v + 0.1 for k, v in HC_BOND_LENGTHS.items()},
                                                    "target_per_atom_eV": 0.7, "xi": 0.2,
                                                    "learning_rate": 1.8,
                                                    "prequench": asdict(prequench)})
                                             for name, fraction in ARMS},
                                "mc": "existing run_ssw default; accepted decision is recorded separately from landing candidate"},
              "runs": [], "search_requests_total": 0, "fresh_requests_total": 0,
              "actual_calculate_calls_total": 0,
              "interpretation_limit": "single-escape paired development ablation only; four seeds do not establish success probability or cross-system generality; MC acceptance is distinct from candidate generation; step-limit LS prequench is recorded but is not a stationary-point certificate"}
    utilities.dump(output / "provenance.json", result)
    utilities.dump(output / "effective_config.json", {
        "source_lifecycle_config": str(prior_path), "source_lifecycle_config_sha256": sha256(prior_path),
        "ssw": asdict(config), "recovered_rotation": asdict(rotation),
        "ls_arms": result["configuration"]["ls_arms"],
        "seeds": SEEDS, "outer_steps": 1,
        "search_cap_per_arm": SEARCH_CAP_PER_ARM, "wall_seconds_per_arm": WALL_SECONDS_PER_ARM,
        "fresh_cap_per_arm": FRESH_CAP_PER_ARM})
    utilities.dump(output / "summary.json", result)

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_started = time.monotonic()
    try:
        calculator = MACECalculator(model_paths=str(MODEL), head="omol", device="cuda",
                                    default_dtype="float64", enable_cueq=False, enable_oeq=False)
    except Exception as error:
        result.update(status="model_load_failed", error=repr(error),
                      model_load_seconds=time.monotonic() - model_started)
        utilities.dump(output / "summary.json", result)
        return result
    result["model_load_seconds"] = time.monotonic() - model_started
    calc_counter = utilities.instrument_calculate(calculator)
    campaign_complete = True

    for start_name, start_path, start_ref in (("gauche", plus_path, torsion_gauche),
                                               ("trans", minus_path, torsion_trans)):
        source_atoms = read(start_path, index=0)
        for seed in SEEDS:
            for arm_name, fraction in ARMS:
                run_id = f"{start_name}-seed{seed}-{arm_name}"
                folder = output / run_id
                folder.mkdir(exist_ok=False)
                shutil.copy2(start_path, folder / "source-input.extxyz")
                case = {"run_id": run_id, "start": start_name, "seed": seed,
                        "arm": arm_name, "initial_fraction": fraction,
                        "source_input": str(start_path), "source_input_sha256": sha256(start_path),
                        "status": "started", "mc_accepted": None,
                        "candidate_produced": False, "candidate_fresh_checked": False,
                        "search_requests": 0, "fresh_requests": 0}
                result["runs"].append(case)
                utilities.dump(folder / "summary.json", case)
                utilities.dump(output / "summary.json", result)
                arm_started = time.monotonic()
                calls_before = calc_counter["calls"]
                calls_after_search = calls_before
                surface = fresh = None
                ssw_result = None
                try:
                    campaign_remaining = CAMPAIGN_WALL_SECONDS - (time.monotonic() - campaign_started)
                    if campaign_remaining <= 0:
                        raise TimeoutError("campaign_wall_cap_before_arm")
                    calculator.reset()
                    base = utilities.CountedSurface(
                        calculator, folder / "requests.jsonl", SEARCH_CAP_PER_ARM,
                        min(WALL_SECONDS_PER_ARM, campaign_remaining))
                    surface = ResetCountedSurface(base)
                    ls_settings = None
                    if fraction is not None:
                        ls_settings = LSSettings(
                            dict(HC_BOND_ENERGIES),
                            {key: value + 0.1 for key, value in HC_BOND_LENGTHS.items()},
                            target_per_atom=0.7, initial_fraction=fraction, xi=0.2,
                            learning_rate=1.8, prequench=prequench)
                    ssw_result = run_ssw(
                        source_atoms.copy(), surface, steps=1, config=config,
                        rng=np.random.default_rng(seed), ls=ls_settings,
                        recovered_rotation=rotation,
                        checkpoint_path=folder / "checkpoint.pkl")
                    calls_after_search = calc_counter["calls"]
                    utilities.dump(folder / "result.json", ssw_result)
                    if ssw_result.checkpoint is not None:
                        utilities.dump(folder / "checkpoint.json", ssw_result.checkpoint)
                    write(folder / "initial-minimum.extxyz", ssw_result.initial.atoms)
                    write(folder / "current.extxyz", ssw_result.current)
                    write(folder / "best.extxyz", ssw_result.best.atoms if hasattr(ssw_result.best, "atoms") else ssw_result.best)
                    if ls_settings is not None:
                        case["initial_effective_W"] = initial_softening_record(
                            ssw_result.initial.atoms, ls_settings, FrozenBondSoftening,
                            folder / "initial-frozen-W.json")
                    else:
                        case["initial_effective_W"] = None

                    case["run_ssw_status"] = ssw_result.status
                    record = ssw_result.records[-1] if ssw_result.records else None
                    case["run_ssw_status"] = ssw_result.status
                    case["returned_outer_records"] = len(ssw_result.records)
                    case["minima_count"] = len(ssw_result.minima)
                    valid_true_landing_statuses = {
                        "gaussian_limit", "lower_true_energy", "stage_release",
                        "true_quench_failed", "mc_failed", "ls_update_failed",
                        "direction_selection_failed", "starter_selection_failed"}
                    landing = None if record is None else record.landing
                    if record is not None:
                        case.update({"outer_step_status": record.status,
                                     "mc_accepted": bool(record.accepted),
                                     "evaluation_requests": int(ssw_result.evaluation_requests),
                                     "request_count_matches_result": surface.requests == ssw_result.evaluation_requests,
                                     "ls_preparation": record.ls_preparation,
                                     "ls_update": record.ls_update,
                                     "landing_reported": landing,
                                     "landing_object_present": landing is not None,
                                     "landing_surface": None if landing is None else landing.surface})
                    is_true_candidate = bool(
                        record is not None and record.status in valid_true_landing_statuses and
                        landing is not None and landing.surface == "true")
                    case["candidate_produced"] = is_true_candidate
                    case["landing_is_candidate_regardless_of_mc"] = is_true_candidate
                    if landing is not None and not is_true_candidate:
                        write(folder / "failed-stage-geometry.extxyz", landing.atoms)
                        case["failed_stage_geometry"] = {
                            "surface": landing.surface, "converged": bool(landing.converged),
                            "energy_eV": float(landing.energy), "max_force_eV_A": float(landing.max_force),
                            "outer_step_status": None if record is None else record.status,
                            "note": "retained diagnostic geometry; not a physical true-surface landing"}
                    if is_true_candidate:
                        candidate_atoms = landing.atoms.copy()
                        case["candidate_status"] = {"energy_eV": float(landing.energy),
                                                     "reported_fmax_eV_A": float(landing.max_force),
                                                     "converged": bool(landing.converged),
                                                     "surface": landing.surface,
                                                     "optimizer_steps": int(landing.optimizer_steps),
                                                     "evaluation_requests": int(landing.evaluation_requests)}
                        write(folder / "candidate.extxyz", candidate_atoms)
                        case["candidate_geometry_diagnostics"] = candidate_diagnostics(
                            candidate_atoms, references, graph_tools.graph,
                            graph_tools.component_formulas, ring_tools)
                        remaining_campaign = CAMPAIGN_WALL_SECONDS - (time.monotonic() - campaign_started)
                        remaining_arm = WALL_SECONDS_PER_ARM - (time.monotonic() - arm_started)
                        calculator.reset()
                        fresh_base = utilities.CountedSurface(
                            calculator, folder / "fresh-requests.jsonl", FRESH_CAP_PER_ARM,
                            max(0.0, min(remaining_campaign, remaining_arm)))
                        fresh = ResetCountedSurface(fresh_base)
                        try:
                            fresh_energy, fresh_forces = fresh.evaluate(candidate_atoms)
                            fresh_fmax = float(np.linalg.norm(fresh_forces, axis=1).max())
                            same_numbers = bool(np.array_equal(candidate_atoms.numbers, source_atoms.numbers))
                            same_cell = bool(np.array_equal(candidate_atoms.cell.array, source_atoms.cell.array))
                            same_pbc = bool(np.array_equal(candidate_atoms.pbc, source_atoms.pbc))
                            case["candidate_fresh_check"] = {
                                "energy_eV": float(fresh_energy),
                                "energy_difference_from_reported_eV": float(fresh_energy - landing.energy),
                                "fmax_eV_A": fresh_fmax,
                                "force_qualified": bool(fresh_fmax <= config.fmax and same_numbers and same_cell and same_pbc),
                                "same_numbers": same_numbers, "same_cell": same_cell, "same_pbc": same_pbc,
                                "fresh_requests": int(fresh.requests)}
                            case["candidate_fresh_checked"] = True
                        except Exception as error:
                            case["candidate_fresh_check"] = {"status": "failed", "error": repr(error),
                                                              "fresh_requests": int(fresh.requests)}
                    case["status"] = "returned_ssw_result"
                    case["physical_landing_completed"] = bool(
                        is_true_candidate and landing.converged and surface.denials == 0)
                    case["complete_escape"] = case["physical_landing_completed"]
                    if not is_true_candidate:
                        case["escape_diagnostic_status"] = "no_valid_true_surface_landing"
                    elif not landing.converged:
                        case["escape_diagnostic_status"] = "true_surface_landing_not_force_converged"
                    elif surface.denials:
                        case["escape_diagnostic_status"] = "surface_budget_or_wall_truncated"
                    else:
                        case["escape_diagnostic_status"] = "force_converged_true_surface_landing"
                except Exception as error:
                    calls_after_search = calc_counter["calls"]
                    case.update(status="exception", error=repr(error))
                    partial = getattr(error, "result", None)
                    if partial is not None:
                        utilities.dump(folder / "partial-result.json", partial)
                        if hasattr(partial, "atoms"):
                            write(folder / "partial-atoms.extxyz", partial.atoms)
                finally:
                    if surface is not None:
                        case.update(search_requests=int(surface.requests),
                                    search_denials=int(surface.denials),
                                    search_boundary=surface.boundary,
                                    search_calculator_calls=int(calls_after_search - calls_before),
                                    search_count_matches_result=bool(
                                        ssw_result is not None and surface.requests == ssw_result.evaluation_requests),
                                    search_truncated=bool(surface.denials),
                                    search_truncation_reason=surface.boundary)
                    if fresh is not None:
                        case.update(fresh_requests=int(fresh.requests), fresh_denials=int(fresh.denials),
                                    fresh_boundary=fresh.boundary,
                                    fresh_calculator_calls=int(calc_counter["calls"] - calls_after_search))
                    case["elapsed_seconds"] = time.monotonic() - arm_started
                    case["actual_calculate_calls_total"] = int(calc_counter["calls"] - calls_before)
                    if (folder / "checkpoint.pkl").is_file():
                        try:
                            restored = paper_reference.load_ssw_checkpoint(folder / "checkpoint.pkl")
                            utilities.dump(folder / "checkpoint.json", restored)
                        except Exception as error:
                            case["checkpoint_read_error"] = repr(error)
                    utilities.dump(folder / "summary.json", case)
                    result["search_requests_total"] = sum(row.get("search_requests", 0) for row in result["runs"])
                    result["fresh_requests_total"] = sum(row.get("fresh_requests", 0) for row in result["runs"])
                    result["actual_calculate_calls_total"] = int(calc_counter["calls"])
                    result["returned_results"] = sum(row.get("run_ssw_status") is not None for row in result["runs"])
                    result["complete_escapes"] = sum(bool(row.get("complete_escape")) for row in result["runs"])
                    result["elapsed_seconds"] = time.monotonic() - campaign_started
                    utilities.dump(output / "summary.json", result)
                if time.monotonic() - campaign_started >= CAMPAIGN_WALL_SECONDS:
                    campaign_complete = False
                    break
            if not campaign_complete:
                break
        if not campaign_complete:
            break

    result["elapsed_seconds"] = time.monotonic() - campaign_started
    result["status"] = "all_24_arms_recorded" if len(result["runs"]) == 24 else "campaign_incomplete"
    result["total_calculator_calls"] = int(calc_counter["calls"])
    result["model_loads"] = 1
    utilities.dump(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torsion_references_dir", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = main_run(args.torsion_references_dir, args.out)
    print(json.dumps({"status": result.get("status"), "arms_returned": len(result.get("runs", [])),
                      "search_requests": result.get("search_requests_total"),
                      "fresh_requests": result.get("fresh_requests_total"),
                      "out": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
