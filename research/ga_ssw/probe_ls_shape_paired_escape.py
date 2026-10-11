#!/usr/bin/env python3
"""Prepared paired one-step comparison of no LS, pair LS, and shape LS.

The default invocation is preparation-only. Pass --execute after review to
run the frozen 24-arm MH-1/omol development experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import platform
import sys as py_sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "research/ga_ssw/evidence/ls-theory-20261010"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
LIFECYCLE = ROOT / "research/ga_ssw/evidence/c4h6-mh1-lifecycle-20260924/effective_config.json"
LEDGER = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
GRAPH_C4 = ROOT / "research/ga_ssw/analyze_c4h6_ls_reaction_coverage.py"
QUALIFICATION = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282"
SEEDS = (701, 702)
ARMS = ("none", "pair-0.03", "shape-0.03")
SEARCH_CAP = 4000
ARM_WALL_SECONDS = 120
CAMPAIGN_WALL_SECONDS = 3600
QUALIFICATION_CAP = 100
FRESH_CAP_PER_ARM = 1

INPUTS = (
    ("c4-gauche", EVIDENCE / "channel-v1/run-1728173/plus-0.050.extxyz", "c4"),
    ("c4-trans", EVIDENCE / "channel-v1/run-1728173/minus-0.050.extxyz", "c4"),
    ("c60-minimum", EVIDENCE / "c60-finite-response-v1/run-1729152/minimum.extxyz", "c60"),
    ("c60-isomer3", QUALIFICATION / "final-candidate.extxyz", "c60-isomer3"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_file(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph_bundle(atoms, family: str, c4_graph=None, c4_lengths=None) -> dict:
    """Return element-labelled graphs with stable edge lists and radial scale."""
    import networkx as nx
    pos = np.asarray(atoms.positions, float)
    center = pos.mean(axis=0)
    radii = np.linalg.norm(pos - center, axis=1)
    out = {"atom_count": len(atoms), "elements": sorted(set(map(int, atoms.numbers))),
           "geometric_center_radius_A": {"norm": float(np.linalg.norm(pos - center)),
                                         "rms": float(np.sqrt(np.mean(radii ** 2))),
                                         "min": float(radii.min()), "max": float(radii.max())}}
    graphs = {}
    if family == "c4":
        if c4_graph is None:
            raise ValueError("C4 graph helper is required")
        g = c4_graph(atoms)
        rows = [("hc-table-plus-0.1A", g,
                 {"rule": "HC_BOND_LENGTHS[element_pair] + 0.1 A",
                  "pair_cutoffs_A": {str(k): float(v) + 0.1 for k, v in c4_lengths.items()}})]
    else:
        rows = []
        cutoffs = (1.64, 1.70, 1.80)
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.linalg.norm(delta, axis=-1)
        for cutoff in cutoffs:
            g = nx.Graph()
            g.add_nodes_from((i, {"number": int(atoms.numbers[i])}) for i in range(len(atoms)))
            g.add_edges_from((int(i), int(j)) for i, j in
                             zip(*np.where(np.triu((distance < cutoff) & (distance > 0), 1))))
            rows.append((f"{cutoff:.2f}A", g, {"distance_cutoff_A": cutoff}))
    for label, g, cutoffs in rows:
        edges = sorted([min(int(i), int(j)), max(int(i), int(j)),
                        int(g.nodes[i]["number"]), int(g.nodes[j]["number"])]
                       for i, j in g.edges())
        graphs[label] = {"cutoff": cutoffs,
                         "nodes": [[int(i), int(g.nodes[i]["number"])] for i in sorted(g.nodes)],
                         "edges": edges,
                         "components": nx.number_connected_components(g),
                         "degree_sequence": sorted(dict(g.degree()).values()),
                         "degree_counts": {str(d): list(dict(g.degree()).values()).count(d)
                                           for d in sorted(set(dict(g.degree()).values()))},
                         "connected": nx.is_connected(g), "all_degree_three": all(d == 3 for d in dict(g.degree()).values())}
    out["graphs"] = graphs
    return out


def compare_graphs(left: dict, right: dict) -> dict:
    import networkx as nx
    result = {}
    for label in sorted(set(left["graphs"]) & set(right["graphs"])):
        def decode(row):
            g = nx.Graph()
            g.add_nodes_from((i, {"number": z}) for i, z in row["nodes"])
            g.add_edges_from((edge[0], edge[1]) for edge in row["edges"])
            return g
        node_match = nx.algorithms.isomorphism.categorical_node_match("number", None)
        result[label] = bool(nx.is_isomorphic(decode(left["graphs"][label]),
                                              decode(right["graphs"][label]),
                                              node_match=node_match))
    return result


class ResetSurface:
    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def evaluate(self, atoms):
        self.base.calculator.reset()
        return self.base.evaluate(atoms)


def execute(out: Path) -> dict:
    import ase
    import mace
    import scipy
    import torch
    from ase.io import read, write
    from mace.calculators import MACECalculator

    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs = []
    for name, path, family in INPUTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        atoms = read(path, index=0)
        if atoms.constraints or atoms.pbc.any() or not np.isfinite(atoms.positions).all():
            raise ValueError(f"shape mode requires finite, unconstrained, isolated geometry: {path}")
        expected = Counter({1: 6, 6: 4}) if family == "c4" else Counter({6: 60})
        if Counter(map(int, atoms.numbers)) != expected:
            raise ValueError(f"unexpected composition in {path}")
        inputs.append((name, path.resolve(), family, atoms))

    if sha256(MODEL) != MODEL_SHA256:
        raise ValueError("MH-1 model hash mismatch")
    if out.exists():
        raise FileExistsError(f"output must be new: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=False)
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone import LSSettings, SSWConfig, run_ssw
    from pamssw.standalone.ls_prequench import LSPrequenchSettings
    from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings
    ledger = import_file("shape_escape_ledger", LEDGER)
    c4_graph_tools = import_file("shape_escape_c4_graph", GRAPH_C4)

    source_config = json.loads(LIFECYCLE.read_text())
    c4_config = SSWConfig(**source_config["config"])
    rotation = RecoveredRotationSettings(**source_config["rotation_settings"])
    expected = (c4_config.width, c4_config.max_gaussians, c4_config.temperature_K,
                c4_config.fmax, c4_config.relax_steps, c4_config.lbfgs_memory)
    if expected != (0.1, 25, 150.0, 0.03, 400, 500):
        raise ValueError(f"unexpected frozen C4 lifecycle config: {expected}")
    configs = {"c4": c4_config,
               "c60": SSWConfig(**{**asdict(c4_config), "fmax": 0.05})}
    prequench = LSPrequenchSettings(fmax=0.1, steps=50, exit_policy="force_or_step_limit")
    ls_configs = {
        "c4": {"bond_energies": dict(HC_BOND_ENERGIES),
               "bond_lengths": {k: v + 0.1 for k, v in HC_BOND_LENGTHS.items()},
               "target_per_atom": 0.7, "xi": 0.2, "learning_rate": 1.8},
        "c60": {"bond_energies": {(6, 6): 3.61}, "bond_lengths": {(6, 6): 1.64},
                "target_per_atom": 0.02, "xi": 0.2, "learning_rate": 1.8},
    }

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    campaign_started = time.monotonic()
    calculator = MACECalculator(model_paths=str(MODEL), head="omol", device="cuda",
                                default_dtype="float64", enable_cueq=False, enable_oeq=False)
    calls = ledger.instrument_calculate(calculator)
    qualification_started = time.monotonic()
    qualifications = []
    qualifications_by_name = {}
    qualification_requests = 0
    ledger.dump(out / "preflight.json", {"status": "started", "inputs": qualifications})
    # Fresh input E/F checks are preflight only; no geometry is optimized here.
    for name, path, family, atoms in inputs:
        calculator.reset()
        qualification_remaining = 120 - (time.monotonic() - qualification_started)
        if qualification_remaining <= 0:
            raise TimeoutError("aggregate input qualification wall cap")
        surface = ledger.CountedSurface(calculator, out / f"preflight-{name}-requests.jsonl",
                                        QUALIFICATION_CAP - qualification_requests, qualification_remaining)
        energy, forces = surface.evaluate(atoms)
        qualification_requests += surface.requests
        item = {"name": name, "path": str(path), "sha256": sha256(path),
                "requests": surface.requests, "energy_eV": energy,
                "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                "graph_and_radius": None, "status": "force_checked",
                "source_qualification_note": (
                    "no Hessian qualification; this experiment does not supply one"
                    if family == "c60-isomer3" else None)}
        qualifications.append(item)
        qualifications_by_name[name] = item
        ledger.dump(out / "preflight.json", {"status": "in_progress", "inputs": qualifications,
                                              "qualification_requests": qualification_requests})
        graph = graph_bundle(atoms, family, c4_graph_tools.graph, HC_BOND_LENGTHS)
        item["graph_and_radius"] = graph
        item["status"] = "qualified"
        if family == "c60-isomer3":
            c60_170 = graph["graphs"]["1.70A"]
            item["source3_preflight"] = {
                "force_at_most_0.05_eV_A": item["fmax_eV_A"] <= 0.05,
                "connected_at_1.70A": c60_170["components"] == 1,
                "all_degree_three_at_1.70A": c60_170["all_degree_three"],
            }
            if not all(item["source3_preflight"].values()):
                item["status"] = "rejected_preflight"
                ledger.dump(out / "preflight.json", {"status": item["status"], "inputs": qualifications,
                                                      "qualification_requests": qualification_requests})
                raise ValueError("C60 isomer 3 failed the frozen force/connectivity/3-regular preflight")
        ledger.dump(out / "preflight.json", {"status": "in_progress", "inputs": qualifications,
                                              "qualification_requests": qualification_requests})
    if qualification_requests > QUALIFICATION_CAP:
        raise RuntimeError("preflight request cap")
    if time.monotonic() - qualification_started > 120:
        raise TimeoutError("input qualification exceeded 120 s")
    ledger.dump(out / "preflight.json", {"status": "qualified", "inputs": qualifications,
                                          "qualification_requests": qualification_requests})

    shutil.copy2(Path(__file__).resolve(), out / Path(__file__).name)
    protocol = EVIDENCE / "shape-single-escape-protocol.md"
    record = {
        "status": "started", "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "model": {"path": str(MODEL), "sha256": MODEL_SHA256, "head": "omol",
                  "device": "cuda", "dtype": "float64", "torch": torch.__version__,
                  "mace": mace.__version__, "ase": ase.__version__, "numpy": np.__version__,
                  "scipy": scipy.__version__, "python": py_sys.version, "platform": platform.platform()},
        "source_hashes": {"runner": sha256(Path(__file__).resolve()), "lifecycle_config": sha256(LIFECYCLE),
                          "ledger": sha256(LEDGER), "c4_graph_helper": sha256(GRAPH_C4),
                          "paper_reference": sha256(ROOT / "pamssw/standalone/paper_reference.py"),
                          "softening": sha256(ROOT / "pamssw/standalone/softening.py"),
                          "shape_softening": sha256(ROOT / "pamssw/standalone/shape_softening.py"),
                          "protocol": sha256(protocol)},
        "limits": {"seeds": SEEDS, "arms": ARMS, "starts": [x[0] for x in inputs],
                   "outer_steps": 1, "search_requests_per_arm": SEARCH_CAP,
                   "wall_seconds_per_arm": ARM_WALL_SECONDS,
                   "campaign_wall_seconds": CAMPAIGN_WALL_SECONDS,
                   "total_qualification_requests_cap": QUALIFICATION_CAP,
                   "fresh_candidate_checks_per_arm": FRESH_CAP_PER_ARM},
        "input_qualifications": qualifications,
        "input_qualification_requests": qualification_requests,
        "configuration": {"c4_ssw": asdict(configs["c4"]), "c60_ssw": asdict(configs["c60"]),
                          "rotation": asdict(rotation),
                          "ls_by_family": {family: {"bond_energies_eV": {str(k): v for k, v in spec["bond_energies"].items()},
                                                     "bond_lengths_A": {str(k): v for k, v in spec["bond_lengths"].items()},
                                                     "target_per_atom_eV": spec["target_per_atom"],
                                                     "xi": spec["xi"], "learning_rate": spec["learning_rate"],
                                                     "prequench": asdict(prequench)}
                                             for family, spec in ls_configs.items()},
                          "arms": {"none": None, "pair-0.03": {"bias_geometry": "pair", "initial_fraction": 0.03},
                                   "shape-0.03": {"bias_geometry": "shape", "initial_fraction": 0.03}}},
        "runs": [], "search_requests_total": 0, "fresh_requests_total": 0,
        "interpretation_limit": "single-step development comparison; four start structures, two paired seeds; not a success-rate, production, or general PES claim"}
    ledger.dump(out / "provenance.json", record)
    ledger.dump(out / "effective_config.json", {k: record[k] for k in ("limits", "configuration", "model", "source_hashes")})
    ledger.dump(out / "summary.json", record)

    for name, path, family, atoms in inputs:
        config = configs["c4" if family == "c4" else "c60"]
        for seed in SEEDS:
            for arm in ARMS:
                folder = out / f"{name}-seed{seed}-{arm}"
                folder.mkdir()
                write(folder / "source-input.extxyz", atoms)
                surface = fresh = None
                started = time.monotonic()
                calls_before = calls["calls"]
                row = {"run_id": folder.name, "start": name, "seed": seed, "arm": arm,
                       "input": str(path), "input_sha256": sha256(path), "status": "started"}
                ledger.dump(folder / "summary.json", row)
                try:
                    if time.monotonic() - campaign_started >= CAMPAIGN_WALL_SECONDS:
                        raise TimeoutError("campaign wall cap")
                    calculator.reset()
                    base = ledger.CountedSurface(calculator, folder / "requests.jsonl", SEARCH_CAP,
                                                  min(ARM_WALL_SECONDS, CAMPAIGN_WALL_SECONDS - (time.monotonic() - campaign_started)))
                    surface = ResetSurface(base)
                    family_spec = ls_configs["c4" if family == "c4" else "c60"]
                    ls = None if arm == "none" else LSSettings(
                        dict(family_spec["bond_energies"]), dict(family_spec["bond_lengths"]),
                        target_per_atom=family_spec["target_per_atom"], initial_fraction=0.03,
                        xi=family_spec["xi"], learning_rate=family_spec["learning_rate"],
                        prequench=prequench,
                        bias_geometry="pair" if arm.startswith("pair") else "shape")
                    result = run_ssw(atoms.copy(), surface, steps=1, config=config,
                                     rng=np.random.default_rng(seed), ls=ls,
                                     recovered_rotation=rotation,
                                     checkpoint_path=folder / "checkpoint.pkl")
                    ledger.dump(folder / "result.json", result)
                    write(folder / "initial-minimum.extxyz", result.initial.atoms)
                    write(folder / "current.extxyz", result.current)
                    rec = result.records[-1] if result.records else None
                    landing = None if rec is None else rec.landing
                    row.update({"run_ssw_status": result.status, "outer_records": len(result.records),
                                "mc_accepted": None if rec is None else bool(rec.accepted),
                                "outer_step_status": None if rec is None else rec.status,
                                "gaussian_count": None if rec is None else len(rec.climb),
                                "ls_preparation": None if rec is None else rec.ls_preparation,
                                "ls_update": None if rec is None else rec.ls_update,
                                "landing_surface": None if landing is None else landing.surface,
                                "landing_converged": None if landing is None else bool(landing.converged),
                                "landing_energy_eV": None if landing is None else float(landing.energy),
                                "landing_fmax_eV_A": None if landing is None else float(landing.max_force),
                                "reported_search_requests": int(result.evaluation_requests),
                                "requests_match_result": surface.requests == result.evaluation_requests})
                    valid_true_landing_statuses = {
                        "gaussian_limit", "lower_true_energy", "stage_release", "true_quench_failed",
                        "mc_failed", "ls_update_failed", "direction_selection_failed",
                        "starter_selection_failed"}
                    is_candidate = bool(rec is not None and rec.status in valid_true_landing_statuses and
                                         landing is not None and landing.surface == "true")
                    row["landing_is_candidate_independent_of_mc"] = is_candidate
                    if landing is not None:
                        write(folder / "landing.extxyz", landing.atoms)
                        candidate_graph = graph_bundle(landing.atoms, family, c4_graph_tools.graph, HC_BOND_LENGTHS)
                        row["landing_graph_and_radius"] = candidate_graph
                        row["graph_isomorphic_to_start"] = compare_graphs(
                            qualifications_by_name[name]["graph_and_radius"], candidate_graph)
                        if is_candidate:
                            fresh = ResetSurface(ledger.CountedSurface(
                                calculator, folder / "fresh-requests.jsonl",
                                min(FRESH_CAP_PER_ARM,
                                    QUALIFICATION_CAP - qualification_requests - record["fresh_requests_total"]),
                                max(0.0, min(ARM_WALL_SECONDS - (time.monotonic() - started),
                                             CAMPAIGN_WALL_SECONDS - (time.monotonic() - campaign_started)))))
                            try:
                                e, f = fresh.evaluate(landing.atoms)
                                fmax = float(np.linalg.norm(f, axis=1).max())
                                same_numbers = bool(np.array_equal(landing.atoms.numbers, atoms.numbers))
                                same_cell = bool(np.array_equal(landing.atoms.cell.array, atoms.cell.array))
                                same_pbc = bool(np.array_equal(landing.atoms.pbc, atoms.pbc))
                                row["fresh_candidate"] = {"status": "completed", "energy_eV": e,
                                                           "reported_energy_error_eV": float(e - landing.energy),
                                                           "fmax_eV_A": fmax, "force_threshold_eV_A": config.fmax,
                                                           "same_numbers": same_numbers, "same_cell": same_cell,
                                                           "same_pbc": same_pbc,
                                                           "force_qualified": bool(fmax <= config.fmax and same_numbers and same_cell and same_pbc),
                                                           "graph_and_radius": row["landing_graph_and_radius"]}
                            except Exception as error:
                                row["fresh_candidate"] = {"status": "failed", "error": repr(error)}
                    row["status"] = "returned"
                except Exception as error:
                    row.update(status="exception", error=repr(error))
                    partial = getattr(error, "result", None)
                    if partial is not None:
                        ledger.dump(folder / "partial-result.json", partial)
                finally:
                    row.update(search_requests=0 if surface is None else surface.requests,
                               search_denials=0 if surface is None else surface.denials,
                               search_boundary=None if surface is None else surface.boundary,
                               fresh_requests=0 if fresh is None else fresh.requests,
                               elapsed_seconds=time.monotonic() - started,
                               actual_calculate_calls=calls["calls"] - calls_before)
                    ledger.dump(folder / "summary.json", row)
                    record["runs"].append(row)
                    record["search_requests_total"] += row["search_requests"]
                    record["fresh_requests_total"] += row["fresh_requests"]
                    record["elapsed_seconds"] = time.monotonic() - campaign_started
                    ledger.dump(out / "summary.json", record)
                if time.monotonic() >= campaign_started + CAMPAIGN_WALL_SECONDS:
                    break
            if time.monotonic() >= campaign_started + CAMPAIGN_WALL_SECONDS:
                break
        if time.monotonic() >= campaign_started + CAMPAIGN_WALL_SECONDS:
            break
    record["status"] = "all_24_arms_recorded" if len(record["runs"]) == 24 else "campaign_incomplete"
    record["qualification_and_fresh_requests_total"] = qualification_requests + record["fresh_requests_total"]
    record["actual_calculate_calls_total"] = calls["calls"]
    record["elapsed_seconds"] = time.monotonic() - campaign_started
    ledger.dump(out / "summary.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new evidence directory")
    parser.add_argument("--execute", action="store_true", help="run the prepared GPU experiment")
    args = parser.parse_args()
    if args.execute:
        result = execute(args.out.expanduser().resolve())
        print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                          "out": str(args.out.resolve())}, indent=2))
    else:
        print("prepared_not_executed: inspect the protocol, then pass --execute to run")


if __name__ == "__main__":
    main()
