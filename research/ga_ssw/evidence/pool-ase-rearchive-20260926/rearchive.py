"""Zero-PES ordered-vs-ASE re-archive of four frozen C4H6 trajectories."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from pamssw.archive import MinimaArchive
from pamssw.state import State
from research.ga_ssw.pool_molecular_archive import ASEPermutationArchive

ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ls-mechanism-panel-20260926")
MAIN_ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification")
AUDIT = ROOT / "research/ga_ssw/evidence/ls-climb-depth-panel-20260926/topology-audit.json"
ARM_ORDER = (("ssw", 61), ("ssw", 67), ("native_ls", 61), ("native_ls", 67))
RMSD_TOL_A = 0.1
ENERGY_TOL_EV = 0.001
TIME_BUDGET_S = 230.0
FORCE_LIMIT_EV_A = 0.03
MATCHER_SOURCE_PATHS = ("research/ga_ssw/pool_molecular_archive.py",
                        "pamssw/archive.py", "pamssw/state.py")
EXPECTED_MATCHER_SHA256 = {
    "research/ga_ssw/pool_molecular_archive.py": "55c7502d73b42d3989272a29e960fb233e49d7105adf017951659ae0a1b878bb",
    "pamssw/archive.py": "b8895c1a428881330e5555d4f308e788d76e433a91480f7309e48ed3ee43b3d9",
    "pamssw/state.py": "00b705c157526a65202a5846927553098fadfd75f7e04b2e2618bb1da1e04131",
}
_TOPOLOGY_PATH = ROOT / "research/ga_ssw/evidence/ls-climb-depth-panel-20260926/audit_topology.py"
_TOPOLOGY_SPEC = importlib.util.spec_from_file_location("frozen_c4h6_topology_audit", _TOPOLOGY_PATH)
TOPOLOGY = importlib.util.module_from_spec(_TOPOLOGY_SPEC)
_TOPOLOGY_SPEC.loader.exec_module(TOPOLOGY)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(arm_name, seed):
    audit = json.loads(AUDIT.read_text())
    descriptor = next(item for item in audit["arms"]
                      if item["arm"] == arm_name and int(item["seed"]) == seed)
    path = Path(descriptor["source"])
    observed_sha = sha256(path)
    if descriptor.get("source_sha256") != observed_sha:
        raise ValueError(f"source SHA-256 mismatch: {path}")
    data = json.loads(path.read_text())
    records = data["records"]
    raw_minima_count = len(data.get("minima", ()))
    topology_arm = next(item for item in audit["arms"]
                        if item["arm"] == arm_name and int(item["seed"]) == seed)
    raw_graph_representatives = [(0, None, TOPOLOGY.make_graph(data["initial"]["atoms"]))]
    for discovery in topology_arm["all_candidate_noninitial_first_discoveries"]:
        record_index = int(discovery["first_record"])
        atoms = records[record_index]["landing"]["atoms"]
        raw_graph_representatives.append((int(discovery["class_id"]), record_index,
                                          TOPOLOGY.make_graph(atoms)))

    candidates = [{
        "observation_index": 0,
        "source_record_index": None,
        "accepted": None,
        "status": "initial",
        "energy": float(data["initial"]["energy"]),
        "numbers": data["initial"]["atoms"]["numbers"],
        "positions": data["initial"]["atoms"]["positions"],
        "recorded_graph_class": match_recorded_graph_class(raw_graph_representatives,
                                                            data["initial"]["atoms"]),
    }]
    excluded = []
    for expected_index, record in enumerate(records):
        index = int(record["index"])
        if index != expected_index:
            raise ValueError(f"noncontiguous record index {index}; expected {expected_index}: {path}")
        landing = record.get("landing")
        force = None if landing is None else landing.get("max_force")
        qualified = (landing is not None and landing.get("converged") is True
                     and landing.get("surface") == "true" and force is not None
                     and math.isfinite(float(force)) and float(force) <= FORCE_LIMIT_EV_A)
        if not qualified:
            excluded.append({"record_index": int(record["index"]),
                             "status": record.get("status"),
                             "accepted": record.get("accepted"),
                             "landing_present": landing is not None,
                             "landing_converged": None if landing is None else landing.get("converged")})
            continue
        atoms = landing["atoms"]
        candidates.append({
            "observation_index": len(candidates),
            "source_record_index": int(record["index"]),
            "accepted": record.get("accepted"),
            "status": record.get("status"),
            "energy": float(landing["energy"]),
            "numbers": atoms["numbers"],
            "positions": atoms["positions"],
            "recorded_graph_class": match_recorded_graph_class(raw_graph_representatives, atoms),
        })
    if raw_minima_count != len(candidates):
        raise ValueError(
            f"raw minima/input denominator mismatch for {arm_name} seed {seed}: "
            f"minima={raw_minima_count}, initial+qualified_landings={len(candidates)}")
    return path, observed_sha, data, candidates, excluded


def match_recorded_graph_class(representatives, atom_data):
    graph = TOPOLOGY.make_graph(atom_data)
    for class_id, _record_index, representative in representatives:
        if TOPOLOGY.nx.is_isomorphic(graph, representative, node_match=TOPOLOGY.NODE_MATCH):
            return class_id
    return None


def state_from(candidate):
    return State(numbers=np.asarray(candidate["numbers"], dtype=int).copy(),
                 positions=np.asarray(candidate["positions"], dtype=float).copy(),
                 cell=np.zeros((3, 3)), pbc=(False, False, False))


def instrument(archive):
    original = archive._rmsd
    metrics = {"rmsd_calls": 0, "rmsd_seconds": 0.0, "add_seconds": 0.0}

    def timed_rmsd(lhs, rhs):
        start = time.perf_counter()
        try:
            return original(lhs, rhs)
        finally:
            metrics["rmsd_calls"] += 1
            metrics["rmsd_seconds"] += time.perf_counter() - start

    archive._rmsd = timed_rmsd
    return metrics


def assigned_pair_diagnostics(candidate, candidate_state, ordered_entry, ase_entry,
                              ordered_rep_observation, ase_rep_observation, candidate_by_index):
    diagnostics = {}
    for label, entry, rep_observation in (("ordered_assigned_reference", ordered_entry,
                                           ordered_rep_observation),
                                          ("ase_assigned_reference", ase_entry,
                                           ase_rep_observation)):
        reference = entry.state
        reference_meta = candidate_by_index[rep_observation]
        energy_delta = float(candidate["energy"] - reference_meta["energy"])
        diagnostics[label] = {
            "representative_observation_index": rep_observation,
            "representative_source_record_index": reference_meta["source_record_index"],
            "candidate_source_record_index": candidate["source_record_index"],
            "representative_recorded_graph_class": reference_meta["recorded_graph_class"],
            "candidate_recorded_graph_class": candidate["recorded_graph_class"],
            "candidate_minus_representative_energy_eV": energy_delta,
            "absolute_energy_difference_eV": abs(energy_delta),
            "different_known_recorded_graph_class": (
                candidate["recorded_graph_class"] is not None
                and reference_meta["recorded_graph_class"] is not None
                and candidate["recorded_graph_class"] != reference_meta["recorded_graph_class"]),
            "ordered_rmsd_A": float(MinimaArchive._rmsd(candidate_state, reference)),
            "ase_rmsd_A": float(ASEPermutationArchive._rmsd(candidate_state, reference)),
        }
    return diagnostics


def process(arm_name, seed):
    source_path, source_sha, data, candidates, excluded = load_source(arm_name, seed)
    ordered = MinimaArchive(ENERGY_TOL_EV, RMSD_TOL_A)
    ase = ASEPermutationArchive(ENERGY_TOL_EV, RMSD_TOL_A)
    ordered_metrics, ase_metrics = instrument(ordered), instrument(ase)
    representative_observation = {"ordered": {}, "ase": {}}
    mapping = {"ordered": [], "ase": []}
    candidate_by_index = {item["observation_index"]: item for item in candidates}
    rows, divergences = [], []
    start_all = time.monotonic()
    stopped_by_budget = False

    for candidate in candidates:
        state = state_from(candidate)
        before_counts = (len(ordered.entries), len(ase.entries))
        ordered_before_calls, ordered_before_rmsd = ordered_metrics["rmsd_calls"], ordered_metrics["rmsd_seconds"]
        ase_before_calls, ase_before_rmsd = ase_metrics["rmsd_calls"], ase_metrics["rmsd_seconds"]

        t0 = time.perf_counter()
        ordered_entry = ordered.add(state, candidate["energy"], parent_id=None)
        ordered_metrics["add_seconds"] += time.perf_counter() - t0
        ordered_new = len(ordered.entries) > before_counts[0]
        if ordered_new:
            representative_observation["ordered"][ordered_entry.entry_id] = candidate["observation_index"]
        mapping["ordered"].append(ordered_entry.entry_id)

        t0 = time.perf_counter()
        ase_entry = ase.add(state, candidate["energy"], parent_id=None)
        ase_metrics["add_seconds"] += time.perf_counter() - t0
        ase_new = len(ase.entries) > before_counts[1]
        if ase_new:
            representative_observation["ase"][ase_entry.entry_id] = candidate["observation_index"]
        mapping["ase"].append(ase_entry.entry_id)

        ordered_rep = representative_observation["ordered"][ordered_entry.entry_id]
        ase_rep = representative_observation["ase"][ase_entry.entry_id]
        row = {
            "observation_index": candidate["observation_index"],
            "source_record_index": candidate["source_record_index"],
            "accepted": candidate["accepted"],
            "status": candidate["status"],
            "energy_eV": candidate["energy"],
            "recorded_graph_class": candidate["recorded_graph_class"],
            "ordered_entry_id": ordered_entry.entry_id,
            "ordered_new_entry": ordered_new,
            "ordered_representative_observation": ordered_rep,
            "ordered_entries_after_add": len(ordered.entries),
            "ordered_rmsd_calls_for_add": ordered_metrics["rmsd_calls"] - ordered_before_calls,
            "ordered_rmsd_seconds_for_add": ordered_metrics["rmsd_seconds"] - ordered_before_rmsd,
            "ase_entry_id": ase_entry.entry_id,
            "ase_new_entry": ase_new,
            "ase_representative_observation": ase_rep,
            "ase_entries_after_add": len(ase.entries),
            "ase_rmsd_calls_for_add": ase_metrics["rmsd_calls"] - ase_before_calls,
            "ase_rmsd_seconds_for_add": ase_metrics["rmsd_seconds"] - ase_before_rmsd,
        }
        mapping_changed = ordered_rep != ase_rep
        classification_changed = ordered_new != ase_new
        if mapping_changed or classification_changed:
            row["assigned_pair_rmsd_diagnostics"] = assigned_pair_diagnostics(
                candidate, state, ordered_entry, ase_entry, ordered_rep, ase_rep,
                candidate_by_index)
            row["mapping_change_kind"] = (
                "ordered_new_ase_duplicate" if ordered_new and not ase_new else
                "ordered_duplicate_ase_new" if not ordered_new and ase_new else
                "different_representative_observation")
            divergences.append(row.copy())
        rows.append(row)
        if time.monotonic() - start_all >= TIME_BUDGET_S:
            stopped_by_budget = len(rows) < len(candidates)
            break

    processed = len(rows)
    source_total_ef = int(data["initial"].get("evaluation_requests", 0)) \
        + sum(int(record.get("evaluation_requests", 0) or 0) for record in data["records"])
    recorded_ef = data.get("evaluation_requests")
    if recorded_ef is not None and source_total_ef != int(recorded_ef):
        raise ValueError(f"archived E/F cost does not close: {source_total_ef} != {recorded_ef}")
    return {
        "status": "time_budget_stop" if stopped_by_budget else "completed",
        "pes_evaluations": 0,
        "arm": arm_name,
        "seed": seed,
        "source_result": str(source_path),
        "source_sha256": source_sha,
        "source_record_count": len(data["records"]),
        "source_minima_count": len(data.get("minima", ())),
        "source_accepted_record_count": sum(bool(record.get("accepted")) for record in data["records"]),
        "qualified_mc_accepted_landing_count": sum(item["accepted"] is True for item in candidates[1:]),
        "qualified_mc_rejected_landing_count": sum(item["accepted"] is False for item in candidates[1:]),
        "source_total_EF_requests": source_total_ef,
        "topology_audit_sha256": sha256(AUDIT),
        "qualified_landing_count": len(candidates) - 1,
        "excluded_record_count": len(excluded),
        "excluded_records": excluded,
        "observation_total": len(candidates),
        "observation_processed": processed,
        "processed_fraction": processed / len(candidates) if candidates else 1.0,
        "rmsd_tol_A": RMSD_TOL_A,
        "energy_tol_eV": ENERGY_TOL_EV,
        "matcher_costs": {
            "ordered_v1": {**ordered_metrics, "entry_count": len(ordered.entries),
                           "energy_mismatch_hits": ordered.energy_mismatch_hits,
                           "max_energy_mismatch_eV": ordered.max_energy_mismatch},
            "ase_permute_v1": {**ase_metrics, "entry_count": len(ase.entries),
                                "energy_mismatch_hits": ase.energy_mismatch_hits,
                                "max_energy_mismatch_eV": ase.max_energy_mismatch},
        },
        "ordered_mapping": mapping["ordered"],
        "ase_mapping": mapping["ase"],
        "mapping_divergence_count": len(divergences),
        "mapping_divergences": divergences,
        "observations": rows,
        "elapsed_seconds": time.monotonic() - start_all,
        "limits": [
            "The order is initial followed by every numerically qualified landing in raw time order, whether MC accepted or rejected; disconnected geometries remain included.",
            "Graph class is an annotation only; graph class, energy, and acceptance never filter archive candidates.",
            "A difference is matcher-defined entry/mapping divergence at RMSD threshold 0.1 A, not proof of distinct or identical physical basins.",
            "This is offline re-archiving only; it cannot infer online restart performance, LS state, or new search outcomes.",
            "RMSD call time counts archive identity comparisons; archive add time also includes bookkeeping and descriptor updates.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-index", type=int, required=True, choices=range(len(ARM_ORDER)))
    parser.add_argument("--output", type=Path, required=True,
                        help="new arm result path; an existing output is refused")
    parser.add_argument("--main-root", type=Path, default=MAIN_ROOT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hashes = {rel: sha256(args.main_root.resolve() / rel) for rel in MATCHER_SOURCE_PATHS}
    if source_hashes != EXPECTED_MATCHER_SHA256:
        raise ValueError(f"matcher source hash mismatch: {source_hashes}")
    arm_name, seed = ARM_ORDER[args.arm_index]
    result = process(arm_name, seed)
    result["main_root"] = str(args.main_root.resolve())
    result["main_commit"] = __import__("subprocess").check_output(
        ["git", "-C", str(args.main_root.resolve()), "rev-parse", "HEAD"], text=True).strip()
    result["matcher_source_sha256"] = source_hashes
    result["matcher_source_paths"] = MATCHER_SOURCE_PATHS
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
