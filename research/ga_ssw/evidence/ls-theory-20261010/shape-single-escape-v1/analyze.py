#!/usr/bin/env python3
"""Offline paired readout for one frozen shape-LS campaign."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

STARTS = ("c4-gauche", "c4-trans", "c60-minimum", "c60-isomer3")
SEEDS = (701, 702)
ARMS = ("none", "pair-0.03", "shape-0.03")
EXPECTED = tuple(f"{start}-seed{seed}-{arm}"
                 for start in STARTS for seed in SEEDS for arm in ARMS)
GRAPH_KEYS = {
    "c4-gauche": ("hc-table-plus-0.1A",), "c4-trans": ("hc-table-plus-0.1A",),
    "c60-minimum": ("1.64A", "1.70A", "1.80A"),
    "c60-isomer3": ("1.64A", "1.70A", "1.80A"),
}


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def arm_key(row):
    if row.get("run_id"):
        return str(row["run_id"])
    if all(key in row for key in ("start", "seed", "arm")):
        return f'{row["start"]}-seed{row["seed"]}-{row["arm"]}'
    return None


def radius_norm(bundle):
    return number((bundle or {}).get("geometric_center_radius_A", {}).get("norm"))


def load_graph_helpers(campaign_dir):
    """Load only the graph functions from the archived campaign source snapshot."""
    source = campaign_dir / "source"
    if not source.is_dir():
        raise FileNotFoundError(f"frozen source snapshot missing: {source}")
    sys.path.insert(0, str(source))

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import helper from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    runner = load("frozen_shape_escape_runner", source / "research/ga_ssw/probe_ls_shape_paired_escape.py")
    c4 = load("frozen_c4_graph_helper", source / "research/ga_ssw/analyze_c4h6_ls_reaction_coverage.py")
    from pamssw.standalone.native_ls import HC_BOND_LENGTHS
    return runner.graph_bundle, runner.compare_graphs, c4.graph, HC_BOND_LENGTHS


def load_artifacts(summary_path, row, input_record, helpers):
    from ase.io import read

    graph_bundle, compare_graphs, c4_graph, c4_lengths = helpers
    run_id = arm_key(row)
    folder = summary_path.parent / run_id if run_id else None
    start = row.get("start")
    family = "c4" if start in ("c4-gauche", "c4-trans") else "c60"
    artifacts = {"result_json": "missing", "source_input": "missing",
                 "initial_geometry": "missing", "landing_geometry": "missing"}
    result = None
    raw_atoms = initial_atoms = landing_atoms = None
    initial_bundle = landing_bundle = None
    if folder is None:
        return artifacts, None, None, None

    result_path = folder / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
            artifacts["result_json"] = "read"
        except (OSError, json.JSONDecodeError) as error:
            artifacts["result_json"] = f"error: {error}"

    raw_path = folder / "source-input.extxyz"
    if raw_path.is_file():
        try:
            raw_atoms = read(raw_path, index=0)
            artifacts["source_input"] = "read"
        except Exception as error:
            artifacts["source_input"] = f"error: {type(error).__name__}: {error}"

    initial_path = folder / "initial-minimum.extxyz"
    if initial_path.is_file():
        try:
            initial_atoms = read(initial_path, index=0)
            initial_bundle = graph_bundle(initial_atoms, family, c4_graph, c4_lengths)
            artifacts["initial_geometry"] = "read"
        except Exception as error:
            artifacts["initial_geometry"] = f"error: {type(error).__name__}: {error}"

    landing_path = folder / "landing.extxyz"
    if landing_path.is_file():
        try:
            landing_atoms = read(landing_path, index=0)
            landing_bundle = graph_bundle(landing_atoms, family, c4_graph, c4_lengths)
            artifacts["landing_geometry"] = "read"
        except Exception as error:
            artifacts["landing_geometry"] = f"error: {type(error).__name__}: {error}"

    raw_bundle = input_record.get("graph_and_radius") or {}
    raw_to_initial = compare_graphs(raw_bundle, initial_bundle) if initial_bundle else {}
    initial_to_landing = compare_graphs(initial_bundle, landing_bundle) if initial_bundle and landing_bundle else {}
    def torsion(atoms):
        if family != "c4" or atoms is None:
            return None
        try:
            return float(atoms.get_dihedral(0, 1, 2, 3, mic=False))
        except Exception:
            return None

    return artifacts, result, {
        "initial_bundle": initial_bundle, "landing_bundle": landing_bundle,
        "raw_to_initial_isomorphic_by_cutoff": raw_to_initial,
        "initial_to_landing_isomorphic_by_cutoff": initial_to_landing,
        "raw_torsion_deg": torsion(raw_atoms), "initial_torsion_deg": torsion(initial_atoms),
        "landing_torsion_deg": torsion(landing_atoms),
    }, folder


def compact_ls_prequench(preparation):
    if not isinstance(preparation, dict):
        return None
    soft = preparation.get("soft_quench") or {}
    telemetry = soft.get("optimizer_telemetry") or {}
    telemetry_keys = ("backend", "termination_reason", "converged", "optimizer_success",
                      "evaluator_calls", "backend_evaluations", "accepted_steps",
                      "rejected_steps", "line_search_evaluations")
    return {
        "qualification": preparation.get("qualification"),
        "exit_policy": preparation.get("exit_policy"),
        "max_force_eV_A": number(soft.get("max_force")),
        "converged": soft.get("converged"),
        "optimizer_steps": soft.get("optimizer_steps"),
        "soft_quench_requests": soft.get("evaluation_requests"),
        "total_preparation_requests": preparation.get("evaluation_requests"),
        "true_energy_before_eV": number(preparation.get("true_energy_before")),
        "true_energy_after_eV": number(preparation.get("true_energy_after")),
        "telemetry": {key: telemetry.get(key) for key in telemetry_keys if key in telemetry},
    }


def make_arm_record(summary_path, row, input_records, helpers):
    start = row.get("start")
    input_record = input_records.get(start, {})
    artifacts, result, geometry, folder = load_artifacts(summary_path, row, input_record, helpers)
    initial_record = (result or {}).get("initial") or {}
    initial_energy = number(initial_record.get("energy"))
    input_energy = number(input_record.get("energy_eV"))
    fresh = row.get("fresh_candidate") or {}
    landing_energy = number(row.get("landing_energy_eV"))
    fresh_energy = number(fresh.get("energy_eV")) if fresh.get("status") == "completed" else None
    true_landing_energy = fresh_energy if fresh_energy is not None else landing_energy
    true_delta = (true_landing_energy - initial_energy
                  if row.get("landing_surface") == "true" and
                  true_landing_energy is not None and initial_energy is not None else None)
    input_to_initial_delta = (initial_energy - input_energy
                              if initial_energy is not None and input_energy is not None else None)

    raw_bundle = input_record.get("graph_and_radius") or {}
    initial_bundle = (geometry or {}).get("initial_bundle") or {}
    landing_bundle = (geometry or {}).get("landing_bundle") or {}
    raw_radius, initial_radius, landing_radius = map(radius_norm, (raw_bundle, initial_bundle, landing_bundle))
    raw_initial_ratio = (initial_radius / raw_radius
                         if raw_radius not in (None, 0.0) and initial_radius is not None else None)
    landing_initial_ratio = (landing_radius / initial_radius
                             if initial_radius not in (None, 0.0) and landing_radius is not None else None)

    comparisons = (geometry or {}).get("initial_to_landing_isomorphic_by_cutoff", {})
    graph_changes = {}
    for cutoff in GRAPH_KEYS.get(start, ()):
        if cutoff not in comparisons:
            graph_changes[cutoff] = "unknown"
        else:
            graph_changes[cutoff] = "unchanged_graph" if comparisons[cutoff] else "changed_graph"

    denials = row.get("search_denials", 0)
    try:
        denials = int(denials or 0)
    except (TypeError, ValueError):
        denials = None
    record_complete = row.get("status") == "returned" and row.get("outer_records") == 1
    driver_completed = row.get("run_ssw_status") == "completed"
    failed_or_censored = denials is None or denials > 0 or not record_complete or not driver_completed
    return {
        "run_id": arm_key(row), "start": start, "seed": row.get("seed"), "arm": row.get("arm"),
        "record_status": row.get("status"), "run_ssw_status": row.get("run_ssw_status"),
        "outer_step_status": row.get("outer_step_status"),
        "complete_one_step_record": record_complete, "driver_run_completed": driver_completed,
        "failure_or_censored": failed_or_censored, "error": row.get("error"),
        "artifact_status": artifacts,
        "source_artifacts": {"result_json": f"{arm_key(row)}/result.json",
                             "input_geometry": f"{arm_key(row)}/source-input.extxyz",
                             "actual_initial_geometry": f"{arm_key(row)}/initial-minimum.extxyz",
                             "landing_geometry": f"{arm_key(row)}/landing.extxyz"},
        "candidate_generated_independent_of_mc": row.get("landing_is_candidate_independent_of_mc") is True,
        "mc_accepted": row.get("mc_accepted"), "true_surface_landing": row.get("landing_surface") == "true",
        "landing_converged": row.get("landing_converged"),
        "fresh_candidate_status": fresh.get("status"), "fresh_force_qualified": fresh.get("force_qualified") is True,
        "graph_change_initial_to_landing_by_cutoff": graph_changes,
        "initial_to_landing_isomorphic_by_cutoff": comparisons,
        "raw_input_to_initial_isomorphic_by_cutoff": (geometry or {}).get("raw_to_initial_isomorphic_by_cutoff", {}),
        "c4_torsion_raw_input_deg": (geometry or {}).get("raw_torsion_deg"),
        "c4_torsion_actual_initial_deg": (geometry or {}).get("initial_torsion_deg"),
        "c4_torsion_landing_deg": (geometry or {}).get("landing_torsion_deg"),
        "input_true_energy_eV": input_energy, "initial_true_energy_eV": initial_energy,
        "input_to_initial_true_energy_delta_eV": input_to_initial_delta,
        "landing_true_energy_eV": landing_energy, "fresh_true_energy_eV": fresh_energy,
        "landing_minus_actual_initial_true_energy_eV": true_delta,
        "true_energy_delta_source": ("fresh_candidate" if fresh_energy is not None else
                                      "reported_true_surface_landing" if row.get("landing_surface") == "true" else None),
        "raw_input_radius_norm_A": raw_radius, "actual_initial_radius_norm_A": initial_radius,
        "landing_radius_norm_A": landing_radius,
        "actual_initial_to_raw_radius_ratio": raw_initial_ratio,
        "landing_to_actual_initial_radius_ratio": landing_initial_ratio,
        "search_requests": row.get("search_requests"), "fresh_requests": row.get("fresh_requests"),
        "search_denials": denials, "search_boundary": row.get("search_boundary"),
        "actual_calculate_calls": row.get("actual_calculate_calls"),
        "elapsed_seconds": number(row.get("elapsed_seconds")),
        "gaussian_count": row.get("gaussian_count"),
        "ls_prequench": compact_ls_prequench(row.get("ls_preparation")),
        "ls_update_present": row.get("ls_update") is not None,
        "_landing_graph_bundle": landing_bundle,
    }


def sum_known(rows, key):
    values = [number(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def graph_pair_outcome(left, right, cutoff):
    a = left["graph_change_initial_to_landing_by_cutoff"].get(cutoff, "unknown")
    b = right["graph_change_initial_to_landing_by_cutoff"].get(cutoff, "unknown")
    if "unknown" in (a, b):
        return "unknown"
    if a == b == "changed_graph":
        return "both_changed_graph"
    if a == b == "unchanged_graph":
        return "both_unchanged_graph"
    return "left_only_changed" if a == "changed_graph" else "right_only_changed"


def analyze(summary_path, summary):
    input_records = {item.get("name"): item for item in summary.get("input_qualifications", [])
                     if isinstance(item, dict) and item.get("name")}
    rows_by_id = defaultdict(list)
    for row in summary.get("runs", []):
        if isinstance(row, dict):
            key = arm_key(row)
            if key:
                rows_by_id[key].append(row)
    duplicates = sorted(key for key, rows in rows_by_id.items() if len(rows) > 1)
    observed = set(rows_by_id)
    missing = [key for key in EXPECTED if key not in observed]
    unexpected = sorted(observed - set(EXPECTED))
    helpers = load_graph_helpers(summary_path.parent.parent)
    runs = [make_arm_record(summary_path, rows_by_id[key][0], input_records, helpers)
            for key in EXPECTED if key in rows_by_id]

    by_arm = {}
    for arm in ARMS:
        selected = [row for row in runs if row["arm"] == arm]
        by_arm[arm] = {
            "expected": len(STARTS) * len(SEEDS), "present": len(selected),
            "complete_one_step_records": sum(row["complete_one_step_record"] for row in selected),
            "driver_runs_completed": sum(row["driver_run_completed"] for row in selected),
            "candidate_generated_independent_of_mc": sum(row["candidate_generated_independent_of_mc"] for row in selected),
            "fresh_force_qualified_candidates": sum(row["fresh_force_qualified"] for row in selected),
            "fresh_checks_completed": sum(row["fresh_candidate_status"] == "completed" for row in selected),
            "fresh_checks_failed": sum(row["fresh_candidate_status"] == "failed" for row in selected),
            "generated_candidates_without_fresh_result": sum(
                row["candidate_generated_independent_of_mc"] and
                row["fresh_candidate_status"] not in ("completed", "failed") for row in selected),
            "search_requests_total": sum_known(selected, "search_requests"),
            "fresh_requests_total": sum_known(selected, "fresh_requests"),
            "actual_calculate_calls_total": sum_known(selected, "actual_calculate_calls"),
            "elapsed_seconds_total": sum_known(selected, "elapsed_seconds"),
            "failures_or_censored": sum(row["failure_or_censored"] for row in selected),
            "mc_accepted": sum(row["mc_accepted"] is True for row in selected),
            "landing_minus_actual_initial_true_energy_deltas_eV": [row["landing_minus_actual_initial_true_energy_eV"] for row in selected],
            "landing_to_actual_initial_radius_ratios": [row["landing_to_actual_initial_radius_ratio"] for row in selected],
            "ls_prequench_records_present": sum(row["ls_prequench"] is not None for row in selected),
        }

    paired_graph = {}
    by_key = {(row["start"], row["seed"], row["arm"]): row for row in runs}
    for left_arm, right_arm in (("shape-0.03", "pair-0.03"), ("shape-0.03", "none"),
                                ("pair-0.03", "none")):
        per_cutoff = {}
        for start in STARTS:
            for cutoff in GRAPH_KEYS[start]:
                counts = Counter()
                for seed in SEEDS:
                    left = by_key.get((start, seed, left_arm))
                    right = by_key.get((start, seed, right_arm))
                    outcome = ("missing_pair" if left is None or right is None else
                               graph_pair_outcome(left, right, cutoff))
                    counts[outcome] += 1
                per_cutoff[f"{start}:{cutoff}"] = dict(sorted(counts.items()))
        paired_graph[f"{left_arm}_vs_{right_arm}"] = per_cutoff

    compare_graphs = helpers[1]
    landing_graph_pairs = {}
    landing_graph_pair_counts = {}
    for left_arm, right_arm in (("shape-0.03", "pair-0.03"), ("shape-0.03", "none"),
                                ("pair-0.03", "none")):
        comparison_groups = {}
        counts_by_start_cutoff = defaultdict(Counter)
        for start in STARTS:
            for seed in SEEDS:
                left = by_key.get((start, seed, left_arm))
                right = by_key.get((start, seed, right_arm))
                label = f"{start}-seed{seed}"
                group = {}
                group_status = ("both_valid_true_candidates" if left and right and
                                left["candidate_generated_independent_of_mc"] and
                                right["candidate_generated_independent_of_mc"] else
                                "candidate_missing_or_not_true_surface")
                for cutoff in GRAPH_KEYS[start]:
                    left_bundle = left.get("_landing_graph_bundle") if left else None
                    right_bundle = right.get("_landing_graph_bundle") if right else None
                    comparable = (group_status == "both_valid_true_candidates" and left_bundle and right_bundle)
                    isomorphic = (None if not comparable else
                                  compare_graphs(left_bundle, right_bundle).get(cutoff))
                    group[cutoff] = isomorphic
                    counts_by_start_cutoff[f"{start}:{cutoff}"][
                        "unknown" if isomorphic is None else "isomorphic" if isomorphic else "nonisomorphic"] += 1
                comparison_groups[label] = {"status": group_status, "by_cutoff": group}
        landing_graph_pairs[f"{left_arm}_vs_{right_arm}"] = comparison_groups
        landing_graph_pair_counts[f"{left_arm}_vs_{right_arm}"] = {
            key: dict(sorted(value.items())) for key, value in counts_by_start_cutoff.items()}

    # Full graph edge lists and atom arrays stay in the per-arm source artifacts.
    for row in runs:
        row.pop("_landing_graph_bundle", None)

    qualifications = [{"name": item.get("name"), "status": item.get("status"),
                       "fmax_eV_A": item.get("fmax_eV_A"), "energy_eV": item.get("energy_eV"),
                       "source3_preflight": item.get("source3_preflight")}
                      for item in summary.get("input_qualifications", []) if isinstance(item, dict)]
    return {
        "analysis_scope": "offline read of campaign summary, each arm result.json, and saved initial/landing geometries; no PES or request ledgers",
        "campaign_status": summary.get("status"), "expected_arms": len(EXPECTED),
        "present_unique_expected_arms": len(set(EXPECTED) & observed),
        "missing_expected_arm_ids": missing, "duplicate_arm_ids": duplicates, "unexpected_arm_ids": unexpected,
        "input_qualifications": qualifications,
        "input_qualification_requests": summary.get("input_qualification_requests"),
        "campaign_search_requests": summary.get("search_requests_total"),
        "campaign_fresh_requests": summary.get("fresh_requests_total"),
        "campaign_qualification_and_fresh_requests": summary.get("qualification_and_fresh_requests_total"),
        "campaign_elapsed_seconds": number(summary.get("elapsed_seconds")),
        "by_arm": by_arm, "paired_graph_change_comparisons_by_cutoff": paired_graph,
        "paired_landing_graph_isomorphism": landing_graph_pairs,
        "paired_landing_graph_isomorphism_counts": landing_graph_pair_counts, "runs": runs,
        "interpretation_notes": [
            "Graph isomorphism is reported separately for raw input to actual initial minimum, actual initial to landing, and landing-to-landing across arms, at every configured cutoff.",
            "Graph changed/unchanged is a connectivity diagnostic only and never labels basin identity.",
            "True-energy change uses the fresh true-surface candidate energy when available, otherwise the reported true-surface landing energy, minus result.initial.energy; LS or bias energies are excluded.",
            "Radius ratios compare geometric-center norms for raw input to actual initial, and actual initial to landing.",
            "Missing, duplicate, failed, and censored arms remain visible; absent candidates are not classified as force failures."
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "run-1739743" / "summary.json")
    args = parser.parse_args()
    summary_path = args.summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    print(json.dumps(analyze(summary_path, summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
