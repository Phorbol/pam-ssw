#!/usr/bin/env python3
"""Offline, cost-complete topology analysis for a saved C4H6 LS transfer run."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CASES = ("cyclobutene", "bicyclobutane")
ARMS = ("ssw", "paper_ls")
EXPECTED_MODEL_SHA = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
EXPECTED_INPUT_SHA = {
    "cyclobutene": "87bb85eb4ec12b4aae64a305c21c9a693bed98c2e8c996dc18055672c3227a1b",
    "bicyclobutane": "e9700f800d495b53674064aea953116c32d32f4ecf701582c222a42a87db7571",
}
OUTER_STEPS = 12


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_parse_error": str(exc), "_line": line_number})
    return rows


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _dict(value):
    return value if isinstance(value, dict) else {}


def ledger_cost(path: Path) -> dict:
    rows = read_jsonl(path)
    counts = Counter(row.get("kind", "malformed") for row in rows)
    attempted = [r for r in rows if r.get("kind") in ("search", "search_failure")]
    ids = [r.get("request") for r in attempted]
    integer_ids = [x for x in ids if isinstance(x, int)]
    return {
        "path": str(path), "exists": path.is_file(), "rows": len(rows),
        "kind_counts": dict(counts), "attempted_ef_calls": len(attempted),
        "successful_ef_calls": counts.get("search", 0),
        "failed_ef_calls": counts.get("search_failure", 0),
        "denials_without_ef_call": counts.get("search_denial", 0),
        "unique_request_ids": len(set(integer_ids)),
        "duplicate_request_ids": len(integer_ids) - len(set(integer_ids)),
        "request_id_min": min(integer_ids) if integer_ids else None,
        "request_id_max": max(integer_ids) if integer_ids else None,
        "malformed_rows": sum("_parse_error" in r for r in rows),
    }


def serialize_graph_entry(entry, graph_fn, graph_label_fn, component_formulas_fn, refs):
    import networkx as nx

    atoms = entry["atoms"]
    graph = graph_fn(atoms)
    labels = graph_label_fn(atoms, refs)
    entry["graph"] = graph
    entry["geometry_summary"] = {
        **labels,
        "component_formulas": component_formulas_fn(atoms, graph),
        "formula": atoms.get_chemical_formula(),
        "atom_count": len(atoms),
        "intact_c4h6": len(atoms) == 10 and atoms.get_chemical_formula() == "C4H6",
        "connected_intact_c4h6": (nx.number_connected_components(graph) == 1 and
                                   len(atoms) == 10 and atoms.get_chemical_formula() == "C4H6"),
    }
    return entry


def extract_result_records(result: dict, progress: list[dict]) -> tuple[list[dict], str]:
    records = result.get("records") if isinstance(result, dict) else None
    progress_rows = [row for row in progress if row.get("kind") == "outer_step"]
    if isinstance(records, list) and (records or not progress_rows):
        return records, "result.json"
    return progress_rows, "progress.jsonl"


def response_data(record: dict, target: float) -> dict:
    prep = _dict(record.get("ls_preparation"))
    soft = _dict(prep.get("soft_quench"))
    before, after = prep.get("true_energy_before"), prep.get("true_energy_after")
    response = record.get("energy_response")
    derived = None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        derived = (after - before) / 10.0
    return {
        "outer_index": record.get("index", record.get("step_index")),
        "energy_response_eV_per_atom": response,
        "derived_response_eV_per_atom": derived,
        "target_eV_per_atom": target,
        "response_minus_target_eV_per_atom": (response - target
            if isinstance(response, (int, float)) else None),
        "true_energy_before_eV": before, "true_energy_after_eV": after,
        "preparation_requests": prep.get("evaluation_requests"),
        "preparation_exit_policy": prep.get("exit_policy"),
        "preparation_qualification": prep.get("qualification"),
        "soft_quench_converged": soft.get("converged"),
        "soft_quench_fmax_eV_A": soft.get("max_force"),
        "soft_quench_requests": soft.get("evaluation_requests"),
        "update_record": record.get("ls_update"),
    }


def reconstruct_ls_strengths(records: list[dict], result: dict, start_atoms, effective: dict) -> dict:
    """Replay the existing pure pair-table response update; this performs no PES calls."""
    try:
        from pamssw.standalone.softening import FrozenBondSoftening, LSResponseState
    except Exception as exc:
        return {"available": False, "reason": f"core import failed: {exc!r}"}

    settings = _dict(_dict(effective.get("ls_settings")).get("paper_ls"))
    def table(value):
        out = {}
        for key, amount in _dict(value).items():
            pair = tuple(int(part.strip()) for part in key.strip("() ").split(","))
            out[pair] = float(amount)
        return out

    energies = table(settings.get("bond_energies"))
    lengths = table(settings.get("bond_lengths"))
    if not energies or not lengths:
        return {"available": False, "reason": "effective paper-LS pair tables are absent"}
    target = float(settings.get("target_per_atom", 0.7))
    learning_rate = float(settings.get("learning_rate", 1.8))
    initial_fraction = float(settings.get("initial_fraction", 0.03))
    xi = float(settings.get("xi", 0.2))
    frozen = FrozenBondSoftening.from_atoms(start_atoms, bond_energies=energies,
        bond_lengths=lengths, initial_fraction=initial_fraction, xi=xi,
        energy_filter=settings.get("energy_filter") or None)
    response = LSResponseState(target, learning_rate=learning_rate)
    states = [{"state": "initial", "pair_count": len(frozen.pairs),
               "total_strength_eV": float(sum(frozen.strengths)),
               "strength_eV_per_atom": float(sum(frozen.strengths) / len(start_atoms))}]
    current = start_atoms.copy()
    replayed = []
    errors = []
    for index, record in enumerate(records):
        prep = _dict(record.get("ls_preparation"))
        before, after = prep.get("true_energy_before"), prep.get("true_energy_after")
        if before is None or after is None:
            replayed.append({"outer_index": record.get("index", index), "updated": False,
                             "reason": "missing LS preparation energies"})
            states.append({"state": "after_outer", "outer_index": record.get("index", index),
                           "pair_count": len(frozen.pairs), "total_strength_eV": float(sum(frozen.strengths)),
                           "strength_eV_per_atom": float(sum(frozen.strengths) / len(start_atoms))})
            continue
        landing = _dict(record.get("landing"))
        next_atoms = current
        if record.get("accepted") and isinstance(landing.get("atoms"), dict):
            from ase import Atoms
            a = landing["atoms"]
            next_atoms = Atoms(numbers=a["numbers"], positions=a["positions"],
                               cell=a.get("cell"), pbc=a.get("pbc", False))
        prior_strength = float(sum(frozen.strengths))
        try:
            updated = response.update(frozen, next_atoms, energy_before=float(before),
                                      energy_after=float(after), bond_energies=energies,
                                      bond_lengths=lengths)
            expected = record.get("energy_response")
            response_value = (float(after) - float(before)) / len(start_atoms)
            replayed.append({"outer_index": record.get("index", index), "updated": True,
                             "pair_count_before": len(frozen.pairs),
                             "total_strength_before_eV": prior_strength,
                             "total_strength_after_eV": float(sum(updated.strengths)),
                             "strength_after_eV_per_atom": float(sum(updated.strengths) / len(start_atoms)),
                             "response_eV_per_atom": response_value,
                             "reported_response_eV_per_atom": expected,
                             "response_matches_reported": (expected is None or
                                 abs(response_value - float(expected)) <= 1e-8)})
            frozen = updated
        except Exception as exc:
            errors.append({"outer_index": record.get("index", index), "error": repr(exc)})
            replayed.append({"outer_index": record.get("index", index), "updated": False,
                             "reason": repr(exc)})
        current = next_atoms
        states.append({"state": "after_outer", "outer_index": record.get("index", index),
                       "pair_count": len(frozen.pairs), "total_strength_eV": float(sum(frozen.strengths)),
                       "strength_eV_per_atom": float(sum(frozen.strengths) / len(start_atoms))})

    checkpoint = _dict(_dict(result.get("checkpoint")).get("frozen"))
    final_strengths = checkpoint.get("strengths")
    final_total = float(sum(final_strengths)) if isinstance(final_strengths, list) else None
    replay_total = float(sum(frozen.strengths))
    return {"available": True, "method": "offline replay of existing FrozenBondSoftening/LSResponseState only; no E/F",
            "energy_unit": "eV", "per_atom_unit": "eV/atom", "target_eV_per_atom": target,
            "learning_rate": learning_rate, "initial_fraction": initial_fraction, "xi_dimensionless": xi,
            "states": states, "updates": replayed, "replay_errors": errors,
            "checkpoint_final_strength_total_eV": final_total,
            "replay_final_strength_total_eV": replay_total,
            "checkpoint_final_strength_matches_replay": (None if final_total is None else
                abs(final_total - replay_total) <= 1e-7)}


def analyze_run(run_dir: Path) -> dict:
    import networkx as nx
    import numpy as np
    from ase.io import read

    sys_path = str(ROOT)
    if sys_path not in __import__("sys").path:
        __import__("sys").path.insert(0, sys_path)
    from research.ga_ssw.analyze_c4h6_ls_reaction_coverage import (
        graph, graph_label, component_formulas, assign_global_classes, atoms_from_dict, landing_match)

    provenance = load_json(run_dir / "provenance.json", {}) or {}
    effective = load_json(run_dir / "effective_config.json", {}) or {}
    campaign = load_json(run_dir / "summary.json", {}) or {}
    warnings = []
    checks = {
        "model_sha256_expected": provenance.get("model_sha256") == EXPECTED_MODEL_SHA,
        "model_head_omol": provenance.get("model_head") == "omol",
        "model_device_cuda_float64": (provenance.get("device"), provenance.get("dtype")) == ("cuda", "float64"),
        "source_core_tree_recorded": bool(provenance.get("core_tree")),
        "native_table_cutoff_boundary_recorded": bool(provenance.get("implementation_parameter_boundary")),
    }
    for case in CASES:
        observed = _dict(_dict(provenance.get("inputs")).get(case)).get("sha256")
        checks[f"{case}_input_sha256_expected"] = observed == EXPECTED_INPUT_SHA[case]
    if not all(checks.values()):
        warnings.append("stored provenance does not match the frozen model/input/source contract")

    refs = {}
    reference_paths = provenance.get("inputs", {})
    for case in CASES:
        path = Path(_dict(reference_paths.get(case)).get("path", ""))
        if path.is_file():
            refs[case] = graph(read(path))
    butadiene_path = Path("/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/c4h6-model-qualification-20260924-r2/butadiene/input.extxyz")
    if butadiene_path.is_file():
        refs["butadiene"] = graph(read(butadiene_path))

    summaries = {}
    result_records = {}
    graph_entries = []
    graph_entry_map = {}
    candidates = []
    raw_inputs = {}
    initial_states = {}
    arm_data = {}

    for case in CASES:
        raw_inputs[case] = {}
        for arm in ARMS:
            folder = run_dir / f"{case}-{arm}"
            arm_summary = load_json(folder / "summary.json", {}) or {}
            result = load_json(folder / "result.json", {}) or {}
            progress = read_jsonl(folder / "progress.jsonl")
            checks_list = load_json(folder / "fresh-checks.json", []) or []
            checks_by_index = {row.get("index"): row for row in checks_list if isinstance(row, dict)}
            request_info = ledger_cost(folder / "requests.jsonl")
            fresh_info = ledger_cost(folder / "fresh-requests.jsonl")
            records, record_source = extract_result_records(result, progress)
            result_records[(case, arm)] = records
            input_path = folder / "input.extxyz"
            input_atoms = read(input_path) if input_path.is_file() else None
            if input_atoms is not None:
                input_graph = graph(input_atoms)
                input_label = graph_label(input_atoms, refs)
                entry = {"key": f"{case}/input", "case": case, "arm": "input",
                         "index": 0, "kind": "input", "atoms": input_atoms,
                         "geometry_summary": None}
                serialize_graph_entry(entry, graph, graph_label, component_formulas, refs)
                graph_entries.append(entry)
                graph_entry_map[entry["key"]] = entry
                raw_inputs[case][arm] = {"path": str(input_path), "sha256": sha256(input_path),
                                         "graph_label": input_label,
                                         "graph": input_graph}
            else:
                raw_inputs[case][arm] = {"path": str(input_path), "missing": True}

            result_initial = _dict(result.get("initial"))
            initial_atoms_dict = _dict(result_initial.get("atoms"))
            initial_atoms = (atoms_from_dict(initial_atoms_dict)
                             if initial_atoms_dict.get("numbers") else None)
            initial_path = folder / "initial.extxyz"
            if initial_atoms is None and initial_path.is_file():
                initial_atoms = read(initial_path)
            if initial_atoms is not None:
                key = f"{case}/{arm}/initial_quench"
                entry = {"key": key, "case": case, "arm": arm, "index": 0,
                         "kind": "initial_quench", "atoms": initial_atoms}
                serialize_graph_entry(entry, graph, graph_label, component_formulas, refs)
                graph_entries.append(entry)
                graph_entry_map[key] = entry
                initial_states[(case, arm)] = initial_atoms

            minima = result.get("minima", []) if isinstance(result.get("minima"), list) else []
            arm_candidates = []
            for index, minimum in enumerate(minima):
                md = _dict(minimum)
                atom_data = _dict(md.get("atoms"))
                if not atom_data.get("numbers"):
                    continue
                atoms = atoms_from_dict(atom_data)
                key = f"{case}/{arm}/minimum/{index}"
                entry = {"key": key, "case": case, "arm": arm, "index": index,
                         "kind": "saved_minimum_candidate", "atoms": atoms}
                serialize_graph_entry(entry, graph, graph_label, component_formulas, refs)
                graph_entries.append(entry)
                graph_entry_map[key] = entry
                check = checks_by_index.get(index)
                own_input = raw_inputs[case][arm]
                same_as_input = False
                if own_input.get("graph") is not None:
                    nm = nx.algorithms.isomorphism.categorical_node_match("number", None)
                    same_as_input = nx.is_isomorphic(graph(atoms), own_input["graph"], node_match=nm)
                geo = entry["geometry_summary"]
                connected_intact = bool(geo["connected_intact_c4h6"])
                fresh_qualified = bool(check and check.get("numerical_force_qualified") is True)
                candidate = {
                    "case": case, "arm": arm, "minimum_index": index,
                    "reported_energy_eV": md.get("energy"), "reported_fmax_eV_A": md.get("max_force"),
                    "reported_converged": md.get("converged"), "fresh_check_present": check is not None,
                    "fresh_qualified": fresh_qualified,
                    "fresh_energy_eV": None if check is None else check.get("energy_eV"),
                    "fresh_fmax_eV_A": None if check is None else check.get("fmax_eV_A"),
                    "connected_intact_c4h6": connected_intact,
                    "same_graph_as_own_raw_input": same_as_input,
                    "graph_changed_from_own_raw_input": not same_as_input,
                    "geometry": geo, "global_graph_class": None,
                }
                arm_candidates.append(candidate)
                candidates.append(candidate)

            # Reconcile all cost ledgers against the run, result, progress, and attempt records.
            progress_initial = next((r for r in progress if r.get("kind") == "initial"), None)
            progress_outer_count = sum(r.get("kind") == "outer_step" for r in progress)
            initial_requests = result_initial.get("evaluation_requests")
            if initial_requests is None and progress_initial is not None:
                initial_requests = progress_initial.get("cumulative_requests")
            record_costs = []
            for ri, record in enumerate(records):
                cost = record.get("evaluation_requests")
                if cost is None:
                    cost = record.get("requests")
                record_costs.append({"index": record.get("index", record.get("step_index", ri)),
                                     "status": record.get("status"),
                                     "accepted": record.get("accepted"),
                                     "evaluation_requests": cost,
                                     "climb_stages": len(record.get("climb", []))
                                         if isinstance(record.get("climb"), list)
                                         else record.get("climb_stages"),
                                     "error": record.get("error")})
            known_record_cost = sum(x["evaluation_requests"] for x in record_costs
                                    if isinstance(x["evaluation_requests"], int))
            attempted = request_info["attempted_ef_calls"]
            initial_cost = initial_requests if isinstance(initial_requests, int) else 0
            tail_cost = attempted - initial_cost - known_record_cost
            reported_search = arm_summary.get("search_requests")
            result_cost = result.get("evaluation_requests")
            ledger_result_match = (result_cost is None or result_cost == attempted)
            summary_ledger_match = (reported_search is None or reported_search == attempted)
            result_parts_match = (result_cost is None or
                (initial_requests is not None and
                 result_cost == initial_cost + known_record_cost))
            closure = {
                "initialization_requests": initial_cost,
                "completed_outer_records": len(records),
                "record_source": record_source,
                "progress_outer_records": progress_outer_count,
                "result_progress_outer_count_match": (not result or
                    len(result.get("records", [])) == progress_outer_count),
                "outer_requests_all_records_including_rejected": known_record_cost,
                "unassigned_tail_or_censored_requests": tail_cost,
                "requested_outer_steps": OUTER_STEPS,
                "missing_outer_records": max(0, OUTER_STEPS - len(records)),
                "mean_requests_per_completed_outer_all_statuses":
                    (known_record_cost / len(records) if records else None),
                "total_search_requests_per_requested_outer_including_init_and_tail": attempted / OUTER_STEPS,
                "raw_ledger": request_info, "summary_reported_search_requests": reported_search,
                "result_reported_search_requests": result_cost,
                "ledger_matches_result": ledger_result_match,
                "ledger_matches_arm_summary": summary_ledger_match,
                "result_equals_init_plus_records": result_parts_match,
                "record_costs": record_costs,
                "stop_boundary": arm_summary.get("search_boundary"),
                "arm_status": arm_summary.get("status"),
                "result_status": result.get("status"),
                "result_returned": bool(arm_summary.get("result_returned", bool(result))),
                "denials_not_charged_as_EF": request_info["denials_without_ef_call"],
            }

            target = float(_dict(_dict(effective.get("ls_settings")).get("paper_ls")).get(
                "target_per_atom", 0.7))
            response_rows = []
            for ri, record in enumerate(records):
                if arm == "paper_ls":
                    rrow = response_data(record, target)
                    rrow["outer_index"] = record.get("index", record.get("step_index", ri))
                    response_rows.append(rrow)
            strength_history = {"available": False, "reason": "SSW arm has no LS response"}
            if arm == "paper_ls" and initial_atoms is not None:
                strength_history = reconstruct_ls_strengths(records, result, initial_atoms, effective)

            # Link each landing to a separately fresh-checked archive candidate where possible.
            landing_events = []
            for ri, record in enumerate(records):
                landing = _dict(record.get("landing"))
                landing_atoms = _dict(landing.get("atoms"))
                if not landing_atoms.get("numbers"):
                    continue
                landing_graph = graph(atoms_from_dict(landing_atoms))
                same_input = False
                if own_input.get("graph") is not None:
                    nm = nx.algorithms.isomorphism.categorical_node_match("number", None)
                    same_input = nx.is_isomorphic(landing_graph, own_input["graph"], node_match=nm)
                matched = []
                for minimum_index, minimum in enumerate(minima):
                    if landing_match(minimum, landing):
                        matched.append(minimum_index)
                matched_fresh = [idx for idx in matched
                                 if checks_by_index.get(idx, {}).get("numerical_force_qualified") is True]
                connected = nx.number_connected_components(landing_graph) == 1
                landing_events.append({
                    "outer_index": record.get("index", record.get("step_index", ri)),
                    "status": record.get("status"), "accepted": record.get("accepted"),
                    "energy_eV": landing.get("energy"), "fmax_eV_A": landing.get("max_force"),
                    "connected": connected,
                    "component_formulas": component_formulas(atoms_from_dict(landing_atoms), landing_graph),
                    "same_graph_as_own_raw_input": same_input,
                    "graph_changed_from_own_raw_input": not same_input,
                    "matched_minimum_indices": matched,
                    "matched_fresh_qualified_minimum_indices": matched_fresh,
                    "fresh_qualified_connected_graph_change": bool(
                        connected and not same_input and matched_fresh),
                })

            arm_data[(case, arm)] = {
                "case": case, "arm": arm, "directory": str(folder),
                "summary": arm_summary, "result_present": bool(result),
                "progress_event_counts": dict(Counter(r.get("kind", "unknown") for r in progress)),
                "request_cost": closure, "fresh_request_cost": fresh_info,
                "fresh_checks_count": len(checks_list), "saved_minima_count": len(minima),
                "candidate_counts": {
                    "saved_minima_candidates": len(arm_candidates),
                    "fresh_checked": sum(c["fresh_check_present"] for c in arm_candidates),
                    "fresh_qualified": sum(c["fresh_qualified"] for c in arm_candidates),
                    "fresh_qualified_connected_intact_c4h6": sum(
                        c["fresh_qualified"] and c["connected_intact_c4h6"] for c in arm_candidates),
                    "fresh_qualified_connected_graph_changes": sum(
                        c["fresh_qualified"] and c["connected_intact_c4h6"] and
                        c["graph_changed_from_own_raw_input"] for c in arm_candidates),
                    "fresh_qualified_same_graph": sum(
                        c["fresh_qualified"] and c["connected_intact_c4h6"] and
                        c["same_graph_as_own_raw_input"] for c in arm_candidates),
                    "fresh_qualified_fragmented_or_non_C4H6": sum(
                        c["fresh_qualified"] and not c["connected_intact_c4h6"] for c in arm_candidates),
                },
                "candidate_indices": [c["minimum_index"] for c in arm_candidates],
                "outer_landing_events": landing_events,
                "paper_ls_responses": response_rows,
                "paper_ls_strength_history": strength_history,
                "fresh_ledger_matches_summary": (
                    arm_summary.get("fresh_requests") is None or
                    arm_summary.get("fresh_requests") == fresh_info["attempted_ef_calls"]),
                "fresh_ledger_matches_fresh_checks": fresh_info["attempted_ef_calls"] == len(checks_list),
            }

    # One species-aware global class assignment makes IDs comparable across both inputs and arms.
    assign_global_classes(graph_entries)
    assigned = graph_entries
    for entry in assigned:
        graph_entry_map[entry["key"]] = entry
    input_classes = {}
    for case in CASES:
        e = graph_entry_map.get(f"{case}/input")
        input_classes[case] = e.get("class_id") if e else None
    for candidate in candidates:
        key = f"{candidate['case']}/{candidate['arm']}/minimum/{candidate['minimum_index']}"
        entry = graph_entry_map.get(key)
        if entry:
            candidate["global_graph_class"] = entry["class_id"]

    for case in CASES:
        for arm in ARMS:
            data = arm_data.get((case, arm))
            if data is None:
                continue
            rows = [c for c in candidates if c["case"] == case and c["arm"] == arm]
            base_class = input_classes.get(case)
            all_classes = sorted({c["global_graph_class"] for c in rows
                                  if c["global_graph_class"] is not None})
            connected_classes = sorted({c["global_graph_class"] for c in rows
                if c["global_graph_class"] is not None and c["connected_intact_c4h6"]})
            successful_classes = sorted({c["global_graph_class"] for c in rows
                if c["global_graph_class"] is not None and c["fresh_qualified"]
                and c["connected_intact_c4h6"] and c["graph_changed_from_own_raw_input"]})
            data["graph_classes"] = {
                "input_global_class": base_class,
                "all_saved_minimum_classes": all_classes,
                "new_distinct_classes_vs_own_raw_input": [x for x in all_classes if x != base_class],
                "connected_intact_classes": connected_classes,
                "fresh_qualified_connected_changed_classes": successful_classes,
                "new_distinct_saved_graph_classes_count": len([x for x in all_classes if x != base_class]),
                "fresh_qualified_connected_changed_class_count": len(successful_classes),
            }

    cross_arm = {}
    for case in CASES:
        left = raw_inputs.get(case, {}).get("ssw", {})
        right = raw_inputs.get(case, {}).get("paper_ls", {})
        left_graph, right_graph = left.get("graph"), right.get("graph")
        input_graph_equal = False
        if left_graph is not None and right_graph is not None:
            nm = nx.algorithms.isomorphism.categorical_node_match("number", None)
            input_graph_equal = nx.is_isomorphic(left_graph, right_graph, node_match=nm)
        ls_init = initial_states.get((case, "paper_ls"))
        ssw_init = initial_states.get((case, "ssw"))
        quenched_equal = False
        if ls_init is not None and ssw_init is not None:
            nm = nx.algorithms.isomorphism.categorical_node_match("number", None)
            quenched_equal = nx.is_isomorphic(graph(ls_init), graph(ssw_init), node_match=nm)
        cross_arm[case] = {
            "raw_input_sha_equal": bool(left.get("sha256") and left.get("sha256") == right.get("sha256")),
            "raw_input_graphs_isomorphic": input_graph_equal,
            "initial_quench_graphs_isomorphic": quenched_equal,
            "initial_quench_graph_labels": {
                arm: graph_label(initial_states[(case, arm)], refs)
                for arm in ARMS if (case, arm) in initial_states},
        }

    # In-memory NetworkX graphs/ASE atoms are intentionally excluded from JSON outputs.
    public_classes = [{"key": e["key"], "case": e["case"], "arm": e["arm"],
                       "index": e["index"], "kind": e["kind"],
                       "global_graph_class": e.get("class_id"),
                       "geometry": e.get("geometry_summary")}
                      for e in assigned]
    for value in raw_inputs.values():
        for row in value.values():
            row.pop("graph", None)
    return {
        "schema": "c4h6-ls-isomer-transfer-analysis-v1", "run_directory": str(run_dir),
        "campaign": campaign, "provenance": provenance, "effective_config": effective,
        "provenance_checks": checks, "warnings": warnings,
        "scientific_limits": [
            "MH-1/omol is not the paper's GGA-PBE PES; the result is an algorithm test on this model.",
            "The paper starts from trans-butadiene A; D/E are paper-listed isomers used here as transfer starting inputs.",
            "Pair energies and +0.1 A bond cutoffs come from the recovered native table and prior local implementation, not the paper SI pair table.",
            "Fresh energy/force qualification and graph changes do not establish Hessian-positive minima, transition states, barriers, or rates.",
            "Single seed and 12 outer attempts do not establish an independent success probability or general LS advantage.",
        ],
        "raw_inputs_by_arm": raw_inputs, "shared_initial_graph_checks": cross_arm,
        "global_graph_classes": public_classes,
        "arms": [arm_data.get((case, arm), {"case": case, "arm": arm,
                    "status": "missing_arm_directory"}) for case in CASES for arm in ARMS],
        "candidate_rows": candidates,
        "aggregation_rules": {
            "topology": "existing C4H6 graph() cutoff from HC_BOND_LENGTHS+0.1 A; atom-element labeled isomorphism; global IDs include input, quenched initial, and all saved-minimum candidates",
            "fresh_success": "fresh-checks.json numerical_force_qualified is true; this is only an independent energy/force gate",
            "connected_intact": "one graph component and total composition C4H6",
            "new_graph": "candidate graph not isomorphic to that case's own raw ASE G2 input graph",
            "cost": "CountedSurface search/search_failure rows are attempted E/F calls; denials are reported separately and not charged. Initialization, every completed outer record including MC-rejected records, and unassigned tail/censored calls are separate.",
            "LS_strengths": "reconstructed offline with the existing FrozenBondSoftening and LSResponseState methods from saved true-energy response and MC state; cross-check against final result checkpoint when present; no calculator is instantiated.",
        },
    }


def markdown(data: dict) -> str:
    lines = ["# C4H6 LS isomer-transfer offline analysis", "",
             f"Input run: `{data['run_directory']}`", "",
             "This report counts force-qualified connected graph changes as candidates only. It does not claim Hessian-positive minima or reaction-path validation.", "",
             "## Provenance and shared starts", "",
             f"Model SHA/head/device/dtype: `{data['provenance'].get('model_sha256')}` / `{data['provenance'].get('model_head')}` / `{data['provenance'].get('device')}` / `{data['provenance'].get('dtype')}`.",
             f"Core tree: `{data['provenance'].get('core_tree')}`. Inputs: " + ", ".join(
                 f"`{case}={_dict(_dict(data['provenance'].get('inputs')).get(case)).get('sha256')}`" for case in CASES) + ".", "",
             "| Input | Raw bytes match across arms | Raw graphs match | Post-quench initial graphs match |",
             "|---|---|---|---|"]
    for case in CASES:
        row = data["shared_initial_graph_checks"].get(case, {})
        lines.append(f"| {case} | {row.get('raw_input_sha_equal')} | {row.get('raw_input_graphs_isomorphic')} | {row.get('initial_quench_graphs_isomorphic')} |")
    lines += ["", "## Cost and graph outcomes", "",
              "All completed outer attempts are included regardless of MC acceptance. Initialization and censored/unassigned tail E/F costs remain separate.", "",
              "| Case | Arm | State | Records / 12 | Search E/F | Init | Outer records E/F | Tail | Mean all outer records | Fresh-qualified connected changed candidates | Fresh-qualified connected changed classes | Fresh-qualified fragmented/non-C4H6 candidates | New distinct saved graph classes* |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        for arm in ARMS:
            row = next((x for x in data["arms"] if x.get("case") == case and x.get("arm") == arm), {})
            cost = _dict(row.get("request_cost")); counts = _dict(row.get("candidate_counts")); classes = _dict(row.get("graph_classes"))
            lines.append("| {case} | {arm} | {status} | {records}/{asked} | {search} | {init} | {outer} | {tail} | {mean} | {qualified} | {changed_classes} | {fragmented} | {newclasses} |".format(
                case=case, arm=arm, status=row.get("summary", {}).get("status", row.get("status", "missing")),
                records=cost.get("completed_outer_records", 0), asked=cost.get("requested_outer_steps", OUTER_STEPS),
                search=cost.get("raw_ledger", {}).get("attempted_ef_calls", 0), init=cost.get("initialization_requests", 0),
                outer=cost.get("outer_requests_all_records_including_rejected", 0), tail=cost.get("unassigned_tail_or_censored_requests", 0),
                mean=(f"{cost['mean_requests_per_completed_outer_all_statuses']:.1f}" if cost.get("mean_requests_per_completed_outer_all_statuses") is not None else "NA"),
                qualified=counts.get("fresh_qualified_connected_graph_changes", 0),
                changed_classes=classes.get("fresh_qualified_connected_changed_class_count", 0),
                fragmented=counts.get("fresh_qualified_fragmented_or_non_C4H6", 0),
                newclasses=classes.get("new_distinct_saved_graph_classes_count", 0)))
    lines += ["", "* Saved-class count is the number of distinct topology classes among all saved-minimum candidates that differ from that case's own raw input graph. It is not restricted to fresh-qualified, connected, intact C4H6 candidates; inspect the fresh-qualified connected changed-class and fragmented/non-C4H6 columns separately.", ""]
    lines += ["", "## LS response", "", "Paper target is 0.7 eV/atom. The response is the true-potential energy rise after soft-only prequench divided by 10 atoms; no biased-climb energy is treated as this response.", ""]
    for row in data["arms"]:
        if row.get("arm") != "paper_ls":
            continue
        strength = _dict(row.get("paper_ls_strength_history"))
        responses = row.get("paper_ls_responses", [])
        values = [x.get("energy_response_eV_per_atom") for x in responses
                  if isinstance(x.get("energy_response_eV_per_atom"), (int, float))]
        lines.append(f"- {row.get('case')}: preparation responses {len(values)}/{len(responses)}; observed range "
                     f"{min(values):.4f}–{max(values):.4f} eV/atom" if values else
                     f"- {row.get('case')}: no saved LS preparation response records.")
        lines.append(f"  Strength replay available={strength.get('available')}; final total={strength.get('replay_final_strength_total_eV')} eV; checkpoint match={strength.get('checkpoint_final_strength_matches_replay')}; replay errors={len(strength.get('replay_errors', []))}.")
    lines += ["", "## Evidence limits", ""]
    lines.extend(f"- {item}" for item in data["scientific_limits"])
    if data["warnings"]:
        lines += ["", "Provenance warnings:", ""]
        lines.extend(f"- {item}" for item in data["warnings"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="saved runner output directory")
    parser.add_argument("--output", type=Path, required=True, help="new directory for JSON and Markdown")
    args = parser.parse_args()
    run_dir = args.run.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not run_dir.is_dir():
        parser.error(f"--run directory does not exist: {run_dir}")
    if out.exists() or out == run_dir or out in run_dir.parents or run_dir in out.parents:
        parser.error(f"--output must be a new directory separate from input: {out}")
    if not out.parent.is_dir():
        parser.error(f"--output parent must exist: {out.parent}")
    data = analyze_run(run_dir)
    out.mkdir(parents=False, exist_ok=False)
    (out / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(markdown(data))
    print(f"wrote {out / 'analysis.json'} and {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
