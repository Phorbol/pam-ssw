#!/usr/bin/env python3
"""Zero-PES aggregation of the 36-arm direction-components panel."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


CASES = ("lj38", "lj55", "c60")
METHODS = ("full", "local_only", "feedback_only")
SEEDS = (26101231, 26101232, 26101233, 26101234)
FMAX = {"lj38": 0.05, "lj55": 0.05, "c60": 0.03}


def read_json(path: Path):
    try:
        return json.loads(path.read_text()), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def read_jsonl(path: Path):
    if not path.is_file():
        return None, "missing"
    rows = []
    try:
        with path.open() as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as exc:
                    return rows, f"invalid line {line_no}: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return rows, f"{type(exc).__name__}: {exc}"
    return rows, None


def ledger_cost(path: Path):
    rows, error = read_jsonl(path)
    if rows is None:
        return {"status": "missing", "error": error, "rows": 0,
                "paid_requests": None, "denied_requests": None,
                "paid_failures": None, "actual_calculator_calls": None}
    if error:
        status = "invalid"
    else:
        status = "ok"
    return {
        "status": status,
        "error": error,
        "rows": len(rows),
        "paid_requests": sum(bool(row.get("charged")) for row in rows),
        "denied_requests": sum(row.get("status") == "denied" for row in rows),
        "paid_failures": sum(bool(row.get("charged")) and row.get("status") == "failed"
                              for row in rows),
        "actual_calculator_calls": sum(int(row.get("actual_calculator_calls", 0) or 0)
                                       for row in rows),
    }


def matches(recounted, reported):
    if reported is None or recounted is None:
        return None
    return recounted == reported


def connected_diagnostic(case, diagnostics):
    """Return connectivity fields only; never assign basin/topology identity."""
    if not isinstance(diagnostics, dict):
        return None
    if case.startswith("lj"):
        return {
            "connected_component_sizes_1p3sigma": diagnostics.get(
                "connected_component_sizes_1p3sigma"),
            "full_cluster": diagnostics.get("full_cluster"),
        }
    graphs = diagnostics.get("graphs")
    if not isinstance(graphs, dict):
        return {"graphs": graphs}
    return {"graphs": {
        str(cutoff): {
            "components": row.get("components") if isinstance(row, dict) else None,
            "connected": (row.get("components") == 1) if isinstance(row, dict) else None,
            "edges": row.get("edges") if isinstance(row, dict) else None,
            "degree_counts": row.get("degree_counts") if isinstance(row, dict) else None,
            "graph_cage_candidate": row.get("graph_cage_candidate") if isinstance(row, dict) else None,
            "reported_ih_graph_match": row.get("ih_graph_match") if isinstance(row, dict) else None,
            "ih_graph_match": None,
            "ih_graph_match_note": "reference_not_supplied; raw value is not a tested negative",
        } for cutoff, row in graphs.items()
    }}


def landing_rows(case, result):
    initial = result.get("initial") if isinstance(result, dict) else None
    initial_energy = initial.get("energy_eV") if isinstance(initial, dict) else None
    rows = []
    for event in (result.get("outer_events") or []):
        energy = event.get("landing_energy_eV")
        if energy is None:
            continue
        force = event.get("landing_max_force_eV_A")
        rows.append({
            "outer_index": event.get("index"),
            "status": event.get("status"),
            "accepted": event.get("accepted"),
            "saved_landing_is_candidate": True,
            "energy_eV": energy,
            "delta_energy_from_initial_eV": (
                None if initial_energy is None else float(energy) - float(initial_energy)),
            "max_force_eV_A": force,
            "fmax_threshold_eV_A": FMAX[case],
            "passes_force_value_gate": (None if force is None else float(force) <= FMAX[case]),
            # Current runner schema does not serialize landing.converged.
            "optimizer_converged": event.get("landing_converged"),
            "connected_diagnostic": connected_diagnostic(
                case, event.get("landing_structure_diagnostics")),
            "structure_reference": event.get("landing_structure"),
        })
    return rows


def exposure_summary(result):
    total = 0
    outer_with_exposure = 0
    details = []
    for event in (result.get("outer_events") or []):
        exposures = event.get("second_gaussian_exposures")
        if exposures is None:
            exposures = [item for item in (event.get("climb") or [])
                         if isinstance(item, dict) and int(item.get("index", -1)) >= 1
                         and item.get("recovered_direction") is not None]
        exposures = [item for item in exposures if isinstance(item, dict)]
        total += len(exposures)
        if exposures:
            outer_with_exposure += 1
        # Full stage arrays stay in the original result; summaries need only
        # the exposure identity, including zero-direction release.
        details.extend({"outer_index": event.get("index"),
                        "index": item.get("index"), "status": item.get("status")}
                       for item in exposures)
    return {"index_ge_1_exposure_count": total,
            "outer_attempts_with_exposure": outer_with_exposure,
            "exposures": details}


def classify_slot(result, result_error, path_exists, ledgers):
    if not path_exists:
        return "incomplete_no_result" if ledgers["search"]["status"] != "missing" else "missing_unrun"
    if result_error:
        return "invalid_result"
    outcome = result.get("execution_outcome")
    status = result.get("result_status", result.get("status"))
    search = ledgers["search"]
    outer_statuses = {str(event.get("status", "")).lower()
                      for event in (result.get("outer_events") or [])}
    explicit_failed_statuses = {
        "failed", "mc_failed", "direction_selection_failed", "initial_failed",
        "quench_failed", "rotation_failed", "evaluation_failure_without_cap_denial",
    }
    fresh = result.get("fresh") or {}
    if (result.get("error") or fresh.get("error") or
            outcome in ("runner_exception", "evaluation_failure_without_cap_denial") or
            status in explicit_failed_statuses or
            outer_statuses.intersection(explicit_failed_statuses) or
            (search["paid_failures"] or 0) > 0):
        return "failed"
    if (outcome == "bounded_search_or_wall_truncation" or
            (status in ("evaluation_failed", "wall_exhausted", "paused") and
             search["denied_requests"] and not search["paid_failures"])):
        return "censored"
    if status in ("evaluation_failed", "exception"):
        return "failed"
    if status == "completed":
        return "completed"
    if status in ("paused", "wall_exhausted"):
        return "censored"
    return "incomplete"


def find_slot(root: Path, case, method, seed):
    return root / case / method / f"seed-{seed}"


def slot_summary(case, method, seed, slot_dirs):
    if not slot_dirs:
        return {"case": case, "method": method, "seed": seed,
                "execution_status": "missing_unrun", "result_path": None,
                "search_cost": ledger_cost(Path("/nonexistent")),
                "fresh_cost": {"status": "missing", "paid_requests": None,
                               "actual_calculator_calls": None}}
    if len(slot_dirs) > 1:
        return {"case": case, "method": method, "seed": seed,
                "execution_status": "duplicate_slot_outputs",
                "slot_directories": [str(p) for p in slot_dirs],
                "error": "multiple roots contain this panel slot; no result selected"}

    slot_dir = slot_dirs[0]
    result_path = slot_dir / "result.json"
    result, result_error = read_json(result_path) if result_path.is_file() else (None, None)
    search_cost = ledger_cost(slot_dir / "search-ef-ledger.jsonl")
    fresh_cost = ledger_cost(slot_dir / "fresh-ef-ledger.jsonl")
    status = classify_slot(result, result_error, result_path.is_file(),
                           {"search": search_cost})
    if fresh_cost["status"] == "missing" and result and (result.get("fresh") or {}).get("requests", 0) == 0:
        fresh_cost["status"] = "not_requested"
        fresh_cost["paid_requests"] = 0
        fresh_cost["actual_calculator_calls"] = 0
    reported_recount = (result or {}).get("search_cost_recount_from_ledger") or {}
    fresh_summary = (result or {}).get("fresh") or {}
    no_fresh_selected = (fresh_summary.get("requests") == 0 and
                         fresh_summary.get("selected_role") is None and
                         fresh_cost["status"] == "not_requested")
    comparisons = {
        "ledger_vs_result_search_requests": matches(search_cost["paid_requests"],
                                                      (result or {}).get("search_requests")),
        "ledger_vs_result_denied_requests": matches(search_cost["denied_requests"],
                                                      (result or {}).get("denied_requests")),
        "ledger_vs_result_actual_calculator_calls": matches(
            search_cost["actual_calculator_calls"],
            (result or {}).get("actual_calculator_calls")),
        "ledger_vs_result_recount_paid_search_requests": matches(
            search_cost["paid_requests"], reported_recount.get("paid_search_requests")),
        "ledger_vs_result_recount_denied_requests": matches(
            search_cost["denied_requests"], reported_recount.get("denied_requests")),
        "ledger_vs_result_recount_paid_failures": matches(
            search_cost["paid_failures"], reported_recount.get("paid_evaluation_failures")),
        "ledger_vs_result_recount_actual_calculator_calls": matches(
            search_cost["actual_calculator_calls"],
            reported_recount.get("actual_calculator_calls")),
        "fresh_ledger_vs_summary_requests": (True if no_fresh_selected else matches(
            fresh_cost["paid_requests"], fresh_summary.get("requests"))),
        "fresh_ledger_vs_summary_calculator_calls": (True if no_fresh_selected else matches(
            fresh_cost["actual_calculator_calls"], fresh_summary.get("actual_calculator_calls"))),
    }
    exposures = exposure_summary(result or {})
    outer_events = (result or {}).get("outer_events") or []
    if not outer_events:
        exposure_state = "no_outer_event_observed"
    elif exposures["index_ge_1_exposure_count"]:
        exposure_state = "exposed"
    else:
        exposure_state = "outer_observed_but_index_ge_1_unexposed"
    return {
        "case": case, "method": method, "seed": seed,
        "execution_status": status,
        "result_status": None if result is None else result.get("result_status", result.get("status")),
        "execution_outcome": None if result is None else result.get("execution_outcome"),
        "error": result_error or ((result or {}).get("error")),
        "slot_directory": str(slot_dir), "result_path": str(result_path) if result_path.exists() else None,
        "source": None if result is None else result.get("source"),
        "outer_steps_requested": None if result is None else result.get("outer_steps_requested"),
        "outer_attempts_observed": len(outer_events),
        "exposure_state": exposure_state,
        **exposures,
        "initial": None if result is None else result.get("initial"),
        "saved_landing_candidates": [] if result is None else landing_rows(case, result),
        "fresh_summary": None if result is None else result.get("fresh"),
        "search_cost": search_cost,
        "fresh_cost": fresh_cost,
        "ledger_consistency": comparisons,
        "elapsed_seconds": None if result is None else result.get("elapsed_seconds"),
    }


def aggregate(roots):
    slots = []
    for case in CASES:
        for method in METHODS:
            for seed in SEEDS:
                slot_dirs = []
                for root in roots:
                    candidate = find_slot(root, case, method, seed)
                    if candidate.is_dir():
                        slot_dirs.append(candidate)
                slots.append(slot_summary(case, method, seed, slot_dirs))

    status_counts = Counter(item["execution_status"] for item in slots)
    groups = []
    for case in CASES:
        for method in METHODS:
            subset = [s for s in slots if s["case"] == case and s["method"] == method]
            costs = [s for s in subset if s.get("search_cost", {}).get("status") in ("ok", "invalid")]
            fresh = [s for s in subset if s.get("fresh_cost", {}).get("status") in ("ok", "invalid", "not_requested")]
            groups.append({
                "case": case, "method": method, "expected_slots": len(SEEDS),
                "execution_status_counts": dict(Counter(s["execution_status"] for s in subset)),
                "exposed_slots": sum(s.get("index_ge_1_exposure_count", 0) > 0 for s in subset),
                "index_ge_1_exposure_count": sum(s.get("index_ge_1_exposure_count", 0) for s in subset),
                "search_paid_requests_known_subtotal": sum(s["search_cost"]["paid_requests"] or 0 for s in costs),
                "search_actual_calculator_calls_known_subtotal": sum(s["search_cost"]["actual_calculator_calls"] or 0 for s in costs),
                "fresh_paid_requests_known_subtotal": sum(s["fresh_cost"]["paid_requests"] or 0 for s in fresh),
                "fresh_actual_calculator_calls_known_subtotal": sum(s["fresh_cost"]["actual_calculator_calls"] or 0 for s in fresh),
                "all_search_ledgers_present": len(costs) == len(SEEDS),
                "all_fresh_costs_accounted": len(fresh) == len(SEEDS),
            })
    ledger_mismatches = []
    for slot in slots:
        for check, value in (slot.get("ledger_consistency") or {}).items():
            if value is False:
                ledger_mismatches.append({"case": slot["case"], "method": slot["method"],
                                          "seed": slot["seed"], "check": check})
    search_ledgers = [s["search_cost"] for s in slots
                      if s.get("search_cost", {}).get("status") in ("ok", "invalid")]
    fresh_ledgers = [s["fresh_cost"] for s in slots
                     if s.get("fresh_cost", {}).get("status") in ("ok", "invalid", "not_requested")]
    return {
        "schema": "direction-components-panel-readout-v1",
        "zero_pes_analysis": True,
        "roots": [str(p) for p in roots],
        "root_existence": [{"path": str(p), "exists": p.is_dir()} for p in roots],
        "contract": {"expected_slots": 36, "cases": list(CASES),
                     "methods": list(METHODS), "seeds": list(SEEDS)},
        "slot_status_counts": dict(status_counts),
        "cost_totals_known_subtotal": {
            "search_paid_requests": sum(x["paid_requests"] or 0 for x in search_ledgers),
            "search_actual_calculator_calls": sum(x["actual_calculator_calls"] or 0 for x in search_ledgers),
            "search_denied_requests": sum(x["denied_requests"] or 0 for x in search_ledgers),
            "fresh_paid_requests": sum(x["paid_requests"] or 0 for x in fresh_ledgers),
            "fresh_actual_calculator_calls": sum(x["actual_calculator_calls"] or 0 for x in fresh_ledgers),
            "search_ledgers_available": len(search_ledgers),
            "fresh_costs_accounted": len(fresh_ledgers),
            "expected_slots": 36,
            "complete": len(search_ledgers) == 36 and len(fresh_ledgers) == 36,
        },
        "groups": groups,
        "ledger_mismatch_count": len(ledger_mismatches),
        "ledger_mismatches": ledger_mismatches,
        "slots": slots,
        "interpretation_limits": [
            "A saved landing remains a candidate regardless of MC acceptance.",
            "Energy differences do not identify basins.",
            "Graph connectivity and graph matches are diagnostics, not basin labels.",
            "The current runner does not serialize landing optimizer convergence; force-value gates are reported separately.",
            "Missing, failed, censored, and unexposed arms remain in the 36-slot denominator.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True,
                        help="a run output root containing case/method/seed-* directories; repeat for each job root")
    parser.add_argument("--out", type=Path, required=True,
                        help="write the JSON summary to this new path")
    args = parser.parse_args(argv)
    roots = [p.resolve() for p in args.root]
    if not any(p.exists() for p in roots):
        parser.error("none of the supplied roots exists")
    report = aggregate(roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except FileExistsError:
        parser.error(f"refusing to overwrite existing output: {args.out}")
    print(f"slots={len(report['slots'])}/36 statuses={report['slot_status_counts']} "
          f"ledger_mismatches={report['ledger_mismatch_count']} summary={args.out}")
    for row in report["groups"]:
        print(f"{row['case']:4s} {row['method']:13s} "
              f"slots={row['expected_slots']} exposed={row['exposed_slots']} "
          f"index>=1={row['index_ge_1_exposure_count']} "
          f"search_paid_known={row['search_paid_requests_known_subtotal']} "
          f"search_calls_known={row['search_actual_calculator_calls_known_subtotal']} "
          f"fresh_paid_known={row['fresh_paid_requests_known_subtotal']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
