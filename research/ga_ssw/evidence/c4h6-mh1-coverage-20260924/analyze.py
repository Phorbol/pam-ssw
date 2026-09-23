"""Offline fixed-prefix and endpoint audit for the frozen six-arm C4H6 run."""
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def stream_ledger(path):
    """Aggregate a potentially large JSONL ledger without retaining its rows."""
    if not path.is_file():
        return None, [f"missing {path.name}"]
    counts = {"search": 0, "search_failure": 0, "search_denial": 0, "other": 0}
    errors, last_request = [], 0
    denial_rows = []
    with path.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"{path.name}:{line_no}: invalid JSON: {exc!r}")
                continue
            kind, request = row.get("kind"), row.get("request")
            if kind in ("search", "search_failure"):
                counts[kind] += 1
                if not isinstance(request, int) or request != last_request + 1:
                    errors.append(f"{path.name}:{line_no}: nonmonotonic charged request {request!r} after {last_request}")
                if isinstance(request, int):
                    last_request = request
            elif kind == "search_denial":
                counts[kind] += 1
                denial_rows.append({"line": line_no, "request": request, "reason": row.get("reason")})
                if request != last_request:
                    errors.append(f"{path.name}:{line_no}: denial request {request!r} != charged total {last_request}")
            else:
                counts["other"] += 1
                errors.append(f"{path.name}:{line_no}: unexpected ledger kind {kind!r}")
    return {"counts": counts, "charged_requests": counts["search"] + counts["search_failure"],
            "last_charged_request": last_request, "denials": denial_rows,
            "monotonic": not any("nonmonotonic" in e or "denial request" in e for e in errors)}, errors


def helpers():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "research" / "ga_ssw"))
    from analyze_c4h6_ls_reaction_coverage import (
        assign_global_classes, atoms_from_dict, component_formulas, graph,
        graph_label, graph_signature,
    )
    from ase.collections import g2
    refs = {name: graph(g2[name].copy()) for name in
            ("butadiene", "cyclobutene", "2-butyne", "methylenecyclopropane", "bicyclobutane")}
    torsion_path = HERE.parent / "c4h6-torsion-audit-20260924" / "analyze.py"
    spec = importlib.util.spec_from_file_location("c4h6_torsion_helpers", torsion_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load torsion helper: {torsion_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return assign_global_classes, atoms_from_dict, component_formulas, graph, graph_label, graph_signature, mod.torsion, refs


def result_source(folder):
    for name in ("result.json", "partial-exception-result.json", "checkpoint.json"):
        path = folder / name
        if path.is_file():
            return name, load(path)
    return None, None


def audit_arm(arm, seed, plan, h):
    assign, atoms_from_dict, formulas, graph, graph_label, signature, torsion, refs = h
    folder = HERE / f"{arm}-seed{seed}"
    summary = load(folder / "summary.json")
    source, result = result_source(folder)
    errors = []
    if summary is None:
        errors.append("missing summary.json")
    if result is None:
        errors.append("missing result.json, partial-exception-result.json, and checkpoint.json")
    elif source != "result.json":
        errors.append(f"partial evidence source used: {source}")
    search, search_errors = stream_ledger(folder / "requests.jsonl")
    fresh_ledger, fresh_ledger_errors = stream_ledger(folder / "fresh-requests.jsonl")
    errors.extend(search_errors + fresh_ledger_errors)
    checks_path = folder / "fresh-checks.json"
    checks = load(checks_path)
    if not isinstance(checks, list):
        checks = []
        errors.append("missing or malformed fresh-checks.json")
    result = result or {}
    initial = result.get("initial") or {}
    records, minima = result.get("records") or [], result.get("minima") or []
    init_cost, result_cost = initial.get("evaluation_requests"), result.get("evaluation_requests")
    record_costs = [r.get("evaluation_requests") for r in records]
    ledger_count = None if search is None else search["charged_requests"]
    fresh_count = None if fresh_ledger is None else fresh_ledger["charged_requests"]
    summary = summary or {}
    required_summary = ("protocol_completed", "budget_censored", "execution_ok",
        "search_requests", "search_calculator_calls", "search_denials", "search_elapsed_seconds",
        "fresh_requests", "fresh_calculator_calls", "fresh_all_qualified", "fresh_checked")
    for key in required_summary:
        if key not in summary:
            errors.append(f"summary missing required field {key}")
    accounting = {
        "initial_requests": init_cost, "record_requests": record_costs,
        "result_requests": result_cost, "ledger_charged_requests": ledger_count,
        "summary_search_requests": summary.get("search_requests"),
        "request_sum_matches_result": (isinstance(init_cost, int) and isinstance(result_cost, int)
            and all(isinstance(x, int) and x >= 0 for x in record_costs)
            and result_cost == init_cost + sum(record_costs)),
        "result_matches_ledger": result_cost == ledger_count,
        "summary_matches_ledger": summary.get("search_requests") == ledger_count,
        "summary_calculator_calls": summary.get("search_calculator_calls"),
        "summary_denials": summary.get("search_denials"),
        "summary_request_count_matches_result": summary.get("request_count_matches_result"),
        "ledger_audit": search,
    }
    accounting["reconciles"] = all(accounting[k] for k in
        ("request_sum_matches_result", "result_matches_ledger", "summary_matches_ledger")) and not search_errors
    accounting["reconciles"] = (accounting["reconciles"]
        and (search is not None and summary.get("search_denials") == search["counts"]["search_denial"])
        and summary.get("request_count_matches_result") is True)
    if not accounting["reconciles"]:
        errors.append("initial + record costs, result, ledger, and summary do not reconcile")

    # Map minima in the same order as core: initial at index 0, then each
    # converged record.landing appended at paper_reference.py:1150.
    mapping_errors, cumulative, min_costs, record_minimum_indices = [], init_cost, {}, {}
    if not minima or not isinstance(init_cost, int):
        mapping_errors.append("initial minimum or initial request count missing")
    else:
        if minima[0] != initial:
            mapping_errors.append("minima[0] differs from result.initial")
        min_costs[0] = init_cost
    next_minimum = 1
    for record in records:
        cost = record.get("evaluation_requests")
        if isinstance(cumulative, int) and isinstance(cost, int):
            cumulative += cost
        else:
            cumulative = None
        landing = record.get("landing")
        if isinstance(landing, dict) and landing.get("converged") is True:
            if next_minimum >= len(minima):
                mapping_errors.append(f"record {record.get('index')} has an unmatched converged landing")
            elif landing != minima[next_minimum]:
                mapping_errors.append(f"record {record.get('index')} landing differs from minima[{next_minimum}]")
            else:
                min_costs[next_minimum] = cumulative
                record_minimum_indices[record.get("index")] = next_minimum
            next_minimum += 1
    if next_minimum != len(minima):
        mapping_errors.append(f"mapped {next_minimum} minima positions, result contains {len(minima)}")
    if mapping_errors:
        errors.extend(mapping_errors)
    mapping_ok = not mapping_errors
    outer_records = []
    for record in records:
        landing, update = record.get("landing") or {}, record.get("ls_update") or {}
        climb = record.get("climb") or []
        outer_records.append({"index": record.get("index"), "status": record.get("status"),
            "minimum_index": record_minimum_indices.get(record.get("index")),
            "evaluation_requests": record.get("evaluation_requests"), "error": record.get("error"),
            "climb_stops": [{k: e[k] for k in ("index", "status", "stop_reason",
                "rotation_stop_reason", "stage_stop_reason", "error", "requests") if k in e}
                for e in climb if isinstance(e, dict)],
            "landing_converged": landing.get("converged") is True,
            "mc_accepted": record.get("accepted"),
            "accepted_converged_landing": bool(landing.get("converged") is True
                                               and record.get("accepted") is True),
            "energy_response_eV_per_atom": record.get("energy_response"),
            "native_observed_response_meV_per_atom": update.get("observed_response_mev_per_atom"),
            "native_update_step": update.get("step"), "native_update_phase": update.get("phase"),
            "native_update_nsoftstep": update.get("nsoftstep"),
            "native_update_actions": update.get("actions"), "ls_update_raw": update})

    check_by_index, duplicates, malformed = {}, [], []
    for check in checks:
        try:
            i = int(check["index"])
        except (KeyError, TypeError, ValueError):
            malformed.append(check)
            continue
        if i in check_by_index:
            duplicates.append(i)
        check_by_index[i] = check
    missing = sorted(set(range(len(minima))) - set(check_by_index))
    outside = sorted(i for i in check_by_index if i < 0 or i >= len(minima))
    minima_rows, class_entries = [], []
    for i, minimum in enumerate(minima):
        atom_data = minimum.get("atoms")
        if not isinstance(atom_data, dict):
            errors.append(f"minimum {i} has no atoms")
            continue
        atoms = atoms_from_dict(atom_data)
        g, check = graph(atoms), check_by_index.get(i)
        identity = None if check is None else (
            check.get("numbers") == atom_data.get("numbers")
            and check.get("cell_A") == atom_data.get("cell")
            and check.get("pbc") == atom_data.get("pbc"))
        label, components = graph_label(atoms, refs), formulas(atoms, g)
        row = {"index": i, "cumulative_requests": min_costs.get(i),
            "energy_eV": minimum.get("energy"), "fmax_eV_A": minimum.get("max_force"),
            "converged": minimum.get("converged"), "graph": label,
            "graph_signature": signature(g), "component_formulas": components,
            "fragmented": len(components) > 1,
            "fresh": {"present": check is not None, "identity_matches": identity, "record": check}}
        if check is not None:
            check["geometry_identity_fields_match"] = identity
        if "butadiene" in label["reference_graphs"]:
            try:
                path, angle, cosine = torsion(atoms)
                row["butadiene_torsion"] = {"path": list(path), "raw_deg": angle,
                    "cosphi": cosine, "region": "positive" if cosine > 0 else "negative" if cosine < 0 else "zero"}
            except Exception as exc:
                row["butadiene_torsion"] = {"error": repr(exc)}
        entry = {"arm": f"{arm}-seed{seed}", "index": i, "graph": g}
        class_entries.append(entry)
        row["_class_entry"] = entry
        minima_rows.append(row)
    fresh_ledger_counts = None if fresh_ledger is None else fresh_ledger["counts"]
    fresh_identity = all(check_by_index.get(i, {}).get("geometry_identity_fields_match") is True
                         for i in range(len(minima)))
    fmax_limit = plan["config"]["fmax"]
    fresh_values_ok = all(
        check_by_index[i].get("numerical_qualified") is True
        and check_by_index[i].get("same_numbers") is True
        and check_by_index[i].get("same_cell") is True
        and check_by_index[i].get("same_pbc") is True
        and check_by_index[i].get("reported_converged") is True
        and isinstance(check_by_index[i].get("energy_error_eV"), (int, float))
        and math.isfinite(check_by_index[i]["energy_error_eV"])
        and abs(check_by_index[i]["energy_error_eV"]) <= 1e-6
        and isinstance(check_by_index[i].get("fmax_eV_A"), (int, float))
        and math.isfinite(check_by_index[i]["fmax_eV_A"])
        and check_by_index[i]["fmax_eV_A"] <= fmax_limit
        for i in range(len(minima)) if i in check_by_index)
    fresh_ok = bool(minima and len(minima_rows) == len(minima) and len(check_by_index) == len(minima)
        and not missing and not outside and not duplicates and not malformed and fresh_identity
        and not fresh_ledger_errors and fresh_count == summary.get("fresh_requests")
        and fresh_values_ok and summary.get("fresh_checked") == len(minima)
        and fresh_ledger_counts == {"search": len(minima), "search_failure": 0,
                                    "search_denial": 0, "other": 0}
        and fresh_count == summary.get("fresh_requests")
        and summary.get("fresh_all_qualified") is True
        and summary.get("fresh_checked") == len(minima))
    if not fresh_ok:
        errors.append("fresh ledger/check coverage or numerical qualification is incomplete")
    result_full = source == "result.json"
    return {"arm": arm, "seed": seed, "label": f"{arm}-seed{seed}", "errors": errors,
        "result_source": source, "result_completeness": "full" if result_full else source or "missing",
        "result_status": result.get("status"), "summary": summary, "accounting": accounting,
        "minima": minima_rows, "class_entries": class_entries, "outer_records": outer_records,
        "mapping": {"valid": mapping_ok, "errors": mapping_errors,
                    "mapped_minima": len(min_costs)},
        "fresh": {"checked_count": len(checks), "missing_indices": missing,
            "duplicate_indices": sorted(set(duplicates)), "out_of_range_indices": outside,
            "malformed": malformed, "ledger_counts": fresh_ledger_counts,
            "ledger_charged_requests": fresh_count,
            "summary_requests": summary.get("fresh_requests"),
            "summary_checked": summary.get("fresh_checked"),
            "summary_calculator_calls": summary.get("fresh_calculator_calls"),
            "all_qualified": fresh_ok},
        "search_ledger_errors": search_errors, "fresh_ledger_errors": fresh_ledger_errors}


def summarize_arm(row, plan):
    charged = row["accounting"]["ledger_charged_requests"] or 0
    prefix = plan["common_request_prefix"]
    mapping_ok = row["mapping"]["valid"]
    eligible = [m for m in row["minima"] if mapping_ok and m.get("converged") is True and
                isinstance(m.get("cumulative_requests"), int) and m["cumulative_requests"] <= prefix]
    connected = {m["graph_class_id"] for m in eligible if m["graph"]["component_count"] == 1}
    fragmented = {m["graph_class_id"] for m in eligible if m["graph"]["component_count"] > 1}
    torsions = [m["butadiene_torsion"] for m in eligible
                if "region" in m.get("butadiene_torsion", {})]
    prefix_fresh_qualified = sum(
        m.get("fresh", {}).get("record", {}).get("numerical_qualified") is True
        and m.get("fresh", {}).get("identity_matches") is True for m in eligible)
    records = row["outer_records"]
    accepted = sum(r["accepted_converged_landing"] for r in records)
    converged = sum(r["landing_converged"] for r in records)
    prefix_accepts = sum(r["accepted_converged_landing"] for r in records
                         if mapping_ok and isinstance(r.get("minimum_index"), int)
                         and isinstance(row["minima"][r["minimum_index"]].get("cumulative_requests"), int)
                         and row["minima"][r["minimum_index"]]["cumulative_requests"] <= prefix)
    native_actions = [action for r in records
                      for action in (r.get("native_update_actions") or [])]
    first_classes = {}
    if mapping_ok:
        for m in row["minima"]:
            cid, cost = m.get("graph_class_id"), m.get("cumulative_requests")
            if cid is None or not isinstance(cost, int):
                continue
            entry = {"class_id": cid, "first_request": cost,
                     "connected": m["graph"]["component_count"] == 1,
                     "component_formulas": m["component_formulas"],
                     "minimum_index": m["index"]}
            if cid not in first_classes or cost < first_classes[cid]["first_request"]:
                first_classes[cid] = entry
    class_first = sorted(first_classes.values(), key=lambda x: (x["first_request"], x["class_id"]))
    no_converged_landing_costs = [{"index": r["index"], "status": r["status"],
        "evaluation_requests": r["evaluation_requests"], "error": r["error"]}
        for r in records if not r["landing_converged"]]
    boundary = row["summary"].get("search_boundary")
    derived_complete = bool(row["result_source"] == "result.json"
        and row["result_status"] == "completed" and len(records) == plan["steps"])
    reported_complete = row["summary"].get("protocol_completed")
    expected_censored = boundary in ("request_cap", "wall_cap")
    reported_censored = row["summary"].get("budget_censored")
    return {"charged_requests": charged, "prefix": prefix,
        "prefix_reached": charged >= prefix, "prefix_censored": charged < prefix,
        "prefix_returned_minima": len(eligible),
        "prefix_fresh_qualified_minima": prefix_fresh_qualified,
        "prefix_fresh_unqualified_or_missing_minima": len(eligible) - prefix_fresh_qualified,
        "prefix_noninitial_minima": sum(m["index"] > 0 for m in eligible),
        "prefix_accepted_landings": prefix_accepts,
        "prefix_connected_graph_classes": len(connected),
        "prefix_fragmented_graph_classes": len(fragmented),
        "prefix_component_splits": sum(m["fragmented"] for m in eligible),
        "prefix_butadiene_torsion_regions": {
            "positive": sum(t["region"] == "positive" for t in torsions),
            "negative": sum(t["region"] == "negative" for t in torsions),
            "zero": sum(t["region"] == "zero" for t in torsions),
            "raw_deg": [t["raw_deg"] for t in torsions],
            "cosphi": [t["cosphi"] for t in torsions]},
        "endpoint_first_class_costs": class_first,
        "no_converged_landing_costs": no_converged_landing_costs,
        "native_update_action_counts": {name: native_actions.count(name)
            for name in sorted(set(native_actions))},
        "native_update_record_count": sum(bool(r.get("ls_update_raw")) for r in records),
        "ls_response_count": sum(r.get("energy_response_eV_per_atom") is not None for r in records),
        "last_core_response_eV_per_atom": next((r["energy_response_eV_per_atom"]
            for r in reversed(records) if r.get("energy_response_eV_per_atom") is not None), None),
        "last_native_response_meV_per_atom": next((r["native_observed_response_meV_per_atom"]
            for r in reversed(records) if r.get("native_observed_response_meV_per_atom") is not None), None),
        "endpoint": {"protocol_completed_recorded": row["summary"].get("protocol_completed"),
        "protocol_completed_derived_from_result": derived_complete,
            "protocol_completed_consistent": (None if not isinstance(reported_complete, bool)
                                               else reported_complete == derived_complete),
            "budget_censored_recorded": reported_censored,
            "budget_censored_consistent": (None if not isinstance(reported_censored, bool)
                                           else reported_censored == expected_censored),
            "execution_ok_recorded": row["summary"].get("execution_ok"),
            "search_boundary": boundary, "outer_records": row["summary"].get("outer_records"),
            "actual_outer_records": len(records), "returned_minima": len(row["minima"]),
            "converged_landings": converged,
            "accepted_landings": accepted, "search_requests": charged,
            "search_calculator_calls": row["summary"].get("search_calculator_calls"),
            "search_wall_seconds": row["summary"].get("search_elapsed_seconds"),
            "fresh_requests": row["fresh"]["ledger_charged_requests"],
            "fresh_calculator_calls": row["fresh"]["summary_calculator_calls"]}}


def main():
    for name in ("analysis.json", "report.md"):
        if (HERE / name).exists():
            raise FileExistsError(f"refusing to overwrite {name}")
    plan = load(HERE / "plan.json")
    if plan is None:
        raise FileNotFoundError(HERE / "plan.json")
    h = helpers()
    arms = [audit_arm(arm, seed, plan, h)
            for seed in plan["seeds"] for arm in plan["arms"]]
    all_entries = [entry for arm in arms for entry in arm["class_entries"]]
    h[0](all_entries)
    for arm in arms:
        entry_by_index = {e["index"]: e for e in arm["class_entries"]}
        connected, fragmented = set(), set()
        for minimum in arm["minima"]:
            entry = entry_by_index.get(minimum["index"])
            if entry is None:
                continue
            minimum["graph_class_id"] = entry["class_id"]
            if minimum["graph"]["component_count"] == 1:
                connected.add(entry["class_id"])
            else:
                fragmented.add(entry["class_id"])
            minimum.pop("_class_entry", None)
        arm["global_class_scope"] = "one isomorphism mapping across all six arms"
        arm["connected_graph_class_count"] = len(connected)
        arm["fragmented_graph_class_count"] = len(fragmented)
        arm["common_prefix_and_endpoint"] = summarize_arm(arm, plan)
        endpoint = arm["common_prefix_and_endpoint"]["endpoint"]
        if endpoint["protocol_completed_consistent"] is False:
            arm["errors"].append("protocol_completed summary flag disagrees with full result/status/400-record evidence")
        if endpoint["budget_censored_consistent"] is False:
            arm["errors"].append("budget_censored summary flag disagrees with request/wall boundary")
        if endpoint["protocol_completed_recorded"] is not True and endpoint["budget_censored_recorded"] is not True:
            arm["errors"].append("non-budget early termination; protocol is incomplete")
        if arm["summary"].get("execution_ok") is False and endpoint["budget_censored_recorded"] is not True:
            arm["errors"].append("runner did not record execution_ok and arm was not budget-censored")
        del arm["class_entries"]
    bad = [a["label"] for a in arms if a["errors"] or not a["mapping"]["valid"]
           or not a["accounting"]["reconciles"] or not a["fresh"]["all_qualified"]]
    payload = {"scope": "frozen six-arm single-task comparison; no ranking",
        "common_prefix_requests": plan["common_request_prefix"],
        "class_identity": "element-labeled graph isomorphism; mapping shared across all arms",
        "prefix_rule": "only result.minima whose mapped initial+record cumulative request count is <=200000; mapping mismatch excludes prefix claims",
        "torsion_limit": "raw CCCC angle and cos sign are geometric regions, not distinct connectivity, isomer stability, TS, or electronic-state proof",
        "qualification_limit": "fresh force qualification is not Hessian stability or chemical validation; MH-1 is not the paper PBE oracle",
        "issues": [{"arm": a["label"], "issues": a["errors"]} for a in arms if a["errors"]],
        "errors": bad, "arms": arms}
    (HERE / "analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# C4H6 MH-1 fixed-protocol coverage audit", "",
        "No method ranking is computed. The common prefix is fixed at 200000 charged E/F requests; arms below it are censored. One element-labeled graph mapping is shared across all six arms. Class counts include the initial structure; repeated observations do not add classes. Connected and fragmented structures are summarized separately; all per-minimum geometry and per-attempt history is in analysis.json.", "",
        "| Seed | Arm | Prefix reached | Minima in prefix | Connected classes | Fragmented classes | Fragmented frames | Accepted prefix landings | Outer records | Returned minima | Requests | Calculator calls | Search wall (s) | Protocol complete | Budget censored |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for arm in arms:
        p, e = arm["common_prefix_and_endpoint"], arm["common_prefix_and_endpoint"]["endpoint"]
        lines.append(f"| {arm['seed']} | {arm['arm']} | {p['prefix_reached']} | {p['prefix_returned_minima']} | {p['prefix_connected_graph_classes']} | {p['prefix_fragmented_graph_classes']} | {p['prefix_component_splits']} | {p['prefix_accepted_landings']} | {e['actual_outer_records']} | {e['returned_minima']} | {e['search_requests']} | {e['search_calculator_calls']} | {e['search_wall_seconds']} | {e['protocol_completed_recorded']} ({e['protocol_completed_consistent']}) | {e['budget_censored_recorded']} ({e['budget_censored_consistent']}) |")
    for arm in arms:
        p, e = arm["common_prefix_and_endpoint"], arm["common_prefix_and_endpoint"]["endpoint"]
        class_rows = p["endpoint_first_class_costs"]
        conn_first = [x for x in class_rows if x["connected"]][:3]
        frag_first = [x for x in class_rows if not x["connected"]][:3]
        torsions = p["prefix_butadiene_torsion_regions"]
        failures = p["no_converged_landing_costs"]
        failure_sample = failures[:5]
        lines += ["", f"## {arm['label']}", "",
            f"Accounting reconciles={arm['accounting']['reconciles']}; fresh coverage={arm['fresh']['checked_count']}/{len(arm['minima'])}; result source={arm['result_completeness']}; issues={arm['errors']}.",
            f"Endpoint: status={arm['result_status']}, records={e['actual_outer_records']}, returned minima={e['returned_minima']}, converged landings={e['converged_landings']}, accepted converged landings={e['accepted_landings']}, charged requests={e['search_requests']}, calculator calls={e['search_calculator_calls']}, wall={e['search_wall_seconds']} s, protocol_complete={e['protocol_completed_recorded']}, budget_censored={e['budget_censored_recorded']}.",
            f"Common prefix: reached={p['prefix_reached']}, minima={p['prefix_returned_minima']} (noninitial={p['prefix_noninitial_minima']}), fresh-qualified minima={p['prefix_fresh_qualified_minima']}, fresh-unqualified/missing={p['prefix_fresh_unqualified_or_missing_minima']}, connected classes={p['prefix_connected_graph_classes']}, fragmented classes={p['prefix_fragmented_graph_classes']}, fragmented frames={p['prefix_component_splits']}, torsion cos-sign regions +/−/0={torsions['positive']}/{torsions['negative']}/{torsions['zero']}; graph counts include every observed converged landing, including fresh failures; they are not physical-qualification counts. Raw angles and per-frame classes are in JSON.",
            f"First connected class costs (up to 3): {conn_first}; first fragmented class costs (up to 3): {frag_first}.",
            f"No-converged-landing records: {len(failures)}; first records (status and charged cost): {failure_sample}; full list remains in analysis.json.",
            f"LS response observations={p['ls_response_count']}; native update records={p['native_update_record_count']}; core last response={p['last_core_response_eV_per_atom']} eV/atom; native last observed response={p['last_native_response_meV_per_atom']} meV/atom; native action counts={p['native_update_action_counts']}.",
            "Climb stop status, landing convergence, MC acceptance, all attempt costs, and full LS response history remain separate in analysis.json. A force-converged landing is not a Hessian or chemical-stability certificate."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    if bad:
        raise SystemExit(f"analysis written with missing/integrity/mapping issues: {bad}")


if __name__ == "__main__":
    main()
