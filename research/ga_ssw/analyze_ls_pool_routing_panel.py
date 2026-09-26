#!/usr/bin/env python3
"""Offline readout for the frozen six-arm LS pool-routing panel.

Reads worker JSON/JSONL artifacts only. It never creates a calculator or
performs a PES evaluation. Missing, failed, and budget-censored arms remain in
the output and make the exit status nonzero.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import networkx as nx
from ase.io import read as ase_read

DEFAULT_PLAN = Path(
    "/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/"
    "research/ga_ssw/evidence/ls-pool-routing-20260926/plan.json"
)
C60_CUTOFFS = (1.64, 1.70, 1.80)
NODE_MATCH = nx.algorithms.isomorphism.categorical_node_match("number", None)


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def iter_jsonl(path: Path):
    """Stream JSONL so ledgers need not be loaded into memory."""
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{number}: invalid JSONL: {exc}") from exc


def atom_graph(atom_data: dict, cutoff_for_pair=None, scalar_cutoff=None, *, inclusive=True):
    numbers = [int(x) for x in atom_data["numbers"]]
    positions = atom_data["positions"]
    graph = nx.Graph()
    graph.add_nodes_from((i, {"number": number}) for i, number in enumerate(numbers))
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            d2 = sum((float(positions[i][k]) - float(positions[j][k])) ** 2
                     for k in range(3))
            cutoff = scalar_cutoff
            if cutoff_for_pair is not None:
                cutoff = cutoff_for_pair[tuple(sorted((numbers[i], numbers[j])))]
            edge = d2 <= float(cutoff) ** 2 if inclusive else d2 < float(cutoff) ** 2
            if edge:
                graph.add_edge(i, j)
    return graph


def c4h6_graph(atom_data: dict, cutoffs: dict):
    pairs = {
        (1, 1): float(cutoffs.get("1,1", cutoffs.get("(1, 1)"))),
        (1, 6): float(cutoffs.get("1,6", cutoffs.get("(1, 6)"))),
        (6, 6): float(cutoffs.get("6,6", cutoffs.get("(6, 6)"))),
    }
    return atom_graph(atom_data, cutoff_for_pair=pairs)


def c60_graph(atom_data: dict, cutoff: float):
    return atom_graph(atom_data, scalar_cutoff=cutoff, inclusive=False)


def graph_class_id(graph, representatives):
    for index, representative in enumerate(representatives):
        if nx.is_isomorphic(graph, representative, node_match=NODE_MATCH):
            return index, False
    representatives.append(graph)
    return len(representatives) - 1, True


def c60_graph_metrics(graph, ih_graph, defect_graph):
    degrees = Counter(dict(graph.degree()).values())
    planar, embedding = nx.check_planarity(graph)
    faces = []
    if planar:
        seen = set()
        for u, v in embedding.edges():
            if (u, v) not in seen:
                faces.append(embedding.traverse_face(u, v, seen))
    face_counts = Counter(map(len, faces))
    connected = nx.is_connected(graph) if graph.number_of_nodes() else False
    fullerene_candidate = bool(
        graph.number_of_nodes() == 60 and graph.number_of_edges() == 90
        and degrees == Counter({3: 60}) and connected and planar
        and face_counts == Counter({5: 12, 6: 20})
        and nx.node_connectivity(graph) >= 3
    )
    return {
        "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "components": nx.number_connected_components(graph),
        "degree_3_all": degrees == Counter({3: 60}),
        "fullerene_cage_candidate": fullerene_candidate,
        "ih_graph_match": nx.is_isomorphic(graph, ih_graph),
        "source_defect_graph_match": nx.is_isomorphic(graph, defect_graph),
    }


def charged_ledger(path: Path, *, stage: str):
    rows = 0
    charged_events = 0
    events = Counter()
    start_ids = []
    paid_ids = set()
    started_attempts = set()
    terminal_attempts = set()
    errors = []
    try:
        for row in iter_jsonl(path):
            rows += 1
            event = row.get("event")
            events[str(event)] += 1
            if row.get("stage") not in (None, stage):
                errors.append(f"unexpected {stage} ledger stage {row.get('stage')!r}")
            if event == "attempt_started":
                started_attempts.add(row.get("attempt"))
            if event in ("attempt_completed", "attempt_error", "budget_censor"):
                terminal_attempts.add(row.get("attempt"))
            if event in ("attempt_completed", "attempt_error") and row.get("charged") is True:
                charged_events += 1
                ident = row.get("request")
                if ident is not None:
                    paid_ids.add(ident)
                    start_ids.append(ident)
            # Failed calculator calls are paid if charged=true; denied requests are not.
    except ValueError as exc:
        # Keep the valid prefix as a lower bound for interrupted/truncated files.
        # The parse error invalidates exact accounting and therefore prefix use.
        errors.append(str(exc))
    if len(paid_ids) != charged_events or len(start_ids) != charged_events:
        errors.append(f"paid request ids ({len(paid_ids)} unique, {len(start_ids)} rows) != charged terminal events {charged_events}")
    paid = charged_events
    monotonic = all(
        isinstance(b, (int, float)) and isinstance(a, (int, float)) and b > a
        for a, b in zip(start_ids, start_ids[1:])
    )
    if start_ids and not monotonic:
        errors.append(f"paid {stage} request identifiers are not strictly increasing")
    incomplete_attempts = sorted(
        (attempt for attempt in started_attempts - terminal_attempts if attempt is not None),
        key=str)
    if incomplete_attempts:
        errors.append(f"{len(incomplete_attempts)} started attempts have no terminal ledger row; charge is unknown")
    return {"rows": rows, "events": dict(events), "charged_requests": paid,
            "charged_event_count": charged_events,
            "paid_request_id_count": len(start_ids),
            "incomplete_attempts": incomplete_attempts,
            "request_ids_strictly_increasing": monotonic, "errors": errors}


def progress_tail(path: Path):
    """Return only the last valid progress event without loading the file."""
    if not path.is_file():
        return None
    last = None
    try:
        for row in iter_jsonl(path):
            last = row
    except ValueError:
        # Progress is diagnostic only; a truncated final line must not hide
        # the last fully written status event.
        pass
    return last


def shared_prefix(rows, modes):
    """Use only arms with closed result/ledger and full fresh coverage."""
    by_mode = {row.get("mode"): row for row in rows}
    selected = [by_mode.get(mode) for mode in modes]
    if any(row is None or row.get("scientific_prefix_eligible") is not True
           or not isinstance(row.get("search_requests"), int) for row in selected):
        return None
    return min(row["search_requests"] for row in selected)


def candidate_costs(payload: dict):
    """Map initial and every non-null landing to cumulative paid search cost."""
    result = payload.get("result", {})
    initial = result.get("initial") or {}
    records = result.get("records") or []
    cumulative = int(initial.get("evaluation_requests", 0) or 0)
    costs = {0: cumulative}
    minimum_by_candidate = {0: 0}
    record_by_candidate = {0: -1}
    candidate_index = 1
    next_minimum = 1
    no_landing = 0
    landing_count = 0
    converged_landing_count = 0
    nonconverged_landing_count = 0
    status_counts = Counter()
    record_costs = []
    for expected_index, record in enumerate(records):
        if record.get("index") != expected_index:
            raise ValueError(f"record index mismatch at {expected_index}")
        step_cost = int(record.get("evaluation_requests", 0) or 0)
        if step_cost < 0:
            raise ValueError(f"negative record cost at {expected_index}")
        cumulative += step_cost
        status_counts[str(record.get("status"))] += 1
        landing = record.get("landing")
        if isinstance(landing, dict):
            costs[candidate_index] = cumulative
            record_by_candidate[candidate_index] = expected_index
            if landing.get("converged") is True:
                minimum_by_candidate[candidate_index] = next_minimum
                next_minimum += 1
                converged_landing_count += 1
            else:
                minimum_by_candidate[candidate_index] = None
                nonconverged_landing_count += 1
            candidate_index += 1
            landing_count += 1
        else:
            no_landing += 1
        record_costs.append(cumulative)
    minima = result.get("minima") or []
    if next_minimum != len(minima):
        raise ValueError(
            f"converged landing/minima order mismatch: mapped={next_minimum}, minima={len(minima)}"
        )
    result_total = result.get("evaluation_requests")
    if result_total is not None and cumulative != int(result_total):
        raise ValueError(f"initial+record cost {cumulative} != result total {result_total}")
    wrapper_total = payload.get("search_requests")
    if wrapper_total is not None and cumulative != int(wrapper_total):
        raise ValueError(f"result cost {cumulative} != wrapper search_requests {wrapper_total}")
    return {"candidate_costs": costs, "minimum_by_candidate": minimum_by_candidate,
            "record_by_candidate": record_by_candidate,
            "record_cumulative_costs": record_costs,
            "landing_count": landing_count, "no_landing_count": no_landing,
            "converged_landing_count": converged_landing_count,
            "nonconverged_landing_count": nonconverged_landing_count,
            "record_status_counts": dict(status_counts), "record_count": len(records),
            "candidate_count": candidate_index,
            "search_requests_from_result": cumulative}


def load_arm(folder: Path, case: dict, mode: str, plan: dict):
    row = {"case": case["name"], "mode": mode, "path": str(folder),
           "present": folder.is_dir(), "issues": []}
    identity_file = "offline-identity.json" if mode == "mc" else "pool-report.json"
    needed = ["summary.json", "search-result.json", "fresh-checks.json",
              "search-ledger.jsonl", "fresh-ledger.jsonl", "selector-contract.json"]
    if mode != "mc":
        needed.append(identity_file)
    missing = [name for name in needed if not (folder / name).is_file()]
    if missing:
        row["missing_artifacts"] = missing
        row["issues"].append("missing artifacts: " + ", ".join(missing))
    identity_present = (folder / identity_file).is_file()
    identity_status = (
        "not_applicable" if mode != "mc" else
        "present" if identity_present else "missing_optional"
    )
    if not (folder / "summary.json").is_file():
        ledger = charged_ledger(folder / "search-ledger.jsonl", stage="search") \
            if (folder / "search-ledger.jsonl").is_file() else None
        fresh_ledger = charged_ledger(folder / "fresh-ledger.jsonl", stage="fresh") \
            if (folder / "fresh-ledger.jsonl").is_file() else None
        wrapped = {}
        if (folder / "search-result.json").is_file():
            try:
                wrapped = read_json(folder / "search-result.json")
            except Exception as exc:
                row["issues"].append(
                    f"partial search-result.json unreadable: {type(exc).__name__}: {exc}")
        result = wrapped.get("result", {}) if isinstance(wrapped, dict) else {}
        search_lb = None if ledger is None else ledger["charged_requests"]
        fresh_lb = None if fresh_ledger is None else fresh_ledger["charged_requests"]
        record_count = len(result.get("records", [])) if isinstance(result, dict) else None
        row.update(status="missing_summary", algorithm_status=wrapped.get("status"),
                   search_requests=None, search_requests_lower_bound=search_lb,
                   fresh_requests=None, fresh_requests_lower_bound=fresh_lb,
                   total_requests=None,
                   total_requests_lower_bound=(None if search_lb is None and fresh_lb is None
                                               else (search_lb or 0) + (fresh_lb or 0)),
                   outer_records=record_count, search_ledger=ledger, fresh_ledger=fresh_ledger,
                   progress_last=progress_tail(folder / "progress.jsonl"),
                   offline_identity_diagnostic=identity_status,
                   pool_report_entries=None,
                   scientific_prefix_eligible=False)
        if ledger is not None:
            row["issues"].extend(ledger["errors"])
        if fresh_ledger is not None:
            row["issues"].extend(fresh_ledger["errors"])
        return row
    try:
        summary = read_json(folder / "summary.json")
        wrapped = read_json(folder / "search-result.json") if (folder / "search-result.json").is_file() else {}
        fresh = read_json(folder / "fresh-checks.json") if (folder / "fresh-checks.json").is_file() else {}
        pool = read_json(folder / identity_file) if (folder / identity_file).is_file() else {}
        contract = read_json(folder / "selector-contract.json") if (folder / "selector-contract.json").is_file() else {}
        ledger = charged_ledger(folder / "search-ledger.jsonl", stage="search") if (folder / "search-ledger.jsonl").is_file() else None
        fresh_ledger = charged_ledger(folder / "fresh-ledger.jsonl", stage="fresh") if (folder / "fresh-ledger.jsonl").is_file() else None
        if ledger is None:
            row["issues"].append("search ledger unavailable; charged search cost cannot be independently checked")
        if fresh_ledger is None:
            row["issues"].append("fresh ledger unavailable; fresh cost cannot be independently checked")
        if not wrapped or not fresh:
            partial_costs = None
            if wrapped.get("result"):
                try:
                    partial_costs = candidate_costs(wrapped)
                    if partial_costs["search_requests_from_result"] != int(summary.get("search_requests", -1)):
                        row["issues"].append("partial search-result cost differs from summary")
                except Exception as exc:
                    row["issues"].append(f"partial result accounting error: {type(exc).__name__}: {exc}")
            if ledger is not None and summary.get("search_requests") is not None \
                    and int(summary["search_requests"]) != ledger["charged_requests"]:
                row["issues"].append("partial search summary cost differs from ledger")
            row.update({"status": summary.get("status", "partial"),
                        "algorithm_status": summary.get("algorithm_status"),
                        "budget_censor": bool(summary.get("budget_censor", False)),
                        "censor_reason": summary.get("censor_reason"),
                        "outer_records": summary.get("outer_attempt_records",
                                                      None if partial_costs is None else partial_costs["record_count"]),
                        "landing_events": summary.get("returned_landings",
                                                       None if partial_costs is None else partial_costs["landing_count"]),
                        "converged_landing_events": None if partial_costs is None else partial_costs["converged_landing_count"],
                        "nonconverged_landing_events": None if partial_costs is None else partial_costs["nonconverged_landing_count"],
                        "outer_records_without_landing": None if partial_costs is None else partial_costs["no_landing_count"],
                        "search_requests": summary.get("search_requests"),
                        "fresh_requests": summary.get("fresh_requests"),
                        "total_requests": summary.get("total_requests"),
                        "search_ledger": ledger,
                        "fresh_ledger": fresh_ledger,
                        "fresh_certified": summary.get("fresh_qualified_count"),
                        "fresh_failed_or_uncertified": None,
                        "fresh_missing": None,
                        "committed_pool_restarts": [],
                        "offline_identity_diagnostic": identity_status,
                        "pool_report_entries": (None if mode == "mc" and not identity_present
                                                 else len(pool.get("entries", []))),
                        "pool_report_cost_closed": pool.get("cost_closed"),
                        "selector_contract": contract,
                        "scientific_prefix_eligible": False})
            return row
        result = wrapped.get("result")
        if not isinstance(result, dict):
            raise ValueError("search-result.json missing serialized result wrapper.result")
        minima = result.get("minima") or []
        costs = candidate_costs(wrapped)
        checks = fresh.get("checks")
        if not isinstance(checks, list):
            raise ValueError("fresh-checks.json checks is not a list")
        if int(fresh.get("candidate_count", -1)) != costs["candidate_count"]:
            row["issues"].append(
                f"fresh candidate_count={fresh.get('candidate_count')} != initial+non-null-landings={costs['candidate_count']}"
            )
        if summary.get("fresh_candidate_count") is not None and int(
                summary["fresh_candidate_count"]) != costs["candidate_count"]:
            row["issues"].append(
                f"summary fresh_candidate_count={summary['fresh_candidate_count']} != derived {costs['candidate_count']}"
            )
        if summary.get("fresh_checks_recorded") is not None and int(
                summary["fresh_checks_recorded"]) != len(checks):
            row["issues"].append("summary fresh_checks_recorded differs from fresh-check rows")
        check_by_index = {}
        duplicate_fresh_indices = []
        for check in checks:
            index = check.get("candidate_index")
            if index in check_by_index:
                duplicate_fresh_indices.append(index)
            check_by_index[index] = check
        if duplicate_fresh_indices:
            row["issues"].append(f"duplicate fresh minimum indices: {duplicate_fresh_indices}")
        expected_indices = set(range(costs["candidate_count"]))
        missing_fresh = sorted(expected_indices - set(check_by_index))
        extra_fresh = sorted(set(check_by_index) - expected_indices, key=str)
        if missing_fresh or extra_fresh:
            row["issues"].append(f"fresh coverage mismatch missing={missing_fresh}, extra={extra_fresh}")

        search_n = int(summary.get("search_requests", -1))
        fresh_n = int(summary.get("fresh_requests", -1))
        result_n = int(wrapped.get("search_requests", costs["search_requests_from_result"]))
        if search_n != result_n or (ledger is not None and search_n != ledger["charged_requests"]):
            row["issues"].append(
                f"search requests disagree summary={search_n}, result={result_n}, ledger={None if ledger is None else ledger['charged_requests']}"
            )
        fresh_reported = int(fresh.get("fresh_requests", fresh_n))
        if fresh_n != fresh_reported or (fresh_ledger is not None and fresh_n != fresh_ledger["charged_requests"]):
            row["issues"].append(
                f"fresh requests disagree summary={fresh_n}, checks={fresh_reported}, ledger={None if fresh_ledger is None else fresh_ledger['charged_requests']}"
            )
        if summary.get("total_requests") is not None and int(summary["total_requests"]) != search_n + fresh_n:
            row["issues"].append("summary total_requests differs from search+fresh charged requests")
        total_accounting_valid = (
            summary.get("total_requests") is None
            or int(summary["total_requests"]) == search_n + fresh_n
        )
        for key, actual in (("outer_attempt_records", costs["record_count"]),
                            ("returned_landings", costs["landing_count"])):
            if summary.get(key) is not None and int(summary[key]) != actual:
                row["issues"].append(f"summary {key}={summary[key]} differs from derived {actual}")
        expected_steps = int(plan.get("steps", 100))
        if costs["record_count"] != expected_steps:
            row["issues"].append(
                f"outer record count {costs['record_count']} differs from protocol steps {expected_steps}"
            )
        for entry in (ledger or {}).get("errors", []) + (fresh_ledger or {}).get("errors", []):
            row["issues"].append(entry)
        search_accounting_valid = bool(
            ledger is not None and not ledger["errors"]
            and search_n == result_n == ledger["charged_requests"]
        )
        fresh_accounting_valid = bool(
            fresh_ledger is not None and not fresh_ledger["errors"]
            and fresh_n == fresh_reported == fresh_ledger["charged_requests"]
            and not missing_fresh and not extra_fresh
            and int(fresh.get("candidate_count", -1)) == costs["candidate_count"]
            and (summary.get("fresh_candidate_count") is None
                 or int(summary["fresh_candidate_count"]) == costs["candidate_count"])
            and (summary.get("fresh_checks_recorded") is None
                 or int(summary["fresh_checks_recorded"]) == len(checks))
        )

        certified = []
        certified_nonconverged = []
        failed = []
        for index in sorted(expected_indices):
            check = check_by_index.get(index)
            if check is None:
                continue
            if check.get("certified") is True:
                certified.append((index, check))
                if check.get("converged") is not True:
                    certified_nonconverged.append((index, check))
            else:
                failed.append((index, check))
        if fresh.get("qualified_count") is not None and int(fresh["qualified_count"]) != len(certified):
            row["issues"].append("fresh-checks qualified_count differs from certified row count")
        if summary.get("fresh_qualified_count") is not None and int(
                summary["fresh_qualified_count"]) != len(certified):
            row["issues"].append("summary fresh_qualified_count differs from certified row count")
        fresh_accounting_valid = bool(
            fresh_accounting_valid and total_accounting_valid
            and (fresh.get("qualified_count") is None or int(fresh["qualified_count"]) == len(certified))
            and (summary.get("fresh_qualified_count") is None
                 or int(summary["fresh_qualified_count"]) == len(certified))
        )
        records = result.get("records") or []
        committed = []
        for record in records:
            selection = record.get("starter_selection") or {}
            if selection.get("restarted") is True and selection.get("restart_failed") is not True:
                committed.append({"record_index": record.get("index"),
                                  "chosen_index": selection.get("chosen_index"),
                                  "mc_current_index": selection.get("mc_current_index"),
                                  "cost": selection.get("cost")})
        if summary.get("committed_restart_count") is not None and int(
                summary["committed_restart_count"]) != len(committed):
            row["issues"].append("summary committed restart count differs from result records")

        row.update({
            "status": summary.get("status", wrapped.get("status", "status_missing")),
            "algorithm_status": wrapped.get("algorithm_status", summary.get("algorithm_status")),
            "budget_censor": bool(wrapped.get("budget_censor", summary.get("budget_censor", False))),
            "censor_reason": wrapped.get("censor_reason", summary.get("censor_reason")),
            "outer_records": costs["record_count"],
            "outer_record_status_counts": costs["record_status_counts"],
            "landing_events": costs["landing_count"],
            "converged_landing_events": costs["converged_landing_count"],
            "nonconverged_landing_events": costs["nonconverged_landing_count"],
            "outer_records_without_landing": costs["no_landing_count"],
            "search_requests": search_n,
            "fresh_requests": fresh_n,
            "total_requests": search_n + fresh_n,
            "search_ledger": ledger,
            "fresh_ledger": fresh_ledger,
            "candidate_denominator": costs["candidate_count"],
            "fresh_checks_present": len(checks),
            "fresh_checks_recorded": summary.get("fresh_checks_recorded", len(checks)),
            "fresh_certified": len(certified),
            "fresh_certified_nonconverged": len(certified_nonconverged),
            "fresh_converged_certified_minima": len(certified) - len(certified_nonconverged),
            "fresh_failed_or_uncertified": len(failed),
            "fresh_missing": missing_fresh,
            "fresh_failure_indices": [i for i, _ in failed],
            "committed_pool_restarts": committed,
            "offline_identity_diagnostic": identity_status,
            "pool_report_entries": (None if mode == "mc" and not identity_present
                                     else len(pool.get("entries", []))),
            "pool_report_cost_closed": pool.get("cost_closed"),
            "selector_contract": contract,
            "search_accounting_valid": search_accounting_valid,
            "fresh_accounting_valid": fresh_accounting_valid,
            "scientific_prefix_eligible": search_accounting_valid and fresh_accounting_valid,
            "_payload": wrapped,
            "_fresh_by_index": check_by_index,
            "_costs": costs,
            "_pool": pool,
        })
    except Exception as exc:
        row.update(status="analysis_error", error=f"{type(exc).__name__}: {exc}")
        row["issues"].append(row["error"])
    return row


def analyze_prefix(row: dict, case: dict, prefix: int, ih_graphs=None, defect_graphs=None):
    payload = row.get("_payload")
    if payload is None:
        return {"prefix_requests": prefix, "status": "unavailable"}
    result = payload["result"]
    cutoffs = case.get("graph_cutoffs_A", {})
    is_c4 = case["name"] == "C4H6"
    candidates = []
    for index, check in row["_fresh_by_index"].items():
        if not isinstance(index, int) or index not in row["_costs"]["candidate_costs"]:
            continue
        cost = row["_costs"]["candidate_costs"][index]
        if cost > prefix or check.get("certified") is not True:
            continue
        if check.get("converged") is not True:
            continue
        atoms = check.get("atoms")
        if not isinstance(atoms, dict):
            minimum_index = row["_costs"]["minimum_by_candidate"].get(index)
            minima = result.get("minima", [])
            if minimum_index is None or minimum_index >= len(minima):
                continue
            atoms = minima[minimum_index].get("atoms")
        if not isinstance(atoms, dict):
            continue
        energy = check.get("fresh_energy_eV", check.get("energy_eV"))
        minimum_index = row["_costs"]["minimum_by_candidate"].get(index)
        if energy is None and minimum_index is not None and minimum_index < len(result.get("minima", [])):
            energy = result["minima"][minimum_index].get("energy")
        candidate = {"candidate_index": index, "minimum_index": minimum_index,
                     "record_index": check.get("record_index"), "search_cost": cost,
                     "energy_eV": None if energy is None else float(energy), "atoms": atoms}
        if is_c4:
            graph = c4h6_graph(atoms, cutoffs)
            candidate["components"] = nx.number_connected_components(graph)
            candidate["graph"] = graph
        else:
            graph_rows = {}
            for cutoff in C60_CUTOFFS:
                graph = c60_graph(atoms, cutoff)
                graph_rows[str(cutoff)] = c60_graph_metrics(
                    graph, ih_graphs[cutoff], defect_graphs[cutoff])
            candidate["graphs"] = graph_rows
        candidates.append(candidate)

    output = {"prefix_requests": prefix,
              "fresh_converged_certified_minima_in_prefix": len(candidates),
              "best_fresh_energy_eV": min(
                  (c["energy_eV"] for c in candidates if c["energy_eV"] is not None),
                  default=None)}
    if is_c4:
        # The starting graph is a reference even if its independent fresh
        # certificate failed; the failed check remains in the denominator.
        initial_check = row["_fresh_by_index"].get(0)
        initial_atoms = initial_check.get("atoms") if initial_check else None
        initial_graph = c4h6_graph(initial_atoms, cutoffs) if isinstance(initial_atoms, dict) else None
        representatives = [initial_graph] if initial_graph is not None else []
        connected_representatives = (
            [initial_graph] if initial_graph is not None and nx.number_connected_components(initial_graph) == 1 else []
        )
        fragmented_representatives = (
            [initial_graph] if initial_graph is not None and nx.number_connected_components(initial_graph) > 1 else []
        )
        new_connected_count = 0
        new_fragmented_count = 0
        connected = fragmented = 0
        class_rows = []
        for candidate in candidates:
            graph = candidate["graph"]
            if candidate["components"] == 1:
                connected += 1
            else:
                fragmented += 1
            class_id, is_new = graph_class_id(graph, representatives)
            if candidate["components"] == 1:
                connected_class_id, connected_is_new = graph_class_id(graph, connected_representatives)
                fragmented_class_id = None
                if connected_is_new and candidate["candidate_index"] != 0:
                    new_connected_count += 1
            else:
                connected_class_id = None
                fragmented_class_id, fragmented_is_new = graph_class_id(graph, fragmented_representatives)
                if fragmented_is_new and candidate["candidate_index"] != 0:
                    new_fragmented_count += 1
            class_rows.append({"candidate_index": candidate["candidate_index"],
                               "minimum_index": candidate["minimum_index"],
                               "record_index": candidate["record_index"],
                               "class_id": class_id,
                               "first_observation_in_prefix": is_new,
                               "connected_class_id": connected_class_id,
                               "fragmented_class_id": fragmented_class_id,
                               "components": candidate["components"],
                               "search_cost": candidate["search_cost"]})
        output.update({"c4h6_topology_classes_including_initial": len(representatives),
                       "c4h6_connected_topology_classes": len(connected_representatives),
                       "c4h6_fragmented_graph_classes": len(fragmented_representatives),
                       "c4h6_noninitial_connected_new_topology_classes": new_connected_count,
                       "c4h6_fragmented_new_graph_classes": new_fragmented_count,
                       "connected_certified_observations": connected,
                       "fragmented_certified_observations": fragmented,
                       "topology_observations": class_rows})
    else:
        graph_summary = {}
        for cutoff in C60_CUTOFFS:
            key = str(cutoff)
            observations = [c for c in candidates if key in c["graphs"]]
            graph_summary[key] = {
                "cage_candidates": sum(c["graphs"][key]["fullerene_cage_candidate"] for c in observations),
                "ih_graph_matches": sum(c["graphs"][key]["ih_graph_match"] for c in observations),
                "source_defect_graph_matches": sum(c["graphs"][key]["source_defect_graph_match"] for c in observations),
                "observations": [{"candidate_index": c["candidate_index"],
                                  "minimum_index": c["minimum_index"],
                                  **c["graphs"][key]} for c in observations],
            }
        reference = case.get("references", {})
        ih_energy = reference.get("ih_energy_eV")
        window = reference.get("energy_window_eV")
        for candidate in candidates:
            candidate["energy_window_met"] = bool(
                ih_energy is not None and window is not None and candidate["energy_eV"] is not None
                and candidate["energy_eV"] <= float(ih_energy) + float(window)
            )
        output.update({"c60_graphs_by_cutoff": graph_summary,
                       "ih_energy_reference_eV": ih_energy,
                       "energy_window_eV": window,
                       "ih_energy_window_observations": sum(
                           c.get("energy_window_met", False) for c in candidates),
                       "energy_window_is_separate_from_graph_match": True})
    # Descriptive linkage only: a restart changes subsequent proposals and does
    # not isolate a causal effect.
    record_to_candidate = {record_index: candidate_index for candidate_index, record_index
                           in row["_costs"]["record_by_candidate"].items()
                           if record_index >= 0}
    observed = {candidate["candidate_index"]: candidate for candidate in candidates}
    followups = []
    for restart in row.get("committed_pool_restarts", []):
        current_record = restart.get("record_index")
        following_record = None if current_record is None else int(current_record) + 1
        candidate_index = record_to_candidate.get(following_record)
        candidate = observed.get(candidate_index)
        item = {"restart_after_record": current_record,
                "chosen_pool_index": restart.get("chosen_index"),
                "following_record_index": following_record,
                "following_landing_in_prefix": candidate is not None}
        if candidate is not None:
            item["following_landing_fresh_certified"] = True
            if is_c4:
                topo = next((x for x in output["topology_observations"]
                             if x["candidate_index"] == candidate_index), None)
                item["following_c4_topology_class"] = None if topo is None else topo["class_id"]
                item["following_new_topology_observation"] = bool(
                    topo is not None and topo["first_observation_in_prefix"])
            else:
                item["following_c60_graphs_by_cutoff"] = candidate["graphs"]
                item["following_ih_energy_window_met"] = candidate.get("energy_window_met")
        else:
            check = row["_fresh_by_index"].get(candidate_index) if candidate_index is not None else None
            item["following_landing_fresh_certified"] = bool(check and check.get("certified") is True)
        followups.append(item)
    output["committed_restart_followups_descriptive"] = followups
    return output


def render_report(analysis: dict):
    lines = ["# LS pool-routing panel: offline analysis", "",
             "This report reads archived result, fresh-check, and request-ledger files only; it performs no PES evaluations.",
             "Candidate counts are diagnostics. Scientific readout uses fresh-certified observations and paid search E/F cost.", "",
             "## Arm summary", "",
             "| Case | Mode | Status | Censored | Outer records | Landings conv / nonconv / no landing | Fresh certified minima / cert nonconv / failed / missing | Search / fresh paid E/F | Committed restarts | Issues |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["arms"]:
        landing_counts = "{}/{}/{}".format(
            row.get("converged_landing_events", "?"),
            row.get("nonconverged_landing_events", "?"),
            row.get("outer_records_without_landing", "?"))
        search_cost = row.get("search_requests")
        if search_cost is None and row.get("search_requests_lower_bound") is not None:
            search_cost = f"≥{row['search_requests_lower_bound']} lower bound"
        if search_cost is None:
            search_cost = "unknown"
        fresh_cost = row.get("fresh_requests")
        if fresh_cost is None and row.get("fresh_requests_lower_bound") is not None:
            fresh_cost = f"≥{row['fresh_requests_lower_bound']} lower bound"
        if fresh_cost is None:
            fresh_cost = "unknown"
        lines.append("| {case} | {mode} | {status} | {censored} | {records} | {landing_counts} | {cert_min} / {cert_nonconv} / {failed} / {missing} | {search} / {fresh} | {restarts} | {issues} |".format(
            case=row.get("case", "?"), mode=row.get("mode", "?"),
            status=row.get("status", "missing"), censored=row.get("budget_censor", "?"),
            records=row.get("outer_records", "?"), landing_counts=landing_counts,
            cert_min=row.get("fresh_converged_certified_minima", "?"),
            cert_nonconv=row.get("fresh_certified_nonconverged", "?"),
            failed=row.get("fresh_failed_or_uncertified", "?"),
            missing=(len(row["fresh_missing"]) if isinstance(row.get("fresh_missing"), list)
                     else "unknown"),
            search=search_cost, fresh=fresh_cost,
            restarts=len(row.get("committed_pool_restarts", [])) if "committed_pool_restarts" in row else "?",
            issues="; ".join(row.get("issues", [])) or "—"))
    lines += ["", "## Common paid-search prefixes", "",
              "An exact common endpoint is shown only when all three modes have closed search/result and fresh accounting with full fresh-check coverage. Ledger-only amounts are lower bounds and are never used as exact prefixes.", ""]
    for case_name, data in analysis["cases"].items():
        lines.append(f"### {case_name}")
        lines.append("")
        lines.append(f"Common search prefix: {data.get('common_search_prefix_requests', 'unavailable')} E/F requests.")
        if data.get("excluded_from_exact_prefix"):
            lines.append(f"Excluded modes: {', '.join(data['excluded_from_exact_prefix'])} (accounting or fresh coverage incomplete).")
        lines.append("")
        for mode, prefix in data.get("prefix_readouts", {}).items():
            if "c4h6_noninitial_connected_new_topology_classes" in prefix:
                compact = {key: prefix.get(key) for key in (
                    "prefix_requests", "fresh_converged_certified_minima_in_prefix",
                    "best_fresh_energy_eV", "c4h6_noninitial_connected_new_topology_classes",
                    "c4h6_fragmented_new_graph_classes",
                    "connected_certified_observations", "fragmented_certified_observations")}
            elif "c60_graphs_by_cutoff" in prefix:
                compact = {key: prefix.get(key) for key in (
                    "prefix_requests", "fresh_converged_certified_minima_in_prefix",
                    "best_fresh_energy_eV", "ih_energy_window_observations", "energy_window_eV")}
                compact["c60_graphs_by_cutoff"] = {
                    cutoff: {metric: value.get(metric) for metric in (
                        "cage_candidates", "ih_graph_matches", "source_defect_graph_matches")}
                    for cutoff, value in prefix["c60_graphs_by_cutoff"].items()}
            else:
                compact = prefix
            lines.append(f"- **{mode}:** {json.dumps(compact, ensure_ascii=False, sort_keys=True)}")
        lines.append("")
    lines += ["## Interpretation limits", "",
              "Committed restarts and their following landing are descriptive associations, not isolated causal effects.",
              "Graph isomorphism classes are topology summaries, not geometrical minima, reaction pathways, barriers, or proof of chemical stability.",
              "C60 cage/Ih graph matches and the IH energy window are separate criteria. This panel is a one-seed-per-case developer screen, not a population ranking.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True,
                        help="root containing <case>/<mode> arm directories")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory for analysis.json and report.md")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    plan = read_json(args.plan)
    arms = []
    for case in plan["cases"]:
        case_name = case["name"]
        for mode in plan["methods"]:
            arms.append(load_arm(args.evidence_root / case_name / mode, case, mode, plan))

    references = {}
    c60_case = next((c for c in plan["cases"] if c["name"] == "C60-isomer2"), None)
    if c60_case:
        refs = c60_case.get("references", {})
        ih_path, defect_path = refs.get("ih_graph_input"), refs.get("source_defect_reference_input")
        try:
            ih_atoms = ase_read(ih_path)
            defect_atoms = ase_read(defect_path)
            def serial_atoms(atoms):
                return {"numbers": [int(x) for x in atoms.numbers],
                        "positions": [[float(v) for v in xyz] for xyz in atoms.positions]}
            references = {"ih": {cutoff: c60_graph(serial_atoms(ih_atoms), cutoff)
                                  for cutoff in C60_CUTOFFS},
                          "defect": {cutoff: c60_graph(serial_atoms(defect_atoms), cutoff)
                                     for cutoff in C60_CUTOFFS}}
        except Exception as exc:
            references = {"error": f"{type(exc).__name__}: {exc}"}

    cases_out = {}
    for case in plan["cases"]:
        name = case["name"]
        selected = [row for row in arms if row.get("case") == name]
        common = shared_prefix(selected, plan["methods"])
        excluded = [row.get("mode") for row in selected
                    if row.get("scientific_prefix_eligible") is not True]
        case_out = {"common_search_prefix_requests": common,
                    "excluded_from_exact_prefix": excluded,
                    "prefix_readouts": {}}
        for row in selected:
            if common is None:
                readout = {"status": "unavailable; at least one arm lacks closed search/fresh ledgers, result accounting, or complete fresh coverage"}
            elif name == "C60-isomer2" and "error" in references:
                readout = {"status": "reference_error", "error": references["error"]}
            else:
                readout = analyze_prefix(
                    row, case, common,
                    ih_graphs=references.get("ih") if name == "C60-isomer2" else None,
                    defect_graphs=references.get("defect") if name == "C60-isomer2" else None)
            case_out["prefix_readouts"][row["mode"]] = readout
        cases_out[name] = case_out

    issues = []
    if "error" in references:
        issues.append(f"C60 reference graph construction failed: {references['error']}")
    for row in arms:
        for issue in row.get("issues", []):
            issues.append(f"{row.get('case')}/{row.get('mode')}: {issue}")
        if row.get("budget_censor"):
            issues.append(f"{row.get('case')}/{row.get('mode')}: budget censored ({row.get('censor_reason')})")
        status = row.get("status")
        if status not in ("completed", "complete", "ok"):
            issues.append(f"{row.get('case')}/{row.get('mode')}: non-complete status {status}")
    if len(arms) != len(plan["cases"]) * len(plan["methods"]):
        issues.append("arm count differs from the six-arm protocol")
    analysis = {
        "status": "offline_analysis_with_issues" if issues else "offline_analysis_complete",
        "pes_evaluations": 0,
        "plan": str(args.plan), "evidence_root": str(args.evidence_root),
        "expected_arms": len(plan["cases"]) * len(plan["methods"]),
        "observed_arms": len(arms),
        "arms": [{key: value for key, value in row.items() if not key.startswith("_")}
                 for row in arms], "cases": cases_out,
        "issues": issues,
        "scope": "one seed per case developer screen; no controller ranking or global-search claim",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(render_report(analysis))
    print(json.dumps({"status": analysis["status"], "arms": len(arms),
                      "issues": len(issues), "output_dir": str(args.output_dir)}, indent=2))
    return 2 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
