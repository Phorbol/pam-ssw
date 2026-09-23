"""Offline accounting and geometry audit for the bounded C4H6 lifecycle pilot."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def jsonl(path):
    return ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.is_file() else None)


def helpers():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "research" / "ga_ssw"))
    from analyze_c4h6_ls_reaction_coverage import (
        assign_global_classes, atoms_from_dict, component_formulas, graph,
        graph_label, graph_signature,
    )
    from ase.collections import g2
    references = {name: graph(g2[name].copy()) for name in
                  ("butadiene", "cyclobutene", "2-butyne", "methylenecyclopropane", "bicyclobutane")}
    path = HERE.parent / "c4h6-torsion-audit-20260924" / "analyze.py"
    spec = importlib.util.spec_from_file_location("c4h6_torsion_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load torsion helpers: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return assign_global_classes, atoms_from_dict, component_formulas, graph, graph_label, graph_signature, mod.torsion, references


def analyze_arm(arm, plan, h):
    assign, atoms_from_dict, formulas, graph, graph_label, signature, torsion, references = h
    folder = HERE / arm
    summary = load(folder / "summary.json")
    root_summary = load(HERE / "summary.json") or {}
    if summary is None:
        summary = next((r for r in root_summary.get("rows", []) if r.get("arm") == arm), None)
    source, result = next(((name, load(folder / name)) for name in
                           ("result.json", "partial-exception-result.json", "checkpoint.json")
                           if (folder / name).is_file()), (None, None))
    search, fresh_ledger = jsonl(folder / "requests.jsonl"), jsonl(folder / "fresh-requests.jsonl")
    fresh = load(folder / "fresh-checks.json")
    fresh = fresh if isinstance(fresh, list) else []
    errors = [msg for condition, msg in (
        (summary is None, "missing arm summary.json and root summary row"),
        (result is None, "missing result/partial-exception/checkpoint JSON"),
        (search is None, "missing requests.jsonl"),
        (fresh_ledger is None, "missing fresh-requests.jsonl"),
        (not (folder / "fresh-checks.json").is_file(), "missing fresh-checks.json")) if condition]
    ledger = search or []
    counts = {k: sum(r.get("kind") == k for r in ledger)
              for k in ("search", "search_failure", "search_denial")}
    charged = counts["search"] + counts["search_failure"]
    result = result or {}
    initial, records, minima = result.get("initial") or {}, result.get("records") or [], result.get("minima") or []
    init_cost = initial.get("evaluation_requests")
    costs = [r.get("evaluation_requests") for r in records]
    result_cost = result.get("evaluation_requests")
    summary_cost = (summary or {}).get("search_requests")
    ledger_ok = (search is not None and isinstance(init_cost, int) and isinstance(result_cost, int)
                 and all(isinstance(x, int) and x >= 0 for x in costs)
                 and result_cost == init_cost + sum(costs) == charged
                 and summary_cost == charged)

    checks, duplicate, malformed = {}, [], []
    for row in fresh:
        try:
            i = int(row["index"])
        except (KeyError, TypeError, ValueError):
            malformed.append(row)
            continue
        if i in checks:
            duplicate.append(i)
        checks[i] = row
    observations, entries = [], []
    for i, minimum in enumerate(minima):
        data = minimum.get("atoms")
        if not isinstance(data, dict):
            observations.append({"index": i, "error": "minimum has no atoms object"})
            continue
        atoms, g = atoms_from_dict(data), None
        g = graph(atoms)
        label, parts = graph_label(atoms, references), formulas(atoms, g)
        check = checks.get(i)
        identity = None if check is None else (
            check.get("numbers") == data.get("numbers")
            and check.get("cell_A") == data.get("cell") and check.get("pbc") == data.get("pbc"))
        if check is not None:
            check["geometry_identity_fields_match"] = identity
        row = {"index": i, "energy_eV": minimum.get("energy"),
               "max_force_eV_A": minimum.get("max_force"), "converged": minimum.get("converged"),
               "graph": label, "graph_signature": signature(g),
               "component_formulas": parts, "fragmented": len(parts) > 1,
               "fresh": {"present": check is not None, "geometry_identity_fields_match": identity,
                         "record": check}}
        if "butadiene" in label["reference_graphs"]:
            try:
                path, angle, cosine = torsion(atoms)
                row["butadiene_torsion"] = {"carbon_path_indices": list(path),
                    "raw_deg": angle, "cosphi": cosine,
                    "interpretation": "geometric torsion only; no stability or isomer claim"}
            except Exception as exc:
                row["butadiene_torsion"] = {"error": repr(exc)}
        entry = {"arm": arm, "index": i, "graph": g}
        row["_entry"] = entry
        entries.append(entry)
        observations.append(row)
    assign(entries)
    class_ids = set()
    for row in observations:
        entry = row.pop("_entry", None)
        if entry:
            row["graph_class_id"] = entry["class_id"]
            class_ids.add(entry["class_id"])

    fresh_counts = {k: sum(r.get("kind") == k for r in (fresh_ledger or []))
                    for k in ("search", "search_failure", "search_denial")}
    missing = sorted(set(range(len(minima))) - set(checks))
    out_of_range = sorted(i for i in checks if i < 0 or i >= len(minima))
    fresh_ok = bool(result and minima and len(observations) == len(minima)
        and len(checks) == len(minima) and not duplicate and not malformed and not missing
        and not out_of_range and fresh_counts == {"search": len(minima), "search_failure": 0, "search_denial": 0}
        and all(checks[i].get("numerical_qualified") is True
                and checks[i].get("geometry_identity_fields_match") is True for i in range(len(minima))))

    boundary = (summary or {}).get("search_boundary")
    censored = bool(counts["search_denial"] or boundary in ("request_cap", "wall_cap", "pilot_wall_cap"))
    censored_records = records[-1:] if censored and records else []
    censored_indices = [r.get("index") for r in censored_records]
    attempts = records[:-1] if censored_records else list(records)
    landing_ok = lambda r: isinstance(r.get("landing"), dict) and r["landing"].get("converged") is True
    no_converged_landing = [r for r in attempts if not landing_ok(r)]
    successful = [r for r in attempts if landing_ok(r)]
    attempt_costs = [r.get("evaluation_requests") for r in attempts]
    mean_all = (sum(attempt_costs) / len(attempt_costs) if attempts and
                all(isinstance(x, int) and x >= 0 for x in attempt_costs) else None)
    success_costs = [r["evaluation_requests"] for r in successful
                     if isinstance(r.get("evaluation_requests"), int)]
    mean_success = sum(success_costs) / len(success_costs) if success_costs else None
    search_wall = (summary or {}).get("search_elapsed_seconds")
    scenario = (None if mean_all is None or not isinstance(init_cost, int) else {
        "outer_attempts": 400, "search_requests_estimate": init_cost + 400 * mean_all,
        "mean_basis": "all non-censored attempts, including attempts without a converged landing",
        "censored_attempt_excluded": True, "not_a_request_cap_or_authorization": True})
    outer = []
    for r in records:
        events = r.get("climb") or []
        outer.append({"index": r.get("index"), "status": r.get("status"),
            "accepted": r.get("accepted"),
            "climb_stops": [{k: e.get(k) for k in
                ("index", "status", "stop_reason", "rotation_stop_reason", "stage_stop_reason", "error", "requests")
                if k in e} for e in events if isinstance(e, dict)],
            "landing_numerical_success": landing_ok(r),
            "evaluation_requests": r.get("evaluation_requests"), "error": r.get("error"),
            "energy_response_eV_per_atom": r.get("energy_response"),
            "native_observed_response_meV_per_atom": (r.get("ls_update") or {}).get("observed_response_mev_per_atom"),
            "ls_update_raw": r.get("ls_update")})
    return {"arm": arm, "errors": errors, "summary": summary, "result_source": source,
        "result_present": bool(result),
        "result_completeness": "full_result" if source == "result.json" else source or "missing",
        "result_status": result.get("status"),
        "accounting": {"initial_requests": init_cost, "record_requests": costs,
                       "result_requests": result_cost, "ledger_requests": charged,
                       "summary_search_requests": summary_cost, "reconciles": ledger_ok},
        "ledger": {"kind_counts": counts, "denials": [r for r in ledger if r.get("kind") == "search_denial"],
                   "failed_requests": [r for r in ledger if r.get("kind") == "search_failure"]},
        "records": outer,
        "no_converged_landing_costs": [{"index": r.get("index"), "status": r.get("status"),
                                  "evaluation_requests": r.get("evaluation_requests"), "error": r.get("error")}
                                 for r in no_converged_landing],
        "censored_attempt": {"present": censored, "indices": censored_indices,
                             "boundary": boundary,
                             "records": censored_records},
        "ls_responses": [{k: r[k] for k in ("index", "energy_response_eV_per_atom",
                         "native_observed_response_meV_per_atom", "ls_update_raw")}
                         for r in outer],
        "fresh": {"checks_present": (folder / "fresh-checks.json").is_file(),
                  "ledger_present": fresh_ledger is not None, "ledger_counts": fresh_counts,
                  "checked_count": len(fresh), "missing_indices": missing,
                  "duplicate_indices": sorted(set(duplicate)), "out_of_range_indices": out_of_range,
                  "malformed_records": malformed, "all_minima_qualified": fresh_ok},
        "minima": observations, "graph_class_count": len(class_ids),
        "cost": {"full_attempt_count": len(attempts),
                 "no_converged_landing_count": len(no_converged_landing),
                 "mean_requests_all_attempts": mean_all,
                 "conditional_success_count": len(successful),
                 "conditional_success_mean_requests": mean_success,
                 "search_wall_seconds": search_wall,
                 "wall_seconds_amortized_over_recorded_attempts": (
                     search_wall / len(records) if isinstance(search_wall, (int, float)) and records else None),
                 "400_outer_attempt_scenario": scenario}}


def main():
    for name in ("analysis.json", "report.md"):
        if (HERE / name).exists():
            raise FileExistsError(f"refusing to overwrite {name}")
    plan = load(HERE / "plan.json")
    if plan is None:
        raise FileNotFoundError(HERE / "plan.json")
    h = helpers()
    arms = [analyze_arm(a, plan, h) for a in plan["arms"]]
    payload = {"scope": "single-seed lifecycle/cost pilot; no method ranking",
        "graph_class_scope": "class IDs are assigned within each arm and cannot be compared across arms",
        "model_qualification_limit": "MH-1's three stationary closed-shell geometries do not establish reaction accuracy",
        "attempt_limit": "12 outer attempts cannot establish convergence, coverage advantage, or rare-event probability",
        "400_attempt_limit": "outer attempts, not retained minima or the paper's 400-minimum benchmark",
        "landing_success_definition": "record.landing.converged is a numerical force-quench result, not a Hessian/minimum certificate",
        "arms": arms}
    (HERE / "analysis.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    lines = ["# C4H6 MH-1 lifecycle pilot analysis", "",
        "This audits costs, completion, minima geometry, and fresh checks. It does not rank methods or establish reaction accuracy.", "",
        "Graph class IDs are assigned within each arm and are not comparable across arms; compare each arm's class count only.", "",
        "| Arm | Result | Attempts | No converged landing | E/F requests | Denials | Search wall (s) | Mean all-attempt requests | Graph classes | Fresh complete | Accounting |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for arm in arms:
        c, q = arm["cost"], arm["ledger"]["kind_counts"]
        lines.append(f"| {arm['arm']} | {arm['result_status']} | {c['full_attempt_count']} | {c['no_converged_landing_count']} | {arm['accounting']['ledger_requests']} | {q['search_denial']} | {c['search_wall_seconds']} | {c['mean_requests_all_attempts']} | {arm['graph_class_count']} | {arm['fresh']['all_minima_qualified']} | {arm['accounting']['reconciles']} |")
        lines += ["", f"## {arm['arm']}", "",
            f"Fresh checks: {arm['fresh']['checked_count']}/{len(arm['minima'])}; missing={arm['fresh']['missing_indices']}, duplicate={arm['fresh']['duplicate_indices']}, out-of-range={arm['fresh']['out_of_range_indices']}.",
            f"E/F ledger: search={q['search']}, failed calls={q['search_failure']}; denials={q['search_denial']} are retained but not charged as calls.",
            f"No-converged-landing costs (numerical landing outcome, separate from climb status and MC acceptance): {arm['no_converged_landing_costs']}. Censored attempt: {arm['censored_attempt']}.",
            f"400 outer-attempt request scenario: {c['400_outer_attempt_scenario']}. It includes all ordinary attempts in the mean, excludes the final budget-censored attempt, and is not 400 minima or paper reproduction.",
            "Core `energy_response` is reported as eV/atom; native `observed_response_mev_per_atom` is reported as meV/atom. analysis.json retains outer status, climb stop reason, numerical landing success, MC acceptance, and raw LS updates as separate fields.", "",
            "| Minimum | Energy eV | Fmax eV/Å | Graph class | Components | Formulas | Butadiene raw CCCC torsion deg | Fresh qualified |",
            "|---:|---:|---:|---:|---:|---|---:|---|"]
        for m in arm["minima"]:
            lines.append(f"| {m.get('index')} | {m.get('energy_eV')} | {m.get('max_force_eV_A')} | {m.get('graph_class_id')} | {m.get('graph', {}).get('component_count')} | {m.get('component_formulas')} | {m.get('butadiene_torsion', {}).get('raw_deg')} | {m.get('fresh', {}).get('record', {}).get('numerical_qualified')} |")
        lines.append("")
    lines.append("Mean request cost uses every non-censored recorded outer attempt, including attempts without a converged landing. Conditional landing-success cost is reported separately in analysis.json. A `gaussian_limit` stop with a converged landing is a numerical force-quench success, not a Hessian-certified minimum or automatic failure. Search wall ends before fresh checks; wall-per-attempt is amortized, not a performance claim. The 400-attempt scenario is not a new cap or authorization.")
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    bad = [a["arm"] for a in arms if a["errors"] or not a["result_present"] or not a["accounting"]["reconciles"] or not a["fresh"]["all_minima_qualified"]]
    if bad:
        raise SystemExit(f"analysis written with explicit missing/integrity issues: {bad}")


if __name__ == "__main__":
    main()
