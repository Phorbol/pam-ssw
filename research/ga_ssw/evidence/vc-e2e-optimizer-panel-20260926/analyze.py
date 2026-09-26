#!/usr/bin/env python3
"""Analyze frozen VC optimizer worker artifacts without calculator calls."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import platform
import time

import numpy as np
from ase import Atoms
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

TOLERANCES = {
    "tight": {"ltol": 0.05, "stol": 0.10, "angle_tol": 2.0},
    "broad": {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0},
}
METHODS = ("safe_total", "ase", "scipy")
NAME_RE = re.compile(r"^case(?P<case>\d+)-(?P<method>safe_total|ase|scipy)-seed(?P<seed>\d+)$")
ADAPTOR = AseAtomsAdaptor()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def matcher(label):
    return StructureMatcher(**TOLERANCES[label], primitive_cell=False,
        scale=False, attempt_supercell=False, comparator=ElementComparator())


def atoms_from_eval(row):
    if row is None:
        return None
    data = row.get("atoms")
    if not isinstance(data, dict):
        return None
    if "numbers" in data:
        return Atoms(numbers=data["numbers"], positions=data["positions"],
                    cell=data["cell"], pbc=data["pbc"])
    return Atoms(symbols=data["symbols"], positions=data["positions"],
                 cell=data["cell"], pbc=data.get("pbc", True))


def eval_volume(row):
    if row is None:
        return None
    if row.get("volume") is not None:
        return float(row["volume"])
    atoms = atoms_from_eval(row)
    return None if atoms is None else float(atoms.get_volume())


def records_from_payload(payload):
    result = payload.get("result")
    if not isinstance(result, dict):
        return None, []
    return result.get("initial"), result.get("records", [])


def geometry(frames):
    """Approximate within-arm matching; variable cells use StructureMatcher."""
    if len(frames) < 2:
        return {key: {"frame_names": [x[0] for x in frames],
            "approximate_group_count": len(frames), "greedy_group_by_frame": list(range(len(frames))),
            "greedy_representative_indices": list(range(len(frames))),
            "pairwise_matches": [[True]] if frames else [], "landing_matches_initial": [],
            "landing_return_count": 0, "distinct_new_groups_vs_initial": 0}
            for key in TOLERANCES}
    base = frames[0][1]
    for name, atoms in frames:
        if (len(atoms) != len(base)
                or sorted(atoms.numbers.tolist()) != sorted(base.numbers.tolist())
                or not np.array_equal(atoms.pbc, base.pbc)):
            raise ValueError(f"{name}: elemental composition or PBC differs from initial")
    structures = [ADAPTOR.get_structure(atoms) for _, atoms in frames]
    output = {}
    for label in TOLERANCES:
        fit = matcher(label).fit
        matrix = [[True if i == j else bool(fit(structures[i], structures[j]))
                   for j in range(len(frames))] for i in range(len(frames))]
        reps, assignments = [], []
        for i in range(len(frames)):
            group = next((g for g, rep in enumerate(reps) if matrix[i][rep]), None)
            if group is None:
                reps.append(i)
                group = len(reps) - 1
            assignments.append(group)
        matches = matrix[0][1:]
        initial_group = assignments[0]
        output[label] = {
            "frame_names": [x[0] for x in frames], "pairwise_matches": matrix,
            "greedy_representative_indices": reps, "greedy_group_by_frame": assignments,
            "approximate_group_count": len(reps), "landing_matches_initial": matches,
            "landing_return_count": int(sum(matches)),
            "distinct_new_groups_vs_initial": len(set(assignments[1:]) - {initial_group}),
        }
    return output


def ledger_counts(path, event_prefix):
    stats = {"lines": 0, "charged_events": 0, "charged_requests": 0,
             "errors": 0, "budget_denials": 0, "started_unresolved": 0,
             "events": {}}
    if not path.exists():
        return stats
    started = set()
    terminal = set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event")
            stats["lines"] += 1
            stats["events"][event] = stats["events"].get(event, 0) + 1
            if event in ("attempt_completed", "attempt_error", "fresh_completed", "fresh_error") and row.get("charged") is True:
                stats["charged_events"] += 1
                stats["charged_requests"] += 1
            if event == "attempt_started":
                started.add(row.get("attempt"))
            if event in ("attempt_completed", "attempt_error", "budget_censor"):
                terminal.add(row.get("attempt"))
            if event == "fresh_started":
                started.add(row.get("fresh_index"))
            if event in ("fresh_completed", "fresh_error"):
                terminal.add(row.get("fresh_index"))
            if event in ("attempt_error", "fresh_error"):
                stats["errors"] += 1
            if event == "budget_censor":
                stats["budget_denials"] += 1
    stats["started_unresolved"] = len(started - terminal)
    return stats


def attempt_stage_counts(records):
    totals = {"initial_true_quench_plus_certificate": 0, "rotation": 0, "height": 0,
              "biased_quench": 0, "true_check": 0,
              "post_climb_true_quench_plus_certificate": 0,
              "landing_optimizer_attempted_calls_descriptive_only": 0,
              "accounting_warnings": []}
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("stage") == "initial":
            totals["initial_true_quench_plus_certificate"] += int(record.get("requests", 0) or 0)
            continue
        climb = record.get("climb") or []
        for height in climb:
            if not isinstance(height, dict):
                continue
            totals["rotation"] += int(height.get("rotation_requests", 0) or 0)
            totals["height"] += int(height.get("height_requests", 0) or 0)
            totals["biased_quench"] += int(height.get("biased_quench_requests", 0) or 0)
            totals["true_check"] += int(height.get("true_check_requests", 0) or 0)
        optimizer = record.get("landing_optimizer")
        if isinstance(optimizer, dict):
            totals["landing_optimizer_attempted_calls_descriptive_only"] += int(optimizer.get("requests", 0) or 0)
        pieces = sum(int(h.get(k, 0) or 0) for h in climb if isinstance(h, dict)
                     for k in ("rotation_requests", "height_requests", "biased_quench_requests", "true_check_requests"))
        # record.requests is measured on the charged surface. The optimizer's
        # own counter is attempted evaluator calls and may include a denied call.
        residual = int(record.get("requests", 0) or 0) - pieces
        totals["post_climb_true_quench_plus_certificate"] += residual
        if residual < 0:
            totals.setdefault("accounting_warnings", []).append({
                "record": record.get("index", record.get("stage")),
                "residual": residual,
                "note": "measured climb EFS exceed outer record paid EFS; stage accounting mismatch"})
    return totals


def read_exits(path):
    result = {}
    if not path.exists():
        return result
    with path.open() as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{lineno}: expected name<TAB>return_code")
            result[fields[0]] = fields[1]
    return result


def classify_fresh(row):
    if row.get("source") == "initial":
        return "initial"
    if row.get("status") != "checked":
        return "fresh_error"
    if row.get("physical_certificate", {}).get("certified") is True:
        return "fresh_valid"
    return "fresh_invalid"


def analyze_arm(run_dir, name, return_code):
    arm_dir = run_dir / name
    match = NAME_RE.match(name)
    if not match:
        return {"name": name, "run_dir": str(run_dir), "worker_return_code": return_code,
                "status": "unexpected_name", "missing_artifacts": []}
    base = {"name": name, "case_index": int(match.group("case")),
            "method": match.group("method"), "seed": int(match.group("seed")),
            "run_dir": str(run_dir), "worker_return_code": return_code}
    required = ("summary.json", "search-result.json", "fresh-checks.json")
    missing = [f for f in required if not (arm_dir / f).is_file()]
    base["missing_artifacts"] = missing
    if missing:
        search_ledger = ledger_counts(arm_dir / "search-ledger.jsonl", "attempt")
        fresh_ledger = ledger_counts(arm_dir / "fresh-ledger.jsonl", "fresh")
        base.update(status="missing_result" if not arm_dir.exists() else "incomplete_result",
                    charged_search_requests_lower_bound=search_ledger["charged_requests"],
                    charged_fresh_requests_lower_bound=fresh_ledger["charged_requests"],
                    search_ledger=search_ledger, fresh_ledger=fresh_ledger,
                    cost_is_lower_bound=True,
                    endpoints=[], geometry=None)
        if return_code in ("124", "137", "143"):
            base["status"] = "timeout_or_killed_missing_result"
        return base

    summary = read_json(arm_dir / "summary.json")
    search_payload = read_json(arm_dir / "search-result.json")
    fresh_payload = read_json(arm_dir / "fresh-checks.json")
    initial, records = records_from_payload(search_payload)
    fresh_checks = fresh_payload.get("checks")
    if not isinstance(fresh_checks, list):
        raise ValueError(f"{arm_dir}: fresh-checks.json must contain checks list")
    initial_fresh = next((r for r in fresh_checks if r.get("source") == "initial"), None)
    initial_atoms = atoms_from_eval(initial)
    endpoints = []
    frames = []
    if initial_atoms is not None:
        frames.append(("initial", initial_atoms))
    landing_rows = {int(r.get("index")): r for r in records
                    if isinstance(r, dict) and r.get("index") is not None}
    for row in fresh_checks:
        idx = row.get("record_index")
        if row.get("source") == "initial":
            continue
        rec = landing_rows.get(int(idx)) if idx is not None else None
        fresh_state = classify_fresh(row)
        landing = None if rec is None else rec.get("landing")
        accepted = None if rec is None else rec.get("accepted")
        alg_cert = None if rec is None else (rec.get("certificate") or {}).get("certified")
        opt = None if rec is None else rec.get("landing_optimizer")
        alg_qualified = bool(rec is not None and landing is not None and alg_cert is True
            and isinstance(opt, dict) and opt.get("status") == "converged")
        endpoint = {"record_index": idx, "record_status": None if rec is None else rec.get("status"),
            "accepted": accepted, "algorithm_certificate": alg_cert,
            "algorithm_qualified": alg_qualified, "fresh_class": fresh_state,
            "fresh_certified": row.get("physical_certificate", {}).get("certified"),
            "fresh_status": row.get("status"), "fresh_fmax_eV_A": row.get("physical_certificate", {}).get("fmax_eV_A"),
            "fresh_stress_residual_max_eV_A3": row.get("physical_certificate", {}).get("stress_residual_max_eV_A3"),
            "energy_eV": row.get("energy_eV"), "objective_eV": row.get("objective_eV"),
            "volume_A3": eval_volume(landing), "delta_objective_vs_initial_eV": None,
            "delta_volume_vs_initial_A3": None}
        if row.get("objective_eV") is not None and initial_fresh and initial_fresh.get("objective_eV") is not None:
            endpoint["delta_objective_vs_initial_eV"] = float(row["objective_eV"] - initial_fresh["objective_eV"])
        if endpoint["volume_A3"] is not None and initial_atoms is not None:
            endpoint["delta_volume_vs_initial_A3"] = endpoint["volume_A3"] - float(initial_atoms.get_volume())
        endpoints.append(endpoint)
        if fresh_state == "fresh_valid" and alg_qualified and landing is not None:
            atoms = atoms_from_eval(landing)
            if atoms is not None:
                frames.append((f"landing_{idx}", atoms))

    search_ledger = ledger_counts(arm_dir / "search-ledger.jsonl", "attempt")
    fresh_ledger = ledger_counts(arm_dir / "fresh-ledger.jsonl", "fresh")
    stages = attempt_stage_counts(records)
    ledger_search = search_ledger["charged_requests"]
    search_count = summary.get("search_requests")
    stages_sum = sum(v for k, v in stages.items()
                     if k not in ("accounting_warnings", "landing_optimizer_attempted_calls_descriptive_only"))
    geometry_metrics = geometry(frames) if frames else {k: None for k in TOLERANCES}
    all_records = [r for r in records if isinstance(r, dict) and r.get("index") is not None]
    base.update({
        "status": summary.get("status"), "algorithm_status": summary.get("algorithm_status"),
        "budget_censor": summary.get("budget_censor"), "censor_reason": summary.get("censor_reason"),
        "run_error": summary.get("run_error"), "search_requests_summary": search_count,
        "fresh_requests_summary": summary.get("fresh_requests"), "total_requests_summary": summary.get("total_requests"),
        "charged_search_requests": ledger_search, "charged_fresh_requests": fresh_ledger["charged_requests"],
        "search_ledger_summary_delta": None if search_count is None else ledger_search - int(search_count),
        "fresh_ledger_summary_delta": None if summary.get("fresh_requests") is None else fresh_ledger["charged_requests"] - int(summary["fresh_requests"]),
        "search_ledger": search_ledger, "fresh_ledger": fresh_ledger,
        "stage_request_counts": stages, "record_stage_sum": stages_sum,
        "unclassified_search_balance": None if search_count is None else int(search_count) - stages_sum,
        "attempts": [{"index": r.get("index"), "status": r.get("status"), "accepted": r.get("accepted"),
                      "requests": r.get("requests"), "landing_present": r.get("landing") is not None,
                      "certificate": r.get("certificate"), "landing_optimizer": r.get("landing_optimizer")}
                     for r in all_records],
        "endpoints": endpoints, "initial_fresh_class": None if initial_fresh is None else classify_fresh(initial_fresh),
        "initial_fresh_certified": None if initial_fresh is None else initial_fresh.get("physical_certificate", {}).get("certified"),
        "outer_step_count": len(all_records),
        "outer_steps_without_landing": sum(r.get("landing") is None for r in all_records),
        "outer_steps_without_landing_detail": [
            {"index": r.get("index"), "status": r.get("status"), "requests": r.get("requests")}
            for r in all_records if r.get("landing") is None],
        "landing_event_count": len(endpoints),
        "fresh_check_count": len(fresh_checks),
        "initial_fresh_check_count": sum(r.get("source") == "initial" for r in fresh_checks),
        "fresh_valid_landing_events": sum(r["fresh_class"] == "fresh_valid" for r in endpoints),
        "algorithm_qualified_landing_events": sum(r["algorithm_qualified"] for r in endpoints),
        "fresh_valid_algorithm_qualified_events": sum(r["fresh_class"] == "fresh_valid" and r["algorithm_qualified"] for r in endpoints),
        "landing_algorithm_qualification_failures": sum(not r["algorithm_qualified"] for r in endpoints),
        "landing_fresh_qualification_failures": sum(r["fresh_class"] != "fresh_valid" for r in endpoints),
        "accepted_landing_events": sum(r["accepted"] is True for r in endpoints),
        "mc_rejected_landing_events": sum(r["algorithm_qualified"] and r["accepted"] is False for r in endpoints),
        "geometry": geometry_metrics,
        "geometry_frame_names": [x[0] for x in frames],
        "wall_seconds": summary.get("wall_seconds"),
    })
    return base


def render(report):
    lines = ["# VC end-to-end optimizer panel: artifact analysis", "",
        "Read-only analysis of the supplied worker artifacts; no calculator/PES calls were made. A fresh-valid endpoint is a separate EFS recheck passing the configured numeric certificate. Algorithm qualification and fresh qualification are reported independently.", "",
        "| Case | Seed | Method | Exit | Status / algorithm | Outer steps | Landing events | No landing | Algorithm-qualified / fresh-valid | Search / fresh requests | Censor |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|"]
    for a in report["arms"]:
        if a.get("missing_artifacts"):
            lines.append(f"| case{a.get('case_index')} | {a.get('seed')} | {a.get('method')} | {a.get('worker_return_code')} | {a.get('status')} | unknown | unknown | unknown | unknown | unknown | — |")
            continue
        case = report["cases"][a["case_index"]]["name"]
        lines.append(f"| {case} | {a['seed']} | {a['method']} | {a['worker_return_code']} | {a['status']} / {a['algorithm_status']} | {a['outer_step_count']} | {a['landing_event_count']} | {a['outer_steps_without_landing']} | {a['algorithm_qualified_landing_events']} / {a['fresh_valid_algorithm_qualified_events']} | {a['charged_search_requests']} / {a['charged_fresh_requests']} | {a.get('censor_reason') or '—'} |")
    lines += ["", "## Paired seeds by case and method", "",
        "| Case | Seed | Method | Outer steps | No landing | Landing events | Algorithm qualification failures | Fresh qualification failures | Accepted / MC-rejected | Tight / broad distinct groups |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in report["arms"]:
        if a.get("missing_artifacts"):
            lines.append(f"| case{a.get('case_index')} | {a.get('seed')} | {a.get('method')} | unknown | unknown | unknown | unknown | unknown | unknown | unknown |")
            continue
        case = report["cases"][a["case_index"]]["name"]
        geo = a.get("geometry") or {}
        t = (geo.get("tight") or {}).get("distinct_new_groups_vs_initial", "—")
        b = (geo.get("broad") or {}).get("distinct_new_groups_vs_initial", "—")
        lines.append(f"| {case} | {a['seed']} | {a['method']} | {a['outer_step_count']} | {a['outer_steps_without_landing']} | {a['landing_event_count']} | {a['landing_algorithm_qualification_failures']} | {a['landing_fresh_qualification_failures']} | {a['accepted_landing_events']} / {a['mc_rejected_landing_events']} | {t} / {b} |")
    totals = report["totals"]
    lines += ["", "## Cost, stages, and interpretation", "",
        f"Across artifacts, charged search requests: {totals['charged_search_requests_known']} known; fresh recheck requests: {totals['charged_fresh_requests_known']} known. Incomplete/missing arms: {totals['incomplete_arms']} of {totals['expected_arms']}; their unknown costs are not treated as zero. Plan ceilings are {report['budget']['search_requests_per_arm']} search and {report['budget']['fresh_requests_per_arm']} fresh requests per arm ({report['budget']['total_search_requests']} and {report['budget']['total_fresh_requests']} total for the full panel).", "",
        f"Denominators: {totals['outer_steps']} recorded outer steps across {totals['expected_arms']} arms; {totals['outer_steps_without_landing']} outer steps produced no landing and {totals['landing_events']} did. There were {totals['fresh_checks']} independent fresh checks: {totals['initial_checks']} initial structures plus landing events. Initial qualification is reported separately per arm. A no-landing outer failure is not counted as a landing qualification failure or MC rejection.", "",
        "No-landing outer records: " + ("; ".join(f"{report['cases'][r['case_index']]['name']} seed{r['seed']} {r['method']} step{d['index']}={d['status']}" for r in report['arms'] for d in r.get('outer_steps_without_landing_detail', [])) or "none") + ".", "",
        "Stage sums use charged surface request counts: the initial quench plus its certification; climb rotation, height, biased-quench, and true-check counters; and each outer record's residual as post-climb true-quench plus certification. `landing_optimizer.requests` is retained as a descriptive attempted-call counter only, since it can include a denied budget call. Any negative post-climb residual is retained with an accounting warning. The arm-level residual `search_requests - stage sum` and ledger-to-summary deltas expose remaining discrepancies. The worker ledger has no stage labels, so it reconciles charged totals rather than imputing stages.", "",
        "Structure matching uses pymatgen StructureMatcher with the prior VC pilot's tight (ltol 0.05, stol 0.10, angle 2°) and broad (0.20, 0.30, 5°) settings, `scale=False`, `primitive_cell=False`, `attempt_supercell=False`, and ElementComparator. It compares each arm's initial plus fresh-valid, algorithm-qualified new landings. Group counts are greedy summaries with full pairwise matrices retained; approximate matches are not basin, phase, or global-minimum identities. Energy/objective and volume changes are reported relative to the separately fresh-checked initial structure.", "",
        "The configured force/stress thresholds are numeric local certificates on this MACE OMAT model PES. This small two-seed panel can describe endpoint qualification and cost for these inputs; it cannot establish stable optimizer benefit, global search efficiency, phase identity, or transfer to other systems.", "",
        f"Analysis elapsed {report['analysis_elapsed_seconds']} s; calculator/PES calls: zero.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="one or more array-task run directories")
    parser.add_argument("--output", required=True, type=Path, help="new output directory; must not exist")
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output directory: {out}")
    started = time.monotonic()
    arms, run_meta, cases = [], [], {}
    expected = set()
    budget = None
    for run_arg in args.runs:
        run = run_arg.resolve()
        plan_path = run / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"missing {plan_path}")
        plan = read_json(plan_path)
        if budget is None:
            budget = plan["budget"]
        elif budget != plan["budget"]:
            raise ValueError("input runs have different budgets")
        exits = read_exits(run / "worker-exits.tsv")
        suffix = run.name.rsplit("-", 1)[-1]
        if not suffix.isdigit():
            raise ValueError(f"cannot infer array case index from run directory {run.name}")
        case_index = int(suffix)
        if not (0 <= case_index < len(plan["cases"])):
            raise ValueError(f"case index {case_index} outside plan")
        cases[case_index] = {"name": plan["cases"][case_index]["name"]}
        names = [f"case{case_index}-{method}-seed{seed}" for seed in plan["seeds"] for method in plan["methods"]]
        expected.update((str(run), name) for name in names)
        found = set(exits)
        found.update(p.name for p in run.iterdir() if p.is_dir() and NAME_RE.match(p.name))
        arm_names = list(dict.fromkeys(names + sorted(found - set(names))))
        run_arms = []
        for name in arm_names:
            row = analyze_arm(run, name, exits.get(name))
            arms.append(row)
            run_arms.append(row)
        run_meta.append({"run_dir": str(run), "case_index": case_index,
                         "worker_exits": exits, "expected_names": names,
                         "observed_names": sorted(found), "arms": [a["name"] for a in run_arms]})
    arms.sort(key=lambda a: (a.get("case_index", 999), a.get("seed", 999), a.get("method", ""), a.get("name", "")))
    charged_search = sum(a.get("charged_search_requests", a.get("charged_search_requests_lower_bound", 0)) or 0 for a in arms)
    charged_fresh = sum(a.get("charged_fresh_requests", a.get("charged_fresh_requests_lower_bound", 0)) or 0 for a in arms)
    expected_count = len(expected)
    incomplete = sum(bool(a.get("missing_artifacts")) for a in arms)
    report = {"analysis": "read-only joint-VC optimizer panel artifact analysis",
        "inputs": [str(p.resolve()) for p in args.runs], "runs": run_meta,
        "cases": cases, "methods": list(METHODS), "arms": arms,
        "budget": {"search_requests_per_arm": budget["search_requests_per_arm"],
                   "fresh_requests_per_arm": budget["fresh_requests_per_arm"],
                   "total_search_requests": budget["total_search_requests"],
                   "total_fresh_requests": budget["total_fresh_requests"]},
        "totals": {"expected_arms": expected_count, "observed_arm_rows": len(arms),
                   "incomplete_arms": incomplete,
                   "charged_search_requests_known": charged_search,
                   "charged_fresh_requests_known": charged_fresh,
                   "charged_total_requests_known": charged_search + charged_fresh,
                   "budget_censored_arms": sum(a.get("budget_censor") is True for a in arms),
                   "nonzero_worker_exit_arms": sum(a.get("worker_return_code") not in (None, "0") for a in arms),
                   "outer_steps": sum(a.get("outer_step_count", 0) for a in arms),
                   "outer_steps_without_landing": sum(a.get("outer_steps_without_landing", 0) for a in arms),
                   "landing_events": sum(a.get("landing_event_count", 0) for a in arms),
                   "fresh_checks": sum(a.get("fresh_check_count", 0) for a in arms),
                   "initial_checks": sum(a.get("initial_fresh_check_count", 0) for a in arms),
                   "all_search_ledger_reconciliations_zero": all(a.get("search_ledger_summary_delta") == 0 for a in arms if not a.get("missing_artifacts")),
                   "all_fresh_ledger_reconciliations_zero": all(a.get("fresh_ledger_summary_delta") == 0 for a in arms if not a.get("missing_artifacts")),
                   "all_stage_balances_zero": all(a.get("unclassified_search_balance") == 0 for a in arms if not a.get("missing_artifacts"))},
        "matcher": {"class": "pymatgen StructureMatcher", "tolerances": TOLERANCES,
                    "scale": False, "primitive_cell": False, "attempt_supercell": False,
                    "comparator": "ElementComparator"},
        "qualification_thresholds": {"source": "per-run task/config.json or plan config; inspect worker artifacts",
                    "interpretation": "fresh physical certificate is separate from source algorithm qualification"},
        "software": {"python": platform.python_version()},
        "analysis_elapsed_seconds": round(time.monotonic() - started, 3),
        "calculator_calls": 0,
        "limitations": ["Missing or timed-out arms remain in the expected denominator; unknown cost is not zero.",
            "Fresh validity means only the worker's independent numeric force/stress recheck passed on its configured model PES.",
            "Approximate structure matches do not establish basin, phase, or global-minimum identity.",
            "Two seeds and three outer steps per arm cannot establish stable optimizer benefit or general efficiency."]}
    # Derive and record thresholds from frozen task configs when available.
    thresholds = []
    for arm in arms:
        task_path = Path(arm["run_dir"]) / arm["name"] / "config.json"
        if task_path.is_file():
            conf = read_json(task_path)
            thresholds.append({"arm": arm["name"], "fmax_eV_A": conf.get("fmax"), "stress_tol_eV_A3": conf.get("stress_tol"), "pressure_eV_A3": conf.get("pressure")})
    report["qualification_thresholds"] = thresholds
    if expected_count == 12 and incomplete == 0:
        observed = report["totals"]
        expected_regression = {"outer_steps": 36, "outer_steps_without_landing": 3,
                               "landing_events": 33, "fresh_checks": 45, "initial_checks": 12}
        for key, value in expected_regression.items():
            if observed[key] != value:
                raise ValueError(f"panel denominator regression: {key}={observed[key]}, expected {value}")
        for key in ("all_search_ledger_reconciliations_zero", "all_fresh_ledger_reconciliations_zero", "all_stage_balances_zero"):
            if not observed[key]:
                raise ValueError(f"panel cost closure regression failed: {key}")
        observed["saved_artifact_regression"] = {"status": "passed", **expected_regression,
            "cost_closure": True}
    out.mkdir(parents=True, exist_ok=False)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(render(report))
    print(f"wrote {out / 'analysis.json'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
