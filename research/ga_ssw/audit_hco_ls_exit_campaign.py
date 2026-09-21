"""Offline, live-safe audit of the HCO LS-exit campaign.

This script reads archived JSON/JSONL only.  It deliberately keeps the
manifest denominator even when an arm has only a plan or a partial ledger.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def ledger_stats(path: Path) -> dict[str, Any]:
    out = dict(events=0, charged=0, denials=0, last_after=None)
    if not path.is_file():
        return out
    try:
        stream = path.open()
    except OSError:
        return out
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out["events"] += 1
            out["charged"] += int(bool(row.get("charged")))
            out["denials"] += int(row.get("kind") == "denial")
            if isinstance(row.get("after"), int):
                out["last_after"] = row["after"]
    return out


def simple_record_metrics(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return dict(records=0, accepted=0, status_counts={}, raw_converged=0, raw_converged_denominator=0,
                    landing_observations=[], ls_preparation=[], ls_preparation_missing=0,
                    prequench_qualification=dict(total=0, qualified=0, rejected=0, status_counts={}),
                    true_energy_response=[], native_ls_update=[], failures=[])
    records = result.get("records") if isinstance(result.get("records"), list) else []
    accepted = [r for r in records if isinstance(r, dict) and r.get("accepted") is True]
    converged = []
    prep, response, updates, landing_observations, failures = [], [], [], [], []
    status_counts = Counter()
    for r in records:
        if not isinstance(r, dict):
            continue
        status_counts[str(r.get("status"))] += 1
        landing = r.get("landing")
        if isinstance(landing, dict) and isinstance(landing.get("converged"), bool):
            converged.append(bool(landing["converged"]))
        if isinstance(landing, dict):
            # Keep failed QuenchResult telemetry too: strict LS rejection can
            # leave landing populated while ls_preparation is null.
            landing_observations.append(dict(
                index=r.get("index"), status=r.get("status"),
                converged=landing.get("converged"), energy=landing.get("energy"),
                max_force=landing.get("max_force"), optimizer_steps=landing.get("optimizer_steps"),
                evaluation_requests=landing.get("evaluation_requests"),
                termination_reason=landing.get("termination_reason"),
                optimizer_telemetry=landing.get("optimizer_telemetry")))
        if r.get("ls_preparation") is not None:
            prep.append(r["ls_preparation"])
        else:
            prep.append(None)
        if r.get("energy_response") is not None:
            response.append(r["energy_response"])
        if r.get("ls_update") is not None:
            updates.append(r["ls_update"])
        if r.get("error") not in (None, ""):
            failures.append(dict(record_index=r.get("index"), field="error", value=r.get("error")))
    initial = result.get("initial") if isinstance(result.get("initial"), dict) else {}
    if isinstance(initial.get("converged"), bool):
        converged.insert(0, bool(initial["converged"]))
    if result.get("status") not in (None, "", "complete", "accepted") and result.get("error") not in (None, ""):
        failures.insert(0, dict(field="result.status", value=result.get("status")))
    prep_dicts = [x for x in prep if isinstance(x, dict)]
    prep_statuses = Counter(str(x.get("status")) for x in prep_dicts if x.get("status") is not None)
    prep_qualified = sum(x.get("qualified") is True or x.get("converged") is True for x in prep_dicts)
    prep_rejected = sum(x.get("qualified") is False or x.get("converged") is False for x in prep_dicts)
    return dict(records=len(records), accepted=len(accepted), status_counts=dict(status_counts),
                raw_converged=sum(converged), raw_converged_denominator=len(converged),
                landing_observations=landing_observations,
                ls_preparation=prep, ls_preparation_missing=sum(x is None for x in prep),
                prequench_qualification=dict(total=len(prep_dicts), qualified=prep_qualified,
                                             rejected=prep_rejected, status_counts=dict(prep_statuses)),
                true_energy_response=response,
                native_ls_update=updates, failures=failures)


def fresh_metrics(fresh: Any, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = fresh if isinstance(fresh, list) else []
    checked, qualified, energies = [], [], []
    request_counters = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        checked.append(dict(index=row.get("index"), status=row.get("status"),
                            energy=row.get("energy"), fmax=row.get("fmax"),
                            energy_error=row.get("energy_error"),
                            qualified=row.get("qualified")))
        if finite(row.get("requests")):
            request_counters.append(row["requests"])
        if row.get("qualified") is True:
            qualified.append(row)
        if finite(row.get("energy")):
            energies.append(row["energy"])
    best = min((r["energy"] for r in qualified if finite(r.get("energy"))), default=None)
    failed = sum(row.get("status") == "failed" for row in rows if isinstance(row, dict))
    expected = None
    if isinstance(summary, dict):
        expected = summary.get("fresh_checked", 0) + summary.get("fresh_failures", 0)
    confirmed = len(checked)
    return dict(rows=len(rows), checked=checked, qualified=len(qualified), failed=failed,
                confirmed_attempts=confirmed,
                censored_attempts=max(0, expected - len(rows)) if isinstance(expected, int) else None,
                request_counter_last=max(request_counters) if request_counters else None,
                request_counter_note="fresh.json requests are cumulative when present; not summed",
                best_qualified_energy=best,
                min_observed_energy=min(energies) if energies else None)


def audit(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, list):
        raise SystemExit(f"manifest is missing or is not a list: {root / 'manifest.json'}")
    stop = read_json(root / "campaign-stop.json")
    backend_switch_stop = isinstance(stop, dict) and stop.get("reason") == "user_backend_switch"
    rows = []
    for item in manifest:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        arm = root / name
        frame = arm / "frame0-safe-lbfgs-total"
        plan = read_json(arm / "plan.json")
        settings = read_json(frame / "resolved-settings.json") or {}
        result = read_json(frame / "result.json")
        summary = read_json(frame / "summary.json")
        fresh = read_json(frame / "fresh.json")
        ledger = ledger_stats(frame / "ledger.jsonl")
        have_terminal = all(isinstance(read_json(frame / n), (dict, list)) for n in ("result.json", "summary.json", "fresh.json"))
        if have_terminal:
            state = "completed_artifacts"
        elif isinstance(summary, dict) and summary.get("status") in {"failed", "error", "evaluation_failed", "censored", "complete", "completed"}:
            state = "finished_without_result"
        elif ledger["events"] or (arm / "execution-claim" / "slurm-job-id").is_file():
            state = "running_or_partial"
        else:
            state = "pending"
        if backend_switch_stop:
            if state == "running_or_partial":
                state = "stopped_by_user_backend_switch"
            elif state == "pending":
                state = "cancelled_unstarted"
        config = settings.get("config") if isinstance(settings, dict) else {}
        ls = settings.get("ls") if isinstance(settings, dict) else None
        pre = ls.get("prequench") if isinstance(ls, dict) else None
        rm = simple_record_metrics(result)
        fm = fresh_metrics(fresh, summary)
        initial = result.get("initial") if isinstance(result, dict) and isinstance(result.get("initial"), dict) else {}
        best = result.get("best") if isinstance(result, dict) and isinstance(result.get("best"), dict) else {}
        rows.append(dict(task=item.get("task"), name=name, case=item.get("case"),
                         variant=item.get("variant"), seed=item.get("seed"), state=state,
                         plan_status=plan.get("status") if isinstance(plan, dict) else None,
                         prequench=pre, force_limit=(pre or {}).get("fmax") if isinstance(pre, dict) else None,
                         step_limit=(pre or {}).get("steps") if isinstance(pre, dict) else None,
                         summary_status=summary.get("status") if isinstance(summary, dict) else None,
                         summary_censor_reason=summary.get("censor_reason") if isinstance(summary, dict) else None,
                         initial_energy=initial.get("energy"), initial_fmax=initial.get("max_force"),
                         best_energy=best.get("energy"), evaluation_requests=(result or {}).get("evaluation_requests") if isinstance(result, dict) else None,
                         ledger=ledger, records=rm, fresh=fm,
                         cost=dict(ledger_charged_requests=ledger["charged"],
                                   ledger_events=ledger["events"],
                                   ledger_denials=ledger["denials"],
                                   search_requests=summary.get("search_requests") if isinstance(summary, dict) else None,
                                   fresh_checked=summary.get("fresh_checked") if isinstance(summary, dict) else None,
                                   fresh_failures=summary.get("fresh_failures") if isinstance(summary, dict) else None,
                                   search_wall_seconds=summary.get("search_wall_seconds") if isinstance(summary, dict) else None,
                                   fresh_wall_seconds=summary.get("fresh_wall_seconds") if isinstance(summary, dict) else None,
                                   elapsed_seconds=summary.get("elapsed_seconds") if isinstance(summary, dict) else None,
                                   fresh_call_count_observed=fm["confirmed_attempts"],
                                   fresh_censored_attempts=fm["censored_attempts"],
                                   fresh_call_count_note="checked+failed rows are confirmed; censored calls are not inferred"),
                         result_present=isinstance(result, dict), summary_present=isinstance(summary, dict),
                         source_paths=dict(plan=str(arm / "plan.json"), result=str(frame / "result.json"),
                                           summary=str(frame / "summary.json"), fresh=str(frame / "fresh.json"),
                                           ledger=str(frame / "ledger.jsonl"))))
    completed = [r for r in rows if r["state"] == "completed_artifacts"]
    by_arm = []
    for key in sorted({(r["case"], r["variant"]) for r in rows}):
        group = [r for r in rows if (r["case"], r["variant"]) == key]
        by_arm.append(dict(case=key[0], variant=key[1], denominator=len(group),
                           completed=sum(r["state"] == "completed_artifacts" for r in group),
                           pending=sum(r["state"] == "pending" for r in group),
                           running_or_partial=sum(r["state"] == "running_or_partial" for r in group),
                           finished_without_result=sum(r["state"] == "finished_without_result" for r in group),
                           stopped_by_user_backend_switch=sum(r["state"] == "stopped_by_user_backend_switch" for r in group),
                           cancelled_unstarted=sum(r["state"] == "cancelled_unstarted" for r in group),
                           qualified=sum(r["fresh"]["qualified"] for r in group),
                           prequench_qualified=sum(r["records"]["prequench_qualification"]["qualified"] for r in group),
                           prequench_rejected=sum(r["records"]["prequench_qualification"]["rejected"] for r in group),
                           observed_records=sum(r["records"]["records"] for r in group),
                           charged_evaluations=sum(r["ledger"]["charged"] for r in group)))
    return dict(campaign=root.name, denominator=len(manifest),
                campaign_stop=stop,
                rows=rows, by_arm=by_arm,
                completed_artifact_rows=len(completed),
                scope=("Offline artifact audit. The 12 manifest rows remain the denominator; "
                       "pending/running rows are not treated as failures or successes. "
                       "Rows and fresh endpoints are observations, not unique minima. "
                       "The planned 150 outer attempts are not counted as completed."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path,
                        default=Path("research/ga_ssw/evidence/hco-ls-exit-policy-20260912"))
    args = parser.parse_args()
    output = args.root / "live-audit.json"
    output.write_text(json.dumps(audit(args.root), indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
