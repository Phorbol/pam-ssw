#!/usr/bin/env python3
"""Derive a sidecar summary when postprocessing left raw MC summary absent.

This reads one completed search-result wrapper, fresh-checks file, and the two
streamed request ledgers. It never edits the raw arm and refuses to overwrite
the requested destination. The recovered status deliberately records that
the ordinary worker summary is missing and does not imply a 100-step run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def summarize_ledger(path: Path, expected_stage: str):
    events = {}
    started = 0
    charged = 0
    request_ids = []
    censor_reasons = []
    terminal_attempts = set()
    starts = set()
    terminal_rows = 0
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != expected_stage:
                raise ValueError(f"{path}:{line_number}: unexpected stage {row.get('stage')!r}")
            event = row.get("event")
            events[event] = events.get(event, 0) + 1
            attempt = row.get("attempt")
            if event == "attempt_started":
                started += 1
                starts.add(attempt)
            elif event in ("attempt_completed", "attempt_error"):
                terminal_rows += 1
                terminal_attempts.add(attempt)
                if row.get("charged") is True:
                    charged += 1
                    request_ids.append(row.get("request"))
            elif event == "budget_censor":
                terminal_rows += 1
                terminal_attempts.add(attempt)
                if row.get("charged") is False:
                    censor_reasons.append(row.get("reason"))
            else:
                raise ValueError(f"{path}:{line_number}: unknown event {event!r}")
    contiguous = request_ids == list(range(1, charged + 1))
    if len(request_ids) != charged or not contiguous:
        raise ValueError(f"{path}: paid request IDs do not close monotonically from 1")
    if starts != terminal_attempts or len(starts) != started or len(terminal_attempts) != terminal_rows:
        raise ValueError(f"{path}: attempt starts and terminal rows differ")
    return {"charged": charged, "events": events, "censor_reasons": censor_reasons}


def derive(raw_arm: Path):
    if (raw_arm / "summary.json").exists():
        raise ValueError("raw summary.json exists; recovery is only for its absence")
    wrapper = read_json(raw_arm / "search-result.json")
    result = wrapper["result"]
    fresh = read_json(raw_arm / "fresh-checks.json")
    search_ledger = summarize_ledger(raw_arm / "search-ledger.jsonl", "search")
    fresh_ledger = summarize_ledger(raw_arm / "fresh-ledger.jsonl", "fresh")

    records = result.get("records", [])
    request_sum = int((result.get("initial") or {}).get("evaluation_requests", 0))
    candidate_map = [(0, "initial", -1, 0,
                      bool((result.get("initial") or {}).get("converged")), True)]
    landing_count = 0
    next_minimum = 1
    for record_index, record in enumerate(records):
        if record.get("index") != record_index:
            raise ValueError(f"record index mismatch at {record_index}")
        request_sum += int(record.get("evaluation_requests", 0) or 0)
        landing = record.get("landing")
        if landing is None:
            continue
        converged = landing.get("converged") is True
        minimum_index = next_minimum if converged else None
        candidate_map.append((len(candidate_map), "landing", record_index,
                              minimum_index, converged, bool(record.get("accepted"))))
        landing_count += 1
        if converged:
            next_minimum += 1

    minima_count = len(result.get("minima", []))
    checks = fresh.get("checks", [])
    mapping_matches = len(checks) == len(candidate_map)
    if mapping_matches:
        for expected, row in zip(candidate_map, checks):
            index, source, record_index, minimum_index, converged, accepted = expected
            if (row.get("candidate_index"), row.get("source"), row.get("record_index"),
                    row.get("minimum_index"), row.get("converged"), row.get("accepted")) != (
                    index, source, record_index, minimum_index, converged, accepted):
                mapping_matches = False
                break
    search_n = int(wrapper["search_requests"])
    fresh_n = int(fresh["fresh_requests"])
    fresh_qualified = sum(row.get("certified") is True for row in checks)
    cost_closed = (request_sum == result.get("evaluation_requests") == search_n
                   == search_ledger["charged"])
    fresh_closed = (fresh_n == fresh_ledger["charged"]
                    and len(checks) == fresh.get("candidate_count") == len(candidate_map)
                    and mapping_matches
                    and fresh_qualified == fresh.get("qualified_count"))
    if not cost_closed or not fresh_closed or next_minimum != minima_count:
        raise ValueError(
            f"raw recovery does not close: cost={cost_closed}, fresh={fresh_closed}, "
            f"minima_order={next_minimum == minima_count}"
        )
    censor_reasons = search_ledger["censor_reasons"] + fresh_ledger["censor_reasons"]
    return {
        "case": "C60-isomer2", "mode": "mc", "seed": 181,
        "status": "recovered_from_raw_artifacts_postprocess_incomplete",
        "algorithm_status": wrapper.get("status"),
        "budget_censor": bool(censor_reasons),
        "censor_reason": censor_reasons[0] if censor_reasons else None,
        "search_requests": search_n, "fresh_requests": fresh_n,
        "total_requests": search_n + fresh_n,
        "outer_attempt_records": len(records), "returned_landings": landing_count,
        "successful_minima_in_core_result": minima_count,
        "fresh_candidate_count": fresh.get("candidate_count"),
        "fresh_checks_recorded": len(checks),
        "fresh_qualified_count": fresh_qualified,
        "adapter_call_count": wrapper.get("adapter_calls"),
        "derivation": "sidecar recovered from raw search-result, fresh-checks, and closed ledgers; raw summary absent",
        "derivation_checks": {
            "initial_plus_records_equals_result_wrapper_and_search_ledger": cost_closed,
            "fresh_ledger_and_candidate_mapping_closed": fresh_closed,
            "fresh_qualified_count_matches_check_rows": fresh_qualified == fresh.get("qualified_count"),
            "all_fresh_rows_completed": all(row.get("status") == "fresh_completed" for row in checks),
            "converged_landings_map_to_result_minima": next_minimum == minima_count,
            "search_ledger_events": search_ledger["events"],
            "fresh_ledger_events": fresh_ledger["events"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-arm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_arm = args.raw_arm.resolve()
    output = args.output.resolve()
    if raw_arm == output or raw_arm in output.parents:
        raise ValueError("recovered summary must be outside the raw arm")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    summary = derive(raw_arm)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(output), "search_requests": summary["search_requests"],
                      "fresh_requests": summary["fresh_requests"],
                      "total_requests": summary["total_requests"]}, indent=2))


if __name__ == "__main__":
    main()
