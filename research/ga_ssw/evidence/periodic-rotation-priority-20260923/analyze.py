#!/usr/bin/env python3
"""Read-only accounting and initial-versus-best geometry audit for four SSW arms."""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import sys
import traceback

from ase.io import read as ase_read
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / "plan.json"
OUT_JSON = HERE / "analysis.json"
OUT_MD = HERE / "report.md"
FMAX_ERROR_TOL_EV = 1.0e-6
TOLERANCES = {
    "tight": {"ltol": 0.05, "stol": 0.10, "angle_tol": 2.0},
    "broad": {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0},
}
ADAPTOR = AseAtomsAdaptor()


def reject_nonfinite(token):
    raise ValueError(f"non-finite JSON constant: {token}")


def load_json(path):
    with Path(path).open() as stream:
        return json.load(stream, parse_constant=reject_nonfinite)


def load_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line, parse_constant=reject_nonfinite))
                except Exception as error:
                    raise ValueError(f"{path}:{line_number}: {type(error).__name__}: {error}") from error
    return rows


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"{label}: expected numeric value, got {type(value).__name__}"
    if not math.isfinite(float(value)):
        return None, f"{label}: non-finite value {value!r}"
    return float(value), None


def nonnegative_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, f"{label}: expected nonnegative integer, got {value!r}"
    return value, None


def pymatgen_version():
    for name in ("pymatgen", "pymatgen-core"):
        try:
            return version(name)
        except PackageNotFoundError:
            pass
    return "unknown"


def expected_arms(plan):
    return [
        {"case": case, "method": method, "seed": seed,
         "directory": HERE / f"{case}-{method}-seed{seed}"}
        for case in plan["input_sources"]
        for seed in plan["seeds"]
        for method in plan["methods"]
    ]


def audit_request_ledger(path):
    rows = load_jsonl(path)
    counted_ids = []
    kinds = {"search": 0, "search_failure": 0, "search_denial": 0}
    unknown_kinds = []
    errors = []
    for index, row in enumerate(rows, 1):
        kind = row.get("kind")
        if kind not in kinds:
            unknown_kinds.append({"line": index, "kind": kind})
            continue
        kinds[kind] += 1
        request = row.get("request")
        if isinstance(request, bool) or not isinstance(request, int) or request < 0:
            errors.append(f"line {index}: invalid request field {request!r}")
            continue
        if kind in ("search", "search_failure"):
            if request < 1:
                errors.append(f"line {index}: counted request must be >=1, got {request}")
            counted_ids.append(request)
    expected_ids = list(range(1, len(counted_ids) + 1))
    continuous = counted_ids == expected_ids
    if not continuous:
        errors.append("counted search request IDs are not exactly the sequence 1..N")
    return {
        "ledger_rows": len(rows),
        "successful_search_rows": kinds["search"],
        "failed_search_rows": kinds["search_failure"],
        "denial_rows_not_counted_as_requests": kinds["search_denial"],
        "counted_search_requests": len(counted_ids),
        "request_ids_continuous": continuous,
        "first_request_id": counted_ids[0] if counted_ids else None,
        "last_request_id": counted_ids[-1] if counted_ids else None,
        "unknown_kinds": unknown_kinds,
        "errors": errors,
    }


def get_nested(mapping, *keys):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def audit_rotation_costs(method, records):
    totals = {
        "rotation_requests": 0,
        "bias_quench_requests": 0,
        "true_quench_requests": 0,
        "outer_record_requests": 0,
        "climb_event_requests": 0,
        "event_unallocated_requests": 0,
        "outer_unallocated_requests": 0,
    }
    known = {key: 0 for key in totals}
    missing = {
        "rotation_cost_events": 0,
        "bias_quench_cost_events": 0,
        "true_quench_cost_records": 0,
        "event_total_costs": 0,
        "outer_record_costs": 0,
    }
    rotation_substages = {}
    anomalies = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            anomalies.append(f"record {record_index}: expected object")
            missing["rotation_cost_events"] += 1
            missing["bias_quench_cost_events"] += 1
            missing["event_total_costs"] += 1
            missing["outer_record_costs"] += 1
            continue
        record_cost, err = nonnegative_int(record.get("evaluation_requests"),
                                           f"record[{record_index}].evaluation_requests")
        if err:
            anomalies.append(err)
        else:
            totals["outer_record_requests"] += record_cost
            known["outer_record_requests"] += 1

        climb = record.get("climb")
        climb_valid = isinstance(climb, list)
        if not climb_valid:
            missing["rotation_cost_events"] += 1
            missing["bias_quench_cost_events"] += 1
            missing["event_total_costs"] += 1
            climb = []
            anomalies.append(f"record {record_index}: climb is missing or not a list")
        event_total = 0
        event_total_complete = climb_valid
        record_event_rotation = 0
        record_event_bias = 0
        record_event_cost_complete = climb_valid
        record_rotation_complete = climb_valid
        record_bias_complete = climb_valid
        recovered_stage_counts = {}
        for event_index, event in enumerate(climb):
            if not isinstance(event, dict):
                anomalies.append(f"record {record_index} climb {event_index}: expected object")
                missing["rotation_cost_events"] += 1
                missing["bias_quench_cost_events"] += 1
                missing["event_total_costs"] += 1
                record_event_cost_complete = record_rotation_complete = record_bias_complete = False
                event_total_complete = False
                continue

            event_requests, request_error = nonnegative_int(event.get("requests"),
                f"record[{record_index}].climb[{event_index}].requests")
            event_rotation_cost = None
            if request_error:
                missing["event_total_costs"] += 1
                record_event_cost_complete = False
                event_total_complete = False
                anomalies.append(request_error)

            # Rotation accounting uses the method-specific serialized diagnostics.
            if method == "recovered":
                trace = get_nested(event, "recovered_rotation", "trace")
                if isinstance(trace, list) and trace:
                    previous = 0
                    stage_costs = {}
                    trace_valid = True
                    for trace_index, stage in enumerate(trace):
                        if not isinstance(stage, dict):
                            trace_valid = False
                            anomalies.append(f"record {record_index} climb {event_index}: invalid recovered trace row {trace_index}")
                            break
                        cumulative, trace_error = nonnegative_int(stage.get("force_calls"),
                            f"record[{record_index}].climb[{event_index}].recovered_rotation.trace[{trace_index}].force_calls")
                        stage_name = stage.get("stage")
                        if trace_error or not isinstance(stage_name, str) or cumulative < previous:
                            trace_valid = False
                            anomalies.append(trace_error or f"record {record_index} climb {event_index}: invalid recovered stage/cumulative force_calls")
                            break
                        stage_costs[stage_name] = stage_costs.get(stage_name, 0) + cumulative - previous
                        previous = cumulative
                    if trace_valid:
                        rotation_cost = previous
                        event_rotation_cost = rotation_cost
                        record_event_rotation += rotation_cost
                        totals["rotation_requests"] += rotation_cost
                        known["rotation_requests"] += 1
                        for stage_name, cost in stage_costs.items():
                            rotation_substages[stage_name] = rotation_substages.get(stage_name, 0) + cost
                        reported = event.get("rotation_force_requests")
                        if reported is not None and reported != rotation_cost:
                            anomalies.append(f"record {record_index} climb {event_index}: recovered trace cost {rotation_cost} differs from top-level rotation_force_requests {reported}")
                    else:
                        missing["rotation_cost_events"] += 1
                        record_rotation_complete = False
                else:
                    missing["rotation_cost_events"] += 1
                    record_rotation_complete = False
            else:
                rotation_cost, rotation_error = nonnegative_int(event.get("rotation_force_requests"),
                    f"record[{record_index}].climb[{event_index}].rotation_force_requests")
                if rotation_error:
                    missing["rotation_cost_events"] += 1
                    record_rotation_complete = False
                    anomalies.append(rotation_error)
                else:
                    event_rotation_cost = rotation_cost
                    record_event_rotation += rotation_cost
                    totals["rotation_requests"] += rotation_cost
                    known["rotation_requests"] += 1
                    pre = get_nested(event, "pre_rotation", "force_calls")
                    main = get_nested(event, "main_rotation", "force_calls")
                    if pre is not None and main is not None and pre + main != rotation_cost:
                        anomalies.append(f"record {record_index} climb {event_index}: Ritz pre+main force_calls {pre}+{main} != rotation_force_requests {rotation_cost}")
                    if pre is not None:
                        rotation_substages["pre_rotation"] = rotation_substages.get("pre_rotation", 0) + pre
                    if main is not None:
                        rotation_substages["main_rotation"] = rotation_substages.get("main_rotation", 0) + main

            bias_cost, bias_error = nonnegative_int(event.get("quench_requests"),
                f"record[{record_index}].climb[{event_index}].quench_requests")
            if bias_error:
                missing["bias_quench_cost_events"] += 1
                record_bias_complete = False
            else:
                record_event_bias += bias_cost
                totals["bias_quench_requests"] += bias_cost
                known["bias_quench_requests"] += 1

            if event_requests is not None:
                event_total += event_requests
                if event_rotation_cost is not None and not bias_error:
                    if event_requests >= event_rotation_cost + bias_cost:
                        totals["event_unallocated_requests"] += event_requests - event_rotation_cost - bias_cost
                        known["event_unallocated_requests"] += 1
                    else:
                        anomalies.append(f"record {record_index} climb {event_index}: event request cost below rotation+bias components")
                else:
                    record_event_cost_complete = False
            else:
                record_event_cost_complete = False

        if event_total_complete:
            totals["climb_event_requests"] += event_total
            known["climb_event_requests"] += 1
        if record_cost is not None:
            landing = record.get("landing")
            landing_cost = None
            if isinstance(landing, dict):
                landing_cost, landing_error = nonnegative_int(landing.get("evaluation_requests"),
                    f"record[{record_index}].landing.evaluation_requests")
                if landing_error:
                    anomalies.append(landing_error)
            error_text = record.get("error")
            true_quench_failed = isinstance(error_text, str) and error_text.startswith("true_quench:")
            if landing_cost is not None:
                totals["true_quench_requests"] += landing_cost
                known["true_quench_requests"] += 1
            elif true_quench_failed:
                missing["true_quench_cost_records"] += 1
                anomalies.append(f"record {record_index}: true-quench attempt failed without a serialized landing cost")
            else:
                # No landing was returned. Preserve as not applicable or unallocated from the outer residual.
                pass
            if record_event_cost_complete and landing_cost is not None:
                residual = record_cost - event_total - landing_cost
                if residual < 0:
                    anomalies.append(f"record {record_index}: outer request cost is below climb plus true-quench costs")
                else:
                    totals["outer_unallocated_requests"] += residual
                    known["outer_unallocated_requests"] += 1
            elif record_event_cost_complete and not landing and not true_quench_failed:
                residual = record_cost - event_total
                if residual < 0:
                    anomalies.append(f"record {record_index}: outer request cost is below climb-event costs")
                else:
                    totals["outer_unallocated_requests"] += residual
                    known["outer_unallocated_requests"] += 1
            else:
                missing["outer_record_costs"] += 1
        else:
            missing["outer_record_costs"] += 1

    return {
        "request_totals": totals,
        "records_with_known_cost": known,
        "missing_cost_classification_counts": missing,
        "rotation_requests_by_serialized_stage": rotation_substages,
        "anomalies": anomalies,
    }


def fresh_check_status(rows, label, threshold, minimum_converged):
    candidates = [row for row in rows if isinstance(row, dict) and row.get("label") == label]
    if not candidates:
        return {"status": "missing_fresh_record", "numerical_qualified": None}
    row = candidates[-1]
    if row.get("status") != "completed":
        return {"status": row.get("status", "missing_status"), "numerical_qualified": None,
                "error": row.get("error")}
    problems = []
    values = {}
    for field in ("energy", "fmax", "energy_error"):
        value, error = finite_number(row.get(field), f"{label}.{field}")
        values[field] = value
        if error:
            problems.append(error)
    geometry_ok = all(row.get(key) is True for key in
                      ("cell_unchanged", "pbc_unchanged", "numbers_unchanged"))
    expected_qualified = bool(
        not problems and values["fmax"] <= threshold and
        abs(values["energy_error"]) <= FMAX_ERROR_TOL_EV and geometry_ok and minimum_converged
    )
    if not geometry_ok:
        problems.append("cell/PBC/composition match is false or missing")
    saved = row.get("numerical_qualified")
    if saved is not expected_qualified:
        problems.append(f"saved numerical_qualified={saved!r} differs from recomputed={expected_qualified}")
    return {
        "status": "completed_with_errors" if problems else "completed",
        "energy_eV": values["energy"],
        "fmax_eV_A": values["fmax"],
        "energy_error_eV": values["energy_error"],
        "cell_unchanged": row.get("cell_unchanged"),
        "pbc_unchanged": row.get("pbc_unchanged"),
        "numbers_unchanged": row.get("numbers_unchanged"),
        "minimum_converged": minimum_converged,
        "saved_numerical_qualified": saved,
        "recomputed_numerical_qualified": expected_qualified,
        "errors": problems,
    }


def initial_best_geometry(directory):
    initial_path = directory / "initial.extxyz"
    best_path = directory / "best.extxyz"
    if not initial_path.exists() or not best_path.exists():
        missing = [name for name, path in (("initial.extxyz", initial_path), ("best.extxyz", best_path)) if not path.exists()]
        return {"status": "missing_geometry_file", "missing": missing,
                "interpretation": "no match result; missing is not a new structure"}
    initial_atoms = ase_read(str(initial_path))
    best_atoms = ase_read(str(best_path))
    if (not initial_atoms.pbc.all() or not best_atoms.pbc.all() or
            not (initial_atoms.numbers == best_atoms.numbers).all()):
        return {"status": "geometry_contract_mismatch", "initial_pbc": initial_atoms.pbc.tolist(),
                "best_pbc": best_atoms.pbc.tolist(), "same_ordered_composition": bool((initial_atoms.numbers == best_atoms.numbers).all()),
                "interpretation": "no match result; not evidence of a new basin"}
    initial = ADAPTOR.get_structure(initial_atoms)
    best = ADAPTOR.get_structure(best_atoms)
    matches = {}
    for label, values in TOLERANCES.items():
        matcher = StructureMatcher(**values, primitive_cell=False, scale=False,
                                   attempt_supercell=False, comparator=ElementComparator())
        matches[label] = bool(matcher.fit(initial, best))
    return {
        "status": "compared",
        "initial_vs_best_matches": matches,
        "pymatgen_comparisons": len(matches),
        "interpretation": "pairwise geometric match only; not strict basin or phase identity",
    }


def analyze_arm(expected, plan, top_summary_rows):
    case, method, seed, directory = (expected[k] for k in ("case", "method", "seed", "directory"))
    out = {"case": case, "method": method, "seed": seed, "directory": str(directory),
           "artifact_status": "incomplete", "errors": []}
    paths = {name: directory / f"{name}.{suffix}" for name, suffix in (
        ("summary", "json"), ("result", "json"), ("requests", "jsonl"), ("fresh", "jsonl"))}
    out["artifacts_present"] = {name: path.exists() for name, path in paths.items()}
    try:
        summary = load_json(paths["summary"])
        result = load_json(paths["result"])
        ledger = audit_request_ledger(paths["requests"])
        fresh_rows = load_jsonl(paths["fresh"])
        if not isinstance(summary, dict) or not isinstance(result, dict):
            raise ValueError("per-arm summary.json/result.json must each be a JSON object")
        out["archived_status"] = summary.get("status")
        out["execution_class"] = summary.get("execution_class")
        out["boundary"] = summary.get("boundary")
        out["errors"].extend(ledger["errors"])
        out["ledger"] = ledger
        out["summary_search_requests"] = summary.get("search_requests")
        out["summary_search_calculator_calls"] = summary.get("search_calculator_calls")
        out["summary_denials"] = summary.get("denials")
        out["result_status"] = result.get("status")
        out["result_evaluation_requests"] = result.get("evaluation_requests")
        initial = result.get("initial")
        records = result.get("records")
        minima = result.get("minima")
        if not isinstance(initial, dict) or not isinstance(records, list) or not isinstance(minima, list):
            raise ValueError("result initial/records/minima fields have unexpected types")
        initial_cost, initial_error = nonnegative_int(initial.get("evaluation_requests"), "result.initial.evaluation_requests")
        if initial_error:
            out["errors"].append(initial_error)
        record_costs = []
        for index, record in enumerate(records):
            cost, error = nonnegative_int(get_nested(record, "evaluation_requests"),
                                          f"result.records[{index}].evaluation_requests")
            if error:
                out["errors"].append(error)
            else:
                record_costs.append(cost)
        record_sum = sum(record_costs) if len(record_costs) == len(records) else None
        result_cost, result_error = nonnegative_int(result.get("evaluation_requests"), "result.evaluation_requests")
        if result_error:
            out["errors"].append(result_error)
        accounted = initial_cost + record_sum if initial_cost is not None and record_sum is not None else None
        out["request_accounting"] = {
            "initial_requests": initial_cost,
            "outer_record_count": len(records),
            "outer_record_request_sum": record_sum,
            "initial_plus_records": accounted,
            "result_evaluation_requests": result_cost,
            "summary_search_requests": summary.get("search_requests"),
            "ledger_counted_requests": ledger["counted_search_requests"],
            "matches_result": accounted == result_cost if accounted is not None and result_cost is not None else None,
            "matches_summary": result_cost == summary.get("search_requests") if result_cost is not None else None,
            "matches_ledger": result_cost == ledger["counted_search_requests"] if result_cost is not None else None,
        }
        for check in ("matches_result", "matches_summary", "matches_ledger"):
            if out["request_accounting"][check] is False:
                out["errors"].append(f"request accounting invariant failed: {check}")
        out["outer_records"] = len(records)
        out["landings"] = sum(isinstance(record, dict) and isinstance(record.get("landing"), dict) for record in records)
        out["result_minima"] = len(minima)
        out["accepted_records"] = sum(bool(record.get("accepted")) for record in records if isinstance(record, dict))
        out["rejected_records"] = sum(not bool(record.get("accepted")) for record in records if isinstance(record, dict))
        out["rotation_and_quench_costs"] = audit_rotation_costs(method, records)
        out["rotation_and_quench_costs"]["audit_status"] = (
            "review_anomalies" if out["rotation_and_quench_costs"]["anomalies"] else "consistent"
        )
        min_rows = []
        for index, minimum in enumerate(minima):
            energy, error = finite_number(get_nested(minimum, "energy"), f"result.minima[{index}].energy")
            if error:
                out["errors"].append(error)
            elif energy is not None:
                min_rows.append((energy, minimum))
        initial_energy, initial_energy_error = finite_number(initial.get("energy"), "result.initial.energy")
        if initial_energy_error:
            out["errors"].append(initial_energy_error)
        best_energy = min((energy for energy, _ in min_rows), default=None)
        best_minimum = min(min_rows, key=lambda item: item[0])[1] if min_rows else None
        best_converged = (best_minimum.get("converged") is True) if isinstance(best_minimum, dict) else None
        out["energetics"] = {
            "initial_energy_eV": initial_energy,
            "best_energy_eV_from_result_minima": best_energy,
            "best_delta_eV_vs_initial": best_energy - initial_energy if best_energy is not None and initial_energy is not None else None,
            "best_energy_minimum_converged": best_converged,
            "best_delta_definition": "min(result.minima[*].energy) - result.initial.energy; stored best.extxyz is geometrically checked separately",
        }
        threshold = float(plan["ssw_config"]["fmax"])
        out["fresh_checks"] = {
            "initial": fresh_check_status(fresh_rows, "initial", threshold, initial.get("converged") is True),
            "best": fresh_check_status(fresh_rows, "best", threshold, best_converged is True),
        }
        for label, check in out["fresh_checks"].items():
            if check.get("status") == "completed_with_errors":
                out["errors"].extend(f"fresh {label}: {message}" for message in check["errors"])
        out["initial_vs_best_geometry"] = initial_best_geometry(directory)
        out["summary_aggregate_rows"] = [row for row in top_summary_rows if isinstance(row, dict)
            and row.get("case") == case and row.get("method") == method and row.get("seed") == seed]
        if len(out["summary_aggregate_rows"]) != 1:
            out["errors"].append(f"top-level summary has {len(out['summary_aggregate_rows'])} rows for this expected arm")
        out["artifact_status"] = "complete" if all(out["artifacts_present"].values()) else "missing_artifacts"
        out["audit_status"] = "consistent" if not out["errors"] else "review_errors"
    except FileNotFoundError as error:
        out["artifact_status"] = "missing_artifacts"
        out["errors"].append(f"missing required output: {error.filename}")
        out["audit_status"] = "incomplete"
    except Exception as error:
        out["artifact_status"] = "read_or_analysis_error"
        out["errors"].append(f"{type(error).__name__}: {error}")
        out["traceback"] = traceback.format_exc()
        out["audit_status"] = "review_errors"
    return out


def cost_prefix_for_arm(arm, budget):
    directory = Path(arm["directory"])
    result = load_json(directory / "result.json")
    initial = result.get("initial") if isinstance(result, dict) else None
    records = result.get("records") if isinstance(result, dict) else None
    if not isinstance(initial, dict) or not isinstance(records, list):
        return {"status": "invalid_result_schema", "common_budget_requests": budget}
    initial_cost, initial_error = nonnegative_int(initial.get("evaluation_requests"), "initial.evaluation_requests")
    initial_energy, initial_energy_error = finite_number(initial.get("energy"), "initial.energy")
    if initial_error or initial_energy_error:
        return {"status": "invalid_initial", "common_budget_requests": budget,
                "errors": [value for value in (initial_error, initial_energy_error) if value]}
    if initial_cost > budget:
        return {
            "status": "initial_exceeds_common_budget",
            "common_budget_requests": budget,
            "initial_requests_required": initial_cost,
            "actual_prefix_requests_including_initial": 0,
            "initial_included": False,
            "completed_records_inside_prefix": 0,
            "failed_records_inside_prefix": 0,
            "converged_landings_inside_prefix": 0,
            "best_qualified_energy_eV": None,
            "interpretation": "common budget is insufficient to complete initial quench; no minimum energy is credited",
        }
    used = initial_cost
    included_energies = []
    initial_qualified = (initial.get("converged") is True and
                         arm.get("fresh_checks", {}).get("initial", {}).get("recomputed_numerical_qualified") is True)
    if initial_qualified and used <= budget:
        included_energies.append(initial_energy)
    records_inside = failures_inside = landings_inside = 0
    landing_energy_failures = 0
    straddling = None
    record_cost_errors = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            record_cost_errors.append(f"record[{index}] is not an object")
            break
        cost, error = nonnegative_int(record.get("evaluation_requests"), f"record[{index}].evaluation_requests")
        if error:
            record_cost_errors.append(error)
            break
        if used + cost > budget:
            straddling = {"record_index": index, "requests_needed_to_finish": cost,
                          "prefix_requests_before_record": used}
            break
        used += cost
        records_inside += 1
        if record.get("error") is not None or (isinstance(record.get("status"), str) and "fail" in record["status"].lower()):
            failures_inside += 1
        landing = record.get("landing")
        if isinstance(landing, dict):
            converged = landing.get("converged") is True
            energy, energy_error = finite_number(landing.get("energy"), f"record[{index}].landing.energy")
            if converged and not energy_error:
                landings_inside += 1
                included_energies.append(energy)
            else:
                landing_energy_failures += 1
    return {
        "status": "complete_prefix" if not record_cost_errors else "prefix_cost_error",
        "common_budget_requests": budget,
        "actual_prefix_requests_including_initial": used,
        "initial_included": bool(initial_qualified and initial_cost <= budget),
        "initial_energy_eV": initial_energy if initial_qualified and initial_cost <= budget else None,
        "completed_records_inside_prefix": records_inside,
        "failed_records_inside_prefix": failures_inside,
        "converged_landings_inside_prefix": landings_inside,
        "unqualified_landing_records_inside_prefix": landing_energy_failures,
        "best_qualified_energy_eV": min(included_energies) if included_energies else None,
        "excluded_straddling_record": straddling,
        "record_cost_errors": record_cost_errors,
        "interpretation": "full-record common request prefix; includes finite converged true landings and a fresh-qualified converged initial state only",
    }


def pairwise_cost_prefixes(arms):
    result = []
    cases = sorted({arm["case"] for arm in arms})
    for case in cases:
        pair = [arm for arm in arms if arm["case"] == case and arm["method"] in ("ritz", "recovered")]
        if len(pair) != 2:
            result.append({"case": case, "status": "missing_or_duplicate_policy_arm",
                           "available_arms": [arm.get("method") for arm in pair]})
            continue
        counts = [arm.get("request_accounting", {}).get("result_evaluation_requests") for arm in pair]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            result.append({"case": case, "status": "missing_actual_search_count", "methods": [a["method"] for a in pair]})
            continue
        budget = min(counts)
        row = {"case": case, "status": "compared", "common_budget_requests": budget, "policies": {}}
        for arm in pair:
            try:
                row["policies"][arm["method"]] = cost_prefix_for_arm(arm, budget)
            except Exception as error:
                row["policies"][arm["method"]] = {"status": "prefix_analysis_error", "error": f"{type(error).__name__}: {error}"}
        result.append(row)
    return result


def write_report(result):
    lines = [
        "# Periodic rotation-priority comparison: offline readout",
        "",
        "This is an audit of the four archived seed-41 arms, not new independent validation. The two inputs are previously used; this run is a prospective strategy discriminator. Geometric identity is an initial-versus-best pairwise match only and does not establish basin or phase identity.",
        "",
        "| Case | Method | Run / audit | Requests: ledger / result | Init E (eV) | Best ΔE (eV) | Records / landings / minima | Rotation requests | Bias-quench requests | True-quench requests | Initial fresh | Best fresh | Initial~best tight / broad |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for arm in result["arms"]:
        accounting = arm.get("request_accounting", {})
        costs = arm.get("rotation_and_quench_costs", {}).get("request_totals", {})
        energy = arm.get("energetics", {})
        fresh = arm.get("fresh_checks", {})
        geom = arm.get("initial_vs_best_geometry", {})
        matches = geom.get("initial_vs_best_matches", {})
        lines.append(
            f"| {arm['case']} | {arm['method']} | {arm.get('archived_status', arm.get('artifact_status'))} / {arm.get('audit_status', 'incomplete')} "
            f"| {accounting.get('ledger_counted_requests', '—')} / {accounting.get('result_evaluation_requests', '—')} "
            f"| {energy.get('initial_energy_eV', '—')} | {energy.get('best_delta_eV_vs_initial', '—')} "
            f"| {arm.get('outer_records', '—')} / {arm.get('landings', '—')} / {arm.get('result_minima', '—')} "
            f"| {costs.get('rotation_requests', 'unallocated')} | {costs.get('bias_quench_requests', 'unallocated')} | {costs.get('true_quench_requests', 'unallocated')} "
            f"| {fresh.get('initial', {}).get('recomputed_numerical_qualified', '—')} "
            f"| {fresh.get('best', {}).get('recomputed_numerical_qualified', '—')} "
            f"| {matches.get('tight', 'unmatched/unavailable')} / {matches.get('broad', 'unmatched/unavailable')} |"
        )
    lines.extend([
        "",
        "Rotation requests use method-specific archived diagnostics: Ritz PreRot/main force-call counts; recovered-CBD per-stage trace increments. Bias-quench and true landing-quench requests are reported separately. Missing fields remain unallocated; outer record/request-ledger totals are authoritative. Search failures are counted separately from successful requests, and denials are not counted as requests.",
        "",
        "A missing geometry file, a false initial~best match, or an incomplete fresh check is not evidence of a new basin. Fresh numerical qualification checks the archived recalculation energy error, force threshold, fixed cell, PBC, composition, convergence, and finiteness; it does not establish physical stability.",
        "",
        f"Expected arms: {result['expected_arm_count']}; top-level summary rows: {result.get('aggregate_summary_row_count', 'missing')}; analyzed arms: {sum(a.get('artifact_status') == 'complete' for a in result['arms'])}.",
        f"Analysis state: {result.get('analysis_state', 'unknown')}; cost anomaly arms: {sum(a.get('rotation_and_quench_costs', {}).get('audit_status') == 'review_anomalies' for a in result['arms'])}.",
        f"Pymatgen {result['pymatgen_version']}; analysis elapsed {result.get('elapsed_seconds', 0.0):.1f} s; calculator/PES calls: zero.",
        "",
        "## Pairwise common-cost prefixes",
        "",
        "| Case | Common requests | Policy | Prefix requests | Records | Failed records | Converged landings | Best qualified E (eV) | Initial included |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ])
    for pair in result.get("pairwise_cost_prefixes", []):
        for method, prefix in pair.get("policies", {}).items():
            lines.append(f"| {pair['case']} | {pair.get('common_budget_requests', '—')} | {method} | {prefix.get('actual_prefix_requests_including_initial', '—')} | {prefix.get('completed_records_inside_prefix', '—')} | {prefix.get('failed_records_inside_prefix', '—')} | {prefix.get('converged_landings_inside_prefix', '—')} | {prefix.get('best_qualified_energy_eV', '—')} | {prefix.get('initial_included', '—')} |")
    lines.append("A record that would cross the common-budget boundary is excluded whole; its required cost is retained in `excluded_straddling_record`. Failed records fully inside the prefix remain in request and failure counts but contribute no landing energy. This is a cost-prefix comparison, not independent validation.")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main():
    started = __import__("time").monotonic()
    plan = load_json(PLAN_PATH)
    aggregate_path = HERE / "summary.json"
    aggregate_rows = load_json(aggregate_path) if aggregate_path.exists() else []
    if not isinstance(aggregate_rows, list):
        aggregate_rows = []
    expected = expected_arms(plan)
    result = {
        "analysis": "read-only request accounting, fresh-qualification, and initial-vs-best geometry audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(PLAN_PATH),
        "plan_status": plan.get("status"),
        "python": sys.version,
        "platform": platform.platform(),
        "pymatgen_version": pymatgen_version(),
        "calculator_or_PES_calls": 0,
        "expected_arm_count": len(expected),
        "expected_arms": [{k: str(v) if isinstance(v, Path) else v for k, v in row.items()} for row in expected],
        "aggregate_summary_row_count": len(aggregate_rows) if aggregate_path.exists() else None,
        "aggregate_summary_path": str(aggregate_path),
        "geometry_matcher": {
            "class": "pymatgen.core.structure_matcher.StructureMatcher",
            "comparator": "ElementComparator",
            "primitive_cell": False,
            "scale": False,
            "attempt_supercell": False,
            "tolerances": TOLERANCES,
            "maximum_pairwise_calls": 2 * len(expected),
            "interpretation": "initial.extxyz vs best.extxyz only; a match/nonmatch is not a strict basin/phase classification",
        },
        "arms": [],
        "analysis_state": "running",
    }
    if not aggregate_path.exists():
        result["aggregate_summary_error"] = "top-level summary.json missing; run may still be active or incomplete"
    elif len(aggregate_rows) != len(expected):
        result["aggregate_summary_error"] = f"expected {len(expected)} aggregate rows, found {len(aggregate_rows)}"
    def persist(state):
        result["analysis_state"] = state
        result["elapsed_seconds"] = __import__("time").monotonic() - started
        result["updated_utc"] = datetime.now(timezone.utc).isoformat()
        OUT_JSON.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_report(result)

    for row in expected:
        result["arms"].append(analyze_arm(row, plan, aggregate_rows))
        result["pairwise_cost_prefixes"] = pairwise_cost_prefixes(result["arms"])
        persist("partial")
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    result["pairwise_cost_prefixes"] = pairwise_cost_prefixes(result["arms"])
    persist("complete")
    audit_errors = (
        bool(result.get("aggregate_summary_error"))
        or any(a.get("errors") for a in result["arms"])
        or any(a.get("rotation_and_quench_costs", {}).get("anomalies") for a in result["arms"])
        or any(pair.get("status") not in ("compared",) or any(p.get("status") != "complete_prefix" for p in pair.get("policies", {}).values()) for pair in result.get("pairwise_cost_prefixes", []))
    )
    incomplete = len(result["arms"]) != len(expected) or any(a.get("artifact_status") != "complete" for a in result["arms"])
    return 1 if incomplete or audit_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
