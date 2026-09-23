#!/usr/bin/env python3
"""Zero-PES geometry/stop audit of the frozen four periodic SSW arms."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "periodic-rotation-priority-20260923"
ARMS = [(case, method) for case in ("aloh3", "brookite48") for method in ("ritz", "recovered")]
TOLS = {"tight": (0.05, 0.10, 2.0), "broad": (0.20, 0.30, 5.0)}
ADAPTOR = AseAtomsAdaptor()


def load(path):
    with path.open() as stream:
        return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def atoms_from(data):
    if not isinstance(data, dict) or not {"numbers", "positions", "cell", "pbc"} <= data.keys():
        return None
    return Atoms(numbers=data["numbers"], positions=data["positions"], cell=data["cell"], pbc=data["pbc"])


def terminal_censor(record):
    error = str(record.get("error") or "")
    return record.get("status") == "evaluation_failed" and ("request_cap" in error or "wall_cap" in error)


def geom_compare(left_data, right_data):
    left, right = atoms_from(left_data), atoms_from(right_data)
    if left is None or right is None:
        return {"status": "missing_atoms"}
    if not np.array_equal(left.numbers, right.numbers) or not np.array_equal(left.pbc, right.pbc):
        return {"status": "composition_or_pbc_mismatch"}
    try:
        structures = (ADAPTOR.get_structure(left), ADAPTOR.get_structure(right))
        matches = {}
        for name, (ltol, stol, angle) in TOLS.items():
            matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle,
                primitive_cell=False, scale=False, attempt_supercell=False,
                comparator=ElementComparator())
            matches[name] = bool(matcher.fit(*structures))
        delta = right.positions - left.positions
        mic, _ = find_mic(delta, left.cell, pbc=left.pbc)
        norms = np.linalg.norm(mic, axis=1)
        return {"status": "compared", "structure_match": matches,
                "ordered_atom_mic_rms_A": float(np.sqrt(np.mean(norms ** 2))),
                "ordered_atom_mic_max_A": float(np.max(norms)),
                "cell_equal": bool(np.allclose(left.cell.array, right.cell.array, rtol=0, atol=1e-10)),
                "displacement_interpretation": "atom order fixed; MIC Cartesian displacement, not permutation invariant"}
    except Exception as error:
        return {"status": "comparison_error", "error": f"{type(error).__name__}: {error}"}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def analyze_arm(case, method, on_sample=None):
    folder = SOURCE / f"{case}-{method}-seed41"
    result, summary = load(folder / "result.json"), load(folder / "summary.json")
    ledger_counts = {"search": 0, "search_failure": 0, "search_denial": 0}
    request_ids = []
    with (folder / "requests.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            kind = row.get("kind")
            if kind in ledger_counts:
                ledger_counts[kind] += 1
                if kind in ("search", "search_failure"):
                    request_ids.append(row.get("request"))
    initial = result["initial"]
    current_atoms, current_energy = initial.get("atoms"), initial.get("energy")
    records = result.get("records", [])
    samples, checks = [], []
    total_record_cost = 0
    complete_records = censored_records = 0
    status_counts = {}
    for index, record in enumerate(records):
        cost = record.get("evaluation_requests")
        if isinstance(cost, int) and not isinstance(cost, bool) and cost >= 0:
            total_record_cost += cost
        status = record.get("status")
        status_counts[status] = status_counts.get(status, 0) + 1
        censored = terminal_censor(record)
        if censored:
            censored_records += 1
        else:
            complete_records += 1
            successful = [event for event in record.get("climb", [])
                          if isinstance(event, dict) and finite(event.get("true_energy"))]
            last_event = successful[-1] if successful else None
            delta = (float(last_event["true_energy"]) - float(current_energy)
                     if last_event is not None and finite(current_energy) else None)
            if status in ("lower_true_energy", "gaussian_limit"):
                expected_lower = status == "lower_true_energy"
                observed_lower = delta is not None and delta < 0.0
                checks.append({"record_index": index, "status": status,
                    "current_energy_before_eV": current_energy,
                    "last_successful_true_energy_eV": last_event.get("true_energy") if last_event else None,
                    "true_energy_delta_eV": delta,
                    "last_successful_biased_energy_eV": last_event.get("biased_energy") if last_event else None,
                    "sign_consistent": (observed_lower == expected_lower) if delta is not None else None})
            if status == "gaussian_limit" and len(samples) < 3:
                landing = record.get("landing")
                samples.append({"record_index": index, "outer_status": status,
                    "record_requests": cost, "cumulative_requests": total_record_cost + initial.get("evaluation_requests", 0),
                    "current_before_energy_eV": current_energy,
                    "last_event": (None if last_event is None else {
                        "index": last_event.get("index"), "true_energy_eV": last_event.get("true_energy"),
                        "biased_energy_total_eV": last_event.get("biased_energy"),
                        "event_status": last_event.get("status"), "requests": last_event.get("requests")}),
                    "landing_energy_eV": landing.get("energy") if isinstance(landing, dict) else None,
                    "landing_converged": landing.get("converged") if isinstance(landing, dict) else None,
                    "geometry": {
                        "current_vs_last_atoms": geom_compare(current_atoms, record.get("last_atoms")),
                        "current_vs_landing": geom_compare(current_atoms, landing.get("atoms") if isinstance(landing, dict) else None),
                        "last_atoms_vs_landing": geom_compare(record.get("last_atoms"), landing.get("atoms") if isinstance(landing, dict) else None)},
                    "cost_to_record_including_initial": initial.get("evaluation_requests", 0) + total_record_cost})
                if on_sample is not None:
                    on_sample({"case": case, "method": method,
                        "selected_samples_completed": len(samples),
                        "selected_first_three_gaussian_limit": samples,
                        "stop_sign_checks_completed_so_far": checks,
                        "record_index_processed": index})
        # State walk includes every completed landing, regardless of sampled status.
        landing = record.get("landing")
        if record.get("accepted") is True and isinstance(landing, dict) and landing.get("converged") is True:
            current_atoms, current_energy = landing.get("atoms"), landing.get("energy")
    accounting = {"initial_requests": initial.get("evaluation_requests"),
        "sum_all_record_requests_including_censored": total_record_cost,
        "initial_plus_records": (initial.get("evaluation_requests", 0) + total_record_cost),
        "result_requests": result.get("evaluation_requests"), "summary_requests": summary.get("search_requests"),
        "complete_record_count": complete_records, "terminal_censored_record_count": censored_records,
        "record_status_counts": status_counts,
        "ledger_success_rows": ledger_counts["search"],
        "ledger_failure_rows": ledger_counts["search_failure"],
        "ledger_denial_rows_not_counted": ledger_counts["search_denial"],
        "ledger_request_ids_continuous_1_to_n": request_ids == list(range(1, len(request_ids) + 1)),
        "ledger_counted_requests_match_result": len(request_ids) == result.get("evaluation_requests")}
    accounting["request_totals_consistent"] = (
        accounting["initial_plus_records"] == accounting["result_requests"] ==
        accounting["summary_requests"] == len(request_ids)
    ) and accounting["ledger_request_ids_continuous_1_to_n"]
    return {"case": case, "method": method, "run_status": summary.get("status"),
        "accounting": accounting, "selected_first_three_gaussian_limit": samples,
        "stop_sign_checks": checks,
        "stop_sign_consistent_count": sum(c["sign_consistent"] is True for c in checks),
        "stop_sign_inconsistent_count": sum(c["sign_consistent"] is False for c in checks),
        "stop_sign_uncheckable_count": sum(c["sign_consistent"] is None for c in checks)}


def write_outputs(data):
    (HERE / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    lines = ["# Periodic return mechanism: zero-PES diagnostic", "",
        "This is a selected diagnostic sample (first three completed `gaussian_limit` records per arm), not a whole-population return/basin estimate. Matching is approximate geometry; ordered MIC displacement is not permutation invariant. Energies are reported by role and not used to rank sub-MLIP differences.", "",
        "| Case | Method | Requests | Complete / censored records | Sampled records | Stop checks consistent / inconsistent / uncheckable |", "|---|---|---:|---:|---:|---:|"]
    for arm in data["arms"]:
        a = arm["accounting"]
        lines.append(f"| {arm['case']} | {arm['method']} | {a['result_requests']} | {a['complete_record_count']} / {a['terminal_censored_record_count']} | {len(arm['selected_first_three_gaussian_limit'])} | {arm['stop_sign_consistent_count']} / {arm['stop_sign_inconsistent_count']} / {arm['stop_sign_uncheckable_count']} |")
    for arm in data["arms"]:
        lines += ["", f"## {arm['case']} / {arm['method']}", "",
            "| Record | Requests | Current E | Biased total E | Physical E at biased endpoint | True landing E | Status | Current~work tight/broad | Current~landing tight/broad | Work~landing tight/broad |", "|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
        for sample in arm["selected_first_three_gaussian_limit"]:
            event = sample.get("last_event") or {}
            geo = sample["geometry"]
            def pair(key):
                value = geo[key]
                if value.get("status") != "compared": return value.get("status")
                m = value["structure_match"]
                return f"{m['tight']}/{m['broad']}"
            lines.append(f"| {sample['record_index']} | {sample.get('record_requests')} | {sample.get('current_before_energy_eV')} | {event.get('biased_energy_total_eV')} | {event.get('true_energy_eV')} | {sample.get('landing_energy_eV')} | {sample['outer_status']} | {pair('current_vs_last_atoms')} | {pair('current_vs_landing')} | {pair('last_atoms_vs_landing')} |")
    lines += ["", "Stop-sign details, full per-record energy changes, geometry displacements, missing comparisons, and all request costs are preserved in `analysis.json`."]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")


def main():
    data = {"protocol": str(HERE / "plan.md"), "source_evidence": str(SOURCE),
        "created_utc": datetime.now(timezone.utc).isoformat(), "calculator_or_PES_calls": 0,
        "maximum_structure_match_fits": 72, "tolerances": TOLS, "arms": [], "state": "running"}
    def persist_progress(progress):
        data["active_arm_progress"] = progress
        data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_outputs(data)

    for case, method in ARMS:
        arm = analyze_arm(case, method, on_sample=persist_progress)
        data["arms"].append(arm)
        data.pop("active_arm_progress", None)
        data["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_outputs(data)
    data["state"] = "complete"
    data["completed_utc"] = datetime.now(timezone.utc).isoformat()
    audit_errors = []
    if len(data["arms"]) != len(ARMS):
        audit_errors.append("expected four arms")
    for arm in data["arms"]:
        if not arm["accounting"].get("request_totals_consistent"):
            audit_errors.append(f"{arm['case']}/{arm['method']}: request accounting mismatch")
        if arm["stop_sign_inconsistent_count"] or arm["stop_sign_uncheckable_count"]:
            audit_errors.append(f"{arm['case']}/{arm['method']}: stop-sign inconsistency or missing event comparison")
        if len(arm["selected_first_three_gaussian_limit"]) != 3:
            audit_errors.append(f"{arm['case']}/{arm['method']}: fewer than three selected records")
        for sample in arm["selected_first_three_gaussian_limit"]:
            if any(pair.get("status") != "compared" for pair in sample["geometry"].values()):
                audit_errors.append(f"{arm['case']}/{arm['method']} record {sample['record_index']}: missing geometry comparison")
    data["audit_errors"] = audit_errors
    write_outputs(data)
    return 1 if audit_errors else 0


if __name__ == "__main__": raise SystemExit(main())
