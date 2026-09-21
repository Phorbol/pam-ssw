"""Offline audit of serialized multi-step SSW/VC state transitions.

This module never constructs a calculator.  It checks only serialized geometry,
accepted-state transitions, landing provenance, and request accounting.  A
missing field is reported as ``unsupported`` rather than treated as success.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _same(a, b):
    """Compare serialized snapshots exactly; they are copied state records."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _atoms(value):
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("atoms"), dict):
        value = value["atoms"]
    if not all(k in value for k in ("positions", "cell")):
        return None
    species = "numbers" if "numbers" in value else "symbols" if "symbols" in value else None
    if species is None:
        return None
    result = {species: value[species], "positions": value["positions"], "cell": value["cell"]}
    if "pbc" in value:
        result["pbc"] = value["pbc"]
    return result


def _add(report, kind, check, detail):
    report[kind].append({"check": check, "detail": detail})


def _qualification(report, label, value):
    """Require both a converged optimizer and a certified physical landing."""
    if not isinstance(value, dict):
        _add(report, "unsupported", f"{label}.optimizer", "optimizer result is missing")
        return False
    optimizer = value.get("optimizer")
    if not isinstance(optimizer, dict):
        _add(report, "unsupported", f"{label}.optimizer", "optimizer result is missing")
        return False
    converged = optimizer.get("converged")
    if not isinstance(converged, bool):
        converged = optimizer.get("status") == "converged" if "status" in optimizer else None
    if converged is None:
        _add(report, "unsupported", f"{label}.optimizer.converged", "optimizer convergence field is missing")
        return False
    if not converged:
        _add(report, "failures", f"{label}.optimizer", "optimizer did not converge")
        return False
    certificate = value.get("certificate")
    if not isinstance(certificate, dict) or "certified" not in certificate:
        _add(report, "unsupported", f"{label}.certificate", "physical certificate is missing")
        return False
    if certificate["certified"] is not True:
        _add(report, "failures", f"{label}.certificate", "physical certificate is not certified")
        return False
    return True


def audit_payload(payload):
    """Return a JSON-compatible support/violation report for one result file."""
    report = {"status": "unsupported", "family": None, "supported": [],
              "failures": [], "unsupported": []}
    if not isinstance(payload, dict):
        _add(report, "unsupported", "payload", "top-level JSON must be an object")
        return report
    data = payload.get("result", payload)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        _add(report, "unsupported", "schema", "requires result.records list")
        return report
    records = data["records"]
    indexed = [r for r in records if isinstance(r, dict) and "index" in r]
    if not indexed:
        _add(report, "unsupported", "steps", "no indexed outer-step records")
        return report
    if not all(isinstance(r, dict) for r in records):
        _add(report, "unsupported", "records", "every record must be an object")
        return report
    report["family"] = "reduced-or-joint-serialized"

    initial = _atoms(data.get("initial"))
    if initial is None:
        _add(report, "unsupported", "initial", "initial atoms geometry is missing")
        return report
    initial_qualified = _qualification(report, "initial", data["initial"])
    expected = initial
    known_landing_atoms = []
    valid_landing_records = []
    request_values = [r["requests"] for r in records
                      if isinstance(r, dict) and isinstance(r.get("requests"), (int, float))]
    if len(request_values) != len(records):
        _add(report, "unsupported", "requests.records", "at least one record lacks a numeric request total")
    transition_complete = True
    for rec in indexed:
        idx = rec.get("index")
        for key in ("accepted", "status", "requests", "chart_reference", "landing"):
            if key not in rec:
                _add(report, "unsupported", f"step[{idx}].{key}", "required transition field is missing")
        if not all(k in rec for k in ("accepted", "status", "requests", "chart_reference", "landing")):
            transition_complete = False
            continue
        if not isinstance(rec["accepted"], bool) or not isinstance(rec["requests"], (int, float)):
            _add(report, "unsupported", f"step[{idx}].types", "accepted/requests have invalid types")
            transition_complete = False
            continue
        reference = _atoms(rec["chart_reference"])
        if reference is None:
            _add(report, "unsupported", f"step[{idx}].chart_reference", "geometry fields are incomplete")
            transition_complete = False
        elif not _same(reference, expected):
            _add(report, "failures", f"step[{idx}].current_source", "chart_reference does not equal prior selected current")
        landing = _atoms(rec["landing"])
        valid = rec["status"] == "valid_landing"
        if valid and landing is None:
            _add(report, "unsupported", f"step[{idx}].landing", "valid landing has no serialized atoms")
        landing_qualified = True
        if valid and isinstance(rec["landing"], dict):
            landing_qualified = _qualification(report, f"step[{idx}].landing", rec["landing"])
            cert = rec["landing"].get("certificate")
            if isinstance(cert, dict) and cert.get("certified") is False:
                _add(report, "failures", f"step[{idx}].landing", "valid_landing has uncertified landing")
            elif not isinstance(cert, dict) or cert.get("certified") is not True:
                _add(report, "unsupported", f"step[{idx}].landing.certificate", "quench certificate is absent")
        if valid and landing is not None and landing_qualified:
            valid_landing_records.append(rec)
            known_landing_atoms.append(landing)
        if rec["accepted"]:
            if not valid or landing is None or not landing_qualified:
                _add(report, "failures", f"step[{idx}].accepted", "accepted step has no valid landing")
            else:
                expected = landing
        # Rejected or failed proposals leave expected selected current unchanged.
    if transition_complete:
        current = _atoms(data.get("current"))
        if current is None:
            _add(report, "unsupported", "current", "current atoms geometry is missing")
        elif not _same(current, expected):
            _add(report, "failures", "final.current_source", "serialized current is not the last selected state")

        minima = data.get("minima")
        if not isinstance(minima, list):
            _add(report, "unsupported", "minima", "minima list is missing")
        else:
            expected_minima = ([initial] if initial_qualified else []) + known_landing_atoms
            if len(minima) != len(expected_minima):
                _add(report, "failures", "minima.count", f"got {len(minima)}, expected initial plus {len(known_landing_atoms)} valid landings")
            for pos, item in enumerate(minima):
                geom = _atoms(item)
                if geom is None:
                    _add(report, "unsupported", f"minima[{pos}]", "minimum geometry is incomplete")
                elif pos < len(expected_minima) and not _same(geom, expected_minima[pos]):
                    _add(report, "failures", f"minima[{pos}].provenance", "minimum is not initial or a valid landing in order")

    if "requests" not in data:
        _add(report, "unsupported", "requests", "top-level request total is missing")
    elif len(request_values) == len(records) and data["requests"] != sum(request_values):
        _add(report, "failures", "requests.total", f"top-level {data['requests']} != record sum {sum(request_values)}")
    elif len(request_values) != len(records):
        _add(report, "unsupported", "requests.total", "cannot reconcile with incomplete record request fields")
    elif not request_values:
        _add(report, "unsupported", "requests.records", "no complete step request fields")
    else:
        _add(report, "supported", "requests.total", "top-level total equals indexed-step record sum")
    if not report["unsupported"] and not report["failures"]:
        report["status"] = "supported_pass"
    elif report["failures"]:
        report["status"] = "violations"
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args(argv)
    report = audit_payload(json.loads(args.result.read_text()))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["status"] == "violations" else 0


if __name__ == "__main__":
    raise SystemExit(main())
