"""Offline analysis for the 12-slot matched material rotation campaign.

This module only reads JSON/EXTXYZ artifacts.  It never imports a calculator or
evaluates a potential.  Missing and incomplete slots remain in the denominator.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list


CASES = ("aloh26", "brookite48")
SOLVERS = ("ritz", "dimer", "broyden-euclidean")
SEEDS = (11, 29)
MATCH_SETTINGS = ((.1, .15, 2), (.2, .3, 5))


def _atoms(value):
    if isinstance(value, Atoms):
        return value.copy()
    if not isinstance(value, dict):
        return None
    try:
        return Atoms(numbers=value["numbers"], positions=value["positions"],
                     cell=value.get("cell"), pbc=value.get("pbc", False))
    except (KeyError, TypeError, ValueError):
        return None


def geometry(value):
    """Return approximate periodic geometry diagnostics for one stored frame."""
    atoms = _atoms(value)
    if atoms is None:
        raise ValueError("frame has no serializable atoms")
    symbols = np.asarray(atoms.get_chemical_symbols())
    i, j, d, shifts = neighbor_list("ijdS", atoms, 6.0, self_interaction=False)
    out = {
        "formula": atoms.get_chemical_formula(),
        "composition": dict(collections.Counter(symbols)),
        "volume_A3": float(atoms.get_volume()),
        "min_periodic_distance_A": float(d.min()) if len(d) else None,
        "pair_minima_A": {}, "coordination": {}, "H_nearest_O": [],
    }
    for x, y in itertools.combinations_with_replacement(sorted(set(symbols)), 2):
        ds = d[(symbols[i] == x) & (symbols[j] == y)]
        out["pair_minima_A"][x + "-" + y] = float(ds.min()) if len(ds) else None
    for x, y, cutoffs in (("Ti", "O", (2.2, 2.3, 2.4)),
                          ("Al", "O", (2.2, 2.3, 2.4))):
        centers = np.flatnonzero(symbols == x)
        if not len(centers):
            continue
        out["coordination"][x + "-" + y] = {
            str(c): {"indices_zero_based": centers.tolist(),
                     "counts": [int(np.sum((i == k) & (symbols[j] == y) & (d < c)))
                                for k in centers]}
            for c in cutoffs}
    centers = np.flatnonzero(symbols == "H")
    oxy = np.flatnonzero(symbols == "O")
    for k in centers:
        mask = np.flatnonzero((i == k) & (symbols[j] == "O"))
        if len(mask):
            pick = mask[np.argmin(d[mask])]
            out["H_nearest_O"].append({"H": int(k), "O": int(j[pick]),
                                        "shift": shifts[pick].tolist(),
                                        "distance_A": float(d[pick])})
    return out


def matching(a, b):
    """Periodic pymatgen matching at the prescribed two sensitivities."""
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as exc:
        return [{"settings": {}, "match": None, "error": repr(exc)} for _ in MATCH_SETTINGS]
    aa, bb = _atoms(a), _atoms(b)
    if aa is None or bb is None:
        return [{"settings": {}, "match": None, "error": "missing atoms"} for _ in MATCH_SETTINGS]
    result = []
    for ltol, stol, angle in MATCH_SETTINGS:
        settings = {"ltol": ltol, "stol": stol, "angle_tol": angle,
                    "scale": False, "primitive_cell": True,
                    "attempt_supercell": True}
        try:
            sm = StructureMatcher(**settings)
            result.append({"settings": settings, "match": bool(sm.fit(
                AseAtomsAdaptor.get_structure(aa), AseAtomsAdaptor.get_structure(bb),
                symmetric=True))})
        except Exception as exc:
            result.append({"settings": settings, "match": None, "error": repr(exc)})
    return result


def _load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, repr(exc)


def _frame(entry):
    if isinstance(entry, dict):
        return entry.get("atoms", entry)
    return None


def _event_records(data):
    records = data.get("records", []) if isinstance(data, dict) else []
    return records if isinstance(records, list) else []


def _contains_width(value):
    if isinstance(value, dict):
        return bool(value.get("width")) or any(_contains_width(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_width(v) for v in value)
    return False


def _row(root, case, solver, seed):
    name = f"{case}-{solver}-seed{seed}"
    folder = root / name
    row = {"name": name, "case": case, "solver": solver, "seed": seed,
           "status": "missing", "result_path": str(folder / "result.json"),
           "search_cost": None, "validation_force_pass": None,
           "gaussian_reached": 0, "rotation_fail_initial": 0,
           "rotation_fail_later": 0, "bestdelta_eV": None,
           "total_raw_minima": None, "geometry": None}
    summary = _load(folder / "summary.json")
    if isinstance(summary, tuple):
        summary = None
    result = _load(folder / "result.json")
    if isinstance(result, tuple):
        result = None
    if summary:
        row["status"] = summary.get("status", row["status"])
        row["search_cost"] = summary.get("search_requests")
        row["summary"] = summary
        row['termination'] = ('search_budget' if summary.get('search_denials',0)
                              else summary.get('status'))
    if not result:
        return row
    row["status"] = result.get("status", row["status"])
    row["search_cost"] = result.get("requests", row["search_cost"])
    records = _event_records(result)
    climbs = [c for e in records if isinstance(e, dict)
              for c in (e.get("climb") or []) if isinstance(c, dict)]
    reached = [c for c in climbs if c.get("width") is not None]
    row["gaussian_reached"] = sum(any(c.get('width') is not None for c in e.get('climb', []))
                                   for e in records)
    row["total_gaussians_reached"] = len(reached)
    row['rotation_budget_released'] = sum(bool(c.get('rotation_budget_released')) for c in climbs)
    row['rotation_stop_reasons'] = dict(collections.Counter(
        c.get('rotation_stop_reason','unrecorded') for c in climbs))
    failures = [e['climb'][-1] for e in records
                if e.get('status') == 'rotation_failed' and e.get('climb')]
    row["rotation_fail_initial"] = sum(c.get("index") == 0 for c in failures)
    row["rotation_fail_later"] = sum(isinstance(c.get("index"), int) and c.get("index") > 0 for c in failures)
    minima = result.get("minima", [])
    row["total_raw_minima"] = len(minima) if isinstance(minima, list) else None
    if minima:
        initial = minima[0]
        best = min((m for m in minima if m.get("energy") is not None),
                   key=lambda m: m["energy"], default=minima[0])
        if initial.get("energy") is not None and best.get("energy") is not None:
            row["bestdelta_eV"] = float(best["energy"] - initial["energy"])
        try:
            row["geometry"] = geometry(_frame(best))
        except (ValueError, TypeError, KeyError):
            row["geometry"] = {"error": "best frame unavailable"}
    validation = _load(folder / "validation.json")
    if isinstance(validation, list):
        checks = [x for x in validation if isinstance(x, dict)]
        row["validation_force_pass"] = sum(bool(x.get("force_qualified")) for x in checks)
        row["validation_count"] = len(checks)
    return row


def deduplicate(frames):
    outputs = []
    for ti, setting in enumerate(MATCH_SETTINGS):
        representatives = []
        unknown = []
        assignments = []
        for i, item in enumerate(frames):
            matches = []
            for j in representatives:
                if frames[j]['case'] != item['case']:
                    continue
                outcome = matching(item['atoms'], frames[j]['atoms'])[ti]
                matches.append((j, outcome))
            found = next((j for j, outcome in matches if outcome['match'] is True), None)
            if found is not None:
                assignments.append(found)
            elif any(outcome['match'] is None for _, outcome in matches):
                unknown.append(i)
                assignments.append(None)
            else:
                representatives.append(i)
                assignments.append(i)
        outputs.append(dict(settings=setting,count=None if unknown else len(representatives),
            known_representatives=representatives,unknown=unknown,assignments=assignments))
    return outputs


def analyze(root):
    rows = [_row(root, c, s, seed) for c in CASES for s in SOLVERS for seed in SEEDS]
    available = [r for r in rows if r["status"] != "missing"]
    frames = []
    for row in available:
        result = _load(root / row["name"] / "result.json")
        if isinstance(result, dict):
            for minimum_index, minimum in enumerate(result.get("minima", [])):
                if _frame(minimum) is not None:
                    frames.append({"run": row["name"], "case": row["case"],
                                   "index": minimum_index,
                                   "atoms": _frame(minimum),
                                   "geometry": geometry(_frame(minimum))})
    for row in rows:
        own = [f for f in frames if f['run'] == row['name']]
        row['identity'] = deduplicate(own) if own else None
    dedup_by_tolerance = deduplicate(frames)
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "planned_slots": 12, "available_slots": len(available),
            "missing_slots": 12 - len(available), "matching_settings": MATCH_SETTINGS,
            "deduplicated_minima_by_tolerance": dedup_by_tolerance, "runs": rows,
            "frames": frames,
            "frozen_rotation_report": _load(root / "frozen-rotation.json")
            if (root / "frozen-rotation.json").exists() else None}


def markdown(report):
    lines = ["# Matched material rotation analysis", "",
             f"Available slots: **{report['available_slots']}/12**; missing: **{report['missing_slots']}**.",
             "", "| Run | Status | Gaussian reached | Rotation fail initial/later | Search requests | Validation force pass | Best ΔE (eV) | Raw minima |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in report["runs"]:
        vf = "unknown" if r["validation_force_pass"] is None else str(r["validation_force_pass"])
        lines.append(f"| {r['name']} | {r['status']} | {r['gaussian_reached']} | {r['rotation_fail_initial']}/{r['rotation_fail_later']} | {r['search_cost'] if r['search_cost'] is not None else 'unknown'} | {vf} | {r['bestdelta_eV'] if r['bestdelta_eV'] is not None else 'unknown'} | {r['total_raw_minima'] if r['total_raw_minima'] is not None else 'unknown'} |")
    lines += ["", "Periodic deduplicated minima by tolerance: " + ", ".join(
                 f"{x['settings']} -> {x['count']}" for x in report["deduplicated_minima_by_tolerance"]),
              "Matching uses scale=False, primitive_cell=True, attempt_supercell=True at (.1,.15,2) and (.2,.3,5). These are approximate sensitivity checks, not chemical identity certificates.",
              "Geometry diagnostics use a 6 Å periodic neighbor list, Al–O/Ti–O coordination cutoffs 2.2/2.3/2.4 Å, H–O nearest distances, and periodic minimum distances. No PES calls are made."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "evidence/matched-direction-materials-20260917")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); out = (args.output or root).resolve(); out.mkdir(parents=True, exist_ok=True)
    report = analyze(root)
    (out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n")
    (out / "summary.md").write_text(markdown(report))
    print(f"Analyzed {report['available_slots']}/12 slots; 0 PES calls")


if __name__ == "__main__":
    main()
