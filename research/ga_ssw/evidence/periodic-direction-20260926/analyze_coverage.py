#!/usr/bin/env python3
"""Pairwise periodic geometry coverage for the preserved three-step pilot."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time

import numpy as np
from ase import Atoms
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
VERIFY = HERE / "verification"
OUT_JSON = HERE / "coverage-summary.json"
OUT_MD = HERE / "coverage-report.md"
TOLERANCES = {
    "tight": {"ltol": 0.05, "stol": 0.10, "angle_tol": 2.0},
    "broad": {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0},
}
ADAPTOR = AseAtomsAdaptor()


def package_version():
    for name in ("pymatgen", "pymatgen-core"):
        try:
            return version(name)
        except PackageNotFoundError:
            pass
    return "unknown"


def matcher(label):
    return StructureMatcher(**TOLERANCES[label], primitive_cell=False,
                            scale=False, attempt_supercell=False,
                            comparator=ElementComparator())


def atoms_from_record(record):
    data = record["atoms"]
    return Atoms(numbers=data["numbers"], positions=data["positions"],
                 cell=data["cell"], pbc=data["pbc"])


def geometry_comparison(frames):
    base = frames[0][1]
    for label, atoms in zip((name for name, _ in frames), (a for _, a in frames)):
        if (len(atoms) != len(base) or not np.array_equal(atoms.numbers, base.numbers)
                or not np.array_equal(atoms.pbc, base.pbc)
                or not np.array_equal(atoms.cell.array, base.cell.array)):
            raise ValueError(f"{label}: ordered composition/cell/PBC differs from initial")
    structures = [ADAPTOR.get_structure(atoms) for _, atoms in frames]
    output = {}
    for label in TOLERANCES:
        fit = matcher(label).fit
        matrix = [[True if i == j else bool(fit(structures[i], structures[j]))
                   for j in range(len(frames))] for i in range(len(frames))]
        # Sequential representative grouping is deterministic; pairwise matrix is
        # retained because approximate matching need not be transitive.
        representatives = []
        assignments = []
        for i in range(len(frames)):
            match = next((group for group, representative in enumerate(representatives)
                          if matrix[i][representative]), None)
            if match is None:
                representatives.append(i)
                match = len(representatives) - 1
            assignments.append(match)
        output[label] = {
            "pairwise_matches": matrix,
            "greedy_representative_indices": representatives,
            "greedy_group_by_frame": assignments,
            "approximate_group_count": len(representatives),
            "landing_matches_initial": matrix[0][1:],
            "landing_return_count": int(sum(matrix[0][1:])),
        }
    return output


def load_arm(case, method):
    directory = RUNS / f"{case}-{method}"
    with (directory / "result.json").open() as stream:
        result = json.load(stream)
    with (directory / "requests.jsonl").open() as stream:
        request_rows = sum(bool(line.strip()) for line in stream)
    initial = atoms_from_record(result["initial"])
    minima = [atoms_from_record(row) for row in result["minima"]]
    if len(minima) != 4:
        raise ValueError(f"{directory.name}: expected initial plus 3 minima, got {len(minima)}")
    if result.get("status") != "completed" or len(result.get("records", [])) != 3:
        raise ValueError(f"{directory.name}: expected three completed outer steps")
    frames = [("initial", initial)] + [(f"landing_{i}", atoms)
                                        for i, atoms in enumerate(minima[1:])]
    metrics = geometry_comparison(frames)
    if result.get("evaluation_requests") != request_rows:
        raise ValueError(f"{directory.name}: result request count differs from ledger lines")
    return {
        "case": case,
        "method": method,
        "status": result.get("status"),
        "outer_steps": len(result.get("records", [])),
        "total_search_requests": result.get("evaluation_requests"),
        "initial_search_requests": result["initial"].get("evaluation_requests"),
        "step_search_requests": [row.get("evaluation_requests") for row in result["records"]],
        "request_ledger_rows": request_rows,
        "frame_order": [name for name, _ in frames],
        "geometry": metrics,
    }


def main():
    started = time.monotonic()
    protocol = json.loads((RUNS / "protocol.json").read_text())
    verification = json.loads((VERIFY / "summary.json").read_text())
    verify_by_case = {row["case"]: row for row in verification["rows"]}
    cases = ("rutile", "anatase", "brookite48")
    methods = ("global", "local_memory")
    arms = [load_arm(case, method) for case in cases for method in methods]
    for arm in arms:
        row = verify_by_case[arm["case"] + "-" + arm["method"]]
        arm["fresh_qualified_endpoints"] = 4 if row.get("qualified") is True else None
        arm["local_resume_replayed_requests"] = row.get("replayed")
        arm["local_resume_state_equal"] = row.get("state_equal")
    comparisons = []
    for case in cases:
        by_method = {a["method"]: a for a in arms if a["case"] == case}
        for label in TOLERANCES:
            global_arm = by_method["global"]["geometry"][label]
            local_arm = by_method["local_memory"]["geometry"][label]
            global_structures = [atoms_from_record(row) for row in
                json.loads((RUNS / f"{case}-global" / "result.json").read_text())["minima"][1:]]
            local_structures = [atoms_from_record(row) for row in
                json.loads((RUNS / f"{case}-local_memory" / "result.json").read_text())["minima"][1:]]
            global_initial = atoms_from_record(json.loads(
                (RUNS / f"{case}-global" / "result.json").read_text())["initial"])
            local_initial = atoms_from_record(json.loads(
                (RUNS / f"{case}-local_memory" / "result.json").read_text())["initial"])
            if (not np.array_equal(global_initial.numbers, local_initial.numbers) or
                    not np.array_equal(global_initial.cell.array, local_initial.cell.array) or
                    not np.array_equal(global_initial.pbc, local_initial.pbc)):
                raise ValueError(f"{case}: global/local starts differ in ordered composition, cell, or PBC")
            fit = matcher(label).fit
            matrix = [[bool(fit(ADAPTOR.get_structure(a), ADAPTOR.get_structure(b)))
                       for b in local_structures] for a in global_structures]
            comparisons.append({"case": case, "tolerance": label,
                                "global_landing_to_local_landing_matches": matrix,
                                "matched_cross_arm_pairs": int(sum(sum(row) for row in matrix))})
    summary = {
        "analysis": "pairwise fixed-cell periodic geometry comparison among initial and three true landing minima per arm",
        "source": "preserved runs/*/result.json; no calculator calls",
        "protocol_config": protocol["config"],
        "direction_settings": protocol["direction"],
        "cases": list(cases),
        "methods": list(methods),
        "tolerances": TOLERANCES,
        "matcher": {"class": "pymatgen StructureMatcher", "scale": False,
                    "primitive_cell": False, "attempt_supercell": False,
                    "comparator": "ElementComparator", "same_ordered_composition_cell_pbc_required": True},
        "arms": arms,
        "tight_broad_pairwise_relations_identical": all(
            arm["geometry"]["tight"]["pairwise_matches"] ==
            arm["geometry"]["broad"]["pairwise_matches"] for arm in arms),
        "search_request_totals_by_method": {
            method: sum(arm["total_search_requests"] for arm in arms
                        if arm["method"] == method) for method in methods},
        "cross_arm_landing_comparisons": comparisons,
        "verification": verification,
        "software": {"python": platform.python_version(), "pymatgen": package_version()},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "interpretation_limits": [
            "A match is approximate geometric similarity for this fixed ordered cell, not basin, phase, or Hessian identity.",
            "Greedy representative group counts depend on frame order when approximate matching is nontransitive; pairwise matrices are retained.",
            "Three outer steps per arm are development evidence and do not establish efficiency or generality.",
            "Fresh endpoint qualification and same-oracle local resume are separate from geometric coverage.",
        ],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    OUT_MD.write_text(render(summary))


def render(summary):
    rows = ["# Periodic direction pilot: approximate structure coverage", "",
            "Geometry-only comparison of each arm’s initial minimum and its three true landing minima. Matching uses the prior periodic rotation audit’s tight and broad tolerance sets, with fixed cell scale, primitive-cell, supercell, and element-comparison settings. Energy values do not enter matching.", "",
            "| Case | Arm | Search E/F requests | Fresh endpoints | Tight groups incl. initial | Tight returns | Broad groups incl. initial | Broad returns | Local replay requests / equal state |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm in summary["arms"]:
        tight, broad = arm["geometry"]["tight"], arm["geometry"]["broad"]
        replay = arm["local_resume_replayed_requests"]
        replay_text = "—" if replay is None else f"{replay} / {arm['local_resume_state_equal']}"
        rows.append(f"| {arm['case']} | {arm['method']} | {arm['total_search_requests']} | {arm['fresh_qualified_endpoints']} / 4 | {tight['approximate_group_count']} | {tight['landing_return_count']} / 3 | {broad['approximate_group_count']} | {broad['landing_return_count']} / 3 | {replay_text} |")
    rows += ["", "The groups are sequential representative clusters over `initial, landing_0, landing_1, landing_2`; the JSON retains the full pairwise matrix and representative assignments at each tolerance. Because approximate matching may not be transitive, group counts are order-dependent summaries, not unique basin counts.", "",
             "Cross-arm landing-to-landing pair matches:", "",
             "| Case | Tolerance | Matching global/local landing pairs |", "|---|---|---:|"]
    for row in summary["cross_arm_landing_comparisons"]:
        rows.append(f"| {row['case']} | {row['tolerance']} | {row['matched_cross_arm_pairs']} / 9 |")
    totals = summary["search_request_totals_by_method"]
    rows += ["", f"Across these particular runs, search totals were {totals['global']} requests for the global arm and {totals['local_memory']} for local memory ({sum(totals.values())} combined). This is a raw cost observation from one seed and three steps per case, not an efficiency estimate.", "",
             f"Tight and broad pairwise relations were identical for all six within-arm panels: `{summary['tight_broad_pairwise_relations_identical']}`. The match/return and grouping readout was insensitive to this tolerance change for these sampled frames.", "",
             "All six runs completed three outer steps. Search request totals include the initial quench and all recorded steps; fresh endpoint checks are four separate recalculations per arm. The local-memory arm’s replay uses the saved oracle stream and is a same-trajectory resume check, not independent sampling.", "",
             "A landing matching the initial structure is counted as a return under the stated tolerance. Nonmatches show geometric difference under that matcher only; neither result certifies a distinct basin, positive Hessian, or a new phase. The three-step, three-case panel is a development pilot and supports no universal efficiency claim.", "",
             f"Matcher: pymatgen {summary['software']['pymatgen']}; analysis elapsed {summary['elapsed_seconds']} s; calculator/PES calls: zero.", ""]
    return "\n".join(rows)


if __name__ == "__main__":
    main()
