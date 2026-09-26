#!/usr/bin/env python3
"""Geometry and cost readout for the completed VC mechanism screen; no PES calls."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import platform
import time

from ase import Atoms
from pymatgen.core.structure_matcher import ElementComparator, StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

HERE = Path(__file__).resolve().parent
RUN = HERE / "run-1499436"
OUT_JSON = HERE / "analysis.json"
OUT_MD = HERE / "report.md"
TOLERANCES = {
    "tight": {"ltol": 0.05, "stol": 0.10, "angle_tol": 2.0},
    "broad": {"ltol": 0.20, "stol": 0.30, "angle_tol": 5.0},
}
ADAPTOR = AseAtomsAdaptor()


def atoms(data):
    return Atoms(numbers=data["numbers"], positions=data["positions"],
                 cell=data["cell"], pbc=data["pbc"])


def matcher(label):
    return StructureMatcher(**TOLERANCES[label], primitive_cell=False,
                            scale=False, attempt_supercell=False,
                            comparator=ElementComparator())


def geometry(frames):
    structures = [ADAPTOR.get_structure(a) for _, a in frames]
    output = {}
    for label in TOLERANCES:
        fit = matcher(label).fit
        matrix = [[True if i == j else bool(fit(structures[i], structures[j]))
                   for j in range(len(frames))] for i in range(len(frames))]
        representatives, assignments = [], []
        for i in range(len(frames)):
            group = next((j for j, r in enumerate(representatives) if matrix[i][r]), None)
            if group is None:
                representatives.append(i)
                group = len(representatives) - 1
            assignments.append(group)
        output[label] = {"pairwise_matches": matrix,
                         "greedy_representative_indices": representatives,
                         "greedy_group_by_frame": assignments,
                         "approximate_group_count": len(representatives),
                         "initial_matches_candidates": matrix[0][1:],
                         "candidate_matches_between_arms": [matrix[i][j]
                             for i in (1, 2) for j in (3, 4)]}
    return output


def ledger_cost(arm):
    counts = Counter(row.get("stage", "<missing>") for row in arm["request_ledger"])
    if len(arm["request_ledger"]) != arm["requests"] or sum(counts.values()) != arm["requests"]:
        raise ValueError(f"{arm['arm']}: ledger row count does not match requests")
    prefix = "cell_cycle_"
    rotation = sum(n for s, n in counts.items() if s.startswith(prefix) and s.endswith("_rotation"))
    partial = sum(n for s, n in counts.items() if s.startswith(prefix) and s.endswith("_partial_fixed_cell_relax"))
    atomic = counts["fixed_cell_atomic_climb"]
    quench = counts["full_cell_true_quench"]
    start = counts["arm_start_fresh_certificate"]
    if rotation + partial + atomic + quench + start != arm["requests"]:
        raise ValueError(f"{arm['arm']}: unclassified cost stages: {dict(counts)}")
    return {"cell_rotation": rotation, "partial_fixed_cell_relax": partial,
            "atomic_climb": atomic, "full_cell_true_quench": quench,
            "arm_start_fresh_certificate": start, "fresh_endpoint_checks": arm["fresh_check_requests"],
            "arm_search_requests": arm["requests"], "arm_total_including_fresh_endpoints": arm["requests"] + arm["fresh_check_requests"],
            "ledger_stage_counts": dict(counts)}


def main():
    started = time.monotonic()
    result = json.loads((RUN / "result.json").read_text())
    if result["status"] != "completed":
        raise ValueError(f"run status is {result['status']}")
    start = atoms(result["cell_on"]["start"])
    frames = [("initial", start)]
    for name in ("cell_on", "cell_off"):
        arm = result[name]
        if len(arm["landings"]) != 2 or len(arm["steps"]) != 2:
            raise ValueError(f"{name}: expected two candidates and two outer steps")
        for i, row in enumerate(arm["landings"]):
            frames.append((f"{name}_landing_{i}", atoms(row["atoms"])))
    # Confirm the arms began from the same serialized configuration.
    off_start = atoms(result["cell_off"]["start"])
    if (start.numbers.tolist() != off_start.numbers.tolist() or
            start.pbc.tolist() != off_start.pbc.tolist() or
            start.cell.array.tolist() != off_start.cell.array.tolist() or
            start.positions.tolist() != off_start.positions.tolist()):
        raise ValueError("cell-on/off serialized starts differ")
    rows = []
    for name in ("cell_on", "cell_off"):
        arm = result[name]
        costs = ledger_cost(arm)
        steps = [{"step": s["step"], "status": s["status"],
                  "requests": s["requests"], "delta_enthalpy_eV": s["delta_enthalpy_eV"],
                  "accepted": s["accepted"],
                  "cell_cycle_request_totals": [c["requests"] for c in s.get("cell_cycles", [])],
                  "cell_cycle_rotation_force_calls": [c["cell_mode_force_calls"] for c in s.get("cell_cycles", [])],
                  "partial_relax_steps": [c["partial_relax_steps"] for c in s.get("cell_cycles", [])]}
                 for s in arm["steps"]]
        rows.append({"arm": name, "status": arm["status"], "wall_seconds": arm["wall_seconds"],
                     "cost": costs, "steps": steps,
                     "fresh_endpoint_checks": arm["fresh_endpoint_checks"]})
    gate = result["reference_gate"]
    summary = {"analysis": "post-run variable-cell mechanism screen cost and approximate geometry readout",
               "run": str(RUN / "result.json"), "run_status": result["status"],
               "scientific_status": result["scientific_status"],
               "input_and_pairing": result["pairing"], "reference_gate": gate,
               "gate_requests": gate["requests_separate_from_arm_caps"],
               "arms": rows,
               "whole_run_paid_requests_including_gate_and_fresh_endpoints": gate["requests_separate_from_arm_caps"] + sum(a["cost"]["arm_total_including_fresh_endpoints"] for a in rows),
               "frame_order": [name for name, _ in frames],
               "geometry": geometry(frames),
               "matcher": {"class": "pymatgen StructureMatcher", "tolerances": TOLERANCES,
                           "scale": False, "primitive_cell": False, "attempt_supercell": False,
                           "comparator": "ElementComparator"},
               "software": {"python": platform.python_version()},
               "analysis_execution": {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                                      "partition": os.environ.get("SLURM_JOB_PARTITION"),
                                      "qos": os.environ.get("SLURM_JOB_QOS"),
                                      "command": "sbatch --wait research/ga_ssw/evidence/vc-paper-panel-20260926/analyze.sbatch"},
               "analysis_elapsed_seconds": round(time.monotonic() - started, 3),
               "calculator_calls_for_analysis": 0,
               "interpretation_limits": [
                   "This is one seed and two outer steps per arm; no efficiency ranking or general VC claim follows.",
                   "Candidates are true-quenched and freshly endpoint-qualified, but approximate structural matching does not establish basin identity, Hessian stability, or a distinct phase.",
                   "The geometry matcher uses the established tight/broad numerical tolerances; because cells are allowed to differ, the fixed-cell exact-cell precondition in the earlier helper is intentionally omitted.",
                   "The cell-on arm performs cell cycles and an atomic climb on every outer step to match the atomic stage; this is not the paper's lambda=2 interleaving schedule.",
                   "A request cap of 6000 per arm was not reached; no budget-censored steps or failed steps occurred in this run."
               ]}
    OUT_JSON.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    OUT_MD.write_text(render(summary))


def render(s):
    lines = ["# VC mechanism screen: cost and geometry readout", "",
             f"Run `{s['run']}` completed with status `{s['run_status']}` and scientific label `{s['scientific_status']}`. This analysis used the saved result and paid-request ledgers only; calculator/PES calls: **0**.", "",
             "The reference gate used 6 requests and qualified the common start. Both arms used the same serialized start, seed 17, MACE-OMAT-small PES, and two outer steps; each arm used one request for its fresh start certificate plus two separate fresh endpoint checks. The arm cap was 6000 requests; no cap censoring or failed steps occurred.", "",
             "| Arm | Cell rotation | Partial fixed-cell relax | Atomic climb | Full-cell true quench | Start certificate | Search total | Fresh endpoints | Total incl. gate | Wall (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in s["arms"]:
        c = a["cost"]
        total = c["arm_total_including_fresh_endpoints"] + s["gate_requests"]
        lines.append(f"| {a['arm']} | {c['cell_rotation']} | {c['partial_fixed_cell_relax']} | {c['atomic_climb']} | {c['full_cell_true_quench']} | {c['arm_start_fresh_certificate']} | {c['arm_search_requests']} | {c['fresh_endpoint_checks']} | {total} | {a['wall_seconds']:.2f} |")
    lines += ["", "The gate is a shared pre-arm cost, so it is listed in each arm's inclusive total for comparability; it is counted once in the whole-run total.", "",
              "| Arm | Step | Step requests | ΔH (eV) | MC accepted | Cell-cycle aggregate requests | Rotation force calls / cycle | Partial-relax steps / cycle |",
              "|---|---:|---:|---:|---|---|---|---|"]
    for a in s["arms"]:
        for st in a["steps"]:
            lines.append(f"| {a['arm']} | {st['step']} | {st['requests']} | {st['delta_enthalpy_eV']:.6f} | {st['accepted']} | {st['cell_cycle_request_totals'] or '—'} | {st['cell_cycle_rotation_force_calls'] or '—'} | {st['partial_relax_steps'] or '—'} |")
    lines += ["", "The cell-on candidates were both valid landings but had positive ΔH (+0.659505 and +0.763256 eV), so MC rejected both. Cell-off produced +0.234709 eV (rejected) and +0.000978 eV (accepted). The cell-off endpoint on step 1 was accepted and became its final current state.", "",
              "All four candidate endpoints passed their separate fresh E/F/stress qualification checks. Their measured force and maximum absolute stress values are retained in `analysis.json`.", "",
              "Approximate structure matching uses the existing periodic-direction pilot's tight/broad tolerances and the same `StructureMatcher` options. Here the frames are the common initial structure and the four true-quenched candidate endpoints; cells may differ because this is variable-cell data.", "",
              "| Tolerance | Approximate groups among initial + four candidates | Initial matches candidates | Cell-on/off candidate matches (2×2) |",
              "|---|---:|---|---|"]
    for label, g in s["geometry"].items():
        lines.append(f"| {label} | {g['approximate_group_count']} | {g['initial_matches_candidates']} | {g['candidate_matches_between_arms']} |")
    lines += ["", "The pairwise matrices and greedy representative assignments are in `analysis.json`. These matches are geometry-based similarity only; they do not certify basin identity, Hessian stability, or phase identity. With one seed and two steps, the observed request and energy differences are descriptive, not an efficiency or generality result; they are insufficient grounds to remove or promote the VC mechanism.", "",
              f"Analysis ran as Slurm job {s['analysis_execution']['slurm_job_id']} on {s['analysis_execution']['partition']} ({s['analysis_execution']['qos']}); submit command: `{s['analysis_execution']['command']}`. Elapsed analysis time {s['analysis_elapsed_seconds']} s (Python {s['software']['python']}); no PES/calculator calls.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
