#!/usr/bin/env python3
"""Post-hoc, zero-PES structural readout for completed LJ38 trajectories."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
CAMPAIGN = ROOT / "research/ga_ssw/evidence/lj38-direction-target-20261012"
SLM_ROOT = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/lj38-source-20260912")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stream_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def qualify_minimum(row: dict, initial: bool) -> tuple[bool, str]:
    if initial:
        converged = row.get("converged") is True
        fmax = row.get("fmax_eV_A")
        components = row.get("connected_component_sizes_1p3sigma")
    else:
        gate = row.get("minimum_gate") or {}
        converged = gate.get("converged") is True
        fmax = gate.get("fmax_eV_A")
        components = row.get("minimum_connected_component_sizes_1p3sigma")
    if not converged:
        return False, "not_converged"
    if fmax is None or float(fmax) > 0.05:
        return False, "force_not_qualified"
    if components != [38]:
        return False, "not_single_connected_38_atom_component"
    return True, "force_qualified_connected_minimum"


def compare(analyzer, reference, atoms):
    try:
        return analyzer.compare_geometry(reference, atoms)
    except Exception as exc:
        return {"classification": "inconclusive",
                "reason": f"matcher_error:{type(exc).__name__}:{exc}"}


def analyze_run(run_dir: Path, runner, target, slm):
    from ase.io import read

    run_dir = run_dir.resolve()
    result_path = run_dir / "result.json"
    progress_path = run_dir / "progress.jsonl"
    structures_path = run_dir / "outer-structures.extxyz"
    ledger_path = run_dir / "search-ef-ledger.jsonl"
    if not all(p.is_file() for p in (result_path, progress_path, structures_path, ledger_path)):
        missing = [str(p) for p in (result_path, progress_path, structures_path, ledger_path) if not p.is_file()]
        raise FileNotFoundError(f"run artifacts missing: {missing}")

    result = json.loads(result_path.read_text())
    frames = read(structures_path, index=":")
    frame_map = {(str(f.info["event_role"]), int(f.info["event_index"])): f for f in frames}
    progress = jsonl(progress_path)
    candidates = []
    gm_found = False
    slm_found = False
    for row in progress:
        initial = row.get("kind") == "initial"
        if not initial and row.get("kind") != "outer_step":
            continue
        qualifies, qualification = qualify_minimum(row, initial)
        cost = row.get("evaluation_requests") if initial else row.get("cumulative_requests")
        structure_ref = row.get("structure") if initial else row.get("minimum_structure")
        record = {"kind": row.get("kind"), "outer_index": None if initial else row.get("index"),
                  "cost_paid_requests": cost, "energy_eV": row.get("energy_eV") if initial else
                  (row.get("minimum_gate") or {}).get("energy_eV"),
                  "fmax_eV_A": row.get("fmax_eV_A") if initial else
                  (row.get("minimum_gate") or {}).get("fmax_eV_A"),
                  "qualification": qualification, "structure_ref": structure_ref}
        if not qualifies:
            candidates.append(record)
            continue
        try:
            if initial:
                atoms = frame_map[("initial", -1)]
            else:
                role, index = structure_ref.split("#", 1)[1].split(":", 1)
                atoms = frame_map[(role, int(index))]
        except (AttributeError, KeyError, ValueError) as exc:
            record["qualification"] = "missing_or_unresolvable_structure_frame"
            record["frame_error"] = f"{type(exc).__name__}: {exc}"
            candidates.append(record)
            continue
        record["atom_count"] = len(atoms)
        if not gm_found:
            record["target_geometry"] = compare(runner.ANALYSIS, target, atoms)
            gm_found = record["target_geometry"].get("classification") == "same"
        else:
            record["target_geometry"] = {"classification": "not_examined_after_first_match"}
        if not slm_found:
            record["second_minimum_geometry"] = compare(runner.ANALYSIS, slm, atoms)
            slm_found = record["second_minimum_geometry"].get("classification") == "same"
        else:
            record["second_minimum_geometry"] = {"classification": "not_examined_after_first_match"}
        candidates.append(record)

    eligible = [r for r in candidates if r["qualification"] == "force_qualified_connected_minimum"]
    def first_class(classification: str):
        matches = [r for r in eligible if r.get("cost_paid_requests") is not None
                   and r.get("target_geometry", {}).get("classification") == classification]
        return min(matches, key=lambda x: int(x["cost_paid_requests"])) if matches else None

    first_target = first_class("same")
    inconclusive = [r for r in eligible if r.get("target_geometry", {}).get("classification") == "inconclusive"]
    first_slm = min((r for r in eligible if r.get("cost_paid_requests") is not None
                     and r.get("second_minimum_geometry", {}).get("classification") == "same"),
                    key=lambda x: int(x["cost_paid_requests"]), default=None)
    uncosted = [r for r in eligible if r.get("cost_paid_requests") is None]
    ledger_counts = {"rows": 0, "denied": 0, "charged_failed": 0,
                     "paid_calculator_errors": 0, "denied_examples": [],
                     "charged_failure_examples": [], "paid_calculator_error_examples": []}
    for row in stream_jsonl(ledger_path):
        ledger_counts["rows"] += 1
        if row.get("status") == "denied":
            ledger_counts["denied"] += 1
            if len(ledger_counts["denied_examples"]) < 2:
                ledger_counts["denied_examples"].append(row)
        if row.get("charged") is True and row.get("status") == "failed":
            ledger_counts["charged_failed"] += 1
            if len(ledger_counts["charged_failure_examples"]) < 3:
                ledger_counts["charged_failure_examples"].append(row)
            if row.get("actual_calculator_called") is True:
                ledger_counts["paid_calculator_errors"] += 1
                if len(ledger_counts["paid_calculator_error_examples"]) < 3:
                    ledger_counts["paid_calculator_error_examples"].append(row)

    best_path = run_dir / "best.extxyz"
    best_readout = None
    fresh_best = next((r for r in result.get("fresh", []) if r.get("role") == "best"), None)
    if best_path.is_file():
        best_atoms = read(best_path)
        best_readout = {
            "file": str(best_path), "energy_eV": result.get("best_energy_eV"),
            "fresh_record": fresh_best,
            "target_geometry": compare(runner.ANALYSIS, target, best_atoms),
            "second_minimum_geometry": compare(runner.ANALYSIS, slm, best_atoms),
            "connected_component_sizes_1p3sigma": runner.PANEL.connected_components(best_atoms),
        }
    return {
        "run": str(run_dir), "slot": result.get("slot"), "seed": result.get("seed"),
        "arm": result.get("arm"), "execution_status_preserved": result.get("status"),
        "execution_error_field": result.get("error"),
        "search_requests_total": result.get("search_requests_total"),
        "ledger_rows": ledger_counts["rows"],
        "result_denied_request_count": result.get("denied_requests"),
        "ledger_denied_request_count": ledger_counts["denied"],
        "denied_request_examples": ledger_counts["denied_examples"],
        "charged_failed_request_count": ledger_counts["charged_failed"],
        "paid_calculator_error_count": ledger_counts["paid_calculator_errors"],
        "paid_failure_examples": ledger_counts["charged_failure_examples"],
        "paid_calculator_error_examples": ledger_counts["paid_calculator_error_examples"],
        "target_joint_hit": result.get("status") == "first_hit",
        "first_structural_target_minimum": first_target,
        "first_structural_target_cost_requests": None if first_target is None else first_target["cost_paid_requests"],
        "first_second_minimum_match": first_slm,
        "target_geometry_inconclusive_minimum_count": len(inconclusive),
        "target_geometry_inconclusive_minima": inconclusive,
        "uncosted_structural_observations": uncosted,
        "eligible_minima_scanned": len(eligible), "minima_records_seen": len(candidates),
        "best_structure_readout": best_readout,
        "structures_sha256": sha256(structures_path), "progress_sha256": sha256(progress_path),
        "search_ledger_sha256": sha256(ledger_path),
        "scope": "post-hoc structural identity only; does not change the predeclared joint energy/force/geometry target hit",
    }


def main():
    from ase.io import read

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, required=True,
                        help="completed run directory; repeat for each member of the pair")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory, distinct from all source run directories")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    runner = load_module(CAMPAIGN / "run.py", "lj38_campaign_readout")
    target = runner.target_atoms()
    slm_path = SLM_ROOT / "optim-finish.extxyz"
    slm_qualification_path = SLM_ROOT / "optim-qualification.json"
    if not slm_path.is_file() or not slm_qualification_path.is_file():
        raise FileNotFoundError(f"qualified archived LJ38 SLM reference unavailable under {SLM_ROOT}")
    slm_qualification = json.loads(slm_qualification_path.read_text())
    slm_record = next(row for row in slm_qualification["properties"] if row["file"] == "optim-finish")
    slm_atoms = read(slm_path)
    payload = {
        "status": "readout_complete", "pes_requests": 0,
        "target_reference": {"file": str(runner.REFERENCE_PATH), "energy_eV": runner.TARGET_E,
                             "sha256": sha256(runner.REFERENCE_PATH)},
        "second_minimum_reference": {"file": str(slm_path), "energy_eV": slm_record["energy"],
                                      "fmax_eV_A": slm_record["fmax"],
                                      "qualification_file": str(slm_qualification_path),
                                      "qualification_sha256": sha256(slm_qualification_path),
                                      "sha256": sha256(slm_path),
                                      "identity_basis": "pre-existing independently qualified OPTIM second-lowest LJ38 minimum"},
        "geometry_method": {"implementation": str(runner.ANALYSIS.__file__),
                            "mapping_cap": runner.ANALYSIS.MAX_MAPPINGS,
                            "proper_rotation_rms_tolerance_A": runner.ANALYSIS.GEOM_RMS_TOL,
                            "connected_component_method": str(runner.PANEL.__file__),
                            "connected_graph_cutoff_sigma": 1.3},
        "runs": [analyze_run(path, runner, target, slm_atoms) for path in args.run],
        "limitations": ["A capped nonmatch is inconclusive, not evidence of structural difference.",
                        "A geometry match to the second-minimum endpoint does not establish a transition path or funnel residence.",
                        "Structural target discovery is reported separately from the frozen joint target criterion.",
                        "Execution status and denied/paid failed requests are preserved as independent fields."],
    }
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    (output / "readout.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
