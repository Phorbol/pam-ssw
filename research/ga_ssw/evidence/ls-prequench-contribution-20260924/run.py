"""Prepare and optionally run a bounded same-state LS prequench ablation.

CPU preflight exports twelve structures from frozen C4H6 MH-1 coverage runs.
GPU work is opt-in via --execute and never reruns SSW.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_RUNS = ROOT / "research/ga_ssw/evidence/c4h6-mh1-coverage-20260924"
LEDGER_PATH = ROOT / "research/ga_ssw/evidence/periodic-rotation-priority-20260923/ledger.py"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_SHA256 = "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47"
HEAD = "422d2b6c82d40e466e4e52bde5b6e3963ea507b9"
RECORD_INDICES = (0, 199, 399)
ARMS = ("paper_ls", "native_ls")
SEEDS = (61, 67)
REQUEST_CAP_PER_QUENCH = 1000
FRESH_REQUESTS_PER_CASE = 2
MAXITER = 400
FMAX = 0.03
LBFGS_MEMORY = 500
TOTAL_REQUEST_CAP = 12 * (REQUEST_CAP_PER_QUENCH + FRESH_REQUESTS_PER_CASE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> bytes:
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def atoms_record(value, label: str) -> dict:
    if not isinstance(value, dict) or not all(k in value for k in ("numbers", "positions", "cell", "pbc")):
        raise ValueError(f"{label} is not a serialized ASE Atoms record")
    numbers, positions = value["numbers"], value["positions"]
    if len(numbers) != 10 or len(positions) != 10 or any(len(x) != 3 for x in positions):
        raise ValueError(f"{label} is not the 10-atom C4H6 structure")
    return {"numbers": numbers, "positions": positions, "cell": value["cell"], "pbc": value["pbc"]}


def selected_inputs():
    """Read each frozen result once and reconstruct pre-record states by MC acceptance."""
    inputs = []
    source_hashes = {}
    for arm in ARMS:
        for seed in SEEDS:
            source = SOURCE_RUNS / f"{arm}-seed{seed}" / "result.json"
            result = json.loads(source.read_text())
            source_summary = json.loads((source.parent / "summary.json").read_text())
            records = result.get("records")
            if (result.get("status") != "completed" or source_summary.get("status") != "completed"
                    or not isinstance(records, list) or len(records) != 400):
                raise ValueError(f"incomplete source result: {source}")
            source_hashes[source] = sha256(source)
            current = atoms_record(result["initial"]["atoms"], f"{source}:initial.atoms")
            targets = set(RECORD_INDICES)
            for index, record in enumerate(records):
                if record.get("index") != index:
                    raise ValueError(f"record index mismatch at {source} record {index}")
                if index in targets:
                    prep = record.get("ls_preparation")
                    if not isinstance(prep, dict) or not isinstance(prep.get("soft_quench"), dict):
                        raise ValueError(f"missing LS soft-prequench at {source} record {index}")
                    soft = prep["soft_quench"]
                    landing = record.get("landing")
                    if not isinstance(landing, dict) or not isinstance(landing.get("atoms"), dict):
                        raise ValueError(f"missing full SSW landing at {source} record {index}")
                    inputs.append({
                        "case_id": f"{arm}-seed{seed}-record{index:03d}",
                        "arm": arm, "seed": seed, "record_index": index,
                        "source_result": str(source.relative_to(ROOT)),
                        "source_sha256": source_hashes[source],
                        "record_status": record.get("status"),
                        "accepted": bool(record.get("accepted")),
                        "evaluation_requests": record.get("evaluation_requests"),
                        "energy_response": record.get("energy_response"),
                        "ls_preparation": {
                            "qualification": prep.get("qualification"),
                            "evaluation_requests": prep.get("evaluation_requests"),
                            "true_energy_before_eV": prep.get("true_energy_before"),
                            "true_energy_after_eV": prep.get("true_energy_after"),
                            "soft_quench_energy_eV": soft.get("energy"),
                            "soft_quench_max_force_eV_A": soft.get("max_force"),
                            "soft_quench_converged": soft.get("converged"),
                            "soft_quench_optimizer_steps": soft.get("optimizer_steps"),
                        },
                        "landing_reported": {
                            "energy_eV": landing.get("energy"),
                            "max_force_eV_A": landing.get("max_force"),
                            "converged": landing.get("converged"),
                        },
                        "step_start_atoms": current,
                        "soft_prequench_atoms": atoms_record(soft.get("atoms"), f"{source}:soft_quench.atoms"),
                        "ssw_landing_atoms": atoms_record(landing["atoms"], f"{source}:landing.atoms"),
                    })
                if record.get("accepted"):
                    landing = record.get("landing")
                    if not isinstance(landing, dict) or not isinstance(landing.get("atoms"), dict):
                        raise ValueError(f"accepted record {index} lacks a landing in {source}")
                    current = atoms_record(landing["atoms"], f"{source}:record{index}.landing.atoms")
    if len(inputs) != 12:
        raise ValueError(f"expected 12 selected records; found {len(inputs)}")
    return inputs


def preflight():
    git_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    if git_head != HEAD:
        raise RuntimeError(f"expected prepared source {HEAD}, found {git_head}")
    core_dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--", "pamssw"],
                                check=True, capture_output=True, text=True).stdout
    if core_dirty:
        raise RuntimeError("tracked pamssw core has uncommitted changes")
    inputs = selected_inputs()
    for case in inputs:
        dump(HERE / "inputs" / f"{case['case_id']}.json", case)
    manifest = {
        "status": "prepared_not_executed", "purpose": "same-state LS prequench contribution ablation",
        "git_head": git_head, "core_tree": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD:pamssw"], check=True,
            capture_output=True, text=True).stdout.strip(),
        "source_root": str(SOURCE_RUNS.relative_to(ROOT)),
        "selected_records": list(RECORD_INDICES), "arms": list(ARMS), "seeds": list(SEEDS),
        "case_count": len(inputs), "request_cap_per_quench": REQUEST_CAP_PER_QUENCH,
        "fresh_requests_per_case": FRESH_REQUESTS_PER_CASE,
        "total_request_cap": TOTAL_REQUEST_CAP,
        "inputs": [{"case_id": c["case_id"], "source_result": c["source_result"],
                    "source_sha256": c["source_sha256"],
                    "input_json_sha256": sha256(HERE / "inputs" / f"{c['case_id']}.json")}
                   for c in inputs],
    }
    dump(HERE / "preflight.json", manifest)
    print(f"prepared_not_executed: {len(inputs)} cases, request cap {TOTAL_REQUEST_CAP}")


def load_ledger():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ls_prequench_shared_ledger", LEDGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ase_atoms(record):
    from ase import Atoms
    return Atoms(numbers=record["numbers"], positions=record["positions"],
                 cell=record["cell"], pbc=record["pbc"])


def execute():
    started_all = time.monotonic()
    import numpy as np
    import torch
    from mace.calculators import MACECalculator
    sys.path.insert(0, str(ROOT))
    from pamssw.standalone.surface import quench

    manifest_path = HERE / "preflight.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("git_head") != HEAD or manifest.get("case_count") != 12:
        raise RuntimeError("missing or stale CPU preflight manifest")
    current_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                                  capture_output=True, text=True).stdout.strip()
    current_core = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:pamssw"], check=True,
                                  capture_output=True, text=True).stdout.strip()
    core_dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "HEAD", "--", "pamssw"],
                                check=True, capture_output=True, text=True).stdout
    if current_core != manifest.get("core_tree") or core_dirty:
        raise RuntimeError("current checkout/core differs from CPU-preflight provenance")
    for item in manifest.get("inputs", []):
        input_path = HERE / "inputs" / f"{item['case_id']}.json"
        if not input_path.is_file() or sha256(input_path) != item.get("input_json_sha256"):
            raise RuntimeError(f"prepared case input changed or missing: {item.get('case_id')}")
    if not MODEL.is_file() or sha256(MODEL) != MODEL_SHA256:
        raise RuntimeError("MH-1 model missing or SHA256 mismatch")
    if sha256(LEDGER_PATH) != "729355f7f85defc9faebb3b128ec09fb70d6359e3fb5e89bb8c6949b91c19e45":
        raise RuntimeError("shared request ledger changed")
    if (HERE / "runs").exists():
        raise FileExistsError("runs/ already exists; refusing retry or overwrite")

    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ledger = load_ledger()
    calc_args = dict(model_paths=str(MODEL), head="omol", device="cuda", default_dtype="float64",
                     enable_cueq=False, enable_oeq=False)
    search_calc = MACECalculator(**calc_args)
    fresh_calc = MACECalculator(**calc_args)
    inputs = [json.loads((HERE / "inputs" / f"{item['case_id']}.json").read_text())
              for item in manifest["inputs"]]
    deadline = started_all + 1140.0  # Leave one minute for final ledger flush before Slurm's 20-minute limit.
    rows = []
    for case in inputs:
        if time.monotonic() >= deadline:
            break
        folder = HERE / "runs" / case["case_id"]
        folder.mkdir(parents=True, exist_ok=False)
        row = {"case_id": case["case_id"], "status": "started", "quench_requests": 0,
               "fresh_requests": 0, "request_cap_per_quench": REQUEST_CAP_PER_QUENCH}
        surface = None
        try:
            search_calc.reset()
            surface = ledger.CountedSurface(search_calc, folder / "quench-requests.jsonl",
                                            REQUEST_CAP_PER_QUENCH, max(0.0, deadline-time.monotonic()))
            quenched = quench(ase_atoms(case["soft_prequench_atoms"]), surface, fmax=FMAX,
                              steps=MAXITER, optimizer="safe-lbfgs-total", lbfgs_memory=LBFGS_MEMORY)
            row["prequench_only_quench"] = {
                "status": "completed", "energy_eV": quenched.energy,
                "max_force_eV_A": quenched.max_force, "converged": quenched.converged,
                "optimizer_steps": quenched.optimizer_steps,
                "optimizer_telemetry": getattr(quenched.optimizer_telemetry, "__dict__", quenched.optimizer_telemetry),
                "terminal_atoms": quenched.atoms,
            }
        except Exception as error:
            row["prequench_only_quench"] = {"status": "failed", "error": repr(error)}
            quenched = None
        row["quench_requests"] = 0 if surface is None else surface.requests
        row["quench_denials"] = 0 if surface is None else surface.denials
        row["quench_boundary"] = None if surface is None else surface.boundary
        fresh_surface = ledger.CountedSurface(fresh_calc, folder / "fresh-requests.jsonl",
                                              FRESH_REQUESTS_PER_CASE,
                                              max(0.0, deadline-time.monotonic()))
        fresh_rows = []
        targets = [("original_ssw_landing", ase_atoms(case["ssw_landing_atoms"])),
                   ("prequench_only_terminal", None if quenched is None else quenched.atoms)]
        for label, atoms in targets:
            item = {"label": label, "status": "not_run"}
            if atoms is not None:
                try:
                    fresh_calc.reset()
                    energy, forces = fresh_surface.evaluate(atoms)
                    fmax = float(np.linalg.norm(forces, axis=1).max())
                    item.update(status="completed", energy_eV=energy,
                                max_force_eV_A=fmax, force_qualified=fmax <= FMAX)
                except Exception as error:
                    item.update(status="failed", error=repr(error))
            fresh_rows.append(item)
        row["fresh_checks"] = fresh_rows
        row["fresh_requests"] = fresh_surface.requests
        row["fresh_denials"] = fresh_surface.denials
        row["fresh_boundary"] = fresh_surface.boundary
        row["total_requests"] = row["quench_requests"] + row["fresh_requests"]
        row["elapsed_seconds"] = time.monotonic() - started_all
        row["status"] = "completed" if row["prequench_only_quench"]["status"] == "completed" else "failed"
        ledger.dump(folder / "result.json", {"input": case, "result": row,
                     "model": str(MODEL), "model_sha256": MODEL_SHA256, "head": "omol",
                     "device": "cuda", "dtype": "float64", "fmax_eV_A": FMAX,
                     "optimizer": "safe-lbfgs-total", "lbfgs_memory": LBFGS_MEMORY,
                     "maxiter": MAXITER})
        rows.append(row)
        ledger.dump(HERE / "runs" / "summary.json", {
            "status": "running", "completed_cases": len(rows), "case_count": 12,
            "total_request_cap": TOTAL_REQUEST_CAP, "requests_so_far": sum(x["total_requests"] for x in rows),
            "rows": rows,
        })
        if time.monotonic() >= deadline:
            break
    ledger.dump(HERE / "runs" / "summary.json", {
        "status": "complete" if len(rows) == 12 else "wall_censored",
        "completed_cases": len(rows), "case_count": 12, "total_request_cap": TOTAL_REQUEST_CAP,
        "requests_so_far": sum(x["total_requests"] for x in rows), "rows": rows,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preflight", action="store_true", help="CPU-only export and field/count checks")
    group.add_argument("--execute", action="store_true", help="execute the prepared 12-case GPU ablation")
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.execute:
        execute()
    else:
        print("prepared_not_executed: use --preflight for CPU export; --execute runs GPU calculations")


if __name__ == "__main__":
    main()
