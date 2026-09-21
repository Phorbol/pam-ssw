#!/usr/bin/env python3
"""Zero-PES integrity check for the prepared composition experiment."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
LONG = HERE.parent / "c60-recovered-rotation-long-20260921"
LS_PLAN = HERE.parent / "mh1-native-ls-equal-budget-20260920" / "plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(label, got, expected, checks):
    if got != expected:
        raise AssertionError(f"{label}: {got!r} != {expected!r}")
    checks[label] = "matched"


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    long = json.loads((LONG / "plan.json").read_text())
    ls = json.loads(LS_PLAN.read_text())["native_ls"]
    checks = {}

    same("cases", plan["cases"], long["cases"], checks)
    same("seeds", plan["seeds"], long["seeds"], checks)
    for key in ("model", "model_sha256", "head", "dtype", "device", "ssw_config",
                "native_mc", "recovered_rotation", "outer_steps", "search_cap",
                "wall_seconds", "runtime"):
        same(key, plan[key], long[key], checks)
    same("native_ls", plan["native_ls"], ls, checks)
    same("source_target", (HERE / "source").resolve(), (LONG / "source").resolve(), checks)
    if sha256(plan["model"]) != plan["model_sha256"]:
        raise AssertionError("model hash changed")
    checks["model_hash"] = "matched"

    for rel, expected in plan["source_sha256"].items():
        same(f"source:{rel}", sha256(HERE / rel), expected, checks)
    for case, expected in plan["input_sha256"].items():
        path = HERE / "inputs" / f"{case}.traj"
        same(f"input:{case}", sha256(path), expected, checks)
        atoms = read(path)
        if len(atoms) != 60 or not np.all(atoms.numbers == 6):
            raise AssertionError(f"{case}: expected 60 carbon atoms")
        if atoms.pbc.any() or not np.allclose(atoms.cell.array, np.diag([50., 50., 50.])):
            raise AssertionError(f"{case}: fixed nonperiodic 50 A cell contract failed")
        if np.any(atoms.positions < 20.0) or np.any(atoms.positions > 30.0):
            raise AssertionError(f"{case}: translated input bounds failed")
        checks[f"geometry:{case}"] = "matched"

    for rel, expected in plan["harness_sha256"].items():
        same(f"harness:{rel}", sha256(HERE / rel), expected, checks)

    from runner import make_ls, SSWConfig, RecoveredRotationSettings
    make_ls(plan)
    SSWConfig(**plan["ssw_config"])
    RecoveredRotationSettings(**plan["recovered_rotation"])
    checks["settings_construction"] = "passed_without_pes"

    result = {"status": "passed", "pes_initialized": False,
              "python": sys.executable, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "checks": len(checks)}
    (HERE / f"preflight-{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
