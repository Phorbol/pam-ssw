#!/usr/bin/env python3
"""Thin phase-139 input/seed binding for the reviewed atomic-period runner."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE_RUNNER_PATH = HERE.parent / "vc-atomic-contribution-20261012" / "run.py"
PHASE139_SOURCE = Path("/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/literature/benchmark-sources/coordinates/phase-139.extxyz")
PHASE139_SHA256 = "38b00df3777900257a1be1e6e10c680fb5d91a441c78e6594c5dba7dee11a8c6"
PHASE139_CASE_ID = "phase139_48"
SEEDS = (26101251, 26101252)


def base_module():
    spec = importlib.util.spec_from_file_location("vc_atomic_phase87_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import reviewed runner {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_plan():
    base = base_module()
    plan = copy.deepcopy(base.load_plan())
    control = next(row for row in plan["cases"] if row["id"] == "phase87_48")
    identity_source = plan.pop("source_qualification", None)
    plan["cases"] = [{"id": PHASE139_CASE_ID, "path": str(PHASE139_SOURCE),
                       "sha256": PHASE139_SHA256, "natoms": 48,
                       "paired_seed": SEEDS[0]}]
    plan["scope"] = "paired full block-VC input-transfer experiment from phase139_48; shared configurations remain frozen"
    plan["transfer_provenance"] = {
        "control_case_id": control["id"],
        "control_input": {"path": control["path"], "sha256": control["sha256"], "natoms": control["natoms"]},
        "phase139_input": {"path": str(PHASE139_SOURCE), "sha256": PHASE139_SHA256, "natoms": 48},
        "control_source_qualification_provenance_only": identity_source,
        "changed_experimental_factors": ["input geometry", "paired random seeds"],
        "unchanged_constructor_configurations": True,
    }
    return plan


def preflight():
    base = base_module()
    plan = build_plan()
    result = base.preflight(plan, PHASE139_CASE_ID)
    from ase.io import read
    sys.path.insert(0, str(base.ROOT))
    from pamssw.standalone.periodic_ga_reference import pymatgen_identity
    raw = read(PHASE139_SOURCE)
    control = read(plan["transfer_provenance"]["control_input"]["path"])
    anatase = read(plan["target"]["path"])
    raw_comparisons = {}
    for label, settings in plan["identity_tolerances"].items():
        matcher = pymatgen_identity(**settings)
        raw_comparisons[label] = {"raw_phase139_vs_phase87": bool(matcher(raw, control)),
                                  "raw_phase139_vs_anatase": bool(matcher(raw, anatase)),
                                  "scope": "geometry identity only; not minimum or phase qualification"}
    result.update({"transfer_seed_pair": list(SEEDS),
                   "control_input": plan["transfer_provenance"]["control_input"],
                   "phase139_source_expected_sha256": PHASE139_SHA256,
                   "raw_input_geometry_identity": raw_comparisons,
                   "real_pes_requests": 0})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.execute == args.preflight:
        parser.error("choose exactly one of --preflight or --execute")
    if args.preflight:
        print(json.dumps(preflight(), indent=2, allow_nan=False))
        return
    if args.out is None:
        parser.error("--execute requires --out")
    base = base_module()
    base.execute(args.out.resolve(), plan=build_plan(), case_id=PHASE139_CASE_ID,
                 seeds=SEEDS, protocol_path=HERE / "protocol.md",
                 entrypoint_path=Path(__file__).resolve())


if __name__ == "__main__":
    main()
