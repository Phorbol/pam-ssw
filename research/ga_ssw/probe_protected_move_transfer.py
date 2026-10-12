#!/usr/bin/env python3
"""Run the frozen protected-move transfer panel; default mode is preflight only."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "research/ga_ssw/evidence/protected-move-transfer-20261013"
PROTOCOL = EVIDENCE / "protocol.md"
BASE = ROOT / "research/ga_ssw/probe_protected_move_paper_cases.py"
MODEL_C60 = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_TIO2 = Path("/home/gengjianrui/.cache/mace/mace-omat-0-small.model")


def input_specs(data_root: Path):
    data_root = Path(data_root).expanduser().resolve()
    return {
    "c60-isomer1": {
        "input": data_root / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-1/final.extxyz",
        "input_sha256": "6a5a4a34abdf618aaaf45be77b0492c4f2fef61b104eb361c9d0030b5e598355",
        "qualification": data_root / "research/ga_ssw/evidence/c60-local-defect-20260925/qualification/isomer-1/result.json",
        "qualification_sha256": "693a23f7e216b26deeabb13956d1e421ebd4db8bf40688b9b94336f1795a57e9",
        "model": MODEL_C60,
        "model_sha256": "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47",
        "head": "omol", "cutoff_A": 3.0, "center": 0, "neighbor_species": (6,),
        "composition": {6: 60}, "periodic": False,
    },
    "tio2-anatase-fs": {
        "input": data_root / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run-1663408/anatase_fs/endpoint.extxyz",
        "input_sha256": "867a034e9e339e273b7a580e397c4245728c6d6ebe9c951a93fc79b9fe6606d8",
        "qualification": data_root / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run-1663408/anatase_fs/result.json",
        "qualification_sha256": "daaa84508a07aa04a3f9f0b5093375d0af4708cdd4b758322122d9b87e7e7645",
        "manifest": data_root / "research/ga_ssw/evidence/tio2-phase-qualification-20261007/run-1663408/effective-plan.json",
        "manifest_sha256": "007fcfcc903efe956c4a9291b50f4917c83bc65dec74421616155b52117de9d1",
        "model": MODEL_TIO2,
        "model_sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
        "head": "omat_pbe", "cutoff_A": 3.0, "center": 8, "neighbor_species": (8,),
        "composition": {8: 8, 22: 4}, "periodic": True,
    },
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(data_root: Path):
    pm = load_module("protected_move_transfer_base", BASE)
    global PROTOCOL
    PROTOCOL = data_root / "research/ga_ssw/evidence/protected-move-transfer-20261013/protocol.md"
    specs = input_specs(data_root)
    pm.PROTOCOL = PROTOCOL
    pm.SEED = 20261013
    original_load_base = pm.load_base

    def transfer_base(root=data_root):
        base = original_load_base(root)
        pm.PROTOCOL = Path(root).expanduser().resolve() / "research/ga_ssw/evidence/protected-move-transfer-20261013/protocol.md"
        base.CASES = {name: dict(spec) for name, spec in specs.items()}

        def validate_inputs():
            from ase.io import read
            prepared = {}
            for name, spec in specs.items():
                for key in ("input", "qualification", "model"):
                    if not spec[key].is_file():
                        raise FileNotFoundError(spec[key])
                if base.sha256(spec["input"]) != spec["input_sha256"]:
                    raise ValueError(f"frozen input hash changed: {spec['input']}")
                if base.sha256(spec["qualification"]) != spec["qualification_sha256"]:
                    raise ValueError(f"qualification record changed: {spec['qualification']}")
                if base.sha256(spec["model"]) != spec["model_sha256"]:
                    raise ValueError(f"frozen model hash changed: {spec['model']}")
                atoms = read(spec["input"], index=0)
                composition = {int(z): int((atoms.numbers == z).sum()) for z in set(atoms.numbers)}
                if composition != spec["composition"] or not __import__("numpy").isfinite(atoms.positions).all():
                    raise ValueError(f"composition or coordinates changed for {name}: {composition}")
                if bool(atoms.pbc.any()) != spec["periodic"] or atoms.constraints:
                    raise ValueError(f"PBC/constraints changed for {name}")
                qualification = json.loads(spec["qualification"].read_text())
                if name == "c60-isomer1":
                    if (not qualification.get("force_qualified") or
                            not all(qualification.get("labeled_edges_preserved", {}).values())):
                        raise ValueError("C60 isomer-1 qualification did not pass")
                else:
                    if (qualification.get("case") != "anatase_fs" or
                            qualification.get("status") != "complete" or
                            not qualification.get("numerical_qualified") or
                            not qualification.get("optimizer_converged") or
                            not all(row.get("qualified") for row in qualification.get("fresh_checks", []))):
                        raise ValueError("anatase FS qualification did not pass")
                    if base.sha256(spec["manifest"]) != spec["manifest_sha256"]:
                        raise ValueError("anatase qualification manifest changed")
                    plan = json.loads(spec["manifest"].read_text())
                    anatase = next(row for row in plan["cases"] if row["id"] == "anatase_fs")
                    if (plan["paper"]["doi"] != "10.1039/c7sc01459g" or
                            plan["model_sha256"] != spec["model_sha256"] or
                            plan["calculator"]["head"] != spec["head"] or
                            anatase["sha256"] != "e1726c173dde3e0007e094f85ab006ae46e14048311867b816ada1faafe331a7"):
                        raise ValueError("anatase paper/model/source manifest mismatch")
                    extxyz = Path(spec["input"]).read_text().splitlines()[1]
                    if ('source_doi=10.1039/c7sc01459g' not in extxyz or
                            'source_section="SI section 7"' not in extxyz or
                            'anatase' not in extxyz.lower()):
                        raise ValueError("anatase endpoint lacks expected paper SI provenance")
                prepared[name] = atoms
            return prepared

        base.validate_inputs = validate_inputs

        original_dump = base.dump

        def transfer_dump(path, value):
            path = Path(path)
            if path.name == "provenance.json":
                out = path.parent
                frozen_wrapper = out / Path(__file__).name
                frozen_protocol = out / "transfer-protocol.md"
                if not frozen_wrapper.exists():
                    import shutil
                    shutil.copy2(Path(__file__).resolve(), frozen_wrapper)
                if not frozen_protocol.exists():
                    import shutil
                    shutil.copy2(pm.PROTOCOL, frozen_protocol)
                value["transfer_wrapper"] = str(Path(__file__).resolve())
                value["transfer_wrapper_sha256"] = base.sha256(Path(__file__).resolve())
                value["transfer_wrapper_snapshot"] = str(frozen_wrapper)
                value["transfer_protocol"] = str(pm.PROTOCOL)
                value["transfer_protocol_sha256"] = base.sha256(pm.PROTOCOL)
                value["transfer_protocol_snapshot"] = str(frozen_protocol)
            return original_dump(path, value)

        pm.dump = transfer_dump
        return base

    pm.load_base = transfer_base
    return pm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="new output directory for --execute/--smoke")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="run frozen one-V100 transfer panel")
    modes.add_argument("--smoke", action="store_true", help="synthetic wiring smoke only")
    args = parser.parse_args()
    pm = configure(args.data_root.resolve())
    if args.execute or args.smoke:
        result = pm.execute(args.out.resolve(), args.data_root.resolve(), smoke=args.smoke)
        print(json.dumps({"status": result["status"], "runs": len(result["runs"]),
                          "search_requests": result["search_requests_total"],
                          "fresh_requests": result["fresh_requests_total"],
                          "scientific_run": result["scientific_run"], "out": str(args.out.resolve())}, indent=2))
    else:
        base = pm.load_base(args.data_root.resolve())
        prepared = pm.api_preflight(base)
        print(json.dumps({"status": "prepared_not_executed", "seed": pm.SEED,
                          "cases": list(input_specs(args.data_root.resolve())), "arms": pm.ARMS,
                          "max_search_requests": 4 * pm.SEARCH_CAP,
                          "max_fresh_requests": 4 * pm.FRESH_CAP,
                          "preflight": prepared,
                          "transfer_wrapper": str(Path(__file__).resolve()),
                          "transfer_wrapper_sha256": pm.sha256(Path(__file__).resolve()),
                          "transfer_protocol": str(pm.PROTOCOL),
                          "transfer_protocol_sha256": pm.sha256(pm.PROTOCOL),
                          "out": str(args.out.resolve())}, indent=2))


if __name__ == "__main__":
    main()
