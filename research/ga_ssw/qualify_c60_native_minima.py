#!/usr/bin/env python3
"""Extract native C60 minima and optionally fresh-qualify them with MACE.

``--prepare NATIVE_DIR --output OUT_DIR`` parses only ``lasp.out`` minimum
events and streams the request ledger to recover matching coordinates.  It
does not relax or change any criterion.  ``--execute OUT_DIR`` is the later
GPU-only fresh MACE qualification stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "research/ga_ssw/evidence/c60-random-inputs-development-20260917"
MODEL = Path("/home/gengjianrui/.cache/mace/mace-omat-0-small.model")
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
EVENT_RE = re.compile(
    r"Minimum found\s+(\d+)\s+(\d+)\s+([-+0-9.]+)\s+([-+0-9.]+).*?\s(F|T)\s+"
    r"[-+0-9.]+\s+([-+0-9.]+).*?\s(\d+)\s*$"
)


def dump(path: Path, value, *, exclusive=False):
    with path.open("x" if exclusive else "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def append(path: Path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def backend_from_native_plan(native_dir: Path):
    plan_path = native_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"native input plan is missing: {plan_path}")
    frozen = json.loads(plan_path.read_text())
    raw = frozen.get("backend") if isinstance(frozen.get("backend"), dict) else {}
    model = frozen.get("model") or raw.get("model")
    expected = frozen.get("model_sha256") or raw.get("model_sha256")
    head = frozen.get("head") or raw.get("head")
    device = frozen.get("device") or raw.get("device") or "cuda"
    dtype = frozen.get("dtype") or raw.get("dtype") or raw.get("default_dtype") or "float64"
    metadata_keys = {"backend", "model", "model_sha256", "head", "device", "dtype"}
    metadata_present = any(key in frozen for key in metadata_keys)
    if not model and not expected:
        if metadata_present:
            raise ValueError(f"native input plan has incomplete backend metadata: {plan_path}")
        model, expected, head = str(MODEL), MODEL_SHA256, None
        device, dtype = "cuda", "float64"
        source = "legacy-omat-fallback"
    elif not model or not expected:
        raise ValueError(f"native input plan has incomplete backend metadata: {plan_path}")
    else:
        source = str(plan_path)
    model_path = Path(model)
    if not model_path.is_file():
        raise FileNotFoundError(f"backend model is missing: {model_path}")
    actual = sha256(model_path)
    if actual != expected:
        raise RuntimeError(f"backend model hash mismatch: expected {expected}, got {actual}")
    return {"name": raw.get("name", frozen.get("backend", "MACE")),
            "model": str(model_path), "model_sha256": actual, "head": head,
            "device": device, "dtype": dtype, "source_plan": source}


def parse_events(native_case: Path, limit=1001):
    events, cumulative = [], 0
    lasp = native_case / "lasp.out"
    if not lasp.is_file():
        return events
    with lasp.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            match = EVENT_RE.search(line)
            if not match:
                continue
            delta = int(match.group(7))
            cumulative += delta
            events.append({"ordinal": int(match.group(1)),
                           "native_previous": int(match.group(2)),
                           "event_energy_eV": float(match.group(4)),
                           "event_force_component": float(match.group(6)),
                           "cumulative_request": cumulative,
                           "lasp_line": line_number,
                           "lasp_text": line.rstrip()})
            if len(events) >= limit:
                break
    return events


def base_atoms(case):
    from ase.io import read
    return read(INPUT_ROOT / f"seed{case.removeprefix('seed')}.extxyz")


def prepare(native_dir: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if not native_dir.is_dir():
        raise FileNotFoundError(native_dir)
    cases = sorted(p for p in native_dir.iterdir() if p.is_dir() and p.name.startswith("seed"))
    if {p.name for p in cases} != {"seed17093", "seed17094"}:
        raise ValueError("native directory must contain seed17093 and seed17094")
    output.mkdir(parents=True)
    (output / "events").mkdir()
    backend = backend_from_native_plan(native_dir)
    plan = {"scope": "CPU extraction only; no local relaxation or criterion changes",
            "native_dir": str(native_dir.resolve()), "cases": [p.name for p in cases],
            "limit_per_case": 1001,
            "criteria": {"energy_tolerance_eV": 1e-6, "max_atom_force_norm_eV_per_A": .03,
                          "pbc": False, "dtype": "float64"},
            "backend": backend,
            "runtime": {"torch_manual_seed": 0, "torch_deterministic_algorithms": True,
                        "torch_num_threads": 1, "tf32": False,
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}}
    all_meta = []
    for case_dir in cases:
        case = case_dir.name
        events = parse_events(case_dir)
        target = {event["cumulative_request"]: event for event in events}
        matched_path = output / "events" / f"{case}.extxyz"
        metadata_path = output / "events" / f"{case}.jsonl"
        base = base_atoms(case)
        matched = 0
        seen_case_requests = 0
        with (native_dir / "requests.jsonl").open() as ledger:
            for line_number, line in enumerate(ledger, 1):
                row = json.loads(line)
                if row.get("case") != case or not row.get("response", {}).get("ok"):
                    continue
                seen_case_requests += 1
                event = target.get(seen_case_requests)
                if event is None:
                    continue
                response = row["response"]
                positions = response.get("positions")
                if positions is None:
                    positions = row.get("positions")
                if positions is None:
                    event = dict(event, matched=False, reason="ledger_missing_positions",
                                 ledger_line=line_number)
                    append(metadata_path, event)
                    continue
                energy = float(response['energy'])
                forces = response.get('forces')
                component = max(abs(float(x)) for xyz in forces for x in xyz) if forces else None
                if abs(energy - event['event_energy_eV']) > 5.1e-7 or component is None or abs(component - event['event_force_component']) > .00051:
                    append(metadata_path, dict(event, matched=False, reason='event_ledger_mismatch',
                        ledger_energy_eV=energy, ledger_component_force_eV_per_A=component))
                    continue
                from ase import Atoms
                atoms = Atoms(numbers=base.numbers, positions=positions,
                              cell=base.cell, pbc=False)
                atoms.info["native_case"] = case
                atoms.info["native_request"] = int(row.get("request", seen_case_requests))
                atoms.info["native_event_energy_eV"] = event["event_energy_eV"]
                atoms.info["native_event_force_component"] = event["event_force_component"]
                from ase.io import write
                write(matched_path, atoms, format="extxyz", append=matched > 0)
                energy = float(response["energy"])
                forces = response.get("forces")
                component = max(abs(float(x)) for xyz in forces for x in xyz) if forces else None
                metadata = dict(event, matched=True, ledger_line=line_number,
                                ledger_request=int(row.get("request", seen_case_requests)),
                                ledger_energy_eV=energy,
                                ledger_component_force_eV_per_A=component,
                                energy_delta_eV=energy - event["event_energy_eV"])
                append(metadata_path, metadata)
                matched += 1
        summary = {"case": case, "event_count": len(events), "matched_count": matched,
                   "unmatched_count": len(events) - matched,
                   "ledger_case_successful_requests": seen_case_requests,
                   "native_search_cost": {"successful_EF_requests": seen_case_requests,
                                          "fresh_EF_requests": 0,
                                          "price": None, "price_note": "scheduler pricing not supplied"},
                   "matched_events": str(metadata_path.relative_to(output)),
                   "matched_structures": str(matched_path.relative_to(output)) if matched else None}
        dump(output / "events" / f"{case}.summary.json", summary, exclusive=True)
        all_meta.append(summary)
    plan["native_dir_sha256"] = {case: sha256(native_dir / case / "lasp.out") for case in plan["cases"]}
    plan["input_sha256"] = {case: sha256(INPUT_ROOT / f"{case}.extxyz") for case in plan["cases"]}
    dump(output / "plan.json", plan, exclusive=True)
    dump(output / "prepare-summary.json", all_meta, exclusive=True)
    print(json.dumps({"prepared": str(output.resolve()), "cases": all_meta}, indent=2))


def execute(output: Path):
    if not output.is_dir() or not (output / "plan.json").is_file():
        raise FileNotFoundError("run --prepare first")
    plan = json.loads((output / "plan.json").read_text())
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", plan["runtime"]["CUBLAS_WORKSPACE_CONFIG"])
    import numpy as np
    import torch
    torch.set_num_threads(plan["runtime"]["torch_num_threads"])
    torch.manual_seed(plan["runtime"]["torch_manual_seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import ASESurface
    backend = plan.get("backend")
    if not isinstance(backend, dict):
        raise ValueError("prepared plan lacks backend metadata")
    model = Path(backend["model"])
    if not model.is_file() or sha256(model) != backend["model_sha256"]:
        raise RuntimeError("prepared backend model missing or hash mismatch")
    calculator_kwargs = {"model_paths": str(model), "device": backend.get("device", "cuda"),
                         "default_dtype": backend.get("dtype", "float64"),
                         "enable_cueq": False, "enable_oeq": False}
    if backend.get("head") is not None:
        calculator_kwargs["head"] = backend["head"]
    rows = []
    for case in plan["cases"]:
        structures = output / "events" / f"{case}.extxyz"
        metadata = output / "events" / f"{case}.jsonl"
        fresh = ASESurface(MACECalculator(**calculator_kwargs))
        checks_path = output / "events" / f"{case}.fresh.jsonl"
        if checks_path.exists():
            raise FileExistsError(f"refusing to overwrite {checks_path}")
        checks_path.touch()
        meta_rows = [row for line in metadata.open() if line.strip() for row in [json.loads(line)] if row.get("matched")]
        fresh_requests = 0
        for index, atoms in enumerate(read(structures, ":") if structures.exists() else []):
            meta = meta_rows[index]
            try:
                energy, forces = fresh.evaluate(atoms)
                fmax = float(np.linalg.norm(forces, axis=1).max())
                energy_delta = energy - meta["event_energy_eV"]
                row = {"case": case, "index": index, "ordinal": meta["ordinal"],
                       "cumulative_request": meta["cumulative_request"], "native_event_energy_eV": meta["event_energy_eV"],
                       "fresh_energy_eV": energy, "energy_delta_eV": energy_delta,
                       "fresh_fmax_eV_per_A": fmax,
                       "energy_match": bool(abs(energy_delta) <= 1e-6),
                       "force_match": bool(fmax <= .03), "qualified": bool(abs(energy_delta) <= 1e-6 and fmax <= .03),
                       "pbc": atoms.pbc.tolist(), "dtype": "float64"}
            except Exception as error:
                row = {"case": case, "index": index, "qualified": False, "error": repr(error)}
            append(checks_path, row)
            fresh_requests += 1
        summary = json.loads((output / "events" / f"{case}.summary.json").read_text())
        summary.update(fresh_EF_requests=fresh_requests,
                       fresh_qualified=sum(bool(json.loads(line).get("qualified"))
                                            for line in checks_path.open() if line.strip()),
                       native_search_cost=dict(summary["native_search_cost"], fresh_EF_requests=fresh_requests))
        dump(output / "events" / f"{case}.summary.json", summary)
        rows.append(summary)
    dump(output / "execute-summary.json", rows, exclusive=True)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path, metavar="NATIVE_DIR")
    group.add_argument("--execute", type=Path, metavar="PREPARED_DIR")
    parser.add_argument("--output", type=Path, help="new output directory for --prepare")
    args = parser.parse_args()
    if args.prepare is not None:
        if args.output is None:
            parser.error("--prepare requires --output")
        prepare(args.prepare.resolve(), args.output.resolve())
    else:
        if args.output is not None:
            parser.error("--output is valid only with --prepare")
        execute(args.execute.resolve())


if __name__ == "__main__":
    main()
