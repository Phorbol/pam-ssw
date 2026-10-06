#!/usr/bin/env python3
"""Bounded explicit-LASP-ranseed interface probe; not a physical search."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INPUT = ROOT / "research/ga_ssw/evidence/c60-ls-source3-qualification-20261007/qualification-1664282/final-candidate.extxyz"
SOURCE = ROOT / "research/ga_ssw/evidence/c60-source3-lasp-input-qualification-20261007/run-1664642"
HELPER_LIVE = ROOT / "research/ga_ssw/lasp_external_ase.py"
BINARY = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")
PYTHON = Path("/home/gengjianrui/.conda/envs/mace_env/bin/python")
MPI_LIB = "/opt/devtools/intel/oneapi/mpi/2021.13/lib"
CELL = np.eye(3) * 50.0
SEEDS = (26100791, 26100791, 26100792)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Quadratic(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, reference):
        super().__init__()
        self.reference = np.asarray(reference, dtype=float).copy()
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        delta = atoms.positions - self.reference
        self.calls += 1
        self.results = {"energy": 0.5 * float(np.sum(delta * delta)), "forces": -delta}


def stage(out: Path, atoms, seed: int, prep):
    prep.write_slot_inputs(out, atoms)
    inp = out / "lasp.in"
    text = inp.read_text().replace("SSW.SSWsteps 1", "SSW.SSWsteps 2").replace(
        "SSW.MaxOptstep 1", "SSW.MaxOptstep 10")
    text += f"ranseed {seed}\n"
    inp.write_text(text)


def counted_command(out: Path, argv, seconds: float):
    return [sys.executable, str(out / "bounded_process.py"), "--cwd", str(out),
            "--timeout", str(seconds), "--log", str(out / "child.log"),
            "--status", str(out / "process.json"), "--", *argv]


def preflight() -> None:
    """Exercise copied socket client, bounded process and E/F service using a fake driver."""
    for path in (INPUT, SOURCE / "qualify.py", SOURCE / "client.py", SOURCE / "bounded_process.py", HELPER_LIVE):
        if not path.is_file():
            raise FileNotFoundError(path)
    old_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix=".seedpf-", dir=Path.home()) as td:
        root = Path(td)
        tempfile.tempdir = str(root)
        periodic = read(INPUT); periodic.pbc = True
        prep = load_module(SOURCE / "qualify.py", "seed_probe_prep")
        shutil.copy2(HELPER_LIVE, root / "lasp_external_ase.py")
        shutil.copy2(HERE / "probe.py", root / "probe.py")
        shutil.copy2(HERE / "cpu.sbatch", root / "cpu.sbatch")
        shutil.copy2(HERE / "protocol.md", root / "protocol.md")
        for name in ("client.py", "bounded_process.py"):
            shutil.copy2(SOURCE / name, root / name)
        dump(root / "source-manifest.json", {
            "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                     capture_output=True, text=True, check=True).stdout.strip(),
            "sources": {name: {"path": str(path), "sha256": sha(path)} for name, path in {
                "probe.py": HERE / "probe.py", "cpu.sbatch": HERE / "cpu.sbatch",
                "protocol.md": HERE / "protocol.md", "runner.py": SOURCE / "qualify.py",
                "validator.py": HELPER_LIVE, "input.extxyz": INPUT}.items()}})
        helper = load_module(root / "lasp_external_ase.py", "seed_probe_helper")
        rows = []
        for index, seed in enumerate(SEEDS):
            case = root / f"case-{index}"
            case.mkdir()
            stage(case, periodic, seed, prep)
            shutil.copy2(root / "client.py", case / "client.py")
            shutil.copy2(root / "bounded_process.py", case / "bounded_process.py")
            dump(case / "provenance.json", {"mode": "synthetic preflight", "seed": seed,
                 "input": str(INPUT), "input_sha256": sha(INPUT), "native_lasp": False,
                 "mace_calls": 0, "physical_pes_calls": 0})
            (case / "allkeys.log").write_text(f"ranseed {seed}\nSSW.SSWsteps 2\nSSW.MaxOptstep 10\n")
            # The fake driver makes bounded, seed-derived coordinate requests
            # through the copied production callback client; it is not LASP.
            driver = case / "fake_driver.py"
            driver.write_text(
                "import os,subprocess,sys\nfrom pathlib import Path\nimport numpy as np\n"
                "p=Path.cwd(); rows=(p/'input.arc').read_text().splitlines()[5:65]\n"
                "base=np.array([[float(x) for x in r.split()[1:4]] for r in rows])\n"
                f"rng=np.random.default_rng({seed})\n"
                "for i in range(3):\n"
                " pos=base.copy() if i==0 else base+rng.normal(0,.01,base.shape)\n"
                " lines=['50 0 0','0 50 0','0 0 50']+[f'C {x:.14f} {y:.14f} {z:.14f} {j+1}' for j,(x,y,z) in enumerate(pos)]\n"
                " (p/'external.coord').write_text('\\n'.join(lines)+'\\n')\n"
                f" subprocess.run([{str(PYTHON)!r},str(p/'client.py')],check=True,timeout=10)\n"
                " assert (p/'external.ene').is_file()\n"
            )
            calculator = Quadratic(periodic.positions)
            result = helper.run_lasp(counted_command(case, [sys.executable, str(driver)], 30),
                                     cwd=case, atoms=periodic, calculator=calculator,
                                     pbc=[True] * 3, max_requests=8, env=os.environ.copy())
            if result["returncode"] != 0 or len(result["requests"]) != 3 or result["errors"]:
                raise RuntimeError(f"synthetic socket preflight failed for case {index}: {result}")
            (case / "lasp.out").write_text("synthetic fake-driver only; no native LASP invoked\n")
            dump(case / "lasp-callback.json", result)
            dump(case / "summary.json", {"status": "synthetic_preflight", "seed": seed,
                 "synthetic_ef": len(result["requests"]), "mace_calls": 0, "physical_pes_calls": 0})
            rows.append({"case": index, "seed": seed, "requests": len(result["requests"]),
                         "actual_synthetic_calculations": calculator.calls, "process_status": json.loads((case / "process.json").read_text())})
        dump(root / "manifest.json", {"preflight": "passed", "cases": rows,
             "mace_calls": 0, "physical_pes_calls": 0, "native_lasp_calls": 0})
        print(json.dumps({"preflight": "passed", "cases": rows, "temporary_only": True}))
    tempfile.tempdir = old_tempdir


def run_case(out: Path, index: int, seed: int, atoms, prep, helper):
    case = out / f"case-{index}-seed-{seed}"
    case.mkdir(exist_ok=False)
    stage(case, atoms, seed, prep)
    shutil.copy2(out / "client.py", case / "client.py")
    shutil.copy2(out / "bounded_process.py", case / "bounded_process.py")
    # Inputs, output locations and original code snapshots are kept per case.
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = MPI_LIB + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    env.update(I_MPI_FABRICS="shm", I_MPI_PIN="0")
    command = counted_command(case, ["/lib64/ld-linux-x86-64.so.2", str(BINARY)], 30)
    calc = Quadratic(atoms.positions)
    dump(case / "provenance.json", {"input_ranseed": seed, "input_path": str(INPUT),
         "input_sha256": sha(INPUT), "native_binary": str(BINARY),
         "native_binary_sha256": sha(BINARY), "synthetic_calculator": "0.5*sum((R-R0)^2)",
         "calculator_force": "-(R-R0)", "max_callback_requests": 8,
         "process_timeout_seconds": 30, "native_ssw_steps": 2, "native_max_optstep": 10,
         "actual_helper_import_path": str(Path(helper.__file__).resolve()),
         "actual_helper_sha256": sha(Path(helper.__file__).resolve()),
         "actual_runner_snapshot": str(out / "runner.py"), "mace_calls": 0,
         "physical_pes_calls": 0})
    result = helper.run_lasp(command, cwd=case, atoms=atoms, calculator=calc,
                             pbc=[True] * 3, max_requests=8, env=env)
    dump(case / "lasp-callback.json", result)
    if not (case / "allkeys.log").exists():
        (case / "allkeys.log").write_text("LASP did not emit allkeys.log\n")
    for src, dst in ((case / "allkeys.log", case / "allkeys-captured.log"),
                     (case / "lasp.out", case / "lasp-out-captured.log"),
                     (case / "external.coord", case / "external-coord-final"),
                     (case / "external.ene", case / "external-ene-final")):
        if src.is_file():
            shutil.copy2(src, dst)
    keys = (case / "allkeys.log").read_text(errors="replace") if (case / "allkeys.log").exists() else ""
    log = (case / "lasp.out").read_text(errors="replace") if (case / "lasp.out").exists() else ""
    seed_values = re.findall(r"(?:randomseed|random seed)\s*[:=]?\s*(-?\d+)", log, flags=re.I)
    coords = [np.asarray(r["positions"], float) for r in result["requests"]]
    displacements = [float(np.max(np.abs(x - atoms.positions))) for x in coords]
    first_nonzero = next((i for i, d in enumerate(displacements) if d > 1e-4), None)
    rec = {"case": index, "input_ranseed": seed, "requested_key_present_in_allkeys": bool(re.search(
          rf"(?m)^\s*ranseed\s+{seed}\s*$", keys)), "allkeys_log": str(case / "allkeys.log"),
          "printed_native_seed_values": seed_values, "callback_request_count": len(result["requests"]),
          "callback_error_count": len(result["errors"]), "requested_coordinates": [x.tolist() for x in coords],
          "max_component_displacement_from_input_A": displacements,
          "first_nonzero_displacement_request_index0": first_nonzero,
          "synthetic_calculations": calc.calls, "mace_calls": 0, "physical_pes_calls": 0,
          "process_status": json.loads((case / "process.json").read_text()) if (case / "process.json").exists() else None}
    dump(case / "summary.json", rec)
    return rec


def execute(out: Path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("--execute requires Slurm; use --preflight for synthetic-only check")
    tmp = os.environ.get("TMPDIR")
    if not tmp or not Path(tmp).is_dir():
        raise RuntimeError("sbatch must provide a private short TMPDIR")
    tempfile.tempdir = tmp
    if len(os.fsencode(tmp)) + len(os.fsencode("/lasp-ase-XXXXXXXX/eval.sock")) >= 108:
        raise RuntimeError("AF_UNIX socket path exceeds Linux limit")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    shutil.copy2(HERE / "probe.py", out / "probe.py")
    shutil.copy2(HERE / "cpu.sbatch", out / "cpu.sbatch")
    shutil.copy2(HERE / "protocol.md", out / "protocol.md")
    shutil.copy2(SOURCE / "qualify.py", out / "runner.py")
    shutil.copy2(INPUT, out / "source-input.extxyz")
    frozen = {"probe.py": out / "probe.py", "cpu.sbatch": out / "cpu.sbatch",
              "protocol.md": out / "protocol.md", "runner.py": out / "runner.py",
              "validator.py": out / "lasp_external_ase.py", "input.extxyz": out / "source-input.extxyz",
              "client.py": out / "client.py", "bounded_process.py": out / "bounded_process.py"}
    shutil.copy2(SOURCE / "client.py", out / "client.py")
    shutil.copy2(SOURCE / "bounded_process.py", out / "bounded_process.py")
    shutil.copy2(HELPER_LIVE, out / "lasp_external_ase.py")
    prep = load_module(out / "runner.py", "seed_probe_prep_exec")
    atoms = read(out / "source-input.extxyz")
    prep.validate_atoms(atoms)
    if sha(out / "source-input.extxyz") != prep.EXPECTED_INPUT_SHA or sha(BINARY) != prep.EXPECTED_BINARY_SHA:
        raise ValueError("qualified source#3 input or LASP binary changed")
    atoms.pbc = True
    helper = load_module(out / "lasp_external_ase.py", "seed_probe_helper_exec")
    source_record = {name: {"path": str(path), "sha256": sha(path)} for name, path in frozen.items()}
    source_record["original_paths"] = {"runner.py": str(SOURCE / "qualify.py"),
         "validator.py": str(HELPER_LIVE), "input.extxyz": str(INPUT),
         "client.py": str(SOURCE / "client.py"), "bounded_process.py": str(SOURCE / "bounded_process.py")}
    source_record["git_head"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    dump(out / "source-manifest.json", source_record)
    records = []
    for i, seed in enumerate(SEEDS):
        records.append(run_case(out, i, seed, atoms, prep, helper))
    first, repeat, other = records
    def coordinate_compare(a, b):
        n = min(len(a["requested_coordinates"]), len(b["requested_coordinates"]))
        maxima = [float(np.max(np.abs(np.asarray(a["requested_coordinates"][i]) -
                                       np.asarray(b["requested_coordinates"][i])))) for i in range(n)]
        ia, ib = a["first_nonzero_displacement_request_index0"], b["first_nonzero_displacement_request_index0"]
        first_delta = None
        if ia is not None and ib is not None and ia < len(a["requested_coordinates"]) and ib < len(b["requested_coordinates"]):
            da = np.asarray(a["requested_coordinates"][ia]) - atoms.positions
            db = np.asarray(b["requested_coordinates"][ib]) - atoms.positions
            first_delta = float(np.max(np.abs(da - db)))
        return {"shared_prefix_requests": n, "max_component_differences_A": maxima,
                "equal_at_1e-4_A": [v <= 1e-4 for v in maxima],
                "first_nonzero_displacement_indices0": [ia, ib],
                "first_nonzero_displacement_max_component_difference_A": first_delta,
                "first_nonzero_displacement_equal_at_1e-4_A":
                    (None if first_delta is None else first_delta <= 1e-4)}
    summary = {"status": "completed_probe", "scope": "synthetic quadratic, early direction requests only",
        "cases": records, "duplicate_seed_coordinate_comparison": coordinate_compare(first, repeat),
        "different_seed_coordinate_comparison": coordinate_compare(first, other),
        "elapsed_seconds": time.monotonic() - started, "mace_calls": 0, "physical_pes_calls": 0,
        "global_rng_independence_claim": False, "global_search_claim": False}
    summary["total_synthetic_calculations"] = sum(r["synthetic_calculations"] for r in records)
    if summary["total_synthetic_calculations"] > 24:
        raise ValueError("synthetic calculation bound exceeded")
    dump(out / "summary.json", summary)


def main():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", type=Path)
    a = p.parse_args()
    if a.preflight: preflight()
    else: execute(a.execute)


if __name__ == "__main__": main()
