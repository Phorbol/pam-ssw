#!/usr/bin/env python3
"""Prepare or execute one bounded native external full-stress diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "research/ga_ssw/evidence/native-vc-lifecycle-trace-20261012"
BINARY = Path("/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp")
EXPECTED_BINARY_SHA256 = "bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704"
MPI_LIB = "/opt/devtools/intel/oneapi/mpi/2021.13/lib"
PYTHON = Path("/home/gengjianrui/.conda/envs/mace_env/bin/python")
CAP = 20
NATIVE_TIMEOUT = 120
SLURM_SECONDS = 300
SEED = 20261012
DEFORMATION = np.array([[1.0080, 0.0020, 0.0000],
                        [0.0000, 0.9930, -0.0015],
                        [0.0005, 0.0000, 1.0120]])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def fcc_cu4():
    from ase.build import bulk
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
    if len(atoms) != 4:
        raise RuntimeError("ASE fcc conventional cell did not produce Cu4")
    cell = np.asarray(atoms.cell.array, dtype=float)
    positions = np.asarray(atoms.positions, dtype=float)
    atoms.set_cell(cell @ DEFORMATION.T, scale_atoms=False)
    atoms.set_positions(positions @ DEFORMATION.T)
    # ARC stores lengths/angles only: rotate the cell and positions together
    # into ASE's canonical cellpar representation before serialization.
    from ase.cell import Cell
    atoms.set_cell(Cell.fromcellpar(atoms.cell.cellpar()), scale_atoms=True)
    atoms.pbc = True
    atoms.wrap()
    if atoms.constraints or atoms.get_chemical_formula() != "Cu4" or atoms.cell.rank != 3:
        raise RuntimeError("Cu4 geometry violates frozen diagnostic contract")
    return atoms


def arc_text(atoms) -> str:
    rows = ["!BIOSYM archive 2", "PBC=ON", "Energy 0 0.0 0.0", "!DATE",
            "PBC " + " ".join(f"{x:.14f}" for x in atoms.cell.cellpar())]
    for i, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions), 1):
        rows.append(f"{symbol} {pos[0]:.14f} {pos[1]:.14f} {pos[2]:.14f} CORE {i} {symbol} {symbol} 0.0 {i}")
    rows.extend(["end", "end", ""])
    return "\n".join(rows)


def native_input() -> str:
    return "\n".join([
        "potential external", "explore_type ssw", "Ewaldflag 0", "Run_Type 15",
        "external.dedcell F", f"ranseed {SEED}", "SSW.quick_setting 1",
        "SSW.SSWsteps 2", "SSW.MaxOptstep 50", "SSW.ftol 0.05", "SSW.strtol 0.05",
        "SSW.Temp 200", "SSW.NG 3", "SSW.ds_atom 0.6", "SSW.NG_cell 3",
        "SSW.ds_cell 1.6", "SSW.Ratio_atomcell 2", "SSW.mdfreq 0",
        "SSW.output T", "SSW.printevery T", ""])


def prepare(out: Path):
    if out.exists():
        raise FileExistsError(f"run directory must be new: {out}")
    if not BINARY.is_file() or sha(BINARY) != EXPECTED_BINARY_SHA256:
        raise RuntimeError("uploaded LASP binary missing or differs from frozen ELF")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    from ase.calculators.emt import EMT
    if "stress" not in EMT.implemented_properties:
        raise RuntimeError("installed ASE EMT does not advertise stress")
    atoms = fcc_cu4()
    out.mkdir(parents=True)
    executable = out / "lasp"
    shutil.copyfile(BINARY, executable)
    if sha(executable) != EXPECTED_BINARY_SHA256:
        raise RuntimeError("diagnostic executable copy hash mismatch")
    executable.chmod(0o700)
    from ase.io import write
    write(out / "input.extxyz", atoms)
    (out / "input.arc").write_text(arc_text(atoms))
    (out / "lasp.in").write_text(native_input())
    for name in ("trace.py", "trace.gdb", "external_service.py", "external_client.py",
                 "protocol.md", "run.py", "cpu.sbatch"):
        shutil.copy2(HERE / name, out / name)
    (out / "lasp.external.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"{PYTHON} \"$(dirname \"$0\")/external_client.py\"\n")
    (out / "lasp.external.sh").chmod(0o755)
    shutil.copy2(ROOT / "research/ga_ssw/bounded_process.py", out / "bounded_process.py")
    source_files = {name: HERE / name for name in
                    ("run.py", "protocol.md", "trace.py", "trace.gdb",
                     "external_service.py", "external_client.py", "cpu.sbatch")}
    provenance = {
        "status": "prepared_not_executed", "question": "native external full-stress handoff only",
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "binary": str(BINARY), "binary_sha256": sha(BINARY),
        "executed_binary": str(executable), "executed_binary_sha256": sha(executable),
        "input": {"formula": atoms.get_chemical_formula(), "natoms": len(atoms),
                  "pbc": atoms.pbc.tolist(), "cell_A": atoms.cell.array.tolist(),
                  "positions_A": atoms.positions.tolist(), "deformation": DEFORMATION.tolist(),
                  "a_fcc_A": 3.6, "sha256": sha(out / "input.extxyz")},
        "potential": "ASE EMT", "stress_property": list(EMT.implemented_properties),
        "external_dedcell_input": False, "wire_stress_hypothesis": "minus ASE full stress",
        "budget": {"arms": 1, "external_ef_stress_cap": CAP,
                   "native_process_seconds": NATIVE_TIMEOUT, "slurm_seconds": SLURM_SECONDS,
                   "stop": "third class_struc_mp_stress2dedlatt entry or first failure/cap/timeout"},
        "native_input": native_input(),
        "source_hashes": {name: sha(path) for name, path in source_files.items()},
        "references": {"key_reset_slices_commit": "deb93e4",
                       "consumer_abi_oracle": "research/ga_ssw/probe_native_cell_force.py",
                       "external_tensor_contract": "docs/research/native-stress-backend-boundary.md"},
        "numeric_ef_calls": 0}
    dump(out / "provenance.json", provenance)
    return provenance


def execute(out: Path):
    provenance_path = out / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("status") != "prepared_not_executed":
        raise ValueError("only a freshly prepared, unexecuted directory may run")
    if sha(out / "lasp") != EXPECTED_BINARY_SHA256:
        raise RuntimeError("staged native binary changed")
    for name, expected in provenance["source_hashes"].items():
        if sha(HERE / name) != expected or sha(out / name) != expected:
            raise RuntimeError(f"frozen source hash changed: {name}")
    spec = importlib.util.spec_from_file_location("native_external_service", out / "external_service.py")
    service_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service_module)
    with tempfile.TemporaryDirectory(prefix="nxst-") as sockdir:
        socket_path = Path(sockdir) / "s"
        service = service_module.CurrentCellService(str(socket_path), out / "callback-events.jsonl", CAP)
        server_thread = threading.Thread(target=service.serve_forever, daemon=True)
        server_thread.start()
        env = os.environ.copy()
        env["LASP_MACE_SOCKET"] = str(socket_path)
        env["LD_LIBRARY_PATH"] = MPI_LIB + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
        env.update(I_MPI_FABRICS="shm", I_MPI_PIN="0", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        command = [sys.executable, str(out / "bounded_process.py"), "--cwd", str(out),
                   "--timeout", str(NATIVE_TIMEOUT), "--log", str(out / "child.log"),
                   "--status", str(out / "process.json"), "--", "gdb", "-q", "-nx",
                   "-batch", "-x", "trace.gdb", str(out / "lasp")]
        provenance["status"] = "execution_requested"
        provenance["command"] = command
        provenance["numeric_ef_calls"] = 0
        dump(provenance_path, provenance)
        started = time.monotonic()
        try:
            completed = subprocess.run(command, cwd=out, env=env, check=False)
            provenance["supervisor_returncode"] = completed.returncode
            provenance["process_status"] = (json.loads((out / "process.json").read_text())
                                             if (out / "process.json").exists() else None)
        finally:
            service.shutdown()
            server_thread.join(timeout=5)
            service.server_close()
            provenance["elapsed_seconds"] = time.monotonic() - started
            provenance["callback_requests_received"] = service.received_requests
            provenance["external_requests"] = service.attempts
            provenance["actual_calculator_calls"] = service.actual_calculations
            provenance["numeric_ef_calls"] = service.actual_calculations
            provenance["callback_error_latched"] = service.failure_latched
            stop_record = (json.loads((out / "trace-stop.json").read_text())
                           if (out / "trace-stop.json").exists() else None)
            provenance["trace_stop"] = stop_record
            if service.failure_latched:
                provenance["status"] = "callback_failed_or_denied"
            elif (not provenance.get("process_status") or
                  provenance["process_status"].get("state") != "completed" or
                  provenance["process_status"].get("returncode") != 0 or
                  provenance["process_status"].get("cleanup_survivors")):
                provenance["status"] = "process_failed_timed_out_or_unclean"
            elif (out / "trace-error.json").exists():
                provenance["status"] = "observer_read_error"
            elif stop_record and stop_record.get("reason") == "third_class_stress2dedlatt_entry":
                provenance["status"] = "stopped_at_third_class_entry"
            else:
                provenance["status"] = "incomplete_without_expected_stop"
            dump(provenance_path, provenance)
    return 0 if provenance["status"] == "stopped_at_third_class_entry" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--execute", type=Path)
    args = parser.parse_args()
    path = args.prepare or args.execute
    path = path.expanduser().resolve()
    if args.prepare:
        print(json.dumps({"status": prepare(path)["status"], "run_dir": str(path)}))
        return 0
    return execute(path)


if __name__ == "__main__":
    raise SystemExit(main())
