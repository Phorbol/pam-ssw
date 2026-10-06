"""Capped native periodic C60 source-3 reference runner."""
from __future__ import annotations

import hashlib
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import vacuum_geometry
from vacuum_geometry import inspect_vacuum

HERE = Path(__file__).resolve().parent
CELL = np.eye(3) * 50.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_external_coord(raw: str):
    from ase import Atoms
    lines = [line.split() for line in raw.splitlines() if line.strip()]
    if len(lines) < 63:
        raise ValueError("external.coord must contain 3 cell rows and 60 atoms")
    parsed_cell = np.asarray([[float(v) for v in row] for row in lines[:3]], dtype=float)
    if parsed_cell.shape != (3, 3) or not np.allclose(parsed_cell, CELL, rtol=0.0, atol=1e-8):
        raise ValueError("external.coord cell is not the required 50 A diagonal cell")
    rows = lines[3:]
    if len(rows) != 60 or any(len(row) < 4 or row[0] != "C" for row in rows):
        raise ValueError("expected 60 C coordinate rows after three cell rows")
    positions = np.asarray([[float(v) for v in row[1:4]] for row in rows], dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("nonfinite external coordinates")
    return Atoms("C60", positions=positions, cell=CELL, pbc=True)


def main(out: Path) -> None:
    plan = json.loads((out / "plan.json").read_text())
    case = plan["case"]
    if (out / "request.jsonl").exists() or (out / "summary.json").exists():
        raise FileExistsError("preserve previous run; prepare a new output directory")
    module_path = Path(vacuum_geometry.__file__).resolve()
    if module_path.parent != out:
        raise RuntimeError(f"geometry helper was not imported from frozen package: {module_path}")
    for name, expected in plan["frozen_source"].items():
        if sha256(out / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    gate = json.loads((out / plan["vacuum_equivalence_result"]).read_text())
    if gate.get("all_rows_pass") is not True:
        raise RuntimeError("vacuum equivalence gate is not all_rows_pass")
    qualification_summary = out / plan["source3_input_qualification_summary"]
    if sha256(qualification_summary) != plan["source3_input_qualification_summary_sha256"]:
        raise RuntimeError("source3 native input qualification summary changed")
    for key in ("model", "binary"):
        path = Path(plan[key])
        if sha256(path) != plan[key + "_sha256"]:
            raise ValueError(key + " changed")

    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mace.calculators import MACECalculator

    calc = MACECalculator(model_paths=plan["model"], head=plan["head"], device=plan["device"],
                          default_dtype=plan["dtype"], enable_cueq=False, enable_oeq=False)
    actual_rmax = float(calc.r_max)
    requests = []
    paid = 0
    failed = False
    lock = threading.Lock()
    actual_calculate_calls = 0
    original_calculate = calc.calculate

    def counted_calculate(*args, **kwargs):
        nonlocal actual_calculate_calls
        with lock:
            actual_calculate_calls += 1
        return original_calculate(*args, **kwargs)

    calc.calculate = counted_calculate

    def persist(record):
        with lock:
            record["request"] = len(requests) + 1
            requests.append(record)
            with (out / "request.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            nonlocal paid, failed
            started = time.monotonic()
            calculate_calls_before = actual_calculate_calls
            record = {"case": case, "paid_ef": False, "cost": {"paid_ef": 0},
                      "model_r_max_A": actual_rmax}
            try:
                raw_request = self.rfile.readline()
                request = json.loads(raw_request)
                record["external_coord"] = request["coord"]
                if failed:
                    record["error_kind"] = "previous_failure"
                    raise RuntimeError("case already stopped after a geometry or backend failure")
                atoms = parse_external_coord(request["coord"])
                record.update(positions=atoms.positions.tolist(), cell=atoms.cell.array.tolist(), pbc=atoms.pbc.tolist())
                canonical, diagnostics = inspect_vacuum(atoms, actual_rmax)
                record["geometry_gate"] = diagnostics
                if not diagnostics["eligible"]:
                    failed = True
                    record["error_kind"] = "geometry"
                    raise RuntimeError("geometry gate failed: " + diagnostics["reason"])
                del canonical
                # Atomic server-side guard before every calculator entry. This is
                # independent of the LASP client; failed E/F calls consume a slot.
                with lock:
                    if failed:
                        record["error_kind"] = "model"
                        raise RuntimeError("case stopped after an earlier backend failure")
                    if paid >= int(plan["request_cap"]):
                        record["error_kind"] = "request_cap"
                        raise RuntimeError("per-case paid E/F request cap")
                    paid += 1
                    record["paid_ef"] = True
                    record["cost"] = {"paid_ef": 1, "paid_index": paid}
                atoms.calc = calc
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                if not np.isfinite(energy) or not np.isfinite(forces).all():
                    record["error_kind"] = "model"
                    raise ValueError("nonfinite MH-1 E/F")
                record.update(energy=energy, forces=forces.tolist(), error_kind=None)
                response = {"ok": True, "energy": energy, "forces": forces.tolist()}
            except Exception as error:
                record.setdefault("error_kind", "model" if record["paid_ef"] else "parse")
                if record["paid_ef"]:
                    with lock:
                        failed = True
                record.update(ok=False, error=repr(error))
                response = {"ok": False, "error": repr(error)}
            record["elapsed_seconds"] = time.monotonic() - started
            record["actual_calculate_calls"] = actual_calculate_calls - calculate_calls_before
            persist(record)
            self.wfile.write((json.dumps(response) + "\n").encode())

    with tempfile.TemporaryDirectory(dir=Path.home(), prefix=".lm-") as temp:
        address = str(Path(temp) / "eval.sock")
        if len(os.fsencode(address)) >= 108:
            raise RuntimeError("Unix socket path is too long")
        with socketserver.UnixStreamServer(address, Handler) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            env = os.environ.copy()
            env["LASP_MACE_SOCKET"] = address
            env["LD_LIBRARY_PATH"] = plan["mpi_lib"] + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
            env.update(I_MPI_FABRICS="shm", I_MPI_PIN="0")
            started = time.monotonic()
            command = [sys.executable, str(out / "bounded_process.py"), "--cwd", str(out),
                       "--timeout", str(plan["wall_seconds"]), "--log", str(out / "stdout.txt"),
                       "--status", str(out / "process.json"), "--",
                       "/lib64/ld-linux-x86-64.so.2", plan["binary"]]
            # The input-specific LASP file is fully staged and frozen before the process starts.
            subprocess.run(command, env=env, check=False)
            status = json.loads((out / "process.json").read_text()) if (out / "process.json").exists() else None
            lasp = (out / "lasp.out").read_text() if (out / "lasp.out").exists() else ""
            summary = {"case": case, "seed": plan["seed"], "process": status,
                       "requests": len(requests), "paid_ef": paid,
                       "actual_calculate_calls": actual_calculate_calls,
                       "geometry_failed": any(r.get("error_kind") == "geometry" for r in requests),
                       "backend_failed": any(r.get("paid_ef") and r.get("ok") is False for r in requests),
                       "ssw_done": "SSW all done" in lasp,
                       "elapsed_seconds": time.monotonic() - started,
                       "wall_cap_seconds": plan["wall_seconds"]}
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (out / "runtime-provenance.json").write_text(json.dumps({
                "python": sys.executable, "runner": str(Path(__file__).resolve()),
                "runner_sha256": sha256(Path(__file__).resolve()), "vacuum_geometry": str(module_path),
                "vacuum_geometry_sha256": sha256(module_path), "model": plan["model"],
                "model_sha256": sha256(Path(plan["model"])), "binary": plan["binary"],
                "binary_sha256": sha256(Path(plan["binary"])), "mace_calculator": str(Path(sys.modules["mace.calculators"].__file__).resolve()),
                "lasp_external_ase": str((out / "lasp_external_ase.py").resolve()),
                "lasp_external_ase_sha256": sha256(out / "lasp_external_ase.py"),
                "client": str((out / "client.py").resolve()), "client_sha256": sha256(out / "client.py"),
                "supervisor": str((out / "bounded_process.py").resolve()),
                "supervisor_sha256": sha256(out / "bounded_process.py"),
                "graph_helper": str((out / "graph_helper.py").resolve()),
                "graph_helper_sha256": sha256(out / "graph_helper.py"),
                "input_sha256": sha256(out / "input.extxyz"), "plan_sha256": sha256(out / "plan.json"),
                "lasp_input_sha256": sha256(out / "lasp.in"), "arc_sha256": sha256(out / "input.arc"),
            }, indent=2) + "\n")
            server.shutdown()
            worker.join()


if __name__ == "__main__":
    main(HERE)
