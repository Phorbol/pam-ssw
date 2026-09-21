"""Bounded LASP external E/F contract helper.

Research-only helper.  It deliberately supports fixed-cell energy/forces only:
no constraints, stress, cell relaxation, or public PAM-SSW API is involved.
The existing ``LASP_MACE_SOCKET`` environment variable is retained because the
LASP external shell script already uses it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading
from typing import Any

import numpy as np
from ase import Atoms


def parse_external_coord(raw: str) -> dict[str, Any]:
    """Parse the observed three-cell-row plus atom-row external.coord format."""
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) < 4:
        raise ValueError("external.coord needs three cell rows and atoms")
    try:
        cell = np.asarray([[float(x) for x in lines[i].split()] for i in range(3)], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("external.coord cell rows are not numeric 3-vectors") from exc
    if cell.shape != (3, 3) or not np.isfinite(cell).all():
        raise ValueError("external.coord cell must be finite 3x3")
    symbols, positions, serials = [], [], []
    for line in lines[3:]:
        fields = line.split()
        if len(fields) < 4:
            raise ValueError("external.coord atom row needs symbol, xyz")
        try:
            xyz = [float(x) for x in fields[1:4]]
            serial = int(fields[4]) if len(fields) >= 5 else len(serials) + 1
        except (TypeError, ValueError) as exc:
            raise ValueError("external.coord atom row is malformed") from exc
        if not np.isfinite(xyz).all():
            raise ValueError("external.coord contains nonfinite coordinates")
        symbols.append(fields[0]); positions.append(xyz); serials.append(serial)
    if not positions:
        raise ValueError("external.coord contains no atoms")
    return {"cell": cell, "symbols": tuple(symbols),
            "positions": np.asarray(positions, dtype=float), "serials": tuple(serials)}


def encode_external_result(energy: float, forces: Any, natoms: int) -> str:
    """Encode the observed external.ene energy/force contract."""
    energy = float(energy)
    forces = np.asarray(forces, dtype=float)
    if not math.isfinite(energy) or forces.shape != (natoms, 3) or not np.isfinite(forces).all():
        raise ValueError("calculator must return finite energy and an (N,3) force array")
    return f"{energy:.17g}\n" + "\n".join(
        " ".join(f"{value:.17g}" for value in row) for row in forces
    ) + "\n"


def _materialize_template(template: Atoms, parsed: dict[str, Any], pbc) -> Atoms:
    """Apply a fixed-cell request while retaining template-side ASE state."""
    if tuple(parsed["symbols"]) != tuple(template.get_chemical_symbols()):
        raise ValueError("requested composition differs from fixed template")
    atoms = template.copy()
    atoms.set_positions(parsed["positions"])
    atoms.set_cell(parsed["cell"], scale_atoms=False)
    atoms.pbc = tuple(bool(x) for x in pbc)
    if atoms.constraints:
        raise ValueError("constraints are outside this helper contract")
    return atoms


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        service = self.server.service  # type: ignore[attr-defined]
        raw_coord = None
        parsed = None
        try:
            request = json.loads(self.rfile.readline())
            raw_coord = request["coord"]
            parsed = parse_external_coord(raw_coord)
            if len(service.log) >= service.max_requests:
                raise RuntimeError("external request cap reached")
            if not np.allclose(parsed["cell"], service.cell, rtol=0., atol=service.cell_atol):
                raise ValueError("requested fixed cell differs from template")
            atoms = _materialize_template(service.template, parsed, service.pbc)
            atoms.calc = service.calculator
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            encode_external_result(energy, forces, len(atoms))
            response = {"ok": True, "energy": float(energy),
                        "forces": np.asarray(forces, dtype=float).tolist()}
            service.log.append({"request": len(service.log) + 1,
                                "raw_coord": raw_coord,
                                "cell": parsed["cell"].tolist(),
                                "positions": parsed["positions"].tolist(),
                                "response": response})
            self.wfile.write((json.dumps(response) + "\n").encode())
        except Exception as error:
            response = {"ok": False, "error": repr(error)}
            failure = dict(response, raw_coord=raw_coord)
            if parsed is not None:
                failure.update(cell=parsed["cell"].tolist(),
                               positions=parsed["positions"].tolist())
            service.errors.append(failure)
            self.wfile.write((json.dumps(response) + "\n").encode())


class _Service(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(self, address, calculator, template, pbc, max_requests, cell_atol=1e-8):
        super().__init__(address, _Handler)
        self.service = self
        self.calculator = calculator
        self.template = template.copy()
        self.symbols = tuple(self.template.get_chemical_symbols())
        self.cell = np.asarray(self.template.cell.array, dtype=float)
        self.pbc = tuple(bool(x) for x in pbc)
        self.max_requests = int(max_requests)
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        self.cell_atol = float(cell_atol)
        if not math.isfinite(self.cell_atol) or self.cell_atol < 0:
            raise ValueError("cell_atol must be finite and nonnegative")
        self.log, self.errors = [], []


def run_lasp(command, *, cwd, atoms, calculator, pbc, max_requests=100,
             env=None, cell_atol=1e-8):
    """Run one already-prepared LASP external case under a bounded supervisor.

    ``command`` must be an argv sequence for an existing bounded wrapper (for
    example ``bounded_process.py --timeout 60 -- lasp``).  This helper never
    owns process supervision: the wrapper must reap LASP/MPI descendants.  The
    caller owns the LASP input directory;
    this function only supplies the socket service and returns an in-memory
    audit record.  It does not submit jobs or write public experiment state.
    """
    if atoms.constraints:
        raise ValueError("constraints are outside this helper contract")
    if len(atoms) == 0 or not np.isfinite(atoms.cell.array).all():
        raise ValueError("template must be nonempty with a finite cell")
    requested_pbc = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,))
    if not np.array_equal(requested_pbc, atoms.pbc):
        raise ValueError("pbc must match the Atoms template")
    if atoms.cell.rank != 3:
        raise ValueError("LASP external requires a full storage cell")
    scaled = atoms.get_scaled_positions(wrap=False)
    nonperiodic = ~atoms.pbc
    if np.any((scaled[:, nonperiodic] < 0) | (scaled[:, nonperiodic] >= 1)):
        raise ValueError("nonperiodic input must lie inside its storage cell; center the cluster and regenerate LASP input")
    if not isinstance(command, (tuple, list)) or not command:
        raise TypeError("command must be a nonempty argv sequence")
    with tempfile.TemporaryDirectory(prefix="lasp-ase-") as tmp:
        address = str(Path(tmp) / "eval.sock")
        server = _Service(address, calculator, atoms, pbc, max_requests, cell_atol)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        child_env = os.environ.copy() if env is None else dict(env)
        child_env["LASP_MACE_SOCKET"] = address
        started = __import__("time").monotonic()
        state, returncode = "launch_failed", None
        try:
            completed = subprocess.run(list(command), cwd=str(cwd), env=child_env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       check=False, text=True)
            state, returncode = "completed", completed.returncode
            output = completed.stdout
            return {"state": state, "returncode": returncode,
                    "wall_seconds": __import__("time").monotonic() - started,
                    "output": output, "requests": list(server.log),
                    "errors": list(server.errors)}
        finally:
            server.shutdown(); worker.join(timeout=5); server.server_close()


def self_test():
    """Parser/encoder regression checks; performs no calculator or LASP call."""
    parsed = parse_external_coord("1 0 0\n0 2 0\n0 0 3\n Cu 1 2 3 1\n")
    assert parsed["cell"].shape == (3, 3) and parsed["positions"].shape == (1, 3)
    assert encode_external_result(1.25, [[1., 2., 3.]], 1).splitlines()[0] == "1.25"
    try:
        encode_external_result(float("nan"), [[1., 2., 3.]], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite calculator energy was accepted")
    template = Atoms("Cu", positions=[[0., 0., 0.]], cell=np.eye(3), pbc=False)
    template.info["fixture"] = "preserve"
    template.set_initial_charges([0.25])
    template.set_initial_magnetic_moments([1.5])
    applied = _materialize_template(template, parsed, (False, False, False))
    assert applied.info["fixture"] == "preserve"
    assert applied.get_initial_charges()[0] == 0.25
    assert applied.get_initial_magnetic_moments()[0] == 1.5
    for bad in ("", "1 0 0\n0 1 0\n0 0 1\nX bad"):
        try:
            parse_external_coord(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed external.coord was accepted")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test(); print("lasp_external_ase self-test: passed"); return
    raise SystemExit("import run_lasp() with a prepared Atoms template and Calculator")


if __name__ == "__main__":
    main()
