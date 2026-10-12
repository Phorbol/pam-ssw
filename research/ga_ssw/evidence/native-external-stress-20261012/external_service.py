"""Private current-cell ASE EMT service for the native stress diagnostic."""
from __future__ import annotations

import json
import math
from pathlib import Path
import socketserver
import threading

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT

from research.ga_ssw.lasp_external_ase import parse_external_coord


class CountedEMT(EMT):
    def __init__(self):
        super().__init__()
        self.actual_calculations = 0

    def calculate(self, *args, **kwargs):
        self.actual_calculations += 1
        return super().calculate(*args, **kwargs)


class CurrentCellService(socketserver.UnixStreamServer):
    """One fresh, auditable E/F/full-stress EMT calculation per socket call."""
    allow_reuse_address = True

    def __init__(self, address: str, log_path: Path, cap: int = 20):
        super().__init__(address, Handler)
        self.log_path = Path(log_path)
        self.cap = int(cap)
        self.received_requests = 0
        self.attempts = 0
        self.actual_calculations = 0
        self.failure_latched = False
        self.lock = threading.Lock()

    def append(self, row):
        with self.lock, self.log_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        service: CurrentCellService = self.server  # type: ignore[assignment]
        request = None
        calc = None
        response = None
        service.received_requests += 1
        row = {"request": service.received_requests}
        try:
            request = json.loads(self.rfile.readline())
            parsed = parse_external_coord(request["coord"])
            cell = np.asarray(parsed["cell"], dtype=float)
            positions = np.asarray(parsed["positions"], dtype=float)
            symbols = tuple(parsed["symbols"])
            if len(symbols) != 4 or symbols != ("Cu",) * 4:
                raise ValueError("native request must contain exactly four Cu atoms")
            if not np.isfinite(cell).all() or not np.isfinite(positions).all():
                raise ValueError("native request contains nonfinite geometry")
            if abs(float(np.linalg.det(cell))) < 1e-8:
                raise ValueError("native request cell is singular")
            if service.failure_latched:
                raise RuntimeError("previous paid EMT call failed; callback latched closed")
            if service.attempts >= service.cap:
                raise RuntimeError("external EMT EFS cap reached")

            # The request slot is charged immediately before calculation; a
            # failed calculator call remains a paid/attempted slot.
            service.attempts += 1
            atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            calc = CountedEMT()
            atoms.calc = calc
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            stress = np.asarray(atoms.get_stress(voigt=False), dtype=float)
            if (not math.isfinite(energy) or forces.shape != (4, 3) or
                    stress.shape != (3, 3) or not np.isfinite(forces).all() or
                    not np.isfinite(stress).all()):
                raise ValueError("ASE EMT returned nonfinite or malformed E/F/stress")
            wire_stress = -stress
            row.update({"status": "calculated", "cell_A": cell.tolist(),
                        "positions_A": positions.tolist(), "energy_eV": energy,
                        "forces_eV_A": forces.tolist(),
                        "ase_stress_eV_A3_tensile": stress.tolist(),
                        "wire_stress_eV_A3_hypothesis": wire_stress.tolist(),
                        "wire_order": "Fortran column-major", "wire_values":
                            wire_stress.flatten(order="F").tolist(),
                        "actual_calculator_calls": calc.actual_calculations})
            response = {"ok": True, "energy": energy, "forces": forces.tolist(),
                        "stress_wire": wire_stress.flatten(order="F").tolist()}
        except Exception as error:
            service.failure_latched = True
            row.update({"status": "failed_or_denied", "error": repr(error)})
        finally:
            if calc is not None:
                calls = int(calc.actual_calculations)
                service.actual_calculations += calls
                row["actual_calculator_calls"] = calls
            service.append(row)
            if response is None:
                response = {"ok": False, "error": row.get("error", "failed without response")}
            self.wfile.write((json.dumps(response, allow_nan=False) + "\n").encode())
