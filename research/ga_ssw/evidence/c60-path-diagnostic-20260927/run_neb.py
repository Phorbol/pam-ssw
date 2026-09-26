#!/usr/bin/env python
"""Bounded endpoint-informed NEB diagnostic for C60 defect connectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np


MODEL = "/home/gengjianrui/.cache/mace/mace-mh-1.model"
CALL_CAP = 3000
DEADLINE_SECONDS = 18 * 60
N_IMAGES = 7
K_EV_A2 = 0.1
FMAX_EV_A = 0.05
PHASE_STEPS = {"ordinary": 50, "climbing": 150}


class DiagnosticLimit(RuntimeError):
    pass


class DiagnosticDeadline(DiagnosticLimit):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True, type=Path)
    parser.add_argument("--final", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--calculator", choices=("mace", "emt"), default="mace")
    parser.add_argument("--smoke", action="store_true",
                        help="use one FIRE step per phase; valid only with EMT")
    parser.add_argument("--call-cap", type=int, default=CALL_CAP)
    parser.add_argument("--deadline-seconds", type=int, default=DEADLINE_SECONDS)
    return parser.parse_args()


def build_counted_calculator(kind, call_cap):
    if kind == "emt":
        from ase.calculators.calculator import all_changes
        from ase.calculators.emt import EMT

        class CountedEMT(EMT):
            def __init__(self):
                super().__init__()
                self.calls_started = 0
                self.calls_completed = 0
                self.cap_hits = 0

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                if self.calls_started >= call_cap:
                    self.cap_hits += 1
                    raise DiagnosticLimit(f"calculator call cap reached ({call_cap})")
                self.calls_started += 1
                result = super().calculate(atoms, properties, system_changes)
                self.calls_completed += 1
                return result

        return CountedEMT()

    from mace.calculators import MACECalculator
    from ase.calculators.calculator import all_changes

    class CountedMACE(MACECalculator):
        def __init__(self):
            super().__init__(model_paths=MODEL, head="omol", device="cuda",
                             default_dtype="float64", enable_cueq=False,
                             enable_oeq=False)
            self.calls_started = 0
            self.calls_completed = 0
            self.cap_hits = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            if self.calls_started >= call_cap:
                self.cap_hits += 1
                raise DiagnosticLimit(f"calculator call cap reached ({call_cap})")
            self.calls_started += 1
            result = super().calculate(atoms, properties, system_changes)
            self.calls_completed += 1
            return result

    return CountedMACE()


def atoms_metadata(atoms):
    return {"natoms": len(atoms), "symbols": atoms.get_chemical_symbols(),
            "pbc": atoms.pbc.tolist(), "cell_A": atoms.cell.array.tolist()}


def validate_pair(initial, final, smoke):
    if len(initial) != len(final):
        raise ValueError(f"endpoint atom counts differ: {len(initial)} vs {len(final)}")
    if not np.array_equal(initial.numbers, final.numbers):
        raise ValueError("endpoint species/order differ; use the separately qualified mapping")
    for name, atoms in (("initial", initial), ("final", final)):
        if not np.isfinite(atoms.positions).all():
            raise ValueError(f"{name} positions contain non-finite values")
        if atoms.constraints:
            raise ValueError(f"{name} has constraints; diagnostic expects free isolated endpoints")
        if np.any(atoms.pbc):
            raise ValueError(f"{name} is periodic; diagnostic expects isolated endpoints")
    if not smoke and (len(initial) != 60 or not np.all(initial.numbers == 6)):
        raise ValueError("MACE C60 diagnostic requires exactly 60 carbon atoms")


def phase_record(name, optimizer, converged, max_steps, started, calculator):
    return {"name": name, "converged": bool(converged), "steps": int(optimizer.nsteps),
            "step_limit": int(max_steps), "elapsed_seconds": time.monotonic() - started,
            "calculator_calls_started_at_end": int(calculator.calls_started),
            "calculator_calls_completed_at_end": int(calculator.calls_completed)}


def main():
    args = parse_args()
    if args.smoke and args.calculator != "emt":
        raise ValueError("--smoke is restricted to --calculator emt")
    if args.call_cap < 1 or args.deadline_seconds < 1:
        raise ValueError("call cap and deadline must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    summary_path = out / "summary.json"
    summary = {
        "status": "initializing", "diagnostic": "endpoint-informed NEB; not SSW",
        "started_unix": time.time(), "initial_path": str(args.initial.resolve()),
        "final_path": str(args.final.resolve()), "calculator": args.calculator,
        "model": MODEL if args.calculator == "mace" else None,
        "backend_head": "omol" if args.calculator == "mace" else None,
        "dtype": "float64" if args.calculator == "mace" else None,
        "device": "cuda" if args.calculator == "mace" else "cpu",
        "call_cap": args.call_cap, "deadline_seconds": args.deadline_seconds,
        "n_images": N_IMAGES, "spring_k_eV_A2": K_EV_A2,
        "neb_method": "improvedtangent", "interpolation": "idpp",
        "fmax_eV_A": FMAX_EV_A,
        "phase_step_limits": ({"ordinary": 1, "climbing": 1} if args.smoke
                               else PHASE_STEPS),
        "phases": [], "total_calculator_calls_started": 0,
        "total_calculator_calls_completed": 0, "calculator_cap_hits": 0,
        "initial": None, "final": None,
    }

    def save_summary():
        summary["updated_unix"] = time.time()
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    timer_installed = False
    calculator = None
    try:
        if args.deadline_seconds and hasattr(signal, "setitimer"):
            def deadline_handler(signum, frame):
                raise DiagnosticDeadline(f"program deadline reached ({args.deadline_seconds}s)")
            signal.signal(signal.SIGALRM, deadline_handler)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_seconds)
            timer_installed = True

        import ase
        import torch
        from ase.io import Trajectory, read, write
        from ase.mep import NEB
        from ase.mep.neb import idpp_interpolate
        from ase.optimize import FIRE

        if args.calculator == "mace":
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
        summary["software"] = {"ase": ase.__version__, "numpy": np.__version__,
                               "torch": torch.__version__}
        initial = read(args.initial, index=0)
        final = read(args.final, index=0)
        validate_pair(initial, final, args.smoke)
        initial.pbc = False
        final.pbc = False
        summary["initial"] = atoms_metadata(initial)
        summary["final"] = atoms_metadata(final)
        summary["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()

        calculator = build_counted_calculator(args.calculator, args.call_cap)
        images = [initial.copy()]
        images.extend(initial.copy() for _ in range(N_IMAGES - 2))
        images.append(final.copy())
        for image in images:
            image.calc = calculator
        neb = NEB(images, k=K_EV_A2, climb=False, method="improvedtangent",
                  allow_shared_calculator=True)
        # Preserve ASE's existing IDPP defaults while directing its own
        # optimizer artifacts into this run directory.
        neb.interpolate(method="linear")
        idpp_interpolate(neb, traj=str(out / "idpp.traj"),
                         log=str(out / "idpp.log"))
        with Trajectory(out / "neb.traj", "w", atoms=neb) as trajectory:
            phase_specs = (("ordinary", False, args.smoke and 1 or PHASE_STEPS["ordinary"]),
                           ("climbing", True, args.smoke and 1 or PHASE_STEPS["climbing"]))
            for name, climb, steps in phase_specs:
                neb.climb = climb
                phase_start = time.monotonic()
                optimizer = FIRE(neb, logfile=str(out / f"{name}.log"))
                optimizer.attach(trajectory.write, interval=1, atoms=neb)
                converged = optimizer.run(fmax=FMAX_EV_A, steps=steps)
                # Ensure the terminal geometry is recorded even if run() took
                # zero steps or stopped without firing its interval observer.
                trajectory.write(neb)
                summary["phases"].append(phase_record(
                    name, optimizer, converged, steps, phase_start, calculator))
                save_summary()

        # Direct calculate() forces a new evaluation of every terminal image,
        # counted by the same calculator cap, and avoids shared-calculator
        # results being mistaken for another image's properties.
        image_rows = []
        path_images = [image.copy() for image in images]
        for index, image in enumerate(images):
            calculator.calculate(image, properties=["energy", "forces"])
            energy = float(calculator.results["energy"])
            forces = np.asarray(calculator.results["forces"], dtype=float).copy()
            image_rows.append({"image": index, "energy_eV": energy,
                               "physical_fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
                               "physical_force_rms_eV_A": float(np.sqrt(np.mean(forces ** 2))),
                               "positions_A": image.positions.tolist(),
                               "forces_eV_A": forces.tolist()})
            path_images[index].info["energy"] = energy
            path_images[index].arrays["forces"] = forces

        neb_force = np.asarray(neb.get_forces(), dtype=float)
        neb_fmax = float(np.linalg.norm(neb_force.reshape((-1, 3)), axis=1).max())
        write(out / "path.extxyz", path_images, format="extxyz")
        energies = [row["energy_eV"] for row in image_rows]
        summary.update({"status": "completed",
                        "total_calculator_calls_started": calculator.calls_started,
                        "total_calculator_calls_completed": calculator.calls_completed,
                        "calculator_cap_hits": calculator.cap_hits,
                        "images": image_rows, "endpoint_delta_eV": energies[-1] - energies[0],
                        "path_energy_span_eV": max(energies) - min(energies),
                        "highest_image_index": int(np.argmax(energies)),
                        "neb_projected_fmax_eV_A": neb_fmax,
                        "neb_fmax_qualified": bool(np.isfinite(neb_fmax) and neb_fmax <= FMAX_EV_A),
                        "max_physical_fmax_eV_A": max(r["physical_fmax_eV_A"] for r in image_rows),
                        "interpretation_limit": "discretized path only; highest image is not a certified TS"})
        save_summary()
        return 0
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        if calculator is not None:
            summary["total_calculator_calls_started"] = calculator.calls_started
            summary["total_calculator_calls_completed"] = calculator.calls_completed
            summary["calculator_cap_hits"] = calculator.cap_hits
        try:
            save_summary()
        except Exception:
            pass
        raise
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    sys.exit(main())
