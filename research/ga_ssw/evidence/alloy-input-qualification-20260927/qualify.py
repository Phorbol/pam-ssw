#!/usr/bin/env python
"""Numerically qualify one Ag30Au30 source geometry and one native exchange."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = "/home/gengjianrui/.cache/mace/mace-omat-0-small.model"
HEAD = "omat_pbe"
SEED = 270927
CALL_CAP = 1000
DEADLINE_SECONDS = 10 * 60
FMAX_EV_A = 0.05
MAX_STEPS = 300
L_BFGS_MEMORY = 500
MIXING_CUTOFFS_A = (3.0, 3.5, 4.0)


class QualificationLimit(RuntimeError):
    pass


class QualificationDeadline(QualificationLimit):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", required=True, type=Path,
                        help="read-only Ag30Au30 source extxyz")
    parser.add_argument("--out", required=True, type=Path,
                        help="new, exclusive output directory")
    return parser.parse_args()


def json_safe(value):
    """Convert non-finite values to JSON null so failures remain serializable."""
    import math
    import numpy as np

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def graph_geometry(atoms, symbols):
    import numpy as np

    positions = np.asarray(atoms.positions, dtype=float)
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    components = {}
    for cutoff in MIXING_CUTOFFS_A:
        adjacency = distances < cutoff
        unseen = set(range(len(atoms)))
        sizes = []
        while unseen:
            start = unseen.pop()
            stack = [start]
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                neighbors = set(np.flatnonzero(adjacency[node]).tolist()) & unseen
                unseen.difference_update(neighbors)
                stack.extend(neighbors)
            sizes.append(size)
        components[f"{cutoff:.1f}"] = {
            "component_count": len(sizes),
            "component_sizes_descending": sorted(sizes, reverse=True),
        }
    counts = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return {
        "n_atoms": len(atoms),
        "composition": counts,
        "symbols_by_atom_index": list(symbols),
        "minimum_pair_distance_A": float(np.min(distances)),
        "bounding_box_extent_A": np.ptp(positions, axis=0).tolist(),
        "coordination_graph_diagnostic_only": {
            "edge_rule": "distance < cutoff",
            "cutoffs_A": list(MIXING_CUTOFFS_A),
            "connected_components": components,
            "interpretation": "cutoff-sensitive geometric description, not a chemical criterion",
        },
    }


def make_calculator():
    import torch
    from ase.calculators.calculator import all_changes
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    class CountedMACE(MACECalculator):
        def __init__(self):
            super().__init__(model_paths=MODEL, head=HEAD, device="cuda",
                             default_dtype="float64", enable_cueq=False,
                             enable_oeq=False)
            self.calls_started = 0
            self.calls_completed = 0
            self.cap_hits = 0

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            if self.calls_started >= CALL_CAP:
                self.cap_hits += 1
                raise QualificationLimit(f"actual calculator call cap reached ({CALL_CAP})")
            self.calls_started += 1
            result = super().calculate(atoms, properties, system_changes)
            self.calls_completed += 1
            return result

    return CountedMACE()


def main():
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "qualify.py").write_bytes(Path(__file__).read_bytes())
    summary_path = out / "summary.json"
    summary = {
        "status": "initializing",
        "purpose": "numerical and geometric qualification on MACE-OMAT-0-small; not a Gupta-GM reproduction",
        "started_unix": time.time(),
        "source_path": str(args.initial.resolve()),
        "output_path": str(out),
        "model": MODEL,
        "head": HEAD,
        "dtype": "float64",
        "device": "cuda",
        "enable_cueq": False,
        "enable_oeq": False,
        "seed": SEED,
        "optimizer": "ASE LBFGSLineSearch",
        "lbfgs_memory": L_BFGS_MEMORY,
        "fmax_eV_A": FMAX_EV_A,
        "max_steps": MAX_STEPS,
        "calculator_call_cap": CALL_CAP,
        "deadline_seconds": DEADLINE_SECONDS,
        "arms": [],
        "calculator_calls_started": 0,
        "calculator_calls_completed": 0,
        "calculator_cap_hits": 0,
    }

    calculator = None
    active_row = None
    fresh_phase_active = False
    fresh_start = None
    fresh_complete = None
    timer_installed = False

    def save_summary():
        summary["updated_unix"] = time.time()
        summary_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n")

    try:
        if hasattr(signal, "setitimer"):
            def deadline_handler(signum, frame):
                raise QualificationDeadline(f"program deadline reached ({DEADLINE_SECONDS}s)")
            signal.signal(signal.SIGALRM, deadline_handler)
            signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
            timer_installed = True

        import ase
        import mace
        import numpy as np
        import torch
        from ase.io import read, write
        from ase.optimize import LBFGSLineSearch

        sys.path.insert(0, str(ROOT))
        from pamssw.standalone.atomic_ga import exchange_atoms

        source = read(args.initial, index=0)
        if len(source) != 60:
            raise ValueError(f"expected 60 atoms, got {len(source)}")
        source_counts = {symbol: source.get_chemical_symbols().count(symbol)
                         for symbol in sorted(set(source.get_chemical_symbols()))}
        if source_counts != {"Ag": 30, "Au": 30}:
            raise ValueError(f"expected Ag30Au30, got {source_counts}")
        if not np.isfinite(source.positions).all():
            raise ValueError("source coordinates contain non-finite values")
        if source.constraints or np.any(source.pbc):
            raise ValueError("expected unconstrained, non-periodic source cluster")

        exchanged, origins = exchange_atoms(source, np.random.default_rng(SEED))
        exchanged.pbc = False
        if not np.array_equal(exchanged.positions, source.positions):
            raise RuntimeError("native exchange unexpectedly changed coordinates")
        if not np.array_equal(np.sort(exchanged.numbers), np.sort(source.numbers)):
            raise RuntimeError("native exchange changed composition")
        if np.array_equal(exchanged.numbers, source.numbers):
            raise RuntimeError("seeded native exchange did not change any species labels")

        summary["software"] = {"python": sys.version.split()[0], "ase": ase.__version__,
                                "mace": mace.__version__, "torch": torch.__version__,
                                "numpy": np.__version__}
        summary["git_head"] = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        summary["source_geometry"] = graph_geometry(source, source.get_chemical_symbols())
        summary["native_exchange_geometry"] = graph_geometry(
            exchanged, exchanged.get_chemical_symbols())
        summary["species_exchange"] = {
            "operator": "pamssw.standalone.atomic_ga.exchange_atoms",
            "calls": 1,
            "rng": "numpy.random.default_rng(270927)",
            "swaps_requested": 10 * len(source),
            "atom_indices_with_changed_species": int(np.count_nonzero(source.numbers != exchanged.numbers)),
            "origin_mapping": list(origins),
            "same_composition": bool(np.array_equal(np.sort(source.numbers),
                                                    np.sort(exchanged.numbers))),
            "same_positions": bool(np.array_equal(source.positions, exchanged.positions)),
            "different_species_assignment": bool(np.any(source.numbers != exchanged.numbers)),
        }
        calculator = make_calculator()
        arms = (("source", source), ("native_exchange", exchanged))
        arm_start = None
        for name, base_atoms in arms:
            folder = out / name
            folder.mkdir(exist_ok=False)
            atoms = base_atoms.copy()
            arm_start = time.monotonic()
            arm_calls_start = calculator.calls_started
            arm_calls_complete = calculator.calls_completed
            row = {"name": name, "status": "started", "folder": str(folder),
                   "calculator_calls_started": 0, "calculator_calls_completed": 0,
                   "fresh_calls_started": 0, "fresh_calls_completed": 0,
                   "optimizer_converged": False, "force_qualified": False}
            summary["arms"].append(row)
            active_row = row
            row["_calls_started_before"] = arm_calls_start
            row["_calls_completed_before"] = arm_calls_complete
            fresh_phase_active = False
            fresh_start = None
            fresh_complete = None
            write(folder / "input.extxyz", atoms, format="extxyz")
            row["initial_geometry"] = graph_geometry(atoms, atoms.get_chemical_symbols())
            row["initial_positions_A"] = atoms.positions.tolist()
            atoms.calc = calculator
            calculator.reset()
            row["initial_energy_eV"] = float(atoms.get_potential_energy())
            initial_forces = np.asarray(atoms.get_forces(), dtype=float).copy()
            row["initial_forces_eV_A"] = initial_forces.tolist()
            row["initial_fmax_eV_A"] = float(np.linalg.norm(initial_forces, axis=1).max())

            optimizer = LBFGSLineSearch(
                atoms, memory=L_BFGS_MEMORY,
                logfile=str(folder / "optimizer.log"),
                trajectory=str(folder / "trajectory.traj"))
            row["optimizer_converged"] = bool(optimizer.run(fmax=FMAX_EV_A, steps=MAX_STEPS))
            row["optimizer_steps"] = int(optimizer.nsteps)
            row["optimizer_terminal_energy_eV"] = float(atoms.get_potential_energy())
            row["optimizer_terminal_fmax_eV_A"] = float(
                np.linalg.norm(atoms.get_forces(), axis=1).max())
            row["final_positions_A"] = atoms.positions.tolist()

            calculator.reset()
            fresh_start = calculator.calls_started
            fresh_complete = calculator.calls_completed
            fresh_phase_active = True
            final_energy = float(atoms.get_potential_energy())
            final_forces = np.asarray(atoms.get_forces(), dtype=float).copy()
            final_fmax = float(np.linalg.norm(final_forces, axis=1).max())
            write(folder / "final.extxyz", atoms, format="extxyz")
            row.update({"final_energy_eV": final_energy,
                        "final_forces_eV_A": final_forces.tolist(),
                        "final_fmax_eV_A": final_fmax,
                        "force_qualified": bool(np.isfinite(final_fmax) and final_fmax <= FMAX_EV_A),
                        "final_geometry": graph_geometry(atoms, atoms.get_chemical_symbols()),
                        "final_positions_A": atoms.positions.tolist(),
                        "fresh_calls_started": calculator.calls_started - fresh_start,
                        "fresh_calls_completed": calculator.calls_completed - fresh_complete,
                        "status": "complete"})
            fresh_phase_active = False
            row["calculator_calls_started"] = calculator.calls_started - arm_calls_start
            row["calculator_calls_completed"] = calculator.calls_completed - arm_calls_complete
            row["elapsed_seconds"] = time.monotonic() - arm_start
            row.pop("_calls_started_before", None)
            row.pop("_calls_completed_before", None)
            summary["calculator_calls_started"] = calculator.calls_started
            summary["calculator_calls_completed"] = calculator.calls_completed
            summary["calculator_cap_hits"] = calculator.cap_hits
            save_summary()
            active_row = None

        summary.update({"status": "complete",
                        "same_initial_composition": True,
                        "native_exchange_changed_assignment": summary["species_exchange"][
                            "different_species_assignment"],
                        "interpretation_limit": (
                            "small forces qualify local numerical termination only; they do not establish "
                            "a Gupta global minimum, MACE global minimum, or agreement between potentials")})
        save_summary()
        return 0
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        summary["traceback"] = traceback.format_exc()
        if calculator is not None:
            summary["calculator_calls_started"] = calculator.calls_started
            summary["calculator_calls_completed"] = calculator.calls_completed
            summary["calculator_cap_hits"] = calculator.cap_hits
        if active_row is not None:
            active_row["status"] = "failed"
            active_row["error_type"] = type(exc).__name__
            active_row["error"] = str(exc)
            if calculator is not None:
                active_row["calculator_calls_started"] = (
                    calculator.calls_started - active_row.get("_calls_started_before", 0))
                active_row["calculator_calls_completed"] = (
                    calculator.calls_completed - active_row.get("_calls_completed_before", 0))
                if fresh_phase_active and fresh_start is not None and fresh_complete is not None:
                    active_row["fresh_calls_started"] = calculator.calls_started - fresh_start
                    active_row["fresh_calls_completed"] = calculator.calls_completed - fresh_complete
            if arm_start is not None:
                active_row["elapsed_seconds"] = time.monotonic() - arm_start
            active_row.pop("_calls_started_before", None)
            active_row.pop("_calls_completed_before", None)
        try:
            save_summary()
        except Exception as save_error:
            print(f"summary write failed: {save_error!r}", file=sys.stderr)
        raise
    finally:
        if timer_installed:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    sys.exit(main())
