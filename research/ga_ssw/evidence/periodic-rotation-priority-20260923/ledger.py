"""Bounded paired direction-utility experiment (prepared; execution is opt-in).

The two arms share the same Gaussian, quench, MC, budget, seed and backend
settings.  They differ only in direction generation: the complete recovered
Run5 controller versus the recovered rotation-only controller.  This runner
does not generate inputs or copy model files.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT / "source"
MODEL_C60 = Path("/home/gengjianrui/.cache/mace/mace-mh-1.model")
MODEL_CU55 = Path("/home/gengjianrui/.cache/mace/mace-omat-0-small.model")


def _jsonable(value):
    from ase import Atoms
    if isinstance(value, Atoms):
        return {"numbers": value.numbers.tolist(), "positions": value.positions.tolist(),
                "cell": value.cell.array.tolist(), "pbc": value.pbc.tolist()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {str(k): _jsonable(v) for k, v in vars(value).items()}
    return value


def dump(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(_jsonable(value), allow_nan=False) + "\n")


class CountedSurface:
    def __init__(self, calculator, ledger, cap, wall):
        self.calculator, self.ledger = calculator, ledger
        self.cap, self.wall = cap, wall
        self.requests = self.denials = 0
        self.started = time.monotonic()
        self.boundary = None

    def evaluate(self, atoms):
        if self.requests >= self.cap or time.monotonic() - self.started >= self.wall:
            self.denials += 1
            self.boundary = "request_cap" if self.requests >= self.cap else "wall_cap"
            append(self.ledger, {"kind": "search_denial", "request": self.requests,
                                 "reason": self.boundary, "atoms": atoms})
            raise RuntimeError(self.boundary)
        self.requests += 1
        work = atoms.copy(); work.calc = self.calculator
        try:
            energy = float(work.get_potential_energy())
            forces = np.asarray(work.get_forces(), float)
            if not np.isfinite(energy) or forces.shape != work.positions.shape or not np.isfinite(forces).all():
                raise ValueError("nonfinite energy/forces")
            append(self.ledger, {"kind": "search", "request": self.requests,
                                 "energy": energy, "fmax": float(np.linalg.norm(forces, axis=1).max()),
                                 "atoms": work, "forces": forces})
            return energy, forces
        except Exception as error:
            append(self.ledger, {"kind": "search_failure", "request": self.requests,
                                 "error": repr(error), "atoms": work})
            raise


def instrument_calculate(calculator):
    """Count the existing bound calculate method without proxying ASE state."""
    original = calculator.calculate
    counter = {"calls": 0}
    def counted(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)
    calculator.calculate = counted
    return counter


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(SOURCE))
    from ase.io import read
    from mace.calculators import MACECalculator
    from pamssw.standalone import ASESurface, SSWConfig, run_ssw
    from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
    from pamssw.standalone.recovered_rotation import RecoveredRotationSettings

    plan = json.loads((OUT / "plan.json").read_text())
    rows = []
    for case, spec in plan["cases"].items():
        model = Path(spec["backend"]["model"])
        if sha256(model) != spec["backend"]["model_sha256"]:
            raise ValueError(f"model hash changed: {model}")
        calculator = MACECalculator(model_paths=str(model), head=spec["backend"]["head"],
                                    device=spec["backend"]["device"],
                                    default_dtype=spec["backend"]["dtype"],
                                    enable_cueq=False, enable_oeq=False)
        search_counter = instrument_calculate(calculator)
        atoms = read(OUT / spec["input"])
        config = SSWConfig(**spec["config"])
        direction = RecoveredDirectionSettings(**plan["direction_settings"])
        rotation = RecoveredRotationSettings(**plan["rotation_settings"])
        for arm in plan["arms"]:
            folder = OUT / f"{case}-{arm}"
            folder.mkdir(exist_ok=False)
            calculator.reset()  # Equal empty ASE cache at both arm starts.
            search_calls_before = search_counter["calls"]
            surface = CountedSurface(calculator, folder / "requests.jsonl",
                                     plan["search_cap_per_arm"], plan["wall_seconds_per_arm"])
            row = {"case": case, "arm": arm, "status": "started"}
            try:
                kwargs = {"recovered_direction": direction} if arm == "recovered_direction" else {"recovered_rotation": rotation}
                from pamssw.standalone import NativeMCSettings
                result = run_ssw(atoms.copy(), surface, steps=1, config=config,
                                 rng=np.random.default_rng(spec["seed"]),
                                 mc=NativeMCSettings(energy_tol=0.1, maxtrap=99999), **kwargs)
                assert result.evaluation_requests == surface.requests
                assert surface.requests == result.initial.evaluation_requests + sum(
                    record.evaluation_requests for record in result.records)
                row["search_elapsed_seconds"] = time.monotonic() - surface.started
                dump(folder / "result.json", result)
                fresh_calculator = MACECalculator(model_paths=str(model), head=spec["backend"]["head"],
                                                  device=spec["backend"]["device"],
                                                  default_dtype=spec["backend"]["dtype"],
                                                  enable_cueq=False, enable_oeq=False)
                fresh_counter = instrument_calculate(fresh_calculator)
                fresh = ASESurface(fresh_calculator)
                checks = []
                for index, minimum in enumerate(result.minima[:plan["fresh_cap_per_arm"]]):
                    try:
                        fresh_calculator.reset()
                        energy, forces = fresh.evaluate(minimum.atoms)
                        fmax = float(np.linalg.norm(forces, axis=1).max())
                        fixed_cell = bool(np.array_equal(minimum.atoms.cell.array, atoms.cell.array))
                        same_numbers = bool(np.array_equal(minimum.atoms.numbers, atoms.numbers))
                        checks.append({"index": index, "energy_eV": energy,
                                       "energy_error_eV": float(energy - minimum.energy),
                                       "fmax_eV_A": fmax,
                                       "qualified": bool(fmax <= config.fmax and abs(energy - minimum.energy) <= 1e-6
                                                         and fixed_cell and same_numbers and not minimum.atoms.pbc.any()),
                                       "fixed_cell": fixed_cell, "same_numbers": same_numbers,
                                       "pbc": minimum.atoms.pbc.tolist()})
                    except Exception as error:
                        checks.append({"index": index, "qualified": False, "error": repr(error)})
                dump(folder / "fresh-checks.json", checks)
                row.update(status=result.status, records=len(result.records),
                           gaussian_counts=[len(record.climb) for record in result.records],
                           outer_statuses=dict(Counter(record.status for record in result.records)),
                           fresh_requests=fresh.requests, fresh_calculator_calls=fresh_counter["calls"],
                           fresh_checks=checks)
            except Exception as error:
                row.update(status="exception", error=repr(error))
            row.update(search_requests=surface.requests, calculator_calls=search_counter["calls"] - search_calls_before,
                       denials=surface.denials, boundary=surface.boundary,
                       elapsed_including_fresh_seconds=time.monotonic() - surface.started)
            rows.append(row); dump(OUT / "summary.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the prepared PES experiment")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print("prepared_not_executed: pass --execute only after review")


if __name__ == "__main__":
    main()
