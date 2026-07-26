from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
from typing import Any

from pamssw import BootstrapConvergenceError
from pamssw.accounting import BudgetExceeded
from pamssw.exploration.runner import _bootstrap_minimum


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
SOURCE_DRIVER = (
    RUN_ROOT.parent
    / "20260727-014407-bias-relaxation-gpu-g1"
    / "run_g1.py"
)
FROZEN_COMMIT = "01bcdc47a002bdf00f0d568f966be06169080756"


def _state_hash(state) -> str:
    digest = hashlib.sha256()
    digest.update(state.numbers.tobytes())
    digest.update(state.positions.tobytes())
    if state.cell is not None:
        digest.update(state.cell.tobytes())
    digest.update(bytes(state.pbc))
    if state.fixed_mask is not None:
        digest.update(state.fixed_mask.tobytes())
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run(system: str) -> None:
    source = runpy.run_path(str(SOURCE_DRIVER))
    state = source["_state"](system)
    config = source["_ssw_config"](system, "ase-fire")
    calculator_factory = source["_calculator"]
    started = time.perf_counter()
    try:
        relaxed, energy, counts = _bootstrap_minimum(
            state,
            calculator_factory,
            config,
            total_force_budget=3000,
        )
    except BootstrapConvergenceError as exc:
        payload = {
            "schema_version": 1,
            "frozen_commit": FROZEN_COMMIT,
            "system": system,
            "status": "uncertified",
            "force_evaluations": exc.evaluation_counts.total,
            "purpose_counts": exc.evaluation_counts.as_dict(),
            "energy_eV": exc.relaxation.energy,
            "max_active_atom_force_eV_per_A": exc.relaxation.gradient_norm,
            "force_tolerance_eV_per_A": config.quench_fmax,
            "iterations": exc.relaxation.n_iter,
            "termination_reason": (
                None
                if exc.relaxation.telemetry is None
                else exc.relaxation.telemetry.termination_reason
            ),
            "wall_time_s": time.perf_counter() - started,
        }
    except BudgetExceeded as exc:
        counts = exc.evaluation_counts
        payload = {
            "schema_version": 1,
            "frozen_commit": FROZEN_COMMIT,
            "system": system,
            "status": "budget_exhausted",
            "force_evaluations": None if counts is None else counts.total,
            "purpose_counts": None if counts is None else counts.as_dict(),
            "wall_time_s": time.perf_counter() - started,
        }
    else:
        payload = {
            "schema_version": 1,
            "frozen_commit": FROZEN_COMMIT,
            "system": system,
            "status": "certified",
            "force_evaluations": counts.total,
            "purpose_counts": counts.as_dict(),
            "energy_eV": energy,
            "force_tolerance_eV_per_A": config.quench_fmax,
            "state_sha256": _state_hash(relaxed),
            "wall_time_s": time.perf_counter() - started,
        }
    OUTPUT_ROOT.mkdir(exist_ok=True)
    _write_json_exclusive(OUTPUT_ROOT / f"{system}.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("system", choices=("c60", "pdo"))
    args = parser.parse_args()
    run(args.system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
