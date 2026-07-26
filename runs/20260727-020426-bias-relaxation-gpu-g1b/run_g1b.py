from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import time
from typing import Any

from pamssw import BootstrapConvergenceError
from pamssw.accounting import BudgetExceeded


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
SOURCE_DRIVER = (
    RUN_ROOT.parent
    / "20260727-014407-bias-relaxation-gpu-g1"
    / "run_g1.py"
)
FROZEN_COMMIT = "c598096170d40625b6d127668935a166cc1507c1"
SYSTEMS = ("c60", "pdo")
OPTIMIZERS = (
    "ase-fire",
    "ase-fire2",
    "safe-lbfgs-total",
    "bias-separated-lbfgs",
)


def _source() -> dict[str, Any]:
    source = runpy.run_path(str(SOURCE_DRIVER))
    for function_name in ("run_preflight", "run_campaign"):
        function_globals = source[function_name].__globals__
        function_globals["OUTPUT_ROOT"] = OUTPUT_ROOT
        function_globals["FROZEN_COMMIT"] = FROZEN_COMMIT
    return source


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def run_campaign(system: str, optimizer: str) -> None:
    source = _source()
    started = time.perf_counter()
    try:
        source["run_campaign"](system, optimizer)
    except BootstrapConvergenceError as exc:
        case_root = OUTPUT_ROOT / f"{system}__{optimizer}"
        payload = {
            "schema_version": 1,
            "frozen_commit": FROZEN_COMMIT,
            "system": system,
            "optimizer": optimizer,
            "status": "bootstrap_uncertified",
            "proposal_started": False,
            "force_evaluations": exc.evaluation_counts.total,
            "purpose_counts": exc.evaluation_counts.as_dict(),
            "energy_eV": exc.relaxation.energy,
            "max_active_atom_force_eV_per_A": exc.relaxation.gradient_norm,
            "iterations": exc.relaxation.n_iter,
            "termination_reason": (
                None
                if exc.relaxation.telemetry is None
                else exc.relaxation.telemetry.termination_reason
            ),
            "wall_time_s": time.perf_counter() - started,
        }
        _write_json_exclusive(case_root / "failure.json", payload)
    except BudgetExceeded as exc:
        case_root = OUTPUT_ROOT / f"{system}__{optimizer}"
        counts = exc.evaluation_counts
        payload = {
            "schema_version": 1,
            "frozen_commit": FROZEN_COMMIT,
            "system": system,
            "optimizer": optimizer,
            "status": "bootstrap_budget_exhausted",
            "proposal_started": False,
            "force_evaluations": None if counts is None else counts.total,
            "purpose_counts": None if counts is None else counts.as_dict(),
            "wall_time_s": time.perf_counter() - started,
        }
        _write_json_exclusive(case_root / "failure.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "campaign"))
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--optimizer", choices=OPTIMIZERS)
    args = parser.parse_args()
    source = _source()
    if args.phase == "preflight":
        if args.system is not None or args.optimizer is not None:
            parser.error("preflight takes no system or optimizer")
        source["run_preflight"]()
    else:
        if args.system is None or args.optimizer is None:
            parser.error("campaign requires system and optimizer")
        run_campaign(args.system, args.optimizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
