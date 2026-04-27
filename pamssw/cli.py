from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from .calculators import ASECalculator, AnalyticCalculator
from .config import LSSSWConfig, SSWConfig
from .potentials import CoupledPairWell, DoubleWell2D
from .runner import run_ls_ssw, run_ssw
from .state import State


def main() -> None:
    parser = argparse.ArgumentParser(description="Run low-parameter SSW or LS-SSW searches.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run-ssw").add_argument("config")
    subparsers.add_parser("run-ls-ssw").add_argument("config")
    args = parser.parse_args()

    config_path = Path(args.config)
    payload = yaml.safe_load(config_path.read_text())
    state = _load_state(payload["state"])
    calculator = _load_calculator(payload["calculator"])
    output = Path(payload.get("output", "pamssw_result.json"))

    if args.command == "run-ssw":
        config = SSWConfig(**payload.get("search", {}))
        result = run_ssw(state, calculator, config)
    else:
        config = LSSSWConfig(**payload.get("search", {}))
        result = run_ls_ssw(state, calculator, config)

    summary = {
        "best_energy": result.best_energy,
        "n_minima": len(result.archive.entries),
        "archive_energies": [entry.energy for entry in result.archive.entries],
    }
    output.write_text(json.dumps(summary, indent=2))


def _load_state(payload: dict) -> State:
    return State(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=None if payload.get("cell") is None else np.asarray(payload["cell"], dtype=float),
        pbc=tuple(payload.get("pbc", (False, False, False))),
        fixed_mask=None if payload.get("fixed_mask") is None else np.asarray(payload["fixed_mask"], dtype=bool),
    )


def _load_calculator(payload: dict):
    kind = payload["kind"]
    if kind == "analytic":
        name = payload["potential"]
        if name == "double_well_2d":
            return AnalyticCalculator(DoubleWell2D())
        if name == "coupled_pair_well":
            return AnalyticCalculator(CoupledPairWell())
        raise ValueError(f"unknown analytic potential: {name}")
    if kind == "ase":
        module_name, class_name = payload["factory"].rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        calculator_cls = getattr(module, class_name)
        kwargs = payload.get("kwargs", {})
        return ASECalculator(calculator_cls(**kwargs))
    raise ValueError(f"unknown calculator kind: {kind}")
