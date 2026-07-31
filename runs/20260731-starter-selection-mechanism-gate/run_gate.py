#!/usr/bin/env python3
"""Matched C60/PdO/CuO gate for uniform, archive-UCB-like, and Metropolis starters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any, Sequence
from zipfile import ZipFile

import numpy as np
from ase.io import read

from pamssw.accounting import EvaluationPurpose
from pamssw.mace_batch import MACEBatchCalculator
from pamssw.state import State
from pamssw.walker import SurfaceWalker
from pamssw.io import write_state


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_REPO_ROOT = Path("/mnt/d/download/trae-research-code/ssw")
SOURCE_GATE = (
    RUN_ROOT.parent / "20260730-starter-cell-online-gate" / "run_gate.py"
)
CUO_ARCHIVE_PATH = SOURCE_REPO_ROOT / "Cu110_Cu10O8.zip"
CUO_INPUT_MEMBER = "Cu110_Cu10O8/CuO_opt_input.arc"
CUO_MODEL_MEMBER = "Cu110_Cu10O8/CuO-OMAT_finetune.model"
SYSTEMS = ("c60", "pdo", "cuo")
STARTER_MODES = ("uniform_archive", "archive_ucb", "metropolis_chain")
MAX_TRIALS = 10_000
DEFAULT_FORCE_BUDGET = 20_000
_SOURCE_MODULE_NAME = "_starter_selection_source_gate"


def _load_source_gate():
    module = sys.modules.get(_SOURCE_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_SOURCE_MODULE_NAME, SOURCE_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen source gate: {SOURCE_GATE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _zip_member_sha256(archive_path: Path, member: str) -> str:
    digest = sha256()
    with ZipFile(archive_path) as archive, archive.open(member) as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _materialize_cuo_resources(
    archive_path: Path,
    target_directory: Path,
) -> dict[str, Path]:
    """Materialize only the declared CuO structure and model from the package."""
    target_directory = Path(target_directory)
    target_directory.mkdir(parents=True, exist_ok=False)
    resources = {}
    with ZipFile(archive_path) as archive:
        for label, member in (
            ("input", CUO_INPUT_MEMBER),
            ("model", CUO_MODEL_MEMBER),
        ):
            destination = target_directory / Path(member).name
            with archive.open(member) as source, destination.open("wb") as sink:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(block)
            resources[label] = destination
    return resources


def _load_state(
    system: str,
    cuo_resources: dict[str, Path] | None = None,
) -> State:
    source_gate = _load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    if system != "cuo":
        return production.load_state(system)
    if cuo_resources is None:
        raise ValueError("CuO resources are required for the CuO case")
    atoms = read(cuo_resources["input"])
    positions = np.asarray(atoms.positions, dtype=float)
    fixed_mask = production.bottom_fixed_mask(positions, 0.35)
    return State(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=positions,
        cell=np.asarray(atoms.cell.array, dtype=float),
        pbc=(True, True, False),
        fixed_mask=fixed_mask,
        metadata={
            "input": str(cuo_resources["input"]),
            "system": "cuo",
            "pbc_mode": "slab",
            "fixed_bottom_fraction": 0.35,
        },
    )


def _state_facts(state: State) -> dict[str, Any]:
    return {
        "n_atoms": int(state.n_atoms),
        "n_fixed_atoms": int(np.count_nonzero(state.fixed_mask)),
        "pbc": [bool(value) for value in state.pbc],
    }


def build_config(
    system: str,
    case_directory: Path,
    *,
    seed: int,
    starter_mode: str,
    force_budget: int,
):
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS!r}")
    if starter_mode not in STARTER_MODES:
        raise ValueError(f"starter_mode must be one of {STARTER_MODES!r}")
    if isinstance(force_budget, bool) or not isinstance(force_budget, int):
        raise ValueError("force_budget must be a positive integer")
    if force_budget <= 0:
        raise ValueError("force_budget must be a positive integer")
    source_gate = _load_source_gate()
    # CuO deliberately inherits the already frozen generic slab action kernel.
    # This admits a third physical system without tuning the SSW machinery to
    # the CuO result; only the structure and calculator model differ at runtime.
    kernel_system = "pdo" if system == "cuo" else system
    frozen = source_gate.build_production_config(
        kernel_system,
        Path(case_directory),
        master_seed=seed,
    )
    return replace(
        frozen,
        max_trials=MAX_TRIALS,
        max_force_evals=force_budget,
        rng_seed=seed,
        seed_selection_mode=starter_mode,
    )


def _preflight(
    expected_commit: str,
    systems: Sequence[str] = SYSTEMS,
) -> dict[str, Any]:
    import mace
    import torch

    actual_commit = _git_commit()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    source_gate = _load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    ordinary_systems = tuple(system for system in systems if system != "cuo")
    inputs = {
        system: {
            "path": str(production.INPUT_PATHS[system]),
            "sha256": _sha256(production.INPUT_PATHS[system]),
        }
        for system in ordinary_systems
    }
    models = {
        system: {
            "path": str(production.MODEL_PATH),
            "sha256": _sha256(production.MODEL_PATH),
        }
        for system in ordinary_systems
    }
    if "cuo" in systems:
        if not CUO_ARCHIVE_PATH.is_file():
            raise FileNotFoundError(CUO_ARCHIVE_PATH)
        inputs["cuo"] = {
            "archive_path": str(CUO_ARCHIVE_PATH),
            "archive_sha256": _sha256(CUO_ARCHIVE_PATH),
            "member": CUO_INPUT_MEMBER,
            "member_sha256": _zip_member_sha256(
                CUO_ARCHIVE_PATH,
                CUO_INPUT_MEMBER,
            ),
        }
        models["cuo"] = {
            "archive_path": str(CUO_ARCHIVE_PATH),
            "archive_sha256": _sha256(CUO_ARCHIVE_PATH),
            "member": CUO_MODEL_MEMBER,
            "member_sha256": _zip_member_sha256(
                CUO_ARCHIVE_PATH,
                CUO_MODEL_MEMBER,
            ),
        }
    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mace": getattr(mace, "__version__", "unknown"),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": str(torch.cuda.get_device_name(0)),
        "model_path": str(production.MODEL_PATH),
        "model_sha256": _sha256(production.MODEL_PATH),
        "models": models,
        "calculator": dict(production.CALCULATOR_CONFIG),
        "inputs": inputs,
        "starter_modes": list(STARTER_MODES),
        "random_stream_protocol": {
            "physical_action_stream": "rng_seed",
            "starter_selection_stream": "SeedSequence([rng_seed, 0x535357])",
            "reason": (
                "starter draws must not shift random, bond, or momentum direction draws"
            ),
        },
        "mechanism_frozen": (
            "direction generation, local softening, Gaussian-bias propagation, "
            "proposal relaxation, true-PES quench"
        ),
    }


def _calculator(
    system: str,
    cuo_resources: dict[str, Path] | None = None,
):
    from mace.calculators import MACECalculator

    source_gate = _load_source_gate()
    prior = source_gate._load_prior_harness()
    production = prior._load_production_runner()
    if system == "cuo":
        if cuo_resources is None:
            raise ValueError("CuO resources are required for the CuO calculator")
        model_path = cuo_resources["model"]
    else:
        model_path = production.MODEL_PATH
    mace_calculator = MACECalculator(
        model_paths=str(model_path),
        **production.CALCULATOR_CONFIG,
    )
    return MACEBatchCalculator(mace_calculator)


def _energy_trace(result) -> list[dict[str, Any]]:
    initial_energy = float(result.archive.entries[0].energy)
    best_energy = initial_energy
    trace = [
        {
            "trial": 0,
            "landing_energy_eV": initial_energy,
            "best_energy_eV": initial_energy,
            "starter_entry_id": 0,
            "landing_entry_id": 0,
            "accepted_new_basin": True,
        }
    ]
    for trial, record in enumerate(result.walk_history, start=1):
        best_energy = min(best_energy, float(record.energy))
        trace.append(
            {
                "trial": trial,
                "landing_energy_eV": float(record.energy),
                "best_energy_eV": best_energy,
                "starter_entry_id": int(record.seed_entry_id),
                "landing_entry_id": int(record.discovered_entry_id),
                "accepted_new_basin": bool(record.accepted_new_basin),
            }
        )
    return trace


def _run_case(
    *,
    system: str,
    seed: int,
    starter_mode: str,
    case_directory: Path,
    force_budget: int,
    cuo_resources: dict[str, Path] | None = None,
) -> dict[str, Any]:
    state = _load_state(system, cuo_resources)
    config = build_config(
        system,
        case_directory,
        seed=seed,
        starter_mode=starter_mode,
        force_budget=force_budget,
    )
    walker = SurfaceWalker(
        calculator=_calculator(system, cuo_resources),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    wall_time_s = perf_counter() - started
    counts = walker.calculator.snapshot()
    if counts.total != int(result.stats["force_evaluations"]):
        raise RuntimeError("force-evaluation ledger does not close")
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("force-evaluation ledger contains unattributed work")
    if counts.total > force_budget:
        raise RuntimeError("case exceeded its force-evaluation budget")

    initial_energy = float(result.archive.entries[0].energy)
    best_energy = float(result.best_energy)
    summary = {
        "schema_version": 1,
        "system": system,
        "seed": seed,
        "starter_mode": starter_mode,
        "state": _state_facts(state),
        "effective_config": asdict(config),
        "initial_energy_eV": initial_energy,
        "best_energy_eV": best_energy,
        "energy_drop_eV": initial_energy - best_energy,
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "completed_trials": int(result.stats["n_trials"]),
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.archive.duplicate_rate()),
        "budget_exhausted": bool(result.stats["budget_exhausted"]),
        "wall_time_s": wall_time_s,
        "stats": result.stats,
    }
    case_directory.mkdir(parents=True, exist_ok=False)
    write_state(case_directory / "best_minimum.xyz", result.best_state)
    _write_json(case_directory / "energy_trace.json", _energy_trace(result))
    _write_json(case_directory / "summary.json", summary)
    return summary


def run_gate(
    *,
    output_directory: Path,
    expected_commit: str,
    systems: Sequence[str],
    seeds: Sequence[int],
    starter_modes: Sequence[str],
    force_budget: int,
    preflight_only: bool = False,
) -> dict[str, Any]:
    systems = tuple(systems)
    seeds = tuple(seeds)
    starter_modes = tuple(starter_modes)
    if not systems or any(system not in SYSTEMS for system in systems):
        raise ValueError(f"systems must be drawn from {SYSTEMS!r}")
    if not seeds or any(isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if not starter_modes or any(mode not in STARTER_MODES for mode in starter_modes):
        raise ValueError(f"starter_modes must be drawn from {STARTER_MODES!r}")
    provenance = _preflight(expected_commit, systems)
    manifest = {
        **provenance,
        "systems": list(systems),
        "seeds": list(seeds),
        "starter_modes": list(starter_modes),
        "force_budget_per_case": force_budget,
        "max_trials_guard": MAX_TRIALS,
        "primary_metric": "best energy at the common force-evaluation budget",
        "secondary_metrics": [
            "energy drop",
            "archive entries",
            "duplicate rate",
            "completed trials",
            "purpose-resolved force evaluations",
            "wall time",
        ],
    }
    if preflight_only:
        return manifest
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    _write_json(output_directory / "manifest.json", manifest)
    cuo_resources = None
    if "cuo" in systems:
        cuo_resources = _materialize_cuo_resources(
            CUO_ARCHIVE_PATH,
            output_directory / "cuo-input",
        )

    cases = []
    for system in systems:
        for seed in seeds:
            for starter_mode in starter_modes:
                relative = Path(system) / f"seed-{seed:08d}" / starter_mode
                summary = _run_case(
                    system=system,
                    seed=seed,
                    starter_mode=starter_mode,
                    case_directory=output_directory / relative,
                    force_budget=force_budget,
                    cuo_resources=cuo_resources,
                )
                cases.append(
                    {
                        "system": system,
                        "seed": seed,
                        "starter_mode": starter_mode,
                        "summary": str(relative / "summary.json"),
                        "best_energy_eV": summary["best_energy_eV"],
                        "force_evaluations": summary["force_evaluations"],
                    }
                )
                _write_json(
                    output_directory / "index.json",
                    {"schema_version": 1, "manifest": "manifest.json", "cases": cases},
                )
    return {"schema_version": 1, "manifest": manifest, "cases": cases}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--starter-modes",
        nargs="+",
        choices=STARTER_MODES,
        default=STARTER_MODES,
    )
    parser.add_argument("--force-budget", type=int, default=DEFAULT_FORCE_BUDGET)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_gate(
        output_directory=args.output,
        expected_commit=args.expected_commit,
        systems=args.systems,
        seeds=args.seeds,
        starter_modes=args.starter_modes,
        force_budget=args.force_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
