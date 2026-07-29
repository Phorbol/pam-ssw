#!/usr/bin/env python3
"""Run one frozen 200-trial C60 or PdO LS-SSW production search."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from ase.io import read

from pamssw import LSSSWConfig, State, write_state
from pamssw.accounting import EvaluationPurpose
from pamssw.calculators import ASECalculator
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_REPO_ROOT = Path("/mnt/d/download/trae-research-code/ssw")
MODEL_PATH = Path("/root/.cache/mace/mace-omat-0-small.model")
INPUT_PATHS = {
    "c60": SOURCE_REPO_ROOT
    / "runs"
    / "20260428-c60-mace-production"
    / "prerelaxed_c60.xyz",
    "pdo": SOURCE_REPO_ROOT / "PdO.xyz",
}

MAX_TRIALS = 200
SEED = 42
SAFE_LBFGS_DEFAULT_HISTORY_LIMIT = 10
CALCULATOR_CONFIG = {
    "device": "cuda",
    "default_dtype": "float32",
    "inference_precision": "float32",
    "enable_cueq": False,
}
RUNTIME_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "ase": "ase",
    "torch": "torch",
    "mace": "mace-torch",
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name, distribution in RUNTIME_PACKAGES.items():
        versions[name] = importlib.metadata.version(distribution)
    return versions


def _cuda_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "available": True,
        "runtime_version": str(torch.version.cuda),
        "device_name": str(torch.cuda.get_device_name(0)),
    }


def _safe_lbfgs_default_history_limit() -> int:
    from pamssw.relax import _SAFE_LBFGS_MEMORY

    return int(_SAFE_LBFGS_MEMORY)


def preflight(
    *,
    system: str,
    expected_git_commit: str,
    runtime_probe: Callable[[], Mapping[str, Any]] = _runtime_versions,
    cuda_probe: Callable[[], Mapping[str, Any]] = _cuda_info,
) -> dict[str, Any]:
    if system not in INPUT_PATHS:
        raise ValueError(f"unknown system: {system}")
    actual_commit = _current_commit()
    if expected_git_commit != actual_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    if _safe_lbfgs_default_history_limit() != SAFE_LBFGS_DEFAULT_HISTORY_LIMIT:
        raise RuntimeError("safe L-BFGS default history limit is not the frozen value 10")

    input_path = INPUT_PATHS[system]
    for path in (input_path, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    runtime_versions = {
        key: str(value) for key, value in runtime_probe().items()
    }
    cuda = dict(cuda_probe())
    if cuda.get("available") is not True:
        raise RuntimeError("CUDA is unavailable")

    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "system": system,
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "model_path": str(MODEL_PATH),
        "model_sha256": _sha256(MODEL_PATH),
        "runtime_versions": runtime_versions,
        "cuda": cuda,
        "calculator": dict(CALCULATOR_CONFIG),
        "safe_lbfgs_default_history_limit": SAFE_LBFGS_DEFAULT_HISTORY_LIMIT,
    }


def bottom_fixed_mask(positions: np.ndarray, fraction: float) -> np.ndarray:
    threshold = float(np.quantile(positions[:, 2], fraction))
    return positions[:, 2] <= threshold


def load_state(system: str) -> State:
    if system not in INPUT_PATHS:
        raise ValueError(f"unknown system: {system}")
    path = INPUT_PATHS[system]
    atoms = read(path)
    if system == "c60":
        return State(
            numbers=np.asarray(atoms.numbers, dtype=int),
            positions=np.asarray(atoms.positions, dtype=float),
            cell=None,
            pbc=(False, False, False),
            fixed_mask=None,
            metadata={"input": str(path), "system": system},
        )

    atoms.pbc = (True, True, False)
    fixed_mask = bottom_fixed_mask(
        np.asarray(atoms.positions, dtype=float), 0.35
    )
    return State(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=np.asarray(atoms.cell.array, dtype=float),
        pbc=(True, True, False),
        fixed_mask=fixed_mask,
        metadata={
            "input": str(path),
            "system": system,
            "pbc_mode": "slab",
            "fixed_bottom_fraction": 0.35,
        },
    )


def _common_config(case_dir: Path) -> dict[str, Any]:
    return {
        "max_trials": MAX_TRIALS,
        "max_force_evals": None,
        "max_steps_per_walk": 8,
        "target_uphill_energy": 0.8,
        "target_negative_curvature": 0.05,
        "quench_maxiter": 400,
        "quench_optimizer": "scipy-lbfgsb",
        "rng_seed": SEED,
        "dedup_energy_tol": 1e-3,
        "fragment_guard_factor": 3.0,
        "anchor_weight": 0.5,
        "continuity_weight": 0.1,
        "history_push_weight": 0.1,
        "enable_anchor_candidate": False,
        "n_bond_pairs": 2,
        "proposal_pool_size": 1,
        "same_seed_max_consecutive": 3,
        "archive_density_weight": 0.5,
        "novelty_weight": 1.0,
        "novelty_probe_scales": (1.0,),
        "frontier_weight": 0.5,
        "bandit_exploration_weight": 0.75,
        "baseline_selection_probability": 0.15,
        "bandit_energy_weight": 1.0,
        "max_prototypes": 500,
        "direction_selection_mode": "discrete",
        "direction_synthesis_mode": "none",
        "direction_score_sigma_mode": "adaptive",
        "direction_type_ucb_enabled": False,
        "direction_archive_enabled": False,
        "direction_probe_enabled": False,
        "plateau_evolution_enabled": False,
        "archive_escape_momentum_enabled": False,
        "random_direction_distribution": "unit_gaussian",
        "enable_bond_form_break_split": False,
        "step_length_mode": "per_atom_rms",
        "direction_diagnostics_enabled": True,
        "direction_diagnostics_path": str(case_dir / "direction_trace.jsonl"),
        "max_energy_drop_per_atom": 5.0,
        "accepted_structures_log": str(case_dir / "accepted_structures.jsonl"),
        "accepted_structures_dir": str(case_dir / "accepted_minima"),
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "min_step_scale": 0.1,
        "max_step_scale": 1.2,
        "proposal_trust_radius": 1.5,
        "local_softening_mode": "active_neighbors",
        "local_softening_strength": 0.15,
        "local_softening_penalty": "buckingham_repulsive",
        "local_softening_xi": 0.3,
        "local_softening_cutoff": 2.0,
        "direction_curvature_source": "inner",
    }


def build_config(system: str, case_dir: Path) -> LSSSWConfig:
    params = _common_config(case_dir)
    if system == "c60":
        params.update(
            quench_fmax=0.01,
            dedup_rmsd_tol=0.15,
            oracle_candidates=12,
            proposal_relax_steps=80,
            walk_trust_radius=5.0,
            target_step_rms=0.08,
            max_step_rms=0.15,
            local_softening_cutoff_scale=1.3,
            local_softening_active_count=3,
        )
    elif system == "pdo":
        params.update(
            quench_fmax=0.03,
            dedup_rmsd_tol=0.4,
            oracle_candidates=8,
            proposal_relax_steps=300,
            min_step_scale=0.05,
            walk_trust_radius=4.0,
            target_step_rms=0.15,
            max_step_rms=0.35,
            local_softening_cutoff_scale=1.15,
            local_softening_active_count=5,
        )
    else:
        raise ValueError(f"unknown system: {system}")
    return LSSSWConfig(**params)


def _calculator():
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(MODEL_PATH), **CALCULATOR_CONFIG)


def _energy_trace(result) -> list[dict[str, Any]]:
    initial_energy = float(result.archive.entries[0].energy)
    best_energy = initial_energy
    points = [
        {
            "trial": 0,
            "energy_eV": initial_energy,
            "best_energy_eV": initial_energy,
            "accepted_new_basin": True,
        }
    ]
    for trial, record in enumerate(result.walk_history, start=1):
        best_energy = min(best_energy, float(record.energy))
        points.append(
            {
                "trial": trial,
                "energy_eV": float(record.energy),
                "best_energy_eV": best_energy,
                "accepted_new_basin": bool(record.accepted_new_basin),
            }
        )
    return points


def _walk_records(result) -> list[dict[str, Any]]:
    return [
        {
            "trial": trial,
            "seed_entry_id": int(record.seed_entry_id),
            "discovered_entry_id": int(record.discovered_entry_id),
            "energy_eV": float(record.energy),
            "accepted_new_basin": bool(record.accepted_new_basin),
        }
        for trial, record in enumerate(result.walk_history, start=1)
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(
    *,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    provenance = preflight(
        system=system,
        expected_git_commit=expected_git_commit,
    )
    if preflight_only:
        return provenance

    output_dir.mkdir(parents=True)
    state = load_state(system)
    config = build_config(system, output_dir)
    walker = SurfaceWalker(
        calculator=ASECalculator(_calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    total_wall_time_s = perf_counter() - started

    counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    force_evaluations = int(result.stats["force_evaluations"])
    if counts.total != force_evaluations:
        raise RuntimeError("force-evaluation total does not close")
    if purpose_counts[EvaluationPurpose.UNATTRIBUTED.value] != 0:
        raise RuntimeError("purpose accounting contains unattributed evaluations")
    if int(result.stats["n_trials"]) != MAX_TRIALS:
        raise RuntimeError("production run did not complete all 200 trials")

    energy_trace = _energy_trace(result)
    walk_records = _walk_records(result)
    optimizer_telemetry = walker.relaxation_diagnostics()
    initial_energy = float(energy_trace[0]["energy_eV"])
    best_energy = float(result.best_energy)
    summary = {
        **provenance,
        "effective_config": asdict(config),
        "input_state": {
            "atom_count": int(len(state.numbers)),
            "fixed_count": int(np.count_nonzero(state.fixed_mask)),
            "pbc": list(state.pbc),
        },
        "initial_energy_eV": initial_energy,
        "best_energy_eV": best_energy,
        "energy_drop_eV": initial_energy - best_energy,
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "optimizer_telemetry": optimizer_telemetry,
        "stats": result.stats,
        "timing": {
            "total_wall_time_s": total_wall_time_s,
            "per_trial_wall_time_s": None,
            "per_trial_wall_time_reason": (
                "not exposed by the current SearchResult or WalkRecord API"
            ),
        },
        "walk_records": walk_records,
    }

    write_state(output_dir / "best_minimum.xyz", result.best_state)
    _write_json(output_dir / "energy_trace.json", energy_trace)
    _write_json(output_dir / "walk_records.json", walk_records)
    _write_json(output_dir / "optimizer_diagnostics.json", optimizer_telemetry)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=tuple(INPUT_PATHS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(
        system=args.system,
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
