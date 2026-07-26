from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from pamssw import SSWConfig, State, run_posterior_ssw
from pamssw.calculators import ASECalculator
from pamssw.exploration import PosteriorExplorationConfig
from pamssw.pbc import mic_displacement, mic_distance_matrix


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
MODEL = Path("/root/.cache/mace/mace-omat-0-small.model")
FROZEN_COMMIT = "47212f7e280671b76d87af95250d469c163353d0"
OPTIMIZERS = (
    "ase-fire",
    "ase-fire2",
    "safe-lbfgs-total",
    "bias-separated-lbfgs",
)
SYSTEMS: dict[str, dict[str, Any]] = {
    "c60": {
        "input": Path(
            "/mnt/d/download/trae-research-code/ssw/"
            "runs/20260428-c60-mace-production/prerelaxed_c60.xyz"
        ),
        "sha256": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
        "n_atoms": 60,
        "pbc": (False, False, False),
        "fix_bottom_fraction": 0.0,
        "expected_n_fixed": 0,
        "dedup_rmsd_tol": 0.15,
        "proposal_relax_steps": 80,
        "fragment_guard_factor": 2.5,
    },
    "pdo": {
        "input": Path("/mnt/d/download/trae-research-code/ssw/PdO.xyz"),
        "sha256": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
        "n_atoms": 115,
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "expected_n_fixed": 50,
        "dedup_rmsd_tol": 0.4,
        "proposal_relax_steps": 120,
        "fragment_guard_factor": 3.0,
    },
}
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _bottom_fixed_mask(positions: np.ndarray, fraction: float) -> np.ndarray | None:
    if fraction <= 0.0:
        return None
    z = np.asarray(positions, dtype=float)[:, 2]
    threshold = float(np.min(z) + fraction * (np.max(z) - np.min(z)))
    return z <= threshold


def _state(system: str) -> State:
    spec = SYSTEMS[system]
    atoms = read(spec["input"])
    atoms.pbc = spec["pbc"]
    fixed_mask = _bottom_fixed_mask(atoms.positions, spec["fix_bottom_fraction"])
    state = State(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=atoms.cell.array,
        pbc=spec["pbc"],
        fixed_mask=fixed_mask,
        metadata={"system": system, "input": str(spec["input"])},
    )
    if state.n_atoms != spec["n_atoms"]:
        raise ValueError(f"{system}: atom count changed")
    n_fixed = 0 if fixed_mask is None else int(np.count_nonzero(fixed_mask))
    if n_fixed != spec["expected_n_fixed"]:
        raise ValueError(f"{system}: fixed-atom count changed")
    return state


def _calculator() -> ASECalculator:
    return ASECalculator(
        MACECalculator(
            model_paths=str(MODEL),
            device="cuda",
            default_dtype="float32",
            inference_precision="float32",
            enable_cueq=False,
        )
    )


def _ssw_config(system: str, optimizer: str) -> SSWConfig:
    spec = SYSTEMS[system]
    return SSWConfig(
        max_trials=1,
        max_steps_per_walk=8,
        target_uphill_energy=0.8,
        target_negative_curvature=0.05,
        quench_fmax=0.03,
        quench_maxiter=400,
        quench_optimizer="scipy-lbfgsb",
        dedup_rmsd_tol=spec["dedup_rmsd_tol"],
        dedup_energy_tol=1e-3,
        rng_seed=42,
        oracle_candidates=8,
        proposal_relax_steps=spec["proposal_relax_steps"],
        proposal_fmax=0.05,
        proposal_optimizer=optimizer,
        min_step_scale=0.05,
        max_step_scale=1.2,
        proposal_trust_radius=None,
        walk_trust_radius=4.0,
        fragment_guard_factor=spec["fragment_guard_factor"],
        n_bond_pairs=2,
        stagnation_bond_pair_boost=2,
        max_stagnation_bond_pairs=10,
        proposal_pool_size=1,
        same_seed_max_consecutive=3,
        max_prototypes=500,
        max_energy_drop_per_atom=5.0,
        direction_curvature_source="inner",
        direction_score_sigma_mode="fixed_reference",
        step_length_mode="per_atom_rms",
        accepted_structures_log=None,
        accepted_structures_dir=None,
        write_proposal_minima=False,
        proposal_minima_dir=None,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        proposal_duplicate_rescue_optimizer=None,
        max_force_evals=None,
    )


def _exploration_config(case_root: Path) -> PosteriorExplorationConfig:
    return PosteriorExplorationConfig(
        policy_name="uniform",
        batch_size=2,
        max_workers=2,
        action_force_budget=1000,
        total_force_budget=3000,
        master_seed=42,
        run_directory=case_root / "campaign",
    )


def _jsonable_config(config: SSWConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        if isinstance(value, Path):
            value = str(value)
        payload[item.name] = value
    return payload


def _single_point(system: str) -> dict[str, Any]:
    state = _state(system)
    started = time.perf_counter()
    evaluated = _calculator().evaluate(state)
    elapsed = time.perf_counter() - started
    force = -np.asarray(evaluated.gradient, dtype=float)
    if not np.isfinite(evaluated.energy) or not np.isfinite(force).all():
        raise ValueError(f"{system}: non-finite MACE single point")
    return {
        "system": system,
        "energy_eV": float(evaluated.energy),
        "force_max_eV_per_A": float(np.max(np.linalg.norm(force, axis=1))),
        "wall_time_s": elapsed,
    }


def run_preflight() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if _sha256(MODEL) != MODEL_SHA256:
        raise ValueError("model hash mismatch")
    for spec in SYSTEMS.values():
        if _sha256(spec["input"]) != spec["sha256"]:
            raise ValueError(f"input hash mismatch: {spec['input']}")
    payload = {
        "schema_version": 1,
        "frozen_commit": FROZEN_COMMIT,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "mace_version": _package_version("mace-torch"),
        "ase_version": _package_version("ase"),
        "numpy_version": np.__version__,
        "model": str(MODEL),
        "model_sha256": MODEL_SHA256,
        "input_sha256": {
            name: spec["sha256"] for name, spec in SYSTEMS.items()
        },
        "single_points": [_single_point(system) for system in SYSTEMS],
    }
    _write_json_exclusive(OUTPUT_ROOT / "preflight.json", payload)


def _event_facts(event_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attempts = [row for row in rows if row["record_type"] == "attempt"]
    return {
        "records": len(rows),
        "attempts": len(attempts),
        "status_counts": dict(sorted(Counter(row["status"] for row in attempts).items())),
        "random_seeds": [row["random_seed"] for row in attempts],
        "starter_ids": [row["starter_id"] for row in attempts],
        "force_evaluations": [row["force_evaluations"] for row in attempts],
    }


def _diagnostic_facts(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    attempts = payload["attempts"]
    summed_names = (
        "proposal_relax_count",
        "proposal_relax_evaluator_calls",
        "proposal_relax_backend_evaluations",
        "proposal_relax_reporting_evaluator_calls",
        "proposal_relax_accepted_steps",
        "proposal_relax_rejected_steps",
        "proposal_relax_accepted_secants",
        "proposal_relax_rejected_secants",
        "proposal_relax_line_search_evaluations",
        "proposal_relax_mic_branch_resets",
        "proposal_relax_termination_converged",
        "proposal_relax_termination_maxiter",
        "proposal_relax_termination_optimizer_stopped",
        "proposal_relax_termination_line_search_failed",
        "proposal_relax_bias_secant_curvature_sum",
        "true_quench_evaluator_calls",
    )
    sums = {
        name: sum(float(item["stats"].get(name, 0)) for item in attempts)
        for name in summed_names
    }
    return {
        "attempt_count": len(attempts),
        "action_ids": [item["action_id"] for item in attempts],
        "stages": [item["stats"]["diagnostic_stage"] for item in attempts],
        "sums": sums,
        "attempts": attempts,
    }


def _geometry_facts(state: State, reference: State, energy: float, entry_id: int) -> dict[str, Any]:
    distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    fixed_displacement_max = 0.0
    if reference.fixed_mask is not None and np.any(reference.fixed_mask):
        displacement = mic_displacement(
            state.positions[reference.fixed_mask],
            reference.positions[reference.fixed_mask],
            state.cell,
            state.pbc,
        )
        fixed_displacement_max = float(np.max(np.linalg.norm(displacement, axis=1)))
    return {
        "entry_id": entry_id,
        "energy_eV": float(energy),
        "finite_positions": bool(np.isfinite(state.positions).all()),
        "nearest_min_A": float(np.min(nearest)),
        "nearest_median_A": float(np.median(nearest)),
        "fixed_displacement_max_A": fixed_displacement_max,
    }


def run_campaign(system: str, optimizer: str) -> None:
    case_root = OUTPUT_ROOT / f"{system}__{optimizer}"
    case_root.mkdir(exist_ok=False)
    reference = _state(system)
    ssw_config = _ssw_config(system, optimizer)
    exploration = _exploration_config(case_root)
    started = time.perf_counter()
    result = run_posterior_ssw(reference, _calculator, ssw_config, exploration)
    wall_time = time.perf_counter() - started

    entries = sorted(result.archive.entries, key=lambda entry: entry.energy)
    bootstrap_entry = next(entry for entry in result.archive.entries if entry.entry_id == 0)
    frames: list[Atoms] = []
    geometry: list[dict[str, Any]] = []
    for rank, entry in enumerate(entries):
        atoms = Atoms(
            numbers=entry.state.numbers,
            positions=entry.state.positions,
            cell=entry.state.cell,
            pbc=entry.state.pbc,
        )
        atoms.info.update(
            {
                "entry_id": entry.entry_id,
                "rank": rank,
                "energy_eV": entry.energy,
                "parent_id": -1 if entry.parent_id is None else entry.parent_id,
            }
        )
        frames.append(atoms)
        geometry.append(_geometry_facts(entry.state, reference, entry.energy, entry.entry_id))
    write(case_root / "archive_minima.xyz", frames)

    campaign_root = exploration.run_directory
    diagnostics = _diagnostic_facts(campaign_root / "optimizer_diagnostics.json")
    payload = {
        "schema_version": 1,
        "frozen_commit": FROZEN_COMMIT,
        "system": system,
        "optimizer": optimizer,
        "input": str(SYSTEMS[system]["input"]),
        "input_sha256": SYSTEMS[system]["sha256"],
        "model": str(MODEL),
        "model_sha256": MODEL_SHA256,
        "device": "cuda",
        "dtype": "float32",
        "enable_cueq": False,
        "ssw_config": _jsonable_config(ssw_config),
        "exploration_config": {
            "policy_name": exploration.policy_name,
            "batch_size": exploration.batch_size,
            "max_workers": exploration.max_workers,
            "action_force_budget": exploration.action_force_budget,
            "total_force_budget": exploration.total_force_budget,
            "master_seed": exploration.master_seed,
        },
        "completed_batches": result.completed_batches,
        "completed_attempts": result.completed_attempts,
        "failed_attempts": result.failed_attempts,
        "posterior_observed_attempts": result.posterior_observed_attempts,
        "bootstrap_evaluations": result.bootstrap_evaluations,
        "action_evaluations": result.action_evaluations,
        "total_evaluations": result.total_evaluations,
        "unused_force_budget": result.unused_force_budget,
        "purpose_counts": result.purpose_counts.as_dict(),
        "stop_reason": result.stop_reason.value,
        "benchmark_eligible": result.benchmark_eligible,
        "benchmark_ineligibility_reasons": list(result.benchmark_ineligibility_reasons),
        "bootstrap_energy_eV": float(bootstrap_entry.energy),
        "best_energy_eV": float(entries[0].energy),
        "best_energy_drop_from_bootstrap_eV": float(bootstrap_entry.energy - entries[0].energy),
        "unique_minima": len(entries),
        "archive_energies_eV": [float(entry.energy) for entry in entries],
        "event_facts": _event_facts(campaign_root / "events.jsonl"),
        "optimizer_diagnostics": diagnostics,
        "geometry_diagnostics": geometry,
        "wall_time_s": wall_time,
    }
    _write_json_exclusive(case_root / "result.json", payload)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "campaign"))
    parser.add_argument("--system", choices=tuple(SYSTEMS))
    parser.add_argument("--optimizer", choices=OPTIMIZERS)
    args = parser.parse_args()
    if args.phase == "preflight":
        if args.system is not None or args.optimizer is not None:
            parser.error("preflight takes no system or optimizer")
        run_preflight()
    else:
        if args.system is None or args.optimizer is None:
            parser.error("campaign requires system and optimizer")
        run_campaign(args.system, args.optimizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
