from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write
from mace.calculators import MACECalculator

from pamssw import LSSSWConfig, State, run_ls_ssw
from pamssw.calculators import ASECalculator
from pamssw.pbc import mic_distance_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = RUN_DIR / "output"
LOG_DIR = RUN_DIR / "logs"
EVENT_LOG = RUN_DIR / "event_log.jsonl"
RESULTS_CSV = RUN_DIR / "results.csv"
RESULTS_JSON = RUN_DIR / "results.json"
SUMMARY_MD = RUN_DIR / "summary.md"
ARTIFACTS_JSON = RUN_DIR / "artifacts.json"

SOURCE_ROOT = Path("/mnt/d/Download/trae-research-code/SSW")
DEFAULT_MODEL = Path("/root/.cache/mace/mace-omat-0-small.model")
SYSTEMS: dict[str, dict[str, Any]] = {
    "c60": {
        "input": SOURCE_ROOT / "runs/20260428-c60-mace-production/prerelaxed_c60.xyz",
        "model": DEFAULT_MODEL,
        "pbc": (False, False, False),
        "fix_bottom_fraction": 0.0,
        "dedup_rmsd_tol": 0.15,
        "proposal_relax_steps": 80,
        "local_softening_active_count": 3,
        "quench_fmax": 0.01,
        "oracle_candidates": 12,
        "walk_trust_radius": 5.0,
        "proposal_trust_radius": 1.5,
    },
    "cuo": {
        "input": SOURCE_ROOT / "runs/20260506-cuo-200t-production/input/Cu110_Cu10O8/CuO_opt_input.arc",
        "model": SOURCE_ROOT / "runs/20260506-cuo-200t-production/input/Cu110_Cu10O8/CuO-OMAT_finetune.model",
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "dedup_rmsd_tol": 0.4,
        "proposal_relax_steps": 300,
        "local_softening_active_count": 5,
        "quench_fmax": 0.03,
        "oracle_candidates": 8,
        "walk_trust_radius": 4.0,
        "proposal_trust_radius": 1.5,
    },
    "pdo": {
        "input": SOURCE_ROOT / "PdO.xyz",
        "model": DEFAULT_MODEL,
        "pbc": (True, True, False),
        "fix_bottom_fraction": 0.35,
        "dedup_rmsd_tol": 0.4,
        "proposal_relax_steps": 120,
        "local_softening_active_count": 5,
        "quench_fmax": 0.03,
        "oracle_candidates": 8,
        "walk_trust_radius": 4.0,
        "proposal_trust_radius": 1.5,
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "bias_relax_current": {
        "proposal_step_mode": "bias_relax",
    },
    "direct_qp_default": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 4.0,
    },
    "direct_qp_curvature_gamma": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 4.0,
    },
    "direct_qp_curvature_gamma_kappa240": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 240.0,
    },
    "direct_qp_rank1_kappa240": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 240.0,
    },
    "direct_qp_rank1_floor16_kappa240": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 16.0,
        "direct_qp_kappa": 240.0,
    },
    "direct_qp_rank1_histq25_kappa240": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "curvature_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_kappa": 240.0,
    },
    "direct_qp_rank1_gated_q25_kappa240": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
    },
    "direct_qp_rank1_gated_q25_micro3": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_optimizer": "scipy-lbfgsb",
        "direct_qp_micro_fmax": 0.2,
        "direct_qp_micro_trust_radius": 0.25,
    },
    "direct_qp_rank1_gated_q25_micro_adaptive20": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 20,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 10.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.08,
        "direct_qp_micro_trust_radius": 0.25,
    },
    "direct_qp_rank1_gated_q25_micro_adaptive50": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 50,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 12.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.05,
        "direct_qp_micro_trust_radius": 0.25,
    },
    "direct_qp_rank1_gated_q25_micro_adaptive50_prodstep": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 50,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 12.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.05,
        "direct_qp_micro_trust_radius": 0.25,
        "target_step_rms": 0.08,
        "max_step_rms": 0.15,
    },
    "direct_qp_rank1_gated_q25_micro_adaptive50_pool2": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 50,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 12.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.05,
        "direct_qp_micro_trust_radius": 0.25,
        "proposal_pool_size": 2,
    },
    "direct_qp_rank1_gated_q25_micro_adaptive50_dup_rescue": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 3,
        "direct_qp_micro_mode": "adaptive_model_error",
        "direct_qp_micro_max_steps": 50,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_model_error_high": 12.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.05,
        "direct_qp_micro_trust_radius": 0.25,
        "proposal_duplicate_rescue_optimizer": "ase-fire",
    },
    "direct_qp_rank1_gated_q25_micro_error20": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_hessian": "rank1",
        "direct_qp_gamma": 1.0,
        "direct_qp_gamma_mode": "model_error_gated_history",
        "direct_qp_gamma_history_quantile": 0.25,
        "direct_qp_gamma_history_min_samples": 4,
        "direct_qp_gamma_history_maxlen": 128,
        "direct_qp_gamma_model_error_threshold": 2.0,
        "direct_qp_gamma_model_error_streak": 2,
        "direct_qp_kappa": 240.0,
        "direct_qp_micro_steps": 0,
        "direct_qp_micro_mode": "model_error",
        "direct_qp_micro_max_steps": 20,
        "direct_qp_micro_model_error_threshold": 2.0,
        "direct_qp_micro_optimizer": "ase-fire",
        "direct_qp_micro_fmax": 0.08,
        "direct_qp_micro_trust_radius": 0.25,
    },
    "direct_qp_adaptive_kappa_r15": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 4.0,
        "direct_qp_kappa_mode": "adaptive_curvature",
        "direct_qp_kappa_curvature_ratio": 15.0,
    },
    "direct_qp_adaptive_kappa_r15_cap240": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 4.0,
        "direct_qp_kappa_mode": "adaptive_curvature",
        "direct_qp_kappa_curvature_ratio": 15.0,
        "direct_qp_kappa_max": 240.0,
    },
    "direct_qp_strong": {
        "proposal_step_mode": "direct_qp",
        "direct_qp_gamma": 1.0,
        "direct_qp_kappa": 8.0,
    },
    "direct_qp_system_baseline": {},
    "direct_qp_system_baseline_regularized_ritz": {
        "direction_synthesis_mode": "regularized_ritz",
        "regularized_ritz_top_k": 5,
    },
    "direct_qp_system_baseline_rayleigh_ritz": {
        "direction_selection_mode": "rayleigh_ritz",
    },
}

SYSTEM_BASELINE_VARIANT = {
    "c60": "direct_qp_rank1_gated_q25_micro_adaptive50",
    "cuo": "direct_qp_curvature_gamma_kappa240",
    "pdo": "direct_qp_rank1_gated_q25_micro_adaptive50",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def append_event(payload: dict[str, Any]) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def bottom_fixed_mask(positions: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0:
        return np.zeros(len(positions), dtype=bool)
    z = np.asarray(positions, dtype=float)[:, 2]
    threshold = float(np.min(z) + fraction * (np.max(z) - np.min(z)))
    return z <= threshold


def state_from_input(system: str, spec: dict[str, Any]) -> State:
    atoms = read(spec["input"])
    input_pbc = tuple(bool(x) for x in atoms.pbc.tolist())
    atoms.pbc = spec["pbc"]
    fixed_mask = bottom_fixed_mask(atoms.positions, float(spec["fix_bottom_fraction"]))
    cell = atoms.cell.array if atoms.cell.rank > 0 else None
    if not any(spec["pbc"]):
        cell = None
    return State(
        numbers=atoms.numbers,
        positions=atoms.positions,
        cell=cell,
        pbc=spec["pbc"],
        fixed_mask=fixed_mask,
        metadata={
            "system": system,
            "input": str(spec["input"]),
            "input_pbc": input_pbc,
            "run_pbc": spec["pbc"],
        },
    )


def config_for(system: str, spec: dict[str, Any], variant: str, seed: int, trials: int, steps_per_walk: int, case_dir: Path) -> LSSSWConfig:
    params: dict[str, Any] = {
        "max_trials": trials,
        "max_steps_per_walk": steps_per_walk,
        "target_uphill_energy": 0.8,
        "target_negative_curvature": 0.05,
        "quench_fmax": spec["quench_fmax"],
        "quench_maxiter": 400,
        "dedup_rmsd_tol": spec["dedup_rmsd_tol"],
        "dedup_energy_tol": 1e-3,
        "rng_seed": seed,
        "oracle_candidates": spec["oracle_candidates"],
        "proposal_relax_steps": spec["proposal_relax_steps"],
        "proposal_fmax": 0.05,
        "proposal_optimizer": "ase-fire",
        "quench_optimizer": "scipy-lbfgsb",
        "min_step_scale": 0.05 if system != "c60" else 0.1,
        "max_step_scale": 1.2,
        "proposal_trust_radius": spec["proposal_trust_radius"],
        "walk_trust_radius": spec["walk_trust_radius"],
        "fragment_guard_factor": 3.0,
        "n_bond_pairs": 2,
        "stagnation_bond_pair_boost": 2,
        "max_stagnation_bond_pairs": 10,
        "proposal_pool_size": 1,
        "same_seed_max_consecutive": 3,
        "max_prototypes": 500,
        "max_energy_drop_per_atom": 5.0,
        "direction_curvature_source": "inner",
        "direction_score_sigma_mode": "adaptive",
        "accepted_structures_log": str(case_dir / "accepted_structures.jsonl"),
        "accepted_structures_dir": str(case_dir / "accepted_minima"),
        "write_proposal_minima": False,
        "write_relaxation_trajectories": False,
        "local_softening_mode": "active_neighbors",
        "local_softening_cutoff_scale": 1.15,
        "local_softening_active_count": spec["local_softening_active_count"],
        "local_softening_strength": 0.15,
        "local_softening_penalty": "buckingham_repulsive",
        "local_softening_xi": 0.3,
        "local_softening_cutoff": 2.0,
    }
    variant_params = dict(VARIANTS[variant])
    if variant.startswith("direct_qp_system_baseline"):
        params.update(VARIANTS[SYSTEM_BASELINE_VARIANT[system]])
    params.update(variant_params)
    return LSSSWConfig(**params)


def make_calculator(model: Path, device: str):
    calc = MACECalculator(
        model_paths=str(model),
        device=device,
        default_dtype="float32",
        inference_precision="float32",
        enable_cueq=False,
    )
    return ASECalculator(calc)


def state_to_atoms(state: State) -> Atoms:
    return Atoms(numbers=state.numbers, positions=state.positions, cell=state.cell, pbc=state.pbc)


def energy_trace(result) -> list[dict[str, Any]]:
    best = result.archive.entries[0].energy if result.archive.entries else result.best_energy
    points: list[dict[str, Any]] = [
        {"step": 0, "energy": float(best), "best_energy": float(best), "accepted_new_basin": True}
    ]
    for index, record in enumerate(result.walk_history, start=1):
        best = min(best, record.energy)
        points.append(
            {
                "step": index,
                "energy": float(record.energy),
                "best_energy": float(best),
                "accepted_new_basin": bool(record.accepted_new_basin),
                "seed_entry_id": int(record.seed_entry_id),
                "discovered_entry_id": int(record.discovered_entry_id),
            }
        )
    return points


def geometry_diagnostics(state: State, energy: float, entry_id: int) -> dict[str, Any]:
    distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
    if state.n_atoms:
        np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1) if state.n_atoms else np.asarray([])
    return {
        "entry_id": int(entry_id),
        "energy": float(energy),
        "pbc": tuple(bool(x) for x in state.pbc),
        "cell_present": state.cell is not None,
        "nearest_min": float(np.min(nearest)) if nearest.size else 0.0,
        "nearest_median": float(np.median(nearest)) if nearest.size else 0.0,
        "nearest_max": float(np.max(nearest)) if nearest.size else 0.0,
        "z_min": float(np.min(state.positions[:, 2])),
        "z_max": float(np.max(state.positions[:, 2])),
    }


def write_outputs(case_dir: Path, result, config: LSSSWConfig, system: str, variant: str, seed: int, spec: dict[str, Any], device: str, wall_time_s: float) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "minima_xyz").mkdir(exist_ok=True)
    archive_entries = sorted(result.archive.entries, key=lambda entry: entry.energy)
    minima_atoms = []
    for rank, entry in enumerate(archive_entries):
        atoms = state_to_atoms(entry.state)
        atoms.info.update({"entry_id": entry.entry_id, "rank": rank, "energy": entry.energy})
        minima_atoms.append(atoms)
        write(case_dir / "minima_xyz" / f"minimum_rank{rank:03d}_entry{entry.entry_id:03d}.xyz", atoms)
    if minima_atoms:
        write(case_dir / "archive_minima.xyz", minima_atoms)
        write(case_dir / "best_minimum.xyz", minima_atoms[0])
    trace = energy_trace(result)
    (case_dir / "energy_trace.json").write_text(safe_json(trace), encoding="utf-8")
    diagnostics = [geometry_diagnostics(entry.state, entry.energy, entry.entry_id) for entry in archive_entries]
    fixed_mask = archive_entries[0].state.fixed_mask if archive_entries else None
    summary = {
        "system": system,
        "variant": variant,
        "seed": seed,
        "input": str(spec["input"]),
        "model": str(spec["model"]),
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dtype": "float32",
        "enable_cueq": False,
        "n_atoms": int(archive_entries[0].state.n_atoms if archive_entries else 0),
        "n_fixed": int(np.count_nonzero(fixed_mask)) if fixed_mask is not None else 0,
        "pbc": tuple(bool(x) for x in spec["pbc"]),
        "config": asdict(config),
        "best_energy": float(result.best_energy),
        "n_minima": len(result.archive.entries),
        "stats": result.stats,
        "archive_energies": [float(entry.energy) for entry in archive_entries],
        "geometry_diagnostics": diagnostics,
        "wall_time_s": float(wall_time_s),
        "outputs": {
            "summary": str(case_dir / "ssw_summary.json"),
            "energy_trace_json": str(case_dir / "energy_trace.json"),
            "accepted_structures_log": str(case_dir / "accepted_structures.jsonl"),
            "accepted_structures_dir": str(case_dir / "accepted_minima"),
            "archive_minima_xyz": str(case_dir / "archive_minima.xyz"),
            "best_minimum_xyz": str(case_dir / "best_minimum.xyz"),
            "minima_dir": str(case_dir / "minima_xyz"),
        },
    }
    (case_dir / "ssw_summary.json").write_text(safe_json(summary), encoding="utf-8")
    return summary


def row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary["stats"]
    geometry = summary.get("geometry_diagnostics") or []
    best_geometry = geometry[0] if geometry else {}
    return {
        "system": summary["system"],
        "variant": summary["variant"],
        "seed": int(summary["seed"]),
        "device": summary["device"],
        "best_energy": float(summary["best_energy"]),
        "n_minima": int(summary["n_minima"]),
        "force_evaluations": int(stats.get("force_evaluations", 0)),
        "energy_evaluations": int(stats.get("energy_evaluations", 0)),
        "duplicate_rate": float(stats.get("duplicate_rate", 0.0)),
        "proposal_relax_count": int(stats.get("proposal_relax_count", 0)),
        "proposal_relax_mean_iterations": float(stats.get("proposal_relax_mean_iterations", 0.0)),
        "bias_steps": int(stats.get("bias_steps", 0)),
        "direct_qp_steps": int(stats.get("direct_qp_steps", 0)),
        "direct_qp_rejected": int(stats.get("direct_qp_rejected", 0)),
        "direct_qp_mean_step_norm": float(stats.get("direct_qp_mean_step_norm", 0.0)),
        "direct_qp_mean_progress": float(stats.get("direct_qp_mean_progress", 0.0)),
        "direct_qp_mean_target_error": float(stats.get("direct_qp_mean_target_error", 0.0)),
        "direct_qp_mean_model_error": float(stats.get("direct_qp_mean_model_error", 0.0)),
        "direct_qp_trust_shrink_steps": int(stats.get("direct_qp_trust_shrink_steps", 0)),
        "direct_qp_trust_expand_steps": int(stats.get("direct_qp_trust_expand_steps", 0)),
        "direct_qp_gamma_mean": float(stats.get("direct_qp_gamma_mean", 0.0)),
        "direct_qp_kappa_mean": float(stats.get("direct_qp_kappa_mean", 0.0)),
        "direct_qp_micro_count": int(stats.get("direct_qp_micro_count", 0)),
        "direct_qp_micro_mean_iterations": float(stats.get("direct_qp_micro_mean_iterations", 0.0)),
        "direct_qp_micro_displacement_rms_mean": float(stats.get("direct_qp_micro_displacement_rms_mean", 0.0)),
        "direct_qp_micro_displacement_max": float(stats.get("direct_qp_micro_displacement_max", 0.0)),
        "direction_selected_ritz": int(stats.get("direction_selected_ritz", 0)),
        "direction_selected_ritz_reg": int(stats.get("direction_selected_ritz_reg", 0)),
        "direction_selected_momentum": int(stats.get("direction_selected_momentum", 0)),
        "direction_selected_bond": int(stats.get("direction_selected_bond", 0)),
        "direction_selected_random": int(stats.get("direction_selected_random", 0)),
        "direction_candidate_evaluations": int(stats.get("direction_candidate_evaluations", 0)),
        "direction_mean_candidate_pool_size": float(stats.get("direction_mean_candidate_pool_size", 0.0)),
        "true_quench_count": int(stats.get("true_quench_count", 0)),
        "true_quench_mean_iterations": float(stats.get("true_quench_mean_iterations", 0.0)),
        "nearest_min": float(best_geometry.get("nearest_min", 0.0)),
        "nearest_median": float(best_geometry.get("nearest_median", 0.0)),
        "wall_time_s": float(summary["wall_time_s"]),
        "summary_path": summary["outputs"]["summary"],
    }


def write_aggregate(rows: list[dict[str, Any]], final: bool = False) -> None:
    if rows:
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    grouped = []
    for system in sorted({row["system"] for row in rows}):
        for variant in sorted({row["variant"] for row in rows}):
            subset = [row for row in rows if row["system"] == system and row["variant"] == variant]
            if not subset:
                continue
            grouped.append(
                {
                    "system": system,
                    "variant": variant,
                    "n": len(subset),
                    "mean_best_energy": float(np.mean([row["best_energy"] for row in subset])),
                    "best_energy_min": float(np.min([row["best_energy"] for row in subset])),
                    "mean_force_evaluations": float(np.mean([row["force_evaluations"] for row in subset])),
                    "mean_duplicate_rate": float(np.mean([row["duplicate_rate"] for row in subset])),
                    "mean_direct_qp_steps": float(np.mean([row["direct_qp_steps"] for row in subset])),
                    "mean_direct_qp_rejected": float(np.mean([row["direct_qp_rejected"] for row in subset])),
                    "mean_direct_qp_micro_count": float(np.mean([row["direct_qp_micro_count"] for row in subset])),
                    "mean_direct_qp_micro_iterations": float(
                        np.mean([row["direct_qp_micro_mean_iterations"] for row in subset])
                    ),
                    "mean_direction_selected_ritz": float(np.mean([row["direction_selected_ritz"] for row in subset])),
                    "mean_direction_selected_ritz_reg": float(
                        np.mean([row["direction_selected_ritz_reg"] for row in subset])
                    ),
                    "mean_direction_candidate_evaluations": float(
                        np.mean([row["direction_candidate_evaluations"] for row in subset])
                    ),
                    "mean_wall_time_s": float(np.mean([row["wall_time_s"] for row in subset])),
                }
            )
    RESULTS_JSON.write_text(safe_json({"rows": rows, "summary": grouped}), encoding="utf-8")
    write_summary(rows, grouped, final=final)
    ARTIFACTS_JSON.write_text(
        safe_json(
            {
                "artifacts": [
                    {"artifact_id": "results_csv", "type": "csv", "path": str(RESULTS_CSV)},
                    {"artifact_id": "results_json", "type": "json", "path": str(RESULTS_JSON)},
                    {"artifact_id": "summary_md", "type": "markdown", "path": str(SUMMARY_MD)},
                ]
            }
        ),
        encoding="utf-8",
    )


def write_summary(rows: list[dict[str, Any]], grouped: list[dict[str, Any]], final: bool) -> None:
    lines = [
        "# Direct-QP SSW C60/CuO/PdO Benchmark",
        "",
        f"- Status: {'completed' if final else 'running'}",
        f"- Completed cases: {len(rows)}",
        "- Variants: direct_qp_system_baseline, direct_qp_system_baseline_regularized_ritz, direct_qp_system_baseline_rayleigh_ritz",
        "- Dtype: float32, cuEq off",
        "",
        "## Aggregate",
        "",
    ]
    for item in grouped:
        lines.append(
            "- {system} / {variant}: n={n}, mean_best={mean_best:.9g} eV, best_single={best_min:.9g} eV, "
            "mean_force={force:.1f}, mean_dup={dup:.3f}, direct_steps={direct_steps:.1f}, "
            "direct_reject={direct_reject:.1f}, ritz={ritz:.1f}, ritz_reg={ritz_reg:.1f}, wall={wall:.1f}s".format(
                system=item["system"],
                variant=item["variant"],
                n=item["n"],
                mean_best=item["mean_best_energy"],
                best_min=item["best_energy_min"],
                force=item["mean_force_evaluations"],
                dup=item["mean_duplicate_rate"],
                direct_steps=item["mean_direct_qp_steps"],
                direct_reject=item["mean_direct_qp_rejected"],
                ritz=item["mean_direction_selected_ritz"],
                ritz_reg=item["mean_direction_selected_ritz_reg"],
                wall=item["mean_wall_time_s"],
            )
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_case(system: str, variant: str, seed: int, trials: int, steps_per_walk: int, device: str) -> dict[str, Any]:
    spec = SYSTEMS[system]
    case_id = f"{system}_{variant}_seed{seed}_trials{trials}_{device}"
    case_dir = OUTPUT_DIR / case_id
    summary_path = case_dir / "ssw_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        return row_from_summary(summary)
    append_event({"event": "start", "case": case_id, "time": now()})
    case_dir.mkdir(parents=True, exist_ok=True)
    state = state_from_input(system, spec)
    write(case_dir / "initial_structure.xyz", state_to_atoms(state))
    start = time.time()
    calculator = make_calculator(spec["model"], device=device)
    config = config_for(system, spec, variant, seed, trials, steps_per_walk, case_dir)
    result = run_ls_ssw(state, calculator, config)
    wall_time_s = time.time() - start
    summary = write_outputs(case_dir, result, config, system, variant, seed, spec, device, wall_time_s)
    row = row_from_summary(summary)
    append_event({"event": "done", "case": case_id, "time": now(), "row": row})
    return row


def parse_csv(value: str, allowed: set[str]) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise ValueError(f"unknown values {unknown}; allowed={sorted(allowed)}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="c60,cuo,pdo")
    parser.add_argument(
        "--variants",
        default=(
            "direct_qp_system_baseline,"
            "direct_qp_system_baseline_regularized_ritz,"
            "direct_qp_system_baseline_rayleigh_ritz"
        ),
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--steps-per-walk", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        if not args.allow_cpu:
            raise RuntimeError("CUDA unavailable; pass --device cpu --allow-cpu for CPU evidence")
        args.device = "cpu"
    if args.device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU execution requires explicit --allow-cpu")

    systems = parse_csv(args.systems, set(SYSTEMS))
    variants = parse_csv(args.variants, set(VARIANTS))
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for system in systems:
        for variant in variants:
            for seed in seeds:
                row = run_case(system, variant, seed, args.trials, args.steps_per_walk, args.device)
                rows.append(row)
                write_aggregate(rows, final=False)
    write_aggregate(rows, final=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
