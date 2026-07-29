from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import LSSSWConfig


_C60_PROFILE = "c60_direction_efficient_validated_20260729"

_PROFILE_METADATA: dict[str, dict[str, Any]] = {
    _C60_PROFILE: {
        "system": "C60",
        "source_commit": "04c7bd011fd3b9beaa7dff775ffa5e203ef00a92",
        "model_sha256": (
            "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
        ),
        "production_default_changed": False,
        "evidence": (
            "runs/20260729-staged-direction-efficiency-ablation/"
            "final_report.md"
        ),
        "claim_ceiling": (
            "C60 fixed-starter direction-efficiency and strict-terminal "
            "validation; this profile is not a universal cluster, slab, "
            "optimizer, or thermodynamic-sampling default."
        ),
    },
}


def available_validated_profiles() -> tuple[str, ...]:
    """Return the stable names of user-facing evidence-backed profiles."""
    return tuple(_PROFILE_METADATA)


def validated_profile_metadata(name: str) -> dict[str, Any]:
    """Return a copy of the evidence scope attached to one profile."""
    try:
        return deepcopy(_PROFILE_METADATA[name])
    except KeyError as error:
        available = ", ".join(available_validated_profiles())
        raise ValueError(
            f"unknown validated profile {name!r}; available: {available}"
        ) from error


def validated_ls_ssw_config(
    name: str,
    *,
    output_dir: str | Path,
    max_trials: int = 200,
    rng_seed: int = 0,
    max_force_evals: int | None = None,
) -> LSSSWConfig:
    """Build an explicit LS-SSW config without changing package defaults."""
    validated_profile_metadata(name)
    output = Path(output_dir)
    return LSSSWConfig(
        max_trials=max_trials,
        max_force_evals=max_force_evals,
        max_steps_per_walk=8,
        target_uphill_energy=0.8,
        target_negative_curvature=0.05,
        quench_fmax=0.01,
        quench_maxiter=400,
        quench_optimizer="ase-lbfgs",
        quench_fallback_optimizer="ase-fire",
        dedup_rmsd_tol=0.15,
        dedup_energy_tol=1e-3,
        rng_seed=rng_seed,
        oracle_candidates=4,
        proposal_relax_steps=80,
        proposal_fmax=0.05,
        proposal_optimizer="safe-lbfgs-total",
        min_step_scale=0.1,
        max_step_scale=1.2,
        proposal_trust_radius=1.5,
        walk_trust_radius=5.0,
        fragment_guard_factor=3.0,
        anchor_weight=0.5,
        continuity_weight=0.1,
        history_push_weight=0.1,
        enable_momentum_candidate=True,
        enable_anchor_candidate=False,
        n_bond_pairs=2,
        proposal_pool_size=1,
        same_seed_max_consecutive=3,
        archive_density_weight=0.5,
        novelty_weight=1.0,
        novelty_probe_scales=(1.0,),
        frontier_weight=0.5,
        bandit_exploration_weight=0.75,
        baseline_selection_probability=0.15,
        bandit_energy_weight=1.0,
        max_prototypes=500,
        direction_curvature_source="inner",
        direction_selection_mode="discrete",
        direction_synthesis_mode="none",
        direction_score_sigma_mode="adaptive",
        direction_type_ucb_enabled=False,
        direction_archive_enabled=False,
        direction_probe_enabled=False,
        plateau_evolution_enabled=False,
        archive_escape_momentum_enabled=False,
        random_direction_distribution="unit_gaussian",
        enable_bond_form_break_split=False,
        step_length_mode="per_atom_rms",
        target_step_rms=0.08,
        max_step_rms=0.15,
        max_energy_drop_per_atom=5.0,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(output / "direction_trace.jsonl"),
        accepted_structures_log=str(output / "accepted_structures.jsonl"),
        accepted_structures_dir=str(output / "accepted_minima"),
        local_softening_mode="active_neighbors",
        local_softening_strength=0.15,
        local_softening_penalty="buckingham_repulsive",
        local_softening_xi=0.3,
        local_softening_cutoff=2.0,
        local_softening_cutoff_scale=1.3,
        local_softening_active_count=3,
    )
