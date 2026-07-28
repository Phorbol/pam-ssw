from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from .acquisition import SearchMode


@dataclass(frozen=True)
class RelaxConfig:
    """Settings for a true-PES local minimization."""
    fmax: float = 1e-3
    maxiter: int = 200

    def __post_init__(self) -> None:
        if self.fmax <= 0:
            raise ValueError("fmax must be positive")
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")


@dataclass(frozen=True)
class SSWConfig:
    """High-level settings for stochastic surface walking searches."""
    max_trials: int = 12
    max_steps_per_walk: int = 6
    target_uphill_energy: float = 0.6
    target_negative_curvature: float = 0.05
    quench_fmax: float = 1e-3
    quench_maxiter: int = 400
    quench_optimizer: str = "scipy-lbfgsb"
    quench_fallback_optimizer: str | None = None
    dedup_rmsd_tol: float = 0.1
    dedup_energy_tol: float = 1e-3
    rng_seed: int = 0
    oracle_candidates: int = 12
    proposal_relax_steps: int = 40
    proposal_fmax: float = 2e-2
    proposal_optimizer: str = "ase-fire"
    hvp_epsilon: float = 1e-3
    min_step_scale: float = 0.15
    max_step_scale: float = 1.5
    bias_weight_min: float = 0.0
    bias_weight_max: float = 10.0
    proposal_trust_radius: float | None = 1.5
    walk_trust_radius: float = 4.0
    fragment_guard_factor: float | None = None
    anchor_weight: float = 0.5
    anchor_mixing_alpha: float | None = None
    continuity_weight: float = 0.1
    enable_outcome_gated_continuity: bool = True
    history_push_weight: float = 0.1
    enable_momentum_candidate: bool = True
    enable_anchor_candidate: bool = False
    n_bond_pairs: int = 2
    random_direction_distribution: str = "unit_gaussian"
    enable_bond_form_break_split: bool = False
    n_bond_formation_pairs: int = 2
    n_bond_breaking_pairs: int = 1
    bond_formation_max_distance: float = 4.0
    bond_breaking_max_distance: float = 2.0
    stagnation_bond_pair_boost: int = 2
    max_stagnation_bond_pairs: int | None = 10
    bond_distance_threshold: float | None = None
    lambda_bond_start: float = 0.1
    lambda_bond_end: float = 1.0
    proposal_pool_size: int = 1
    same_seed_max_consecutive: int | None = 3
    use_archive_acquisition: bool = True
    seed_selection_mode: str = "archive_ucb"
    metropolis_temperature: float = 0.26
    archive_density_weight: float = 0.5
    novelty_weight: float = 1.0
    novelty_probe_scales: tuple[float, ...] = (1.0,)
    frontier_weight: float = 0.5
    bandit_exploration_weight: float = 0.75
    baseline_selection_probability: float = 0.15
    bandit_energy_weight: float = 1.0
    search_mode: SearchMode | str = SearchMode.GLOBAL_MINIMUM
    max_prototypes: int = 1000
    max_force_evals: int | None = None
    accepted_structures_log: str | None = None
    accepted_structures_dir: str | None = None
    write_proposal_minima: bool = False
    proposal_minima_dir: str | None = None
    write_relaxation_trajectories: bool = False
    relaxation_trajectory_dir: str | None = None
    relaxation_trajectory_stride: int = 1
    direction_curvature_source: str = "inner"
    direction_selection_mode: str = "discrete"
    block_krylov_blocks: int = 2
    block_krylov_depth: int = 3
    direction_synthesis_mode: str = "none"
    regularized_ritz_top_k: int = 5
    direction_score_sigma_mode: str = "adaptive"
    direction_type_ucb_enabled: bool = False
    direction_type_success_weight: float = 0.0
    direction_type_exploration_weight: float = 0.1
    direction_type_ucb_window: int = 40
    direction_archive_enabled: bool = False
    direction_archive_max_records: int = 10000
    direction_archive_success_only: bool = False
    direction_archive_path: str | None = None
    direction_probe_enabled: bool = False
    direction_probe_top_k: int = 5
    direction_probe_ds_scale: float = 0.5
    direction_probe_uphill_low: float = 0.05
    direction_probe_uphill_high: float = 1.0
    direction_probe_collision_distance: float = 0.5
    plateau_evolution_enabled: bool = False
    plateau_patience_trials: int = 20
    plateau_evolution_children: int = 5
    plateau_evolution_crossover_pairs: int = 3
    plateau_evolution_mutation_count: int = 2
    plateau_evolution_history_limit: int = 10
    archive_escape_momentum_enabled: bool = False
    archive_escape_momentum_limit: int = 2
    archive_escape_momentum_history_limit: int = 16
    archive_escape_momentum_same_seed_first: bool = True
    step_length_mode: str = "per_atom_rms"
    target_step_rms: float = 0.15
    max_step_rms: float = 0.35
    step_rms_scope: str = "all_atoms"
    step_active_threshold: float = 1e-4
    step_error_tolerance: float = 1.0
    step_gamma_down: float = 0.5
    step_gamma_up: float = 1.15
    min_escape_energy_delta: float = 0.1
    min_escape_descriptor_delta: float = 0.1
    # coverage_gain is bounded by 1.0; 1.01 disables novelty-only escape by default.
    min_escape_novelty: float = 1.01
    trial_progress_patience: int = 0
    trial_progress_boost_factor: float = 1.5
    trial_progress_max_boost: float = 2.0
    trial_progress_duplicate_tolerance: float = 0.75
    proposal_optimizer_alt: str | None = None
    proposal_duplicate_rescue_optimizer: str | None = None
    max_energy_drop_per_atom: float | None = 5.0
    direction_diagnostics_enabled: bool = False
    direction_diagnostics_path: str | None = None

    def __post_init__(self) -> None:
        positive_ints = {
            "max_trials": self.max_trials,
            "max_steps_per_walk": self.max_steps_per_walk,
            "oracle_candidates": self.oracle_candidates,
            "quench_maxiter": self.quench_maxiter,
            "proposal_pool_size": self.proposal_pool_size,
            "max_prototypes": self.max_prototypes,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.proposal_relax_steps < 0:
            raise ValueError("proposal_relax_steps must be non-negative")
        if self.n_bond_pairs < 0:
            raise ValueError("n_bond_pairs must be non-negative")
        if self.random_direction_distribution not in {"unit_gaussian", "mass_weighted"}:
            raise ValueError("random_direction_distribution must be unit_gaussian or mass_weighted")
        if not isinstance(self.enable_bond_form_break_split, bool):
            raise ValueError("enable_bond_form_break_split must be a boolean")
        for name in ("n_bond_formation_pairs", "n_bond_breaking_pairs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.trial_progress_patience < 0:
            raise ValueError("trial_progress_patience must be non-negative")
        if not isinstance(self.plateau_evolution_enabled, bool):
            raise ValueError("plateau_evolution_enabled must be a boolean")
        if not isinstance(self.archive_escape_momentum_enabled, bool):
            raise ValueError("archive_escape_momentum_enabled must be a boolean")
        if not isinstance(self.archive_escape_momentum_same_seed_first, bool):
            raise ValueError("archive_escape_momentum_same_seed_first must be a boolean")
        for name in (
            "plateau_patience_trials",
            "plateau_evolution_children",
            "plateau_evolution_crossover_pairs",
            "plateau_evolution_mutation_count",
            "plateau_evolution_history_limit",
            "archive_escape_momentum_limit",
            "archive_escape_momentum_history_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.regularized_ritz_top_k, bool) or not isinstance(self.regularized_ritz_top_k, int):
            raise ValueError("regularized_ritz_top_k must be a positive integer")
        if self.regularized_ritz_top_k <= 0:
            raise ValueError("regularized_ritz_top_k must be a positive integer")
        for name in ("block_krylov_blocks", "block_krylov_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.direction_type_ucb_window, bool) or not isinstance(self.direction_type_ucb_window, int):
            raise ValueError("direction_type_ucb_window must be a positive integer")
        if self.direction_type_ucb_window <= 0:
            raise ValueError("direction_type_ucb_window must be a positive integer")
        if not isinstance(self.direction_type_ucb_enabled, bool):
            raise ValueError("direction_type_ucb_enabled must be a boolean")
        if not isinstance(self.direction_archive_enabled, bool):
            raise ValueError("direction_archive_enabled must be a boolean")
        if not isinstance(self.direction_archive_success_only, bool):
            raise ValueError("direction_archive_success_only must be a boolean")
        if isinstance(self.direction_archive_max_records, bool) or not isinstance(self.direction_archive_max_records, int):
            raise ValueError("direction_archive_max_records must be a positive integer")
        if self.direction_archive_max_records <= 0:
            raise ValueError("direction_archive_max_records must be a positive integer")
        if self.direction_archive_path is not None and (
            not isinstance(self.direction_archive_path, str) or self.direction_archive_path == ""
        ):
            raise ValueError("direction_archive_path must be None or a non-empty string")
        if self.stagnation_bond_pair_boost < 0:
            raise ValueError("stagnation_bond_pair_boost must be non-negative")
        if self.max_stagnation_bond_pairs is not None and self.max_stagnation_bond_pairs <= 0:
            raise ValueError("max_stagnation_bond_pairs must be positive when set")
        if self.max_force_evals is not None and self.max_force_evals <= 0:
            raise ValueError("max_force_evals must be positive when set")
        if self.max_energy_drop_per_atom is not None and self.max_energy_drop_per_atom <= 0:
            raise ValueError("max_energy_drop_per_atom must be positive when set")
        if self.same_seed_max_consecutive is not None and self.same_seed_max_consecutive <= 0:
            raise ValueError("same_seed_max_consecutive must be positive when set")
        positive_floats = {
            "target_uphill_energy": self.target_uphill_energy,
            "target_negative_curvature": self.target_negative_curvature,
            "quench_fmax": self.quench_fmax,
            "dedup_rmsd_tol": self.dedup_rmsd_tol,
            "dedup_energy_tol": self.dedup_energy_tol,
            "proposal_fmax": self.proposal_fmax,
            "hvp_epsilon": self.hvp_epsilon,
            "min_step_scale": self.min_step_scale,
            "max_step_scale": self.max_step_scale,
            "walk_trust_radius": self.walk_trust_radius,
            "anchor_weight": self.anchor_weight,
            "lambda_bond_start": self.lambda_bond_start,
            "lambda_bond_end": self.lambda_bond_end,
            "archive_density_weight": self.archive_density_weight,
            "novelty_weight": self.novelty_weight,
            "frontier_weight": self.frontier_weight,
            "bandit_exploration_weight": self.bandit_exploration_weight,
            "baseline_selection_probability": self.baseline_selection_probability,
            "bandit_energy_weight": self.bandit_energy_weight,
            "metropolis_temperature": self.metropolis_temperature,
            "step_error_tolerance": self.step_error_tolerance,
            "step_gamma_down": self.step_gamma_down,
            "step_gamma_up": self.step_gamma_up,
            "trial_progress_boost_factor": self.trial_progress_boost_factor,
            "trial_progress_max_boost": self.trial_progress_max_boost,
        }
        for name, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bias_weight_min < 0:
            raise ValueError("bias_weight_min must be non-negative")
        if self.bias_weight_max <= 0:
            raise ValueError("bias_weight_max must be positive")
        if self.bias_weight_min > self.bias_weight_max:
            raise ValueError("bias_weight_min cannot exceed bias_weight_max")
        if self.history_push_weight < 0:
            raise ValueError("history_push_weight must be non-negative")
        if self.continuity_weight < 0:
            raise ValueError("continuity_weight must be non-negative")
        if not isfinite(self.direction_type_success_weight) or self.direction_type_success_weight < 0:
            raise ValueError("direction_type_success_weight must be finite and non-negative")
        if not isfinite(self.direction_type_exploration_weight) or self.direction_type_exploration_weight < 0:
            raise ValueError("direction_type_exploration_weight must be finite and non-negative")
        if self.min_escape_energy_delta < 0:
            raise ValueError("min_escape_energy_delta must be non-negative")
        if self.min_escape_descriptor_delta < 0:
            raise ValueError("min_escape_descriptor_delta must be non-negative")
        if self.min_escape_novelty < 0:
            raise ValueError("min_escape_novelty must be non-negative")
        if self.trial_progress_boost_factor <= 1.0:
            raise ValueError("trial_progress_boost_factor must be greater than 1")
        if self.trial_progress_max_boost < 1.0:
            raise ValueError("trial_progress_max_boost must be at least 1")
        if not 0.0 <= self.trial_progress_duplicate_tolerance <= 1.0:
            raise ValueError("trial_progress_duplicate_tolerance must be between 0 and 1")
        if not self.novelty_probe_scales or any(scale <= 0 for scale in self.novelty_probe_scales):
            raise ValueError("novelty_probe_scales must contain positive values")
        if self.proposal_trust_radius is not None and self.proposal_trust_radius <= 0:
            raise ValueError("proposal_trust_radius must be positive when set")
        if self.seed_selection_mode not in {"archive_ucb", "metropolis_chain"}:
            raise ValueError("seed_selection_mode must be archive_ucb or metropolis_chain")
        if self.anchor_mixing_alpha is not None and not 0.0 <= self.anchor_mixing_alpha <= 1.0:
            raise ValueError("anchor_mixing_alpha must be between 0 and 1 when set")
        quench_optimizers = {"scipy-lbfgsb", "ase-fire", "ase-lbfgs"}
        proposal_optimizers = quench_optimizers | {
            "ase-fire2",
            "safe-lbfgs-total",
            "bias-separated-lbfgs",
        }
        if self.quench_optimizer not in quench_optimizers:
            raise ValueError("quench_optimizer must be one of scipy-lbfgsb, ase-fire, ase-lbfgs")
        if (
            self.quench_fallback_optimizer is not None
            and self.quench_fallback_optimizer not in quench_optimizers
        ):
            raise ValueError(
                "quench_fallback_optimizer must be one of "
                "scipy-lbfgsb, ase-fire, ase-lbfgs when set"
            )
        if self.proposal_optimizer not in proposal_optimizers:
            raise ValueError(
                "proposal_optimizer must be one of scipy-lbfgsb, ase-fire, "
                "ase-lbfgs, ase-fire2, safe-lbfgs-total, bias-separated-lbfgs"
            )
        if self.proposal_optimizer_alt is not None and self.proposal_optimizer_alt not in proposal_optimizers:
            raise ValueError(
                "proposal_optimizer_alt must be one of scipy-lbfgsb, ase-fire, "
                "ase-lbfgs, ase-fire2, safe-lbfgs-total, bias-separated-lbfgs when set"
            )
        if (
            self.proposal_duplicate_rescue_optimizer is not None
            and self.proposal_duplicate_rescue_optimizer not in proposal_optimizers
        ):
            raise ValueError(
                "proposal_duplicate_rescue_optimizer must be one of scipy-lbfgsb, "
                "ase-fire, ase-lbfgs, ase-fire2, safe-lbfgs-total, "
                "bias-separated-lbfgs when set"
            )
        if self.direction_curvature_source not in {"inner", "true"}:
            raise ValueError("direction_curvature_source must be inner or true")
        direction_selection_modes = {
            "discrete",
            "rayleigh_ritz",
            "block_krylov",
            "exact_anchor",
            "anchor_krylov",
        }
        if self.direction_selection_mode not in direction_selection_modes:
            raise ValueError(
                "direction_selection_mode must be discrete, rayleigh_ritz, "
                "block_krylov, exact_anchor, or anchor_krylov"
            )
        if self.direction_synthesis_mode not in {"none", "regularized_ritz"}:
            raise ValueError("direction_synthesis_mode must be none or regularized_ritz")
        if (
            self.direction_selection_mode
            in {
                "rayleigh_ritz",
                "block_krylov",
                "exact_anchor",
                "anchor_krylov",
            }
            and self.direction_synthesis_mode == "regularized_ritz"
        ):
            raise ValueError(
                "explicit direction_selection_mode cannot be combined with "
                "regularized_ritz synthesis"
            )
        if self.direction_score_sigma_mode not in {"adaptive", "trust_scaled", "fixed_reference"}:
            raise ValueError("direction_score_sigma_mode must be adaptive, trust_scaled, or fixed_reference")
        if self.step_length_mode not in {"curvature_adaptive", "per_atom_rms"}:
            raise ValueError("step_length_mode must be curvature_adaptive or per_atom_rms")
        if self.step_rms_scope not in {"all_atoms", "active_atoms"}:
            raise ValueError("step_rms_scope must be all_atoms or active_atoms")
        if not isfinite(self.target_step_rms) or self.target_step_rms <= 0:
            raise ValueError("target_step_rms must be positive")
        if not isfinite(self.max_step_rms) or self.max_step_rms <= 0:
            raise ValueError("max_step_rms must be positive")
        if self.target_step_rms > self.max_step_rms:
            raise ValueError("target_step_rms cannot exceed max_step_rms")
        if not isfinite(self.step_active_threshold) or self.step_active_threshold <= 0:
            raise ValueError("step_active_threshold must be positive")
        if self.direction_diagnostics_enabled and self.direction_diagnostics_path is None:
            raise ValueError("direction_diagnostics_path must be set when direction diagnostics are enabled")
        if self.write_proposal_minima and self.proposal_minima_dir is None:
            raise ValueError("proposal_minima_dir must be set when write_proposal_minima is enabled")
        if self.write_relaxation_trajectories and self.relaxation_trajectory_dir is None:
            raise ValueError("relaxation_trajectory_dir must be set when write_relaxation_trajectories is enabled")
        if self.relaxation_trajectory_stride <= 0:
            raise ValueError("relaxation_trajectory_stride must be positive")
        if self.fragment_guard_factor is not None and self.fragment_guard_factor <= 0:
            raise ValueError("fragment_guard_factor must be positive when set")
        if self.bond_distance_threshold is not None and self.bond_distance_threshold <= 0:
            raise ValueError("bond_distance_threshold must be positive when set")
        if not isfinite(self.bond_formation_max_distance) or self.bond_formation_max_distance <= 0:
            raise ValueError("bond_formation_max_distance must be finite and positive")
        if not isfinite(self.bond_breaking_max_distance) or self.bond_breaking_max_distance <= 0:
            raise ValueError("bond_breaking_max_distance must be finite and positive")
        if self.min_step_scale > self.max_step_scale:
            raise ValueError("min_step_scale cannot exceed max_step_scale")
        if self.lambda_bond_start > self.lambda_bond_end:
            raise ValueError("lambda_bond_start cannot exceed lambda_bond_end")
        try:
            SearchMode(self.search_mode)
        except ValueError as exc:
            raise ValueError("search_mode must be a documented SearchMode") from exc


@dataclass(frozen=True)
class LSSSWConfig(SSWConfig):
    """Settings for locally softened stochastic surface walking."""
    local_softening_strength: float = 0.6
    local_softening_pairs: list[tuple[int, int]] = field(default_factory=list)
    local_softening_mode: str = "neighbor_auto"
    local_softening_cutoff_scale: float = 1.25
    local_softening_active_count: int | None = None
    local_softening_penalty: str = "buckingham_repulsive"
    local_softening_xi: float = 0.3
    local_softening_cutoff: float | None = 2.0
    local_softening_adaptive_strength: bool = False
    local_softening_max_strength_scale: float = 3.0
    local_softening_deviation_scale: float = 0.25
    choice_aligned_softening_enabled: bool = False
    choice_aligned_softening_cos_threshold: float = 0.3

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.local_softening_strength <= 0:
            raise ValueError("local_softening_strength must be positive")
        if self.local_softening_mode not in {"manual", "neighbor_auto", "active_neighbors"}:
            raise ValueError("local_softening_mode must be manual, neighbor_auto, or active_neighbors")
        if self.local_softening_cutoff_scale <= 0:
            raise ValueError("local_softening_cutoff_scale must be positive")
        if self.local_softening_active_count is not None and self.local_softening_active_count <= 0:
            raise ValueError("local_softening_active_count must be positive when set")
        if self.local_softening_penalty not in {"gaussian_well", "buckingham_repulsive"}:
            raise ValueError("local_softening_penalty must be gaussian_well or buckingham_repulsive")
        if self.local_softening_xi <= 0:
            raise ValueError("local_softening_xi must be positive")
        if self.local_softening_cutoff is not None and self.local_softening_cutoff <= 0:
            raise ValueError("local_softening_cutoff must be positive when set")
        if self.local_softening_max_strength_scale < 1.0:
            raise ValueError("local_softening_max_strength_scale must be at least 1")
        if self.local_softening_deviation_scale <= 0:
            raise ValueError("local_softening_deviation_scale must be positive")
        if not isinstance(self.choice_aligned_softening_enabled, bool):
            raise ValueError("choice_aligned_softening_enabled must be a boolean")
        if not -1.0 <= self.choice_aligned_softening_cos_threshold <= 1.0:
            raise ValueError("choice_aligned_softening_cos_threshold must be between -1 and 1")
        for pair in self.local_softening_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("local_softening_pairs must contain distinct atom pairs")
