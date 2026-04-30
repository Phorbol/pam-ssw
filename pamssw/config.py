from __future__ import annotations

from dataclasses import dataclass, field

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
    n_bond_pairs: int = 2
    stagnation_bond_pair_boost: int = 2
    max_stagnation_bond_pairs: int | None = 10
    bond_distance_threshold: float | None = None
    lambda_bond_start: float = 0.1
    lambda_bond_end: float = 1.0
    proposal_pool_size: int = 1
    same_seed_max_consecutive: int | None = 3
    use_archive_acquisition: bool = True
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
    direction_score_sigma_mode: str = "adaptive"
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
    coordinate_mode: str = "fixed_cell"
    cell_dof_mode: str = "fixed_cell"
    external_pressure: float = 0.0
    cell_metric_weight: float | None = None
    atom_metric_weight: float | None = None
    max_cell_strain_step: float = 0.5
    max_volume_change_per_step: float = 0.3
    min_cell_length: float = 0.5
    min_volume: float = 1e-6
    variable_cell_requires_stress: bool = True
    finite_diff_cell_gradient: bool = False
    fixed_atom_cell_semantics: str = "fixed_fractional"
    lattice_descriptor_weight: float = 1.0
    n_cell_random_candidates: int = 2
    n_coupled_random_candidates: int = 2
    cell_soft_mode_enabled: bool = True
    coupled_soft_mode_enabled: bool = True

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
        if self.trial_progress_patience < 0:
            raise ValueError("trial_progress_patience must be non-negative")
        if self.stagnation_bond_pair_boost < 0:
            raise ValueError("stagnation_bond_pair_boost must be non-negative")
        if self.max_stagnation_bond_pairs is not None and self.max_stagnation_bond_pairs <= 0:
            raise ValueError("max_stagnation_bond_pairs must be positive when set")
        if self.max_force_evals is not None and self.max_force_evals <= 0:
            raise ValueError("max_force_evals must be positive when set")
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
        if self.anchor_mixing_alpha is not None and not 0.0 <= self.anchor_mixing_alpha <= 1.0:
            raise ValueError("anchor_mixing_alpha must be between 0 and 1 when set")
        allowed_optimizers = {"scipy-lbfgsb", "ase-fire", "ase-lbfgs"}
        if self.quench_optimizer not in allowed_optimizers:
            raise ValueError("quench_optimizer must be one of scipy-lbfgsb, ase-fire, ase-lbfgs")
        if self.proposal_optimizer not in allowed_optimizers:
            raise ValueError("proposal_optimizer must be one of scipy-lbfgsb, ase-fire, ase-lbfgs")
        if self.proposal_optimizer_alt is not None and self.proposal_optimizer_alt not in allowed_optimizers:
            raise ValueError("proposal_optimizer_alt must be one of scipy-lbfgsb, ase-fire, ase-lbfgs when set")
        if (
            self.proposal_duplicate_rescue_optimizer is not None
            and self.proposal_duplicate_rescue_optimizer not in allowed_optimizers
        ):
            raise ValueError(
                "proposal_duplicate_rescue_optimizer must be one of scipy-lbfgsb, ase-fire, ase-lbfgs when set"
            )
        if self.direction_curvature_source not in {"inner", "true"}:
            raise ValueError("direction_curvature_source must be inner or true")
        if self.direction_score_sigma_mode not in {"adaptive", "trust_scaled", "fixed_reference"}:
            raise ValueError("direction_score_sigma_mode must be adaptive, trust_scaled, or fixed_reference")
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
        if self.min_step_scale > self.max_step_scale:
            raise ValueError("min_step_scale cannot exceed max_step_scale")
        if self.lambda_bond_start > self.lambda_bond_end:
            raise ValueError("lambda_bond_start cannot exceed lambda_bond_end")
        if self.coordinate_mode not in {"fixed_cell", "variable_cell"}:
            raise ValueError("coordinate_mode must be fixed_cell or variable_cell")
        allowed_cell_modes = {"fixed_cell", "volume_only", "shape_6", "full_9", "slab_xy"}
        if self.cell_dof_mode not in allowed_cell_modes:
            raise ValueError("cell_dof_mode must be fixed_cell, volume_only, shape_6, full_9, or slab_xy")
        if self.coordinate_mode == "fixed_cell" and self.cell_dof_mode != "fixed_cell":
            raise ValueError("cell_dof_mode must be fixed_cell when coordinate_mode is fixed_cell")
        if self.coordinate_mode == "variable_cell" and self.cell_dof_mode == "fixed_cell":
            raise ValueError("variable_cell coordinate_mode requires a non-fixed cell_dof_mode")
        if self.cell_metric_weight is not None and self.cell_metric_weight <= 0:
            raise ValueError("cell_metric_weight must be positive when set")
        if self.atom_metric_weight is not None and self.atom_metric_weight <= 0:
            raise ValueError("atom_metric_weight must be positive when set")
        if self.max_cell_strain_step <= 0:
            raise ValueError("max_cell_strain_step must be positive")
        if self.max_volume_change_per_step <= 0:
            raise ValueError("max_volume_change_per_step must be positive")
        if self.min_cell_length <= 0:
            raise ValueError("min_cell_length must be positive")
        if self.min_volume <= 0:
            raise ValueError("min_volume must be positive")
        if self.fixed_atom_cell_semantics not in {"fixed_fractional", "fixed_cartesian"}:
            raise ValueError("fixed_atom_cell_semantics must be fixed_fractional or fixed_cartesian")
        if self.lattice_descriptor_weight <= 0:
            raise ValueError("lattice_descriptor_weight must be positive")
        if self.n_cell_random_candidates < 0:
            raise ValueError("n_cell_random_candidates must be non-negative")
        if self.n_coupled_random_candidates < 0:
            raise ValueError("n_coupled_random_candidates must be non-negative")
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
        for pair in self.local_softening_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("local_softening_pairs must contain distinct atom pairs")
