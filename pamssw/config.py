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
    dedup_rmsd_tol: float = 0.1
    dedup_energy_tol: float = 1e-3
    rng_seed: int = 0
    oracle_candidates: int = 12
    proposal_relax_steps: int = 40
    proposal_fmax: float = 2e-2
    hvp_epsilon: float = 1e-3
    min_step_scale: float = 0.15
    max_step_scale: float = 1.5
    bias_weight_max: float = 10.0
    proposal_trust_radius: float = 1.5
    walk_trust_radius: float = 4.0
    fragment_guard_factor: float | None = None
    anchor_weight: float = 0.5
    n_bond_pairs: int = 2
    bond_distance_threshold: float | None = None
    lambda_bond_start: float = 0.1
    lambda_bond_end: float = 1.0
    proposal_pool_size: int = 1
    use_archive_acquisition: bool = True
    archive_density_weight: float = 0.5
    novelty_weight: float = 1.0
    frontier_weight: float = 0.5
    bandit_exploration_weight: float = 0.75
    baseline_selection_probability: float = 0.15
    bandit_energy_weight: float = 1.0
    search_mode: SearchMode | str = SearchMode.GLOBAL_MINIMUM
    max_prototypes: int = 1000
    max_force_evals: int | None = None

    def __post_init__(self) -> None:
        positive_ints = {
            "max_trials": self.max_trials,
            "max_steps_per_walk": self.max_steps_per_walk,
            "oracle_candidates": self.oracle_candidates,
            "proposal_relax_steps": self.proposal_relax_steps,
            "proposal_pool_size": self.proposal_pool_size,
            "max_prototypes": self.max_prototypes,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_bond_pairs < 0:
            raise ValueError("n_bond_pairs must be non-negative")
        if self.max_force_evals is not None and self.max_force_evals <= 0:
            raise ValueError("max_force_evals must be positive when set")
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
            "bias_weight_max": self.bias_weight_max,
            "proposal_trust_radius": self.proposal_trust_radius,
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
        }
        for name, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fragment_guard_factor is not None and self.fragment_guard_factor <= 0:
            raise ValueError("fragment_guard_factor must be positive when set")
        if self.bond_distance_threshold is not None and self.bond_distance_threshold <= 0:
            raise ValueError("bond_distance_threshold must be positive when set")
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

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.local_softening_strength <= 0:
            raise ValueError("local_softening_strength must be positive")
        for pair in self.local_softening_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("local_softening_pairs must contain distinct atom pairs")
