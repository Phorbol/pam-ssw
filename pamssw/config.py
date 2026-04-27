from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RelaxConfig:
    fmax: float = 1e-3
    maxiter: int = 200

    def __post_init__(self) -> None:
        if self.fmax <= 0:
            raise ValueError("fmax must be positive")
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")


@dataclass(frozen=True)
class SSWConfig:
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
    proposal_fmax: float = 5e-3
    min_step_scale: float = 0.15
    max_step_scale: float = 1.5
    use_archive_acquisition: bool = True

    def __post_init__(self) -> None:
        positive_ints = {
            "max_trials": self.max_trials,
            "max_steps_per_walk": self.max_steps_per_walk,
            "oracle_candidates": self.oracle_candidates,
            "proposal_relax_steps": self.proposal_relax_steps,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        positive_floats = {
            "target_uphill_energy": self.target_uphill_energy,
            "target_negative_curvature": self.target_negative_curvature,
            "quench_fmax": self.quench_fmax,
            "dedup_rmsd_tol": self.dedup_rmsd_tol,
            "dedup_energy_tol": self.dedup_energy_tol,
            "proposal_fmax": self.proposal_fmax,
            "min_step_scale": self.min_step_scale,
            "max_step_scale": self.max_step_scale,
        }
        for name, value in positive_floats.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_step_scale > self.max_step_scale:
            raise ValueError("min_step_scale cannot exceed max_step_scale")


@dataclass(frozen=True)
class LSSSWConfig(SSWConfig):
    local_softening_strength: float = 0.6
    local_softening_pairs: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.local_softening_strength <= 0:
            raise ValueError("local_softening_strength must be positive")
        for pair in self.local_softening_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("local_softening_pairs must contain distinct atom pairs")
