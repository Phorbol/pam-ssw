from __future__ import annotations

import json
import sys
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from math import log1p, sqrt
from numbers import Real
from pathlib import Path

from ase import Atoms
from ase.data import atomic_masses
from ase.io import write
import numpy as np

from .acquisition import AcquisitionPolicy, BanditSelector, ProposalOutcome, ProposalScorer
from .accounting import BudgetExceeded, EvalCounter
from .bias import GaussianBiasTerm
from .config import LSSSWConfig, RelaxConfig, SSWConfig
from .coordinates import CartesianCoordinates, TangentVector
from .fingerprint import descriptor_distance, structural_descriptor
from .pbc import mic_displacement, mic_distance_matrix, wrap_positions
from .reference_dimer import ReferenceDimerResult, ReferenceDimerRotator, sample_mixed_mode
from .relax import Relaxer
from .result import RelaxOutcomeClass, RelaxResult, SearchResult, StatsValue, WalkRecord
from .rigid import project_out_rigid_body_modes, rigid_body_overlap
from .softening import LocalSofteningModel
from .state import State


class ProposalPotential:
    def __init__(
        self,
        calculator,
        biases: list[GaussianBiasTerm] | None = None,
        softening: LocalSofteningModel | None = None,
    ) -> None:
        self.calculator = calculator
        self.biases = biases or []
        self.softening = softening

    def evaluate(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        energy, gradient = self.calculator.evaluate_flat(flat_positions, template)
        total_gradient = gradient.copy()
        total_energy = energy
        for bias in self.biases:
            bias_energy, bias_gradient = bias.evaluate(flat_positions, cell=template.cell, pbc=template.pbc)
            total_energy += bias_energy
            total_gradient += bias_gradient
        if self.softening is not None:
            soft_energy, soft_gradient = self.softening.evaluate(flat_positions)
            total_energy += soft_energy
            total_gradient += soft_gradient
        return float(total_energy), total_gradient


@dataclass(frozen=True)
class GeometryValidator:
    min_distance: float = 0.5
    covalent_collision_scale: float = 0.65

    def is_valid_state(self, state: State) -> bool:
        if not np.all(np.isfinite(state.positions)):
            return False
        if state.cell is not None and not np.all(np.isfinite(state.cell)):
            return False
        if state.n_atoms < 2:
            return True
        if any(state.pbc) and state.cell is not None:
            distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
        else:
            distances = np.linalg.norm(state.positions[:, None, :] - state.positions[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        thresholds = self._pair_distance_thresholds(state.numbers)
        return bool(np.all(distances >= thresholds))

    def _pair_distance_thresholds(self, numbers: np.ndarray) -> np.ndarray:
        radii = np.asarray([self._covalent_radius(int(number)) for number in numbers], dtype=float)
        thresholds = self.covalent_collision_scale * (radii[:, None] + radii[None, :])
        thresholds = np.maximum(thresholds, self.min_distance)
        np.fill_diagonal(thresholds, -np.inf)
        return thresholds

    @staticmethod
    def _covalent_radius(number: int) -> float:
        return {
            1: 0.31,
            5: 0.84,
            6: 0.76,
            7: 0.71,
            8: 0.66,
            14: 1.11,
            15: 1.07,
            16: 1.05,
            46: 1.39,
        }.get(number, 0.4)

    def is_valid_evaluation(self, state: State, calculator) -> bool:
        if not self.is_valid_state(state):
            return False
        try:
            energy, gradient = calculator.evaluate_flat(state.flatten_positions(), state)
        except Exception:
            return False
        return bool(np.isfinite(energy) and np.all(np.isfinite(gradient)))


@dataclass
class DirectionChoice:
    direction: np.ndarray
    curvature: float
    kind: DirectionCandidateKind
    candidate_count: int
    mean_rigid_body_overlap: float = 0.0
    mean_post_projection_rigid_body_overlap: float = 0.0
    score: float | None = None
    evolved_candidate_count: int = 0
    archive_momentum_candidate_count: int = 0
    true_curvature: float | None = None
    biased_curvature: float | None = None


@dataclass(frozen=True)
class TrustRegionUpdate:
    predicted_delta: float
    true_delta: float
    model_error: float
    damaged: bool
    sigma_scale: float
    weight_scale: float
    action: str
    sigma_action: str = "hold"
    weight_action: str = "hold"


@dataclass(frozen=True)
class DirectQPStepResult:
    state: State
    step: np.ndarray
    energy_before: float
    energy_after: float
    predicted_delta: float
    true_delta: float
    model_error: float
    progress: float
    target_error: float
    gamma: float
    kappa: float
    action: str
    rejected: bool = False


@dataclass(frozen=True)
class StepLengthUpdate:
    predicted_delta: float
    true_delta: float
    model_error: float
    damaged: bool
    sigma_scale: float
    action: str


@dataclass(frozen=True)
class BiasStrengthUpdate:
    post_bias_curvature: float
    curvature_flipped: bool
    weight_scale: float
    action: str


@dataclass(frozen=True)
class StepLengthController:
    error_tolerance: float = 1.0
    gamma_down: float = 0.5
    gamma_up: float = 1.15
    min_scale: float = 0.25
    max_scale: float = 2.0
    damage_ratio: float = 8.0
    active_bound_tolerance: float = 0.35
    epsilon: float = 1e-8

    def update(
        self,
        curvature: float,
        sigma: float,
        true_delta: float,
        sigma_scale: float,
        g_parallel: float = 0.0,
        error_floor: float = 0.0,
        active_bound_fraction: float = 0.0,
    ) -> StepLengthUpdate:
        predicted_delta = self.predicted_delta(curvature, sigma, g_parallel=g_parallel)
        denominator = max(abs(predicted_delta), float(error_floor)) + self.epsilon
        model_error = abs(true_delta - predicted_delta) / denominator
        damaged = true_delta > max(1.0, self.damage_ratio * denominator)
        if damaged or model_error > self.error_tolerance or active_bound_fraction > self.active_bound_tolerance:
            return StepLengthUpdate(
                predicted_delta=predicted_delta,
                true_delta=float(true_delta),
                model_error=float(model_error),
                damaged=damaged,
                sigma_scale=self._clip(sigma_scale * self.gamma_down),
                action="shrink",
            )
        return StepLengthUpdate(
            predicted_delta=predicted_delta,
            true_delta=float(true_delta),
            model_error=float(model_error),
            damaged=False,
            sigma_scale=self._clip(sigma_scale * self.gamma_up),
            action="expand",
        )

    @staticmethod
    def predicted_delta(curvature: float, sigma: float, g_parallel: float = 0.0) -> float:
        return float(sigma * g_parallel + 0.5 * sigma * sigma * curvature)

    def _clip(self, value: float) -> float:
        return float(np.clip(value, self.min_scale, self.max_scale))


@dataclass(frozen=True)
class BiasStrengthController:
    gamma_down: float = 0.5
    gamma_up: float = 1.15
    min_scale: float = 0.25
    max_scale: float = 2.0

    def update(
        self,
        curvature: float,
        sigma: float,
        bias_weight: float,
        weight_scale: float,
        bias_induced_damage: bool = False,
    ) -> BiasStrengthUpdate:
        post_bias_curvature = float(curvature - bias_weight / max(sigma * sigma, 1e-12))
        curvature_flipped = post_bias_curvature < 0.0
        if bias_induced_damage:
            return BiasStrengthUpdate(
                post_bias_curvature=post_bias_curvature,
                curvature_flipped=curvature_flipped,
                weight_scale=self._clip(weight_scale * self.gamma_down),
                action="shrink",
            )
        if not curvature_flipped:
            return BiasStrengthUpdate(
                post_bias_curvature=post_bias_curvature,
                curvature_flipped=False,
                weight_scale=self._clip(weight_scale * self.gamma_up),
                action="expand",
            )
        return BiasStrengthUpdate(
            post_bias_curvature=post_bias_curvature,
            curvature_flipped=True,
            weight_scale=self._clip(weight_scale),
            action="hold",
        )

    def _clip(self, value: float) -> float:
        return float(np.clip(value, self.min_scale, self.max_scale))


@dataclass(frozen=True)
class TrustRegionBiasController:
    step_length: StepLengthController = field(default_factory=StepLengthController)
    bias_strength: BiasStrengthController = field(default_factory=BiasStrengthController)

    @property
    def error_tolerance(self) -> float:
        return self.step_length.error_tolerance

    def update(
        self,
        curvature: float,
        sigma: float,
        true_delta: float,
        sigma_scale: float,
        weight_scale: float,
        g_parallel: float = 0.0,
        error_floor: float = 0.0,
        active_bound_fraction: float = 0.0,
        bias_weight: float = 0.0,
        bias_induced_damage: bool = False,
    ) -> TrustRegionUpdate:
        sigma_update = self.step_length.update(
            curvature=curvature,
            sigma=sigma,
            true_delta=true_delta,
            sigma_scale=sigma_scale,
            g_parallel=g_parallel,
            error_floor=error_floor,
            active_bound_fraction=active_bound_fraction,
        )
        weight_update = self.bias_strength.update(
            curvature=curvature,
            sigma=sigma,
            bias_weight=bias_weight,
            weight_scale=weight_scale,
            bias_induced_damage=bias_induced_damage,
        )
        if sigma_update.action == "shrink" and weight_update.action == "expand":
            weight_update = BiasStrengthUpdate(
                post_bias_curvature=weight_update.post_bias_curvature,
                curvature_flipped=weight_update.curvature_flipped,
                weight_scale=weight_scale,
                action="hold",
            )
        action = sigma_update.action if sigma_update.action != "expand" else weight_update.action
        return TrustRegionUpdate(
            predicted_delta=sigma_update.predicted_delta,
            true_delta=sigma_update.true_delta,
            model_error=sigma_update.model_error,
            damaged=sigma_update.damaged,
            sigma_scale=sigma_update.sigma_scale,
            weight_scale=weight_update.weight_scale,
            action=action,
            sigma_action=sigma_update.action,
            weight_action=weight_update.action,
        )

    @staticmethod
    def predicted_delta(curvature: float, sigma: float, g_parallel: float = 0.0) -> float:
        return StepLengthController.predicted_delta(curvature, sigma, g_parallel=g_parallel)


@dataclass
class StepTargetController:
    fallback_target: float
    eta_energy_scale: float = 0.2
    min_fraction: float = 0.05
    max_factor: float = 5.0
    target_escape_rate: float = 0.2
    damage_tolerance: float = 0.3
    feedback_warmup_trials: int = 4
    gamma_up: float = 1.1
    min_escape_energy_delta: float = 0.1
    min_escape_descriptor_delta: float = 0.1
    min_escape_novelty: float = 1.01
    progress_patience: int = 0
    progress_boost_factor: float = 1.5
    progress_max_boost: float = 2.0
    progress_duplicate_tolerance: float = 0.75

    def __post_init__(self) -> None:
        if self.fallback_target <= 0.0:
            raise ValueError("fallback_target must be positive")
        self.min_target = self.min_fraction * self.fallback_target
        self.max_target = self.max_factor * self.fallback_target
        self.multiplier = 1.0
        self.trials = 0
        self.escapes = 0
        self.raw_escapes = 0
        self.damage_events = 0
        self.last_target = self.fallback_target
        self.max_escape_energy_delta_seen = 0.0
        self.max_escape_descriptor_delta_seen = 0.0
        self.max_escape_novelty_seen = 0.0
        self.progress_boost = 1.0
        self.no_global_progress_trials = 0

    def target(self, archive=None) -> float:
        raw_target = self._archive_target(archive)
        self.last_target = float(np.clip(raw_target * self.multiplier * self.progress_boost, self.min_target, self.max_target))
        return self.last_target

    def record_trial(
        self,
        escaped: bool,
        damaged: bool,
        seed_energy: float | None = None,
        new_energy: float | None = None,
        energy_delta: float | None = None,
        descriptor_delta: float | None = None,
        novelty_gain: float | None = None,
        global_improved: bool | None = None,
        duplicate_rate: float | None = None,
    ) -> None:
        self.trials += 1
        self.raw_escapes += int(escaped)
        meaningful_escape = self._meaningful_escape(
            escaped=escaped,
            seed_energy=seed_energy,
            new_energy=new_energy,
            energy_delta=energy_delta,
            descriptor_delta=descriptor_delta,
            novelty_gain=novelty_gain,
        )
        self.escapes += int(meaningful_escape)
        self.damage_events += int(damaged)
        if energy_delta is None and seed_energy is not None and new_energy is not None:
            energy_delta = float(new_energy) - float(seed_energy)
        if energy_delta is not None:
            self.max_escape_energy_delta_seen = max(self.max_escape_energy_delta_seen, abs(float(energy_delta)))
        if descriptor_delta is not None:
            self.max_escape_descriptor_delta_seen = max(
                self.max_escape_descriptor_delta_seen,
                float(descriptor_delta),
            )
        if novelty_gain is not None:
            self.max_escape_novelty_seen = max(self.max_escape_novelty_seen, float(novelty_gain))
        escape_rate = self.escapes / self.trials
        damage_rate = self.damage_events / self.trials
        if escape_rate < self.target_escape_rate:
            self.multiplier = float(np.clip(self.multiplier * self.gamma_up, 0.75, 4.0))
        self._update_progress_boost(
            meaningful_escape=meaningful_escape,
            damaged=damaged,
            global_improved=global_improved,
            duplicate_rate=duplicate_rate,
        )

    def _update_progress_boost(
        self,
        *,
        meaningful_escape: bool,
        damaged: bool,
        global_improved: bool | None,
        duplicate_rate: float | None,
    ) -> None:
        if self.progress_patience <= 0:
            return
        duplicate_spike = (
            duplicate_rate is not None
            and float(duplicate_rate) >= self.progress_duplicate_tolerance
        )
        if damaged or duplicate_spike:
            self.no_global_progress_trials = 0
            self.progress_boost = 1.0
            return
        improved = meaningful_escape if global_improved is None else bool(global_improved)
        if improved and meaningful_escape:
            self.no_global_progress_trials = 0
            self.progress_boost = 1.0
            return
        self.no_global_progress_trials += 1
        if self.no_global_progress_trials >= self.progress_patience:
            self.progress_boost = float(
                np.clip(self.progress_boost * self.progress_boost_factor, 1.0, self.progress_max_boost)
            )

    def _meaningful_escape(
        self,
        *,
        escaped: bool,
        seed_energy: float | None,
        new_energy: float | None,
        energy_delta: float | None,
        descriptor_delta: float | None,
        novelty_gain: float | None,
    ) -> bool:
        if not escaped:
            return False
        if energy_delta is None and seed_energy is not None and new_energy is not None:
            energy_delta = float(new_energy) - float(seed_energy)
        has_evidence = energy_delta is not None or descriptor_delta is not None or novelty_gain is not None
        if not has_evidence:
            return True
        energy_ok = energy_delta is not None and abs(float(energy_delta)) >= self.min_escape_energy_delta
        descriptor_ok = (
            descriptor_delta is not None and float(descriptor_delta) >= self.min_escape_descriptor_delta
        )
        novelty_ok = novelty_gain is not None and float(novelty_gain) >= self.min_escape_novelty
        return bool(energy_ok or descriptor_ok or novelty_ok)

    def stats(self) -> dict[str, float | int]:
        escape_rate = self.escapes / self.trials if self.trials else 0.0
        raw_escape_rate = self.raw_escapes / self.trials if self.trials else 0.0
        damage_rate = self.damage_events / self.trials if self.trials else 0.0
        return {
            "adaptive_step_target": float(self.last_target),
            "adaptive_step_multiplier": float(self.multiplier),
            "adaptive_progress_boost": float(self.progress_boost),
            "adaptive_no_global_progress_trials": int(self.no_global_progress_trials),
            "adaptive_escape_rate": float(escape_rate),
            "adaptive_raw_escape_rate": float(raw_escape_rate),
            "adaptive_damage_rate": float(damage_rate),
            "adaptive_max_escape_energy_delta": float(self.max_escape_energy_delta_seen),
            "adaptive_max_escape_descriptor_delta": float(self.max_escape_descriptor_delta_seen),
            "adaptive_max_escape_novelty": float(self.max_escape_novelty_seen),
            "adaptive_damage_warning": int(self.trials >= self.feedback_warmup_trials and damage_rate > self.damage_tolerance),
        }

    def _archive_target(self, archive) -> float:
        if archive is None or len(archive.entries) < 2:
            return self.fallback_target
        energies = np.asarray([entry.energy for entry in archive.entries], dtype=float)
        best = float(energies.min())
        deltas = energies - best
        median = float(np.median(deltas))
        mad = float(np.median(np.abs(deltas - median)))
        scale = max(mad, float(np.median(deltas[deltas > 1e-12])) if np.any(deltas > 1e-12) else 0.0)
        if scale <= 1e-12:
            return self.fallback_target
        return float(np.clip(self.eta_energy_scale * scale, self.min_target, self.max_target))


class DirectionCandidateKind(str, Enum):
    MOMENTUM = "momentum"
    ANCHOR = "anchor"
    RANDOM = "random"
    BOND = "bond"
    BOND_FORM = "bond_form"
    BOND_BREAK = "bond_break"
    RITZ = "ritz"
    RITZ_REG = "ritz_reg"
    EVOLVED = "evolved"
    ARCHIVE_MOMENTUM = "archive_momentum"
    REFERENCE_DIMER = "reference_dimer"


@dataclass(frozen=True, eq=False)
class DirectionRecord:
    record_id: int
    trial_index: int | None
    proposal_index: int | None
    step_index: int
    seed_entry_id: int | None
    kind: DirectionCandidateKind
    direction: np.ndarray
    curvature: float
    score: float | None
    anchor_cosine: float | None
    accepted_new_basin: bool | None = None
    global_improved: bool | None = None
    productive: bool | None = None
    final_energy: float | None = None

    def __post_init__(self) -> None:
        self._validate_nonnegative_int("record_id", self.record_id)
        self._validate_nonnegative_int("step_index", self.step_index)
        for name in ("trial_index", "proposal_index", "seed_entry_id"):
            value = getattr(self, name)
            if value is not None:
                self._validate_nonnegative_int(name, value)
        if not isinstance(self.kind, DirectionCandidateKind):
            raise ValueError("kind must be a DirectionCandidateKind")
        direction = np.asarray(self.direction, dtype=float)
        if direction.ndim != 1 or not np.all(np.isfinite(direction)):
            raise ValueError("direction must be a finite 1D float array")
        direction = direction.copy()
        direction.setflags(write=False)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "curvature", self._validate_finite_float("curvature", self.curvature))
        for name in ("score", "anchor_cosine", "final_energy"):
            object.__setattr__(self, name, self._validate_optional_finite_float(name, getattr(self, name)))
        for name in ("accepted_new_basin", "global_improved", "productive"):
            self._validate_optional_bool(name, getattr(self, name))

    @staticmethod
    def _validate_nonnegative_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _validate_finite_float(name: str, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a finite real number")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _validate_optional_finite_float(cls, name: str, value: float | None) -> float | None:
        if value is None:
            return None
        return cls._validate_finite_float(name, value)

    @staticmethod
    def _validate_optional_bool(name: str, value: bool | None) -> None:
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean when set")


@dataclass
class DirectionTypeMemory:
    enabled: bool
    success_weight: float
    exploration_weight: float
    window: int
    selected_counts: dict[DirectionCandidateKind, int] = field(default_factory=dict)
    productive_counts: dict[DirectionCandidateKind, int] = field(default_factory=dict)
    recent_events: deque[tuple[DirectionCandidateKind, bool]] = field(default_factory=deque)

    def bonus(self, kind: DirectionCandidateKind) -> float:
        if not self.enabled:
            return 0.0
        selected = self.selected_counts.get(kind, 0)
        productive = self.productive_counts.get(kind, 0)
        total_selected = sum(self.selected_counts.values())
        success_rate = productive / max(1, selected)
        exploration = sqrt(log1p(total_selected) / max(1, selected))
        return self.success_weight * success_rate + self.exploration_weight * exploration

    def record_trial(self, kinds: Iterable[DirectionCandidateKind], productive: bool) -> None:
        for kind in sorted(set(kinds), key=lambda item: item.value):
            self.selected_counts[kind] = self.selected_counts.get(kind, 0) + 1
            if productive:
                self.productive_counts[kind] = self.productive_counts.get(kind, 0) + 1
            else:
                self.productive_counts.setdefault(kind, 0)
            self.recent_events.append((kind, bool(productive)))
            self._trim_window()

    def _trim_window(self) -> None:
        while len(self.recent_events) > self.window:
            kind, productive = self.recent_events.popleft()
            self.selected_counts[kind] = max(0, self.selected_counts.get(kind, 0) - 1)
            if productive:
                self.productive_counts[kind] = max(0, self.productive_counts.get(kind, 0) - 1)


@dataclass(frozen=True)
class DirectionCandidate:
    kind: DirectionCandidateKind
    direction: np.ndarray
    damage_risk: float = 0.0
    rigid_body_overlap: float = 0.0
    post_projection_rigid_body_overlap: float = 0.0


@dataclass(frozen=True)
class DirectionScorer:
    damage_weight: float = 1.0
    continuity_weight: float = 0.1
    anchor_weight: float = 0.5
    novelty_weight: float = 0.5
    history_push_weight: float = 0.1
    novelty_probe_scales: tuple[float, ...] = (1.0,)

    def score(
        self,
        curvature: float,
        sigma: float,
        direction: np.ndarray,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
        damage_risk: float,
        history_push: float = 0.0,
        continuity_weight: float | None = None,
    ) -> float:
        continuity = self.continuity_weight if continuity_weight is None else continuity_weight
        energy_cost = 0.5 * sigma * sigma * curvature
        discontinuity = 0.0
        if previous_direction is not None:
            prev = previous_direction / (np.linalg.norm(previous_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            discontinuity = float(np.linalg.norm(cur - prev) ** 2)
        anchor_penalty = 0.0
        if anchor_direction is not None:
            anchor = anchor_direction / (np.linalg.norm(anchor_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            anchor_penalty = float(np.linalg.norm(cur - anchor) ** 2)
        return float(
            -energy_cost
            - self.damage_weight * damage_risk
            - continuity * discontinuity
            - self.anchor_weight * anchor_penalty
            + self.history_push_weight * history_push
        )

    def score_candidate(
        self,
        state: State,
        candidate: DirectionCandidate,
        curvature: float,
        sigma: float,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
        archive,
        history_push: float = 0.0,
        continuity_weight: float | None = None,
    ) -> float:
        score = self.score(
            curvature=curvature,
            sigma=sigma,
            direction=candidate.direction,
            previous_direction=previous_direction,
            anchor_direction=anchor_direction,
            damage_risk=candidate.damage_risk,
            history_push=history_push,
            continuity_weight=continuity_weight,
        )
        if archive is None:
            return score
        coordinates = CartesianCoordinates.from_state(state)
        novelty_gain = max(
            archive.coverage_gain(
                structural_descriptor(coordinates.displace(TangentVector(candidate.direction), scale * sigma))
            )
            for scale in self.novelty_probe_scales
        )
        return float(score + self.novelty_weight * novelty_gain)


class CandidateDirectionGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        n_random: int,
        bond_pairs: list[tuple[int, int]] | None = None,
        n_bond_pairs: int = 0,
        bond_distance_threshold: float | None = None,
        enable_momentum_candidate: bool = True,
        enable_anchor_candidate: bool = False,
        random_direction_distribution: str = "unit_gaussian",
        enable_bond_form_break_split: bool = False,
        n_bond_formation_pairs: int = 2,
        n_bond_breaking_pairs: int = 1,
        bond_formation_max_distance: float = 4.0,
        bond_breaking_max_distance: float = 2.0,
    ) -> None:
        self.rng = rng
        self.n_random = n_random
        self.bond_pairs = bond_pairs or []
        self.n_bond_pairs = n_bond_pairs
        self.bond_distance_threshold = bond_distance_threshold
        self.enable_momentum_candidate = enable_momentum_candidate
        self.enable_anchor_candidate = enable_anchor_candidate
        if random_direction_distribution not in {"unit_gaussian", "mass_weighted"}:
            raise ValueError("random_direction_distribution must be unit_gaussian or mass_weighted")
        self.random_direction_distribution = random_direction_distribution
        self.enable_bond_form_break_split = enable_bond_form_break_split
        self.n_bond_formation_pairs = n_bond_formation_pairs
        self.n_bond_breaking_pairs = n_bond_breaking_pairs
        self.bond_formation_max_distance = bond_formation_max_distance
        self.bond_breaking_max_distance = bond_breaking_max_distance
        self.last_initial_bond_pair: tuple[int, int] | None = None
        self.last_random_bond_pairs_requested = 0
        self.last_random_bond_pairs_generated = 0
        self.last_fallback_bond_pairs_generated = 0
        self.last_random_bond_candidates_valid = 0

    def generate(
        self,
        state: State,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None = None,
        anchor_mixing_alpha: float | None = None,
        n_bond_pairs: int | None = None,
    ) -> list[DirectionCandidate]:
        coordinates = CartesianCoordinates.from_state(state)
        candidates: list[DirectionCandidate] = []
        if self.enable_momentum_candidate and previous_direction is not None:
            momentum_direction = self._anchor_mixed_direction(previous_direction, anchor_direction, anchor_mixing_alpha)
            candidates.append(self._candidate(state, DirectionCandidateKind.MOMENTUM, momentum_direction))
        for atom_i, atom_j in self.bond_pairs:
            direction = self._bond_direction(state, atom_i, atom_j)
            if direction is not None:
                candidates.append(self._candidate(state, DirectionCandidateKind.BOND, direction))
        dynamic_pairs_count = 0
        if self.enable_bond_form_break_split:
            n_form_pairs, n_break_pairs = self._split_bond_pair_counts(n_bond_pairs)
            formation_pairs = self._random_bond_formation_pairs(state, n_form_pairs)
            breaking_pairs = self._random_bond_breaking_pairs(state, n_break_pairs)
            self.last_random_bond_pairs_requested = n_form_pairs + n_break_pairs
            self.last_random_bond_pairs_generated = len(formation_pairs) + len(breaking_pairs)
            self.last_random_bond_candidates_valid = 0
            for atom_i, atom_j in formation_pairs:
                direction = self._bond_form_direction(state, atom_i, atom_j)
                if direction is not None:
                    candidates.append(self._candidate(state, DirectionCandidateKind.BOND_FORM, direction))
                    self.last_random_bond_candidates_valid += 1
            for atom_i, atom_j in breaking_pairs:
                direction = self._bond_break_direction(state, atom_i, atom_j)
                if direction is not None:
                    candidates.append(self._candidate(state, DirectionCandidateKind.BOND_BREAK, direction))
                    self.last_random_bond_candidates_valid += 1
            dynamic_pairs_count = len(formation_pairs) + len(breaking_pairs)
        else:
            dynamic_pairs = self._random_non_neighbor_pairs(
                state,
                n_pairs=self.n_bond_pairs if n_bond_pairs is None else n_bond_pairs,
                distance_threshold=self.bond_distance_threshold,
            )
            self.last_random_bond_pairs_requested = self.n_bond_pairs if n_bond_pairs is None else n_bond_pairs
            self.last_random_bond_pairs_generated = len(dynamic_pairs)
            self.last_random_bond_candidates_valid = 0
            for atom_i, atom_j in dynamic_pairs:
                direction = self._bond_direction(state, atom_i, atom_j)
                if direction is not None:
                    candidates.append(self._candidate(state, DirectionCandidateKind.BOND, direction))
                    self.last_random_bond_candidates_valid += 1
            dynamic_pairs_count = len(dynamic_pairs)
        # Raw anchor as an executable candidate was withdrawn after C60 smokes
        # showed strong anchor-collapse and worse minima.  Keep the config flag
        # as a compatibility no-op; anchor remains available as a prior.
        n_random = max(0, self.n_random - dynamic_pairs_count)
        for _ in range(n_random):
            active = self._random_active_direction(state, coordinates)
            active /= np.linalg.norm(active) + 1e-12
            direction = coordinates.full_tangent_from_active(active).values
            candidates.append(self._candidate(state, DirectionCandidateKind.RANDOM, direction))
        return candidates

    def generate_initial_direction(
        self,
        state: State,
        step_index: int,
        max_steps: int,
        lambda_start: float,
        lambda_end: float,
        n_bond_pairs: int,
        bond_distance_threshold: float | None,
    ) -> np.ndarray:
        coordinates = CartesianCoordinates.from_state(state)
        active = self._random_active_direction(state, coordinates)
        active /= np.linalg.norm(active) + 1e-12
        random_direction = coordinates.full_tangent_from_active(active).values
        random_direction = self._candidate(state, DirectionCandidateKind.RANDOM, random_direction).direction

        bond_direction = np.zeros_like(random_direction)
        self.last_initial_bond_pair = None
        pairs = self._random_non_neighbor_pairs(
            state,
            n_pairs=n_bond_pairs,
            distance_threshold=bond_distance_threshold,
        )
        if pairs:
            atom_i, atom_j = pairs[int(self.rng.integers(0, len(pairs)))]
            raw_bond = self._bond_direction(state, atom_i, atom_j)
            if raw_bond is not None:
                self.last_initial_bond_pair = (atom_i, atom_j)
                bond_direction = self._candidate(state, DirectionCandidateKind.BOND, raw_bond).direction

        progress = step_index / max(1, max_steps - 1)
        lambda_t = lambda_start + (lambda_end - lambda_start) * progress
        mixed = random_direction + lambda_t * bond_direction
        if np.linalg.norm(mixed) <= 1e-12:
            return random_direction
        return self._normalized(mixed)

    def _candidate(self, state: State, kind: DirectionCandidateKind, direction: np.ndarray) -> DirectionCandidate:
        raw = self._normalized(direction)
        overlap = rigid_body_overlap(state, raw)
        projected = project_out_rigid_body_modes(state, raw)
        if np.linalg.norm(projected) <= 1e-12:
            projected = raw
        projected = self._normalized(projected)
        post_overlap = rigid_body_overlap(state, projected)
        return DirectionCandidate(
            kind=kind,
            direction=projected,
            rigid_body_overlap=overlap,
            post_projection_rigid_body_overlap=post_overlap,
        )

    def _anchor_mixed_direction(
        self,
        direction: np.ndarray,
        anchor_direction: np.ndarray | None,
        alpha: float | None,
    ) -> np.ndarray:
        if anchor_direction is None or alpha is None:
            return direction
        alpha = float(np.clip(alpha, 0.0, 1.0))
        anchor = self._normalized(anchor_direction)
        base = self._normalized(direction)
        perpendicular = base - float(np.dot(base, anchor)) * anchor
        if np.linalg.norm(perpendicular) <= 1e-12:
            return anchor
        perpendicular = self._normalized(perpendicular)
        mixed = alpha * anchor + np.sqrt(max(0.0, 1.0 - alpha * alpha)) * perpendicular
        return self._normalized(mixed)

    @staticmethod
    def _normalized(direction: np.ndarray) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        return direction / (np.linalg.norm(direction) + 1e-12)

    def _random_active_direction(self, state: State, coordinates: CartesianCoordinates) -> np.ndarray:
        active = self.rng.normal(size=coordinates.active_size)
        if self.random_direction_distribution == "mass_weighted":
            masses = np.asarray([self._atomic_mass(int(number)) for number in state.numbers[state.movable_mask]])
            active = active.reshape(-1, 3) / np.sqrt(masses)[:, None]
            return active.reshape(-1)
        return active

    @staticmethod
    def _atomic_mass(number: int) -> float:
        if 0 < number < len(atomic_masses):
            mass = float(atomic_masses[number])
            if np.isfinite(mass) and mass > 0:
                return mass
        raise ValueError(f"unsupported atomic number for mass-weighted random direction: {number}")

    def _split_bond_pair_counts(self, override_total: int | None) -> tuple[int, int]:
        base_form = self.n_bond_formation_pairs
        base_break = self.n_bond_breaking_pairs
        base_total = base_form + base_break
        if override_total is None or override_total == base_total:
            return base_form, base_break
        if override_total <= 0 or base_total <= 0:
            return 0, 0
        if base_form == 0:
            return 0, override_total
        if base_break == 0:
            return override_total, 0
        n_break = int(round(override_total * base_break / base_total))
        n_break = min(max(1, n_break), override_total - 1)
        return override_total - n_break, n_break

    def _bond_direction(self, state: State, atom_i: int, atom_j: int) -> np.ndarray | None:
        return self._pair_direction(state, atom_i, atom_j, sign=-1.0)

    def _bond_form_direction(self, state: State, atom_i: int, atom_j: int) -> np.ndarray | None:
        return self._pair_direction(state, atom_i, atom_j, sign=1.0)

    def _bond_break_direction(self, state: State, atom_i: int, atom_j: int) -> np.ndarray | None:
        return self._pair_direction(state, atom_i, atom_j, sign=-1.0)

    def _pair_direction(self, state: State, atom_i: int, atom_j: int, sign: float) -> np.ndarray | None:
        if atom_i < 0 or atom_j < 0 or atom_i >= state.n_atoms or atom_j >= state.n_atoms or atom_i == atom_j:
            return None
        delta = mic_displacement(
            state.positions[atom_j : atom_j + 1],
            state.positions[atom_i : atom_i + 1],
            state.cell,
            state.pbc,
        )[0]
        norm = np.linalg.norm(delta)
        if norm <= 1e-12:
            return None
        axis = delta / norm
        values = np.zeros(state.n_atoms * 3, dtype=float).reshape(state.n_atoms, 3)
        if state.movable_mask[atom_i]:
            values[atom_i] += sign * axis
        if state.movable_mask[atom_j]:
            values[atom_j] -= sign * axis
        flat = values.reshape(-1)
        if np.linalg.norm(flat) <= 1e-12:
            return None
        return self._normalized(flat)

    def _random_bond_formation_pairs(self, state: State, n_pairs: int) -> list[tuple[int, int]]:
        threshold = self._adaptive_non_neighbor_threshold(state, self.bond_distance_threshold)
        return self._random_pairs_in_distance_window(
            state,
            n_pairs=n_pairs,
            min_distance=threshold,
            max_distance=self.bond_formation_max_distance,
        )

    def _random_bond_breaking_pairs(self, state: State, n_pairs: int) -> list[tuple[int, int]]:
        return self._random_pairs_in_distance_window(
            state,
            n_pairs=n_pairs,
            min_distance=0.0,
            max_distance=self.bond_breaking_max_distance,
        )

    def _random_pairs_in_distance_window(
        self,
        state: State,
        n_pairs: int,
        min_distance: float,
        max_distance: float,
    ) -> list[tuple[int, int]]:
        if n_pairs <= 0 or max_distance <= min_distance:
            return []
        movable_indices = np.where(state.movable_mask)[0]
        eligible: list[tuple[int, int]] = []
        for left_index, atom_i in enumerate(movable_indices):
            for atom_j in movable_indices[left_index + 1 :]:
                pair = tuple(sorted((int(atom_i), int(atom_j))))
                distance = float(
                    np.linalg.norm(
                        mic_displacement(
                            state.positions[pair[1] : pair[1] + 1],
                            state.positions[pair[0] : pair[0] + 1],
                            state.cell,
                            state.pbc,
                        )[0]
                    )
                )
                if min_distance < distance < max_distance:
                    eligible.append(pair)
        if not eligible:
            return []
        selected = self.rng.choice(len(eligible), size=min(n_pairs, len(eligible)), replace=False)
        return [eligible[int(index)] for index in np.atleast_1d(selected)]

    def _random_non_neighbor_pairs(
        self,
        state: State,
        n_pairs: int,
        distance_threshold: float | None = None,
    ) -> list[tuple[int, int]]:
        self.last_fallback_bond_pairs_generated = 0
        if n_pairs <= 0:
            return []
        movable_indices = np.where(state.movable_mask)[0]
        if len(movable_indices) < 2:
            return []
        threshold = self._adaptive_non_neighbor_threshold(state, distance_threshold)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        max_attempts = max(50, n_pairs * 50)
        while len(pairs) < n_pairs and attempts < max_attempts:
            atom_i, atom_j = self.rng.choice(movable_indices, size=2, replace=False)
            pair = tuple(sorted((int(atom_i), int(atom_j))))
            if pair not in seen:
                distance = float(
                    np.linalg.norm(
                        mic_displacement(
                            state.positions[pair[1] : pair[1] + 1],
                            state.positions[pair[0] : pair[0] + 1],
                            state.cell,
                            state.pbc,
                        )[0]
                    )
                )
                if distance > threshold:
                    seen.add(pair)
                    pairs.append(pair)
            attempts += 1
        if len(pairs) < max(1, n_pairs // 2):
            fallback_pairs = self._closest_mic_pairs(
                state,
                n_pairs=n_pairs - len(pairs),
                exclude=set(pairs),
            )
            pairs.extend(fallback_pairs)
            self.last_fallback_bond_pairs_generated = len(fallback_pairs)
        return pairs

    def _closest_mic_pairs(
        self,
        state: State,
        n_pairs: int,
        exclude: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        if n_pairs <= 0:
            return []
        exclude = exclude or set()
        movable_indices = np.where(state.movable_mask)[0]
        ranked: list[tuple[float, tuple[int, int]]] = []
        for left_index, atom_i in enumerate(movable_indices):
            for atom_j in movable_indices[left_index + 1 :]:
                pair = tuple(sorted((int(atom_i), int(atom_j))))
                if pair in exclude:
                    continue
                distance = float(
                    np.linalg.norm(
                        mic_displacement(
                            state.positions[pair[1] : pair[1] + 1],
                            state.positions[pair[0] : pair[0] + 1],
                            state.cell,
                            state.pbc,
                        )[0]
                    )
                )
                ranked.append((distance, pair))
        ranked.sort(key=lambda item: item[0])
        return [pair for _, pair in ranked[:n_pairs]]

    @staticmethod
    def _adaptive_non_neighbor_threshold(state: State, configured_threshold: float | None) -> float:
        if configured_threshold is not None:
            return configured_threshold
        if state.n_atoms < 2:
            return 0.0
        distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
        np.fill_diagonal(distances, np.inf)
        nearest = np.min(distances, axis=1)
        finite = nearest[np.isfinite(nearest)]
        if finite.size == 0:
            return 0.0
        return float(1.5 * np.median(finite))


@dataclass(frozen=True)
class CandidateProposal:
    label: str
    state: State
    allow_duplicate_rescue: bool = True
    selected_direction_kinds: frozenset[DirectionCandidateKind] = field(default_factory=frozenset)


class SoftModeOracle:
    def __init__(
        self,
        calculator,
        rng: np.random.Generator,
        candidates: int,
        bond_pairs: list[tuple[int, int]] | None = None,
        n_bond_pairs: int = 0,
        bond_distance_threshold: float | None = None,
        anchor_weight: float = 0.5,
        continuity_weight: float = 0.1,
        history_push_weight: float = 0.1,
        novelty_probe_scales: tuple[float, ...] = (1.0,),
        enable_momentum_candidate: bool = True,
        enable_anchor_candidate: bool = False,
        anchor_mixing_alpha: float | None = None,
        hvp_epsilon: float = 1e-3,
        random_direction_distribution: str = "unit_gaussian",
        enable_bond_form_break_split: bool = False,
        n_bond_formation_pairs: int = 2,
        n_bond_breaking_pairs: int = 1,
        bond_formation_max_distance: float = 4.0,
        bond_breaking_max_distance: float = 2.0,
        direction_selection_mode: str = "discrete",
        direction_synthesis_mode: str = "none",
        regularized_ritz_top_k: int = 5,
        direction_probe_enabled: bool = False,
        direction_probe_top_k: int = 5,
        direction_probe_ds_scale: float = 0.5,
        direction_probe_uphill_low: float = 0.05,
        direction_probe_uphill_high: float = 1.0,
        direction_probe_collision_distance: float = 0.5,
    ) -> None:
        self.calculator = calculator
        self.rng = rng
        self.candidates = candidates
        self.hvp_epsilon = hvp_epsilon
        self.anchor_mixing_alpha = anchor_mixing_alpha
        self.direction_selection_mode = direction_selection_mode
        self.direction_synthesis_mode = direction_synthesis_mode
        self.regularized_ritz_top_k = regularized_ritz_top_k
        self.direction_probe_enabled = direction_probe_enabled
        self.direction_probe_top_k = direction_probe_top_k
        self.direction_probe_ds_scale = direction_probe_ds_scale
        self.direction_probe_uphill_low = direction_probe_uphill_low
        self.direction_probe_uphill_high = direction_probe_uphill_high
        self.direction_probe_collision_distance = direction_probe_collision_distance
        self.generator = CandidateDirectionGenerator(
            rng,
            candidates,
            bond_pairs=bond_pairs,
            n_bond_pairs=n_bond_pairs,
            bond_distance_threshold=bond_distance_threshold,
            enable_momentum_candidate=enable_momentum_candidate,
            enable_anchor_candidate=enable_anchor_candidate,
            random_direction_distribution=random_direction_distribution,
            enable_bond_form_break_split=enable_bond_form_break_split,
            n_bond_formation_pairs=n_bond_formation_pairs,
            n_bond_breaking_pairs=n_bond_breaking_pairs,
            bond_formation_max_distance=bond_formation_max_distance,
            bond_breaking_max_distance=bond_breaking_max_distance,
        )
        self.scorer = DirectionScorer(
            anchor_weight=anchor_weight,
            continuity_weight=continuity_weight,
            history_push_weight=history_push_weight,
            novelty_probe_scales=novelty_probe_scales,
        )

    def choose_direction(
        self,
        state: State,
        proposal: ProposalPotential,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None = None,
        step_scale_fn=None,
        archive=None,
        history_gradient: np.ndarray | None = None,
        continuity_weight: float | None = None,
        n_bond_pairs: int | None = None,
        score_sigma: float | None = None,
        score_sigma_fn=None,
        direction_type_bonus_fn: Callable[[DirectionCandidateKind], float] | None = None,
        plateau_evolution_active: bool = False,
        plateau_history: list[DirectionRecord] | None = None,
        plateau_evolution_children: int = 0,
        plateau_evolution_crossover_pairs: int = 0,
        plateau_evolution_mutation_count: int = 0,
        archive_momentum_history: list[DirectionRecord] | None = None,
        archive_momentum_limit: int = 0,
        candidate_filter: Callable[[list[DirectionCandidate]], list[DirectionCandidate]] | None = None,
    ) -> DirectionChoice:
        best_direction: np.ndarray | None = None
        best_curvature: float | None = None
        best_score: float | None = None
        candidates = self.generator.generate(
            state,
            previous_direction,
            anchor_direction=anchor_direction,
            anchor_mixing_alpha=self.anchor_mixing_alpha,
            n_bond_pairs=n_bond_pairs,
        )
        archive_momentum_candidates = self._archive_momentum_candidates(
            state,
            archive_momentum_history or [],
            archive_momentum_limit,
        )
        candidates.extend(archive_momentum_candidates)
        if candidate_filter is not None:
            candidates = candidate_filter(candidates)
        scoring_anchor_direction = None if self.anchor_mixing_alpha is not None else anchor_direction
        best_kind: DirectionCandidateKind | None = None
        rigid_overlap_sum = 0.0
        post_projection_rigid_overlap_sum = 0.0
        candidate_hvps: list[np.ndarray] = []
        scored_candidates: list[tuple[DirectionCandidate, np.ndarray, float, float]] = []
        for candidate in candidates:
            rigid_overlap_sum += candidate.rigid_body_overlap
            post_projection_rigid_overlap_sum += candidate.post_projection_rigid_body_overlap
            hvp = self._directional_hvp(state, proposal, candidate.direction)
            candidate_hvps.append(hvp)
            curvature = float(np.dot(hvp, candidate.direction))
            candidate_score_sigma = self._candidate_score_sigma(
                curvature=curvature,
                score_sigma=score_sigma,
                score_sigma_fn=score_sigma_fn,
                step_scale_fn=step_scale_fn,
            )
            history_push = 0.0 if history_gradient is None else -float(np.dot(history_gradient, candidate.direction))
            score = self.scorer.score_candidate(
                state=state,
                candidate=candidate,
                curvature=curvature,
                sigma=candidate_score_sigma,
                previous_direction=previous_direction,
                anchor_direction=scoring_anchor_direction,
                archive=archive,
                history_push=history_push,
                continuity_weight=continuity_weight,
            )
            score = self._add_direction_type_bonus(score, candidate.kind, direction_type_bonus_fn)
            scored_candidates.append((candidate, hvp, curvature, score))
            if best_score is None or score > best_score:
                best_score = score
                best_curvature = curvature
                best_direction = candidate.direction
                best_kind = candidate.kind
        if self.direction_probe_enabled:
            probe_sigma = score_sigma if score_sigma is not None else (
                score_sigma_fn(1.0) if score_sigma_fn is not None else 1.0
            )
            best = self._probe_refine(
                [(candidate, curvature, score) for candidate, _, curvature, score in scored_candidates],
                state,
                best_direction,
                best_curvature,
                best_kind,
                best_score,
                probe_sigma,
            )
            if best is not None:
                best_direction, best_curvature, best_kind, best_score = best
        synthetic_count = 0
        evolved_candidates = self._plateau_evolution_candidates(
            scored_candidates,
            state=state,
            proposal=proposal,
            previous_direction=previous_direction,
            anchor_direction=scoring_anchor_direction,
            archive=archive,
            history_gradient=history_gradient,
            continuity_weight=continuity_weight,
            score_sigma=score_sigma,
            score_sigma_fn=score_sigma_fn,
            step_scale_fn=step_scale_fn,
            direction_type_bonus_fn=direction_type_bonus_fn,
            plateau_evolution_active=plateau_evolution_active,
            plateau_history=plateau_history or [],
            plateau_evolution_children=plateau_evolution_children,
            plateau_evolution_crossover_pairs=plateau_evolution_crossover_pairs,
            plateau_evolution_mutation_count=plateau_evolution_mutation_count,
        )
        synthetic_count += len(evolved_candidates)
        for evolved_candidate, _, evolved_curvature, evolved_score in evolved_candidates:
            if best_score is None or evolved_score > best_score:
                best_score = evolved_score
                best_curvature = evolved_curvature
                best_direction = evolved_candidate.direction
                best_kind = evolved_candidate.kind
        if self.direction_synthesis_mode == "regularized_ritz":
            ritz_reg = self._regularized_ritz_candidate(
                scored_candidates,
                previous_direction=previous_direction,
                anchor_direction=scoring_anchor_direction,
            )
            if ritz_reg is not None:
                ritz_reg_direction, ritz_reg_curvature = ritz_reg
                ritz_reg_candidate = DirectionCandidate(DirectionCandidateKind.RITZ_REG, ritz_reg_direction)
                ritz_reg_score_sigma = self._candidate_score_sigma(
                    curvature=ritz_reg_curvature,
                    score_sigma=score_sigma,
                    score_sigma_fn=score_sigma_fn,
                    step_scale_fn=step_scale_fn,
                )
                ritz_reg_history_push = (
                    0.0 if history_gradient is None else -float(np.dot(history_gradient, ritz_reg_direction))
                )
                ritz_reg_score = self.scorer.score_candidate(
                    state=state,
                    candidate=ritz_reg_candidate,
                    curvature=ritz_reg_curvature,
                    sigma=ritz_reg_score_sigma,
                    previous_direction=previous_direction,
                    anchor_direction=scoring_anchor_direction,
                    archive=archive,
                    history_push=ritz_reg_history_push,
                    continuity_weight=continuity_weight,
                )
                ritz_reg_score = self._add_direction_type_bonus(
                    ritz_reg_score,
                    ritz_reg_candidate.kind,
                    direction_type_bonus_fn,
                )
                synthetic_count += 1
                if best_score is None or ritz_reg_score > best_score:
                    best_score = ritz_reg_score
                    best_curvature = ritz_reg_curvature
                    best_direction = ritz_reg_direction
                    best_kind = DirectionCandidateKind.RITZ_REG
        if self.direction_selection_mode == "rayleigh_ritz":
            ritz = self._rayleigh_ritz_candidate(candidates, candidate_hvps)
            if ritz is not None:
                ritz_direction, ritz_curvature = ritz
                ritz_candidate = DirectionCandidate(DirectionCandidateKind.RITZ, ritz_direction)
                synthetic_count += 1
                ritz_score_sigma = self._candidate_score_sigma(
                    curvature=ritz_curvature,
                    score_sigma=score_sigma,
                    score_sigma_fn=score_sigma_fn,
                    step_scale_fn=step_scale_fn,
                )
                ritz_history_push = 0.0 if history_gradient is None else -float(np.dot(history_gradient, ritz_direction))
                ritz_score = self.scorer.score_candidate(
                    state=state,
                    candidate=ritz_candidate,
                    curvature=ritz_curvature,
                    sigma=ritz_score_sigma,
                    previous_direction=previous_direction,
                    anchor_direction=scoring_anchor_direction,
                    archive=archive,
                    history_push=ritz_history_push,
                    continuity_weight=continuity_weight,
                )
                ritz_score = self._add_direction_type_bonus(ritz_score, ritz_candidate.kind, direction_type_bonus_fn)
                if best_score is None or ritz_score > best_score:
                    best_score = ritz_score
                    best_curvature = ritz_curvature
                    best_direction = ritz_direction
                    best_kind = DirectionCandidateKind.RITZ
        assert best_direction is not None and best_curvature is not None and best_kind is not None

        return DirectionChoice(
            best_direction,
            best_curvature,
            best_kind,
            len(candidates) + synthetic_count,
            rigid_overlap_sum / len(candidates),
            post_projection_rigid_overlap_sum / len(candidates),
            score=best_score,
            evolved_candidate_count=len(evolved_candidates),
            archive_momentum_candidate_count=len(archive_momentum_candidates),
        )

    def _archive_momentum_candidates(
        self,
        state: State,
        history: list[DirectionRecord],
        limit: int,
    ) -> list[DirectionCandidate]:
        if limit <= 0 or not history:
            return []
        expected_shape = (state.positions.size,)
        candidates: list[DirectionCandidate] = []
        accepted_directions: list[np.ndarray] = []
        for record in history:
            direction = self._normalized_or_none(record.direction)
            if direction is None or direction.shape != expected_shape:
                continue
            if any(abs(float(np.dot(direction, seen))) > 0.999 for seen in accepted_directions):
                continue
            candidates.append(self.generator._candidate(state, DirectionCandidateKind.ARCHIVE_MOMENTUM, direction))
            accepted_directions.append(direction)
            if len(candidates) >= limit:
                break
        return candidates

    def _plateau_evolution_candidates(
        self,
        scored_candidates: list[tuple[DirectionCandidate, np.ndarray, float, float]],
        *,
        state: State,
        proposal: ProposalPotential,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
        archive,
        history_gradient: np.ndarray | None,
        continuity_weight: float | None,
        score_sigma: float | None,
        score_sigma_fn,
        step_scale_fn,
        direction_type_bonus_fn: Callable[[DirectionCandidateKind], float] | None,
        plateau_evolution_active: bool,
        plateau_history: list[DirectionRecord],
        plateau_evolution_children: int,
        plateau_evolution_crossover_pairs: int,
        plateau_evolution_mutation_count: int,
    ) -> list[tuple[DirectionCandidate, np.ndarray, float, float]]:
        if (
            not plateau_evolution_active
            or plateau_evolution_children <= 0
            or len(scored_candidates) < 2
            or not plateau_history
        ):
            return []
        selected = sorted(scored_candidates, key=lambda item: item[3], reverse=True)[
            : max(2, min(len(scored_candidates), self.regularized_ritz_top_k))
        ]
        basis = self._orthonormal_basis([item[0].direction for item in selected])
        if basis is None or basis.shape[1] < 2:
            return []

        def project(vector: np.ndarray) -> np.ndarray | None:
            normalized = self._normalized_or_none(vector)
            if normalized is None or normalized.shape != (basis.shape[0],):
                return None
            coeff = basis.T @ normalized
            norm = float(np.linalg.norm(coeff))
            if norm <= 1e-12 or not np.all(np.isfinite(coeff)):
                return None
            return coeff / norm

        current = []
        for candidate, _, _, score in selected[:3]:
            coeff = project(candidate.direction)
            if coeff is not None:
                current.append((coeff, float(score)))
        historical = []
        for record in plateau_history:
            coeff = project(record.direction)
            if coeff is None:
                continue
            record_score = record.score if record.score is not None else -float(record.curvature)
            historical.append((coeff, float(record_score)))
        if not current or not historical:
            return []

        child_coefficients: list[np.ndarray] = []
        pair_scores = sorted(
            ((score_a + score_b, coeff_a, coeff_b) for coeff_a, score_a in current for coeff_b, score_b in historical),
            key=lambda item: item[0],
            reverse=True,
        )
        for _, coeff_a, coeff_b in pair_scores[:plateau_evolution_crossover_pairs]:
            child = self._normalized_or_none(coeff_a + coeff_b)
            if child is None:
                child = self._normalized_or_none(coeff_a - coeff_b)
            if child is not None:
                child_coefficients.append(child)
            if len(child_coefficients) >= plateau_evolution_children:
                break

        parents = sorted(current + historical, key=lambda item: item[1], reverse=True)
        for parent, _ in parents[:plateau_evolution_mutation_count]:
            if len(child_coefficients) >= plateau_evolution_children:
                break
            perturbation = self.rng.normal(size=parent.shape)
            perturbation = self._normalized_or_none(perturbation)
            if perturbation is None:
                continue
            child = self._normalized_or_none(parent + 0.1 * perturbation)
            if child is not None:
                child_coefficients.append(child)

        evolved: list[tuple[DirectionCandidate, np.ndarray, float, float]] = []
        for coeff in child_coefficients[:plateau_evolution_children]:
            direction = basis @ coeff
            direction = self._normalized_or_none(direction)
            if direction is None:
                continue
            candidate = DirectionCandidate(DirectionCandidateKind.EVOLVED, direction)
            hvp = self._directional_hvp(state, proposal, candidate.direction)
            curvature = float(np.dot(hvp, candidate.direction))
            child_score_sigma = self._candidate_score_sigma(
                curvature=curvature,
                score_sigma=score_sigma,
                score_sigma_fn=score_sigma_fn,
                step_scale_fn=step_scale_fn,
            )
            history_push = 0.0 if history_gradient is None else -float(np.dot(history_gradient, candidate.direction))
            score = self.scorer.score_candidate(
                state=state,
                candidate=candidate,
                curvature=curvature,
                sigma=child_score_sigma,
                previous_direction=previous_direction,
                anchor_direction=anchor_direction,
                archive=archive,
                history_push=history_push,
                continuity_weight=continuity_weight,
            )
            score = self._add_direction_type_bonus(score, candidate.kind, direction_type_bonus_fn)
            evolved.append((candidate, hvp, curvature, score))
        return evolved

    @staticmethod
    def _orthonormal_basis(directions: list[np.ndarray]) -> np.ndarray | None:
        if len(directions) < 2:
            return None
        matrix = np.column_stack(directions)
        try:
            u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if singular_values.size == 0:
            return None
        tol = max(matrix.shape) * np.finfo(float).eps * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > tol))
        if rank < 2:
            return None
        return u[:, :rank]

    @staticmethod
    def _add_direction_type_bonus(
        score: float,
        kind: DirectionCandidateKind,
        direction_type_bonus_fn: Callable[[DirectionCandidateKind], float] | None,
    ) -> float:
        if direction_type_bonus_fn is None:
            return score
        bonus = float(direction_type_bonus_fn(kind))
        if not np.isfinite(bonus):
            raise ValueError("direction_type_bonus_fn returned a non-finite bonus")
        return float(score + bonus)

    def _candidate_score_sigma(self, curvature: float, score_sigma: float | None, score_sigma_fn, step_scale_fn) -> float:
        if score_sigma is not None:
            return float(score_sigma)
        if score_sigma_fn is not None:
            return float(score_sigma_fn(curvature))
        if step_scale_fn is not None:
            return float(step_scale_fn(1.0))
        return self._step_scale_from_curvature(1.0)

    def _probe_refine(
        self,
        scored_candidates: list[tuple[DirectionCandidate, float, float]],
        state: State,
        best_direction: np.ndarray,
        best_curvature: float,
        best_kind: DirectionCandidateKind,
        best_score: float,
        sigma: float,
    ) -> tuple[np.ndarray, float, DirectionCandidateKind, float] | None:
        k = min(self.direction_probe_top_k, len(scored_candidates))
        top_k = sorted(scored_candidates, key=lambda item: -item[2])[:k]

        probe_best_score = float("-inf")
        probe_best = None
        ds = self.direction_probe_ds_scale * sigma
        state_energy = self.calculator.evaluate(state).energy

        for candidate, curvature, base_score in top_k:
            trial_state = CartesianCoordinates.from_state(state).displace(TangentVector(candidate.direction), ds)
            try:
                probe_energy = self.calculator.evaluate(trial_state).energy
            except Exception:
                continue
            delta_e = probe_energy - state_energy
            uphill_bonus = 0.0 if self.direction_probe_uphill_low <= delta_e <= self.direction_probe_uphill_high else -1.0

            collision_penalty = 0.0
            if trial_state.n_atoms >= 2:
                dists = np.linalg.norm(
                    trial_state.positions[:, None, :] - trial_state.positions[None, :, :],
                    axis=2,
                )
                np.fill_diagonal(dists, np.inf)
                if np.min(dists) < self.direction_probe_collision_distance:
                    collision_penalty = -10.0

            probe_score = base_score + uphill_bonus + collision_penalty
            if probe_score > probe_best_score:
                probe_best_score = probe_score
                probe_best = (candidate.direction, curvature, candidate.kind, probe_score)

        if probe_best is not None and probe_best_score > best_score:
            return probe_best
        return None

    def _regularized_ritz_candidate(
        self,
        scored_candidates: list[tuple[DirectionCandidate, np.ndarray, float, float]],
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,
    ) -> tuple[np.ndarray, float] | None:
        if len(scored_candidates) < 2:
            return None
        selected = sorted(scored_candidates, key=lambda item: item[3], reverse=True)[: self.regularized_ritz_top_k]
        if len(selected) < 2:
            return None
        candidates = [item[0] for item in selected]
        hvps = [item[1] for item in selected]
        directions = np.column_stack([candidate.direction for candidate in candidates])
        h_directions = np.column_stack(hvps)
        try:
            u, singular_values, vt = np.linalg.svd(directions, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if singular_values.size == 0:
            return None
        tol = max(directions.shape) * np.finfo(float).eps * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > tol))
        if rank < 2:
            return None
        q = u[:, :rank]
        coeffs = vt[:rank, :].T / singular_values[:rank]
        hq = h_directions @ coeffs
        projected = q.T @ hq
        projected = 0.5 * (projected + projected.T)
        linear = np.zeros(rank)
        if previous_direction is not None:
            previous = self._normalized_or_none(previous_direction)
            if previous is not None and previous.shape == (q.shape[0],):
                linear += self.scorer.continuity_weight * (q.T @ previous)
        if anchor_direction is not None:
            anchor = self._normalized_or_none(anchor_direction)
            if anchor is not None and anchor.shape == (q.shape[0],):
                linear += self.scorer.anchor_weight * (q.T @ anchor)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(projected)
        except np.linalg.LinAlgError:
            return None
        if np.linalg.norm(linear) <= 1e-12:
            coeff = eigenvectors[:, int(np.argmin(eigenvalues))]
        else:
            coeff = self._regularized_ritz_coefficients(eigenvalues, eigenvectors.T @ linear)
            if coeff is None:
                return None
            coeff = eigenvectors @ coeff
        coeff_norm = float(np.linalg.norm(coeff))
        if coeff_norm <= 1e-12 or not np.all(np.isfinite(coeff)):
            return None
        coeff = coeff / coeff_norm
        direction = q @ coeff
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12 or not np.all(np.isfinite(direction)):
            return None
        direction = direction / norm
        curvature = float(coeff @ projected @ coeff)
        if not np.isfinite(curvature):
            return None
        return direction, curvature

    @staticmethod
    def _regularized_ritz_coefficients(eigenvalues: np.ndarray, linear: np.ndarray) -> np.ndarray | None:
        min_eigenvalue = float(np.min(eigenvalues))
        lower = max(0.0, -min_eigenvalue) + 1e-12

        def coefficients(mu: float) -> np.ndarray:
            return linear / (2.0 * (eigenvalues + mu))

        def norm_at(mu: float) -> float:
            return float(np.linalg.norm(coefficients(mu)))

        if norm_at(lower) < 1.0:
            coeff = coefficients(lower)
            min_indices = np.flatnonzero(np.isclose(eigenvalues, min_eigenvalue, rtol=1e-10, atol=1e-12))
            if min_indices.size == 0:
                return None
            fill = max(0.0, 1.0 - float(coeff @ coeff))
            coeff[min_indices[0]] += np.sqrt(fill)
            return coeff
        upper = max(1.0, float(np.linalg.norm(linear)), lower * 2.0)
        for _ in range(100):
            if norm_at(upper) <= 1.0:
                break
            upper *= 2.0
        else:
            return None
        for _ in range(100):
            mid = 0.5 * (lower + upper)
            if norm_at(mid) > 1.0:
                lower = mid
            else:
                upper = mid
        coeff = coefficients(upper)
        norm = float(np.linalg.norm(coeff))
        if norm <= 1e-12 or not np.all(np.isfinite(coeff)):
            return None
        return coeff / norm

    @staticmethod
    def _normalized_or_none(vector: np.ndarray) -> np.ndarray | None:
        array = np.asarray(vector, dtype=float).reshape(-1)
        norm = float(np.linalg.norm(array))
        if norm <= 1e-12 or not np.all(np.isfinite(array)):
            return None
        return array / norm

    def _rayleigh_ritz_candidate(
        self,
        candidates: list[DirectionCandidate],
        hvps: list[np.ndarray],
    ) -> tuple[np.ndarray, float] | None:
        if len(candidates) < 2 or len(candidates) != len(hvps):
            return None
        directions = np.column_stack([candidate.direction for candidate in candidates])
        h_directions = np.column_stack(hvps)
        try:
            u, singular_values, vt = np.linalg.svd(directions, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if singular_values.size == 0:
            return None
        tol = max(directions.shape) * np.finfo(float).eps * float(singular_values[0])
        rank = int(np.count_nonzero(singular_values > tol))
        if rank < 2:
            return None
        q = u[:, :rank]
        coeffs = vt[:rank, :].T / singular_values[:rank]
        hq = h_directions @ coeffs
        projected = q.T @ hq
        projected = 0.5 * (projected + projected.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(projected)
        except np.linalg.LinAlgError:
            return None
        direction = q @ eigenvectors[:, int(np.argmin(eigenvalues))]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            return None
        direction = direction / norm
        curvature = float(eigenvalues[int(np.argmin(eigenvalues))])
        return direction, curvature

    def _directional_curvature(
        self,
        state: State,
        proposal: ProposalPotential,
        direction: np.ndarray,
        epsilon: float | None = None,
    ) -> float:
        hvp = self._directional_hvp(state, proposal, direction, epsilon=epsilon)
        return float(np.dot(hvp, direction))

    def _directional_hvp(
        self,
        state: State,
        proposal: ProposalPotential,
        direction: np.ndarray,
        epsilon: float | None = None,
    ) -> np.ndarray:
        epsilon = self.hvp_epsilon if epsilon is None else epsilon
        coordinates = CartesianCoordinates.from_state(state)
        tangent = TangentVector(direction)
        plus = coordinates.displace(tangent, epsilon)
        minus = coordinates.displace(tangent, -epsilon)
        _, grad_plus = proposal.evaluate(plus.flatten_positions(), plus)
        _, grad_minus = proposal.evaluate(minus.flatten_positions(), minus)
        return (grad_plus - grad_minus) / (2.0 * epsilon)

    @staticmethod
    def _step_scale_from_curvature(curvature: float) -> float:
        effective = max(abs(curvature), 1e-4)
        return float(np.sqrt(2.0 / effective))


class SurfaceWalker:
    def __init__(self, calculator, config: SSWConfig, softening_enabled: bool) -> None:
        self.calculator = EvalCounter(calculator, max_force_evals=config.max_force_evals)
        self.config = config
        self.softening_enabled = softening_enabled
        self.rng = np.random.default_rng(config.rng_seed)
        bond_pairs = config.local_softening_pairs if softening_enabled and isinstance(config, LSSSWConfig) else []
        self.oracle = SoftModeOracle(
            calculator,
            self.rng,
            config.oracle_candidates,
            bond_pairs=bond_pairs,
            n_bond_pairs=config.n_bond_pairs,
            bond_distance_threshold=config.bond_distance_threshold,
            anchor_weight=config.anchor_weight,
            continuity_weight=config.continuity_weight,
            history_push_weight=config.history_push_weight,
            novelty_probe_scales=tuple(config.novelty_probe_scales),
            enable_momentum_candidate=config.enable_momentum_candidate,
            enable_anchor_candidate=config.enable_anchor_candidate,
            anchor_mixing_alpha=config.anchor_mixing_alpha,
            hvp_epsilon=config.hvp_epsilon,
            random_direction_distribution=config.random_direction_distribution,
            enable_bond_form_break_split=config.enable_bond_form_break_split,
            n_bond_formation_pairs=config.n_bond_formation_pairs,
            n_bond_breaking_pairs=config.n_bond_breaking_pairs,
            bond_formation_max_distance=config.bond_formation_max_distance,
            bond_breaking_max_distance=config.bond_breaking_max_distance,
            direction_selection_mode=config.direction_selection_mode,
            direction_synthesis_mode=config.direction_synthesis_mode,
            regularized_ritz_top_k=config.regularized_ritz_top_k,
            direction_probe_enabled=config.direction_probe_enabled,
            direction_probe_top_k=config.direction_probe_top_k,
            direction_probe_ds_scale=config.direction_probe_ds_scale,
            direction_probe_uphill_low=config.direction_probe_uphill_low,
            direction_probe_uphill_high=config.direction_probe_uphill_high,
            direction_probe_collision_distance=config.direction_probe_collision_distance,
        )
        self.proposal_scorer = ProposalScorer.for_mode(config.search_mode)
        self.selector = BanditSelector(
            policy=AcquisitionPolicy(
                archive_density_weight=config.archive_density_weight,
                novelty_weight=config.novelty_weight,
                frontier_weight=config.frontier_weight,
                exploration_weight=config.bandit_exploration_weight,
                baseline_probability=config.baseline_selection_probability,
                beta_energy=config.bandit_energy_weight,
            )
        )
        self.trust_controller = TrustRegionBiasController(
            step_length=StepLengthController(
                error_tolerance=config.step_error_tolerance,
                gamma_down=config.step_gamma_down,
                gamma_up=config.step_gamma_up,
            )
        )
        self.step_target_controller = StepTargetController(
            config.target_uphill_energy,
            min_escape_energy_delta=config.min_escape_energy_delta,
            min_escape_descriptor_delta=config.min_escape_descriptor_delta,
            min_escape_novelty=config.min_escape_novelty,
            progress_patience=config.trial_progress_patience,
            progress_boost_factor=config.trial_progress_boost_factor,
            progress_max_boost=config.trial_progress_max_boost,
            progress_duplicate_tolerance=config.trial_progress_duplicate_tolerance,
        )
        self.geometry_validator = GeometryValidator()
        self._missing_trajectory_context_warned = False
        self._reset_trust_stats()
        self._reset_direction_stats()
        self._reset_relax_stats()
        self._reset_bias_stats()
        self._reset_direct_qp_stats()
        self._reset_local_softening_stats()
        self._reset_seed_diversity_stats()
        self._reset_reference_dimer_stats()
        self._proposal_optimizer_alt_steps = 0
        self._proposal_duplicate_rescue_attempts = 0
        self._proposal_duplicate_rescue_successes = 0
        self._energy_sanity_rejections = 0
        self.direction_type_memory = self._new_direction_type_memory()
        self._reset_direction_archive_records()
        self._reset_metropolis_stats()

    def _should_rebuild_softening_for_choice(
        self,
        anchor_direction: np.ndarray | None,
        choice_direction: np.ndarray | None,
    ) -> bool:
        if not getattr(self.config, "choice_aligned_softening_enabled", False):
            return False
        if not self.softening_enabled:
            return False
        if anchor_direction is None or choice_direction is None:
            return False
        anchor = np.asarray(anchor_direction, dtype=float).reshape(-1)
        chosen = np.asarray(choice_direction, dtype=float).reshape(-1)
        if anchor.shape != chosen.shape or anchor.size == 0:
            return False
        if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(chosen)):
            return False
        eps = 1e-12
        anchor_norm = float(np.linalg.norm(anchor))
        chosen_norm = float(np.linalg.norm(chosen))
        if anchor_norm < eps or chosen_norm < eps:
            return False
        cosine = float(np.dot(anchor, chosen) / (anchor_norm * chosen_norm))
        threshold = float(getattr(self.config, "choice_aligned_softening_cos_threshold", 0.3))
        return bool(cosine < threshold)

    def _reset_direction_archive_pending_records(self) -> None:
        self._direction_archive_pending_records = [] if self._direction_archive_storage_enabled() else None

    def _reset_direction_archive_records(self) -> None:
        self._direction_archive_records = [] if self._direction_archive_storage_enabled() else None
        self._direction_archive_next_record_id = 0
        self._direction_archive_written_record_ids = set() if self.config.direction_archive_enabled else None
        self._reset_direction_archive_pending_records()

    def _direction_archive_storage_enabled(self) -> bool:
        return bool(
            self.config.direction_archive_enabled
            or self.config.plateau_evolution_enabled
            or self.config.archive_escape_momentum_enabled
        )

    def _archive_momentum_history_for_seed(self, seed_entry_id: int | None) -> list[DirectionRecord]:
        if not self.config.archive_escape_momentum_enabled:
            return []
        limit = self.config.archive_escape_momentum_history_limit
        if not self.config.archive_escape_momentum_same_seed_first:
            return self.successful_records(limit=limit)
        records = self.successful_records(seed_entry_id=seed_entry_id, limit=limit)
        seen = {record.record_id for record in records}
        if len(records) >= limit:
            return records
        for record in self.successful_records(limit=limit):
            if record.record_id in seen:
                continue
            records.append(record)
            seen.add(record.record_id)
            if len(records) >= limit:
                break
        return records

    def successful_records(
        self,
        seed_entry_id: int | None = None,
        kind: DirectionCandidateKind | None = None,
        limit: int = 16,
    ) -> list[DirectionRecord]:
        self._validate_direction_archive_query(kind=kind, limit=limit)
        records = self._direction_archive_records
        if records is None:
            return []
        matches = (
            record
            for record in reversed(records)
            if record.productive is True
            and (seed_entry_id is None or record.seed_entry_id == seed_entry_id)
            and (kind is None or record.kind is kind)
        )
        return self._normalized_direction_record_copies(matches, limit)

    def recent_records(
        self,
        kind: DirectionCandidateKind | None = None,
        limit: int = 16,
    ) -> list[DirectionRecord]:
        self._validate_direction_archive_query(kind=kind, limit=limit)
        records = self._direction_archive_records
        if records is None:
            return []
        matches = (record for record in reversed(records) if kind is None or record.kind is kind)
        return self._normalized_direction_record_copies(matches, limit)

    @staticmethod
    def _validate_direction_archive_query(kind: DirectionCandidateKind | None, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if kind is not None and not isinstance(kind, DirectionCandidateKind):
            raise ValueError("kind must be a DirectionCandidateKind when provided")

    @classmethod
    def _normalized_direction_record_copies(
        cls,
        records: Iterable[DirectionRecord],
        limit: int,
    ) -> list[DirectionRecord]:
        copies: list[DirectionRecord] = []
        for record in records:
            copies.append(cls._normalized_direction_record_copy(record))
            if len(copies) >= limit:
                break
        return copies

    @staticmethod
    def _normalized_direction_record_copy(record: DirectionRecord) -> DirectionRecord:
        direction = np.asarray(record.direction, dtype=float).copy()
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("stored direction vector must be nonzero and finite")
        return replace(record, direction=direction / norm)

    @staticmethod
    def _direction_anchor_cosine(
        anchor_direction: np.ndarray | None,
        choice_direction: np.ndarray | None,
    ) -> float | None:
        if anchor_direction is None or choice_direction is None:
            return None
        anchor = np.asarray(anchor_direction, dtype=float).reshape(-1)
        chosen = np.asarray(choice_direction, dtype=float).reshape(-1)
        if anchor.shape != chosen.shape or anchor.size == 0:
            return None
        if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(chosen)):
            return None
        anchor_norm = float(np.linalg.norm(anchor))
        chosen_norm = float(np.linalg.norm(chosen))
        if anchor_norm < 1e-12 or chosen_norm < 1e-12:
            return None
        return float(np.dot(anchor, chosen) / (anchor_norm * chosen_norm))

    def _capture_direction_record(
        self,
        *,
        trial_index: int | None,
        proposal_index: int | None,
        step_index: int,
        seed_entry_id: int | None,
        choice: DirectionChoice,
        anchor_direction: np.ndarray | None,
    ) -> None:
        pending_records = self._direction_archive_pending_records
        if pending_records is None:
            return
        score = getattr(choice, "score", None)
        pending_records.append(
            DirectionRecord(
                record_id=self._direction_archive_next_record_id,
                trial_index=trial_index,
                proposal_index=proposal_index,
                step_index=step_index,
                seed_entry_id=seed_entry_id,
                kind=choice.kind,
                direction=choice.direction,
                curvature=choice.curvature,
                score=score,
                anchor_cosine=self._direction_anchor_cosine(anchor_direction, choice.direction),
            )
        )
        self._direction_archive_next_record_id += 1

    def _finalize_direction_archive_trial(
        self,
        trial_index: int,
        *,
        accepted_new_basin: bool,
        global_improved: bool,
        final_energy: float | None,
        proposal_index: int | None = None,
    ) -> None:
        pending_records = self._direction_archive_pending_records
        finalized_records = self._direction_archive_records
        if pending_records is None or finalized_records is None:
            return
        productive = bool(accepted_new_basin or global_improved)
        remaining = []
        for record in pending_records:
            if record.trial_index != trial_index:
                remaining.append(record)
                continue
            if proposal_index is not None and record.proposal_index != proposal_index:
                remaining.append(record)
                continue
            finalized = replace(
                record,
                accepted_new_basin=bool(accepted_new_basin),
                global_improved=bool(global_improved),
                productive=productive,
                final_energy=final_energy,
            )
            if not self.config.direction_archive_success_only or finalized.productive:
                finalized_records.append(finalized)
                self._write_direction_archive_record(finalized)
        pending_records[:] = remaining
        max_records = self.config.direction_archive_max_records
        overflow = len(finalized_records) - max_records
        if overflow > 0:
            del finalized_records[:overflow]

    def _discard_direction_archive_trial(self, trial_index: int) -> None:
        pending_records = self._direction_archive_pending_records
        if pending_records is None:
            return
        pending_records[:] = [record for record in pending_records if record.trial_index != trial_index]

    def _direction_archive_output_path(self) -> Path | None:
        if not self.config.direction_archive_enabled:
            return None
        path = self.config.direction_archive_path
        return Path(path) if path is not None else None

    def _reset_direction_archive_output(self) -> None:
        path = self._direction_archive_output_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def _write_direction_archive_record(self, record: DirectionRecord) -> None:
        path = self._direction_archive_output_path()
        written_record_ids = self._direction_archive_written_record_ids
        if path is None or written_record_ids is None:
            return
        if record.record_id in written_record_ids:
            return
        payload = {
            "record_id": int(record.record_id),
            "trial_index": None if record.trial_index is None else int(record.trial_index),
            "proposal_index": None if record.proposal_index is None else int(record.proposal_index),
            "step_index": int(record.step_index),
            "seed_entry_id": None if record.seed_entry_id is None else int(record.seed_entry_id),
            "kind": record.kind.value,
            "curvature": float(record.curvature),
            "score": None if record.score is None else float(record.score),
            "anchor_cosine": None if record.anchor_cosine is None else float(record.anchor_cosine),
            "accepted_new_basin": record.accepted_new_basin,
            "global_improved": record.global_improved,
            "productive": record.productive,
            "final_energy": None if record.final_energy is None else float(record.final_energy),
            "direction": [float(value) for value in record.direction],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        written_record_ids.add(record.record_id)

    def _direction_archive_stats_summary(self) -> dict[str, int]:
        records = self._direction_archive_records
        if records is None:
            return {
                "direction_archive_enabled": 0,
                "direction_archive_active": 0,
                "direction_archive_records": 0,
                "direction_archive_productive_records": 0,
            }
        return {
            "direction_archive_enabled": int(self.config.direction_archive_enabled),
            "direction_archive_active": 1,
            "direction_archive_records": len(records),
            "direction_archive_productive_records": sum(1 for record in records if record.productive is True),
        }

    def relax_true_minimum(self, state: State, trajectory_name: str | None = None) -> RelaxResult:
        if not self.geometry_validator.is_valid_state(state):
            raise BudgetExceeded("invalid geometry before true relaxation")
        relaxer = Relaxer(self.calculator.evaluate_flat, optimizer=self.config.quench_optimizer)
        relax_config = RelaxConfig(fmax=self.config.quench_fmax, maxiter=self.config.quench_maxiter)
        result = relaxer.relax(
            state,
            fmax=relax_config.fmax,
            maxiter=relax_config.maxiter,
            trajectory_callback=self._relaxation_trajectory_callback(trajectory_name),
            trajectory_stride=self.config.relaxation_trajectory_stride,
        )
        if not self.geometry_validator.is_valid_evaluation(result.state, self.calculator):
            raise BudgetExceeded("invalid geometry after true relaxation")
        self._record_relax_result("true_quench", result, relax_config.fmax)
        return result

    def run(self, initial_state: State):
        from .archive import MinimaArchive

        self._reset_trust_stats()
        self._reset_direction_stats()
        self._reset_relax_stats()
        self._reset_bias_stats()
        self._reset_direct_qp_stats()
        self._reset_local_softening_stats()
        self._reset_seed_diversity_stats()
        self._reset_reference_dimer_stats()
        self._proposal_optimizer_alt_steps = 0
        self._proposal_duplicate_rescue_attempts = 0
        self._proposal_duplicate_rescue_successes = 0
        self._energy_sanity_rejections = 0
        self._reset_metropolis_stats()
        self._reset_accepted_structure_log()
        self._reset_direction_diagnostics()
        self.direction_type_memory = self._new_direction_type_memory()
        self._reset_direction_archive_records()
        self._reset_direction_archive_output()
        self._prepare_structure_output_dirs()
        initial = self.relax_true_minimum(initial_state, trajectory_name="initial_true_quench")
        archive = MinimaArchive(
            energy_tol=self.config.dedup_energy_tol,
            rmsd_tol=self.config.dedup_rmsd_tol,
            max_prototypes=self.config.max_prototypes,
        )
        best_entry = archive.add(initial.state, initial.energy, parent_id=None)
        metropolis_entry = best_entry
        walk_history: list[WalkRecord] = []
        local_relaxations = 1

        completed_trials = 0
        budget_exhausted = False
        trials_since_last_improvement = 0
        _t0 = __import__("time").time()
        _p = lambda msg: print(f"[ssw] {msg}", flush=True)
        _p(f"starting {self.config.max_trials} trials...")
        for trial_index in range(self.config.max_trials):
            if self.calculator.exhausted():
                budget_exhausted = True
                break
            step_target = self.step_target_controller.target(archive)
            damage_events_before = self._trust_damage_events
            if self.config.seed_selection_mode == "metropolis_chain":
                seed_entry = self._select_metropolis_seed_entry(metropolis_entry)
            else:
                seed_entry = self._select_seed_entry(archive)
            plateau_evolution_active = bool(
                self.config.plateau_evolution_enabled
                and trials_since_last_improvement >= self.config.plateau_patience_trials
            )
            try:
                proposals = self._proposal_pool(
                    seed_entry.state,
                    archive,
                    trial_index,
                    step_target,
                    seed_entry_id=seed_entry.entry_id,
                    plateau_evolution_active=plateau_evolution_active,
                )
            except BudgetExceeded:
                budget_exhausted = True
                self._discard_direction_archive_trial(trial_index)
                break
            best_discovered = None
            best_discovered_is_new = False
            best_rank_key: tuple[float, ...] | None = None
            best_reward = 0.0
            any_new = False
            max_escape_energy_delta = 0.0
            max_escape_descriptor_delta = 0.0
            max_escape_novelty = 0.0
            duplicate_failures = 0
            previous_best_energy = best_entry.energy
            trial_direction_kinds: set[DirectionCandidateKind] = set()
            for proposal in proposals:
                trial_direction_kinds.update(proposal.selected_direction_kinds)
            proposal_index = 0
            while proposal_index < len(proposals):
                proposal = proposals[proposal_index]
                try:
                    candidate = self.relax_true_minimum(
                        proposal.state,
                        trajectory_name=(
                            f"trial{trial_index + 1:04d}_proposal{proposal_index + 1:03d}_true_quench"
                        ),
                    )
                except BudgetExceeded:
                    budget_exhausted = True
                    self._discard_direction_archive_trial(trial_index)
                    break
                local_relaxations += 1
                if self._is_fragmented_cluster(seed_entry.state, candidate.state):
                    self._write_proposal_minimum(
                        trial_index=trial_index + 1,
                        proposal_index=proposal_index + 1,
                        state=candidate.state,
                        energy=candidate.energy,
                        seed_entry_id=seed_entry.entry_id,
                        status="fragment_rejected",
                    )
                    self._fragment_rejections += 1
                    duplicate_failures += 1
                    self._finalize_direction_archive_trial(
                        trial_index,
                        proposal_index=proposal_index,
                        accepted_new_basin=False,
                        global_improved=False,
                        final_energy=candidate.energy,
                    )
                    proposal_index += 1
                    continue
                if self._is_unphysical_energy_drop(candidate.energy, best_entry.energy, candidate.state.n_atoms):
                    self._write_proposal_minimum(
                        trial_index=trial_index + 1,
                        proposal_index=proposal_index + 1,
                        state=candidate.state,
                        energy=candidate.energy,
                        seed_entry_id=seed_entry.entry_id,
                        status="energy_sanity_rejected",
                    )
                    self._energy_sanity_rejections += 1
                    duplicate_failures += 1
                    self._finalize_direction_archive_trial(
                        trial_index,
                        proposal_index=proposal_index,
                        accepted_new_basin=False,
                        global_improved=False,
                        final_energy=candidate.energy,
                    )
                    proposal_index += 1
                    continue
                descriptor = structural_descriptor(candidate.state)
                coverage_gain = archive.coverage_gain(descriptor)
                before_count = len(archive.entries)
                discovered = archive.add(candidate.state, candidate.energy, parent_id=seed_entry.entry_id)
                is_new = len(archive.entries) > before_count
                is_duplicate = not is_new
                duplicate_failures += int(is_duplicate)
                any_new = any_new or is_new
                if is_new:
                    max_escape_energy_delta = max(
                        max_escape_energy_delta,
                        abs(float(candidate.energy) - float(seed_entry.energy)),
                    )
                    if seed_entry.descriptor is not None:
                        max_escape_descriptor_delta = max(
                            max_escape_descriptor_delta,
                            descriptor_distance(descriptor, seed_entry.descriptor),
                        )
                    max_escape_novelty = max(max_escape_novelty, float(coverage_gain))
                self._write_proposal_minimum(
                    trial_index=trial_index + 1,
                    proposal_index=proposal_index + 1,
                    state=candidate.state,
                    energy=candidate.energy,
                    seed_entry_id=seed_entry.entry_id,
                    discovered_entry_id=discovered.entry_id,
                    status="accepted" if is_new else "duplicate",
                )
                outcome = ProposalOutcome(
                    energy=candidate.energy,
                    previous_best_energy=previous_best_energy,
                    is_new_minimum=is_new,
                    is_duplicate=is_duplicate,
                    descriptor_coverage_gain=coverage_gain,
                )
                reward = self.proposal_scorer.score(outcome)
                rank_key = self.proposal_scorer.rank_key(outcome)
                proposal_global_improved = discovered.energy < best_entry.energy - 1e-12
                self._finalize_direction_archive_trial(
                    trial_index,
                    proposal_index=proposal_index,
                    accepted_new_basin=is_new,
                    global_improved=proposal_global_improved,
                    final_energy=discovered.energy,
                )
                if discovered.energy < best_entry.energy:
                    best_entry = discovered
                if is_new:
                    self._record_accepted_structure(
                        trial_index=trial_index + 1,
                        seed_entry_id=seed_entry.entry_id,
                        discovered_entry_id=discovered.entry_id,
                        state=discovered.state,
                        energy=discovered.energy,
                        best_energy=best_entry.energy,
                    )
                if best_rank_key is None or rank_key > best_rank_key:
                    best_rank_key = rank_key
                    best_reward = reward
                    best_discovered = discovered
                    best_discovered_is_new = is_new
                if (
                    is_duplicate
                    and proposal.allow_duplicate_rescue
                    and self.config.proposal_duplicate_rescue_optimizer is not None
                    and not self.calculator.exhausted()
                ):
                    self._proposal_duplicate_rescue_attempts += 1
                    rescue_direction_kinds: set[DirectionCandidateKind] = set()
                    try:
                        rescue_state = self._walk_candidate_from_seed(
                            seed_entry.state,
                            archive,
                            step_target,
                            trial_index=trial_index,
                            proposal_index=len(proposals),
                            seed_entry_id=seed_entry.entry_id,
                            proposal_optimizer_override=self.config.proposal_duplicate_rescue_optimizer,
                            selected_direction_kinds=rescue_direction_kinds,
                            plateau_evolution_active=plateau_evolution_active,
                        )
                    except BudgetExceeded:
                        budget_exhausted = True
                        self._discard_direction_archive_trial(trial_index)
                        break
                    trial_direction_kinds.update(rescue_direction_kinds)
                    proposals.append(
                        CandidateProposal(
                            "duplicate_rescue",
                            rescue_state,
                            allow_duplicate_rescue=False,
                            selected_direction_kinds=frozenset(rescue_direction_kinds),
                        )
                    )
                if proposal.label == "duplicate_rescue" and is_new:
                    self._proposal_duplicate_rescue_successes += 1
                proposal_index += 1
            if best_discovered is None:
                if budget_exhausted:
                    break
                trial_duplicate_rate = min(1.0, duplicate_failures / max(1, len(proposals)))
                self.step_target_controller.record_trial(
                    escaped=False,
                    damaged=self._trust_damage_events > damage_events_before,
                    global_improved=False,
                    duplicate_rate=trial_duplicate_rate,
                )
                archive.record_success(seed_entry, 0.0, duplicate_failures=max(1, duplicate_failures))
                if self.config.seed_selection_mode == "metropolis_chain":
                    self._metropolis_rejects += 1
                self.direction_type_memory.record_trial(trial_direction_kinds, productive=False)
                self._finalize_direction_archive_trial(
                    trial_index,
                    accepted_new_basin=False,
                    global_improved=False,
                    final_energy=None,
                )
                trials_since_last_improvement += 1
                completed_trials += 1
                _el = __import__("time").time() - _t0
                _p(f"trial {completed_trials}/{self.config.max_trials}  best={best_entry.energy:.3f} eV  minima={len(archive.entries)}  elapsed={_el:.0f}s")
                continue
            trial_duplicate_rate = min(1.0, duplicate_failures / max(1, len(proposals)))
            global_improved = best_entry.energy < previous_best_energy - 1e-12
            self.step_target_controller.record_trial(
                escaped=any_new,
                damaged=self._trust_damage_events > damage_events_before,
                seed_energy=seed_entry.energy,
                new_energy=best_discovered.energy,
                energy_delta=max_escape_energy_delta,
                descriptor_delta=max_escape_descriptor_delta,
                novelty_gain=max_escape_novelty,
                global_improved=global_improved,
                duplicate_rate=trial_duplicate_rate,
            )
            self.direction_type_memory.record_trial(trial_direction_kinds, productive=any_new or global_improved)
            self._discard_direction_archive_trial(trial_index)
            archive.record_success(seed_entry, best_reward, duplicate_failures=duplicate_failures)
            if global_improved:
                trials_since_last_improvement = 0
            else:
                trials_since_last_improvement += 1
            if self.config.seed_selection_mode == "metropolis_chain":
                metropolis_entry = self._update_metropolis_chain(
                    current_entry=metropolis_entry,
                    candidate_entry=best_discovered,
                    is_new=best_discovered_is_new,
                )
            walk_history.append(
                WalkRecord(
                    seed_entry_id=seed_entry.entry_id,
                    discovered_entry_id=best_discovered.entry_id,
                    energy=best_discovered.energy,
                    accepted_new_basin=best_discovered.entry_id != seed_entry.entry_id,
                )
            )
            completed_trials += 1
            _el = __import__("time").time() - _t0
            _p(f"trial {completed_trials}/{self.config.max_trials}  best={best_entry.energy:.3f} eV  minima={len(archive.entries)}  elapsed={_el:.0f}s")

        _el = __import__("time").time() - _t0
        _p(f"done: {completed_trials} trials, {len(archive.entries)} minima, best={best_entry.energy:.3f} eV, elapsed={_el:.0f}s")
        archive.refresh_frontier_status()
        prototype_stats = archive.prototype_occupancy()
        frontier_stats = archive.frontier_diagnostics()
        return SearchResult(
            best_state=best_entry.state,
            best_energy=best_entry.energy,
            archive=archive,
            walk_history=walk_history,
            stats={
                "n_trials": completed_trials,
                "configured_max_trials": self.config.max_trials,
                "n_minima": len(archive.entries),
                "local_relaxations": local_relaxations,
                "force_evaluations": self.calculator.force_evaluations,
                "energy_evaluations": self.calculator.energy_evaluations,
                "max_force_evals": self.config.max_force_evals if self.config.max_force_evals is not None else 0,
                "budget_exhausted": int(budget_exhausted or self.calculator.exhausted()),
                "duplicate_rate": archive.duplicate_rate(),
                "descriptor_degeneracy_rate": archive.descriptor_degeneracy_rate(),
                "archive_prototypes": prototype_stats["n_prototypes"],
                "archive_max_prototypes": prototype_stats["max_prototypes"],
                "archive_max_prototype_weight": prototype_stats["max_prototype_weight"],
                "archive_mean_prototype_weight": prototype_stats["mean_prototype_weight"],
                "frontier_nodes": frontier_stats["frontier_nodes"],
                "dead_nodes": frontier_stats["dead_nodes"],
                "mean_frontier_score": frontier_stats["mean_frontier_score"],
                "mean_node_duplicate_failure_rate": frontier_stats["mean_node_duplicate_failure_rate"],
                "max_node_duplicate_failure_rate": frontier_stats["max_node_duplicate_failure_rate"],
                "coordinate_system": "cartesian_fixed_cell",
                "variable_cell_supported": 0,
                "quench_optimizer": self.config.quench_optimizer,
                "proposal_optimizer": self.config.proposal_optimizer,
                "proposal_optimizer_alt": self.config.proposal_optimizer_alt,
                "proposal_optimizer_alt_steps": self._proposal_optimizer_alt_steps,
                "proposal_duplicate_rescue_optimizer": self.config.proposal_duplicate_rescue_optimizer,
                "proposal_duplicate_rescue_attempts": self._proposal_duplicate_rescue_attempts,
                "proposal_duplicate_rescue_successes": self._proposal_duplicate_rescue_successes,
                "energy_sanity_rejections": self._energy_sanity_rejections,
                **self._metropolis_stats_summary(metropolis_entry),
                "local_softening_terms_last": self._local_softening_terms_last,
                "local_softening_terms_total": self._local_softening_terms_built_total,
                "local_softening_builds": self._local_softening_builds,
                "local_softening_terms_built_total": self._local_softening_terms_built_total,
                **self._trust_stats_summary(),
                **self._direction_stats_summary(),
                **self._direction_archive_stats_summary(),
                **self._reference_dimer_stats_summary(),
                **self._relax_stats_summary(),
                **self._direct_qp_stats_summary(),
                **self.step_target_controller.stats(),
            },
        )

    def _proposal_pool(
        self,
        seed_state: State,
        archive,
        trial_index: int,
        step_target: float | None = None,
        seed_entry_id: int | None = None,
        proposal_optimizer_override: str | None = None,
        label: str = "ssw_walk",
        allow_duplicate_rescue: bool = True,
        plateau_evolution_active: bool = False,
    ) -> list[CandidateProposal]:
        proposals: list[CandidateProposal] = []
        for proposal_index in range(self.config.proposal_pool_size):
            selected_direction_kinds: set[DirectionCandidateKind] = set()
            state = self._walk_candidate_from_seed(
                seed_state,
                archive,
                step_target,
                trial_index=trial_index,
                proposal_index=proposal_index,
                seed_entry_id=seed_entry_id,
                proposal_optimizer_override=proposal_optimizer_override,
                selected_direction_kinds=selected_direction_kinds,
                plateau_evolution_active=plateau_evolution_active,
            )
            proposals.append(
                CandidateProposal(
                    label,
                    state,
                    allow_duplicate_rescue=allow_duplicate_rescue,
                    selected_direction_kinds=frozenset(selected_direction_kinds),
                )
            )
        return proposals

    def _is_unphysical_energy_drop(self, energy: float, reference_energy: float, n_atoms: int) -> bool:
        limit = self.config.max_energy_drop_per_atom
        if limit is None:
            return False
        if n_atoms <= 0:
            return False
        if not np.isfinite(energy) or not np.isfinite(reference_energy):
            return True
        return bool((float(reference_energy) - float(energy)) > float(limit) * float(n_atoms))

    def _choose_walk_direction(
        self,
        *,
        current: State,
        proposal: ProposalPotential,
        scoring_proposal: ProposalPotential,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray,
        archive,
        step_target: float | None,
        sigma_scale: float,
        previous_relax_outcome: RelaxOutcomeClass | None,
        trial_index: int | None,
        proposal_index: int | None,
        seed_entry_id: int | None,
        step_index: int,
        plateau_evolution_active: bool,
        reference_dimer_initial_direction: np.ndarray | None = None,
        reference_dimer_initial_info: dict[str, object] | None = None,
    ) -> DirectionChoice:
        if self.config.direction_engine == "reference_dimer":
            return self._choose_reference_dimer_direction(
                current,
                initial_direction=reference_dimer_initial_direction,
                initial_info=reference_dimer_initial_info,
            )

        score_sigma_fn = self._direction_score_sigma_fn(sigma_scale, step_target=step_target)
        return self.oracle.choose_direction(
            current,
            scoring_proposal,
            previous_direction,
            anchor_direction=anchor_direction,
            step_scale_fn=lambda curvature: self._scaled_step_scale(
                curvature,
                sigma_scale,
                step_target=step_target,
            ),
            archive=archive,
            history_gradient=self._history_bias_gradient(current, proposal.biases),
            continuity_weight=self._continuity_weight_for_outcome(previous_relax_outcome),
            n_bond_pairs=self._n_bond_pairs_for_outcome(previous_relax_outcome),
            score_sigma=(
                None
                if score_sigma_fn is not None
                else self._direction_score_sigma(sigma_scale, step_target=step_target)
            ),
            score_sigma_fn=score_sigma_fn,
            direction_type_bonus_fn=(
                self.direction_type_memory.bonus if self.config.direction_type_ucb_enabled else None
            ),
            plateau_evolution_active=plateau_evolution_active,
            plateau_history=(
                self.successful_records(
                    seed_entry_id=seed_entry_id,
                    limit=self.config.plateau_evolution_history_limit,
                )
                if plateau_evolution_active
                else []
            ),
            plateau_evolution_children=self.config.plateau_evolution_children,
            plateau_evolution_crossover_pairs=self.config.plateau_evolution_crossover_pairs,
            plateau_evolution_mutation_count=self.config.plateau_evolution_mutation_count,
            archive_momentum_history=self._archive_momentum_history_for_seed(seed_entry_id),
            archive_momentum_limit=self.config.archive_escape_momentum_limit,
            candidate_filter=self._filter_direction_candidates,
        )

    def _filter_direction_candidates(self, candidates: list[DirectionCandidate]) -> list[DirectionCandidate]:
        if not self.config.direction_pool_disable_momentum:
            return candidates
        filtered = [
            candidate
            for candidate in candidates
            if candidate.kind not in {DirectionCandidateKind.MOMENTUM, DirectionCandidateKind.ARCHIVE_MOMENTUM}
        ]
        return filtered if filtered else candidates

    def _sample_reference_dimer_initial_mode(self, current: State) -> tuple[np.ndarray, dict[str, object]]:
        lam = float(
            self.rng.uniform(
                self.config.reference_dimer_lambda_min,
                self.config.reference_dimer_lambda_max,
            )
        )
        initial_direction, info = sample_mixed_mode(
            current.positions,
            min_distance=self.config.reference_dimer_min_pair_distance,
            cell=current.cell,
            pbc=current.pbc,
            rng=self.rng,
            lam=lam,
        )
        info = dict(info)
        info["lambda"] = lam
        return initial_direction, info

    def _choose_reference_dimer_direction(
        self,
        current: State,
        *,
        initial_direction: np.ndarray | None = None,
        initial_info: dict[str, object] | None = None,
    ) -> DirectionChoice:
        if initial_direction is None:
            initial_direction, info = self._sample_reference_dimer_initial_mode(current)
        else:
            info = dict(initial_info or {})
        initial_direction = self._mask_fixed_direction(current, initial_direction)
        lam = float(info.get("lambda", 0.0))

        def evaluate_forces(positions: np.ndarray) -> tuple[float, np.ndarray]:
            trial_positions = np.asarray(positions, dtype=float).copy()
            if np.any(current.fixed_mask):
                trial_positions[current.fixed_mask] = current.positions[current.fixed_mask]
            trial_state = replace(current, positions=trial_positions)
            result = self.calculator.evaluate(trial_state)
            forces = -np.asarray(result.gradient, dtype=float)
            if np.any(current.fixed_mask):
                forces = forces.copy()
                forces[current.fixed_mask] = 0.0
            return float(result.energy), forces

        rotator = ReferenceDimerRotator(
            delta=self.config.reference_dimer_delta,
            bias_strength=self.config.reference_dimer_bias_strength,
            max_steps=self.config.reference_dimer_max_steps,
            rotation_tol=self.config.reference_dimer_rotation_tol,
            angular_step=self.config.reference_dimer_angular_step,
        )
        pair = info.get("pair", (0, 0))
        result = rotator.rotate(
            current.positions,
            initial_direction,
            evaluate_forces,
            lambda_value=lam,
            local_pair=pair,
        )
        self._record_reference_dimer_result(result)

        direction = self._mask_fixed_direction(current, result.direction).reshape(-1)
        return DirectionChoice(
            direction=direction,
            curvature=float(result.curvature),
            kind=DirectionCandidateKind.REFERENCE_DIMER,
            candidate_count=1,
            score=None,
            true_curvature=(
                float(result.curvature_true)
                if result.curvature_true is not None
                else float(result.curvature)
            ),
            biased_curvature=(
                float(result.curvature_biased)
                if result.curvature_biased is not None
                else float(result.curvature)
            ),
        )

    @staticmethod
    def _mask_fixed_direction(state: State, direction: np.ndarray) -> np.ndarray:
        direction_matrix = np.asarray(direction, dtype=float).copy()
        if direction_matrix.shape != state.positions.shape:
            raise ValueError("reference dimer direction must match positions shape")
        if not np.any(state.fixed_mask):
            direction_norm = float(np.linalg.norm(direction_matrix))
            if direction_norm <= 1e-15 or not np.isfinite(direction_norm):
                raise ValueError("reference dimer direction cannot be zero")
            return direction_matrix / direction_norm

        direction_matrix[state.fixed_mask] = 0.0
        if not np.any(~state.fixed_mask):
            raise ValueError("reference dimer requires at least one movable atom")
        direction_norm = float(np.linalg.norm(direction_matrix))
        if direction_norm <= 1e-15 or not np.isfinite(direction_norm):
            raise ValueError("reference dimer requires at least one movable atom")
        return direction_matrix / direction_norm

    def _curvatures_for_choice(
        self,
        current: State,
        proposal: ProposalPotential,
        choice: DirectionChoice,
        rebuild_softening_for_choice: bool,
    ) -> tuple[float, float]:
        if self.config.direction_engine == "reference_dimer":
            true_curvature = (
                float(choice.true_curvature)
                if choice.true_curvature is not None
                else float(choice.curvature)
            )
            return true_curvature, true_curvature
        true_curvature = self._true_directional_curvature(current, choice.direction)
        inner_curvature = (
            choice.curvature
            if self.config.direction_curvature_source == "inner" and not rebuild_softening_for_choice
            else self.oracle._directional_curvature(current, proposal, choice.direction)
        )
        return true_curvature, inner_curvature

    def _walk_candidate_from_seed(
        self,
        seed_state: State,
        archive=None,
        step_target: float | None = None,
        trial_index: int | None = None,
        proposal_index: int | None = None,
        seed_entry_id: int | None = None,
        proposal_optimizer_override: str | None = None,
        selected_direction_kinds: set[DirectionCandidateKind] | None = None,
        plateau_evolution_active: bool = False,
    ) -> State:
        current = seed_state
        previous_direction: np.ndarray | None = None
        anchor_direction: np.ndarray | None = None
        previous_relax_outcome: RelaxOutcomeClass | None = None
        biases: list[GaussianBiasTerm] = []
        sigma_scale = 1.0
        weight_scale = 1.0
        direct_qp_trust_radius = self._direct_qp_initial_trust_radius()
        seed_energy = self.calculator.evaluate(seed_state).energy if self.config.early_exit_enabled else None
        reference_dimer_initial_direction: np.ndarray | None = None
        reference_dimer_initial_info: dict[str, object] | None = None
        if self.config.direction_engine == "reference_dimer":
            reference_dimer_initial_direction, reference_dimer_initial_info = (
                self._sample_reference_dimer_initial_mode(seed_state)
            )

        for step_index in range(self.config.max_steps_per_walk):
            if anchor_direction is None:
                anchor_progress_index = 0 if trial_index is None else max(0, min(trial_index, self.config.max_trials - 1))
                anchor_direction = self.oracle.generator.generate_initial_direction(
                    current,
                    step_index=anchor_progress_index,
                    max_steps=self.config.max_trials,
                    lambda_start=self.config.lambda_bond_start,
                    lambda_end=self.config.lambda_bond_end,
                    n_bond_pairs=self.config.n_bond_pairs,
                    bond_distance_threshold=self.config.bond_distance_threshold,
                )
            softening = self._build_softening(current, anchor_direction)
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            scoring_proposal = self._direction_scoring_proposal(proposal)
            if plateau_evolution_active:
                self._plateau_evolution_active_steps += 1
            choice = self._choose_walk_direction(
                current=current,
                proposal=proposal,
                scoring_proposal=scoring_proposal,
                previous_direction=previous_direction,
                anchor_direction=anchor_direction,
                archive=archive,
                step_target=step_target,
                sigma_scale=sigma_scale,
                previous_relax_outcome=previous_relax_outcome,
                trial_index=trial_index,
                proposal_index=proposal_index,
                seed_entry_id=seed_entry_id,
                step_index=step_index,
                plateau_evolution_active=plateau_evolution_active,
                reference_dimer_initial_direction=reference_dimer_initial_direction,
                reference_dimer_initial_info=reference_dimer_initial_info,
            )
            if selected_direction_kinds is not None:
                selected_direction_kinds.add(choice.kind)
            self._capture_direction_record(
                trial_index=trial_index,
                proposal_index=proposal_index,
                step_index=step_index,
                seed_entry_id=seed_entry_id,
                choice=choice,
                anchor_direction=anchor_direction,
            )
            self._record_direction_choice(choice)
            self._record_direction_diagnostics(
                trial_index=trial_index,
                proposal_index=proposal_index,
                step_index=step_index,
                choice=choice,
                anchor_direction=anchor_direction,
            )
            rebuild_softening_for_choice = self._should_rebuild_softening_for_choice(anchor_direction, choice.direction)
            if rebuild_softening_for_choice:
                softening = self._build_softening(current, choice.direction)
                proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            true_curvature, inner_curvature = self._curvatures_for_choice(
                current,
                proposal,
                choice,
                rebuild_softening_for_choice,
            )
            self._record_direction_curvatures(
                choice.kind,
                choice_curvature=choice.curvature,
                true_curvature=true_curvature,
                inner_curvature=inner_curvature,
            )
            sigma = self._execution_step_scale(
                current,
                choice.direction,
                true_curvature,
                sigma_scale,
                step_target=step_target,
            )
            self._record_step_displacement_metrics(current, choice.direction, sigma)
            true_before = self.calculator.evaluate(current)
            true_energy_before = true_before.energy
            g_parallel = float(np.dot(true_before.gradient.reshape(-1), choice.direction))
            if self.config.proposal_step_mode == "direct_qp":
                direct_qp_gamma = self._direct_qp_scalar_gamma(true_curvature)
                if self.config.direct_qp_hessian == "rank1":
                    direct_qp_gamma = self._direct_qp_rank1_gamma_floor()
                result = self._execute_direct_qp_step(
                    current=current,
                    choice=choice,
                    sigma=sigma,
                    true_before=true_before,
                    trust_radius=direct_qp_trust_radius,
                    gamma=direct_qp_gamma,
                    kappa=self._direct_qp_scalar_kappa(direct_qp_gamma),
                    directional_curvature=true_curvature,
                )
                self._record_direct_qp_curvature(true_curvature)
                if result is None:
                    break
                current_candidate, clipped = self._clip_walk_displacement(
                    reference=seed_state,
                    candidate=result.state,
                    max_displacement=self.config.walk_trust_radius,
                )
                self._walk_displacement_clips += int(clipped)
                if not self.geometry_validator.is_valid_state(current_candidate):
                    self._record_direct_qp_result(
                        step_norm=float(np.linalg.norm(result.step)),
                        progress=result.progress,
                        target_error=result.target_error,
                        predicted_delta=result.predicted_delta,
                        true_delta=result.true_delta,
                        model_error=result.model_error,
                        gamma=result.gamma,
                        kappa=result.kappa,
                        action="reject",
                        rejected=True,
                    )
                    break
                true_energy_after = result.energy_after
                if current_candidate is not result.state:
                    true_energy_after = self.calculator.evaluate(current_candidate).energy
                if not np.isfinite(true_energy_after):
                    self._record_direct_qp_result(
                        step_norm=float(np.linalg.norm(result.step)),
                        progress=result.progress,
                        target_error=result.target_error,
                        predicted_delta=result.predicted_delta,
                        true_delta=result.true_delta,
                        model_error=result.model_error,
                        gamma=result.gamma,
                        kappa=result.kappa,
                        action="reject",
                        rejected=True,
                    )
                    break
                micro_result = self._direct_qp_micro_correct(current_candidate, model_error=result.model_error)
                if micro_result is not None:
                    current_candidate = micro_result.state
                    true_energy_after = micro_result.energy
                    self._record_direct_qp_micro_result(micro_result)
                    if not self.geometry_validator.is_valid_state(current_candidate) or not np.isfinite(true_energy_after):
                        self._record_direct_qp_result(
                            step_norm=float(np.linalg.norm(result.step)),
                            progress=result.progress,
                            target_error=result.target_error,
                            predicted_delta=result.predicted_delta,
                            true_delta=result.true_delta,
                            model_error=result.model_error,
                            gamma=result.gamma,
                            kappa=result.kappa,
                            action="reject",
                            rejected=True,
                        )
                        break
                if result.action == "shrink":
                    direct_qp_trust_radius = max(
                        self.config.direct_qp_min_trust_radius,
                        direct_qp_trust_radius * self.config.direct_qp_shrink_factor,
                    )
                elif result.action == "expand":
                    direct_qp_trust_radius = min(
                        self.config.walk_trust_radius,
                        direct_qp_trust_radius * self.config.direct_qp_expand_factor,
                    )
                displacement = mic_displacement(
                    current_candidate.positions,
                    current.positions,
                    current.cell,
                    current.pbc,
                ).reshape(-1)
                if np.linalg.norm(displacement) > 1e-8:
                    previous_direction = displacement / np.linalg.norm(displacement)
                previous_relax_outcome = None
                current = current_candidate
                if self._should_early_exit_walk(seed_energy, true_energy_after):
                    self._walk_early_stops += 1
                    break
                if clipped:
                    break
                continue
            weight_curvature = self._bias_weight_curvature_for_choice(choice, inner_curvature)
            weight = self._bias_weight(weight_curvature, sigma) * weight_scale
            self._record_bias_weight(weight)
            biases.append(
                GaussianBiasTerm(
                    center=current.flatten_positions(),
                    direction=choice.direction,
                    sigma=sigma,
                    weight=weight,
                )
            )
            proposal = ProposalPotential(self.calculator, biases=biases, softening=softening)
            trial_state = CartesianCoordinates.from_state(current).displace(TangentVector(choice.direction), sigma)
            if not self.geometry_validator.is_valid_state(trial_state):
                break
            proposal_optimizer = self._proposal_optimizer_for_outcome(
                previous_relax_outcome,
                override=proposal_optimizer_override,
            )
            if proposal_optimizer != self.config.proposal_optimizer:
                self._proposal_optimizer_alt_steps += 1
            proposal_relax = Relaxer(proposal.evaluate, optimizer=proposal_optimizer).relax(
                trial_state,
                fmax=self.config.proposal_fmax,
                maxiter=self.config.proposal_relax_steps,
                coordinate_trust_radius=self.config.proposal_trust_radius,
                trajectory_callback=self._relaxation_trajectory_callback(
                    self._trajectory_name(
                        "proposal_relax",
                        trial_index=trial_index,
                        proposal_index=proposal_index,
                        step_index=step_index,
                    )
                ),
                trajectory_stride=self.config.relaxation_trajectory_stride,
            )
            current_candidate, clipped = self._clip_walk_displacement(
                reference=seed_state,
                candidate=proposal_relax.state,
                max_displacement=self.config.walk_trust_radius,
            )
            self._walk_displacement_clips += int(clipped)
            if not self.geometry_validator.is_valid_state(current_candidate):
                break
            true_energy_after = self.calculator.evaluate(current_candidate).energy
            if not np.isfinite(true_energy_after):
                break
            proposal_relax = replace(
                proposal_relax,
                outcome_class=Relaxer.classify_outcome(
                    initial_energy=true_energy_before,
                    final_energy=proposal_relax.energy,
                    gradient_norm=proposal_relax.gradient_norm,
                    fmax=self.config.proposal_fmax,
                    displacement_rms=proposal_relax.displacement_rms,
                    displacement_max=proposal_relax.displacement_max,
                    active_bound_fraction=proposal_relax.active_bound_fraction,
                    true_delta=true_energy_after - true_energy_before,
                ),
            )
            self._record_relax_result("proposal_relax", proposal_relax, self.config.proposal_fmax)
            previous_relax_outcome = proposal_relax.outcome_class
            trust_update = self.trust_controller.update(
                curvature=true_curvature,
                sigma=sigma,
                true_delta=true_energy_after - true_energy_before,
                sigma_scale=sigma_scale,
                weight_scale=weight_scale,
                g_parallel=g_parallel,
                error_floor=0.1 * (step_target if step_target is not None else self.config.target_uphill_energy),
                active_bound_fraction=proposal_relax.active_bound_fraction,
                bias_weight=weight,
            )
            sigma_scale = trust_update.sigma_scale
            weight_scale = trust_update.weight_scale
            self._record_trust_update(trust_update)
            displacement = mic_displacement(
                current_candidate.positions,
                current.positions,
                current.cell,
                current.pbc,
            ).reshape(-1)
            if np.linalg.norm(displacement) > 1e-8:
                previous_direction = displacement / np.linalg.norm(displacement)
            current = current_candidate
            if self._should_early_exit_walk(seed_energy, true_energy_after):
                self._walk_early_stops += 1
                break
            if clipped:
                break
        return current

    def _should_early_exit_walk(self, seed_energy: float | None, true_energy_after: float) -> bool:
        if seed_energy is None:
            return False
        return bool(true_energy_after < seed_energy - self.config.early_exit_energy_tol)

    def _walk_from_seed(self, seed_state: State) -> RelaxResult:
        return self.relax_true_minimum(self._walk_candidate_from_seed(seed_state))

    def _select_seed_entry(self, archive):
        if self.config.use_archive_acquisition:
            primary = archive.select_seed(self.selector, self.rng)
        else:
            primary = archive.next_seed()
            primary.visits += 1
            primary.node_trials += 1
        selected = self._seed_diversity_override(archive, primary)
        self._record_seed_selection(selected)
        return selected

    def _select_metropolis_seed_entry(self, entry):
        entry.visits += 1
        entry.node_trials += 1
        self._record_seed_selection(entry)
        return entry

    def _update_metropolis_chain(self, current_entry, candidate_entry, is_new: bool):
        self._metropolis_trials += 1
        if not is_new:
            self._metropolis_duplicate_rejects += 1
            self._metropolis_rejects += 1
            return current_entry
        delta = float(candidate_entry.energy) - float(current_entry.energy)
        if delta <= 0.0:
            self._metropolis_downhill_accepts += 1
            self._metropolis_accepts += 1
            return candidate_entry
        probability = float(np.exp(-delta / self.config.metropolis_temperature))
        if self.rng.random() < probability:
            self._metropolis_uphill_accepts += 1
            self._metropolis_accepts += 1
            return candidate_entry
        self._metropolis_uphill_rejects += 1
        self._metropolis_rejects += 1
        return current_entry

    def _seed_diversity_override(self, archive, primary):
        limit = self.config.same_seed_max_consecutive
        if limit is None or self._last_seed_entry_id != primary.entry_id or self._same_seed_consecutive < limit:
            return primary
        alternatives = [
            entry
            for entry in archive.entries
            if entry.entry_id != primary.entry_id and entry.is_frontier and not entry.is_dead
        ]
        if not alternatives:
            alternatives = [
                entry
                for entry in archive.entries
                if entry.entry_id != primary.entry_id and not entry.is_dead
            ]
        if not alternatives:
            return primary
        replacement = max(
            alternatives,
            key=lambda entry: (
                self.selector.score_entry(archive, entry),
                -entry.entry_id,
            ),
        )
        primary.visits = max(0, primary.visits - 1)
        primary.node_trials = max(0, primary.node_trials - 1)
        replacement.visits += 1
        replacement.node_trials += 1
        self._seed_diversity_reseeds += 1
        return replacement

    def _record_seed_selection(self, entry) -> None:
        if self._last_seed_entry_id == entry.entry_id:
            self._same_seed_consecutive += 1
        else:
            self._last_seed_entry_id = entry.entry_id
            self._same_seed_consecutive = 1

    def _step_scale(self, curvature: float) -> float:
        effective = max(abs(curvature), 1e-4)
        sigma = np.sqrt(2.0 * self.config.target_uphill_energy / effective)
        return float(np.clip(sigma, self.config.min_step_scale, self.config.max_step_scale))

    def _scaled_step_scale(self, curvature: float, sigma_scale: float, step_target: float | None = None) -> float:
        target = self.config.target_uphill_energy if step_target is None else step_target
        effective = max(abs(curvature), 1e-4)
        sigma = np.sqrt(2.0 * target / effective) * sigma_scale
        return float(np.clip(sigma, self.config.min_step_scale, self.config.max_step_scale))

    def _execution_step_scale(
        self,
        state: State,
        direction: np.ndarray,
        curvature: float,
        sigma_scale: float,
        step_target: float | None = None,
    ) -> float:
        if self.config.step_length_mode == "curvature_adaptive":
            return self._scaled_step_scale(curvature, sigma_scale, step_target=step_target)
        metrics = self._direction_step_metrics(
            state,
            direction,
            sigma=1.0,
            active_threshold=self.config.step_active_threshold,
        )
        rms_key = (
            "direction_per_atom_rms_active"
            if self.config.step_rms_scope == "active_atoms"
            else "direction_per_atom_rms_all"
        )
        direction_rms = max(float(metrics[rms_key]), 1e-12)
        target_rms = min(self.config.target_step_rms * sigma_scale, self.config.max_step_rms)
        return float(target_rms / direction_rms)

    def _direction_score_sigma(self, sigma_scale: float, step_target: float | None = None) -> float:
        if self.config.direction_score_sigma_mode == "fixed_reference":
            sigma = np.sqrt(2.0 * self.config.target_uphill_energy)
            return float(np.clip(sigma, self.config.min_step_scale, self.config.max_step_scale))
        if self.config.direction_score_sigma_mode == "adaptive":
            raise ValueError("adaptive direction scoring uses per-candidate score_sigma_fn")
        return self._scaled_step_scale(1.0, sigma_scale, step_target=step_target)

    def _direction_score_sigma_fn(self, sigma_scale: float, step_target: float | None = None):
        if self.config.direction_score_sigma_mode != "adaptive":
            return None
        return lambda curvature: self._scaled_step_scale(curvature, sigma_scale, step_target=step_target)

    def _bias_weight(self, curvature: float, sigma: float) -> float:
        raw = sigma * sigma * max(curvature + self.config.target_negative_curvature, 0.0)
        return float(np.clip(raw, self.config.bias_weight_min, self.config.bias_weight_max))

    @staticmethod
    def _bias_weight_curvature_for_choice(choice: DirectionChoice, inner_curvature: float) -> float:
        if (
            choice.kind is DirectionCandidateKind.REFERENCE_DIMER
            and choice.biased_curvature is not None
        ):
            return float(choice.biased_curvature)
        return float(inner_curvature)

    def _true_directional_curvature(self, state: State, direction: np.ndarray) -> float:
        proposal = ProposalPotential(self.calculator)
        return self.oracle._directional_curvature(state, proposal, direction)

    @staticmethod
    def _direction_step_metrics(
        state: State,
        direction: np.ndarray,
        sigma: float,
        active_threshold: float,
    ) -> dict[str, float]:
        values = np.asarray(direction, dtype=float).reshape(state.n_atoms, 3)
        atom_norms = np.linalg.norm(values, axis=1)
        movable_norms = atom_norms[state.movable_mask]
        if movable_norms.size == 0:
            return {
                "direction_full_norm": 0.0,
                "direction_per_atom_rms_all": 0.0,
                "direction_per_atom_rms_active": 0.0,
                "step_displacement_rms_all": 0.0,
                "step_displacement_rms_active": 0.0,
                "predicted_max_atom_displacement": 0.0,
            }
        full_norm = float(np.linalg.norm(values.reshape(-1)))
        rms_all = float(np.sqrt(np.mean(movable_norms * movable_norms)))
        max_norm = float(np.max(movable_norms))
        active_mask = movable_norms > active_threshold * max(max_norm, 1e-12)
        active_norms = movable_norms[active_mask] if np.any(active_mask) else movable_norms
        rms_active = float(np.sqrt(np.mean(active_norms * active_norms)))
        step = abs(float(sigma))
        return {
            "direction_full_norm": full_norm,
            "direction_per_atom_rms_all": rms_all,
            "direction_per_atom_rms_active": rms_active,
            "step_displacement_rms_all": step * rms_all,
            "step_displacement_rms_active": step * rms_active,
            "predicted_max_atom_displacement": step * max_norm,
        }

    def _record_step_displacement_metrics(self, state: State, direction: np.ndarray, sigma: float) -> None:
        metrics = self._direction_step_metrics(
            state,
            direction,
            sigma,
            active_threshold=self.config.step_active_threshold,
        )
        step_rms = float(metrics["step_displacement_rms_all"])
        max_atom = float(metrics["predicted_max_atom_displacement"])
        self._step_displacement_records += 1
        self._step_displacement_rms_sum += step_rms
        self._step_displacement_rms_max = max(self._step_displacement_rms_max, step_rms)
        self._step_displacement_max_atom_sum += max_atom
        self._step_displacement_max_atom_max = max(self._step_displacement_max_atom_max, max_atom)

    @staticmethod
    def _history_bias_gradient(state: State, biases: list[GaussianBiasTerm]) -> np.ndarray | None:
        if not biases:
            return None
        flat_positions = state.flatten_positions()
        gradient = np.zeros_like(flat_positions)
        for bias in biases:
            _, bias_gradient = bias.evaluate(flat_positions, cell=state.cell, pbc=state.pbc)
            gradient += bias_gradient
        return gradient

    def _direction_scoring_proposal(self, inner_proposal: ProposalPotential) -> ProposalPotential:
        if self.config.direction_curvature_source == "true":
            return ProposalPotential(self.calculator)
        return inner_proposal

    def _continuity_weight_for_outcome(self, outcome: RelaxOutcomeClass | None) -> float:
        if not self.config.enable_outcome_gated_continuity:
            return self.config.continuity_weight
        if outcome in {RelaxOutcomeClass.STAGNATED, RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE}:
            return 0.0
        if outcome in {RelaxOutcomeClass.DAMAGED, RelaxOutcomeClass.ENERGY_EXPLODED, RelaxOutcomeClass.GEOMETRY_INVALID}:
            return 0.5 * self.config.continuity_weight
        return self.config.continuity_weight

    def _n_bond_pairs_for_outcome(self, outcome: RelaxOutcomeClass | None) -> int:
        n_pairs = self.config.n_bond_pairs
        if outcome in {RelaxOutcomeClass.STAGNATED, RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE}:
            n_pairs += self.config.stagnation_bond_pair_boost
            if self.config.max_stagnation_bond_pairs is not None:
                n_pairs = min(n_pairs, self.config.max_stagnation_bond_pairs)
        return n_pairs

    def _proposal_optimizer_for_outcome(
        self,
        outcome: RelaxOutcomeClass | None,
        override: str | None = None,
    ) -> str:
        if override is not None:
            return override
        if self.config.proposal_optimizer_alt is not None and outcome in {
            RelaxOutcomeClass.STAGNATED,
            RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE,
        }:
            return self.config.proposal_optimizer_alt
        return self.config.proposal_optimizer

    def _reset_trust_stats(self) -> None:
        self._trust_steps = 0
        self._trust_model_error_sum = 0.0
        self._trust_shrink_steps = 0
        self._trust_expand_steps = 0
        self._trust_damage_events = 0

    def _reset_direction_stats(self) -> None:
        self._direction_choices = 0
        self._direction_candidate_evaluations = 0
        self._direction_selected = {kind: 0 for kind in DirectionCandidateKind}
        self._direction_curvature_stats = {kind: self._new_curvature_stats() for kind in DirectionCandidateKind}
        self._direction_true_curvature_stats = {kind: self._new_curvature_stats() for kind in DirectionCandidateKind}
        self._direction_inner_curvature_stats = {kind: self._new_curvature_stats() for kind in DirectionCandidateKind}
        self._direction_rigid_overlap_sum = 0.0
        self._direction_post_projection_rigid_overlap_sum = 0.0
        self._step_displacement_records = 0
        self._step_displacement_rms_sum = 0.0
        self._step_displacement_rms_max = 0.0
        self._step_displacement_max_atom_sum = 0.0
        self._step_displacement_max_atom_max = 0.0
        self._direction_bond_pairs_requested = 0
        self._direction_bond_pairs_generated = 0
        self._direction_fallback_bond_pairs_generated = 0
        self._direction_bond_candidates_valid = 0
        self._plateau_evolution_active_steps = 0
        self._plateau_evolution_candidate_steps = 0
        self._plateau_evolution_candidates_generated = 0
        self._archive_escape_momentum_candidate_steps = 0
        self._archive_escape_momentum_candidates_generated = 0
        self._walk_displacement_clips = 0
        self._walk_early_stops = 0
        self._fragment_rejections = 0

    @staticmethod
    def _new_curvature_stats() -> dict[str, float | int | None]:
        return {"count": 0, "sum": 0.0, "min": None, "max": None}

    def _new_direction_type_memory(self) -> DirectionTypeMemory:
        return DirectionTypeMemory(
            enabled=self.config.direction_type_ucb_enabled,
            success_weight=self.config.direction_type_success_weight,
            exploration_weight=self.config.direction_type_exploration_weight,
            window=self.config.direction_type_ucb_window,
        )

    def _reset_seed_diversity_stats(self) -> None:
        self._last_seed_entry_id: int | None = None
        self._same_seed_consecutive = 0
        self._seed_diversity_reseeds = 0

    def _reset_reference_dimer_stats(self) -> None:
        self._reference_dimer_steps = 0
        self._reference_dimer_rotation_sum = 0
        self._reference_dimer_converged = 0
        self._reference_dimer_curvature_sum = 0.0
        self._reference_dimer_true_curvature_sum = 0.0
        self._reference_dimer_biased_curvature_sum = 0.0
        self._reference_dimer_abs_dot_sum = 0.0

    def _reset_relax_stats(self) -> None:
        self._relax_stats = {
            "true_quench": {
                "count": 0,
                "unconverged": 0,
                "max_gradient": 0.0,
                "n_iter_sum": 0,
                "n_iter_values": [],
                "bound_fraction_sum": 0.0,
                "max_bound_fraction": 0.0,
                "displacement_rms_sum": 0.0,
                "max_displacement": 0.0,
                "outcome_counts": {outcome.value: 0 for outcome in RelaxOutcomeClass},
            },
            "proposal_relax": {
                "count": 0,
                "unconverged": 0,
                "max_gradient": 0.0,
                "n_iter_sum": 0,
                "n_iter_values": [],
                "bound_fraction_sum": 0.0,
                "max_bound_fraction": 0.0,
                "displacement_rms_sum": 0.0,
                "max_displacement": 0.0,
                "outcome_counts": {outcome.value: 0 for outcome in RelaxOutcomeClass},
            },
        }

    def _reset_bias_stats(self) -> None:
        self._bias_steps = 0
        self._bias_zero_steps = 0
        self._bias_weight_sum = 0.0
        self._bias_weight_max = 0.0

    def _reset_direct_qp_stats(self) -> None:
        self._direct_qp_steps = 0
        self._direct_qp_rejected = 0
        self._direct_qp_step_norm_sum = 0.0
        self._direct_qp_progress_sum = 0.0
        self._direct_qp_target_error_sum = 0.0
        self._direct_qp_model_error_sum = 0.0
        self._direct_qp_shrink_steps = 0
        self._direct_qp_expand_steps = 0
        self._direct_qp_gamma_sum = 0.0
        self._direct_qp_kappa_sum = 0.0
        self._direct_qp_curvature_history: list[float] = []
        self._direct_qp_high_model_error_streak = 0
        self._direct_qp_high_model_error_streak_max = 0
        self._direct_qp_micro_count = 0
        self._direct_qp_micro_iteration_sum = 0
        self._direct_qp_micro_displacement_rms_sum = 0.0
        self._direct_qp_micro_displacement_max = 0.0

    def _reset_local_softening_stats(self) -> None:
        self._local_softening_terms_last = 0
        self._local_softening_terms_total = 0
        self._local_softening_builds = 0
        self._local_softening_terms_built_total = 0

    def _reset_metropolis_stats(self) -> None:
        self._metropolis_trials = 0
        self._metropolis_accepts = 0
        self._metropolis_rejects = 0
        self._metropolis_downhill_accepts = 0
        self._metropolis_uphill_accepts = 0
        self._metropolis_uphill_rejects = 0
        self._metropolis_duplicate_rejects = 0

    def _metropolis_stats_summary(self, current_entry) -> dict[str, float | int | str]:
        total = self._metropolis_accepts + self._metropolis_rejects
        return {
            "seed_selection_mode": self.config.seed_selection_mode,
            "metropolis_temperature": float(self.config.metropolis_temperature),
            "metropolis_trials": self._metropolis_trials,
            "metropolis_accepts": self._metropolis_accepts,
            "metropolis_rejects": self._metropolis_rejects,
            "metropolis_downhill_accepts": self._metropolis_downhill_accepts,
            "metropolis_uphill_accepts": self._metropolis_uphill_accepts,
            "metropolis_uphill_rejects": self._metropolis_uphill_rejects,
            "metropolis_duplicate_rejects": self._metropolis_duplicate_rejects,
            "metropolis_acceptance_rate": float(self._metropolis_accepts / total) if total else 0.0,
            "metropolis_current_entry_id": int(current_entry.entry_id),
            "metropolis_current_energy": float(current_entry.energy),
        }

    def _accepted_structure_log_path(self) -> Path | None:
        path = self.config.accepted_structures_log
        return Path(path) if path is not None else None

    def _reset_accepted_structure_log(self) -> None:
        path = self._accepted_structure_log_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def _direction_diagnostics_path(self) -> Path | None:
        if not self.config.direction_diagnostics_enabled:
            return None
        path = self.config.direction_diagnostics_path
        if path is None:
            return None
        return Path(path)

    def _reset_direction_diagnostics(self) -> None:
        path = self._direction_diagnostics_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def _record_direction_diagnostics(
        self,
        *,
        trial_index: int | None,
        proposal_index: int | None,
        step_index: int,
        choice: DirectionChoice,
        anchor_direction: np.ndarray | None,
    ) -> None:
        path = self._direction_diagnostics_path()
        if path is None:
            return
        anchor_cosine = self._direction_anchor_cosine(anchor_direction, choice.direction)
        payload = {
            "trial": None if trial_index is None else int(trial_index),
            "proposal": None if proposal_index is None else int(proposal_index),
            "step": int(step_index),
            "selected_kind": choice.kind.value,
            "curvature": float(choice.curvature),
            "candidate_count": int(choice.candidate_count),
            "anchor_cosine": anchor_cosine,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _prepare_structure_output_dirs(self) -> None:
        for path_value in (
            self.config.accepted_structures_dir,
            self.config.proposal_minima_dir if self.config.write_proposal_minima else None,
            self.config.relaxation_trajectory_dir if self.config.write_relaxation_trajectories else None,
        ):
            if path_value is not None:
                Path(path_value).mkdir(parents=True, exist_ok=True)

    def _record_accepted_structure(
        self,
        *,
        trial_index: int,
        seed_entry_id: int,
        discovered_entry_id: int,
        state: State,
        energy: float,
        best_energy: float,
    ) -> None:
        path = self._accepted_structure_log_path()
        if path is not None:
            payload = {
                "trial_index": int(trial_index),
                "seed_entry_id": int(seed_entry_id),
                "discovered_entry_id": int(discovered_entry_id),
                "energy": float(energy),
                "best_energy": float(best_energy),
                "descriptor": structural_descriptor(state).astype(float).tolist(),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._write_accepted_minimum(
            trial_index=trial_index,
            state=state,
            energy=energy,
            seed_entry_id=seed_entry_id,
            discovered_entry_id=discovered_entry_id,
        )

    def _write_accepted_minimum(
        self,
        *,
        trial_index: int,
        state: State,
        energy: float,
        seed_entry_id: int,
        discovered_entry_id: int,
    ) -> None:
        directory = self.config.accepted_structures_dir
        if directory is None:
            return
        filename = f"trial{trial_index:04d}_entry{discovered_entry_id:04d}_accepted.xyz"
        self._write_state_xyz(
            Path(directory) / filename,
            state,
            {
                "trial_index": trial_index,
                "seed_entry_id": seed_entry_id,
                "discovered_entry_id": discovered_entry_id,
                "energy": energy,
                "status": "accepted",
            },
        )

    def _write_proposal_minimum(
        self,
        *,
        trial_index: int,
        proposal_index: int,
        state: State,
        energy: float,
        seed_entry_id: int,
        status: str,
        discovered_entry_id: int | None = None,
    ) -> None:
        if not self.config.write_proposal_minima or self.config.proposal_minima_dir is None:
            return
        entry_part = "none" if discovered_entry_id is None else f"{discovered_entry_id:04d}"
        filename = f"trial{trial_index:04d}_proposal{proposal_index:03d}_entry{entry_part}_{status}.xyz"
        self._write_state_xyz(
            Path(self.config.proposal_minima_dir) / filename,
            state,
            {
                "trial_index": trial_index,
                "proposal_index": proposal_index,
                "seed_entry_id": seed_entry_id,
                "discovered_entry_id": -1 if discovered_entry_id is None else discovered_entry_id,
                "energy": energy,
                "status": status,
            },
        )

    def _trajectory_name(
        self,
        phase: str,
        *,
        trial_index: int | None,
        proposal_index: int | None,
        step_index: int | None,
    ) -> str | None:
        if trial_index is None or proposal_index is None or step_index is None:
            return None
        return (
            f"trial{trial_index + 1:04d}_proposal{proposal_index + 1:03d}_"
            f"step{step_index + 1:03d}_{phase}"
        )

    def _relaxation_trajectory_callback(self, trajectory_name: str | None):
        if not self.config.write_relaxation_trajectories or self.config.relaxation_trajectory_dir is None:
            return None
        if trajectory_name is None:
            if not self._missing_trajectory_context_warned:
                print(
                    "[ssw] warning: missing trajectory context; skipping unnamed relaxation trajectory",
                    file=sys.stderr,
                    flush=True,
                )
                self._missing_trajectory_context_warned = True
            return None
        directory = Path(self.config.relaxation_trajectory_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{trajectory_name}.xyz"
        counter = {"step": 0}

        def record(state: State) -> None:
            step = counter["step"]
            counter["step"] = step + 1
            self._write_state_xyz(path, state, {"trajectory": trajectory_name, "trajectory_step": step}, append=step > 0)

        return record

    @staticmethod
    def _write_state_xyz(path: Path, state: State, info: dict[str, object], append: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=state.cell,
            pbc=state.pbc,
        )
        atoms.info.update(info)
        write(path, atoms, append=append)

    def _record_relax_result(self, label: str, result: RelaxResult, fmax: float) -> None:
        stats = self._relax_stats[label]
        stats["count"] += 1
        stats["n_iter_sum"] += result.n_iter
        stats["n_iter_values"].append(result.n_iter)
        stats["max_gradient"] = max(float(stats["max_gradient"]), result.gradient_norm)
        stats["unconverged"] += int(result.gradient_norm > fmax)
        stats["bound_fraction_sum"] += result.active_bound_fraction
        stats["max_bound_fraction"] = max(float(stats["max_bound_fraction"]), result.active_bound_fraction)
        stats["displacement_rms_sum"] += result.displacement_rms
        stats["max_displacement"] = max(float(stats["max_displacement"]), result.displacement_max)
        stats["outcome_counts"][result.outcome_class.value] += 1

    def _record_bias_weight(self, weight: float) -> None:
        self._bias_steps += 1
        self._bias_zero_steps += int(abs(weight) <= 1e-14)
        self._bias_weight_sum += float(weight)
        self._bias_weight_max = max(self._bias_weight_max, float(weight))

    def _record_direct_qp_result(
        self,
        *,
        step_norm: float,
        progress: float,
        target_error: float,
        predicted_delta: float,
        true_delta: float,
        model_error: float,
        gamma: float,
        kappa: float,
        action: str,
        rejected: bool,
    ) -> None:
        self._direct_qp_steps += 1
        self._direct_qp_rejected += int(rejected)
        self._direct_qp_step_norm_sum += float(step_norm)
        self._direct_qp_progress_sum += float(progress)
        self._direct_qp_target_error_sum += float(target_error)
        self._direct_qp_model_error_sum += float(model_error)
        self._direct_qp_gamma_sum += float(gamma)
        self._direct_qp_kappa_sum += float(kappa)
        if model_error > self.config.direct_qp_gamma_model_error_threshold:
            self._direct_qp_high_model_error_streak += 1
            self._direct_qp_high_model_error_streak_max = max(
                self._direct_qp_high_model_error_streak_max,
                self._direct_qp_high_model_error_streak,
            )
        else:
            self._direct_qp_high_model_error_streak = 0
        if action == "shrink":
            self._direct_qp_shrink_steps += 1
        if action == "expand":
            self._direct_qp_expand_steps += 1

    def _record_direct_qp_micro_result(self, result: RelaxResult) -> None:
        self._direct_qp_micro_count += 1
        self._direct_qp_micro_iteration_sum += int(result.n_iter)
        self._direct_qp_micro_displacement_rms_sum += float(result.displacement_rms)
        self._direct_qp_micro_displacement_max = max(
            self._direct_qp_micro_displacement_max,
            float(result.displacement_max),
        )

    def _record_reference_dimer_result(self, result: ReferenceDimerResult) -> None:
        self._reference_dimer_steps += 1
        self._reference_dimer_rotation_sum += int(result.rotations)
        self._reference_dimer_converged += int(result.converged)
        self._reference_dimer_curvature_sum += float(result.curvature)
        if result.curvature_true is not None:
            self._reference_dimer_true_curvature_sum += float(result.curvature_true)
        if result.curvature_biased is not None:
            self._reference_dimer_biased_curvature_sum += float(result.curvature_biased)
        self._reference_dimer_abs_dot_sum += abs(float(result.dot_initial))

    def _record_direction_curvatures(
        self,
        kind: DirectionCandidateKind,
        *,
        choice_curvature: float,
        true_curvature: float,
        inner_curvature: float,
    ) -> None:
        self._update_curvature_stats(self._direction_curvature_stats[kind], choice_curvature)
        self._update_curvature_stats(self._direction_true_curvature_stats[kind], true_curvature)
        self._update_curvature_stats(self._direction_inner_curvature_stats[kind], inner_curvature)

    @staticmethod
    def _update_curvature_stats(stats: dict[str, float | int | None], value: float) -> None:
        value = float(value)
        if not np.isfinite(value):
            return
        stats["count"] = int(stats["count"]) + 1
        stats["sum"] = float(stats["sum"]) + value
        stats["min"] = value if stats["min"] is None else min(float(stats["min"]), value)
        stats["max"] = value if stats["max"] is None else max(float(stats["max"]), value)

    def _record_direction_choice(self, choice: DirectionChoice) -> None:
        self._direction_choices += 1
        self._direction_candidate_evaluations += choice.candidate_count
        self._direction_selected[choice.kind] += 1
        self._direction_rigid_overlap_sum += choice.mean_rigid_body_overlap
        self._direction_post_projection_rigid_overlap_sum += choice.mean_post_projection_rigid_body_overlap
        evolved_candidate_count = int(choice.evolved_candidate_count)
        if evolved_candidate_count:
            self._plateau_evolution_candidate_steps += 1
            self._plateau_evolution_candidates_generated += evolved_candidate_count
        archive_momentum_candidate_count = int(choice.archive_momentum_candidate_count)
        if archive_momentum_candidate_count:
            self._archive_escape_momentum_candidate_steps += 1
            self._archive_escape_momentum_candidates_generated += archive_momentum_candidate_count
        self._direction_bond_pairs_requested += self.oracle.generator.last_random_bond_pairs_requested
        self._direction_bond_pairs_generated += self.oracle.generator.last_random_bond_pairs_generated
        self._direction_fallback_bond_pairs_generated += self.oracle.generator.last_fallback_bond_pairs_generated
        self._direction_bond_candidates_valid += self.oracle.generator.last_random_bond_candidates_valid

    def _record_trust_update(self, update: TrustRegionUpdate) -> None:
        self._trust_steps += 1
        self._trust_model_error_sum += update.model_error
        if update.action == "shrink":
            self._trust_shrink_steps += 1
        if update.action == "expand":
            self._trust_expand_steps += 1
        if update.damaged:
            self._trust_damage_events += 1

    def _trust_stats_summary(self) -> dict[str, float | int]:
        mean_error = self._trust_model_error_sum / self._trust_steps if self._trust_steps else 0.0
        return {
            "trust_region_steps": self._trust_steps,
            "trust_model_error_mean": float(mean_error),
            "trust_shrink_steps": self._trust_shrink_steps,
            "trust_expand_steps": self._trust_expand_steps,
            "trust_damage_events": self._trust_damage_events,
        }

    def _reference_dimer_stats_summary(self) -> dict[str, float | int]:
        steps = self._reference_dimer_steps
        return {
            "reference_dimer_steps": steps,
            "reference_dimer_mean_rotations": (
                float(self._reference_dimer_rotation_sum / steps) if steps else 0.0
            ),
            "reference_dimer_converged_fraction": (
                float(self._reference_dimer_converged / steps) if steps else 0.0
            ),
            "reference_dimer_mean_curvature": (
                float(self._reference_dimer_curvature_sum / steps) if steps else 0.0
            ),
            "reference_dimer_mean_true_curvature": (
                float(self._reference_dimer_true_curvature_sum / steps) if steps else 0.0
            ),
            "reference_dimer_mean_biased_curvature": (
                float(self._reference_dimer_biased_curvature_sum / steps) if steps else 0.0
            ),
            "reference_dimer_mean_abs_dot_initial": (
                float(self._reference_dimer_abs_dot_sum / steps) if steps else 0.0
            ),
        }

    def _direction_stats_summary(self) -> dict[str, StatsValue]:
        mean_pool_size = (
            self._direction_candidate_evaluations / self._direction_choices if self._direction_choices else 0.0
        )
        mean_rigid_overlap = self._direction_rigid_overlap_sum / self._direction_choices if self._direction_choices else 0.0
        mean_post_projection_overlap = (
            self._direction_post_projection_rigid_overlap_sum / self._direction_choices
            if self._direction_choices
            else 0.0
        )
        summary = {
            "direction_choices": self._direction_choices,
            "direction_candidate_evaluations": self._direction_candidate_evaluations,
            "direction_mean_candidate_pool_size": float(mean_pool_size),
            "direction_rigid_body_overlap_mean": float(mean_rigid_overlap),
            "direction_post_projection_rigid_body_overlap_mean": float(mean_post_projection_overlap),
            "step_length_mode": self.config.step_length_mode,
            "step_rms_scope": self.config.step_rms_scope,
            "target_step_rms": float(self.config.target_step_rms),
            "max_step_rms": float(self.config.max_step_rms),
            "step_displacement_rms_mean": float(
                self._step_displacement_rms_sum / self._step_displacement_records
                if self._step_displacement_records
                else 0.0
            ),
            "step_displacement_rms_max": float(self._step_displacement_rms_max),
            "step_displacement_max_atom_mean": float(
                self._step_displacement_max_atom_sum / self._step_displacement_records
                if self._step_displacement_records
                else 0.0
            ),
            "step_displacement_max_atom_max": float(self._step_displacement_max_atom_max),
            "direction_selected_momentum": self._direction_selected[DirectionCandidateKind.MOMENTUM],
            "direction_selected_random": self._direction_selected[DirectionCandidateKind.RANDOM],
            "direction_selected_bond": self._direction_selected[DirectionCandidateKind.BOND],
            "direction_selected_bond_form": self._direction_selected[DirectionCandidateKind.BOND_FORM],
            "direction_selected_bond_break": self._direction_selected[DirectionCandidateKind.BOND_BREAK],
            "direction_selected_anchor": self._direction_selected[DirectionCandidateKind.ANCHOR],
            "direction_selected_ritz": self._direction_selected[DirectionCandidateKind.RITZ],
            "direction_selected_ritz_reg": self._direction_selected[DirectionCandidateKind.RITZ_REG],
            "direction_selected_evolved": self._direction_selected[DirectionCandidateKind.EVOLVED],
            "direction_selected_archive_momentum": self._direction_selected[DirectionCandidateKind.ARCHIVE_MOMENTUM],
            "direction_selected_reference_dimer": self._direction_selected[DirectionCandidateKind.REFERENCE_DIMER],
            "plateau_evolution_enabled": int(self.config.plateau_evolution_enabled),
            "plateau_evolution_active_steps": self._plateau_evolution_active_steps,
            "plateau_evolution_candidate_steps": self._plateau_evolution_candidate_steps,
            "plateau_evolution_candidates_generated": self._plateau_evolution_candidates_generated,
            "archive_escape_momentum_enabled": int(self.config.archive_escape_momentum_enabled),
            "archive_escape_momentum_candidate_steps": self._archive_escape_momentum_candidate_steps,
            "archive_escape_momentum_candidates_generated": self._archive_escape_momentum_candidates_generated,
            "direction_bond_pairs_requested": self._direction_bond_pairs_requested,
            "direction_bond_pairs_generated": self._direction_bond_pairs_generated,
            "direction_fallback_bond_pairs_generated": self._direction_fallback_bond_pairs_generated,
            "direction_bond_candidates_valid": self._direction_bond_candidates_valid,
            "walk_displacement_clips": self._walk_displacement_clips,
            "walk_early_stops": self._walk_early_stops,
            "fragment_rejections": self._fragment_rejections,
            "seed_diversity_reseeds": self._seed_diversity_reseeds,
        }
        summary["direction_type_ucb_enabled"] = int(self.config.direction_type_ucb_enabled)
        for kind in DirectionCandidateKind:
            summary[f"direction_type_selected_{kind.value}"] = self.direction_type_memory.selected_counts.get(kind, 0)
            summary[f"direction_type_productive_{kind.value}"] = self.direction_type_memory.productive_counts.get(kind, 0)
            self._add_curvature_summary(
                summary,
                f"direction_curvature_{kind.value}",
                self._direction_curvature_stats[kind],
            )
            self._add_curvature_summary(
                summary,
                f"direction_true_curvature_{kind.value}",
                self._direction_true_curvature_stats[kind],
            )
            self._add_curvature_summary(
                summary,
                f"direction_inner_curvature_{kind.value}",
                self._direction_inner_curvature_stats[kind],
            )
        return summary

    @staticmethod
    def _add_curvature_summary(
        summary: dict[str, StatsValue],
        prefix: str,
        stats: dict[str, float | int | None],
    ) -> None:
        count = int(stats["count"])
        summary[f"{prefix}_count"] = count
        summary[f"{prefix}_mean"] = float(float(stats["sum"]) / count) if count else 0.0
        summary[f"{prefix}_min"] = float(stats["min"]) if stats["min"] is not None else 0.0
        summary[f"{prefix}_max"] = float(stats["max"]) if stats["max"] is not None else 0.0

    def _relax_stats_summary(self) -> dict[str, float | int]:
        summary: dict[str, float | int] = {}
        for label, stats in self._relax_stats.items():
            count = int(stats["count"])
            summary[f"{label}_count"] = count
            summary[f"{label}_unconverged"] = int(stats["unconverged"])
            summary[f"{label}_max_gradient"] = float(stats["max_gradient"])
            summary[f"{label}_mean_iterations"] = float(stats["n_iter_sum"] / count) if count else 0.0
            n_iter_values = np.asarray(stats["n_iter_values"], dtype=float)
            summary[f"{label}_min_iterations"] = float(np.min(n_iter_values)) if count else 0.0
            summary[f"{label}_median_iterations"] = float(np.median(n_iter_values)) if count else 0.0
            summary[f"{label}_p90_iterations"] = float(np.percentile(n_iter_values, 90)) if count else 0.0
            summary[f"{label}_max_iterations"] = float(np.max(n_iter_values)) if count else 0.0
            summary[f"{label}_active_bound_fraction_mean"] = (
                float(stats["bound_fraction_sum"] / count) if count else 0.0
            )
            summary[f"{label}_active_bound_fraction_max"] = float(stats["max_bound_fraction"])
            summary[f"{label}_displacement_rms_mean"] = (
                float(stats["displacement_rms_sum"] / count) if count else 0.0
            )
            summary[f"{label}_displacement_max"] = float(stats["max_displacement"])
            for outcome in RelaxOutcomeClass:
                outcome_count = int(stats["outcome_counts"][outcome.value])
                summary[f"{label}_outcome_{outcome.value}"] = outcome_count
                summary[f"{label}_outcome_{outcome.value}_rate"] = (
                    float(outcome_count / count) if count else 0.0
                )
        summary["bias_steps"] = self._bias_steps
        summary["bias_zero_weight_steps"] = self._bias_zero_steps
        summary["bias_zero_weight_fraction"] = (
            float(self._bias_zero_steps / self._bias_steps) if self._bias_steps else 0.0
        )
        summary["bias_weight_mean"] = float(self._bias_weight_sum / self._bias_steps) if self._bias_steps else 0.0
        summary["bias_weight_max"] = float(self._bias_weight_max)
        return summary

    def _direct_qp_stats_summary(self) -> dict[str, float | int]:
        count = self._direct_qp_steps
        return {
            "direct_qp_steps": count,
            "direct_qp_rejected": self._direct_qp_rejected,
            "direct_qp_mean_step_norm": float(self._direct_qp_step_norm_sum / count) if count else 0.0,
            "direct_qp_mean_progress": float(self._direct_qp_progress_sum / count) if count else 0.0,
            "direct_qp_mean_target_error": float(self._direct_qp_target_error_sum / count) if count else 0.0,
            "direct_qp_mean_model_error": float(self._direct_qp_model_error_sum / count) if count else 0.0,
            "direct_qp_trust_shrink_steps": self._direct_qp_shrink_steps,
            "direct_qp_trust_expand_steps": self._direct_qp_expand_steps,
            "direct_qp_gamma_mean": float(self._direct_qp_gamma_sum / count) if count else 0.0,
            "direct_qp_kappa_mean": float(self._direct_qp_kappa_sum / count) if count else 0.0,
            "direct_qp_high_model_error_streak_max": self._direct_qp_high_model_error_streak_max,
            "direct_qp_micro_count": self._direct_qp_micro_count,
            "direct_qp_micro_mean_iterations": (
                float(self._direct_qp_micro_iteration_sum / self._direct_qp_micro_count)
                if self._direct_qp_micro_count
                else 0.0
            ),
            "direct_qp_micro_displacement_rms_mean": (
                float(self._direct_qp_micro_displacement_rms_sum / self._direct_qp_micro_count)
                if self._direct_qp_micro_count
                else 0.0
            ),
            "direct_qp_micro_displacement_max": float(self._direct_qp_micro_displacement_max),
        }

    def _build_softening(self, seed_state: State, direction: np.ndarray | None = None) -> LocalSofteningModel | None:
        if not self.softening_enabled or not isinstance(self.config, LSSSWConfig):
            self._local_softening_terms_last = 0
            return None
        if self.config.local_softening_mode == "manual" and not self.config.local_softening_pairs:
            self._local_softening_terms_last = 0
            return None
        softening = LocalSofteningModel.from_state(
            seed_state,
            pairs=self.config.local_softening_pairs,
            strength=self.config.local_softening_strength,
            mode=self.config.local_softening_mode,
            cutoff_scale=self.config.local_softening_cutoff_scale,
            active_indices=self._softening_active_indices(seed_state, direction),
            penalty=self.config.local_softening_penalty,
            xi=self.config.local_softening_xi,
            cutoff=self.config.local_softening_cutoff,
            adaptive_strength=self.config.local_softening_adaptive_strength,
            max_strength_scale=self.config.local_softening_max_strength_scale,
            deviation_scale=self.config.local_softening_deviation_scale,
        )
        self._local_softening_terms_last = len(softening.terms)
        if self._local_softening_terms_last == 0:
            return None
        self._local_softening_builds += 1
        self._local_softening_terms_built_total += self._local_softening_terms_last
        self._local_softening_terms_total = self._local_softening_terms_built_total
        return softening

    def _softening_active_indices(self, seed_state: State, direction: np.ndarray | None = None) -> np.ndarray | None:
        if not isinstance(self.config, LSSSWConfig) or self.config.local_softening_mode != "active_neighbors":
            return None
        movable_indices = np.where(seed_state.movable_mask)[0]
        active_count = self.config.local_softening_active_count
        if active_count is None:
            return movable_indices
        if direction is None:
            return movable_indices[:active_count]
        displacement = np.asarray(direction, dtype=float).reshape(seed_state.n_atoms, 3)
        scores = np.linalg.norm(displacement, axis=1)
        movable_scores = scores[movable_indices]
        selected_positions = np.argsort(-movable_scores, kind="stable")[:active_count]
        selected = movable_indices[selected_positions]
        return np.sort(selected)

    @staticmethod
    def _clip_walk_displacement(reference: State, candidate: State, max_displacement: float) -> tuple[State, bool]:
        displacement = mic_displacement(candidate.positions, reference.positions, reference.cell, reference.pbc)
        norms = np.linalg.norm(displacement, axis=1)
        movable = candidate.movable_mask
        clipped = movable & (norms > max_displacement)
        if not np.any(clipped):
            return candidate, False
        scale = np.ones(candidate.n_atoms, dtype=float)
        scale[clipped] = max_displacement / (norms[clipped] + 1e-12)
        positions = reference.positions + displacement * scale[:, None]
        if np.any(candidate.fixed_mask):
            positions[candidate.fixed_mask] = reference.positions[candidate.fixed_mask]
        positions = wrap_positions(positions, candidate.cell, candidate.pbc)
        return (
            State(
                numbers=candidate.numbers.copy(),
                positions=positions,
                cell=None if candidate.cell is None else candidate.cell.copy(),
                pbc=candidate.pbc,
                fixed_mask=candidate.fixed_mask.copy(),
                metadata=candidate.metadata.copy(),
            ),
            True,
        )

    @staticmethod
    def _solve_direct_qp_scalar_step(
        *,
        gradient: np.ndarray,
        direction: np.ndarray,
        sigma: float,
        gamma: float,
        kappa: float,
        trust_radius: float,
    ) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        direction = np.asarray(direction, dtype=float).reshape(-1)
        if gradient.shape != direction.shape:
            raise ValueError("gradient and direction must have the same shape")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            return np.zeros_like(gradient)
        direction = direction / direction_norm
        denominator = float(gamma) + float(kappa)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError("direct-QP scalar denominator must be positive and finite")
        radius = float(trust_radius)
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("direct-QP trust radius must be positive and finite")
        step = -(gradient - float(kappa) * float(sigma) * direction) / denominator
        step_norm = float(np.linalg.norm(step))
        if step_norm > radius and step_norm > 1e-12:
            step = step * (radius / step_norm)
        return step

    @staticmethod
    def _solve_direct_qp_rank1_step(
        *,
        gradient: np.ndarray,
        direction: np.ndarray,
        sigma: float,
        gamma_floor: float,
        directional_curvature: float,
        kappa: float,
        trust_radius: float,
    ) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        direction = np.asarray(direction, dtype=float).reshape(-1)
        if gradient.shape != direction.shape:
            raise ValueError("gradient and direction must have the same shape")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            return np.zeros_like(gradient)
        direction = direction / direction_norm
        gamma_floor = float(gamma_floor)
        directional_curvature = float(directional_curvature)
        if not np.isfinite(directional_curvature):
            directional_curvature = gamma_floor
        directional_curvature = max(directional_curvature, gamma_floor)
        kappa = float(kappa)
        parallel_denominator = directional_curvature + kappa
        perpendicular_denominator = gamma_floor + kappa
        if (
            parallel_denominator <= 0.0
            or perpendicular_denominator <= 0.0
            or not np.isfinite(parallel_denominator)
            or not np.isfinite(perpendicular_denominator)
        ):
            raise ValueError("direct-QP rank1 denominators must be positive and finite")
        radius = float(trust_radius)
        if radius <= 0.0 or not np.isfinite(radius):
            raise ValueError("direct-QP trust radius must be positive and finite")
        g_parallel = float(np.dot(gradient, direction))
        gradient_parallel = g_parallel * direction
        gradient_perpendicular = gradient - gradient_parallel
        step = -gradient_perpendicular / perpendicular_denominator
        step += -((g_parallel - kappa * float(sigma)) / parallel_denominator) * direction
        step_norm = float(np.linalg.norm(step))
        if step_norm > radius and step_norm > 1e-12:
            step = step * (radius / step_norm)
        return step

    def _direct_qp_initial_trust_radius(self) -> float:
        radius = self.config.proposal_trust_radius
        if radius is None:
            radius = self.config.walk_trust_radius
        return max(float(radius), float(self.config.direct_qp_min_trust_radius))

    def _direct_qp_trust_action(self, *, model_error: float, progress: float, sigma: float) -> str:
        min_progress = self.config.direct_qp_accept_min_progress_fraction * max(float(sigma), 1e-12)
        if model_error > self.config.direct_qp_accept_model_error or progress < min_progress:
            return "shrink"
        return "expand"

    def _direct_qp_scalar_gamma(self, curvature: float) -> float:
        if not np.isfinite(curvature):
            return float(self.config.direct_qp_gamma)
        return float(max(curvature, self.config.direct_qp_gamma))

    def _direct_qp_rank1_gamma_floor(self) -> float:
        floor = float(self.config.direct_qp_gamma)
        if self.config.direct_qp_gamma_mode == "constant":
            return floor
        if (
            self.config.direct_qp_gamma_mode == "model_error_gated_history"
            and self._direct_qp_high_model_error_streak < int(self.config.direct_qp_gamma_model_error_streak)
        ):
            return floor
        history = np.asarray(self._direct_qp_curvature_history, dtype=float)
        history = history[np.isfinite(history) & (history > 0.0)]
        if history.size < int(self.config.direct_qp_gamma_history_min_samples):
            return floor
        quantile = float(np.quantile(history, float(self.config.direct_qp_gamma_history_quantile)))
        if not np.isfinite(quantile):
            return floor
        return float(max(floor, quantile))

    def _record_direct_qp_curvature(self, curvature: float) -> None:
        if not np.isfinite(curvature) or curvature <= 0.0:
            return
        self._direct_qp_curvature_history.append(float(curvature))
        maxlen = int(self.config.direct_qp_gamma_history_maxlen)
        if len(self._direct_qp_curvature_history) > maxlen:
            del self._direct_qp_curvature_history[: len(self._direct_qp_curvature_history) - maxlen]

    def _direct_qp_scalar_kappa(self, gamma: float) -> float:
        kappa = float(self.config.direct_qp_kappa)
        if self.config.direct_qp_kappa_mode == "adaptive_curvature" and np.isfinite(gamma):
            kappa = max(kappa, float(self.config.direct_qp_kappa_curvature_ratio) * float(gamma))
            kappa = min(kappa, float(self.config.direct_qp_kappa_max))
        return kappa

    def _direct_qp_micro_steps_for_model_error(self, model_error: float) -> int:
        if self.config.direct_qp_micro_mode == "off":
            return 0
        base_steps = int(self.config.direct_qp_micro_steps)
        max_steps = int(self.config.direct_qp_micro_max_steps)
        if self.config.direct_qp_micro_mode == "always":
            return base_steps
        if not np.isfinite(model_error):
            model_error = float("inf")
        threshold = float(self.config.direct_qp_micro_model_error_threshold)
        if self.config.direct_qp_micro_mode == "model_error":
            return max_steps if model_error > threshold else 0
        if model_error <= threshold:
            return base_steps
        high = float(self.config.direct_qp_micro_model_error_high)
        if high <= threshold:
            return max_steps
        fraction = min(1.0, max(0.0, (float(model_error) - threshold) / (high - threshold)))
        return int(round(base_steps + fraction * (max_steps - base_steps)))

    def _direct_qp_micro_correct(self, state: State, *, model_error: float) -> RelaxResult | None:
        maxiter = self._direct_qp_micro_steps_for_model_error(model_error)
        if maxiter <= 0:
            return None
        return Relaxer(self.calculator.evaluate_flat, optimizer=self.config.direct_qp_micro_optimizer).relax(
            state,
            fmax=self.config.direct_qp_micro_fmax,
            maxiter=maxiter,
            coordinate_trust_radius=self.config.direct_qp_micro_trust_radius,
        )

    def _execute_direct_qp_step(
        self,
        *,
        current: State,
        choice: DirectionChoice,
        sigma: float,
        true_before,
        trust_radius: float,
        gamma: float,
        kappa: float,
        directional_curvature: float | None = None,
    ) -> DirectQPStepResult | None:
        gradient = true_before.gradient.reshape(-1)
        if self.config.direct_qp_hessian == "rank1":
            step = self._solve_direct_qp_rank1_step(
                gradient=gradient,
                direction=choice.direction,
                sigma=sigma,
                gamma_floor=gamma,
                directional_curvature=gamma if directional_curvature is None else directional_curvature,
                kappa=kappa,
                trust_radius=trust_radius,
            )
        else:
            step = self._solve_direct_qp_scalar_step(
                gradient=gradient,
                direction=choice.direction,
                sigma=sigma,
                gamma=gamma,
                kappa=kappa,
                trust_radius=trust_radius,
            )
        direction = np.asarray(choice.direction, dtype=float).reshape(-1)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm > 1e-12:
            direction = direction / direction_norm
        progress = float(np.dot(direction, step))
        target_error = float(np.linalg.norm(step - float(sigma) * direction))

        def record_reject() -> None:
            self._record_direct_qp_result(
                step_norm=float(np.linalg.norm(step)),
                progress=progress,
                target_error=target_error,
                predicted_delta=0.0,
                true_delta=0.0,
                model_error=0.0,
                gamma=gamma,
                kappa=kappa,
                action="reject",
                rejected=True,
            )

        candidate = CartesianCoordinates.from_state(current).displace(TangentVector(step), 1.0)
        if not self.geometry_validator.is_valid_state(candidate):
            record_reject()
            return None
        after = self.calculator.evaluate(candidate)
        if not np.isfinite(after.energy):
            record_reject()
            return None
        predicted_delta = float(gradient @ step + 0.5 * float(gamma) * (step @ step))
        true_delta = float(after.energy - true_before.energy)
        model_error = float(abs(true_delta - predicted_delta) / max(abs(predicted_delta), 1e-8))
        action = self._direct_qp_trust_action(model_error=model_error, progress=progress, sigma=sigma)
        self._record_direct_qp_result(
            step_norm=float(np.linalg.norm(step)),
            progress=progress,
            target_error=target_error,
            predicted_delta=predicted_delta,
            true_delta=true_delta,
            model_error=model_error,
            gamma=gamma,
            kappa=kappa,
            action=action,
            rejected=False,
        )
        return DirectQPStepResult(
            state=candidate,
            step=step,
            energy_before=float(true_before.energy),
            energy_after=float(after.energy),
            predicted_delta=predicted_delta,
            true_delta=true_delta,
            model_error=model_error,
            progress=progress,
            target_error=target_error,
            gamma=float(gamma),
            kappa=float(kappa),
            action=action,
            rejected=False,
        )

    def _is_fragmented_cluster(self, reference: State, candidate: State) -> bool:
        if self.config.fragment_guard_factor is None:
            return False
        if all(reference.pbc) or all(candidate.pbc) or candidate.n_atoms < 2:
            return False
        reference_scale = self._max_nearest_neighbor_distance(reference)
        candidate_scale = self._max_nearest_neighbor_distance(candidate)
        if reference_scale <= 1e-12:
            return False
        candidate_median = self._median_nearest_neighbor_distance(candidate)
        parent_fragmented = candidate_scale > self.config.fragment_guard_factor * reference_scale
        self_fragmented = (
            candidate_median > 1e-12
            and candidate_scale > self.config.fragment_guard_factor * candidate_median
        )
        return bool(parent_fragmented or self_fragmented)

    @staticmethod
    def _max_nearest_neighbor_distance(state: State) -> float:
        nearest = SurfaceWalker._nearest_neighbor_distances(state)
        return float(np.max(nearest, initial=0.0))

    @staticmethod
    def _median_nearest_neighbor_distance(state: State) -> float:
        nearest = SurfaceWalker._nearest_neighbor_distances(state)
        return float(np.median(nearest)) if nearest.size else 0.0

    @staticmethod
    def _nearest_neighbor_distances(state: State) -> np.ndarray:
        distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
        np.fill_diagonal(distances, np.inf)
        return np.min(distances, axis=1)
