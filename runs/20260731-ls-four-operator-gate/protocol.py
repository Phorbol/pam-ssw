"""Pure numerical helpers for the LS four-operator mechanism gate."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import statistics

import numpy as np

from pamssw.pbc import mic_displacement
from pamssw.softening import LocalSofteningModel
from pamssw.state import State


@dataclass(frozen=True)
class PairOperatorAction:
    hvp: np.ndarray
    radial_curvature: float
    transverse_curvature: float
    pair_expressivity_numerator: float


@dataclass(frozen=True)
class RigidAlignment:
    positions_in_reference_frame: np.ndarray
    prestrained_to_reference_rotation: np.ndarray

    def direction_to_prestrained(self, direction: np.ndarray) -> np.ndarray:
        values = np.asarray(direction, dtype=float).reshape(-1, 3)
        return (values @ self.prestrained_to_reference_rotation.T).reshape(-1)

    def direction_to_reference(self, direction: np.ndarray) -> np.ndarray:
        values = np.asarray(direction, dtype=float).reshape(-1, 3)
        return (values @ self.prestrained_to_reference_rotation).reshape(-1)


def align_prestrained_to_reference(
    reference: State,
    prestrained: State,
) -> RigidAlignment:
    """Align an indexed non-periodic prestrained geometry to its reference."""

    if reference.n_atoms != prestrained.n_atoms or not np.array_equal(
        reference.numbers,
        prestrained.numbers,
    ):
        raise ValueError("reference and prestrained states must have identical atoms")
    if any(reference.pbc) or any(prestrained.pbc):
        raise ValueError("the C60 four-operator alignment requires non-periodic states")
    reference_center = np.mean(reference.positions, axis=0)
    prestrained_center = np.mean(prestrained.positions, axis=0)
    reference_centered = reference.positions - reference_center
    prestrained_centered = prestrained.positions - prestrained_center
    left, _, right_transpose = np.linalg.svd(
        prestrained_centered.T @ reference_centered
    )
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_transpose
    aligned = prestrained_centered @ rotation + reference_center
    return RigidAlignment(
        positions_in_reference_frame=aligned,
        prestrained_to_reference_rotation=rotation,
    )


def build_operator_row(
    *,
    direction_x0: np.ndarray,
    direction_xr: np.ndarray,
    true_hvp_x0: np.ndarray,
    true_hvp_xr: np.ndarray,
    ls_action_x0: PairOperatorAction,
    ls_action_xr: PairOperatorAction,
) -> dict[str, float]:
    """Build the A/B/C/D curvature decomposition for one frozen candidate."""

    arrays = {
        "direction_x0": np.asarray(direction_x0, dtype=float),
        "direction_xr": np.asarray(direction_xr, dtype=float),
        "true_hvp_x0": np.asarray(true_hvp_x0, dtype=float),
        "true_hvp_xr": np.asarray(true_hvp_xr, dtype=float),
        "ls_hvp_x0": np.asarray(ls_action_x0.hvp, dtype=float),
        "ls_hvp_xr": np.asarray(ls_action_xr.hvp, dtype=float),
    }
    shapes = {values.shape for values in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes), ()) == ():
        raise ValueError("directions and HVPs must be nonempty arrays of one shape")
    if any(values.ndim != 1 or not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("directions and HVPs must be finite flat arrays")
    norm_x0_sq = float(np.dot(arrays["direction_x0"], arrays["direction_x0"]))
    norm_xr_sq = float(np.dot(arrays["direction_xr"], arrays["direction_xr"]))
    if norm_x0_sq <= 1.0e-30 or norm_xr_sq <= 1.0e-30:
        raise ValueError("directions must have nonzero norm")

    kappa_a = float(np.dot(arrays["direction_x0"], arrays["true_hvp_x0"]))
    operator_x0 = float(np.dot(arrays["direction_x0"], arrays["ls_hvp_x0"]))
    kappa_b = kappa_a + operator_x0
    kappa_c = float(np.dot(arrays["direction_xr"], arrays["true_hvp_xr"]))
    operator_xr = float(np.dot(arrays["direction_xr"], arrays["ls_hvp_xr"]))
    kappa_d = kappa_c + operator_xr
    row = {
        "kappa_a": kappa_a,
        "kappa_b": kappa_b,
        "kappa_c": kappa_c,
        "kappa_d": kappa_d,
        "operator_effect_x0": operator_x0,
        "prestrain_effect": kappa_c - kappa_a,
        "operator_effect_xr": operator_xr,
        "operator_geometry_interaction": operator_xr - operator_x0,
        "ls_radial_curvature_x0": float(ls_action_x0.radial_curvature),
        "ls_transverse_curvature_x0": float(ls_action_x0.transverse_curvature),
        "ls_radial_curvature_xr": float(ls_action_xr.radial_curvature),
        "ls_transverse_curvature_xr": float(ls_action_xr.transverse_curvature),
        "pair_expressivity_x0": float(
            ls_action_x0.pair_expressivity_numerator / norm_x0_sq
        ),
        "pair_expressivity_xr": float(
            ls_action_xr.pair_expressivity_numerator / norm_xr_sq
        ),
        "direction_norm_x0": float(np.sqrt(norm_x0_sq)),
        "direction_norm_xr": float(np.sqrt(norm_xr_sq)),
    }
    if not all(np.isfinite(value) for value in row.values()):
        raise ValueError("operator row contains non-finite values")
    return row


_OPERATOR_ROW_KEYS = (
    "kappa_a",
    "kappa_b",
    "kappa_c",
    "kappa_d",
    "operator_effect_x0",
    "prestrain_effect",
    "operator_effect_xr",
    "operator_geometry_interaction",
    "ls_radial_curvature_x0",
    "ls_transverse_curvature_x0",
    "ls_radial_curvature_xr",
    "ls_transverse_curvature_xr",
    "pair_expressivity_x0",
    "pair_expressivity_xr",
    "direction_norm_x0",
    "direction_norm_xr",
)


def build_evidence(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate and summarize the preregistered nine-block mechanism cohort."""

    expected_states = ("bootstrap", "mid", "late")
    expected_seeds = (42, 43, 44)
    expected_blocks = {
        (state_id, seed) for state_id in expected_states for seed in expected_seeds
    }
    observed_blocks = {
        (str(case["state_id"]), int(case["seed"])) for case in cases
    }
    if len(cases) != len(expected_blocks) or observed_blocks != expected_blocks:
        raise ValueError("four-operator evidence requires the complete nine-block cohort")

    candidate_count: int | None = None
    all_candidate_rows: list[Mapping[str, object]] = []
    selection_changes = {"b_vs_a": 0, "c_vs_a": 0, "d_vs_a": 0}
    total_force_evaluations = 0
    total_unattributed = 0
    for case in cases:
        pool_hash = case.get("candidate_pool_sha256")
        if not isinstance(pool_hash, str) or not pool_hash:
            raise ValueError("each block requires one nonempty candidate-pool hash")
        rows = case.get("candidate_rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("each block requires a nonempty candidate row list")
        if candidate_count is None:
            candidate_count = len(rows)
        elif len(rows) != candidate_count:
            raise ValueError("candidate count drifted between blocks")
        indices = [int(row["candidate_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError("candidate indices must be contiguous and ordered")
        for row in rows:
            for key in _OPERATOR_ROW_KEYS:
                value = float(row[key])
                if not np.isfinite(value):
                    raise ValueError(f"candidate operator field {key} must be finite")
        all_candidate_rows.extend(rows)

        selected = case.get("selected_candidate")
        if not isinstance(selected, Mapping) or set(selected) != {"a", "b", "c", "d"}:
            raise ValueError("selected_candidate must contain A/B/C/D identities")
        selected_indices = {key: int(value) for key, value in selected.items()}
        if any(value < 0 or value >= len(rows) for value in selected_indices.values()):
            raise ValueError("selected candidate identity is outside the frozen pool")
        selection_changes["b_vs_a"] += int(selected_indices["b"] != selected_indices["a"])
        selection_changes["c_vs_a"] += int(selected_indices["c"] != selected_indices["a"])
        selection_changes["d_vs_a"] += int(selected_indices["d"] != selected_indices["a"])

        purpose_counts = case.get("purpose_counts")
        if not isinstance(purpose_counts, Mapping):
            raise ValueError("purpose_counts are required")
        integer_counts = {str(key): int(value) for key, value in purpose_counts.items()}
        force_evaluations = int(case["force_evaluations"])
        if force_evaluations != sum(integer_counts.values()):
            raise ValueError("purpose accounting does not close")
        unattributed = integer_counts.get("unattributed", 0)
        if unattributed != 0:
            raise ValueError("unattributed force evaluations are forbidden")
        total_force_evaluations += force_evaluations
        total_unattributed += unattributed

    effect_keys = (
        "operator_effect_x0",
        "prestrain_effect",
        "operator_effect_xr",
        "operator_geometry_interaction",
        "pair_expressivity_x0",
        "pair_expressivity_xr",
    )
    median_effects = {
        key: float(statistics.median(float(row[key]) for row in all_candidate_rows))
        for key in effect_keys
    }
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(expected_states),
            "seeds": list(expected_seeds),
            "blocks": len(cases),
            "candidates_per_block": int(candidate_count or 0),
        },
        "force_accounting": {
            "total": total_force_evaluations,
            "unattributed": total_unattributed,
        },
        "selection_changes": selection_changes,
        "median_candidate_effects": median_effects,
    }


def pair_operator_action(
    model: LocalSofteningModel,
    state: State,
    direction: np.ndarray,
) -> PairOperatorAction:
    """Apply the analytic Hessian of a non-adaptive pair penalty."""

    if model.adaptive_strength:
        raise ValueError("the four-operator gate requires non-adaptive pair strength")
    vector = np.asarray(direction, dtype=float)
    if vector.shape != (3 * state.n_atoms,) or not np.all(np.isfinite(vector)):
        raise ValueError("direction must be a finite flat Cartesian vector")
    atom_vectors = vector.reshape(state.n_atoms, 3)
    hvp = np.zeros_like(atom_vectors)
    radial_curvature = 0.0
    transverse_curvature = 0.0
    pair_expressivity_numerator = 0.0

    for term in model.terms:
        delta = mic_displacement(
            state.positions[term.atom_j : term.atom_j + 1],
            state.positions[term.atom_i : term.atom_i + 1],
            state.cell,
            state.pbc,
        )[0]
        distance = float(np.linalg.norm(delta))
        if distance <= 1.0e-12:
            continue
        if model.cutoff is not None and distance > term.reference_distance + model.cutoff:
            continue
        deviation = distance - term.reference_distance
        if model.penalty == "gaussian_well":
            energy = term.strength * np.exp(
                -0.5 * (deviation / term.width) ** 2
            )
            first_derivative = -energy * deviation / term.width**2
            second_derivative = energy * (
                deviation**2 / term.width**4 - 1.0 / term.width**2
            )
        else:
            decay_length = (
                model.xi * term.reference_distance
                if model.reference_scaled_xi
                else model.xi
            )
            energy = term.strength * np.exp(-deviation / decay_length)
            first_derivative = -energy / decay_length
            second_derivative = energy / decay_length**2

        axis = delta / distance
        relative = atom_vectors[term.atom_j] - atom_vectors[term.atom_i]
        radial_projection = float(np.dot(axis, relative))
        transverse = relative - radial_projection * axis
        radial_action = second_derivative * radial_projection * axis
        transverse_action = (first_derivative / distance) * transverse
        relative_action = radial_action + transverse_action
        hvp[term.atom_i] -= relative_action
        hvp[term.atom_j] += relative_action
        radial_curvature += second_derivative * radial_projection**2
        transverse_curvature += (
            first_derivative / distance * float(np.dot(transverse, transverse))
        )
        pair_expressivity_numerator += radial_projection**2

    return PairOperatorAction(
        hvp=hvp.reshape(-1),
        radial_curvature=float(radial_curvature),
        transverse_curvature=float(transverse_curvature),
        pair_expressivity_numerator=float(pair_expressivity_numerator),
    )
