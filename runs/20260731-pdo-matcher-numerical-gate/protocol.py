"""Pure decision protocol for the PdO matcher/numerical ambiguity gate."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def decompose_pair(
    *,
    energy_delta_eV: float,
    energy_tol_eV: float,
    indexed_mic_rmsd_A: float,
    rmsd_tol_A: float,
    descriptor_delta: float,
    descriptor_tol: float,
) -> dict[str, Any]:
    """Decompose the current archive and descriptor decisions."""

    energy_same = abs(float(energy_delta_eV)) <= float(energy_tol_eV)
    indexed_geometry_same = (
        float(indexed_mic_rmsd_A) <= float(rmsd_tol_A)
    )
    descriptor_same = float(descriptor_delta) < float(descriptor_tol)
    archive_same = energy_same and indexed_geometry_same
    if archive_same == descriptor_same:
        mechanism = "agreement"
    elif (
        not energy_same
        and indexed_geometry_same
        and descriptor_same
    ):
        mechanism = "energy_only_archive_split"
    elif not indexed_geometry_same and descriptor_same:
        mechanism = "descriptor_collision_geometry_split"
    elif archive_same and not descriptor_same:
        mechanism = "descriptor_split_archive_merge"
    else:
        mechanism = "other_disagreement"
    return {
        "energy_same": energy_same,
        "indexed_geometry_same": indexed_geometry_same,
        "descriptor_same": descriptor_same,
        "archive_same": archive_same,
        "mechanism": mechanism,
    }


def decompose_local_region(
    *,
    movable_indexed_mic_rmsd_A: float,
    rmsd_tol_A: float,
) -> dict[str, Any]:
    """Resolve global-RMSD dilution without introducing another threshold."""

    movable_geometry_same = (
        float(movable_indexed_mic_rmsd_A) <= float(rmsd_tol_A)
    )
    return {
        "movable_geometry_same": movable_geometry_same,
        "local_region_mechanism": (
            "energy_only_local_same"
            if movable_geometry_same
            else "local_event_global_dilution"
        ),
    }


def evaluate_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one ambiguous pair is required")
    integrity = all(
        bool(row["certificate"])
        and bool(row["geometry_valid"])
        and not bool(row["fragmented"])
        for row in rows
    )
    mechanisms = Counter(str(row["mechanism"]) for row in rows)
    uniform_geometry_split = all(
        not bool(row["indexed_geometry_same"])
        and bool(row["descriptor_same"])
        for row in rows
    )
    uniform_energy_only = all(
        not bool(row["energy_same"])
        and bool(row["indexed_geometry_same"])
        and bool(row["descriptor_same"])
        for row in rows
    )
    if integrity and uniform_geometry_split:
        classification = "descriptor_collision"
    elif integrity and uniform_energy_only:
        classification = "energy_only_archive_split"
    else:
        classification = "mixed_unresolved"
    return {
        "classification": classification,
        "pair_count": len(rows),
        "mechanism_counts": dict(sorted(mechanisms.items())),
        "integrity_pass": integrity,
        "relabel_as_escaped_allowed": (
            classification == "descriptor_collision"
        ),
        "strict_requench_gate_required": (
            classification == "energy_only_archive_split"
        ),
        "production_matcher_change_allowed": False,
    }


def evaluate_local_region_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered movable-atom subgate to residual pairs only."""

    decision = dict(initial_gate)
    residual = [
        row
        for row in rows
        if str(row["mechanism"]) == "energy_only_archive_split"
    ]
    decision["local_region_pair_count"] = len(residual)
    decision["strict_requench_pair_count"] = 0
    if str(initial_gate["classification"]) != "mixed_unresolved" or not residual:
        return decision

    integrity = all(
        bool(row["certificate"])
        and bool(row["geometry_valid"])
        and not bool(row["fragmented"])
        for row in residual
    )
    all_local_split = all(
        not bool(row["movable_geometry_same"]) for row in residual
    )
    all_local_same = all(
        bool(row["movable_geometry_same"]) for row in residual
    )
    if integrity and all_local_split:
        decision.update(
            {
                "classification": (
                    "descriptor_collision_with_local_dilution"
                ),
                "relabel_as_escaped_allowed": True,
                "strict_requench_gate_required": False,
            }
        )
    elif integrity and all_local_same:
        decision.update(
            {
                "classification": "residual_energy_only_local_same",
                "relabel_as_escaped_allowed": False,
                "strict_requench_gate_required": True,
                "strict_requench_pair_count": len(residual),
            }
        )
    return decision


def evaluate_strict_requench(
    *,
    starter_converged: bool,
    landing_converged: bool,
    strict_energy_delta_eV: float,
    energy_tol_eV: float,
    strict_indexed_mic_rmsd_A: float,
    rmsd_tol_A: float,
) -> dict[str, Any]:
    """Classify the residual pair after independent strict true-PES quenches."""

    strict_certificate = bool(starter_converged) and bool(landing_converged)
    energy_same = abs(float(strict_energy_delta_eV)) <= float(energy_tol_eV)
    geometry_same = (
        float(strict_indexed_mic_rmsd_A) <= float(rmsd_tol_A)
    )
    archive_same = energy_same and geometry_same
    if not strict_certificate:
        classification = "strict_requench_unresolved"
    elif archive_same:
        classification = "return_starter_after_strict_requench"
    else:
        classification = "escaped_certified_after_strict_requench"
    return {
        "classification": classification,
        "strict_certificate": strict_certificate,
        "energy_same": energy_same,
        "indexed_geometry_same": geometry_same,
        "archive_same": archive_same,
        "offline_label": (
            "RETURN_STARTER"
            if classification == "return_starter_after_strict_requench"
            else (
                "ESCAPED_CERTIFIED"
                if classification
                == "escaped_certified_after_strict_requench"
                else "AMBIGUOUS_MATCH"
            )
        ),
        "production_matcher_change_allowed": False,
    }
