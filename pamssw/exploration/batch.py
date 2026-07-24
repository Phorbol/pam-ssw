"""Deterministic, side-effect-free starter action planning for one batch.

Regenerating a batch is tied to the explicit PCG64 construction below. Durable
replay uses recorded ``StarterAction`` values rather than regeneration.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .actions import PolicySnapshot, StarterAction


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _cantor_pair(left: int, right: int) -> int:
    """Encode two nonnegative integers injectively with the Cantor pairing function."""
    total = left + right
    return total * (total + 1) // 2 + right


def derive_action_seed(master_seed: int, batch_id: int, slot_id: int) -> int:
    """Derive an injective nonnegative seed from one action identity.

    This returns ``pi(pi(master_seed, batch_id), slot_id)``, where
    ``pi(a, b) = (a + b) * (a + b + 1) // 2 + b`` is Cantor pairing.
    """
    master_seed = _nonnegative_int("master_seed", master_seed)
    batch_id = _nonnegative_int("batch_id", batch_id)
    slot_id = _nonnegative_int("slot_id", slot_id)
    return _cantor_pair(_cantor_pair(master_seed, batch_id), slot_id)


def plan_batch(
    snapshot: PolicySnapshot,
    batch_id: int,
    batch_size: int,
    master_seed: int,
    force_budget: int | None,
) -> tuple[StarterAction, ...]:
    """Plan one reproducible batch from an immutable policy snapshot.

    Regeneration is pinned to the explicit PCG64 generator below; durable
    replay consumes the recorded action metadata instead.
    """
    if not isinstance(snapshot, PolicySnapshot):
        raise ValueError("snapshot must be a PolicySnapshot")
    batch_id = _nonnegative_int("batch_id", batch_id)
    batch_size = _positive_int("batch_size", batch_size)
    master_seed = _nonnegative_int("master_seed", master_seed)
    if force_budget is not None:
        force_budget = _positive_int("force_budget", force_budget)

    batch_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([master_seed, batch_id, 0x42415443]))
    )
    selected_starter_ids = batch_rng.choice(
        snapshot.eligible_starter_ids,
        size=batch_size,
        replace=True,
        p=snapshot.probabilities,
    )
    return tuple(
        StarterAction(
            action_id=f"batch-{batch_id:08d}-slot-{slot_id:04d}",
            batch_id=batch_id,
            slot_id=slot_id,
            policy_name=snapshot.policy_name,
            policy_version=snapshot.version,
            archive_version=snapshot.archive_version,
            starter_id=int(starter_id),
            selection_probability=snapshot.probability_for(int(starter_id)),
            random_seed=derive_action_seed(master_seed, batch_id, slot_id),
            force_budget=force_budget,
        )
        for slot_id, starter_id in enumerate(selected_starter_ids)
    )


__all__ = ["derive_action_seed", "plan_batch"]
