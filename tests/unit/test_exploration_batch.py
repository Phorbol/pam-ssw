from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import pamssw.exploration as exploration
from pamssw.exploration.actions import PolicySnapshot
from pamssw.exploration.batch import derive_action_seed, plan_batch
from pamssw.exploration.policies import build_policy_snapshot
from pamssw.exploration.posterior import StarterProductivityPosterior


def _snapshot(
    *,
    starter_ids: tuple[int, ...] = (3, 8),
    probabilities: tuple[float, ...] = (0.2, 0.8),
) -> PolicySnapshot:
    return PolicySnapshot(
        version=7,
        archive_version=11,
        policy_name="posterior_proportional",
        eligible_starter_ids=starter_ids,
        probabilities=probabilities,
        support_complete=all(probability > 0.0 for probability in probabilities),
    )


def test_derive_action_seed_matches_its_stable_uint32_seed_sequence():
    expected = int(
        np.random.SeedSequence([19, 4, 2, 0x535357]).generate_state(1, dtype=np.uint32)[0]
    )

    seed = derive_action_seed(19, 4, 2)

    assert seed == expected
    assert isinstance(seed, int)
    assert 0 <= seed <= np.iinfo(np.uint32).max


@pytest.mark.parametrize(
    ("master_seed", "batch_id", "slot_id", "message"),
    [
        (True, 0, 0, "master_seed"),
        (0, False, 0, "batch_id"),
        (0, 0, True, "slot_id"),
        (-1, 0, 0, "master_seed"),
        (0, -1, 0, "batch_id"),
        (0, 0, -1, "slot_id"),
        (0.0, 0, 0, "master_seed"),
    ],
)
def test_derive_action_seed_rejects_nonnegative_integer_contract_violations(
    master_seed, batch_id, slot_id, message
):
    with pytest.raises(ValueError, match=message):
        derive_action_seed(master_seed, batch_id, slot_id)


def test_plan_batch_is_reproducible_and_copies_snapshot_metadata_exactly():
    snapshot = _snapshot()

    first = plan_batch(snapshot, batch_id=4, batch_size=5, master_seed=19, force_budget=12)
    second = plan_batch(snapshot, batch_id=4, batch_size=5, master_seed=19, force_budget=12)

    assert exploration.plan_batch is plan_batch
    assert exploration.derive_action_seed is derive_action_seed
    assert isinstance(first, tuple)
    assert first == second
    assert [action.action_id for action in first] == [
        "batch-00000004-slot-0000",
        "batch-00000004-slot-0001",
        "batch-00000004-slot-0002",
        "batch-00000004-slot-0003",
        "batch-00000004-slot-0004",
    ]
    assert [action.slot_id for action in first] == list(range(5))
    for action in first:
        assert action.batch_id == 4
        assert action.policy_name == snapshot.policy_name
        assert action.policy_version == snapshot.version
        assert action.archive_version == snapshot.archive_version
        assert action.selection_probability == snapshot.probability_for(action.starter_id)
        assert action.random_seed == derive_action_seed(19, 4, action.slot_id)
        assert action.force_budget == 12
        with pytest.raises(FrozenInstanceError):
            action.slot_id = 9


def test_plan_batch_uses_the_snapshot_probability_vector_without_adjustment():
    snapshot = _snapshot(starter_ids=(2, 5, 9), probabilities=(0.1, 0.3, 0.6))
    expected_starters = np.random.default_rng(
        np.random.SeedSequence([17, 3, 0x42415443])
    ).choice(
        snapshot.eligible_starter_ids,
        size=12,
        replace=True,
        p=snapshot.probabilities,
    )

    actions = plan_batch(snapshot, batch_id=3, batch_size=12, master_seed=17, force_budget=None)

    assert tuple(action.starter_id for action in actions) == tuple(expected_starters)
    assert tuple(action.selection_probability for action in actions) == tuple(
        snapshot.probability_for(int(starter_id)) for starter_id in expected_starters
    )


def test_plan_batch_samples_with_replacement_when_batch_exceeds_eligible_starters():
    snapshot = _snapshot(starter_ids=(6,), probabilities=(1.0,))

    actions = plan_batch(snapshot, batch_id=2, batch_size=4, master_seed=5, force_budget=1)

    assert len(actions) == 4
    assert tuple(action.starter_id for action in actions) == (6, 6, 6, 6)


def test_plan_batch_assigns_distinct_slot_specific_seeds():
    snapshot = _snapshot()

    actions = plan_batch(snapshot, batch_id=9, batch_size=16, master_seed=23, force_budget=None)

    assert len({action.action_id for action in actions}) == 16
    assert len({action.random_seed for action in actions}) == 16
    assert tuple(action.random_seed for action in actions) == tuple(
        derive_action_seed(23, 9, slot_id) for slot_id in range(16)
    )


@pytest.mark.parametrize(
    ("snapshot", "batch_id", "batch_size", "master_seed", "force_budget", "message"),
    [
        (object(), 0, 1, 0, None, "snapshot"),
        (_snapshot(), True, 1, 0, None, "batch_id"),
        (_snapshot(), -1, 1, 0, None, "batch_id"),
        (_snapshot(), 0, True, 0, None, "batch_size"),
        (_snapshot(), 0, 0, 0, None, "batch_size"),
        (_snapshot(), 0, 1, False, None, "master_seed"),
        (_snapshot(), 0, 1, -1, None, "master_seed"),
        (_snapshot(), 0, 1, 0, True, "force_budget"),
        (_snapshot(), 0, 1, 0, 0, "force_budget"),
        (_snapshot(), 0, 1, 0, -1, "force_budget"),
    ],
)
def test_plan_batch_validates_all_inputs_before_planning(
    snapshot, batch_id, batch_size, master_seed, force_budget, message
):
    with pytest.raises(ValueError, match=message):
        plan_batch(snapshot, batch_id, batch_size, master_seed, force_budget)


def test_plan_batch_does_not_mutate_the_snapshot():
    snapshot = _snapshot()
    before = snapshot

    plan_batch(snapshot, batch_id=2, batch_size=3, master_seed=7, force_budget=None)

    assert snapshot == before
    assert snapshot.eligible_starter_ids == (3, 8)
    assert snapshot.probabilities == (0.2, 0.8)


def test_plan_batch_accepts_a_minimal_ucb_one_hot_snapshot():
    posterior = StarterProductivityPosterior()
    posterior.update(9, discovered=True)
    snapshot = build_policy_snapshot(
        "minimal_ucb", (9, 7, 2), posterior, version=1, archive_version=2
    )

    actions = plan_batch(snapshot, batch_id=1, batch_size=5, master_seed=3, force_budget=None)

    assert snapshot.probabilities == (1.0, 0.0, 0.0)
    assert tuple(action.starter_id for action in actions) == (2, 2, 2, 2, 2)
    assert tuple(action.selection_probability for action in actions) == (1.0,) * 5


def test_plan_batch_changes_batch_identity_and_master_seed_streams():
    snapshot = _snapshot()

    baseline = plan_batch(snapshot, batch_id=4, batch_size=4, master_seed=19, force_budget=None)
    changed_batch = plan_batch(snapshot, batch_id=5, batch_size=4, master_seed=19, force_budget=None)
    changed_seed = plan_batch(snapshot, batch_id=4, batch_size=4, master_seed=20, force_budget=None)

    assert baseline != changed_batch
    assert tuple(action.random_seed for action in baseline) != tuple(
        action.random_seed for action in changed_seed
    )


def test_plan_batch_is_independent_from_batch_scheduling_and_interleaving_order():
    snapshot = _snapshot()

    a_then_b_a = plan_batch(snapshot, batch_id=4, batch_size=6, master_seed=19, force_budget=8)
    a_then_b_b = plan_batch(snapshot, batch_id=12, batch_size=6, master_seed=19, force_budget=8)
    b_then_a_b = plan_batch(snapshot, batch_id=12, batch_size=6, master_seed=19, force_budget=8)
    b_then_a_a = plan_batch(snapshot, batch_id=4, batch_size=6, master_seed=19, force_budget=8)
    plan_batch(snapshot, batch_id=99, batch_size=3, master_seed=19, force_budget=8)
    interleaved_a = plan_batch(snapshot, batch_id=4, batch_size=6, master_seed=19, force_budget=8)

    assert a_then_b_a == b_then_a_a == interleaved_a
    assert a_then_b_b == b_then_a_b
