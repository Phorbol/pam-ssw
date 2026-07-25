import math

import pytest

import pamssw.exploration as exploration
from pamssw.exploration import StarterProductivityPosterior
from pamssw.exploration.policies import SUPPORTED_POLICIES, build_policy_snapshot


def _record(posterior, starter_id, *, successes, failures):
    for _ in range(successes):
        posterior.update(starter_id, discovered=True)
    for _ in range(failures):
        posterior.update(starter_id, discovered=False)


def test_uniform_has_sorted_complete_support_and_exact_equal_probabilities():
    posterior = StarterProductivityPosterior()

    snapshot = build_policy_snapshot(
        "uniform", (9, 2, 7), posterior, version=4, archive_version=11
    )

    assert exploration.build_policy_snapshot is build_policy_snapshot
    assert set(SUPPORTED_POLICIES) == {"uniform", "posterior_proportional", "minimal_ucb"}
    assert snapshot.policy_name == "uniform"
    assert snapshot.version == 4
    assert snapshot.archive_version == 11
    assert snapshot.eligible_starter_ids == (2, 7, 9)
    assert snapshot.probabilities == (1.0 / 3, 1.0 / 3, 1.0 / 3)
    assert snapshot.support_complete is True


def test_posterior_proportional_normalizes_the_fixed_beta_means():
    posterior = StarterProductivityPosterior()
    _record(posterior, 1, successes=1, failures=0)
    _record(posterior, 3, successes=0, failures=1)

    snapshot = build_policy_snapshot(
        "posterior_proportional", (3, 1), posterior, version=5, archive_version=12
    )

    assert snapshot.eligible_starter_ids == (1, 3)
    assert snapshot.probabilities == (2.0 / 3, 1.0 / 3)
    assert snapshot.support_complete is True


def test_unseen_posterior_proportional_starters_are_uniform_and_registered():
    posterior = StarterProductivityPosterior()

    snapshot = build_policy_snapshot(
        "posterior_proportional", (5, 1, 3), posterior, version=0, archive_version=0
    )

    assert snapshot.eligible_starter_ids == (1, 3, 5)
    assert snapshot.probabilities == (1.0 / 3, 1.0 / 3, 1.0 / 3)
    assert tuple(posterior.counts(starter_id) for starter_id in snapshot.eligible_starter_ids) == (
        (0, 0),
        (0, 0),
        (0, 0),
    )


def test_minimal_ucb_chooses_lowest_untried_starter_with_one_hot_probability():
    posterior = StarterProductivityPosterior()
    _record(posterior, 9, successes=3, failures=1)

    snapshot = build_policy_snapshot(
        "minimal_ucb", (9, 7, 2), posterior, version=1, archive_version=2
    )

    assert snapshot.eligible_starter_ids == (2, 7, 9)
    assert snapshot.probabilities == (1.0, 0.0, 0.0)
    assert snapshot.support_complete is False


def test_minimal_ucb_uses_the_canonical_tried_starter_formula():
    posterior = StarterProductivityPosterior()
    _record(posterior, 3, successes=4, failures=1)
    _record(posterior, 7, successes=1, failures=3)

    snapshot = build_policy_snapshot(
        "minimal_ucb", (7, 3), posterior, version=2, archive_version=3
    )

    total_attempts = posterior.completed_attempts
    score_three = 4 / 5 + math.sqrt(2 * math.log(total_attempts) / 5)
    score_seven = 1 / 4 + math.sqrt(2 * math.log(total_attempts) / 4)
    assert score_three > score_seven
    assert snapshot.eligible_starter_ids == (3, 7)
    assert snapshot.probabilities == (1.0, 0.0)


def test_minimal_ucb_breaks_tried_starter_ties_by_lowest_id():
    posterior = StarterProductivityPosterior()
    _record(posterior, 8, successes=1, failures=1)
    _record(posterior, 1, successes=1, failures=1)

    snapshot = build_policy_snapshot(
        "minimal_ucb", (8, 1), posterior, version=2, archive_version=3
    )

    assert snapshot.eligible_starter_ids == (1, 8)
    assert snapshot.probabilities == (1.0, 0.0)


def test_policies_are_deterministic_for_the_same_posterior_and_action_set():
    posterior = StarterProductivityPosterior()
    _record(posterior, 2, successes=1, failures=3)
    _record(posterior, 5, successes=2, failures=1)

    first = build_policy_snapshot(
        "posterior_proportional", (5, 2), posterior, version=8, archive_version=13
    )
    second = build_policy_snapshot(
        "posterior_proportional", (2, 5), posterior, version=8, archive_version=13
    )

    assert first == second


@pytest.mark.parametrize(
    ("policy_name", "message"),
    [
        ("legacy_ucb", "legacy UCB is an external comparator"),
        ("other", "unsupported policy"),
    ],
)
def test_legacy_and_unknown_policies_are_rejected(policy_name, message):
    with pytest.raises(ValueError, match=message):
        build_policy_snapshot(
            policy_name,
            (1,),
            StarterProductivityPosterior(),
            version=0,
            archive_version=0,
        )


@pytest.mark.parametrize(
    ("eligible_starter_ids", "message"),
    [
        ((), "eligible_starter_ids cannot be empty"),
        ((2, 2), "eligible_starter_ids must be unique"),
    ],
)
def test_builder_rejects_empty_or_duplicate_action_sets(eligible_starter_ids, message):
    with pytest.raises(ValueError, match=message):
        build_policy_snapshot(
            "uniform",
            eligible_starter_ids,
            StarterProductivityPosterior(),
            version=0,
            archive_version=0,
        )


def test_posterior_proportional_retains_strictly_positive_probability_after_imbalanced_outcomes():
    posterior = StarterProductivityPosterior()
    _record(posterior, 1, successes=0, failures=100)
    _record(posterior, 2, successes=100, failures=0)

    snapshot = build_policy_snapshot(
        "posterior_proportional", (2, 1), posterior, version=3, archive_version=7
    )

    assert snapshot.eligible_starter_ids == (1, 2)
    assert all(probability > 0.0 for probability in snapshot.probabilities)
    assert snapshot.support_complete is True
