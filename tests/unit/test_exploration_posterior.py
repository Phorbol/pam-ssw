import pytest

from pamssw.exploration.posterior import StarterProductivityPosterior


def test_new_starter_has_fixed_uniform_beta_prior():
    posterior = StarterProductivityPosterior()

    posterior.ensure((starter_id for starter_id in (3, 7)))

    assert posterior.PRIOR_ALPHA == 1.0
    assert posterior.PRIOR_BETA == 1.0
    assert posterior.counts(3) == (0, 0)
    assert posterior.counts(7) == (0, 0)
    assert posterior.mean(3) == pytest.approx(0.5)
    assert posterior.completed_attempts == 0


def test_posterior_updates_successes_and_failures_exactly_once():
    posterior = StarterProductivityPosterior()

    posterior.update(3, discovered=True)
    posterior.update(3, discovered=False)
    posterior.update(3, discovered=True)

    assert posterior.counts(3) == (2, 1)
    assert posterior.mean(3) == pytest.approx(3 / 5)
    assert posterior.completed_attempts == 3


def test_unseen_lookup_uses_prior_without_mutating_completed_attempts():
    posterior = StarterProductivityPosterior()

    assert posterior.counts(99) == (0, 0)
    assert posterior.mean(99) == pytest.approx(0.5)
    assert posterior.completed_attempts == 0


def test_ensure_creates_zero_count_entries_without_completed_attempts():
    posterior = StarterProductivityPosterior()

    posterior.update(4, discovered=True)
    posterior.ensure((4, 8, 10))

    assert posterior.counts(4) == (1, 0)
    assert posterior.counts(8) == (0, 0)
    assert posterior.counts(10) == (0, 0)
    assert posterior.completed_attempts == 1


def test_posterior_clone_is_independent():
    posterior = StarterProductivityPosterior()
    posterior.update(1, discovered=True)
    cloned = posterior.clone()

    cloned.update(1, discovered=False)
    cloned.update(2, discovered=True)

    assert posterior.counts(1) == (1, 0)
    assert posterior.counts(2) == (0, 0)
    assert posterior.completed_attempts == 1
    assert cloned.counts(1) == (1, 1)
    assert cloned.counts(2) == (1, 0)
    assert cloned.completed_attempts == 3


@pytest.mark.parametrize("starter_id", [True, False, -1])
def test_posterior_rejects_boolean_or_negative_starter_ids(starter_id):
    posterior = StarterProductivityPosterior()

    with pytest.raises(ValueError, match="starter_id"):
        posterior.update(starter_id, discovered=True)


def test_posterior_rejects_nonboolean_discovered_outcomes():
    posterior = StarterProductivityPosterior()

    with pytest.raises(ValueError, match="discovered"):
        posterior.update(0, discovered=1)

    assert posterior.counts(0) == (0, 0)
    assert posterior.completed_attempts == 0
