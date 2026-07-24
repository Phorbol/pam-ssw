from __future__ import annotations

from math import fsum, log, sqrt

from .actions import PolicySnapshot
from .posterior import StarterProductivityPosterior


SUPPORTED_POLICIES = frozenset({"uniform", "posterior_proportional", "minimal_ucb"})


def _canonical_eligible_starter_ids(
    eligible_starter_ids: object,
    posterior: StarterProductivityPosterior,
) -> tuple[int, ...]:
    try:
        starter_ids = tuple(eligible_starter_ids)
    except TypeError as exc:
        raise ValueError("eligible_starter_ids must be a nonempty sequence") from exc
    if not starter_ids:
        raise ValueError("eligible_starter_ids cannot be empty")

    for starter_id in starter_ids:
        posterior.counts(starter_id)
    if len(set(starter_ids)) != len(starter_ids):
        raise ValueError("eligible_starter_ids must be unique")

    sorted_starter_ids = tuple(sorted(starter_ids))
    posterior.ensure(sorted_starter_ids)
    return sorted_starter_ids


def build_policy_snapshot(
    policy_name: str,
    eligible_starter_ids: object,
    posterior: StarterProductivityPosterior,
    version: int,
    archive_version: int,
) -> PolicySnapshot:
    """Build one deterministic, auditable starter-selection policy snapshot."""
    if not isinstance(policy_name, str) or not policy_name.strip():
        raise ValueError("policy_name must be a nonempty string")
    if policy_name == "legacy_ucb":
        raise ValueError("legacy UCB is an external comparator and is not a supported policy")
    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy_name}")
    if not isinstance(posterior, StarterProductivityPosterior):
        raise ValueError("posterior must be a StarterProductivityPosterior")

    starter_ids = _canonical_eligible_starter_ids(eligible_starter_ids, posterior)

    if policy_name == "uniform":
        probabilities = (1.0 / len(starter_ids),) * len(starter_ids)
        return PolicySnapshot(
            version=version,
            archive_version=archive_version,
            policy_name=policy_name,
            eligible_starter_ids=starter_ids,
            probabilities=probabilities,
            support_complete=True,
        )

    if policy_name == "posterior_proportional":
        means = tuple(posterior.mean(starter_id) for starter_id in starter_ids)
        total_mean = fsum(means)
        probabilities = tuple(mean / total_mean for mean in means)
        return PolicySnapshot(
            version=version,
            archive_version=archive_version,
            policy_name=policy_name,
            eligible_starter_ids=starter_ids,
            probabilities=probabilities,
            support_complete=True,
        )

    counts_by_starter = {starter_id: posterior.counts(starter_id) for starter_id in starter_ids}
    untried_starter_ids = [
        starter_id
        for starter_id in starter_ids
        if sum(counts_by_starter[starter_id]) == 0
    ]
    if untried_starter_ids:
        selected_starter_id = untried_starter_ids[0]
    else:
        total_attempts = max(2, posterior.completed_attempts)

        def ucb_score(starter_id: int) -> float:
            successes, failures = counts_by_starter[starter_id]
            attempts = successes + failures
            return successes / attempts + sqrt(2.0 * log(total_attempts) / attempts)

        selected_starter_id = max(
            starter_ids,
            key=lambda starter_id: (ucb_score(starter_id), -starter_id),
        )

    probabilities = tuple(1.0 if starter_id == selected_starter_id else 0.0 for starter_id in starter_ids)
    return PolicySnapshot(
        version=version,
        archive_version=archive_version,
        policy_name=policy_name,
        eligible_starter_ids=starter_ids,
        probabilities=probabilities,
        support_complete=False,
    )


__all__ = ["SUPPORTED_POLICIES", "build_policy_snapshot"]
