from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.exploration import (
    CampaignBudget,
    CampaignBudgetSnapshot,
    CampaignStopReason,
    PosteriorExplorationConfig,
    PosteriorExplorationResult,
)
from pamssw.exploration.policies import SUPPORTED_POLICIES
from pamssw.exploration.posterior import StarterProductivityPosterior


def _config(**changes) -> PosteriorExplorationConfig:
    values = {
        "policy_name": "uniform",
        "batch_size": 4,
        "max_workers": 2,
        "action_force_budget": 8,
        "total_force_budget": 101,
        "master_seed": 7,
        "calculator_label": " MACE-OMAT ",
        "calculator_fingerprint": " model-sha256 ",
        "run_directory": "runs/campaign",
    }
    values.update(changes)
    return PosteriorExplorationConfig(**values)


def test_posterior_exploration_config_has_the_fixed_fidelity_contract_and_normalizes_values():
    config = _config()

    assert tuple(field.name for field in fields(config)) == (
        "policy_name",
        "batch_size",
        "max_workers",
        "action_force_budget",
        "total_force_budget",
        "master_seed",
        "calculator_label",
        "calculator_fingerprint",
        "run_directory",
        "mode",
    )
    assert config.calculator_label == "MACE-OMAT"
    assert config.calculator_fingerprint == "model-sha256"
    assert config.run_directory == Path("runs/campaign")
    assert config.mode == "new"


def test_posterior_exploration_config_is_frozen_and_accepts_all_current_policies():
    config = _config()

    with pytest.raises(FrozenInstanceError):
        config.batch_size = 5

    assert {_config(policy_name=name).policy_name for name in SUPPORTED_POLICIES} == SUPPORTED_POLICIES


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_name": "unknown"},
        {"policy_name": None},
        {"batch_size": 0},
        {"batch_size": True},
        {"batch_size": 1.0},
        {"max_workers": 0},
        {"max_workers": True},
        {"max_workers": 1.0},
        {"max_workers": 5},
        {"action_force_budget": 0},
        {"action_force_budget": True},
        {"action_force_budget": 1.0},
        {"total_force_budget": 0},
        {"total_force_budget": True},
        {"total_force_budget": 1.0},
        {"master_seed": -1},
        {"master_seed": True},
        {"master_seed": 1.0},
        {"calculator_label": " \t "},
        {"calculator_label": None},
        {"calculator_fingerprint": "\n"},
        {"calculator_fingerprint": None},
        {"run_directory": 1},
        {"run_directory": object()},
        {"mode": "restart"},
        {"mode": True},
    ],
)
def test_posterior_exploration_config_rejects_invalid_values(changes):
    with pytest.raises((TypeError, ValueError)):
        _config(**changes)


def test_posterior_exploration_config_accepts_path_objects_and_resume_mode():
    config = _config(run_directory=Path("runs/resume"), mode="resume")

    assert config.run_directory == Path("runs/resume")
    assert config.mode == "resume"


def _counts(total: int) -> EvaluationCounts:
    return EvaluationCounts.from_mapping({EvaluationPurpose.DIRECTION_ORACLE: total})


def _bootstrapped_budget(
    total: int = 101, bootstrap_cost: int = 11, action_force_budget: int = 10
) -> CampaignBudget:
    budget = CampaignBudget(total, action_force_budget)
    budget.record_bootstrap(_counts(bootstrap_cost))
    return budget


def test_campaign_stop_reason_is_closed_to_the_two_budget_terminal_causes():
    assert issubclass(CampaignStopReason, str)
    assert set(CampaignStopReason) == {
        CampaignStopReason.BUDGET_TAIL,
        CampaignStopReason.ZERO_COST_STALL,
    }


def test_campaign_budget_records_exact_costs_not_reserved_costs():
    budget = _bootstrapped_budget()

    assert budget.next_batch_size(2) == 2
    budget.commit_batch((_counts(8), _counts(10)))
    assert budget.next_batch_size(2) == 2
    budget.commit_batch((_counts(3),))

    assert budget.bootstrap_counts == _counts(11)
    assert budget.action_counts == _counts(21)
    assert budget.spent == 32
    assert budget.remaining == 69
    assert budget.unused == 69
    assert budget.committed_batches == 2
    assert budget.committed_attempts == 3
    assert budget.stop_reason is None


def test_campaign_budget_marks_a_tail_without_spending_a_residual_action():
    budget = _bootstrapped_budget(total=25, bootstrap_cost=6, action_force_budget=7)

    assert budget.next_batch_size(3) == 2
    budget.commit_batch((_counts(7), _counts(6)))

    assert budget.stop_reason is CampaignStopReason.BUDGET_TAIL
    assert budget.next_batch_size(3) == 0
    assert budget.stop_reason is CampaignStopReason.BUDGET_TAIL
    assert budget.spent == 19
    assert budget.remaining == 6
    assert budget.unused == 6


def test_campaign_budget_releases_unspent_reservation_after_an_early_attempt_termination():
    budget = _bootstrapped_budget(total=30, bootstrap_cost=5, action_force_budget=8)

    assert budget.next_batch_size(3) == 3
    budget.commit_batch((_counts(8), _counts(1), _counts(0)))

    assert budget.spent == 14
    assert budget.remaining == 16
    assert budget.next_batch_size(3) == 2


def test_campaign_budget_stops_after_a_zero_cost_batch():
    budget = _bootstrapped_budget()

    budget.commit_batch((_counts(0),))

    assert budget.stop_reason is CampaignStopReason.ZERO_COST_STALL
    assert budget.next_batch_size(1) == 0


@pytest.mark.parametrize("total", [0, True, 1.0, "10"])
def test_campaign_budget_rejects_invalid_total(total):
    with pytest.raises((TypeError, ValueError)):
        CampaignBudget(total, 1)


@pytest.mark.parametrize("action_force_budget", [0, True, 1.0, "1"])
def test_campaign_budget_requires_a_strict_fixed_action_force_budget(action_force_budget):
    with pytest.raises((TypeError, ValueError)):
        CampaignBudget(10, action_force_budget)
    with pytest.raises(TypeError):
        CampaignBudget(10)


def test_campaign_budget_bootstrap_is_one_time_counted_and_within_total():
    budget = CampaignBudget(10, 1)
    budget.record_bootstrap(_counts(4))

    with pytest.raises(RuntimeError):
        budget.record_bootstrap(_counts(1))
    with pytest.raises(ValueError):
        CampaignBudget(10, 1).record_bootstrap(_counts(11))
    with pytest.raises(TypeError):
        CampaignBudget(10, 1).record_bootstrap(object())


@pytest.mark.parametrize(
    "batch_size",
    [0, True, 1.0],
)
def test_campaign_budget_next_batch_size_validates_a_positive_maximum_batch_size(batch_size):
    with pytest.raises((TypeError, ValueError)):
        CampaignBudget(10, 1).next_batch_size(batch_size)


def test_campaign_budget_next_batch_size_requires_bootstrap_before_scheduling():
    with pytest.raises(RuntimeError):
        CampaignBudget(10, 1).next_batch_size(1)


@pytest.mark.parametrize(
    "counts",
    [
        [],
        (),
        (object(),),
        (_counts(11),),
        (_counts(8), _counts(8), _counts(1)),
    ],
)
def test_campaign_budget_commit_batch_rejects_invalid_reservations(counts):
    budget = _bootstrapped_budget(total=25, bootstrap_cost=5)

    with pytest.raises((TypeError, ValueError)):
        budget.commit_batch(counts)


def test_campaign_budget_does_not_accept_per_call_force_budget_overrides():
    budget = _bootstrapped_budget()

    with pytest.raises(TypeError):
        budget.next_batch_size(1, 10)
    with pytest.raises(TypeError):
        budget.commit_batch((_counts(1),), 10)


def test_campaign_budget_commit_batch_requires_bootstrap_and_rejects_terminal_commits():
    with pytest.raises(RuntimeError):
        CampaignBudget(10, 1).commit_batch((_counts(1),))

    budget = _bootstrapped_budget()
    budget.commit_batch((_counts(0),))
    with pytest.raises(RuntimeError):
        budget.commit_batch((_counts(1),))


def test_campaign_budget_snapshot_restores_validated_state_and_recomputes_spend():
    budget = _bootstrapped_budget(total=30, bootstrap_cost=6, action_force_budget=8)
    budget.commit_batch((_counts(8), _counts(3)))

    snapshot = budget.snapshot()
    restored = CampaignBudget.from_snapshot(snapshot, action_force_budget=8)
    restored_via_restore = CampaignBudget.restore(snapshot, action_force_budget=8)

    assert tuple(field.name for field in fields(snapshot)) == (
        "total",
        "action_force_budget",
        "bootstrap_counts",
        "action_counts",
        "committed_batches",
        "committed_attempts",
        "bootstrap_recorded",
        "stop_reason",
        "last_batch_spend",
    )
    assert restored.snapshot() == snapshot
    assert restored_via_restore.snapshot() == snapshot
    assert restored.spent == 17
    assert restored.remaining == 13
    assert restored.last_batch_spend == 11
    assert restored.action_force_budget == 8
    with pytest.raises(FrozenInstanceError):
        snapshot.total = 31


def test_campaign_budget_restore_rejects_inconsistent_accounting_and_preserves_terminal_state():
    tail = _bootstrapped_budget(total=25, bootstrap_cost=6, action_force_budget=7)
    tail.commit_batch((_counts(7), _counts(6)))
    assert tail.next_batch_size(3) == 0
    restored_tail = CampaignBudget.from_snapshot(tail.snapshot(), action_force_budget=7)

    assert restored_tail.stop_reason is CampaignStopReason.BUDGET_TAIL
    assert restored_tail.next_batch_size(3) == 0

    invalid = CampaignBudgetSnapshot(
        total=10,
        action_force_budget=1,
        bootstrap_counts=_counts(1),
        action_counts=_counts(0),
        committed_batches=0,
        committed_attempts=0,
        bootstrap_recorded=False,
        stop_reason=None,
        last_batch_spend=None,
    )
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(invalid, action_force_budget=1)

    impossible_batch_history = CampaignBudgetSnapshot(
        total=10,
        action_force_budget=1,
        bootstrap_counts=_counts(1),
        action_counts=_counts(1),
        committed_batches=0,
        committed_attempts=1,
        bootstrap_recorded=True,
        stop_reason=None,
        last_batch_spend=1,
    )
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(impossible_batch_history, action_force_budget=1)


def test_campaign_budget_restore_derives_and_validates_terminal_reason_from_manifest_facts():
    tail = _bootstrapped_budget(total=25, bootstrap_cost=6, action_force_budget=7)
    tail.commit_batch((_counts(7), _counts(6)))
    assert tail.next_batch_size(3) == 0
    tail_snapshot = tail.snapshot()

    assert (
        CampaignBudget.from_snapshot(tail_snapshot, action_force_budget=7).stop_reason
        is CampaignStopReason.BUDGET_TAIL
    )
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(
            replace(tail_snapshot, stop_reason=None), action_force_budget=7
        )
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(tail_snapshot, action_force_budget=6)

    zero = _bootstrapped_budget()
    zero.commit_batch((_counts(0),))
    zero_snapshot = zero.snapshot()
    assert zero_snapshot.last_batch_spend == 0
    assert (
        CampaignBudget.from_snapshot(zero_snapshot, action_force_budget=10).stop_reason
        is CampaignStopReason.ZERO_COST_STALL
    )
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(
            replace(zero_snapshot, stop_reason=CampaignStopReason.BUDGET_TAIL),
            action_force_budget=10,
        )


@pytest.mark.parametrize("action_force_budget", [0, True, 1.0])
def test_campaign_budget_restore_validates_manifest_action_budget(action_force_budget):
    snapshot = _bootstrapped_budget(total=30, bootstrap_cost=6).snapshot()

    with pytest.raises((TypeError, ValueError)):
        CampaignBudget.from_snapshot(snapshot, action_force_budget=action_force_budget)


def test_campaign_budget_restore_rejects_manifest_q_that_differs_from_snapshot_identity():
    budget = _bootstrapped_budget(total=35, bootstrap_cost=5, action_force_budget=10)
    budget.commit_batch((_counts(8),))
    snapshot = budget.snapshot()

    assert snapshot.stop_reason is None
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(snapshot, action_force_budget=5)
    with pytest.raises(TypeError):
        CampaignBudget.from_snapshot(snapshot, 10)


def test_campaign_budget_bootstrap_only_tail_snapshot_round_trips_with_the_bound_q():
    budget = _bootstrapped_budget(total=5, bootstrap_cost=5, action_force_budget=1)
    snapshot = budget.snapshot()

    assert budget.stop_reason is CampaignStopReason.BUDGET_TAIL
    assert CampaignBudget.from_snapshot(snapshot, action_force_budget=1).snapshot() == snapshot


@pytest.mark.parametrize(
    "snapshot",
    [
        CampaignBudgetSnapshot(
            total=10,
            action_force_budget=5,
            bootstrap_counts=_counts(1),
            action_counts=_counts(1),
            committed_batches=1,
            committed_attempts=1,
            bootstrap_recorded=True,
            stop_reason=None,
            last_batch_spend=None,
        ),
        CampaignBudgetSnapshot(
            total=10,
            action_force_budget=5,
            bootstrap_counts=_counts(1),
            action_counts=_counts(0),
            committed_batches=0,
            committed_attempts=0,
            bootstrap_recorded=True,
            stop_reason=None,
            last_batch_spend=0,
        ),
        CampaignBudgetSnapshot(
            total=10,
            action_force_budget=5,
            bootstrap_counts=_counts(1),
            action_counts=_counts(1),
            committed_batches=1,
            committed_attempts=1,
            bootstrap_recorded=True,
            stop_reason=CampaignStopReason.ZERO_COST_STALL,
            last_batch_spend=1,
        ),
    ],
)
def test_campaign_budget_restore_rejects_inconsistent_last_batch_facts(snapshot):
    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(snapshot, action_force_budget=5)


def test_campaign_budget_restore_rejects_action_counts_beyond_attempt_capacity():
    snapshot = CampaignBudgetSnapshot(
        total=100,
        action_force_budget=10,
        bootstrap_counts=_counts(10),
        action_counts=_counts(50),
        committed_batches=2,
        committed_attempts=2,
        bootstrap_recorded=True,
        stop_reason=None,
        last_batch_spend=10,
    )

    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(snapshot, action_force_budget=10)


def test_campaign_budget_restore_rejects_last_batch_beyond_possible_batch_width():
    snapshot = CampaignBudgetSnapshot(
        total=40,
        action_force_budget=10,
        bootstrap_counts=_counts(10),
        action_counts=_counts(20),
        committed_batches=2,
        committed_attempts=2,
        bootstrap_recorded=True,
        stop_reason=None,
        last_batch_spend=20,
    )

    with pytest.raises(ValueError):
        CampaignBudget.from_snapshot(snapshot, action_force_budget=10)


def _result(**changes) -> PosteriorExplorationResult:
    values = {
        "archive": MinimaArchive(energy_tol=0.01, rmsd_tol=0.1),
        "posterior": StarterProductivityPosterior(),
        "policy_name": "uniform",
        "completed_batches": 2,
        "completed_attempts": 2,
        "failed_attempts": 1,
        "posterior_observed_attempts": 2,
        "bootstrap_evaluations": 6,
        "action_evaluations": 11,
        "total_evaluations": 17,
        "purpose_counts": _counts(17),
        "total_force_budget": 25,
        "unused_force_budget": 8,
        "stop_reason": CampaignStopReason.BUDGET_TAIL,
        "benchmark_eligible": False,
        "benchmark_ineligibility_reasons": ("budget tail",),
        "run_directory": "runs/campaign",
    }
    values.update(changes)
    return PosteriorExplorationResult(**values)


def test_posterior_exploration_result_has_only_the_campaign_summary_contract():
    result = _result()

    assert tuple(field.name for field in fields(result)) == (
        "archive",
        "posterior",
        "policy_name",
        "completed_batches",
        "completed_attempts",
        "failed_attempts",
        "posterior_observed_attempts",
        "bootstrap_evaluations",
        "action_evaluations",
        "total_evaluations",
        "purpose_counts",
        "total_force_budget",
        "unused_force_budget",
        "stop_reason",
        "benchmark_eligible",
        "benchmark_ineligibility_reasons",
        "run_directory",
    )
    assert result.run_directory == Path("runs/campaign")
    with pytest.raises(FrozenInstanceError):
        result.total_evaluations = 0


def test_posterior_exploration_result_defensively_clones_archive_and_posterior():
    archive = MinimaArchive(energy_tol=0.01, rmsd_tol=0.1)
    posterior = StarterProductivityPosterior()

    result = _result(archive=archive, posterior=posterior)
    archive.energy_tol = 99.0
    posterior.update(4, True)

    assert result.archive is not archive
    assert result.posterior is not posterior
    assert result.archive.energy_tol == pytest.approx(0.01)
    assert result.posterior.counts(4) == (0, 0)


def test_posterior_exploration_result_validates_accounting_and_attempt_invariants():
    result = _result()

    assert result.completed_attempts + result.failed_attempts == 3
    assert result.posterior_observed_attempts <= 3
    assert result.total_evaluations == result.bootstrap_evaluations + result.action_evaluations
    assert result.total_evaluations == result.purpose_counts.total
    assert result.total_evaluations + result.unused_force_budget == result.total_force_budget
    assert result.completed_batches <= 3


@pytest.mark.parametrize(
    "changes",
    [
        {"archive": object()},
        {"posterior": object()},
        {"policy_name": "unknown"},
        {"completed_batches": -1},
        {"completed_batches": True},
        {"completed_attempts": -1},
        {"completed_attempts": True},
        {"failed_attempts": -1},
        {"failed_attempts": True},
        {"posterior_observed_attempts": 4},
        {"posterior_observed_attempts": True},
        {"bootstrap_evaluations": -1},
        {"action_evaluations": True},
        {"total_evaluations": 16},
        {"purpose_counts": _counts(16)},
        {"total_force_budget": 24},
        {"unused_force_budget": -1},
        {"stop_reason": None},
        {"stop_reason": "budget_tail"},
        {"benchmark_eligible": 1},
        {"benchmark_ineligibility_reasons": []},
        {"benchmark_ineligibility_reasons": ("",)},
        {"benchmark_ineligibility_reasons": (1,)},
        {"run_directory": object()},
    ],
)
def test_posterior_exploration_result_rejects_invalid_field_values(changes):
    with pytest.raises((TypeError, ValueError)):
        _result(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_batches": 4},
        {"completed_batches": 1, "completed_attempts": 0, "failed_attempts": 0},
        {"benchmark_eligible": True},
        {"benchmark_eligible": False, "benchmark_ineligibility_reasons": ()},
    ],
)
def test_posterior_exploration_result_rejects_broken_summary_invariants(changes):
    with pytest.raises(ValueError):
        _result(**changes)


def test_posterior_exploration_result_accepts_a_clean_benchmark_summary():
    result = _result(
        stop_reason=CampaignStopReason.BUDGET_TAIL,
        benchmark_eligible=True,
        benchmark_ineligibility_reasons=(),
    )

    assert result.benchmark_eligible is True
    assert result.benchmark_ineligibility_reasons == ()


@pytest.mark.parametrize(
    "changes",
    [
        {
            "stop_reason": CampaignStopReason.ZERO_COST_STALL,
            "benchmark_eligible": True,
            "benchmark_ineligibility_reasons": (),
        },
        {
            "stop_reason": CampaignStopReason.ZERO_COST_STALL,
            "benchmark_eligible": False,
            "benchmark_ineligibility_reasons": (),
        },
        {
            "stop_reason": CampaignStopReason.BUDGET_TAIL,
            "benchmark_eligible": True,
            "benchmark_ineligibility_reasons": ("tail",),
        },
    ],
)
def test_posterior_exploration_result_validates_stop_reason_benchmark_eligibility(changes):
    with pytest.raises(ValueError):
        _result(**changes)


def test_posterior_exploration_result_allows_a_zero_cost_stop_with_an_ineligibility_reason():
    result = _result(stop_reason=CampaignStopReason.ZERO_COST_STALL)

    assert result.benchmark_eligible is False
    assert result.benchmark_ineligibility_reasons
