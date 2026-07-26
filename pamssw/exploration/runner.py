"""Fixed-budget posterior-exploration campaign runners."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from math import isfinite
from typing import Callable

from ..accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from ..archive import MinimaArchive
from ..config import LSSSWConfig, SSWConfig
from ..relax import Relaxer
from ..state import State
from ..walker import GeometryValidator
from .actions import AttemptStatus, CreditedOutcome
from .campaign import (
    CampaignBudget,
    CampaignStopReason,
    PosteriorExplorationConfig,
    PosteriorExplorationResult,
)
from .controller import ExplorationController
from .event_log import ExplorationEventLog
from .ssw_worker import SSWAttemptWorker


def _bootstrap_minimum(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    *,
    total_force_budget: int,
) -> tuple[State, float, EvaluationCounts]:
    """True-quench one raw state under an exact, local force-evaluation budget."""
    if not isinstance(initial_state, State):
        raise TypeError("initial_state must be a State")

    geometry_validator = GeometryValidator()
    if not geometry_validator.is_valid_state(initial_state):
        raise ValueError("invalid initial geometry")
    if not callable(calculator_factory):
        raise TypeError("calculator_factory must be callable")
    if not isinstance(ssw_config, SSWConfig):
        raise TypeError("ssw_config must be an SSWConfig")
    _positive_force_budget(total_force_budget)

    calculator = calculator_factory()
    if not _is_calculator(calculator):
        raise TypeError("calculator must provide callable evaluate and evaluate_flat")

    counter = EvalCounter(calculator, max_force_evals=total_force_budget)
    relaxer = Relaxer(counter.evaluate_flat, optimizer=ssw_config.quench_optimizer)
    with counter.purpose(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH):
        relaxed = relaxer.relax(
            deepcopy(initial_state),
            fmax=ssw_config.quench_fmax,
            maxiter=ssw_config.quench_maxiter,
        )
    with counter.purpose(EvaluationPurpose.POST_RELAX_VALIDATION):
        valid_final_evaluation = geometry_validator.is_valid_evaluation(relaxed.state, counter)

    energy = float(relaxed.energy)
    if not isfinite(relaxed.gradient_norm) or relaxed.gradient_norm > ssw_config.quench_fmax:
        raise ValueError(
            "bootstrap relaxation did not converge to the configured per-atom force tolerance"
        )
    if not valid_final_evaluation or not isfinite(energy):
        raise ValueError("invalid final relaxed state")
    return deepcopy(relaxed.state), energy, counter.snapshot()


def _positive_force_budget(total_force_budget: object) -> int:
    if isinstance(total_force_budget, bool) or not isinstance(total_force_budget, int):
        raise TypeError("total_force_budget must be an integer")
    if total_force_budget <= 0:
        raise ValueError("total_force_budget must be positive")
    return total_force_budget


def _is_calculator(calculator: object) -> bool:
    return callable(getattr(calculator, "evaluate", None)) and callable(
        getattr(calculator, "evaluate_flat", None)
    )


def run_posterior_ssw(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    """Run a fixed-budget posterior campaign with unsoftened SSW attempts."""
    if type(ssw_config) is not SSWConfig:
        raise TypeError("ssw_config must be exactly an SSWConfig for run_posterior_ssw")
    if not isinstance(exploration_config, PosteriorExplorationConfig):
        raise TypeError("exploration_config must be a PosteriorExplorationConfig")
    return _run_posterior_campaign(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        softening_enabled=False,
    )


def run_posterior_ls_ssw(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: LSSSWConfig,
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    """Run a fixed-budget posterior campaign with locally softened SSW attempts."""
    if not isinstance(ssw_config, LSSSWConfig):
        raise TypeError("ssw_config must be an LSSSWConfig for run_posterior_ls_ssw")
    if not isinstance(exploration_config, PosteriorExplorationConfig):
        raise TypeError("exploration_config must be a PosteriorExplorationConfig")
    return _run_posterior_campaign(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        softening_enabled=True,
    )


def _run_posterior_campaign(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
    *,
    softening_enabled: bool,
) -> PosteriorExplorationResult:
    """Compose the existing exact-accounting exploration primitives once."""
    if not isinstance(ssw_config, SSWConfig):
        raise TypeError("ssw_config must be an SSWConfig")
    if not isinstance(exploration_config, PosteriorExplorationConfig):
        raise TypeError("exploration_config must be a PosteriorExplorationConfig")
    if not isinstance(softening_enabled, bool):
        raise TypeError("softening_enabled must be a boolean")

    run_directory = exploration_config.run_directory
    _preflight_run_directory(run_directory)
    worker = SSWAttemptWorker(
        calculator_factory,
        ssw_config,
        softening_enabled=softening_enabled,
    )
    bootstrap_state, bootstrap_energy, bootstrap_counts = _bootstrap_minimum(
        initial_state,
        calculator_factory,
        ssw_config,
        total_force_budget=exploration_config.total_force_budget,
    )

    budget = CampaignBudget(
        exploration_config.total_force_budget,
        exploration_config.action_force_budget,
    )
    budget.record_bootstrap(bootstrap_counts)
    run_directory.mkdir()
    event_log = ExplorationEventLog(run_directory / "events.jsonl")
    archive = MinimaArchive(
        energy_tol=ssw_config.dedup_energy_tol,
        rmsd_tol=ssw_config.dedup_rmsd_tol,
        max_prototypes=ssw_config.max_prototypes,
    )
    archive.add(deepcopy(bootstrap_state), bootstrap_energy, parent_id=None)
    controller = ExplorationController(
        archive,
        exploration_config.policy_name,
        exploration_config.master_seed,
        event_log,
        require_exact_cost=True,
    )
    outcomes: list[CreditedOutcome] = []
    with ThreadPoolExecutor(max_workers=exploration_config.max_workers) as executor:
        while True:
            width = budget.next_batch_size(exploration_config.batch_size)
            if width == 0:
                break
            batch_outcomes = controller.run_batch(
                executor,
                worker,
                width,
                exploration_config.action_force_budget,
            )
            budget.commit_batch(tuple(outcome.evaluation_counts for outcome in batch_outcomes))
            outcomes.extend(batch_outcomes)

    if budget.stop_reason is None:
        raise RuntimeError("posterior campaign reached no terminal budget state")
    return _campaign_result(
        controller,
        budget,
        outcomes,
        exploration_config,
    )


def _preflight_run_directory(run_directory) -> None:
    if run_directory.exists() or run_directory.is_symlink():
        raise FileExistsError(f"run_directory already exists: {run_directory}")
    parent = run_directory.parent
    if not parent.exists():
        raise FileNotFoundError(f"run_directory parent does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"run_directory parent is not a directory: {parent}")


def _campaign_result(
    controller: ExplorationController,
    budget: CampaignBudget,
    outcomes: list[CreditedOutcome],
    exploration_config: PosteriorExplorationConfig,
) -> PosteriorExplorationResult:
    purpose_counts = EvaluationCounts.sum((budget.bootstrap_counts, budget.action_counts))
    completed_attempts = sum(
        outcome.status is AttemptStatus.COMPLETED for outcome in outcomes
    )
    posterior_observed_attempts = sum(outcome.posterior_observed for outcome in outcomes)
    ineligibility_reasons: list[str] = []
    if posterior_observed_attempts != len(outcomes):
        ineligibility_reasons.append("non_posterior_observed_attempt")
    if purpose_counts.count(EvaluationPurpose.UNATTRIBUTED) > 0:
        ineligibility_reasons.append("unattributed_evaluations")
    if budget.stop_reason is CampaignStopReason.ZERO_COST_STALL:
        ineligibility_reasons.append("zero_cost_stall")
    reasons = tuple(sorted(ineligibility_reasons))
    return PosteriorExplorationResult(
        archive=controller.archive,
        posterior=controller.posterior,
        policy_name=exploration_config.policy_name,
        completed_batches=budget.committed_batches,
        completed_attempts=completed_attempts,
        failed_attempts=len(outcomes) - completed_attempts,
        posterior_observed_attempts=posterior_observed_attempts,
        bootstrap_evaluations=budget.bootstrap_counts.total,
        action_evaluations=budget.action_counts.total,
        total_evaluations=budget.spent,
        purpose_counts=purpose_counts,
        total_force_budget=budget.total,
        unused_force_budget=budget.unused,
        stop_reason=budget.stop_reason,
        benchmark_eligible=not reasons,
        benchmark_ineligibility_reasons=reasons,
        run_directory=exploration_config.run_directory,
    )


__all__ = [
    "run_posterior_ssw",
    "run_posterior_ls_ssw",
]
