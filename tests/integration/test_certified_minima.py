"""Search must never promote an incomplete true quench to a minimum."""

import numpy as np
import pytest

from pamssw import SSWConfig, State, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw import relax as relax_module
from pamssw.walker import CandidateProposal, SurfaceWalker


def state(x):
    return State(numbers=np.array([1]), positions=np.array([[x, 0.0, 0.0]]))


def config(**overrides):
    return SSWConfig(**dict({
        "max_trials": 2,
        "quench_optimizer": "ase-fire",
        "quench_maxiter": 1,
        "quench_fmax": 1e-4,
    }, **overrides))


def test_search_rejects_uncertified_bootstrap_with_exact_cost():
    error_type = getattr(relax_module, "QuenchConvergenceError", RuntimeError)
    with pytest.raises(error_type) as caught:
        run_ssw(state(0.5), AnalyticCalculator(DoubleWell2D()), config())
    assert caught.value.relaxation.gradient_norm > 1e-4
    assert caught.value.evaluation_counts.total > 0
    assert caught.value.evaluation_counts.as_dict()["unattributed"] == 0


class FixedProposals(SurfaceWalker):
    """Feed prescribed states to the real quench/archive boundary."""

    def _proposal_pool(self, seed_state, archive, trial_index, step_target=None, **kwargs):
        return [CandidateProposal("test", state(0.5 if trial_index == 0 else 1.0))]


def test_rejected_landing_does_not_enter_archive_and_search_continues(tmp_path):
    walker = FixedProposals(
        AnalyticCalculator(DoubleWell2D()),
        config(write_proposal_minima=True, proposal_minima_dir=str(tmp_path)),
        softening_enabled=False,
    )
    result = walker.run(state(-1.0))
    assert result.stats["n_trials"] == 2
    assert result.stats["quench_certificate_rejections"] == 1
    assert result.stats["budget_exhausted"] == 0
    assert len(result.archive.entries) == 2
    assert sorted(round(e.state.positions[0, 0], 4) for e in result.archive.entries) == [-1., 1.]
    assert len(result.walk_history) == 1
    assert len(result.quench_failures) == 1
    failure = result.quench_failures[0]
    assert failure.trial_index == 1
    assert failure.relaxation.gradient_norm > 1e-4
    from ase.io import read
    paths = list(tmp_path.glob("*uncertified*.xyz"))
    assert paths
    assert read(paths[0]).info["force_max"] > 1e-4
    assert walker.calculator.snapshot().as_dict()["unattributed"] == 0


@pytest.mark.parametrize("norm", [-1., float("nan"), float("inf")])
def test_force_certificate_rejects_nonphysical_norm(norm):
    from pamssw.result import RelaxResult
    from pamssw.relax import has_force_convergence_certificate
    result = RelaxResult(state(-1.), 0., norm, 0)
    assert not has_force_convergence_certificate(result, 0.01)


def test_uncertified_worker_starter_is_invalid_not_a_worker_or_budget_error():
    from pamssw.exploration import SSWAttemptWorker
    from pamssw.exploration.actions import AttemptStatus, StarterAction
    worker = SSWAttemptWorker(lambda: AnalyticCalculator(DoubleWell2D()), config())
    action = StarterAction(
        action_id="batch-00000000-slot-0000", batch_id=0, slot_id=0,
        policy_name="uniform", policy_version=1, archive_version=0,
        starter_id=0, selection_probability=1.0, random_seed=3, force_budget=100,
    )
    result = worker(action, state(0.5))
    assert result.status is AttemptStatus.INVALID
    assert result.failure_reason == "uncertified_starter"
    assert result.landing_state is None
    assert 0 < result.force_evaluations < 100
    assert result.evaluation_counts.total == result.force_evaluations
    assert result.evaluation_counts.as_dict()["unattributed"] == 0
