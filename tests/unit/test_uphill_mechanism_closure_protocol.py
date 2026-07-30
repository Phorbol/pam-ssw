from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from pamssw.bias import GaussianBiasTerm
from pamssw.softening import LocalSofteningModel, PairSofteningTerm
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT / "runs" / "20260731-uphill-mechanism-closure" / "protocol.py"
)


def _protocol():
    spec = importlib.util.spec_from_file_location(
        "uphill_mechanism_closure_protocol",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task() -> ProposalRelaxationTask:
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]]),
    )
    first = GaussianBiasTerm(
        center=state.flatten_positions(),
        direction=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]),
        sigma=0.4,
        weight=0.2,
    )
    second = GaussianBiasTerm(
        center=state.flatten_positions() + 0.1,
        direction=np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0]),
        sigma=0.3,
        weight=0.1,
    )
    softening = LocalSofteningModel(
        [
            PairSofteningTerm(
                atom_i=0,
                atom_j=1,
                reference_distance=1.4,
                width=0.35,
                strength=0.15,
            )
        ],
        penalty="buckingham_repulsive",
        xi=0.3,
    )
    return ProposalRelaxationTask(
        initial_state=state,
        biases=(first, second),
        softening=softening,
        fmax=0.05,
        maxiter=80,
        coordinate_trust_radius=1.5,
    )


def test_maxiter_arms_change_only_iteration_capacity():
    protocol = _protocol()
    task = _task()

    arms = protocol.maxiter_arms(task, extended_maxiter=300)

    assert set(arms) == {"maxiter80", "maxiter300"}
    assert arms["maxiter80"].maxiter == 80
    assert arms["maxiter300"].maxiter == 300
    assert protocol.physical_task_fingerprint(
        arms["maxiter80"]
    ) == protocol.physical_task_fingerprint(arms["maxiter300"])


def test_history_arms_change_only_historical_bias_retention():
    protocol = _protocol()
    task = _task()

    arms = protocol.history_arms(task)

    assert len(arms["cumulative"].biases) == 2
    assert len(arms["newest_only"].biases) == 1
    assert protocol.bias_fingerprint(
        arms["cumulative"].biases[-1]
    ) == protocol.bias_fingerprint(arms["newest_only"].biases[0])
    assert protocol.task_component_diff(
        arms["cumulative"],
        arms["newest_only"],
    ) == {"bias_history"}


def test_softening_arms_change_only_proposal_softening():
    protocol = _protocol()
    task = _task()

    arms = protocol.softening_arms(task)

    assert arms["both"].softening is not None
    assert arms["oracle_only"].softening is None
    assert protocol.task_component_diff(
        arms["both"],
        arms["oracle_only"],
    ) == {"proposal_softening"}


def test_purpose_delta_requires_exact_total_closure():
    protocol = _protocol()

    delta = protocol.purpose_delta(
        {"total": 10, "direction_oracle": 4, "biased_proposal_relax": 6},
        {"total": 17, "direction_oracle": 4, "biased_proposal_relax": 13},
    )

    assert delta == {
        "total": 7,
        "direction_oracle": 0,
        "biased_proposal_relax": 7,
    }
