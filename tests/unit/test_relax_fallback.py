from __future__ import annotations

import numpy as np

from pamssw.relax import relax_with_certificate_fallback
from pamssw.result import RelaxResult
from pamssw.state import State


def _state(x: float) -> State:
    return State(numbers=np.array([1]), positions=np.array([[x, 0.0, 0.0]]))


class _ScriptedRelaxer:
    def __init__(self, result: RelaxResult) -> None:
        self.result = result
        self.starts: list[State] = []
        self.calls: list[dict[str, object]] = []

    def relax(self, state: State, **kwargs) -> RelaxResult:
        self.starts.append(state)
        self.calls.append(kwargs)
        return self.result


def test_certificate_fallback_skips_fallback_after_certified_primary():
    primary_result = RelaxResult(
        state=_state(1.0),
        energy=-1.0,
        gradient_norm=0.01,
        n_iter=3,
    )
    primary = _ScriptedRelaxer(primary_result)
    fallback = _ScriptedRelaxer(
        RelaxResult(state=_state(2.0), energy=-2.0, gradient_norm=0.0, n_iter=2)
    )
    fallback_starts: list[str] = []

    outcome = relax_with_certificate_fallback(
        primary,
        _state(0.0),
        fmax=0.01,
        maxiter=20,
        fallback_relaxer=fallback,
        on_fallback_start=lambda: fallback_starts.append("started"),
    )

    assert outcome.primary is primary_result
    assert outcome.fallback is None
    assert outcome.final is primary_result
    assert outcome.fallback_used is False
    assert fallback.starts == []
    assert fallback_starts == []


def test_certificate_fallback_starts_once_from_uncertified_primary_terminal_state():
    primary_terminal = _state(1.0)
    fallback_terminal = _state(2.0)
    primary_result = RelaxResult(
        state=primary_terminal,
        energy=-1.0,
        gradient_norm=float("nan"),
        n_iter=20,
    )
    fallback_result = RelaxResult(
        state=fallback_terminal,
        energy=-2.0,
        gradient_norm=0.005,
        n_iter=4,
    )
    primary = _ScriptedRelaxer(primary_result)
    fallback = _ScriptedRelaxer(fallback_result)
    trajectory_states: list[State] = []
    fallback_starts: list[str] = []

    outcome = relax_with_certificate_fallback(
        primary,
        _state(0.0),
        fmax=0.01,
        maxiter=20,
        fallback_relaxer=fallback,
        on_fallback_start=lambda: fallback_starts.append("started"),
        trajectory_callback=trajectory_states.append,
        trajectory_stride=3,
    )

    assert outcome.primary is primary_result
    assert outcome.fallback is fallback_result
    assert outcome.final is fallback_result
    assert outcome.fallback_used is True
    assert fallback.starts == [primary_terminal]
    assert fallback_starts == ["started"]
    assert primary.calls == [
        {
            "fmax": 0.01,
            "maxiter": 20,
            "trajectory_callback": trajectory_states.append,
            "trajectory_stride": 3,
        }
    ]
    assert fallback.calls == primary.calls


def test_certificate_fallback_returns_uncertified_fallback_without_retrying():
    primary = _ScriptedRelaxer(
        RelaxResult(state=_state(1.0), energy=-1.0, gradient_norm=0.02, n_iter=20)
    )
    fallback_result = RelaxResult(
        state=_state(2.0),
        energy=-1.5,
        gradient_norm=0.03,
        n_iter=20,
    )
    fallback = _ScriptedRelaxer(fallback_result)

    outcome = relax_with_certificate_fallback(
        primary,
        _state(0.0),
        fmax=0.01,
        maxiter=20,
        fallback_relaxer=fallback,
    )

    assert outcome.final is fallback_result
    assert outcome.fallback_used is True
    assert len(fallback.starts) == 1
