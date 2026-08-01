from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pamssw.result import ActionRecord, UphillStepRecord, UphillWalkTrace


def _step(
    step_index: int,
    energy_before: float,
    energy_after: float,
    *,
    target_eV: float = 0.8,
    direction_oracle_force_evaluations: int = 2,
) -> UphillStepRecord:
    return UphillStepRecord(
        step_index=step_index,
        direction_kind="random",
        target_eV=target_eV,
        true_energy_before_eV=energy_before,
        true_energy_after_eV=energy_after,
        requested_sigma=0.2,
        executed_sigma=0.2,
        base_bias_weight=0.4,
        final_bias_weight=0.4,
        true_curvature=-0.1,
        inner_curvature=-0.2,
        proposal_relax_iterations=3,
        proposal_relax_outcome="useful_progress",
        proposal_relax_termination="maxiter",
        direction_oracle_force_evaluations=direction_oracle_force_evaluations,
        biased_relax_force_evaluations=4,
        true_pes_check_force_evaluations=2,
        displacement_clipped=False,
        step_termination_reason="continued",
    )


def _walk() -> UphillWalkTrace:
    return UphillWalkTrace(
        target_eV=0.8,
        termination_reason="reached_step_cap",
        steps=(
            _step(0, -10.0, -9.6),
            _step(1, -9.6, -9.2),
        ),
    )


def _action(**overrides) -> ActionRecord:
    values = {
        "trial_index": 0,
        "proposal_index": 0,
        "seed_entry_id": 0,
        "seed_energy_eV": -10.0,
        "walk": _walk(),
        "escape_energy_eV": -9.2,
        "landing_energy_eV": -10.2,
        "landing_gradient_norm": 0.04,
        "landing_iterations": 8,
        "landing_converged": True,
        "landing_force_evaluations": 10,
        "accepted_new_basin": True,
        "is_duplicate": False,
        "global_improved": True,
        "status": "accepted",
    }
    values.update(overrides)
    return ActionRecord(**values)


def test_walk_trace_derives_observed_true_pes_heights() -> None:
    trace = _walk()

    assert trace.observed_max_height_eV == pytest.approx(0.8)
    assert trace.observed_terminal_height_eV == pytest.approx(0.8)
    assert trace.target_delivery_ratio == pytest.approx(1.0)


def test_walk_trace_is_immutable() -> None:
    trace = _walk()

    with pytest.raises(FrozenInstanceError):
        trace.target_eV = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("target", [0.0, -0.1])
def test_walk_trace_rejects_non_positive_target(target: float) -> None:
    with pytest.raises(ValueError, match="target_eV must be positive"):
        UphillWalkTrace(target, "reached_step_cap", ())


def test_walk_trace_rejects_unordered_step_indices() -> None:
    with pytest.raises(ValueError, match="ordered from zero"):
        UphillWalkTrace(
            0.8,
            "reached_step_cap",
            (_step(1, -10.0, -9.6), _step(0, -9.6, -9.2)),
        )


def test_step_rejects_negative_purpose_count() -> None:
    with pytest.raises(ValueError, match="force-evaluation counts must be non-negative"):
        _step(0, -10.0, -9.6, direction_oracle_force_evaluations=-1)


def test_action_rejects_escape_energy_inconsistent_with_walk_endpoint() -> None:
    with pytest.raises(ValueError, match="escape_energy_eV must match"):
        _action(escape_energy_eV=-9.0)


def test_action_rejects_simultaneous_new_and_duplicate_outcome() -> None:
    with pytest.raises(ValueError, match="cannot be both new and duplicate"):
        _action(accepted_new_basin=True, is_duplicate=True)


def test_empty_walk_has_no_derived_height() -> None:
    trace = UphillWalkTrace(0.8, "budget_exhausted", ())

    assert trace.observed_max_height_eV is None
    assert trace.observed_terminal_height_eV is None
    assert trace.target_delivery_ratio is None

