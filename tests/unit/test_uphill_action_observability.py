from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pamssw.result import ActionRecord, UphillStepRecord, UphillWalkTrace


ROOT = Path(__file__).resolve().parents[2]
ANALYZER_PATH = ROOT / "runs" / "20260801-uphill-action-observability" / "analyze.py"


def _load_analyzer():
    name = "_uphill_action_observability_analysis_test"
    spec = importlib.util.spec_from_file_location(name, ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyze = _load_analyzer()


def _walk(*, one_step: bool = False) -> UphillWalkTrace:
    steps = (
        UphillStepRecord(
            step_index=0,
            direction_kind="random",
            target_eV=0.8,
            true_energy_before_eV=-10.0,
            true_energy_after_eV=-9.6,
            requested_sigma=0.2,
            executed_sigma=0.2,
            base_bias_weight=0.4,
            final_bias_weight=0.4,
            true_curvature=-0.1,
            inner_curvature=-0.2,
            proposal_relax_iterations=3,
            proposal_relax_outcome="useful_progress",
            proposal_relax_termination="maxiter",
            direction_oracle_force_evaluations=2,
            biased_relax_force_evaluations=0,
            true_pes_check_force_evaluations=2,
            displacement_clipped=False,
            step_termination_reason="continued" if not one_step else "reached_step_cap",
        ),
    )
    if not one_step:
        steps += (
            UphillStepRecord(
                **{
                    **steps[0].__dict__,
                    "step_index": 1,
                    "true_energy_before_eV": -9.6,
                    "true_energy_after_eV": -9.2,
                    "step_termination_reason": "reached_step_cap",
                }
            ),
        )
    return UphillWalkTrace(0.8, "reached_step_cap", steps)


def _record(**overrides) -> ActionRecord:
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


def _summary(*, unattributed: int = 0, force_evaluations: int = 18):
    return {
        "system": "c60",
        "seed": 49,
        "force_evaluations": force_evaluations,
        "purpose_counts": {
            "bootstrap_true_quench": 0,
            "starter_true_quench": 0,
            "local_softening_pre_relax": 0,
            "direction_oracle": 4,
            "escape_true_pes_check": 4,
            "biased_proposal_relax": 0,
            "landing_true_quench": 10,
            "post_relax_validation": 0,
            "unattributed": unattributed,
        },
    }


def test_serialize_action_contains_no_state_or_coordinate_arrays() -> None:
    record = _record()
    assert isinstance(record, ActionRecord)

    row = analyze.serialize_action(record)

    assert row["walk"]["target_delivery_ratio"] == pytest.approx(1.0)
    rendered = json.dumps(row).lower()
    assert "positions" not in rendered
    assert "numbers" not in rendered
    assert '"direction":' not in rendered


def test_analyze_case_uses_exact_ratio_boundary_and_landing_drop() -> None:
    delivered = analyze.serialize_action(_record())
    unattained = analyze.serialize_action(
        _record(
            proposal_index=1,
            walk=_walk(one_step=True),
            escape_energy_eV=-9.6,
            accepted_new_basin=False,
            is_duplicate=True,
            global_improved=False,
            status="duplicate",
        )
    )

    result = analyze.analyze_case(_summary(), [delivered, unattained])

    assert result["delivered_action_count"] == 1
    assert result["unattained_action_count"] == 1
    assert result["new_basin_rate"] == pytest.approx(0.5)
    assert result["duplicate_rate"] == pytest.approx(0.5)
    assert result["landing_quench_drop_eV"]["max"] == pytest.approx(1.0)


def test_analyzer_rejects_missing_actions_duplicate_keys_and_open_ledgers() -> None:
    row = analyze.serialize_action(_record())
    with pytest.raises(ValueError, match="no completed action"):
        analyze.analyze_case(_summary(), [])
    with pytest.raises(ValueError, match="duplicate action key"):
        analyze.analyze_cases([(_summary(), [row, row], "hash")])
    with pytest.raises(ValueError, match="does not close"):
        analyze.analyze_case(_summary(force_evaluations=19), [row])
    with pytest.raises(ValueError, match="unattributed"):
        analyze.analyze_case(_summary(unattributed=1, force_evaluations=19), [row])
