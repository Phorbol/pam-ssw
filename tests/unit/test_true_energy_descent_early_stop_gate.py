from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pamssw.accounting import EvalCounter
from pamssw.state import State


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT / "runs" / "20260801-true-energy-descent-early-stop-gate"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
RUNNER_PATH = RUN_ROOT / "run_gate.py"
MANIFEST_PATH = RUN_ROOT / "manifest.json"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load(PROTOCOL_PATH, "_true_energy_descent_protocol_test")


def _runner():
    return _load(RUNNER_PATH, "_true_energy_descent_runner_test")


def test_accepted_steps_exclude_rejected_attempts():
    protocol = _protocol()
    case = {"reached_macro_steps": 2, "attempted_macro_steps": 3}

    assert protocol.accepted_step_indices(case) == (1, 2)


def test_accepted_steps_reject_inconsistent_counts():
    protocol = _protocol()

    with pytest.raises(ValueError, match="reached/attempted"):
        protocol.accepted_step_indices(
            {"reached_macro_steps": 3, "attempted_macro_steps": 2}
        )


def test_first_crossing_is_strict_and_does_not_look_ahead():
    protocol = _protocol()
    rows = [
        {"step": 1, "checkpoint_delta_eV": 0.2},
        {"step": 2, "checkpoint_delta_eV": -0.001},
        {"step": 3, "checkpoint_delta_eV": -0.4},
    ]

    assert protocol.first_descent_crossing(
        rows,
        tolerance=0.001,
    ) == rows[2]


def test_first_crossing_requires_consecutive_accepted_steps():
    protocol = _protocol()
    rows = [
        {"step": 1, "checkpoint_delta_eV": 0.2},
        {"step": 3, "checkpoint_delta_eV": -0.4},
    ]

    with pytest.raises(ValueError, match="consecutive"):
        protocol.first_descent_crossing(rows, tolerance=0.001)


def test_outcome_keeps_overshoot_and_deeper_terminal_separate():
    protocol = _protocol()

    assert protocol.classify_tradeoff(-7.0, 0.2, 0.001) == (
        "AVOIDED_OVERSHOOT"
    )
    assert protocol.classify_tradeoff(-7.0, -9.0, 0.001) == (
        "FORGONE_DEEPER_TERMINAL"
    )
    assert protocol.classify_tradeoff(-7.0, -7.0, 0.001) == (
        "ENERGY_EQUIVALENT"
    )
    assert protocol.classify_tradeoff(None, -7.0, 0.001) == "UNLEARNABLE"


def test_manifest_uses_only_reached_steps(tmp_path):
    protocol = _protocol()
    case_dir = tmp_path / "case"
    checkpoint_dir = case_dir / "macro_checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for step in (1, 2, 3):
        (checkpoint_dir / f"step{step:03d}_checkpoint.xyz").write_text(
            f"frame {step}\n",
            encoding="utf-8",
        )
    source_case = {
        "system": "c60",
        "state_id": "plateau_accepted",
        "seed": 42,
        "arm": "D0_exact_anchor",
        "reached_macro_steps": 2,
        "attempted_macro_steps": 3,
    }

    rows = protocol.project_manifest_case(
        source_case,
        case_dir,
        root=tmp_path,
    )

    assert [row["step"] for row in rows] == [1, 2]
    assert [row["checkpoint_path"] for row in rows] == [
        "case/macro_checkpoints/step001_checkpoint.xyz",
        "case/macro_checkpoints/step002_checkpoint.xyz",
    ]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in rows)


def test_frozen_manifest_closes_exact_cohort():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["case_count"] == 24
    assert manifest["accepted_endpoint_count"] == 82
    assert manifest["attempted_endpoint_count"] == 93
    assert len(manifest["cases"]) == 24
    assert sum(len(case["endpoints"]) for case in manifest["cases"]) == 82


def test_existing_checkpoint_energy_reuses_zero_force_evaluations():
    runner = _runner()

    row = runner.reused_energy_row(
        {
            "horizon": 2,
            "checkpoint_energy_eV": -10.5,
            "checkpoint_delta_eV": -0.5,
        }
    )

    assert row["step"] == 2
    assert row["new_force_evaluations"] == 0
    assert row["evidence_origin"] == "reused_first_passage"


class _FakeCalculator:
    def evaluate(self, _state):
        return SimpleNamespace(energy=-10.5)


def test_missing_energy_costs_one_true_pes_check():
    runner = _runner()
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    calculator = EvalCounter(_FakeCalculator())

    row = runner.evaluate_missing_energy(
        state,
        starter_energy=-10.0,
        calculator=calculator,
    )

    assert row["checkpoint_energy_eV"] == pytest.approx(-10.5)
    assert row["checkpoint_delta_eV"] == pytest.approx(-0.5)
    assert row["new_force_evaluations"] == 1
    assert row["purpose_counts"]["escape_true_pes_check"] == 1
    assert row["purpose_counts"]["direction_oracle"] == 0
    assert row["purpose_counts"]["biased_proposal_relax"] == 0
    assert row["purpose_counts"]["unattributed"] == 0


def test_source_validation_closes_manifest_before_runtime():
    runner = _runner()

    context = runner.validate_source_inputs()

    assert len(context["cases_by_key"]) == 24
    assert context["accepted_endpoint_count"] == 82
    assert context["attempted_endpoint_count"] == 93
    assert context["source_raw_sha256"] == context["manifest"][
        "source_raw_evidence_sha256"
    ]


def test_existing_first_crossing_quench_is_reused():
    runner = _runner()
    source_checkpoint = {
        "horizon": 2,
        "label": "ESCAPED_CERTIFIED",
        "landing_energy_eV": -17.0,
        "landing_delta_eV": -7.0,
        "landing_path": "landing.xyz",
        "landing_sha256": "a" * 64,
    }

    row = runner.reused_quench_row(source_checkpoint)

    assert row["step"] == 2
    assert row["new_force_evaluations"] == 0
    assert row["evidence_origin"] == "reused_first_passage"


def test_only_crossing_and_terminal_can_request_new_quench():
    runner = _runner()

    assert runner.required_quench_steps(
        first_crossing=3,
        terminal=8,
        reusable={3},
    ) == (8,)
    assert runner.required_quench_steps(
        first_crossing=3,
        terminal=3,
        reusable=set(),
    ) == (3,)
    assert runner.required_quench_steps(
        first_crossing=None,
        terminal=8,
        reusable=set(),
    ) == ()


def test_case_summary_separates_certificate_savings_and_energy_tradeoff():
    runner = _runner()
    energies = [
        {"step": 1, "checkpoint_delta_eV": 0.2},
        {"step": 2, "checkpoint_delta_eV": -0.5},
        {"step": 3, "checkpoint_delta_eV": 0.1},
    ]
    quenches = {
        2: {
            "step": 2,
            "label": "ESCAPED_CERTIFIED",
            "landing_delta_eV": -7.0,
        },
        3: {
            "step": 3,
            "label": "ESCAPED_CERTIFIED",
            "landing_delta_eV": 0.2,
        },
    }

    summary = runner.summarize_case_outcome(
        energy_rows=energies,
        quench_rows=quenches,
        tolerance=0.001,
    )

    assert summary["first_crossing_step"] == 2
    assert summary["saved_outer_micro_steps"] == 1
    assert summary["crossing_is_certified_lower_basin"] is True
    assert summary["tradeoff_class"] == "AVOIDED_OVERSHOOT"
