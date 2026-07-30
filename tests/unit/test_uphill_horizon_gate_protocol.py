from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260731-uphill-horizon-gate" / "run_gate.py"
)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "uphill_horizon_gate_runner",
        RUNNER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_paired_config_diff_allows_only_horizon():
    runner = _runner()

    assert runner.paired_config_diff(
        {"max_steps_per_walk": 8, "proposal_relax_steps": 80},
        {"max_steps_per_walk": 14, "proposal_relax_steps": 80},
    ) == {"max_steps_per_walk": (8, 14)}


def test_paired_config_diff_rejects_second_mechanism_change():
    runner = _runner()

    with pytest.raises(
        ValueError,
        match="differ only in max_steps_per_walk",
    ):
        runner.paired_config_diff(
            {"max_steps_per_walk": 8, "proposal_relax_steps": 80},
            {"max_steps_per_walk": 14, "proposal_relax_steps": 120},
        )


def test_walk_termination_summary_requires_exactly_one_reason():
    runner = _runner()
    complete = {
        "walk_terminations": 1,
        "walk_termination_reached_step_cap": 1,
        "walk_termination_walk_displacement_clipped": 0,
        "walk_termination_last_reason": "reached_step_cap",
    }

    assert runner.walk_termination_reason(complete) == "reached_step_cap"
    with pytest.raises(ValueError, match="exactly one"):
        runner.walk_termination_reason(
            {
                "walk_terminations": 1,
                "walk_termination_reached_step_cap": 1,
                "walk_termination_walk_displacement_clipped": 1,
            }
        )
