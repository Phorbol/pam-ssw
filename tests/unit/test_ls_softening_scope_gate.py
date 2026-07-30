import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260730-ls-softening-scope-gate"
    / "run_gate.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("_ls_softening_scope_gate", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LS softening scope gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_participation_ratio_has_physical_limits():
    runner = _load_runner()
    mask = np.array([True, True, True, False])

    localized = np.zeros((4, 3))
    localized[0, 0] = 1.0
    delocalized = np.zeros((4, 3))
    delocalized[:3, 0] = 1.0

    assert runner.participation_ratio(localized.reshape(-1), mask) == pytest.approx(1.0 / 3.0)
    assert runner.participation_ratio(delocalized.reshape(-1), mask) == pytest.approx(1.0)


def test_factorial_effects_separate_oracle_proposal_and_interaction():
    runner = _load_runner()
    rows = [
        {
            "system": "c60",
            "state_id": "bootstrap",
            "seed": 42,
            "scope": scope,
            "landing_delta_eV": value,
        }
        for scope, value in {
            "none": 10.0,
            "oracle": 12.0,
            "proposal": 13.0,
            "both": 20.0,
        }.items()
    ]

    effects = runner.factorial_effects(rows, "landing_delta_eV")

    assert effects == [
        {
            "system": "c60",
            "state_id": "bootstrap",
            "seed": 42,
            "oracle_main_effect": pytest.approx(4.5),
            "proposal_main_effect": pytest.approx(5.5),
            "interaction": pytest.approx(5.0),
        }
    ]


def test_factorial_effects_require_exact_four_scope_block():
    runner = _load_runner()
    rows = [
        {
            "system": "c60",
            "state_id": "bootstrap",
            "seed": 42,
            "scope": "none",
            "landing_delta_eV": 1.0,
        }
    ]

    with pytest.raises(ValueError, match="four scopes"):
        runner.factorial_effects(rows, "landing_delta_eV")

