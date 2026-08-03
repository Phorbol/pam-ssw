from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "runs/20260803-bias-history-action-gate/run_gate.py"
ANALYZER_PATH = ROOT / "runs/20260803-bias-history-action-gate/analyze.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pair_cost_charges_shared_prefix_once_and_fully_loads_each_arm():
    runner = load(RUNNER_PATH, "_bias_history_runner_test")
    result = runner.split_pair_cost(
        starter_counts={"starter_true_quench": 2, "unattributed": 0},
        shared_prefix_counts={"direction_oracle": 8, "biased_proposal_relax": 12},
        arm_counts={
            "cumulative": {"biased_proposal_relax": 100, "landing_true_quench": 20},
            "newest_only": {"biased_proposal_relax": 80, "landing_true_quench": 25},
        },
    )
    assert result["shared_prefix_force_evaluations"] == 20
    assert result["pair_force_evaluations"] == 247
    assert result["fully_loaded_force_evaluations"] == {
        "cumulative": 140,
        "newest_only": 125,
    }


def test_pairing_summary_separates_support_from_energy_ordering():
    analyzer = load(ANALYZER_PATH, "_bias_history_analyzer_test")
    rows = [
        {
            "system": "c60",
            "starter_context": "bootstrap",
            "seed": 55,
            "operator_family": "cumulative",
            "certified": True,
            "same_starter_basin": False,
            "landing_delta_eV": -1.0,
        },
        {
            "system": "c60",
            "starter_context": "bootstrap",
            "seed": 55,
            "operator_family": "newest_only",
            "certified": True,
            "same_starter_basin": True,
            "landing_delta_eV": -2.0,
        },
    ]
    summary = analyzer.pairing_summary(rows)
    assert summary["cumulative_only_escape"] == 1
    assert summary["newest_only_escape"] == 0
    assert summary["both_escape"] == 0
    assert summary["neither_escape"] == 0
    assert summary["newest_lower_energy"] == 1
