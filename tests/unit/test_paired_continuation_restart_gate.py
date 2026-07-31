from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "20260801-paired-continuation-restart-gate"
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
RUNNER_PATH = RUN_ROOT / "run_gate.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol():
    return _load(PROTOCOL_PATH, "_paired_continuation_restart_protocol_test")


def _runner():
    return _load(RUNNER_PATH, "_paired_continuation_restart_runner_test")


def test_scr1_matrix_has_twelve_cases_and_one_shared_bootstrap_per_system():
    protocol = _protocol()

    cases = protocol.case_matrix(systems=("c60", "pdo", "cuo"), seeds=(45,))

    assert len(cases) == 12
    assert {row["starter_mode"] for row in cases} == {
        "uniform_archive",
        "archive_ucb",
        "metropolis_chain",
        "paired_best_uniform",
    }
    assert {(row["system"], row["seed"]) for row in cases} == {
        ("c60", 45),
        ("pdo", 45),
        ("cuo", 45),
    }


def test_gain_auc_integrates_best_improvement_on_complete_fe_axis():
    protocol = _protocol()
    rows = [
        {"force_evaluations": 20, "best_energy": -11.0},
        {"force_evaluations": 60, "best_energy": -13.0},
        {"force_evaluations": 80, "best_energy": -12.0},
    ]

    auc = protocol.gain_auc(
        initial_energy_eV=-10.0,
        bootstrap_force_evaluations=10,
        accepted_rows=rows,
        total_force_budget=100,
    )

    assert auc == pytest.approx((40.0 * 1.0 + 40.0 * 3.0) / 100.0)


def _decision_rows():
    rows = []
    values = {
        "c60": {"archive_ucb": 2.0, "metropolis_chain": 1.0, "paired_best_uniform": 3.0},
        "pdo": {"archive_ucb": 2.0, "metropolis_chain": 3.0, "paired_best_uniform": 4.0},
        "cuo": {"archive_ucb": 4.0, "metropolis_chain": 3.0, "paired_best_uniform": 2.0},
    }
    for system, modes in values.items():
        for mode, auc in modes.items():
            rows.append({"system": system, "seed": 45, "starter_mode": mode, "gain_auc_eV": auc})
    return rows


def test_scr1_admission_requires_strict_win_over_both_comparators_in_two_systems():
    protocol = _protocol()
    rows = _decision_rows()

    assert protocol.scr1_decision(rows) == {
        "decision": "ADMIT_S_CR2",
        "paired_winning_system_count": 2,
        "paired_winning_systems": ["c60", "pdo"],
    }

    tied = deepcopy(rows)
    next(row for row in tied if row["system"] == "pdo" and row["starter_mode"] == "paired_best_uniform")["gain_auc_eV"] = 3.0
    assert protocol.scr1_decision(tied)["decision"] == "DO_NOT_ADMIT_S_CR2"


def test_runner_changes_only_selector_log_path_and_cuo_scope(tmp_path):
    runner = _runner()

    for system in runner.SYSTEMS:
        configs = {
            mode: runner.build_config(
                system,
                tmp_path / system / mode,
                seed=45,
                starter_mode=mode,
                force_budget=20_000,
            )
            for mode in runner.STARTER_MODES
        }
        reference = configs["uniform_archive"]
        for mode, config in configs.items():
            assert config.seed_selection_mode == mode
            assert config.accepted_structures_log is not None
            differing = {
                name
                for name in config.__dataclass_fields__
                if getattr(config, name) != getattr(reference, name)
            }
            assert differing <= {
                "seed_selection_mode",
                "accepted_structures_log",
                "accepted_structures_dir",
                "direction_diagnostics_path",
                "direction_archive_path",
                "proposal_minima_dir",
                "relaxation_trajectory_dir",
            }
        if system == "cuo":
            assert reference.local_softening_scope == "oracle"
