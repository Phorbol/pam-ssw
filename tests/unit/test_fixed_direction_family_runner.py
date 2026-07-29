from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260729-fixed-direction-family-terminal-labels"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_case_base_runner_changes_only_the_family_generation_control(
    tmp_path,
):
    protocol = load_module(
        RUN_ROOT / "protocol.py",
        "_fixed_family_protocol_for_runner_test",
    )
    runner = load_module(
        RUN_ROOT / "run_experiment.py",
        "_fixed_family_runner_test",
    )

    @dataclass(frozen=True)
    class Config:
        n_bond_pairs: int = 2
        stagnation_bond_pair_boost: int = 2
        oracle_candidates: int = 12
        max_steps_per_walk: int = 8

    class Base:
        def build_config(self, system, case_dir):
            assert system == "c60"
            assert case_dir == tmp_path
            return Config()

        def write_state(self, path, state):
            return (path, state)

    random_case = next(
        case for case in protocol.case_matrix()
        if case.arm == "random_only"
    )
    bond_case = next(
        case for case in protocol.case_matrix()
        if case.arm == "bond_only"
    )

    random_config = runner.CaseBaseRunner(
        Base(), random_case
    ).build_config("c60", tmp_path)
    bond_config = runner.CaseBaseRunner(
        Base(), bond_case
    ).build_config("c60", tmp_path)

    assert random_config.n_bond_pairs == 0
    assert bond_config.n_bond_pairs == 4
    assert random_config.stagnation_bond_pair_boost == 0
    assert bond_config.stagnation_bond_pair_boost == 0
    assert random_config.oracle_candidates == 12
    assert bond_config.max_steps_per_walk == 8


def test_conclusion_reports_gate_without_promoting_a_selector():
    runner = load_module(
        RUN_ROOT / "run_experiment.py",
        "_fixed_family_runner_conclusion_test",
    )
    evidence = {
        "cohort": {"completed_cases": 24},
        "stable_meaningful_by_family": {
            "random_only": 2,
            "bond_only": 5,
        },
        "held_out_audit": {
            "starter_only_brier": 0.2,
            "starter_plus_family_brier": 0.1,
        },
        "posterior_gate": {
            "enter_posterior_stage": False,
            "reason": "insufficient_stable_meaningful_labels_per_family",
        },
    }

    report = runner.conclusion(evidence)

    assert "random_only: 2" in report
    assert "bond_only: 5" in report
    assert "enter_posterior_stage: `false`" in report
    assert "does not change production defaults" in report


def test_case_annotation_exposes_the_only_arm_specific_config_change():
    protocol = load_module(
        RUN_ROOT / "protocol.py",
        "_fixed_family_protocol_for_annotation_test",
    )
    runner = load_module(
        RUN_ROOT / "run_experiment.py",
        "_fixed_family_runner_annotation_test",
    )
    case = next(
        case for case in protocol.case_matrix()
        if case.arm == "bond_only"
    )
    row = {
        "source_to_effective_config_diff": {
            "oracle_candidates": [12, 4],
        }
    }

    annotated = runner.annotate_case_row(
        row,
        case=case,
        baseline_n_bond_pairs=2,
        baseline_stagnation_bond_pair_boost=2,
    )

    assert annotated["ablation_control"] == {
        "field": "n_bond_pairs",
        "source_value": 2,
        "effective_value": 4,
        "expected_direction_kind": "bond",
    }
    assert annotated["source_to_effective_config_diff"][
        "n_bond_pairs"
    ] == [2, 4]
    assert annotated["source_to_effective_config_diff"][
        "stagnation_bond_pair_boost"
    ] == [2, 0]
    assert annotated["frozen_feedback_controls"] == {
        "stagnation_bond_pair_boost": 0,
    }
