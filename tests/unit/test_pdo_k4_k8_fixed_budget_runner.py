"""Contract tests for the PdO K4/K8 fixed-budget GPU experiment."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-pdo-k4-k8-fixed-budget"
    / "run_arm.py"
)


def load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_projection_changes_only_budget_seed_and_candidate_count(tmp_path):
    runner = load_runner("pdo_k4_k8_projection")
    base = runner._base_runner()

    source, k4, k4_diff = runner.config_projection(
        arm="k4",
        seed=43,
        output_dir=tmp_path / "same-output",
        base_runner=base,
    )
    _, k8, k8_diff = runner.config_projection(
        arm="k8",
        seed=43,
        output_dir=tmp_path / "same-output",
        base_runner=base,
    )

    assert source["oracle_candidates"] == 8
    assert k4["oracle_candidates"] == 4
    assert k8["oracle_candidates"] == 8
    assert k4["max_force_evals"] == k8["max_force_evals"] == 20_000
    assert k4["max_trials"] == k8["max_trials"] == 200
    assert k4["rng_seed"] == k8["rng_seed"] == 43
    assert k4["max_steps_per_walk"] == k8["max_steps_per_walk"] == 8
    assert k4["proposal_optimizer"] == k8["proposal_optimizer"] == (
        "safe-lbfgs-total"
    )
    assert k4["proposal_relax_steps"] == k8["proposal_relax_steps"] == 300
    assert k4["quench_optimizer"] == k8["quench_optimizer"] == "scipy-lbfgsb"
    assert k4["quench_fmax"] == k8["quench_fmax"] == pytest.approx(0.03)
    assert k4["direction_type_ucb_enabled"] is False
    assert k8["direction_type_ucb_enabled"] is False
    assert k4_diff == {
        "max_force_evals": [None, 20_000],
        "oracle_candidates": [8, 4],
        "rng_seed": [42, 43],
    }
    assert k8_diff == {
        "max_force_evals": [None, 20_000],
        "rng_seed": [42, 43],
    }
    differing = {
        field: [k8[field], k4[field]]
        for field in k4
        if k4[field] != k8[field]
    }
    assert differing == {"oracle_candidates": [8, 4]}


def test_projection_rejects_unknown_arm_or_seed(tmp_path):
    runner = load_runner("pdo_k4_k8_bad_projection")
    base = runner._base_runner()
    with pytest.raises(ValueError, match="unknown arm"):
        runner.config_projection(
            arm="k2",
            seed=42,
            output_dir=tmp_path,
            base_runner=base,
        )
    with pytest.raises(ValueError, match="seed"):
        runner.config_projection(
            arm="k4",
            seed=41,
            output_dir=tmp_path,
            base_runner=base,
        )


def test_cli_requires_registered_arm_seed_output_and_commit():
    runner = load_runner("pdo_k4_k8_cli")
    with pytest.raises(SystemExit):
        runner._parse_args([])
    args = runner._parse_args(
        [
            "--arm",
            "k4",
            "--seed",
            "42",
            "--output",
            "run-output",
            "--expected-git-commit",
            "a" * 40,
            "--preflight-only",
        ]
    )
    assert args.arm == "k4"
    assert args.seed == 42
    assert args.output == Path("run-output")
    assert args.preflight_only is True
