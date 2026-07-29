"""Contract tests for the public-profile C60 200-trial production run."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pamssw import validated_ls_ssw_config


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-c60-k4-profile-200"
    / "run_production.py"
)
BASE_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)
PROFILE = "c60_direction_efficient_validated_20260729"
EXPECTED_COMMIT = "e" * 40


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeBaseRunner:
    def __init__(self) -> None:
        self._real = load_module(BASE_RUNNER_PATH, "c60_k4_real_base")
        self.build_config = self._real.build_config
        self.preflight_calls: list[dict[str, object]] = []
        self.run_calls: list[dict[str, object]] = []

    def preflight(self, *, system: str, expected_git_commit: str):
        self.preflight_calls.append(
            {"system": system, "expected_git_commit": expected_git_commit}
        )
        return {
            "schema_version": 1,
            "execution_commit": expected_git_commit,
            "system": system,
            "input_path": "/input/c60.xyz",
            "input_sha256": "a" * 64,
            "model_path": "/model/mace.model",
            "model_sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
            "runtime_versions": {"python": "test"},
            "cuda": {"available": True},
            "calculator": {"device": "cuda"},
        }

    def run(
        self,
        *,
        system: str,
        output_dir: Path,
        expected_git_commit: str,
        preflight_only: bool = False,
    ):
        assert preflight_only is False
        config = self.build_config(system, output_dir)
        self.run_calls.append(
            {
                "system": system,
                "expected_git_commit": expected_git_commit,
                "config": asdict(config),
            }
        )
        output_dir.mkdir(parents=True)
        summary = {
            "effective_config": asdict(config),
            "initial_energy_eV": -450.0,
            "best_energy_eV": -451.5,
            "energy_drop_eV": 1.5,
            "force_evaluations": 14,
            "purpose_counts": {
                "bootstrap_true_quench": 2,
                "direction_oracle": 3,
                "biased_proposal_relax": 4,
                "landing_true_quench": 5,
                "unattributed": 0,
            },
            "optimizer_telemetry": {},
            "stats": {"n_trials": 200, "force_evaluations": 14},
            "timing": {"total_wall_time_s": 12.0},
        }
        for name in (
            "best_minimum.xyz",
            "energy_trace.json",
            "walk_records.json",
            "optimizer_diagnostics.json",
        ):
            (output_dir / name).write_text("fixture\n", encoding="utf-8")
        (output_dir / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary


def test_projection_is_exact_public_profile(tmp_path):
    runner = load_module(RUNNER_PATH, "c60_k4_projection")
    output = tmp_path / "output"

    projected = runner.config_projection(output)
    expected = json.loads(
        json.dumps(
            asdict(
                validated_ls_ssw_config(
                    PROFILE,
                    output_dir=output,
                    max_trials=200,
                    rng_seed=42,
                    max_force_evals=None,
                )
            )
        )
    )

    assert projected == expected
    assert projected["oracle_candidates"] == 4
    assert projected["max_steps_per_walk"] == 8
    assert projected["proposal_relax_steps"] == 80
    assert projected["proposal_optimizer"] == "safe-lbfgs-total"
    assert projected["quench_optimizer"] == "ase-lbfgs"
    assert projected["quench_fallback_optimizer"] == "ase-fire"
    assert projected["direction_type_ucb_enabled"] is False


def test_preflight_pins_profile_commit_runtime_and_model(tmp_path, monkeypatch):
    runner = load_module(RUNNER_PATH, "c60_k4_preflight")
    base = FakeBaseRunner()
    frozen = tmp_path / "base.py"
    frozen.write_text("base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)

    checked = runner.preflight(
        output_dir=tmp_path / "output",
        expected_git_commit=EXPECTED_COMMIT,
        base_runner=base,
    )

    assert base.preflight_calls == [
        {"system": "c60", "expected_git_commit": EXPECTED_COMMIT}
    ]
    assert checked["profile"] == PROFILE
    assert checked["execution_commit"] == EXPECTED_COMMIT
    assert checked["effective_config"]["oracle_candidates"] == 4
    assert checked["profile_metadata"]["model_sha256"] == (
        checked["base_preflight"]["model_sha256"]
    )
    assert checked["base_runner"]["sha256"] == runner._sha256(frozen)


def test_run_injects_exact_profile_and_closes_accounting(tmp_path, monkeypatch):
    runner = load_module(RUNNER_PATH, "c60_k4_run")
    base = FakeBaseRunner()
    frozen = tmp_path / "base.py"
    frozen.write_text("base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)
    output = tmp_path / "output"

    summary = runner.run(
        output_dir=output,
        expected_git_commit=EXPECTED_COMMIT,
        base_runner_loader=lambda: base,
    )

    assert len(base.run_calls) == 1
    assert base.run_calls[0]["system"] == "c60"
    assert json.loads(json.dumps(base.run_calls[0]["config"])) == (
        summary["effective_config"]
    )
    assert summary["effective_config"]["oracle_candidates"] == 4
    assert summary["stats"]["n_trials"] == 200
    assert sum(summary["purpose_counts"].values()) == summary["force_evaluations"]
    assert summary["purpose_counts"]["unattributed"] == 0
    assert summary["validated_profile_run"]["profile"] == PROFILE
    assert json.loads((output / "summary.json").read_text()) == summary


def test_preflight_only_does_not_execute_or_create_output(tmp_path, monkeypatch):
    runner = load_module(RUNNER_PATH, "c60_k4_preflight_only")
    base = FakeBaseRunner()
    frozen = tmp_path / "base.py"
    frozen.write_text("base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)
    output = tmp_path / "output"

    checked = runner.run(
        output_dir=output,
        expected_git_commit=EXPECTED_COMMIT,
        preflight_only=True,
        base_runner_loader=lambda: base,
    )

    assert checked["effective_config"]["max_trials"] == 200
    assert base.run_calls == []
    assert not output.exists()


def test_cli_exposes_only_output_commit_and_preflight():
    runner = load_module(RUNNER_PATH, "c60_k4_cli")
    with pytest.raises(SystemExit):
        runner._parse_args([])

    args = runner._parse_args(
        [
            "--output",
            "run-output",
            "--expected-git-commit",
            EXPECTED_COMMIT,
            "--preflight-only",
        ]
    )

    assert args.output == Path("run-output")
    assert args.expected_git_commit == EXPECTED_COMMIT
    assert args.preflight_only is True
