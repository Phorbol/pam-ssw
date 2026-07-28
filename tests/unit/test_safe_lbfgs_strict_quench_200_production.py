"""Contract tests for the strict-true-quench 200-trial production wrapper."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-safe-lbfgs-strict-quench-200-production"
)
RUNNER_PATH = ROOT / "run_production.py"
BASE_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)
EXPECTED_COMMIT = "e" * 40


def load_runner(name: str = "safe_lbfgs_strict_quench_200"):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeBaseRunner:
    """A tiny frozen-runner surrogate that observes injected configuration."""

    def __init__(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "strict_quench_real_base", BASE_RUNNER_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self._real = module
        self.build_config = self._build_config
        self.run_calls: list[dict[str, object]] = []
        self.preflight_calls: list[dict[str, object]] = []
        self.raise_during_run = False

    def _build_config(self, system: str, case_dir: Path):
        return self._real.build_config(system, case_dir)

    def preflight(self, *, system: str, expected_git_commit: str):
        self.preflight_calls.append(
            {"system": system, "expected_git_commit": expected_git_commit}
        )
        return {
            "schema_version": 1,
            "execution_commit": expected_git_commit,
            "system": system,
            "input_path": f"/input/{system}.xyz",
            "input_sha256": "a" * 64,
            "model_path": "/model/mace.model",
            "model_sha256": "b" * 64,
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
                "output_dir": Path(output_dir),
                "expected_git_commit": expected_git_commit,
                "config": asdict(config),
            }
        )
        if self.raise_during_run:
            raise RuntimeError("frozen runner failed")
        Path(output_dir).mkdir(parents=True)
        purpose_counts = {
            "bootstrap_true_quench": 2,
            "direction_oracle": 3,
            "biased_proposal_relax": 4,
            "landing_true_quench": 5,
            "unattributed": 0,
        }
        summary = {
            "effective_config": asdict(config),
            "force_evaluations": 14,
            "purpose_counts": purpose_counts,
            "optimizer_telemetry": {
                "quench_fallback_attempts": 1,
                "quench_fallback_converged": 1,
            },
            "stats": {"n_trials": 200, "force_evaluations": 14},
        }
        for name in (
            "best_minimum.xyz",
            "energy_trace.json",
            "walk_records.json",
            "optimizer_diagnostics.json",
        ):
            (Path(output_dir) / name).write_text("fixture\n", encoding="utf-8")
        (Path(output_dir) / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        return summary


def test_per_system_effective_config_has_only_the_strict_quench_diffs(tmp_path):
    runner = load_runner("strict_quench_config_contract")
    base = runner._base_runner()

    expected_diffs = {
        "c60": {
            "quench_fallback_optimizer": [None, "ase-fire"],
            "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
        },
        "pdo": {
            "quench_fallback_optimizer": [None, "ase-fire"],
            "quench_fmax": [0.03, 0.01],
            "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
        },
    }
    for system, expected_diff in expected_diffs.items():
        source, effective, diff = runner.config_projection(
            system, tmp_path / system, base
        )
        assert diff == expected_diff
        assert source["max_trials"] == effective["max_trials"] == 200
        assert source["rng_seed"] == effective["rng_seed"] == 42
        assert (
            source["proposal_optimizer"]
            == effective["proposal_optimizer"]
            == "safe-lbfgs-total"
        )
        assert source["proposal_fmax"] == effective["proposal_fmax"] == pytest.approx(
            0.05
        )
        assert source["quench_maxiter"] == effective["quench_maxiter"] == 400
        assert effective["quench_optimizer"] == "ase-lbfgs"
        assert effective["quench_fallback_optimizer"] == "ase-fire"
        assert effective["quench_fmax"] == pytest.approx(0.01)


def test_preflight_pins_wrapper_commit_delegates_runtime_and_records_hash_and_configs(
    tmp_path, monkeypatch
):
    runner = load_runner("strict_quench_preflight_contract")
    base = FakeBaseRunner()
    frozen_path = tmp_path / "frozen_base.py"
    frozen_path.write_text("frozen base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen_path)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)

    checked = runner.preflight(
        system="pdo",
        output_dir=tmp_path / "output",
        expected_git_commit=EXPECTED_COMMIT,
        base_runner=base,
    )

    assert base.preflight_calls == [
        {"system": "pdo", "expected_git_commit": EXPECTED_COMMIT}
    ]
    assert checked["execution_commit"] == EXPECTED_COMMIT
    assert checked["base_runner"]["path"] == str(frozen_path)
    assert checked["base_runner"]["sha256"] == runner._sha256(frozen_path)
    assert checked["source_config"]["quench_fmax"] == pytest.approx(0.03)
    assert checked["effective_config"]["quench_fmax"] == pytest.approx(0.01)
    assert checked["config_diff"] == {
        "quench_fallback_optimizer": [None, "ase-fire"],
        "quench_fmax": [0.03, 0.01],
        "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
    }
    assert checked["base_preflight"]["cuda"] == {"available": True}

    monkeypatch.setattr(runner, "_current_commit", lambda: "f" * 40)
    with pytest.raises(RuntimeError, match="execution commit mismatch"):
        runner.preflight(
            system="pdo",
            output_dir=tmp_path / "other-output",
            expected_git_commit=EXPECTED_COMMIT,
            base_runner=base,
        )


def test_run_delegates_once_injects_strict_config_and_rewrites_augmented_summary(
    tmp_path, monkeypatch
):
    runner = load_runner("strict_quench_run_contract")
    base = FakeBaseRunner()
    frozen_path = tmp_path / "frozen_base.py"
    frozen_path.write_text("frozen base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen_path)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)
    output = tmp_path / "output"
    checked = runner.preflight(
        system="pdo",
        output_dir=output,
        expected_git_commit=EXPECTED_COMMIT,
        base_runner=base,
    )
    original_build_config = base.build_config

    summary = runner.run(
        system="pdo",
        output_dir=output,
        expected_git_commit=EXPECTED_COMMIT,
        base_runner_loader=lambda: base,
        preflight_fn=lambda **_: checked,
    )

    assert base.build_config is original_build_config
    assert len(base.run_calls) == 1
    assert runner._json_mapping(base.run_calls[0]["config"]) == checked[
        "effective_config"
    ]
    assert summary["effective_config"] == checked["effective_config"]
    assert summary["stats"]["n_trials"] == 200
    assert sum(summary["purpose_counts"].values()) == summary["force_evaluations"]
    assert summary["purpose_counts"]["unattributed"] == 0
    assert summary["strict_quench_wrapper"] == {
        "schema_version": 1,
        "base_runner": checked["base_runner"],
        "base_preflight": checked["base_preflight"],
        "source_config": checked["source_config"],
        "effective_config": checked["effective_config"],
        "overrides": runner.STRICT_QUENCH_PROTOCOL,
        "config_diff": checked["config_diff"],
        "fallback_counts": {"attempts": 1, "converged": 1},
    }
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    for name in runner.EXPECTED_OUTPUT_FILES:
        assert (output / name).is_file()
    assert not (output / "summary.json.tmp").exists()

    with pytest.raises(FileExistsError):
        runner.run(
            system="pdo",
            output_dir=output,
            expected_git_commit=EXPECTED_COMMIT,
            base_runner_loader=lambda: base,
            preflight_fn=lambda **_: checked,
        )


def test_injected_build_config_is_restored_when_the_frozen_runner_fails(
    tmp_path, monkeypatch
):
    runner = load_runner("strict_quench_restore_failure")
    base = FakeBaseRunner()
    base.raise_during_run = True
    frozen_path = tmp_path / "frozen_base.py"
    frozen_path.write_text("frozen base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen_path)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)
    output = tmp_path / "output"
    checked = runner.preflight(
        system="c60",
        output_dir=output,
        expected_git_commit=EXPECTED_COMMIT,
        base_runner=base,
    )
    original_build_config = base.build_config

    with pytest.raises(RuntimeError, match="frozen runner failed"):
        runner.run(
            system="c60",
            output_dir=output,
            expected_git_commit=EXPECTED_COMMIT,
            base_runner_loader=lambda: base,
            preflight_fn=lambda **_: checked,
        )
    assert base.build_config is original_build_config


def test_preflight_only_writes_nothing_and_does_not_delegate_execution(
    tmp_path, monkeypatch
):
    runner = load_runner("strict_quench_preflight_only")
    base = FakeBaseRunner()
    frozen_path = tmp_path / "frozen_base.py"
    frozen_path.write_text("frozen base", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_RUNNER_PATH", frozen_path)
    monkeypatch.setattr(runner, "_current_commit", lambda: EXPECTED_COMMIT)
    output = tmp_path / "output"

    checked = runner.run(
        system="c60",
        output_dir=output,
        expected_git_commit=EXPECTED_COMMIT,
        preflight_only=True,
        base_runner_loader=lambda: base,
    )
    assert checked["effective_config"]["quench_optimizer"] == "ase-lbfgs"
    assert base.run_calls == []
    assert not output.exists()


def test_cli_requires_system_output_and_execution_commit():
    runner = load_runner("strict_quench_cli")
    with pytest.raises(SystemExit):
        runner._parse_args([])
    args = runner._parse_args(
        [
            "--system",
            "c60",
            "--output",
            "run-output",
            "--expected-git-commit",
            EXPECTED_COMMIT,
            "--preflight-only",
        ]
    )
    assert args.system == "c60"
    assert args.output == Path("run-output")
    assert args.preflight_only is True
