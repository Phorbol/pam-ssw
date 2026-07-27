"""Contract tests for the fixed-total-FE proposal-fmax GPU wrapper."""

from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from pamssw.config import SSWConfig


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-proposal-fmax-end-to-end-gpu-ablation"
    / "run_ablation.py"
)


def _runner_module():
    module_name = "_proposal_fmax_end_to_end_gpu_ablation_test_runner"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _frozen_projection(config: SSWConfig) -> dict[str, object]:
    values = asdict(config)
    return {
        "source_config_type": "LSSSWConfig",
        "source_config": {"production": "frozen"},
        "effective_ssw_config": values,
        "softening_enabled": False,
        "removed_ls_fields": ["local_softening_strength"],
        "overrides": {"frozen": "posterior-harness"},
    }


def _frozen_manifest(runner, *, systems=("c60", "pdo")) -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_commit": "a" * 40,
        "systems": list(systems),
        "master_seeds": [42],
        "policies": ["uniform", "posterior_proportional", "minimal_ucb"],
        "batch_size": 1,
        "max_workers": 1,
        "action_force_budget": 30,
        "total_force_budget": 120,
        "inputs": {system: {"path": system, "sha256": "b" * 64} for system in systems},
        "model": {"path": "model", "sha256": "c" * 64},
        "calculator": {"device": "cuda"},
        "runtime_versions": {"python": "3"},
        "cuda": {"available": True},
        "projections": {},
    }


def _base_config() -> SSWConfig:
    return SSWConfig(
        proposal_optimizer="safe-lbfgs-total",
        proposal_fmax=0.05,
        proposal_pool_size=1,
        quench_optimizer="ase-lbfgs",
        quench_fallback_optimizer="ase-fire",
        quench_fmax=0.01,
        quench_maxiter=400,
    )


def test_arm_projection_changes_one_and_only_one_scientific_config_value(tmp_path):
    runner = _runner_module()
    posterior = runner._posterior_runner()
    base = _base_config()
    original = posterior.build_ssw_config
    try:
        posterior.build_ssw_config = lambda system, case_directory: (
            base,
            _frozen_projection(base),
        )
        strict, strict_projection = runner.build_ssw_config(
            "c60", tmp_path / "strict", proposal_fmax=0.05
        )
        loose, loose_projection = runner.build_ssw_config(
            "c60", tmp_path / "loose", proposal_fmax=0.10
        )
    finally:
        posterior.build_ssw_config = original

    changed = {
        key: (asdict(strict)[key], asdict(loose)[key])
        for key in asdict(strict)
        if asdict(strict)[key] != asdict(loose)[key]
    }
    assert changed == {"proposal_fmax": (0.05, 0.10)}
    assert strict_projection["arm_overrides"] == {"proposal_fmax": 0.05}
    assert loose_projection["arm_overrides"] == {"proposal_fmax": 0.10}
    assert strict_projection["source_config"] == {"production": "frozen"}
    assert strict_projection["proposal_protocol"] == {
        "optimizer": "safe-lbfgs-total",
        "history_limit": 10,
        "proposal_pool_size": 1,
        "softening_enabled": False,
    }
    assert strict_projection["true_quench_protocol"] == {
        "optimizer": "ase-lbfgs",
        "fallback_optimizer": "ase-fire",
        "fmax_eV_per_A": 0.01,
        "maxiter": 400,
    }
    assert strict_projection["nonoperative_config_outputs"] == {
        "accepted_structures_log": None,
        "accepted_structures_dir": None,
        "write_proposal_minima": False,
        "proposal_minima_dir": None,
        "write_relaxation_trajectories": False,
        "relaxation_trajectory_dir": None,
        "direction_diagnostics_enabled": False,
        "direction_diagnostics_path": None,
        "direction_archive_enabled": False,
        "direction_archive_path": None,
    }


@pytest.mark.parametrize("arms", ((0.05, 0.05), (0.05, 0.20), (0.05,), (float("nan"), 0.10)))
def test_only_the_two_distinct_prespecified_arms_are_valid(arms):
    runner = _runner_module()

    with pytest.raises(ValueError):
        runner._validated_arms(arms)


def test_preflight_reuses_frozen_harness_checks_and_records_full_arm_projections(
    tmp_path, monkeypatch
):
    runner = _runner_module()
    posterior = runner._posterior_runner()
    calls = []
    base = _base_config()

    def frozen_preflight(**kwargs):
        calls.append(kwargs)
        return _frozen_manifest(runner)

    monkeypatch.setattr(posterior, "preflight", frozen_preflight)
    monkeypatch.setattr(
        posterior,
        "build_ssw_config",
        lambda system, case_directory: (base, _frozen_projection(base)),
    )
    monkeypatch.setattr(runner, "_safe_lbfgs_history_limit", lambda: 10)

    manifest = runner.preflight(
        expected_git_commit="a" * 40,
        systems=("c60", "pdo"),
        master_seeds=(42,),
        action_force_budget=30,
        total_force_budget=120,
        arms=(0.05, 0.10),
    )

    assert calls == [
        {
            "expected_git_commit": "a" * 40,
            "systems": ("c60", "pdo"),
            "master_seeds": (42,),
            "action_force_budget": 30,
            "total_force_budget": 120,
        }
    ]
    assert manifest["policy_name"] == "uniform"
    assert manifest["batch_size"] == manifest["max_workers"] == 1
    assert manifest["arms"] == [
        {"arm_index": 0, "arm_id": "proposal-fmax-0.05", "proposal_fmax": 0.05},
        {"arm_index": 1, "arm_id": "proposal-fmax-0.10", "proposal_fmax": 0.10},
    ]
    assert len(manifest["matrix"]) == 4
    for projections in manifest["projections"].values():
        assert len(projections) == 2
        assert projections[0]["effective_ssw_config"]["proposal_fmax"] == 0.05
        assert projections[1]["effective_ssw_config"]["proposal_fmax"] == 0.10
        assert projections[0]["arm_overrides"] == {"proposal_fmax": 0.05}
        assert projections[1]["arm_overrides"] == {"proposal_fmax": 0.10}


def test_atomic_matrix_publish_is_exclusive_and_delegates_to_frozen_run_campaign(
    tmp_path, monkeypatch
):
    runner = _runner_module()
    posterior = runner._posterior_runner()
    base = _base_config()
    called = []
    monkeypatch.setattr(
        posterior,
        "preflight",
        lambda **kwargs: _frozen_manifest(runner),
    )
    monkeypatch.setattr(
        posterior,
        "build_ssw_config",
        lambda system, case_directory: (base, _frozen_projection(base)),
    )
    monkeypatch.setattr(runner, "_safe_lbfgs_history_limit", lambda: 10)
    monkeypatch.setattr(posterior, "_mace_calculator_factory", lambda: object())
    monkeypatch.setattr(posterior, "_load_production_runner", lambda: type(
        "Production", (), {"load_state": staticmethod(lambda system: {"system": system})}
    ))

    def frozen_run_campaign(**kwargs):
        config = kwargs["ssw_config"]
        exploration = kwargs["exploration_config"]
        case_directory = exploration.run_directory
        case_directory.mkdir(parents=True, exist_ok=True)
        action_metrics = case_directory / "action_metrics.jsonl"
        event_log = case_directory / "events.jsonl"
        action_metrics.write_text('{"record_type": "metric"}\n', encoding="utf-8")
        event_log.write_text('{"record_type": "attempt"}\n', encoding="utf-8")
        (case_directory / "campaign_summary.json").write_text(
            json.dumps(
                {
                    "action_metrics_path": str(action_metrics),
                    "event_log_path": str(event_log),
                }
            ),
            encoding="utf-8",
        )
        called.append(
            (exploration.policy_name, exploration.action_force_budget,
             exploration.total_force_budget, config.proposal_fmax,
             exploration.run_directory)
        )
        return {"schema_version": 1, "delegated": True}

    monkeypatch.setattr(posterior, "run_campaign", frozen_run_campaign)
    output = tmp_path / "matrix"
    index = runner.run_ablation(
        output_root=output,
        expected_git_commit="a" * 40,
        systems=("c60", "pdo"),
        master_seeds=(42,),
        action_force_budget=30,
        total_force_budget=120,
        arms=(0.05, 0.10),
    )

    assert [row[:4] for row in called] == [
        ("uniform", 30, 120, 0.05),
        ("uniform", 30, 120, 0.10),
        ("uniform", 30, 120, 0.05),
        ("uniform", 30, 120, 0.10),
    ]
    assert output.is_dir()
    assert not output.with_name(".matrix.partial").exists()
    assert len(index["campaigns"]) == len(index["manifest"]["matrix"]) == 4
    for campaign in index["campaigns"]:
        assert campaign["arm_id"] in campaign["campaign_summary_path"]
        assert Path(campaign["campaign_summary_path"]).is_relative_to(output)
        summary = json.loads(
            Path(campaign["campaign_summary_path"]).read_text(encoding="utf-8")
        )
        for key in ("action_metrics_path", "event_log_path"):
            recorded_path = Path(summary[key])
            assert recorded_path.is_file()
            assert recorded_path.is_relative_to(output)
            assert ".matrix.partial" not in str(recorded_path)
    saved = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert saved == index
    with pytest.raises(FileExistsError, match="output root already exists"):
        runner.run_ablation(
            output_root=output,
            expected_git_commit="a" * 40,
            systems=("c60", "pdo"),
            master_seeds=(42,),
            action_force_budget=30,
            total_force_budget=120,
            arms=(0.05, 0.10),
        )


def test_preflight_only_never_creates_an_output_or_staging_directory(tmp_path, monkeypatch):
    runner = _runner_module()
    posterior = runner._posterior_runner()
    base = _base_config()
    monkeypatch.setattr(posterior, "preflight", lambda **kwargs: _frozen_manifest(runner))
    monkeypatch.setattr(
        posterior,
        "build_ssw_config",
        lambda system, case_directory: (base, _frozen_projection(base)),
    )
    monkeypatch.setattr(runner, "_safe_lbfgs_history_limit", lambda: 10)
    output = tmp_path / "preflight"

    manifest = runner.run_ablation(
        output_root=output,
        expected_git_commit="a" * 40,
        systems=("c60",),
        master_seeds=(42,),
        action_force_budget=30,
        total_force_budget=120,
        arms=(0.05, 0.10),
        preflight_only=True,
    )

    assert manifest["policy_name"] == "uniform"
    assert not output.exists()
    assert not output.with_name(".preflight.partial").exists()


def test_published_summary_verification_rejects_a_stale_staging_prefix(tmp_path):
    runner = _runner_module()
    output = tmp_path / "matrix"
    staging = tmp_path / ".matrix.partial"
    case_directory = output / "c60" / "seed-00000042" / "proposal-fmax-0.05"
    case_directory.mkdir(parents=True)
    stale_artifact = staging / "c60" / "seed-00000042" / "proposal-fmax-0.05"
    summary_path = case_directory / "campaign_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "action_metrics_path": str(stale_artifact / "action_metrics.jsonl"),
                "event_log_path": str(stale_artifact / "events.jsonl"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="retains staging path"):
        runner._verify_published_campaign_summary_paths(
            campaigns=[{"campaign_summary_path": str(summary_path)}],
            output_root=output,
            staging_root=staging,
        )
