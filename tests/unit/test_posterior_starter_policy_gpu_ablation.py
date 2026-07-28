"""Contract tests for the serial posterior starter-policy GPU harness."""

from __future__ import annotations

from dataclasses import fields
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.calculators import AnalyticCalculator
from pamssw.config import LSSSWConfig, SSWConfig
from pamssw.exploration import run_posterior_ssw
from pamssw.potentials import DoubleWell2D
from pamssw.state import State


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-posterior-starter-policy-gpu-ablation"
    / "run_ablation.py"
)


def _runner_module():
    module_name = "_posterior_starter_policy_gpu_ablation_test_runner"
    spec = importlib.util.spec_from_file_location(module_name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load posterior starter-policy harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _analytic_state() -> State:
    return State(
        numbers=np.ones(4, dtype=int),
        positions=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def _analytic_config() -> SSWConfig:
    return SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=2,
        proposal_pool_size=1,
        proposal_relax_steps=2,
        quench_maxiter=30,
        rng_seed=11,
    )


@pytest.mark.parametrize("system", ("c60", "pdo"))
def test_production_config_projects_to_unsoftened_serial_ssw_without_worker_outputs(tmp_path, system):
    runner = _runner_module()

    config, projection = runner.build_ssw_config(system, tmp_path / system)

    assert type(config) is SSWConfig
    assert projection["source_config_type"] == "LSSSWConfig"
    assert projection["softening_enabled"] is False
    assert config.quench_optimizer == "ase-lbfgs"
    assert config.quench_fallback_optimizer == "ase-fire"
    assert config.quench_fmax == pytest.approx(0.01)
    assert config.quench_maxiter == projection["source_config"]["quench_maxiter"] == 400
    assert config.proposal_pool_size == 1
    assert config.proposal_duplicate_rescue_optimizer is None
    assert config.accepted_structures_log is None
    assert config.accepted_structures_dir is None
    assert config.direction_diagnostics_enabled is False
    assert config.direction_diagnostics_path is None
    assert config.write_proposal_minima is False
    assert config.write_relaxation_trajectories is False
    assert projection["effective_ssw_config"] == {
        item.name: getattr(config, item.name) for item in fields(SSWConfig)
    }
    assert projection["overrides"]["quench_optimizer"] == "ase-lbfgs"
    assert projection["overrides"]["quench_fallback_optimizer"] == "ase-fire"
    assert projection["overrides"]["quench_fmax"] == pytest.approx(0.01)
    assert projection["removed_ls_fields"]


def test_exploration_config_rejects_nonserial_or_unknown_policy(tmp_path):
    runner = _runner_module()

    with pytest.raises(ValueError, match="batch_size=1"):
        runner.build_exploration_config(
            policy_name="uniform",
            run_directory=tmp_path / "parallel",
            master_seed=42,
            action_force_budget=1000,
            total_force_budget=6000,
            batch_size=2,
            max_workers=1,
        )
    with pytest.raises(ValueError, match="unsupported policy"):
        runner.build_exploration_config(
            policy_name="other",
            run_directory=tmp_path / "other",
            master_seed=42,
            action_force_budget=1000,
            total_force_budget=6000,
        )


def test_thread_owned_factory_creates_one_bootstrap_and_one_worker_calculator():
    runner = _runner_module()
    constructed: list[object] = []

    def build_calculator():
        calculator = object()
        constructed.append(calculator)
        return calculator

    factory = runner.ThreadOwnedCalculatorFactory(build_calculator)
    bootstrap_calculator = factory()
    with ThreadPoolExecutor(max_workers=1) as executor:
        action_calculator_one = executor.submit(factory).result()
        action_calculator_two = executor.submit(factory).result()

    assert bootstrap_calculator is not action_calculator_one
    assert action_calculator_one is action_calculator_two
    assert constructed == [bootstrap_calculator, action_calculator_one]
    assert factory.snapshot() == {
        "bootstrap_instances": 1,
        "action_instances": 1,
        "action_thread_count": 1,
    }


def test_parameterized_ablation_creates_all_policy_sibling_run_directories(tmp_path, monkeypatch):
    runner = _runner_module()
    input_paths = {system: tmp_path / f"{system}.xyz" for system in runner.SYSTEMS}
    model_path = tmp_path / "model.model"
    for input_path in input_paths.values():
        input_path.write_text("fixture", encoding="utf-8")
    model_path.write_text("fixture", encoding="utf-8")

    class FakeProduction:
        INPUT_PATHS = input_paths
        MODEL_PATH = model_path
        CALCULATOR_CONFIG = {"device": "cuda", "default_dtype": "float32"}

        @staticmethod
        def build_config(system, case_directory):
            del case_directory
            assert system in runner.SYSTEMS
            return LSSSWConfig(
                max_trials=1,
                max_steps_per_walk=1,
                oracle_candidates=2,
                proposal_pool_size=1,
                quench_optimizer="scipy-lbfgsb",
                quench_fallback_optimizer=None,
                quench_fmax=0.02 if system == "c60" else 0.03,
                quench_maxiter=400,
            )

        @staticmethod
        def load_state(system):
            assert system in runner.SYSTEMS
            return _analytic_state()

    called_policies: list[str] = []

    def fake_run_campaign(**kwargs):
        called_policies.append(kwargs["exploration_config"].policy_name)
        return {"schema_version": 1}

    monkeypatch.setattr(runner, "_load_production_runner", lambda: FakeProduction)
    monkeypatch.setattr(runner, "_current_commit", lambda: "expected")
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(
        runner,
        "_runtime_versions",
        lambda: {"python": "x", "numpy": "x", "scipy": "x", "ase": "x", "torch": "x", "mace": "x"},
    )
    monkeypatch.setattr(
        runner,
        "_cuda_info",
        lambda: {"requested_device": "cuda", "available": True, "runtime_version": "x", "device_name": "fake"},
    )
    monkeypatch.setattr(runner, "run_campaign", fake_run_campaign)

    index = runner.run_ablation(
        output_root=tmp_path / "output",
        expected_git_commit="expected",
        systems=runner.SYSTEMS,
        master_seeds=(42,),
        action_force_budget=30,
        total_force_budget=120,
    )

    assert called_policies == list(runner.POLICIES) * len(runner.SYSTEMS)
    assert [item["policy_name"] for item in index["campaigns"]] == called_policies
    for system in runner.SYSTEMS:
        projection = index["manifest"]["projections"][system]
        assert projection["source_config"]["quench_optimizer"] == "scipy-lbfgsb"
        assert projection["source_config"]["quench_fallback_optimizer"] is None
        assert projection["effective_ssw_config"]["quench_optimizer"] == "ase-lbfgs"
        assert projection["effective_ssw_config"]["quench_fallback_optimizer"] == "ase-fire"
        assert projection["effective_ssw_config"]["quench_fmax"] == pytest.approx(0.01)
        assert projection["effective_ssw_config"]["quench_maxiter"] == 400
        assert projection["overrides"]["quench_optimizer"] == "ase-lbfgs"
        assert projection["overrides"]["quench_fallback_optimizer"] == "ase-fire"
        assert projection["overrides"]["quench_fmax"] == pytest.approx(0.01)
        assert "quench_maxiter" not in projection["overrides"]


def test_zero_fe_attempts_are_explicitly_untimed_and_ambiguous_factory_records_fail_closed(
    tmp_path, monkeypatch
):
    runner = _runner_module()
    zero_fe_attempt = {
        "attempt": {
            "action_id": "batch-00000000-slot-0000",
            "batch_id": 0,
            "starter_id": 0,
            "selection_probability": 1.0,
            "policy_name": "uniform",
            "status": "invalid",
            "failure_reason": "invalid_starter_geometry",
            "discovered_against_snapshot": False,
            "inserted_into_archive": False,
            "landing_energy": None,
            "force_evaluations": 0,
            "evaluation_counts": {
                "bootstrap_true_quench": 0,
                "starter_true_quench": 0,
                "direction_oracle": 0,
                "escape_true_pes_check": 0,
                "biased_proposal_relax": 0,
                "landing_true_quench": 0,
                "post_relax_validation": 0,
                "unattributed": 0,
            },
            "cost_is_exact": True,
            "posterior_observed": False,
        },
        "snapshot": {
            "support_complete": True,
            "eligible_starter_ids": [0],
            "probabilities": [1.0],
        },
    }
    monkeypatch.setattr(runner, "_event_attempts", lambda _: [zero_fe_attempt])

    metrics = runner._action_metrics(
        event_path=tmp_path / "events.jsonl",
        bootstrap_energy_eV=0.0,
        action_timing=(),
    )
    assert metrics[0]["evaluator_wall_time_s"] is None
    assert metrics[0]["evaluator_calls"] is None

    with pytest.raises(RuntimeError, match="zero-FE action timing cannot be unambiguously aligned"):
        runner._action_metrics(
            event_path=tmp_path / "events.jsonl",
            bootstrap_energy_eV=0.0,
            action_timing=({"evaluator_calls": 0, "evaluator_wall_time_s": 0.0},),
        )


def test_run_campaign_emits_closed_action_metrics_from_real_posterior_runner(tmp_path):
    runner = _runner_module()
    factory = runner.InstrumentedCalculatorFactory(
        runner.ThreadOwnedCalculatorFactory(
            lambda: AnalyticCalculator(DoubleWell2D())
        )
    )
    exploration = runner.build_exploration_config(
        policy_name="posterior_proportional",
        run_directory=tmp_path / "campaign",
        master_seed=42,
        action_force_budget=30,
        total_force_budget=120,
    )

    summary = runner.run_campaign(
        initial_state=_analytic_state(),
        calculator_factory=factory,
        ssw_config=_analytic_config(),
        exploration_config=exploration,
        run_posterior=run_posterior_ssw,
    )

    assert summary["total_evaluations"] + summary["unused_force_budget"] == 120
    assert summary["purpose_counts"]["unattributed"] == 0
    assert summary["event_log_replayed"] is True
    assert summary["batch_size"] == summary["max_workers"] == 1
    assert summary["bootstrap_evaluator_wall_time_s"] >= 0.0
    assert summary["calculator_factory"] == {
        "bootstrap_instances": 1,
        "action_instances": 1,
        "action_thread_count": 1,
    }
    action_metrics_path = Path(summary["action_metrics_path"])
    assert action_metrics_path.is_file()
    metrics = runner.read_jsonl(action_metrics_path)
    assert len(metrics) == summary["completed_attempts"] + summary["failed_attempts"]
    assert all(metric["force_evaluations"] <= 30 for metric in metrics)
    assert all(metric["evaluation_counts"]["unattributed"] == 0 for metric in metrics)
    assert all(metric["evaluator_wall_time_s"] is not None for metric in metrics)


def test_run_campaign_requires_an_instrumented_factory_for_action_telemetry(tmp_path):
    runner = _runner_module()
    exploration = runner.build_exploration_config(
        policy_name="uniform",
        run_directory=tmp_path / "campaign",
        master_seed=42,
        action_force_budget=30,
        total_force_budget=120,
    )

    with pytest.raises(TypeError, match="InstrumentedCalculatorFactory"):
        runner.run_campaign(
            initial_state=_analytic_state(),
            calculator_factory=runner.ThreadOwnedCalculatorFactory(
                lambda: AnalyticCalculator(DoubleWell2D())
            ),
            ssw_config=_analytic_config(),
            exploration_config=exploration,
            run_posterior=run_posterior_ssw,
        )
