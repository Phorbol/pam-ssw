"""Contract tests for the fixed-task proposal-fmax tier ablation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.bias import GaussianBiasTerm
from pamssw.proposal_replay import CapturedProposalTask, ProposalReplayResult
from pamssw.result import (
    RelaxOutcomeClass,
    RelaxResult,
    RelaxTelemetry,
)
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-proposal-fmax-tier-ablation"
)
RUNNER_PATH = ROOT / "run_ablation.py"
ANALYZER_PATH = ROOT / "analyze_ablation.py"


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def counts(**items: int) -> EvaluationCounts:
    return EvaluationCounts.from_mapping(items)


def state(system: str = "c60", shift: float = 0.0) -> State:
    return State(
        numbers=np.array([6, 8]),
        positions=np.array([[shift, 0.0, 0.0], [1.0 + shift, 0.0, 0.0]]),
        cell=None if system == "c60" else np.diag([5.0, 5.0, 10.0]),
        pbc=(False, False, False) if system == "c60" else (True, True, False),
        fixed_mask=np.array([False, system == "pdo"]),
    )


def task(system: str = "c60") -> ProposalRelaxationTask:
    initial = state(system)
    return ProposalRelaxationTask(
        initial_state=initial,
        biases=(
            GaussianBiasTerm(
                center=initial.flatten_positions(),
                direction=np.ones(initial.n_atoms * 3),
                sigma=0.2,
                weight=0.3,
            ),
        ),
        softening=None,
        fmax=0.05,
        maxiter=80 if system == "c60" else 300,
        coordinate_trust_radius=1.5,
    )


def result(final_state: State, *, fmax: float, calls: int) -> RelaxResult:
    return RelaxResult(
        state=final_state,
        energy=-1.0 - float(final_state.positions[0, 0]),
        gradient_norm=0.5 * fmax,
        n_iter=2,
        outcome_class=RelaxOutcomeClass.CONVERGED_PRODUCTIVE,
        telemetry=RelaxTelemetry(
            backend="safe-lbfgs-total",
            evaluator_calls=calls,
            backend_evaluations=calls,
            converged=True,
            termination_reason="converged",
        ),
    )


def preflight(runner):
    return runner.Preflight(
        metadata={
            "schema_version": 1,
            "execution_commit": "a" * 40,
            "model": {"path": "model", "sha256": "b" * 64},
            "inputs": {
                system: {"path": system, "sha256": "c" * 64}
                for system in runner.SYSTEMS
            },
            "runtime_versions": {"python": "3"},
            "cuda": {"available": True},
            "calculator": {"device": "cuda"},
            "safe_lbfgs_default_history_limit": 10,
        },
        production_states={system: state(system) for system in runner.SYSTEMS},
    )


def test_protocol_changes_only_proposal_force_certificate():
    runner = module(RUNNER_PATH, "proposal_fmax_contract")
    assert runner.SYSTEMS == ("c60", "pdo")
    assert runner.CANDIDATE_SEEDS == tuple(range(42, 58))
    assert runner.TARGET_ELIGIBLE_TASKS == 8
    assert runner.DEFAULT_OUTPUT == runner.RUN_ROOT / "output-v2"
    assert [
        (arm.arm_id, arm.fmax_eV_per_A) for arm in runner.ARMS
    ] == [("fmax-0.05", 0.05), ("fmax-0.10", 0.10)]
    assert runner.PROPOSAL_OPTIMIZER == "safe-lbfgs-total"
    assert runner.SAFE_HISTORY_LIMIT == 10
    assert runner.TARGET_BIAS_COUNT == 1
    assert runner.SOFTENING_ENABLED is False
    assert runner.LANDING_OPTIMIZER == "scipy-lbfgsb"
    assert runner.LANDING_FMAX == 0.05
    assert runner.LANDING_MAXITER == 400


def test_capture_config_keeps_current_production_bias_and_maxiter(tmp_path):
    runner = module(RUNNER_PATH, "proposal_fmax_config")
    production = runner._production_module()
    for system in runner.SYSTEMS:
        baseline = asdict(production.build_config(system, tmp_path / system))
        captured = asdict(
            runner.build_capture_config(system, tmp_path / system, seed=49)
        )
        changed = {
            key: (baseline[key], captured[key])
            for key in baseline
            if baseline[key] != captured[key]
        }
        assert changed == {
            "max_steps_per_walk": (8, 1),
            "rng_seed": (42, 49),
        }
        assert captured["proposal_fmax"] == 0.05
        assert captured["proposal_relax_steps"] == baseline[
            "proposal_relax_steps"
        ]
        assert captured["proposal_optimizer"] == "safe-lbfgs-total"


def test_arm_task_has_same_biased_pes_and_maxiter_but_different_fmax():
    runner = module(RUNNER_PATH, "proposal_fmax_fixed_task")
    source = task()
    strict = runner.task_for_arm(source, runner.ARMS[0])
    loose = runner.task_for_arm(source, runner.ARMS[1])
    assert strict.fmax == 0.05
    assert loose.fmax == 0.10
    assert strict.maxiter == loose.maxiter == source.maxiter
    assert runner.fixed_biased_pes_sha256(strict) == (
        runner.fixed_biased_pes_sha256(loose)
    )
    assert strict.softening is loose.softening is None
    np.testing.assert_array_equal(
        strict.initial_state.positions, loose.initial_state.positions
    )
    np.testing.assert_array_equal(
        strict.biases[0].center, loose.biases[0].center
    )


def test_observable_capture_attempt_keeps_failed_seed_counts_and_reason():
    runner = module(RUNNER_PATH, "proposal_fmax_capture_attempt")

    class FailedWalker:
        def __init__(self, **kwargs):
            self.calculator = type(
                "Counter",
                (),
                {"snapshot": lambda self: counts(direction_oracle=2)},
            )()
            self.capture_failure_reason = "trial_state_geometry_invalid"
            self.capture_selected_direction_kind = "bond"

        def _walk_candidate_from_seed(self, seed_state):
            return seed_state

    attempt = runner.capture_proposal_attempt(
        state("pdo"),
        object(),
        type("Config", (), {"max_steps_per_walk": 1})(),
        target_bias_count=1,
        walker_factory=FailedWalker,
    )
    assert attempt.status == "ineligible"
    assert attempt.reason == "trial_state_geometry_invalid"
    assert attempt.selected_direction_kind == "bond"
    assert attempt.task is None
    assert attempt.evaluation_counts.total == 2
    assert attempt.evaluation_counts.as_dict()["direction_oracle"] == 2
    assert attempt.evaluation_counts.as_dict()["unattributed"] == 0


def test_execute_proposal_closes_biased_relax_purpose_and_records_certificate():
    runner = module(RUNNER_PATH, "proposal_fmax_execute_proposal")
    seen = []

    def replay(source, calculator, *, optimizer):
        seen.append((source.fmax, source.maxiter, optimizer, calculator))
        calls = 3
        return ProposalReplayResult(
            result=result(state(shift=source.fmax), fmax=source.fmax, calls=calls),
            evaluation_counts=counts(biased_proposal_relax=calls),
            wall_time_s=0.2,
            certificate_satisfied=True,
        )

    row = runner.execute_proposal(
        task(),
        runner.ARMS[1],
        object(),
        replay_fn=replay,
    )
    assert seen == [(0.10, 80, "safe-lbfgs-total", seen[0][3])]
    assert row["force_evaluations"] == 3
    assert row["purpose_counts"]["biased_proposal_relax"] == 3
    assert row["purpose_counts"]["unattributed"] == 0
    assert row["certificate_satisfied"] is True
    assert row["final"]["max_active_force_eV_per_A"] == pytest.approx(0.05)
    assert row["safe_history_limit"] == 10


def test_execute_landing_is_true_pes_scipy_quench_with_closed_purposes():
    runner = module(RUNNER_PATH, "proposal_fmax_execute_landing")
    seen = []

    class Walker:
        def __init__(self, *, calculator, config, softening_enabled):
            seen.append((config, softening_enabled))
            self.calculator = type(
                "Counter",
                (),
                {
                    "snapshot": lambda self: counts(
                        landing_true_quench=4,
                        post_relax_validation=1,
                    )
                },
            )()

        def relax_true_minimum(self, initial, *, quench_purpose):
            assert quench_purpose is EvaluationPurpose.LANDING_TRUE_QUENCH
            return result(state(shift=0.2), fmax=0.05, calls=4)

    production = runner._production_module()
    config = production.build_config("c60", Path("/tmp/c60"))
    row = runner.execute_landing(
        state(shift=0.1),
        config,
        object(),
        walker_factory=Walker,
    )
    landing_config, softening = seen[0]
    assert landing_config.quench_optimizer == "scipy-lbfgsb"
    assert landing_config.quench_fmax == 0.05
    assert landing_config.quench_maxiter == 400
    assert softening is False
    assert row["initial"]["positions_sha256"] == runner.position_sha256(
        state(shift=0.1).positions
    )
    assert row["force_evaluations"] == 5
    assert row["purpose_counts"]["landing_true_quench"] == 4
    assert row["purpose_counts"]["post_relax_validation"] == 1
    assert row["purpose_counts"]["unattributed"] == 0


def test_bootstrap_uses_shared_scipy_true_pes_point05_certificate():
    runner = module(RUNNER_PATH, "proposal_fmax_bootstrap")
    seen = []

    class Walker:
        def __init__(self, *, calculator, config, softening_enabled):
            seen.append((config, softening_enabled))
            self.calculator = type(
                "Counter",
                (),
                {
                    "snapshot": lambda self: counts(
                        starter_true_quench=3,
                        post_relax_validation=1,
                    )
                },
            )()

        def relax_true_minimum(self, initial, *, quench_purpose):
            assert quench_purpose is EvaluationPurpose.STARTER_TRUE_QUENCH
            return result(state(), fmax=0.05, calls=3)

    production = runner._production_module()
    strict_config = production.build_config("c60", Path("/tmp/c60"))
    assert strict_config.quench_fmax == 0.01
    bootstrap = runner.bootstrap_minimum(
        "c60",
        state(),
        strict_config,
        object(),
        walker_factory=Walker,
    )
    effective, softening = seen[0]
    assert effective.quench_optimizer == "scipy-lbfgsb"
    assert effective.quench_fmax == 0.05
    assert effective.quench_maxiter == 400
    assert softening is False
    assert bootstrap.certificate_satisfied is True


def test_preflight_only_precedes_calculator_and_writes_nothing(tmp_path):
    runner = module(RUNNER_PATH, "proposal_fmax_preflight")
    events = []
    output = tmp_path / "output"

    def checked(**kwargs):
        events.append(("preflight", kwargs))
        return preflight(runner)

    def calculator():
        events.append(("calculator", None))
        raise AssertionError("preflight-only must not construct calculator")

    returned = runner.run(
        output_dir=output,
        expected_git_commit="a" * 40,
        preflight_only=True,
        preflight_fn=checked,
        calculator_factory=calculator,
    )
    assert returned["execution_commit"] == "a" * 40
    assert events == [
        ("preflight", {"expected_git_commit": "a" * 40})
    ]
    assert not output.exists()
    assert not output.with_name("output.partial").exists()


def test_run_uses_one_bootstrap_and_one_fixed_task_per_seed_for_both_arms(
    tmp_path, monkeypatch
):
    runner = module(RUNNER_PATH, "proposal_fmax_run")
    events = []

    def bootstrap(system, raw_state, config, calculator):
        events.append(("bootstrap", system))
        return runner.BootstrapResult(
            state=state(system, shift=0.01),
            energy_eV=-0.5,
            gradient_norm=0.01,
            n_iter=1,
            wall_time_s=0.1,
            evaluation_counts=counts(
                starter_true_quench=2, post_relax_validation=1
            ),
            termination_reason="converged",
            certificate_satisfied=True,
        )

    def capture(seed_state, calculator, config, *, target_bias_count):
        events.append(("capture", config.rng_seed, target_bias_count))
        failed = any(seed_state.pbc) and config.rng_seed == 46
        captured = task("pdo" if any(seed_state.pbc) else "c60")
        return runner.CaptureAttempt(
            status="ineligible" if failed else "eligible",
            reason="trial_state_geometry_invalid" if failed else None,
            selected_direction_kind="bond" if failed else "random",
            task=None if failed else captured,
            evaluation_counts=counts(direction_oracle=2),
        )

    def replay(source, calculator, *, optimizer):
        events.append(("proposal", source.fmax))
        arm_shift = source.fmax
        calls = 2 if source.fmax == 0.10 else 4
        return ProposalReplayResult(
            result=result(
                state(
                    "pdo" if any(source.initial_state.pbc) else "c60",
                    shift=arm_shift,
                ),
                fmax=source.fmax,
                calls=calls,
            ),
            evaluation_counts=counts(biased_proposal_relax=calls),
            wall_time_s=0.1,
            certificate_satisfied=True,
        )

    def landing(initial, config, calculator, **kwargs):
        events.append(("landing", float(initial.positions[0, 0])))
        final = state("pdo" if any(initial.pbc) else "c60", shift=0.3)
        return {
            "optimizer": "scipy-lbfgsb",
            "fmax_eV_per_A": 0.05,
            "maxiter": 400,
            "objective": "true_mace_pes_no_bias_no_softening",
            "initial": {
                "positions_sha256": runner.position_sha256(initial.positions),
                "state": runner.state_payload(initial),
            },
            "final": {
                "energy_eV": -2.0,
                "max_active_force_eV_per_A": 0.01,
                "positions_sha256": runner.position_sha256(final.positions),
                "state": runner.state_payload(final),
            },
            "certificate_satisfied": True,
            "n_iter": 2,
            "termination_reason": "converged",
            "outcome_class": "converged_productive",
            "telemetry": {"evaluator_calls": 3},
            "force_evaluations": 4,
            "purpose_counts": {
                purpose.value: (
                    3
                    if purpose is EvaluationPurpose.LANDING_TRUE_QUENCH
                    else 1
                    if purpose is EvaluationPurpose.POST_RELAX_VALIDATION
                    else 0
                )
                for purpose in EvaluationPurpose
            },
            "wall_time_s": 0.1,
        }

    monkeypatch.setattr(runner, "bootstrap_minimum", bootstrap)
    monkeypatch.setattr(runner, "capture_proposal_attempt", capture)
    monkeypatch.setattr(runner, "replay_proposal_task", replay)
    monkeypatch.setattr(runner, "execute_landing", landing)

    output = tmp_path / "output"
    calculator_calls = 0

    def calculator():
        nonlocal calculator_calls
        calculator_calls += 1
        return object()

    summary = runner.run(
        output_dir=output,
        expected_git_commit="a" * 40,
        preflight_fn=lambda **_: preflight(runner),
        calculator_factory=calculator,
    )
    assert calculator_calls == 12
    assert sum(event[0] == "bootstrap" for event in events) == 2
    assert sum(event[0] == "capture" for event in events) == 17
    assert sum(event[0] == "landing" for event in events) == 32
    capture_indices = [
        index for index, event in enumerate(events) if event[0] == "capture"
    ]
    proposal_indices = [
        index for index, event in enumerate(events) if event[0] == "proposal"
    ]
    assert max(capture_indices) < min(proposal_indices)
    assert summary["task_count"] == 16
    assert summary["row_count"] == 32
    assert not output.with_name("output.partial").exists()
    persisted = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    for system in runner.SYSTEMS:
        system_data = persisted["systems"][system]
        assert len(system_data["tasks"]) == 8
        assert len(system_data["rows"]) == 16
        expected_seeds = (
            tuple(range(42, 50))
            if system == "c60"
            else (42, 43, 44, 45, 47, 48, 49, 50)
        )
        assert tuple(
            attempt["seed"] for attempt in system_data["capture_attempts"]
        ) == (
            tuple(range(42, 50))
            if system == "c60"
            else tuple(range(42, 51))
        )
        failures = [
            attempt
            for attempt in system_data["capture_attempts"]
            if attempt["status"] == "ineligible"
        ]
        assert failures == (
            []
            if system == "c60"
            else [
                {
                    "direction_trace_path": (
                        "pdo/seed-46/direction_trace.jsonl"
                    ),
                    "force_evaluations": 2,
                    "purpose_counts": {
                        purpose.value: (
                            2
                            if purpose is EvaluationPurpose.DIRECTION_ORACLE
                            else 0
                        )
                        for purpose in EvaluationPurpose
                    },
                    "reason": "trial_state_geometry_invalid",
                    "seed": 46,
                    "selected_direction_kind": "bond",
                    "selected_for_arms": False,
                    "status": "ineligible",
                }
            ]
        )
        for seed in expected_seeds:
            paired = [
                row for row in system_data["rows"] if row["seed"] == seed
            ]
            assert len(paired) == 2
            assert len({row["fixed_biased_pes_sha256"] for row in paired}) == 1
            assert len({row["source_task_sha256"] for row in paired}) == 1
            for row in paired:
                assert row["landing"]["initial"]["positions_sha256"] == (
                    row["proposal"]["final"]["positions_sha256"]
                )


def test_partial_output_is_left_unpublished_on_failure(tmp_path, monkeypatch):
    runner = module(RUNNER_PATH, "proposal_fmax_atomic_failure")

    def fail(*args, **kwargs):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(runner, "bootstrap_minimum", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        runner.run(
            output_dir=output,
            expected_git_commit="a" * 40,
            preflight_fn=lambda **_: preflight(runner),
            calculator_factory=lambda: object(),
        )
    assert not output.exists()
    assert output.with_name("output.partial").is_dir()


def test_analyzer_reports_paired_cost_and_current_archive_landing_equivalence(
    tmp_path, monkeypatch
):
    runner = module(RUNNER_PATH, "proposal_fmax_analysis_fixture")
    analyzer = module(ANALYZER_PATH, "proposal_fmax_analyzer")

    def bootstrap(system, raw_state, config, calculator):
        return runner.BootstrapResult(
            state=state(system),
            energy_eV=-0.5,
            gradient_norm=0.01,
            n_iter=1,
            wall_time_s=0.1,
            evaluation_counts=counts(
                starter_true_quench=2, post_relax_validation=1
            ),
            termination_reason="converged",
            certificate_satisfied=True,
        )

    def capture(seed_state, calculator, config, *, target_bias_count):
        failed = any(seed_state.pbc) and config.rng_seed == 46
        return runner.CaptureAttempt(
            status="ineligible" if failed else "eligible",
            reason="trial_state_geometry_invalid" if failed else None,
            selected_direction_kind="bond" if failed else "random",
            task=(
                None
                if failed
                else task("pdo" if any(seed_state.pbc) else "c60")
            ),
            evaluation_counts=counts(direction_oracle=2),
        )

    def replay(source, calculator, *, optimizer):
        calls = 2 if source.fmax == 0.10 else 4
        return ProposalReplayResult(
            result=result(
                state(
                    "pdo" if any(source.initial_state.pbc) else "c60",
                    shift=source.fmax,
                ),
                fmax=source.fmax,
                calls=calls,
            ),
            evaluation_counts=counts(biased_proposal_relax=calls),
            wall_time_s=0.1,
            certificate_satisfied=True,
        )

    def landing(initial, config, calculator, **kwargs):
        system = "pdo" if any(initial.pbc) else "c60"
        final = state(system, shift=0.3)
        return {
            "optimizer": "scipy-lbfgsb",
            "fmax_eV_per_A": 0.05,
            "maxiter": 400,
            "objective": "true_mace_pes_no_bias_no_softening",
            "initial": {
                "positions_sha256": runner.position_sha256(initial.positions),
                "state": runner.state_payload(initial),
            },
            "final": {
                "energy_eV": -2.0,
                "max_active_force_eV_per_A": 0.01,
                "positions_sha256": runner.position_sha256(final.positions),
                "state": runner.state_payload(final),
            },
            "certificate_satisfied": True,
            "n_iter": 2,
            "termination_reason": "converged",
            "outcome_class": "converged_productive",
            "telemetry": {"evaluator_calls": 3},
            "force_evaluations": 4,
            "purpose_counts": {
                purpose.value: (
                    3
                    if purpose is EvaluationPurpose.LANDING_TRUE_QUENCH
                    else 1
                    if purpose is EvaluationPurpose.POST_RELAX_VALIDATION
                    else 0
                )
                for purpose in EvaluationPurpose
            },
            "wall_time_s": 0.1,
        }

    monkeypatch.setattr(runner, "bootstrap_minimum", bootstrap)
    monkeypatch.setattr(runner, "capture_proposal_attempt", capture)
    monkeypatch.setattr(runner, "replay_proposal_task", replay)
    monkeypatch.setattr(runner, "execute_landing", landing)
    output = tmp_path / "output"
    runner.run(
        output_dir=output,
        expected_git_commit="a" * 40,
        preflight_fn=lambda **_: preflight(runner),
        calculator_factory=lambda: object(),
    )
    evidence = analyzer.analyze(output)
    assert evidence["claim_boundary"] == {
        "fixed_one_bias_local_ablation": True,
        "softening_disabled": True,
        "shared_bootstrap_true_pes_fmax_eV_per_A": 0.05,
        "capture_eligibility_conditioned": True,
        "capture_eligibility_may_filter_direction_kinds": True,
        "full_ssw_superiority_supported": False,
    }
    for system in runner.SYSTEMS:
        paired = evidence["systems"][system]["paired"]
        assert paired["task_count"] == 8
        assert paired["loose_lower_proposal_calls_count"] == 8
        assert paired["proposal_call_savings_total"] == 16
        assert paired["landing_same_basin_count"] == 8
        assert paired["landing_equivalence_rate"] == pytest.approx(1.0)
        assert paired["both_landing_certified_count"] == 8
        assert paired["arm_costs"] == {
            "fmax-0.05": {
                "proposal_force_evaluations": 32,
                "proposal_wall_time_s": pytest.approx(0.8),
                "landing_force_evaluations": 32,
                "landing_wall_time_s": pytest.approx(0.8),
                "combined_force_evaluations": 64,
                "combined_wall_time_s": pytest.approx(1.6),
            },
            "fmax-0.10": {
                "proposal_force_evaluations": 16,
                "proposal_wall_time_s": pytest.approx(0.8),
                "landing_force_evaluations": 32,
                "landing_wall_time_s": pytest.approx(0.8),
                "combined_force_evaluations": 48,
                "combined_wall_time_s": pytest.approx(1.6),
            },
        }
        assert paired["combined_force_evaluation_savings"] == 16
        assert paired["combined_force_evaluation_savings_fraction"] == pytest.approx(
            0.25
        )
    assert evidence["systems"]["pdo"]["capture_selection"][
        "ineligible_direction_kind_counts"
    ] == {"bond": 1}
    assert evidence["systems"]["c60"]["shared_pre_arm_costs"] == {
        "bootstrap_force_evaluations": 3,
        "bootstrap_wall_time_s": pytest.approx(0.1),
        "capture_force_evaluations": 16,
        "combined_force_evaluations": 19,
    }
    assert evidence["systems"]["pdo"]["shared_pre_arm_costs"] == {
        "bootstrap_force_evaluations": 3,
        "bootstrap_wall_time_s": pytest.approx(0.1),
        "capture_force_evaluations": 18,
        "combined_force_evaluations": 21,
    }

    raw = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    invalid_prefix = deepcopy(raw)
    invalid_prefix["systems"]["pdo"]["capture_attempts"][0]["seed"] = 43
    (output / "summary.json").write_text(
        json.dumps(invalid_prefix), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="capture attempt prefix"):
        analyzer.analyze(output)

    invalid_bootstrap = deepcopy(raw)
    invalid_bootstrap["systems"]["c60"]["bootstrap"]["purpose_counts"][
        "unattributed"
    ] = 1
    (output / "summary.json").write_text(
        json.dumps(invalid_bootstrap), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bootstrap purpose accounting"):
        analyzer.analyze(output)

    raw["systems"]["c60"]["rows"][0]["proposal"]["purpose_counts"][
        "unattributed"
    ] = 1
    (output / "summary.json").write_text(
        json.dumps(raw), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="proposal purpose accounting"):
        analyzer.analyze(output)


def test_nonfinite_archive_rmsd_is_explicit_null_not_invalid_json(monkeypatch):
    runner = module(RUNNER_PATH, "proposal_fmax_nonfinite_runner")
    analyzer = module(ANALYZER_PATH, "proposal_fmax_nonfinite_analyzer")
    final = {
        "energy_eV": -1.0,
        "state": runner.state_payload(state()),
    }
    row = {"landing": {"final": final}}
    monkeypatch.setattr(
        analyzer.MinimaArchive,
        "_rmsd",
        staticmethod(lambda first, second: float("inf")),
    )
    same, energy_delta, rmsd, status = analyzer._same_basin(
        row,
        row,
        energy_tol=1e-3,
        rmsd_tol=0.1,
    )
    assert same is False
    assert energy_delta == 0.0
    assert rmsd is None
    assert status == "nonfinite_rejected"
    json.dumps(
        {"landing_rmsd_A": rmsd, "landing_rmsd_status": status},
        allow_nan=False,
    )
