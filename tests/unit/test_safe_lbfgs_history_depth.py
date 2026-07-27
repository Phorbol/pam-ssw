"""Fail-closed contract tests for the safe L-BFGS history-depth runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import errno
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.bias import GaussianBiasTerm
from pamssw.result import RelaxResult, RelaxTelemetry
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-safe-lbfgs-history-depth-ablation"
    / "run_gpu_ablation.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("safe_lbfgs_history_depth_runner", _RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state() -> State:
    return State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.2, 0.0, 0.0], [1.0, 0.2, 0.0]]),
        fixed_mask=np.array([True, False]),
    )


def _task() -> ProposalRelaxationTask:
    return ProposalRelaxationTask(
        initial_state=_state(),
        biases=(
            GaussianBiasTerm(
                center=np.zeros(6),
                direction=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                sigma=1.0,
                weight=0.0,
            ),
        ),
        softening=None,
        fmax=3.0e-3,
        maxiter=80,
        coordinate_trust_radius=None,
    )


class _QuadraticCalculator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self.calls += 1
        target = np.zeros_like(flat_positions, dtype=float)
        target[3] = 0.35
        displacement = np.asarray(flat_positions, dtype=float) - target
        del template
        return float(0.5 * np.dot(displacement, displacement)), displacement


def _trusted_source(tmp_path: Path, calculator_factory):
    model = tmp_path / "model.pt"
    input_path = tmp_path / "input.xyz"
    model.write_bytes(b"trusted model")
    input_path.write_bytes(b"trusted input")
    return {
        "MODEL": model,
        "MODEL_SHA256": _sha256(model),
        "SYSTEMS": {
            "c60": {"input": input_path, "sha256": _sha256(input_path)},
            "pdo": {"input": input_path, "sha256": _sha256(input_path)},
        },
        "_calculator": calculator_factory,
    }


def _runtime_versions():
    return {
        "python": "3.test",
        "python_implementation": "CPython",
        "numpy": "test",
        "scipy": "test",
        "ase": "test",
        "torch": "test",
        "mace": "test",
    }


def _platform_provenance():
    return {
        "sys_platform": "linux",
        "system": "Linux",
        "release": "test",
        "machine": "x86_64",
    }


def _cuda_provenance(_: object):
    return {
        "requested_device": "cuda",
        "cuda_device_name": "test GPU",
        "cuda_runtime_version": "12.test",
    }


def test_runner_module_is_created_for_the_ablation_contract():
    assert _RUNNER_PATH.is_file()


def test_run_local_output_is_ignored_exactly_and_is_not_tracked():
    runner = _runner_module()
    assert runner.OUTPUT_DIR == runner.RUN_ROOT / "output"
    assert (runner.RUN_ROOT / ".gitignore").read_text(encoding="utf-8") == "output/\n"
    relative = runner.OUTPUT_DIR.relative_to(runner.REPO_ROOT)
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(relative)],
        cwd=runner.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--no-index", str(relative / "future.json")],
        cwd=runner.REPO_ROOT,
    )
    assert ignored.returncode == 0


def test_exact_history_depth_arms_and_32_row_frozen_matrix():
    runner = _runner_module()
    assert runner.SYSTEMS == ("c60", "pdo")
    assert runner.SEEDS == tuple(range(42, 50))
    assert runner.MAXITER == 400
    assert tuple(
        (arm.arm_id, arm.kernel, arm.history_limit, arm.scale_policy, arm.secant_policy,
         arm.adaptive_scale_without_history)
        for arm in runner.ARMS
    ) == (
        (
            "adaptive-scale-history1",
            "safe-lbfgs-total",
            1,
            "latest-history-pair-gamma-plus-one-two-loop-correction",
            "total-biased-gradient",
            False,
        ),
        (
            "adaptive-scale-history10",
            "safe-lbfgs-total",
            10,
            "latest-history-pair-gamma-plus-up-to-ten-two-loop-corrections",
            "total-biased-gradient",
            False,
        ),
    )
    rows = runner._expected_row_keys()
    assert len(rows) == len({(row["system"], row["task_id"], row["arm_id"]) for row in rows}) == 32
    assert {row["arm_id"] for row in rows} == {
        "adaptive-scale-history1",
        "adaptive-scale-history10",
    }


def test_source_summary_and_each_canonical_task_hash_are_pinned():
    runner = _runner_module()
    assert runner.EXPECTED_SOURCE_SUMMARY_SHA256 == (
        "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
    )
    assert _sha256(runner.SOURCE_SUMMARY_PATH) == runner.EXPECTED_SOURCE_SUMMARY_SHA256
    tasks_by_system, measured = runner.load_fixed_tasks(runner.SOURCE_SUMMARY_PATH)
    assert measured == runner.EXPECTED_SOURCE_SUMMARY_SHA256
    assert tuple(tasks_by_system) == runner.SYSTEMS
    assert sum(len(tasks) for tasks in tasks_by_system.values()) == 16
    for system, tasks in tasks_by_system.items():
        assert [task.task_id for task in tasks] == [
            f"{system}-seed-{seed}-bias-1" for seed in runner.SEEDS
        ]
        assert [task.task_sha256 for task in tasks] == [
            runner.EXPECTED_TASK_SHA256[f"{system}-seed-{seed}-bias-1"]
            for seed in runner.SEEDS
        ]
        assert all(task.relax_task.maxiter == runner.MAXITER for task in tasks)

    summary = json.loads(runner.SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["systems"][0]["tasks"][0]["task"]["fmax"] *= 2.0
    with pytest.raises(ValueError, match="canonical task SHA256 mismatch"):
        runner._tasks_from_summary(summary)


def test_pamssw_bundle_helpers_and_kernel_descriptor_are_pinned():
    runner = _runner_module()
    assert runner.EXPECTED_PAMSSW_BUNDLE_SHA256 == (
        "83a2b845c4bbe0235e8c584a579e1b9f1e89690084c8abdd9ef3aa20c2a09b50"
    )
    assert runner._pamssw_bundle_sha256(runner.PAMSSW_SOURCE_ROOT) == runner.EXPECTED_PAMSSW_BUNDLE_SHA256
    assert _sha256(runner.FIXED_REPLAY_DRIVER) == runner.EXPECTED_FIXED_REPLAY_DRIVER_SHA256
    assert _sha256(runner.G1_DRIVER_PATH) == runner.EXPECTED_G1_DRIVER_SHA256
    assert _sha256(runner.TRACE_RECORDER_PATH) == runner.EXPECTED_TRACE_RECORDER_SHA256
    descriptor, digest = runner._verified_safe_kernel_descriptor()
    assert descriptor == runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR
    assert digest == runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256


@pytest.mark.parametrize("failure", ("helper", "model", "input", "runtime", "cuda"))
def test_preflight_rejects_tampering_before_any_calculator_construction(
    tmp_path, monkeypatch, failure
):
    runner = _runner_module()
    calculator_calls = 0

    def forbidden_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        raise AssertionError("calculator must not be constructed during preflight")

    source = _trusted_source(tmp_path, forbidden_factory)
    monkeypatch.setattr(runner, "_worktree_is_clean", lambda: True)
    if failure == "helper":
        tampered = tmp_path / "helper.py"
        tampered.write_text("# tampered\n", encoding="utf-8")
        monkeypatch.setattr(runner, "TRACE_RECORDER_PATH", tampered)
    elif failure == "model":
        Path(source["MODEL"]).write_bytes(b"tampered")
    elif failure == "input":
        Path(source["SYSTEMS"]["c60"]["input"]).write_bytes(b"tampered")

    runtime_probe = _runtime_versions
    cuda_probe = _cuda_provenance
    if failure == "runtime":
        runtime_probe = lambda: {"python": "only"}
    elif failure == "cuda":
        cuda_probe = lambda _: {"requested_device": "cpu"}

    with pytest.raises((RuntimeError, ValueError), match="(SHA256|runtime|CUDA|cuda)"):
        runner.preflight(
            source_summary_path=runner.SOURCE_SUMMARY_PATH,
            output_dir=tmp_path / "output",
            expected_git_commit=runner._current_commit(),
            source_loader=lambda: source,
            runtime_probe=runtime_probe,
            platform_probe=_platform_provenance,
            cuda_probe=cuda_probe,
        )
    assert calculator_calls == 0


def test_preflight_only_runs_all_gates_and_constructs_no_calculator(tmp_path, monkeypatch):
    runner = _runner_module()
    calculator_calls = 0

    def forbidden_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        raise AssertionError("preflight-only must not construct calculators")

    monkeypatch.setattr(runner, "_worktree_is_clean", lambda: True)
    checked = runner.preflight(
        source_summary_path=runner.SOURCE_SUMMARY_PATH,
        output_dir=tmp_path / "output",
        expected_git_commit=runner._current_commit(),
        source_loader=lambda: _trusted_source(tmp_path, forbidden_factory),
        runtime_probe=_runtime_versions,
        platform_probe=_platform_provenance,
        cuda_probe=_cuda_provenance,
    )
    assert checked.git_provenance["worktree_clean"] is True
    assert set(checked.runtime_versions) == set(runner.RUNTIME_VERSION_KEYS)
    assert set(checked.platform_provenance) == set(runner.PLATFORM_PROVENANCE_KEYS)
    assert calculator_calls == 0
    assert not (tmp_path / "output").exists()
    assert runner._parse_args(["--expected-git-commit", runner._current_commit(), "--preflight-only"]).preflight_only
    with pytest.raises(SystemExit):
        runner._parse_args([])


def test_replay_records_zero_extra_calls_and_resolves_only_contract_parameters():
    runner = _runner_module()
    calculator = _QuadraticCalculator()
    replay = runner.replay_task_with_trace(
        _task(),
        calculator,
        arm=runner.ARMS[0],
    )
    purposes = replay.evaluation_counts.as_dict()
    assert len(replay.trace_records) == replay.evaluation_counts.total
    assert replay.result.telemetry.backend_evaluations == replay.evaluation_counts.total
    assert purposes["biased_proposal_relax"] == replay.evaluation_counts.total
    assert purposes["unattributed"] == 0
    assert calculator.calls == replay.evaluation_counts.total
    assert replay.resolved_arm == {
        "history_limit": 1,
        "scale_policy": "latest-history-pair-gamma-plus-one-two-loop-correction",
        "secant_policy": "total-biased-gradient",
        "adaptive_scale_without_history": False,
    }


def _fake_preflight(runner, calculator_factory):
    tasks_by_system, source_sha256 = runner.load_fixed_tasks(runner.SOURCE_SUMMARY_PATH)
    return runner.Preflight(
        tasks_by_system=tasks_by_system,
        source_summary_sha256=source_sha256,
        source={"_calculator": calculator_factory},
        safe_kernel_descriptor=runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR,
        safe_kernel_descriptor_sha256=runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256,
        helper_provenance={
            "fixed_replay_driver": {"path": "/tmp/a", "sha256": "a" * 64},
            "g1_driver": {"path": "/tmp/b", "sha256": "b" * 64},
            "trace_recorder": {"path": "/tmp/c", "sha256": "c" * 64},
        },
        pamssw_source_provenance={
            "source_root": "/tmp/pamssw",
            "bundle_sha256": "d" * 64,
            "imported_module_paths": {},
            "imported_symbol_paths": {},
        },
        git_provenance={
            "expected_git_commit": "e" * 40,
            "actual_git_commit": "e" * 40,
            "repo_root": "/tmp/repo",
            "worktree_clean": True,
        },
        runtime_versions=_runtime_versions(),
        platform_provenance=_platform_provenance(),
        cuda_model_input_provenance={
            "requested_device": "cuda",
            "cuda_device_name": "GPU",
            "cuda_runtime_version": "12.test",
            "model": {"path": "/tmp/model", "declared_sha256": "f" * 64, "measured_sha256": "f" * 64},
            "inputs": {
                system: {"path": f"/tmp/{system}", "declared_sha256": "f" * 64, "measured_sha256": "f" * 64}
                for system in runner.SYSTEMS
            },
        },
    )


def _finite_replay(runner, task, arm):
    calls = 2
    counts = EvaluationCounts.from_mapping({EvaluationPurpose.BIASED_PROPOSAL_RELAX: calls})
    result = RelaxResult(
        state=task.initial_state,
        energy=-1.0,
        gradient_norm=0.1,
        n_iter=400,
        telemetry=RelaxTelemetry(
            backend="safe-lbfgs-total",
            evaluator_calls=calls,
            backend_evaluations=calls,
            converged=False,
            termination_reason="maxiter",
        ),
    )
    trace = [
        {
            "evaluation_index": index,
            "positions_sha256": "0" * 64,
            "true_energy_eV": -1.0,
            "bias_energy_eV": 0.0,
            "softening_energy_eV": 0.0,
            "total_energy_eV": -1.0,
            "active_max_total_force_eV_per_A": 0.1,
            "accepted_state": True,
        }
        for index in range(1, calls + 1)
    ]
    return runner.ReplayResult(
        result,
        counts,
        0.0,
        trace,
        (),
        False,
        runner.resolve_arm(arm),
    )


def test_run_uses_four_distinct_serial_calculators_and_publishes_closed_rows(tmp_path, monkeypatch):
    runner = _runner_module()
    created: list[object] = []
    replay_events: list[tuple[object, str, int]] = []

    def factory():
        calculator = object()
        created.append(calculator)
        return calculator

    monkeypatch.setattr(runner, "preflight", lambda **_: _fake_preflight(runner, factory))

    def replay(task, calculator, *, arm):
        replay_events.append((calculator, arm.arm_id, task.initial_state.n_atoms))
        return _finite_replay(runner, task, arm)

    monkeypatch.setattr(runner, "replay_task_with_trace", replay)
    output_dir = tmp_path / "output"
    summary = runner.run(output_dir=output_dir, expected_git_commit="e" * 40)

    assert len(created) == len(runner.SYSTEMS) * len(runner.ARMS) == 4
    assert [event[0] for event in replay_events[:8]] == [created[0]] * 8
    assert [event[0] for event in replay_events[8:16]] == [created[1]] * 8
    assert [event[0] for event in replay_events[16:24]] == [created[2]] * 8
    assert [event[0] for event in replay_events[24:]] == [created[3]] * 8
    assert summary["row_count"] == 32
    assert summary["certificate_all_satisfied"] is False
    assert summary["termination_reason_counts"] == {"maxiter": 32}
    saved = json.loads((output_dir / "c60.json").read_text(encoding="utf-8"))["rows"]
    row = saved[0]
    assert set(row) == set(runner.ROW_KEYS)
    assert row["arm"]["requested_history_limit"] in {1, 10}
    assert row["force_evaluations"] == row["telemetry"]["backend_evaluations"]
    assert row["purpose_counts"]["biased_proposal_relax"] == row["force_evaluations"]
    assert len(row["zero_extra_call_trace"]) == row["force_evaluations"]
    assert row["purpose_counts"]["unattributed"] == 0


@pytest.mark.parametrize("failure", ("nonfinite", "open_ledger", "schema"))
def test_nonfinite_open_or_schema_mismatch_is_fatal_without_partial_output(
    tmp_path, monkeypatch, failure
):
    runner = _runner_module()
    monkeypatch.setattr(runner, "preflight", lambda **_: _fake_preflight(runner, lambda: object()))

    def invalid_replay(task, calculator, *, arm):
        del calculator
        replay = _finite_replay(runner, task, arm)
        if failure == "nonfinite":
            replay = replay._replace(result=RelaxResult(
                state=replay.result.state,
                energy=float("nan"),
                gradient_norm=0.1,
                n_iter=0,
                telemetry=replay.result.telemetry,
            ))
        elif failure == "open_ledger":
            replay = replay._replace(trace_records=replay.trace_records[:1])
        return replay

    monkeypatch.setattr(runner, "replay_task_with_trace", invalid_replay)
    output_dir = tmp_path / "output"
    if failure == "schema":
        with pytest.raises(ValueError, match="schema"):
            runner.publish_ledger_atomically(output_dir, runner._expected_row_keys(), {"schema_version": 1})
    else:
        with pytest.raises(RuntimeError, match="(non-finite|ledger mismatch)"):
            runner.run(output_dir=output_dir, expected_git_commit="e" * 40)
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_atomic_publish_refuses_overwrite_and_cleans_same_parent_staging(tmp_path):
    runner = _runner_module()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.publish_ledger_atomically(output_dir, [], {})
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_summary_rejects_descriptor_or_arm_schema_drift():
    runner = _runner_module()
    checked = _fake_preflight(runner, lambda: object())
    summary = {
        "schema_version": 1,
        "claim_ceiling": "fixed claim ceiling",
        "source_summary_sha256": checked.source_summary_sha256,
        "systems": list(runner.SYSTEMS),
        "task_count": 16,
        "row_count": 32,
        "certificate_all_satisfied": False,
        "certificate_unsatisfied_count": 32,
        "termination_reason_counts": {"maxiter": 32},
        "arms": [runner.asdict(arm) for arm in runner.ARMS],
        "safe_kernel_descriptor": dict(checked.safe_kernel_descriptor),
        "safe_kernel_descriptor_sha256": checked.safe_kernel_descriptor_sha256,
        "runner_helper_provenance": dict(checked.helper_provenance),
        "pamssw_source_provenance": dict(checked.pamssw_source_provenance),
        "git_provenance": dict(checked.git_provenance),
        "runtime_versions": dict(checked.runtime_versions),
        "platform_provenance": dict(checked.platform_provenance),
        "cuda_model_input_provenance": dict(checked.cuda_model_input_provenance),
        "wall_time_total_s": 0.0,
    }
    runner._validate_summary(summary)

    drifted_descriptor = dict(summary)
    drifted_descriptor["safe_kernel_descriptor"] = {"optimizer": "forged"}
    with pytest.raises(ValueError, match="safe kernel descriptor"):
        runner._validate_summary(drifted_descriptor)

    drifted_arms = dict(summary)
    drifted_arms["arms"] = []
    with pytest.raises(ValueError, match="arms"):
        runner._validate_summary(drifted_arms)


def test_trace_schema_rejects_bool_or_integer_substitutes_for_typed_fields():
    runner = _runner_module()
    valid = {
        "evaluation_index": 1,
        "positions_sha256": "0" * 64,
        "true_energy_eV": -1.0,
        "bias_energy_eV": 0.0,
        "softening_energy_eV": 0.0,
        "total_energy_eV": -1.0,
        "active_max_total_force_eV_per_A": 0.1,
        "accepted_state": True,
    }
    assert runner._trace_values_are_finite([valid])
    bool_index = dict(valid, evaluation_index=True)
    integer_energy = dict(valid, total_energy_eV=-1)
    assert not runner._trace_values_are_finite([bool_index])
    assert not runner._trace_values_are_finite([integer_energy])


def test_preflight_orders_trust_gates_before_source_loader_or_calculator(tmp_path, monkeypatch):
    runner = _runner_module()
    events: list[str] = []
    source = _trusted_source(tmp_path, lambda: (_ for _ in ()).throw(AssertionError("no calculator")))

    monkeypatch.setattr(runner, "load_fixed_tasks", lambda _: (events.append("tasks") or ({}, "a" * 64)))
    monkeypatch.setattr(runner, "_verified_safe_kernel_descriptor", lambda: (events.append("descriptor") or ({}, "b" * 64)))
    monkeypatch.setattr(runner, "_verified_helper_provenance", lambda: (events.append("helpers") or {}))
    monkeypatch.setattr(runner, "_verified_pamssw_source", lambda: (events.append("pamssw") or {}))
    monkeypatch.setattr(runner, "_verified_git_provenance", lambda _: (events.append("git") or {}))
    monkeypatch.setattr(runner, "_verified_model_input_provenance", lambda _: (events.append("model_inputs") or {"model": {}, "inputs": {}}))

    def source_loader():
        events.append("source")
        return source

    def runtime_probe():
        events.append("runtime")
        return _runtime_versions()

    def platform_probe():
        events.append("platform")
        return _platform_provenance()

    def cuda_probe(_: object):
        events.append("cuda")
        return _cuda_provenance(source)

    monkeypatch.setattr(runner, "_validated_cuda_model_input_provenance", lambda _: {})
    runner.preflight(
        source_summary_path=tmp_path / "summary.json",
        output_dir=tmp_path / "output",
        expected_git_commit="a" * 40,
        source_loader=source_loader,
        runtime_probe=runtime_probe,
        platform_probe=platform_probe,
        cuda_probe=cuda_probe,
    )
    assert events == [
        "tasks",
        "helpers",
        "pamssw",
        "descriptor",
        "git",
        "runtime",
        "platform",
        "source",
        "model_inputs",
        "cuda",
    ]


def test_publish_race_keeps_concurrently_created_target_and_cleans_staging(
    tmp_path, monkeypatch
):
    runner = _runner_module()
    output_dir = tmp_path / "output"
    sentinel = output_dir / "sentinel"
    helper_calls = 0

    def race_at_noreplace_boundary(source, target):
        nonlocal helper_calls
        helper_calls += 1
        assert source.parent == target.parent
        target.mkdir()
        (target / "sentinel").write_text("preserve", encoding="utf-8")
        raise FileExistsError(target)

    monkeypatch.setattr(
        runner,
        "_rename_directory_noreplace",
        race_at_noreplace_boundary,
        raising=False,
    )
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda **_: _fake_preflight(runner, lambda: object()),
    )
    monkeypatch.setattr(
        runner,
        "replay_task_with_trace",
        lambda task, calculator, *, arm: _finite_replay(runner, task, arm),
    )

    with pytest.raises(FileExistsError):
        runner.run(output_dir=output_dir, expected_git_commit="e" * 40)

    assert helper_calls == 1
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not list(tmp_path.glob(".output.staging-*"))


@pytest.mark.parametrize("with_sentinel", (False, True))
def test_noreplace_helper_never_overwrites_existing_empty_or_nonempty_target(
    tmp_path, with_sentinel
):
    runner = _runner_module()
    source = tmp_path / ".staging"
    source.mkdir()
    (source / "summary.json").write_text("staged", encoding="utf-8")
    target = tmp_path / "output"
    target.mkdir()
    sentinel = target / "sentinel"
    if with_sentinel:
        sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner._rename_directory_noreplace(source, target)

    assert source.is_dir()
    assert (source / "summary.json").read_text(encoding="utf-8") == "staged"
    assert target.is_dir()
    if with_sentinel:
        assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_noreplace_helper_fails_closed_when_platform_support_is_absent(
    tmp_path, monkeypatch
):
    runner = _runner_module()
    source = tmp_path / ".staging"
    source.mkdir()
    target = tmp_path / "output"
    monkeypatch.setattr(runner.sys, "platform", "darwin")

    with pytest.raises(OSError) as error:
        runner._rename_directory_noreplace(source, target)

    assert error.value.errno in {errno.ENOSYS, errno.ENOTSUP}
    assert source.is_dir()
    assert not target.exists()


def test_noreplace_helper_fails_closed_for_cross_parent_publication(tmp_path):
    runner = _runner_module()
    source_parent = tmp_path / "source-parent"
    target_parent = tmp_path / "target-parent"
    source_parent.mkdir()
    target_parent.mkdir()
    source = source_parent / ".staging"
    source.mkdir()
    target = target_parent / "output"

    with pytest.raises(OSError) as error:
        runner._rename_directory_noreplace(source, target)

    assert error.value.errno == errno.EXDEV
    assert source.is_dir()
    assert not target.exists()
