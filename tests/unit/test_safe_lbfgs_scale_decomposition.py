from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pamssw.accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from pamssw.bias import GaussianBiasTerm
from pamssw.relax import Relaxer
from pamssw.result import RelaxResult, RelaxTelemetry
from pamssw.state import State
from pamssw.walker import ProposalPotential, ProposalRelaxationTask


_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-safe-lbfgs-scale-decomposition"
    / "run_gpu_ablation.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("safe_lbfgs_scale_decomposition_runner", _RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


class _QuadraticCountingCalculator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self.calls += 1
        target = np.zeros_like(flat_positions, dtype=float)
        target[3] = 0.35
        displacement = np.asarray(flat_positions, dtype=float) - target
        del template
        return float(0.5 * np.dot(displacement, displacement)), displacement


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_module_is_created_for_the_ablation_contract():
    assert _RUNNER_PATH.is_file()


def test_default_output_path_is_untracked_and_run_local_ignore_covers_generated_ledger():
    runner = _runner_module()
    output_path = runner.RUN_ROOT / "output"
    ignore_path = runner.RUN_ROOT / ".gitignore"
    output_relative = output_path.relative_to(runner.REPO_ROOT)

    assert runner.OUTPUT_DIR == output_path
    assert ignore_path.read_text(encoding="utf-8") == "output/\n"
    tracked = subprocess.run(
        ["git", "ls-files", "--", str(output_relative)],
        cwd=runner.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked.stdout == ""
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            str(output_relative / "hypothetical-result.json"),
        ],
        cwd=runner.REPO_ROOT,
    )
    assert ignored.returncode == 0


def test_arm_contract_is_exact_and_contains_no_other_capacity_or_kernel():
    runner = _runner_module()

    assert tuple(
        (
            arm.arm_id,
            arm.kernel,
            arm.history_limit,
            arm.adaptive_scale_without_history,
            arm.scale_policy,
        )
        for arm in runner.ARMS
    ) == (
        (
            "fixed-scale-history0",
            "safe-lbfgs-total",
            0,
            False,
            "fixed-1-over-70",
        ),
        (
            "adaptive-scale-history0",
            "safe-lbfgs-total",
            0,
            True,
            "latest-accepted-secant-gamma",
        ),
        (
            "adaptive-scale-history10",
            "safe-lbfgs-total",
            10,
            False,
            "latest-history-pair-gamma-plus-two-loop",
        ),
    )


def test_pinned_source_sha_and_exact_16_task_ids_and_seeds_are_verified():
    runner = _runner_module()

    assert runner.EXPECTED_SOURCE_SUMMARY_SHA256 == (
        "62cc771e2aa24e9addef0e870d0524f901f02bddc34152cf2a2eeea91a671b04"
    )
    assert _sha256(runner.SOURCE_SUMMARY_PATH) == runner.EXPECTED_SOURCE_SUMMARY_SHA256
    tasks_by_system, source_sha = runner.load_fixed_tasks(runner.SOURCE_SUMMARY_PATH)

    assert source_sha == runner.EXPECTED_SOURCE_SUMMARY_SHA256
    assert tuple(tasks_by_system) == ("c60", "pdo")
    for system in ("c60", "pdo"):
        expected_ids = tuple(f"{system}-seed-{seed}-bias-1" for seed in range(42, 50))
        assert tuple(task.task_id for task in tasks_by_system[system]) == expected_ids
        assert tuple(task.seed for task in tasks_by_system[system]) == tuple(range(42, 50))
        assert len(tasks_by_system[system]) == 8
        assert all(task.relax_task.maxiter == 400 for task in tasks_by_system[system])


def test_canonical_task_sha_changes_when_task_payload_is_mutated():
    runner = _runner_module()
    source = json.loads(runner.SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))
    payload = source["systems"][0]["tasks"][0]["task"]

    original = runner.canonical_task_sha256(payload)
    mutated = json.loads(json.dumps(payload))
    mutated["fmax"] = float(mutated["fmax"]) * 2.0

    assert len(original) == 64
    assert runner.canonical_task_sha256(mutated) != original


def test_kernel_descriptor_binds_current_safe_lbfgs_constants_and_memory10():
    runner = _runner_module()
    import pamssw.relax as relax_module

    descriptor = runner.objective_descriptor()

    assert descriptor["optimizer"] == "safe-lbfgs-total"
    assert descriptor["safe_lbfgs_memory"] == 10
    for name in (
        "_SAFE_LBFGS_EMPTY_HISTORY_SCALE",
        "_SAFE_LBFGS_MAX_ATOM_STEP",
        "_SAFE_LBFGS_ARMIJO_C1",
        "_SAFE_LBFGS_BACKTRACK",
        "_SAFE_LBFGS_MAX_LINE_TRIALS",
        "_SAFE_LBFGS_MIN_ALPHA",
        "_SAFE_LBFGS_CURVATURE_REL",
    ):
        assert descriptor["kernel_constants"][name] == getattr(relax_module, name)


def _trusted_source(tmp_path, calculator_factory):
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.xyz"
    model_path.write_bytes(b"trusted model")
    input_path.write_bytes(b"trusted input")
    return {
        "MODEL": model_path,
        "MODEL_SHA256": _sha256(model_path),
        "SYSTEMS": {
            "c60": {"input": input_path, "sha256": _sha256(input_path)},
            "pdo": {"input": input_path, "sha256": _sha256(input_path)},
        },
        "_calculator": calculator_factory,
    }


def _replay_without_observer(
    task,
    calculator,
    *,
    history_limit,
    adaptive_scale_without_history,
    maxiter,
):
    counter = EvalCounter(calculator)
    proposal = ProposalPotential(
        counter,
        biases=list(task.biases),
        softening=task.softening,
    )
    relaxer = Relaxer(
        proposal.evaluate,
        optimizer="safe-lbfgs-total",
        component_evaluator=proposal.evaluate_parts,
    )
    with counter.purpose(EvaluationPurpose.BIASED_PROPOSAL_RELAX):
        result = relaxer.relax(
            task.initial_state,
            fmax=task.fmax,
            maxiter=maxiter,
            coordinate_trust_radius=task.coordinate_trust_radius,
            _safe_lbfgs_history_limit=history_limit,
            _safe_lbfgs_adaptive_scale_without_history=(
                adaptive_scale_without_history
            ),
        )
    return result, counter.snapshot()


def test_pamssw_source_bundle_and_import_paths_are_pinned_to_this_worktree():
    runner = _runner_module()

    assert runner.EXPECTED_PAMSSW_BUNDLE_SHA256 == (
        "96761a45dfe7c8af459ba8112adb79efee4d53ce073a74f7d87fea09c35c9d2a"
    )
    provenance = runner._verified_pamssw_source()

    assert provenance["bundle_sha256"] == runner.EXPECTED_PAMSSW_BUNDLE_SHA256
    assert provenance["source_root"] == str(runner.REPO_ROOT / "pamssw")
    assert set(provenance["imported_module_paths"]) >= {
        "pamssw.accounting",
        "pamssw.relax",
        "pamssw.walker",
    }
    assert all(
        Path(path).is_relative_to(runner.REPO_ROOT / "pamssw")
        for path in provenance["imported_module_paths"].values()
    )


@pytest.mark.parametrize("failure", ("commit", "dirty", "import_path", "bundle"))
def test_repo_or_pamssw_preflight_failures_precede_calculator_factory(
    tmp_path,
    monkeypatch,
    failure,
):
    runner = _runner_module()
    calculator_calls = 0

    def forbidden_calculator_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        raise AssertionError("calculator construction must follow source preflight")

    expected_commit = runner._current_commit()
    if failure == "commit":
        expected_commit = "0" * 40
    elif failure == "dirty":
        monkeypatch.setattr(runner, "_worktree_is_clean", lambda: False, raising=False)
    elif failure == "import_path":
        monkeypatch.setattr(
            runner,
            "_imported_pamssw_source_paths",
            lambda: {"pamssw.relax": Path("/tmp/foreign-site-packages/pamssw/relax.py")},
            raising=False,
        )
    elif failure == "bundle":
        monkeypatch.setattr(
            runner,
            "_pamssw_bundle_sha256",
            lambda _: "0" * 64,
            raising=False,
        )

    with pytest.raises((RuntimeError, ValueError), match="(commit|worktree|import|bundle)"):
        runner.preflight(
            source_summary_path=runner.SOURCE_SUMMARY_PATH,
            expected_git_commit=expected_commit,
            source_loader=lambda: _trusted_source(tmp_path, forbidden_calculator_factory),
            cuda_probe=lambda _: {"device": "cuda"},
        )

    assert calculator_calls == 0


def test_cli_requires_expected_git_commit():
    runner = _runner_module()

    with pytest.raises(SystemExit):
        runner._parse_args([])


def test_safe_kernel_descriptor_has_pinned_literal_values_and_canonical_sha():
    runner = _runner_module()
    expected = {
        "optimizer": "safe-lbfgs-total",
        "safe_lbfgs_memory": 10,
        "scale_policies": {
            "fixed-1-over-70": "gamma=1/70 with empty two-loop history",
            "latest-accepted-secant-gamma": (
                "gamma=(s.T@y)/(y.T@y) with empty two-loop history"
            ),
            "latest-history-pair-gamma-plus-two-loop": (
                "gamma=(s.T@y)/(y.T@y) from newest history pair "
                "plus two-loop corrections"
            ),
        },
        "kernel_constants": {
            "_SAFE_LBFGS_EMPTY_HISTORY_SCALE": 1.0 / 70.0,
            "_SAFE_LBFGS_MAX_ATOM_STEP": 0.2,
            "_SAFE_LBFGS_ARMIJO_C1": 1.0e-4,
            "_SAFE_LBFGS_BACKTRACK": 0.5,
            "_SAFE_LBFGS_MAX_LINE_TRIALS": 20,
            "_SAFE_LBFGS_MIN_ALPHA": 2.0**-20,
            "_SAFE_LBFGS_CURVATURE_REL": 1.4901161193847656e-08,
        },
    }

    assert runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR == expected
    assert runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256 == (
        "ba0c261cd588ce72e8175332385389e69cf8840c00399318723af7e0358e7f24"
    )
    assert runner.canonical_descriptor_sha256(expected) == (
        runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256
    )
    assert runner.objective_descriptor() == expected


def test_kernel_constant_drift_fails_preflight_before_calculator_factory(tmp_path, monkeypatch):
    runner = _runner_module()
    import pamssw.relax as relax_module

    calculator_calls = 0

    def forbidden_calculator_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        raise AssertionError("calculator construction must follow kernel validation")

    monkeypatch.setattr(relax_module, "_SAFE_LBFGS_BACKTRACK", 0.4)
    monkeypatch.setattr(runner, "_worktree_is_clean", lambda: True)

    with pytest.raises(ValueError, match="safe kernel descriptor SHA256 mismatch"):
        runner.preflight(
            source_summary_path=runner.SOURCE_SUMMARY_PATH,
            expected_git_commit=runner._current_commit(),
            source_loader=lambda: _trusted_source(tmp_path, forbidden_calculator_factory),
            cuda_probe=lambda _: {"device": "cuda"},
        )

    assert calculator_calls == 0


def test_helper_file_provenance_is_pinned_and_fails_before_calculator_factory(
    tmp_path,
    monkeypatch,
):
    runner = _runner_module()
    monkeypatch.setattr(runner, "_worktree_is_clean", lambda: True)

    assert runner.EXPECTED_FIXED_REPLAY_DRIVER_SHA256 == (
        "f9c9602e42985891a6ca2a84ca70dda69c52b9d794a6ea6c76857c398345fa8f"
    )
    assert runner.EXPECTED_TRACE_RECORDER_SHA256 == (
        "c6feeaabf0062f6654f8ea4b4fff610b258dcab754e1907dd3f4c3779e7165de"
    )
    assert runner.EXPECTED_G1_DRIVER_SHA256 == (
        "0e69736d5d92372a2f4e449c440c09307613a55f0028126c7bb36d77296bbfbb"
    )
    assert _sha256(runner.FIXED_REPLAY_DRIVER) == runner.EXPECTED_FIXED_REPLAY_DRIVER_SHA256
    assert _sha256(runner.TRACE_RECORDER_PATH) == runner.EXPECTED_TRACE_RECORDER_SHA256
    assert _sha256(runner.G1_DRIVER_PATH) == runner.EXPECTED_G1_DRIVER_SHA256

    for path_name in ("FIXED_REPLAY_DRIVER", "G1_DRIVER_PATH", "TRACE_RECORDER_PATH"):
        calculator_calls = 0

        def forbidden_calculator_factory():
            nonlocal calculator_calls
            calculator_calls += 1
            raise AssertionError("calculator construction must follow helper validation")

        tampered = tmp_path / f"{path_name}.py"
        tampered.write_text("# tampered helper\n", encoding="utf-8")
        with monkeypatch.context() as patched:
            patched.setattr(runner, path_name, tampered)
            with pytest.raises(ValueError, match="SHA256 mismatch"):
                runner.preflight(
                    source_summary_path=runner.SOURCE_SUMMARY_PATH,
                    expected_git_commit=runner._current_commit(),
                    source_loader=lambda: _trusted_source(tmp_path, forbidden_calculator_factory),
                    cuda_probe=lambda _: {"device": "cuda"},
                )
        assert calculator_calls == 0


@pytest.mark.parametrize("tamper_target", ("task", "model", "input", "cuda"))
def test_tampered_preflight_fails_before_any_calculator_call(tmp_path, monkeypatch, tamper_target):
    runner = _runner_module()
    monkeypatch.setattr(runner, "_worktree_is_clean", lambda: True)
    model_path = tmp_path / "model.pt"
    input_path = tmp_path / "input.xyz"
    model_path.write_bytes(b"trusted model")
    input_path.write_bytes(b"trusted input")
    calculator_calls = 0

    def forbidden_calculator_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        raise AssertionError("calculator construction must follow all preflight checks")

    source = {
        "MODEL": model_path,
        "MODEL_SHA256": _sha256(model_path),
        "SYSTEMS": {
            "c60": {"input": input_path, "sha256": _sha256(input_path)},
            "pdo": {"input": input_path, "sha256": _sha256(input_path)},
        },
        "_calculator": forbidden_calculator_factory,
    }
    source_summary = runner.SOURCE_SUMMARY_PATH
    if tamper_target == "task":
        source_summary = tmp_path / "tampered-summary.json"
        payload = json.loads(runner.SOURCE_SUMMARY_PATH.read_text(encoding="utf-8"))
        payload["systems"][0]["tasks"][0]["task_id"] = "forged-task-id"
        source_summary.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper_target == "model":
        model_path.write_bytes(b"tampered model")
    elif tamper_target == "input":
        input_path.write_bytes(b"tampered input")

    def cuda_probe(_: object):
        if tamper_target == "cuda":
            raise RuntimeError("torch.cuda.is_available() is False")
        return {"device": "cuda"}

    with pytest.raises((ValueError, RuntimeError), match="(SHA256 mismatch|cuda|CUDA|source summary)"):
        runner.preflight(
            source_summary_path=source_summary,
            expected_git_commit=runner._current_commit(),
            source_loader=lambda: source,
            cuda_probe=cuda_probe,
        )

    assert calculator_calls == 0


def test_atomic_publish_validates_complete_48_rows_and_cleans_staging_on_failure(tmp_path):
    runner = _runner_module()
    output_dir = tmp_path / "ledger"
    rows = runner._expected_row_keys()
    incomplete = rows[:-1]

    with pytest.raises(ValueError, match="48-row ledger"):
        runner.publish_ledger_atomically(output_dir, incomplete, {"schema_version": 1})

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".ledger.staging-*"))

    write_attempts: list[str] = []

    def fail_on_summary(path, payload):
        del payload
        write_attempts.append(path.name)
        if path.name == "summary.json":
            raise OSError("injected summary write failure")
        runner._write_json_atomic(path, {"written": path.name})

    with pytest.raises(OSError, match="injected summary write failure"):
        runner.publish_ledger_atomically(
            output_dir,
            rows,
            {"schema_version": 1},
            write_json=fail_on_summary,
        )

    assert write_attempts == ["c60.json", "pdo.json", "summary.json"]
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".ledger.staging-*"))


def test_atomic_publish_refuses_to_overwrite_existing_output(tmp_path):
    runner = _runner_module()
    output_dir = tmp_path / "ledger"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError):
        runner.publish_ledger_atomically(
            output_dir,
            runner._expected_row_keys(),
            {"schema_version": 1},
        )

    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_replay_observer_adds_zero_pes_calls_and_closes_the_ledger():
    runner = _runner_module()
    calculator = _QuadraticCountingCalculator()

    replay = runner.replay_task_with_trace(
        _task(),
        calculator,
        history_limit=0,
        adaptive_scale_without_history=True,
    )

    assert replay.trace_records
    assert len(replay.trace_records) == replay.evaluation_counts.total
    assert len(replay.trace_records) == replay.result.telemetry.evaluator_calls
    assert calculator.calls == len(replay.trace_records)
    assert replay.evaluation_counts.as_dict()["biased_proposal_relax"] == len(replay.trace_records)
    assert replay.evaluation_counts.as_dict()["unattributed"] == 0
    assert replay.certificate_satisfied
    assert all(np.isfinite(record["total_energy_eV"]) for record in replay.trace_records)


def test_recording_observer_matches_plain_relaxer_without_extra_pes_calls():
    runner = _runner_module()
    task = _task()
    history_limit = 0
    adaptive_scale_without_history = True
    plain_calculator = _QuadraticCountingCalculator()
    observed_calculator = _QuadraticCountingCalculator()

    plain_result, plain_counts = _replay_without_observer(
        task,
        plain_calculator,
        history_limit=history_limit,
        adaptive_scale_without_history=adaptive_scale_without_history,
        maxiter=runner.MAXITER,
    )
    observed = runner.replay_task_with_trace(
        task,
        observed_calculator,
        history_limit=history_limit,
        adaptive_scale_without_history=adaptive_scale_without_history,
    )

    assert plain_calculator.calls == observed_calculator.calls
    assert plain_calculator.calls == plain_counts.total
    assert observed_calculator.calls == observed.evaluation_counts.total
    assert np.array_equal(plain_result.state.positions, observed.result.state.positions)
    assert plain_result.energy == pytest.approx(observed.result.energy, abs=0.0)
    assert plain_result.telemetry.evaluator_calls == observed.result.telemetry.evaluator_calls
    assert len(observed.trace_records) == observed_calculator.calls


def _finite_trace_record() -> dict[str, float | bool]:
    return {
        "true_energy_eV": -1.0,
        "bias_energy_eV": 0.0,
        "softening_energy_eV": 0.0,
        "total_energy_eV": -1.0,
        "active_max_total_force_eV_per_A": 0.1,
        "accepted_state": True,
    }


def _fake_preflight_for_full_run(runner):
    tasks_by_system, source_sha256 = runner.load_fixed_tasks(runner.SOURCE_SUMMARY_PATH)
    return runner.Preflight(
        tasks_by_system=tasks_by_system,
        source_summary_sha256=source_sha256,
        source={"_calculator": lambda: object()},
        safe_kernel_descriptor=runner.objective_descriptor(),
        safe_kernel_descriptor_sha256=runner.EXPECTED_SAFE_KERNEL_DESCRIPTOR_SHA256,
        helper_provenance={},
        pamssw_source_provenance={},
        repo_provenance={"actual_git_commit": "f" * 40},
        cuda_provenance={},
    )


def _fake_replay(task, *, certificate_satisfied, energy=-1.0, trace_records=None, evaluator_calls=2):
    counts = EvaluationCounts.from_mapping(
        {EvaluationPurpose.BIASED_PROPOSAL_RELAX: evaluator_calls}
    )
    if trace_records is None:
        trace_records = [_finite_trace_record() for _ in range(evaluator_calls)]
    return RelaxResult(
        state=task.initial_state,
        energy=energy,
        gradient_norm=0.1,
        n_iter=400,
        telemetry=RelaxTelemetry(
            backend="safe-lbfgs-total",
            evaluator_calls=evaluator_calls,
            backend_evaluations=evaluator_calls,
            converged=False,
            termination_reason="maxiter",
        ),
    ), counts, trace_records, certificate_satisfied


def test_finite_certificate_failure_is_published_as_a_scientific_result(tmp_path, monkeypatch):
    runner = _runner_module()
    checked = _fake_preflight_for_full_run(runner)
    calculator_calls = 0

    def calculator_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        return object()

    checked.source["_calculator"] = calculator_factory
    monkeypatch.setattr(runner, "preflight", lambda **_: checked)

    def finite_maxiter_replay(
        task,
        calculator,
        *,
        history_limit,
        adaptive_scale_without_history,
    ):
        del calculator, history_limit, adaptive_scale_without_history
        result, counts, records, certificate = _fake_replay(
            task,
            certificate_satisfied=False,
        )
        return runner.ReplayResult(result, counts, 0.0, records, (), certificate)

    monkeypatch.setattr(runner, "replay_task_with_trace", finite_maxiter_replay)
    output_dir = tmp_path / "output"

    summary = runner.run(
        output_dir=output_dir,
        expected_git_commit="f" * 40,
    )

    assert summary["row_count"] == 48
    assert summary["certificate_all_satisfied"] is False
    assert summary["certificate_unsatisfied_count"] == 48
    assert summary["termination_reason_counts"] == {"maxiter": 48}
    assert calculator_calls == 6
    assert output_dir.is_dir()
    assert not list(tmp_path.glob(".output.staging-*"))
    rows = json.loads((output_dir / "c60.json").read_text(encoding="utf-8"))["rows"]
    row = next(
        item
        for item in rows
        if item["task_id"] == "c60-seed-44-bias-1"
        and item["arm_id"] == "adaptive-scale-history0"
    )
    assert row["adaptive_scale_without_history"] is True
    assert row["scale_policy"] == "latest-accepted-secant-gamma"
    assert row["certificate_satisfied"] is False
    assert row["termination_reason"] == "maxiter"


@pytest.mark.parametrize("failure", ("nonfinite", "ledger"))
def test_nonfinite_or_open_ledger_remains_fatal(tmp_path, monkeypatch, failure):
    runner = _runner_module()
    monkeypatch.setattr(runner, "preflight", lambda **_: _fake_preflight_for_full_run(runner))

    def invalid_replay(
        task,
        calculator,
        *,
        history_limit,
        adaptive_scale_without_history,
    ):
        del calculator, history_limit, adaptive_scale_without_history
        if failure == "nonfinite":
            result, counts, records, certificate = _fake_replay(
                task,
                certificate_satisfied=False,
                energy=float("nan"),
            )
        else:
            result, counts, records, certificate = _fake_replay(
                task,
                certificate_satisfied=False,
                trace_records=[_finite_trace_record()],
            )
        return runner.ReplayResult(result, counts, 0.0, records, (), certificate)

    monkeypatch.setattr(runner, "replay_task_with_trace", invalid_replay)
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="(non-finite|ledger mismatch)"):
        runner.run(output_dir=output_dir, expected_git_commit="f" * 40)

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".output.staging-*"))
