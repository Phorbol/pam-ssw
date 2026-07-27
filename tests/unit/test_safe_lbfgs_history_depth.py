"""Behavior tests for the thin safe L-BFGS history-depth runner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.bias import GaussianBiasTerm
from pamssw.result import RelaxOutcomeClass, RelaxResult, RelaxTelemetry
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-safe-lbfgs-history-depth-ablation"
    / "run_gpu_ablation.py"
)
ROW_KEYS = {
    "system",
    "seed",
    "task_id",
    "task_sha256",
    "arm_id",
    "history_limit",
    "fmax_eV_per_A",
    "final_total_biased_energy_eV",
    "final_active_max_force_eV_per_A",
    "iterations",
    "termination_reason",
    "displacement_rms_A",
    "displacement_max_A",
    "outcome_class",
    "force_evaluations",
    "purpose_counts",
    "telemetry",
    "trace",
    "final_positions",
    "final_positions_sha256",
    "wall_time_s",
}
SUMMARY_KEYS = {
    "schema_version",
    "execution_commit",
    "source_summary_sha256",
    "pamssw_bundle_sha256",
    "model_sha256",
    "input_sha256",
    "helper_sha256",
    "runtime_versions",
    "cuda",
    "systems",
    "task_count",
    "row_count",
    "arms",
}


def load_runner():
    spec = importlib.util.spec_from_file_location("history_depth_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state() -> State:
    return State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.2, 0.0, 0.0], [1.0, 0.2, 0.0]]),
        fixed_mask=np.array([True, False]),
    )


def task() -> ProposalRelaxationTask:
    return ProposalRelaxationTask(
        initial_state=state(),
        biases=(
            GaussianBiasTerm(
                center=np.zeros(6),
                direction=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
                sigma=1.0,
                weight=0.0,
            ),
        ),
        softening=None,
        fmax=0.05,
        maxiter=400,
        coordinate_trust_radius=None,
    )


def fake_result(initial_state: State, *, calls: int = 2) -> RelaxResult:
    return RelaxResult(
        state=initial_state,
        energy=-1.25,
        gradient_norm=0.04,
        n_iter=7,
        displacement_rms=0.1,
        displacement_max=0.2,
        outcome_class=RelaxOutcomeClass.USEFUL_PROGRESS,
        telemetry=RelaxTelemetry(
            backend="safe-lbfgs-total",
            evaluator_calls=calls,
            backend_evaluations=max(0, calls - 1),
            reporting_cache_hits=1,
            finalization_requests=1,
            converged=False,
            termination_reason="line_search_failed",
            accepted_steps=7,
        ),
    )


def trace_records(calls: int = 2) -> list[dict[str, object]]:
    return [
        {
            "evaluation_index": index,
            "positions_sha256": "0" * 64,
            "true_energy_eV": -1.0,
            "bias_energy_eV": -0.25,
            "softening_energy_eV": 0.0,
            "total_energy_eV": -1.25,
            "active_max_total_force_eV_per_A": 0.04,
            "accepted_state": index == 1,
        }
        for index in range(1, calls + 1)
    ]


def frozen_task(runner, system: str, seed: int):
    task_id = f"{system}-seed-{seed}-bias-1"
    return SimpleNamespace(
        system=system,
        seed=seed,
        task_id=task_id,
        task_sha256=runner.EXPECTED_TASK_SHA256[task_id],
        relax_task=task(),
    )


def replay(runner, frozen, *, calls: int = 2, wall_time_s: float = 0.25):
    counts = EvaluationCounts.from_mapping(
        {EvaluationPurpose.BIASED_PROPOSAL_RELAX: calls}
    )
    return SimpleNamespace(
        result=fake_result(frozen.relax_task.initial_state, calls=calls),
        evaluation_counts=counts,
        wall_time_s=wall_time_s,
        trace_records=trace_records(calls),
    )


def full_rows(runner):
    return [
        runner._row_payload(
            frozen_task(runner, system, seed),
            arm,
            replay(runner, frozen_task(runner, system, seed)),
        )
        for system in runner.SYSTEMS
        for arm in runner.ARMS
        for seed in runner.SEEDS
    ]


def summary(runner):
    return {
        "schema_version": 1,
        "execution_commit": "e" * 40,
        "source_summary_sha256": runner.EXPECTED_SOURCE_SUMMARY_SHA256,
        "pamssw_bundle_sha256": runner.EXPECTED_PAMSSW_BUNDLE_SHA256,
        "model_sha256": "a" * 64,
        "input_sha256": {"c60": "b" * 64, "pdo": "c" * 64},
        "helper_sha256": runner.EXPECTED_FIXED_REPLAY_DRIVER_SHA256,
        "runtime_versions": {
            "python": "3.test",
            "numpy": "test",
            "scipy": "test",
            "ase": "test",
            "torch": "test",
            "mace": "test",
        },
        "cuda": {
            "requested_device": "cuda",
            "available": True,
            "runtime_version": "12.test",
            "device_name": "test GPU",
        },
        "systems": ["c60", "pdo"],
        "task_count": 16,
        "row_count": 32,
        "arms": [
            {"arm_id": "adaptive-scale-history1", "history_limit": 1},
            {"arm_id": "adaptive-scale-history10", "history_limit": 10},
        ],
    }


def trusted_source(tmp_path: Path, calculator_factory):
    model = tmp_path / "model.pt"
    input_path = tmp_path / "input.xyz"
    model.write_bytes(b"model")
    input_path.write_bytes(b"input")
    return {
        "MODEL": model,
        "MODEL_SHA256": sha256(model),
        "SYSTEMS": {
            "c60": {"input": input_path, "sha256": sha256(input_path)},
            "pdo": {"input": input_path, "sha256": sha256(input_path)},
        },
        "_calculator": calculator_factory,
    }


def runtime_versions():
    return summary(load_runner())["runtime_versions"]


def cuda_info(_):
    return summary(load_runner())["cuda"]


def fake_preflight(runner, calculator_factory):
    return SimpleNamespace(
        tasks_by_system={
            system: tuple(frozen_task(runner, system, seed) for seed in runner.SEEDS)
            for system in runner.SYSTEMS
        },
        source={"_calculator": calculator_factory},
        metadata=summary(runner),
    )


def test_exact_two_arms_and_32_cell_matrix():
    runner = load_runner()
    assert [(arm.arm_id, arm.history_limit) for arm in runner.ARMS] == [
        ("adaptive-scale-history1", 1),
        ("adaptive-scale-history10", 10),
    ]
    assert set(vars(runner.ARMS[0])) == {"arm_id", "history_limit"}
    assert runner.SYSTEMS == ("c60", "pdo")
    assert runner.SEEDS == tuple(range(42, 50))
    assert runner.MAXITER == 400
    assert len(runner._expected_row_keys()) == 32


@pytest.mark.parametrize("tamper", ("source", "pamssw", "helper", "model", "input"))
def test_preflight_rejects_every_pinned_hash_before_calculator(tmp_path, monkeypatch, tamper):
    runner = load_runner()
    calculator_calls = 0

    def calculator_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        return object()

    source = trusted_source(tmp_path, calculator_factory)
    source_summary = runner.SOURCE_SUMMARY_PATH
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True, raising=False)
    if tamper == "source":
        source_summary = tmp_path / "summary.json"
        source_summary.write_text("{}", encoding="utf-8")
    elif tamper == "pamssw":
        monkeypatch.setattr(runner, "_pamssw_bundle_sha256", lambda _: "0" * 64)
    elif tamper == "helper":
        helper = tmp_path / "run_fixed_replay.py"
        helper.write_text("# tampered\n", encoding="utf-8")
        monkeypatch.setattr(runner, "FIXED_REPLAY_DRIVER", helper)
    elif tamper == "model":
        Path(source["MODEL"]).write_bytes(b"tampered")
    elif tamper == "input":
        Path(source["SYSTEMS"]["c60"]["input"]).write_bytes(b"tampered")

    with pytest.raises((ValueError, RuntimeError)):
        runner.preflight(
            source_summary_path=source_summary,
            expected_git_commit=runner._current_commit(),
            source_loader=lambda: source,
            runtime_probe=runtime_versions,
            cuda_probe=cuda_info,
        )
    assert calculator_calls == 0


def test_preflight_only_returns_metadata_before_calculator(tmp_path, monkeypatch):
    runner = load_runner()
    calculator_calls = 0

    def calculator_factory():
        nonlocal calculator_calls
        calculator_calls += 1
        return object()

    source = trusted_source(tmp_path, calculator_factory)
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True, raising=False)
    checked = runner.preflight(
        source_summary_path=runner.SOURCE_SUMMARY_PATH,
        expected_git_commit=runner._current_commit(),
        source_loader=lambda: source,
        runtime_probe=runtime_versions,
        cuda_probe=cuda_info,
    )
    assert set(checked.metadata) == SUMMARY_KEYS
    assert calculator_calls == 0

    monkeypatch.setattr(runner, "preflight", lambda **_: checked)
    result = runner.run(
        output_dir=tmp_path / "output",
        expected_git_commit=runner._current_commit(),
        preflight_only=True,
    )
    assert result == checked.metadata
    assert calculator_calls == 0
    assert not (tmp_path / "output").exists()


def test_relaxation_calls_differ_only_by_history_limit(monkeypatch):
    runner = load_runner()
    calls: list[dict[str, object]] = []

    class FakeRelaxer:
        def __init__(self, *args, **kwargs):
            pass

        def relax(self, initial_state, **kwargs):
            calls.append(kwargs)
            return fake_result(initial_state, calls=0)

    monkeypatch.setattr(runner, "Relaxer", FakeRelaxer)
    runner.replay_task_with_trace(task(), object(), arm=runner.ARMS[0])
    runner.replay_task_with_trace(task(), object(), arm=runner.ARMS[1])
    first = dict(calls[0])
    second = dict(calls[1])
    assert first.pop("_safe_lbfgs_history_limit") == 1
    assert second.pop("_safe_lbfgs_history_limit") == 10
    assert callable(first.pop("trajectory_callback"))
    assert callable(second.pop("trajectory_callback"))
    assert first == second
    assert "_safe_lbfgs_adaptive_scale_without_history" not in first


def test_row_is_raw_finite_and_closes_on_evaluator_calls_not_backend_calls():
    runner = load_runner()
    frozen = frozen_task(runner, "c60", 42)
    row = runner._row_payload(frozen, runner.ARMS[0], replay(runner, frozen))
    assert set(row) == ROW_KEYS
    assert "certificate_satisfied" not in row
    assert len(row["trace"]) == row["force_evaluations"] == 2
    assert row["telemetry"]["evaluator_calls"] == 2
    assert row["telemetry"]["backend_evaluations"] == 1
    assert row["purpose_counts"]["biased_proposal_relax"] == 2
    assert row["purpose_counts"]["unattributed"] == 0
    runner._validate_row(row)


def test_row_rejects_open_accounting_or_nonfinite_raw_values():
    runner = load_runner()
    frozen = frozen_task(runner, "c60", 42)
    row = runner._row_payload(frozen, runner.ARMS[0], replay(runner, frozen))
    open_row = dict(row, trace=row["trace"][:-1])
    nonfinite_row = dict(row, final_total_biased_energy_eV=float("nan"))
    with pytest.raises(ValueError):
        runner._validate_row(open_row)
    with pytest.raises(ValueError):
        runner._validate_row(nonfinite_row)


def test_run_uses_four_isolated_calculators_and_emits_raw_summary(tmp_path, monkeypatch):
    runner = load_runner()
    calculators: list[object] = []
    replayed: list[tuple[object, str]] = []
    captured: dict[str, object] = {}

    def calculator_factory():
        calculator = object()
        calculators.append(calculator)
        return calculator

    monkeypatch.setattr(
        runner,
        "preflight",
        lambda **_: fake_preflight(runner, calculator_factory),
    )

    def fake_replay(relax_task, calculator, *, arm):
        replayed.append((calculator, arm.arm_id))
        frozen = SimpleNamespace(relax_task=relax_task)
        return replay(runner, frozen)

    monkeypatch.setattr(runner, "replay_task_with_trace", fake_replay)
    monkeypatch.setattr(
        runner,
        "_write_output",
        lambda output_dir, rows, metadata: captured.update(rows=rows, metadata=metadata),
        raising=False,
    )
    result = runner.run(
        output_dir=tmp_path / "output",
        expected_git_commit="e" * 40,
    )
    assert len(calculators) == 4
    assert [item[0] for item in replayed[:8]] == [calculators[0]] * 8
    assert [item[0] for item in replayed[8:16]] == [calculators[1]] * 8
    assert [item[0] for item in replayed[16:24]] == [calculators[2]] * 8
    assert [item[0] for item in replayed[24:]] == [calculators[3]] * 8
    assert len(captured["rows"]) == 32
    assert set(captured["metadata"]) == SUMMARY_KEYS
    assert result == captured["metadata"]


def test_output_partial_writes_summary_last_then_renames(tmp_path, monkeypatch):
    runner = load_runner()
    output = tmp_path / "output"
    writes: list[str] = []
    original = runner._write_json

    def record_write(path, payload):
        writes.append(path.name)
        original(path, payload)

    monkeypatch.setattr(runner, "_write_json", record_write)
    runner._write_output(output, full_rows(runner), summary(runner))
    assert writes == ["c60.json", "pdo.json", "summary.json"]
    assert output.is_dir()
    assert not (tmp_path / "output.partial").exists()
    assert set(json.loads((output / "summary.json").read_text())) == SUMMARY_KEYS
    assert (runner.RUN_ROOT / ".gitignore").read_text() == "output/\noutput.partial/\n"


@pytest.mark.parametrize("existing", ("output", "output.partial"))
def test_output_refuses_existing_paths_and_incomplete_partial_has_no_marker(
    tmp_path, monkeypatch, existing
):
    runner = load_runner()
    output = tmp_path / "output"
    occupied = tmp_path / existing
    occupied.mkdir()
    sentinel = occupied / "sentinel"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner._write_output(output, full_rows(runner), summary(runner))
    assert sentinel.read_text() == "preserve"

    other = tmp_path / "other-output"
    original = runner._write_json

    def fail_summary(path, payload):
        if path.name == "summary.json":
            raise OSError("summary failure")
        original(path, payload)

    monkeypatch.setattr(runner, "_write_json", fail_summary)
    with pytest.raises(OSError, match="summary failure"):
        runner._write_output(other, full_rows(runner), summary(runner))
    partial = tmp_path / "other-output.partial"
    assert partial.is_dir()
    assert not (partial / "summary.json").exists()
    assert not other.exists()


def test_cli_requires_execution_commit():
    runner = load_runner()
    with pytest.raises(SystemExit):
        runner._parse_args([])
