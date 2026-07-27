from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.bias import GaussianBiasTerm
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


_RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260727-safe-history-capacity-ablation"
    / "run_gpu_ablation.py"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("safe_history_capacity_runner", _RUNNER_PATH)
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


def test_arm_contract_is_exact_and_contains_no_other_capacity_or_kernel():
    runner = _runner_module()

    assert tuple(
        (arm.arm_id, arm.kernel, arm.history_limit)
        for arm in runner.ARMS
    ) == (
        ("safe-total-gradient-history10", "safe-lbfgs-total", 10),
        ("safe-total-gradient-history0", "safe-lbfgs-total", 0),
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


def test_safe_kernel_descriptor_has_pinned_literal_values_and_canonical_sha():
    runner = _runner_module()
    expected = {
        "optimizer": "safe-lbfgs-total",
        "safe_lbfgs_memory": 10,
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
        "1846e340e762b79f50897dfacd40a16288ed52bb481f36f24ab01594c4724102"
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

    with pytest.raises(ValueError, match="safe kernel descriptor SHA256 mismatch"):
        runner.preflight(
            source_summary_path=runner.SOURCE_SUMMARY_PATH,
            source_loader=lambda: _trusted_source(tmp_path, forbidden_calculator_factory),
            cuda_probe=lambda _: {"device": "cuda"},
        )

    assert calculator_calls == 0


def test_helper_file_provenance_is_pinned_and_fails_before_calculator_factory(
    tmp_path,
    monkeypatch,
):
    runner = _runner_module()

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
        monkeypatch.setattr(runner, path_name, tampered)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            runner.preflight(
                source_summary_path=runner.SOURCE_SUMMARY_PATH,
                source_loader=lambda: _trusted_source(tmp_path, forbidden_calculator_factory),
                cuda_probe=lambda _: {"device": "cuda"},
            )
        assert calculator_calls == 0
        monkeypatch.undo()


@pytest.mark.parametrize("tamper_target", ("task", "model", "input", "cuda"))
def test_tampered_preflight_fails_before_any_calculator_call(tmp_path, tamper_target):
    runner = _runner_module()
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
            source_loader=lambda: source,
            cuda_probe=cuda_probe,
        )

    assert calculator_calls == 0


def test_atomic_publish_validates_complete_32_rows_and_cleans_staging_on_failure(tmp_path):
    runner = _runner_module()
    output_dir = tmp_path / "ledger"
    rows = runner._expected_row_keys()
    incomplete = rows[:-1]

    with pytest.raises(ValueError, match="32-row ledger"):
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
    )

    assert replay.trace_records
    assert len(replay.trace_records) == replay.evaluation_counts.total
    assert len(replay.trace_records) == replay.result.telemetry.evaluator_calls
    assert calculator.calls == len(replay.trace_records)
    assert replay.evaluation_counts.as_dict()["biased_proposal_relax"] == len(replay.trace_records)
    assert replay.evaluation_counts.as_dict()["unattributed"] == 0
    assert replay.certificate_satisfied
    assert all(np.isfinite(record["total_energy_eV"]) for record in replay.trace_records)
