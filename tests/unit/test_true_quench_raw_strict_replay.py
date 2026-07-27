"""Contract tests for the corpus-backed strict true-quench replay."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import sys

import numpy as np
import pytest
from pathlib import Path

from pamssw.accounting import EvalCounter, EvaluationPurpose


ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-true-quench-raw-strict-replay"
)
RUNNER_PATH = ROOT / "run_replay.py"


def module(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def test_runner_is_a_new_independent_corpus_replay_artifact():
    assert RUNNER_PATH.is_file()


def test_protocol_is_fixed_to_raw_corpus_and_five_strict_true_pes_arms():
    runner = module("raw_strict_protocol")
    assert getattr(runner, "SYSTEMS", None) == ("c60", "pdo")
    assert getattr(runner, "TASKS_PER_SYSTEM", None) == 16
    assert getattr(runner, "STRICT_FMAX", None) == 0.01
    assert getattr(runner, "MAXITER", None) == 400
    assert getattr(runner, "OBJECTIVE", None) == "true_mace_pes_no_bias_no_softening"
    assert getattr(runner, "FROZEN_CORPUS_SHA256", None) == (
        "100759e1871cdefb54972f91751c452763f74d0e34ee576f268a5808219a552b"
    )
    assert [
        (arm.arm_id, arm.optimizer, arm.safe_history_limit)
        for arm in runner.ARMS
    ] == [
        ("scipy-lbfgsb", "scipy-lbfgsb", None),
        ("safe-lbfgs-total", "safe-lbfgs-total", 10),
        ("ase-fire", "ase-fire", None),
        ("ase-fire2", "ase-fire2", None),
        ("ase-lbfgs", "ase-lbfgs", None),
    ]


def position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def corpus_payload() -> dict:
    systems = {}
    for system in ("c60", "pdo"):
        entries = []
        for task_index in range(16):
            positions = [[float(task_index), 0.0, 0.0], [0.0, 1.0, 0.0]]
            state = {
                "numbers": [6, 6] if system == "c60" else [46, 8],
                "positions": positions,
                "cell": None if system == "c60" else [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 10.0]],
                "pbc": [False, False, False] if system == "c60" else [True, True, False],
                "fixed_mask": [False, False] if system == "c60" else [True, False],
            }
            entries.append(
                {
                    "system": system,
                    "task_index": task_index,
                    "trial_index": task_index + 1,
                    "proposal_index": 1,
                    "source_trajectory_path": f"stage_a/{system}/{task_index}.xyz",
                    "source_trajectory_sha256": "a" * 64,
                    "source_frame_index": 0,
                    "initial_positions_sha256": position_sha256(positions),
                    "state": state,
                }
            )
        systems[system] = {
            "system": system,
            "task_count": 16,
            "capture_policy_conditioned": True,
            "entries": entries,
        }
    return {"capture_policy_conditioned": True, "systems": systems}


def test_load_corpus_validates_all_raw_states_and_records_source_sha(tmp_path):
    runner = module("raw_strict_corpus")
    assert callable(getattr(runner, "load_corpus", None))
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus_payload()), encoding="utf-8")

    corpus = runner.load_corpus(corpus_path)

    assert corpus.source_sha256 == sha256(corpus_path.read_bytes()).hexdigest()
    assert tuple(corpus.tasks_by_system) == ("c60", "pdo")
    for system, tasks in corpus.tasks_by_system.items():
        assert len(tasks) == 16
        assert [task.task_index for task in tasks] == list(range(16))
        assert all(task.system == system for task in tasks)
        assert all(
            task.initial_positions_sha256
            == position_sha256(task.state.positions)
            for task in tasks
        )


def test_preflight_requires_pinned_clean_tracked_corpus_and_records_provenance(
    tmp_path, monkeypatch
):
    runner = module("raw_strict_preflight")
    assert callable(getattr(runner, "preflight", None))
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus_payload()), encoding="utf-8")
    model_path = tmp_path / "model.model"
    model_path.write_bytes(b"model")
    input_paths = {}
    for system in ("c60", "pdo"):
        input_path = tmp_path / f"{system}.cif"
        input_path.write_text(system, encoding="utf-8")
        input_paths[system] = input_path
    production = type(
        "Production",
        (),
        {
            "MODEL_PATH": model_path,
            "INPUT_PATHS": input_paths,
            "CALCULATOR_CONFIG": {"device": "cuda", "default_dtype": "float64"},
        },
    )()
    monkeypatch.setattr(runner, "_production_module", lambda: production)
    monkeypatch.setattr(runner, "_safe_lbfgs_default_history_limit", lambda: 10)
    monkeypatch.setattr(runner, "_current_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "_is_tracked_file", lambda path: True)
    monkeypatch.setattr(
        runner,
        "FROZEN_CORPUS_SHA256",
        sha256(corpus_path.read_bytes()).hexdigest(),
        raising=False,
    )

    checked = runner.preflight(
        expected_git_commit="a" * 40,
        corpus_path=corpus_path,
        runtime_probe=lambda: {"python": "3.12"},
        cuda_probe=lambda: {"available": True, "device_name": "fake-gpu"},
    )

    assert checked.corpus.source_sha256 == sha256(corpus_path.read_bytes()).hexdigest()
    assert checked.metadata["execution_commit"] == "a" * 40
    assert checked.metadata["corpus"]["sha256"] == checked.corpus.source_sha256
    assert checked.metadata["model"]["sha256"] == sha256(b"model").hexdigest()
    assert checked.metadata["inputs"]["c60"]["sha256"] == sha256(b"c60").hexdigest()
    assert checked.metadata["runtime_versions"] == {"python": "3.12"}
    assert checked.metadata["cuda"]["device_name"] == "fake-gpu"

    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: False)
    with pytest.raises(RuntimeError, match="tracked worktree is not clean"):
        runner.preflight(
            expected_git_commit="a" * 40,
            corpus_path=corpus_path,
            runtime_probe=lambda: {},
            cuda_probe=lambda: {"available": True},
        )

    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "FROZEN_CORPUS_SHA256", "d" * 64)
    with pytest.raises(ValueError, match="frozen corpus SHA-256 mismatch"):
        runner.preflight(
            expected_git_commit="a" * 40,
            corpus_path=corpus_path,
            runtime_probe=lambda: {},
            cuda_probe=lambda: {"available": True},
        )


def test_run_reuses_one_analytic_calculator_per_system_arm_and_closes_ledger(
    tmp_path,
):
    runner = module("raw_strict_run")
    assert callable(getattr(runner, "run", None))
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus_payload()), encoding="utf-8")
    corpus = runner.load_corpus(corpus_path)
    checked = runner.Preflight(
        metadata={
            "schema_version": 1,
            "execution_commit": "a" * 40,
            "corpus": {"sha256": corpus.source_sha256},
            "model": {"sha256": "b" * 64},
            "inputs": {system: {"sha256": "c" * 64} for system in runner.SYSTEMS},
            "runtime_versions": {"python": "3.12"},
            "cuda": {"available": True, "device_name": "analytic-fake"},
            "calculator": {"device": "cuda"},
            "safe_lbfgs_default_history_limit": 10,
        },
        corpus=corpus,
    )
    calculators = []

    class Quadratic:
        def __init__(self):
            self.calls = 0

        def evaluate_flat(self, flat, template):
            self.calls += 1
            vector = np.asarray(flat, dtype=float)
            return 0.5 * float(vector @ vector), vector.copy()

    def calculator_factory():
        calculator = Quadratic()
        calculators.append(calculator)
        return calculator

    output = tmp_path / "output"
    summary = runner.run(
        output_dir=output,
        expected_git_commit="a" * 40,
        preflight_fn=lambda **_: checked,
        calculator_factory=calculator_factory,
        counter_factory=EvalCounter,
    )

    assert len(calculators) == len(runner.SYSTEMS) * len(runner.ARMS)
    assert all(calculator.calls >= 16 for calculator in calculators)
    assert summary["row_count"] == 160
    assert summary["strict_protocol"] == {
        "fmax_eV_per_A": 0.01,
        "maxiter": 400,
        "objective": "true_mace_pes_no_bias_no_softening",
    }
    rows = json.loads((output / "rows.json").read_text(encoding="utf-8"))
    assert len(rows) == 160
    assert not output.with_name("output.partial").exists()
    for system in runner.SYSTEMS:
        for task_index in range(16):
            task_rows = [
                row
                for row in rows
                if row["system"] == system and row["task_index"] == task_index
            ]
            assert len(task_rows) == len(runner.ARMS)
            assert {
                row["initial"]["positions_sha256"] for row in task_rows
            } == {corpus.tasks_by_system[system][task_index].initial_positions_sha256}
    for row in rows:
        assert row["fmax_eV_per_A"] == 0.01
        assert row["maxiter"] == 400
        assert row["purpose_count_delta"]["landing_true_quench"] == row["evaluator_calls"]
        assert row["purpose_count_delta"]["unattributed"] == 0
        assert row["purpose_count_delta"][EvaluationPurpose.LANDING_TRUE_QUENCH.value] > 0
