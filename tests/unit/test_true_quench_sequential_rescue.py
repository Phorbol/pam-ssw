"""Contract tests for the certificate-triggered strict-quench rescue replay."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.accounting import EvalCounter, EvaluationPurpose


ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-true-quench-sequential-rescue"
)
BUILDER_PATH = ROOT / "build_corpus.py"
RUNNER_PATH = ROOT / "run_rescue.py"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def state_payload(system: str, task_index: int, *, terminal: bool) -> dict:
    shift = 0.01 if terminal else 0.0
    positions = [
        [float(task_index) + shift, 0.0, 0.0],
        [0.0, 1.0 + shift, 0.0],
    ]
    return {
        "numbers": [6, 6] if system == "c60" else [46, 8],
        "positions": positions,
        "cell": (
            None
            if system == "c60"
            else [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 10.0]]
        ),
        "pbc": [False, False, False] if system == "c60" else [True, True, False],
        "fixed_mask": [False, False] if system == "c60" else [True, False],
    }


def replay_payloads() -> tuple[list[dict], dict]:
    failures = {("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4)}
    arms = (
        ("scipy-lbfgsb", "scipy-lbfgsb"),
        ("safe-lbfgs-total", "safe-lbfgs-total"),
        ("ase-fire", "ase-fire"),
        ("ase-fire2", "ase-fire2"),
        ("ase-lbfgs", "ase-lbfgs"),
    )
    rows = []
    for system in ("c60", "pdo"):
        for arm_id, optimizer in arms:
            for task_index in range(16):
                initial = state_payload(system, task_index, terminal=False)
                final = state_payload(system, task_index, terminal=True)
                force = (
                    0.02
                    if arm_id == "ase-lbfgs" and (system, task_index) in failures
                    else 0.005
                )
                force_evaluations = 100 + task_index
                rows.append(
                    {
                        "system": system,
                        "task_index": task_index,
                        "trial_index": task_index + 1,
                        "proposal_index": 1,
                        "source_trajectory_path": f"stage_a/{system}/{task_index}.xyz",
                        "source_trajectory_sha256": "a" * 64,
                        "source_frame_index": 0,
                        "arm_id": arm_id,
                        "optimizer": optimizer,
                        "safe_history_limit": 10 if arm_id == "safe-lbfgs-total" else None,
                        "fmax_eV_per_A": 0.01,
                        "maxiter": 400,
                        "coordinate_trust_radius_A": None,
                        "objective": "true_mace_pes_no_bias_no_softening",
                        "initial": {
                            "energy_eV": -10.0,
                            "max_active_force_eV_per_A": 1.0,
                            "positions_sha256": position_sha256(initial["positions"]),
                            "state": initial,
                        },
                        "final": {
                            "energy_eV": -11.0 - task_index,
                            "max_active_force_eV_per_A": force,
                            "positions_sha256": position_sha256(final["positions"]),
                            "state": final,
                        },
                        "n_iter": 400,
                        "termination_reason": (
                            "maxiter" if force > 0.01 else "converged"
                        ),
                        "outcome_class": (
                            "useful_progress" if force > 0.01 else "converged"
                        ),
                        "telemetry": {"backend": optimizer},
                        "evaluator_calls": force_evaluations,
                        "force_evaluations": force_evaluations,
                        "purpose_count_delta": {
                            "unattributed": 0,
                            "bootstrap_true_quench": 0,
                            "starter_true_quench": 0,
                            "direction_oracle": 0,
                            "biased_proposal_relax": 0,
                            "escape_true_pes_check": 0,
                            "landing_true_quench": force_evaluations,
                            "post_relax_validation": 0,
                        },
                        "wall_time_s": 1.0,
                    }
                )
    summary = {
        "schema_version": 1,
        "execution_commit": "a" * 40,
        "corpus": {"sha256": "b" * 64},
        "model": {"path": "model.model", "sha256": "c" * 64},
        "inputs": {
            "c60": {"path": "c60.xyz", "sha256": "d" * 64},
            "pdo": {"path": "pdo.xyz", "sha256": "e" * 64},
        },
        "runtime_versions": {"python": "3.12"},
        "cuda": {"available": True, "device_name": "test"},
        "calculator": {"device": "cuda", "default_dtype": "float32"},
        "row_count": 160,
        "strict_protocol": {
            "fmax_eV_per_A": 0.01,
            "maxiter": 400,
            "objective": "true_mace_pes_no_bias_no_softening",
        },
        "arms": [
            {
                "arm_id": arm_id,
                "optimizer": optimizer,
                "safe_history_limit": (
                    10 if arm_id == "safe-lbfgs-total" else None
                ),
            }
            for arm_id, optimizer in arms
        ],
    }
    return rows, summary


def write_replay(tmp_path: Path) -> tuple[Path, Path]:
    rows, summary = replay_payloads()
    rows_path = tmp_path / "rows.json"
    summary_path = tmp_path / "summary.json"
    rows_path.write_text(
        json.dumps(rows, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return rows_path, summary_path


def test_builder_extracts_only_four_expected_ase_lbfgs_certificate_failures(
    tmp_path, monkeypatch
):
    builder = module("sequential_rescue_builder", BUILDER_PATH)
    rows_path, summary_path = write_replay(tmp_path)
    output_path = tmp_path / "corpus.json"
    monkeypatch.setattr(
        builder, "EXPECTED_ROWS_SHA256", sha256(rows_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        builder,
        "EXPECTED_SUMMARY_SHA256",
        sha256(summary_path.read_bytes()).hexdigest(),
    )

    corpus = builder.build_corpus(
        rows_path=rows_path,
        summary_path=summary_path,
        output_path=output_path,
    )

    assert corpus["source"]["rows"]["sha256"] == sha256(
        rows_path.read_bytes()
    ).hexdigest()
    assert corpus["source"]["summary"]["sha256"] == sha256(
        summary_path.read_bytes()
    ).hexdigest()
    assert [
        (entry["system"], entry["task_index"]) for entry in corpus["entries"]
    ] == [("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4)]
    assert corpus["selection"]["primary_cohort"] == {
        "overall": {
            "task_count": 32,
            "certificate_success_count": 28,
            "trigger_count": 4,
            "total_force_evaluations": 3440,
            "total_wall_time_s": 32.0,
        },
        "by_system": {
            "c60": {
                "task_count": 16,
                "certificate_success_count": 14,
                "trigger_count": 2,
                "total_force_evaluations": 1720,
                "total_wall_time_s": 16.0,
            },
            "pdo": {
                "task_count": 16,
                "certificate_success_count": 14,
                "trigger_count": 2,
                "total_force_evaluations": 1720,
                "total_wall_time_s": 16.0,
            },
        },
    }
    for entry in corpus["entries"]:
        assert entry["primary"]["arm_id"] == "ase-lbfgs"
        assert entry["primary"]["certificate_passed"] is False
        assert entry["primary"]["force_evaluations"] > 0
        assert entry["fallback_start"]["positions_sha256"] == position_sha256(
            entry["fallback_start"]["state"]["positions"]
        )
    assert json.loads(output_path.read_text(encoding="utf-8")) == corpus


def test_builder_fails_closed_on_source_hash_matrix_and_failure_set(
    tmp_path, monkeypatch
):
    builder = module("sequential_rescue_builder_fail_closed", BUILDER_PATH)
    rows_path, summary_path = write_replay(tmp_path)
    monkeypatch.setattr(
        builder, "EXPECTED_ROWS_SHA256", sha256(rows_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        builder,
        "EXPECTED_SUMMARY_SHA256",
        sha256(summary_path.read_bytes()).hexdigest(),
    )
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    rows.pop()
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(
        builder, "EXPECTED_ROWS_SHA256", sha256(rows_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="complete 160-row matrix"):
        builder.build_corpus(
            rows_path=rows_path,
            summary_path=summary_path,
            output_path=tmp_path / "bad-matrix.json",
        )

    rows_path, summary_path = write_replay(tmp_path)
    monkeypatch.setattr(builder, "EXPECTED_ROWS_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="rows SHA-256 mismatch"):
        builder.build_corpus(
            rows_path=rows_path,
            summary_path=summary_path,
            output_path=tmp_path / "bad-hash.json",
        )

    rows, _ = replay_payloads()
    target = next(
        row
        for row in rows
        if row["arm_id"] == "ase-lbfgs"
        and row["system"] == "c60"
        and row["task_index"] == 5
    )
    target["final"]["max_active_force_eV_per_A"] = 0.005
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(
        builder, "EXPECTED_ROWS_SHA256", sha256(rows_path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="certificate-failure set"):
        builder.build_corpus(
            rows_path=rows_path,
            summary_path=summary_path,
            output_path=tmp_path / "bad-failures.json",
        )


def rescue_corpus_payload() -> dict:
    entries = []
    for system, task_index in (("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4)):
        state = state_payload(system, task_index, terminal=True)
        entries.append(
            {
                "system": system,
                "task_index": task_index,
                "trial_index": task_index + 1,
                "proposal_index": 1,
                "source_trajectory_path": f"stage_a/{system}/{task_index}.xyz",
                "source_trajectory_sha256": "a" * 64,
                "source_frame_index": 0,
                "primary": {
                    "arm_id": "ase-lbfgs",
                    "optimizer": "ase-lbfgs",
                    "force_evaluations": 401,
                    "wall_time_s": 1.0,
                    "initial_energy_eV": -10.0,
                    "final_energy_eV": -20.0,
                    "initial_max_active_force_eV_per_A": 1.0,
                    "final_max_active_force_eV_per_A": 0.02,
                    "certificate_passed": False,
                    "termination_reason": "maxiter",
                },
                "fallback_start": {
                    "energy_eV": -20.0,
                    "max_active_force_eV_per_A": 0.02,
                    "positions_sha256": position_sha256(state["positions"]),
                    "state": state,
                },
            }
        )
    return {
        "schema_version": 1,
        "selection": {
            "primary_arm_id": "ase-lbfgs",
            "strict_fmax_eV_per_A": 0.01,
            "expected_failure_keys": [
                ["c60", 5],
                ["c60", 11],
                ["pdo", 3],
                ["pdo", 4],
            ],
            "primary_cohort": {
                "overall": {
                    "task_count": 32,
                    "certificate_success_count": 28,
                    "trigger_count": 4,
                    "total_force_evaluations": 5096,
                    "total_wall_time_s": 98.84687816997757,
                },
                "by_system": {
                    "c60": {
                        "task_count": 16,
                        "certificate_success_count": 14,
                        "trigger_count": 2,
                        "total_force_evaluations": 2490,
                        "total_wall_time_s": 42.58624181896448,
                    },
                    "pdo": {
                        "task_count": 16,
                        "certificate_success_count": 14,
                        "trigger_count": 2,
                        "total_force_evaluations": 2606,
                        "total_wall_time_s": 56.26063635101309,
                    },
                },
            },
        },
        "source": {
            "rows": {
                "path": "rows.json",
                "sha256": (
                    "b9c12ca40f2fa183d8573a46b66ec630aeb25ab54ea8eacfc3e88640d163e1ac"
                ),
            },
            "summary": {
                "path": "summary.json",
                "sha256": (
                    "3e15c150ef20ad85e143346a55918aa7510ec3eeadae727168cd36a6e33c9b36"
                ),
            },
            "execution_commit": "d" * 40,
            "raw_landing_corpus_sha256": "e" * 64,
            "model": {
                "path": "model.model",
                "sha256": (
                    "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
                ),
            },
            "inputs": {
                "c60": {"path": "c60.xyz", "sha256": "1" * 64},
                "pdo": {"path": "pdo.xyz", "sha256": "2" * 64},
            },
            "calculator": {"device": "cuda", "default_dtype": "float32"},
            "runtime_versions": {"python": "3.12"},
            "cuda": {"available": True, "device_name": "source-gpu"},
        },
        "entry_count": 4,
        "entries": entries,
    }


def test_runner_protocol_and_preflight_are_pinned_and_fail_closed(
    tmp_path, monkeypatch
):
    runner = module("sequential_rescue_preflight", RUNNER_PATH)
    assert runner.STRICT_FMAX == 0.01
    assert runner.MAXITER == 400
    assert runner.EXPECTED_MODEL_SHA256 == (
        "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
    )
    assert runner.FROZEN_CORPUS_SHA256 == (
        "82b475eb0bcf0a6fb631bff0ee47cdd0223401db0db6262090c6f721d8624e67"
    )
    assert [
        (arm.arm_id, arm.optimizer, arm.safe_history_limit) for arm in runner.ARMS
    ] == [
        ("ase-lbfgs-restart", "ase-lbfgs", None),
        ("safe-lbfgs-total", "safe-lbfgs-total", 10),
        ("ase-fire", "ase-fire", None),
    ]
    corpus_path = tmp_path / "corpus.json"
    model_path = tmp_path / "model.model"
    model_path.write_bytes(b"model")
    corpus_payload = rescue_corpus_payload()
    corpus_payload["source"]["model"]["sha256"] = sha256(b"model").hexdigest()
    corpus_path.write_text(
        json.dumps(corpus_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    production = type(
        "Production",
        (),
        {
            "MODEL_PATH": model_path,
            "CALCULATOR_CONFIG": {"device": "cuda", "default_dtype": "float32"},
        },
    )()
    monkeypatch.setattr(runner, "_production_module", lambda: production)
    monkeypatch.setattr(runner, "_current_commit", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "_is_tracked_file", lambda path: True)
    monkeypatch.setattr(
        runner,
        "FROZEN_CORPUS_SHA256",
        sha256(corpus_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_MODEL_SHA256",
        sha256(model_path.read_bytes()).hexdigest(),
    )
    checked = runner.preflight(
        expected_git_commit="a" * 40,
        corpus_path=corpus_path,
        runtime_probe=lambda: {"python": "3.12"},
        cuda_probe=lambda: {"available": True, "device_name": "fake-gpu"},
    )
    assert checked.metadata["execution_commit"] == "a" * 40
    assert checked.metadata["corpus"]["sha256"] == sha256(
        corpus_path.read_bytes()
    ).hexdigest()
    assert checked.metadata["model"]["sha256"] == sha256(b"model").hexdigest()
    assert checked.metadata["source_replay"]["rows"]["sha256"] == (
        "b9c12ca40f2fa183d8573a46b66ec630aeb25ab54ea8eacfc3e88640d163e1ac"
    )
    assert checked.metadata["primary_cohort"]["overall"][
        "total_force_evaluations"
    ] == 5096

    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: False)
    with pytest.raises(RuntimeError, match="tracked worktree is not clean"):
        runner.preflight(
            expected_git_commit="a" * 40,
            corpus_path=corpus_path,
            runtime_probe=lambda: {},
            cuda_probe=lambda: {"available": True},
        )

    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "FROZEN_CORPUS_SHA256", "9" * 64)
    with pytest.raises(ValueError, match="frozen rescue corpus SHA-256 mismatch"):
        runner.preflight(
            expected_git_commit="a" * 40,
            corpus_path=corpus_path,
            runtime_probe=lambda: {},
            cuda_probe=lambda: {"available": True},
        )


def test_run_builds_twelve_row_matrix_reuses_six_calculators_and_closes_ledger(
    tmp_path,
):
    runner = module("sequential_rescue_run", RUNNER_PATH)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(rescue_corpus_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corpus = runner.load_corpus(corpus_path)
    checked = runner.Preflight(
        metadata={
            "schema_version": 1,
            "execution_commit": "a" * 40,
            "corpus": {"sha256": corpus.source_sha256},
            "source_replay": corpus.provenance["source"],
            "model": {"sha256": "b" * 64},
            "runtime_versions": {"python": "3.12"},
            "cuda": {"available": True, "device_name": "analytic"},
            "calculator": {"device": "cuda"},
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

    assert len(calculators) == 6
    assert summary["row_count"] == 12
    assert summary["calculator_instances"] == 6
    rows = json.loads((output / "rows.json").read_text(encoding="utf-8"))
    assert len(rows) == 12
    assert not output.with_name("output.partial").exists()
    assert {
        (row["system"], row["task_index"], row["arm_id"]) for row in rows
    } == {
        (system, task_index, arm.arm_id)
        for system, task_index in (("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4))
        for arm in runner.ARMS
    }
    for row in rows:
        source = next(
            entry
            for entry in rescue_corpus_payload()["entries"]
            if entry["system"] == row["system"]
            and entry["task_index"] == row["task_index"]
        )
        assert row["fallback"]["initial"]["positions_sha256"] == source[
            "fallback_start"
        ]["positions_sha256"]
        assert row["primary"]["force_evaluations"] == 401
        assert row["cost"]["offline_fallback_force_evaluations"] == row[
            "fallback"
        ]["force_evaluations"]
        assert row["cost"]["offline_combined_replay_force_evaluations"] == (
            401 + row["fallback"]["force_evaluations"]
        )
        assert row["cost"]["offline_repeats_primary_terminal_evaluation"] is True
        assert row["cost"]["continuous_implementation_avoidable_force_evaluations"] == 1
        assert (
            row["fallback"]["purpose_count_delta"]["landing_true_quench"]
            == row["fallback"]["force_evaluations"]
        )
        assert row["fallback"]["purpose_count_delta"]["unattributed"] == 0
    by_arm = summary["pipeline_cost_by_fallback_arm"]
    assert set(by_arm) == {arm.arm_id for arm in runner.ARMS}
    for arm in runner.ARMS:
        arm_rows = [row for row in rows if row["arm_id"] == arm.arm_id]
        fallback_fe = sum(
            row["fallback"]["force_evaluations"] for row in arm_rows
        )
        successes = sum(
            bool(row["fallback"]["certificate_passed"]) for row in arm_rows
        )
        assert by_arm[arm.arm_id] == {
            "trigger_count": 4,
            "fallback_certificate_success_count": successes,
            "offline_fallback_force_evaluations": fallback_fe,
            "offline_pipeline_total_force_evaluations": 5096 + fallback_fe,
            "continuous_pipeline_projected_force_evaluations": (
                5096 + fallback_fe - 4
            ),
        }


def test_load_corpus_rejects_primary_cohort_inner_outer_mismatch(tmp_path):
    runner = module("sequential_rescue_cohort_mismatch", RUNNER_PATH)
    payload = rescue_corpus_payload()
    payload["selection"]["primary_cohort"]["overall"][
        "total_force_evaluations"
    ] = 1606
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="primary cohort"):
        runner.load_corpus(corpus_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["entries"][0]["primary"].__setitem__(
                "final_energy_eV", float("nan")
            ),
            "primary numeric",
        ),
        (
            lambda payload: payload["entries"][0]["primary"].__setitem__(
                "wall_time_s", -1.0
            ),
            "primary numeric",
        ),
        (
            lambda payload: payload["entries"][0]["fallback_start"]["state"][
                "positions"
            ][0].__setitem__(0, float("nan")),
            "fallback state",
        ),
        (
            lambda payload: payload["entries"][2]["fallback_start"]["state"][
                "cell"
            ][0].__setitem__(0, float("nan")),
            "fallback state",
        ),
        (
            lambda payload: payload["entries"][0].__setitem__("task_index", True),
            "integer identity",
        ),
        (
            lambda payload: payload["entries"][0].__setitem__("system", "unknown"),
            "unknown system",
        ),
    ],
)
def test_load_corpus_rejects_nonfinite_bool_integer_and_unknown_system(
    tmp_path, mutate, message
):
    runner = module(f"sequential_rescue_invalid_{message}", RUNNER_PATH)
    payload = rescue_corpus_payload()
    mutate(payload)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        runner.load_corpus(corpus_path)


def test_complete_matrix_rejects_nonfinite_and_inconsistent_certificates(tmp_path):
    runner = module("sequential_rescue_matrix_mutations", RUNNER_PATH)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(rescue_corpus_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corpus = runner.load_corpus(corpus_path)
    checked = runner.Preflight(
        metadata={"execution_commit": "a" * 40},
        corpus=corpus,
    )

    class Quadratic:
        def evaluate_flat(self, flat, template):
            vector = np.asarray(flat, dtype=float)
            return 0.5 * float(vector @ vector), vector.copy()

    output = tmp_path / "output"
    runner.run(
        output_dir=output,
        expected_git_commit="a" * 40,
        preflight_fn=lambda **_: checked,
        calculator_factory=Quadratic,
        counter_factory=EvalCounter,
    )
    valid_rows = json.loads((output / "rows.json").read_text(encoding="utf-8"))

    mutations = (
        ("certificate", lambda row: row["fallback"].__setitem__("certificate_passed", 1)),
        (
            "telemetry convergence",
            lambda row: row["fallback"]["telemetry"].__setitem__(
                "converged", not row["fallback"]["certificate_passed"]
            ),
        ),
        (
            "termination",
            lambda row: row["fallback"].__setitem__(
                "termination_reason",
                "maxiter"
                if row["fallback"]["certificate_passed"]
                else "converged",
            ),
        ),
        (
            "telemetry termination",
            lambda row: row["fallback"]["telemetry"].__setitem__(
                "termination_reason", "inconsistent"
            ),
        ),
        (
            "finite numeric",
            lambda row: row["fallback"]["final"].__setitem__(
                "energy_eV", float("nan")
            ),
        ),
        (
            "finite numeric",
            lambda row: row["fallback"].__setitem__("wall_time_s", -1.0),
        ),
    )
    for message, mutate in mutations:
        rows = json.loads(json.dumps(valid_rows))
        mutate(rows[0])
        with pytest.raises(RuntimeError, match=message):
            runner._validate_complete_matrix(rows, corpus)


def test_run_is_all_or_nothing_when_a_fallback_fails(tmp_path):
    runner = module("sequential_rescue_atomic", RUNNER_PATH)
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(rescue_corpus_payload(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    corpus = runner.load_corpus(corpus_path)
    checked = runner.Preflight(
        metadata={"execution_commit": "a" * 40},
        corpus=corpus,
    )

    class Broken:
        def evaluate_flat(self, flat, template):
            raise RuntimeError("synthetic failure")

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="synthetic failure"):
        runner.run(
            output_dir=output,
            expected_git_commit="a" * 40,
            preflight_fn=lambda **_: checked,
            calculator_factory=Broken,
            counter_factory=EvalCounter,
        )
    assert not output.exists()
    assert output.with_name("output.partial").exists()
