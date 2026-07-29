"""Contract tests for the focused C60/PdO 200-trial production runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.result import WalkRecord
from pamssw.state import State


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-safe-lbfgs-200-production"
    / "run_production.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "safe_lbfgs_200_production", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state(*, fixed: bool = False) -> State:
    return State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
        fixed_mask=np.array([True, False]) if fixed else None,
    )


def _result():
    initial = SimpleNamespace(entry_id=0, energy=-10.0, state=_state())
    best = SimpleNamespace(entry_id=1, energy=-11.5, state=_state())
    archive = SimpleNamespace(entries=[initial, best])
    return SimpleNamespace(
        best_state=best.state,
        best_energy=best.energy,
        archive=archive,
        walk_history=[
            WalkRecord(
                seed_entry_id=0,
                discovered_entry_id=1,
                energy=-11.5,
                accepted_new_basin=True,
            )
        ],
        stats={"n_trials": 200, "force_evaluations": 17, "n_minima": 2},
    )


def _preflight(system: str):
    return {
        "schema_version": 1,
        "execution_commit": "e" * 40,
        "system": system,
        "input_path": f"/input/{system}.xyz",
        "input_sha256": "a" * 64,
        "model_path": "/model/mace.model",
        "model_sha256": "b" * 64,
        "runtime_versions": {
            "python": "3.test",
            "numpy": "test",
            "scipy": "test",
            "ase": "test",
            "torch": "test",
            "mace": "test",
        },
        "cuda": {
            "available": True,
            "runtime_version": "12.test",
            "device_name": "test GPU",
        },
        "calculator": {
            "device": "cuda",
            "default_dtype": "float32",
            "inference_precision": "float32",
            "enable_cueq": False,
        },
        "safe_lbfgs_default_history_limit": 10,
    }


def test_frozen_configs_change_only_the_requested_production_controls(tmp_path):
    runner = load_runner()
    c60 = runner.build_config("c60", tmp_path / "c60")
    pdo = runner.build_config("pdo", tmp_path / "pdo")

    for config in (c60, pdo):
        assert config.max_trials == 200
        assert config.max_force_evals is None
        assert config.rng_seed == 42
        assert config.max_steps_per_walk == 8
        assert config.proposal_optimizer == "safe-lbfgs-total"
        assert config.quench_optimizer == "scipy-lbfgsb"
        assert config.target_uphill_energy == pytest.approx(0.8)
        assert config.local_softening_strength == pytest.approx(0.15)
        assert config.local_softening_penalty == "buckingham_repulsive"
        assert config.local_softening_xi == pytest.approx(0.3)

    assert c60.oracle_candidates == 12
    assert c60.proposal_relax_steps == 80
    assert c60.quench_fmax == pytest.approx(0.01)
    assert c60.target_step_rms == pytest.approx(0.08)
    assert c60.max_step_rms == pytest.approx(0.15)
    assert c60.local_softening_cutoff_scale == pytest.approx(1.3)
    assert c60.local_softening_active_count == 3

    assert pdo.oracle_candidates == 8
    assert pdo.proposal_relax_steps == 300
    assert pdo.quench_fmax == pytest.approx(0.03)
    assert pdo.target_step_rms == pytest.approx(0.15)
    assert pdo.max_step_rms == pytest.approx(0.35)
    assert pdo.local_softening_cutoff_scale == pytest.approx(1.15)
    assert pdo.local_softening_active_count == 5
    assert runner.SAFE_LBFGS_DEFAULT_HISTORY_LIMIT == 10


def test_real_inputs_are_the_frozen_c60_and_pdo_states():
    runner = load_runner()
    c60 = runner.load_state("c60")
    pdo = runner.load_state("pdo")

    assert len(c60.numbers) == 60
    assert int(np.count_nonzero(c60.fixed_mask)) == 0
    assert c60.pbc == (False, False, False)
    assert Path(c60.metadata["input"]).name == "prerelaxed_c60.xyz"

    assert len(pdo.numbers) == 115
    assert pdo.pbc == (True, True, False)
    assert int(np.count_nonzero(pdo.fixed_mask)) == 40
    assert pdo.metadata["fixed_bottom_fraction"] == pytest.approx(0.35)


def test_preflight_records_hashes_runtime_gpu_and_default_history(
    tmp_path, monkeypatch
):
    runner = load_runner()
    model = tmp_path / "model"
    structure = tmp_path / "input.xyz"
    model.write_bytes(b"model")
    structure.write_bytes(b"structure")
    monkeypatch.setattr(runner, "MODEL_PATH", model)
    monkeypatch.setitem(runner.INPUT_PATHS, "c60", structure)
    monkeypatch.setattr(runner, "_current_commit", lambda: "e" * 40)
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    monkeypatch.setattr(runner, "_safe_lbfgs_default_history_limit", lambda: 10)

    checked = runner.preflight(
        system="c60",
        expected_git_commit="e" * 40,
        runtime_probe=lambda: _preflight("c60")["runtime_versions"],
        cuda_probe=lambda: _preflight("c60")["cuda"],
    )
    assert checked["execution_commit"] == "e" * 40
    assert checked["input_sha256"] != checked["model_sha256"]
    assert checked["cuda"]["available"] is True
    assert checked["calculator"]["enable_cueq"] is False
    assert checked["safe_lbfgs_default_history_limit"] == 10

    monkeypatch.setattr(runner, "_safe_lbfgs_default_history_limit", lambda: 1)
    with pytest.raises(RuntimeError, match="history"):
        runner.preflight(
            system="c60",
            expected_git_commit="e" * 40,
            runtime_probe=lambda: _preflight("c60")["runtime_versions"],
            cuda_probe=lambda: _preflight("c60")["cuda"],
        )


def test_run_closes_purpose_accounting_and_preserves_existing_trial_records(
    tmp_path, monkeypatch
):
    runner = load_runner()
    result = _result()
    counts = EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH: 3,
            EvaluationPurpose.DIRECTION_ORACLE: 4,
            EvaluationPurpose.BIASED_PROPOSAL_RELAX: 7,
            EvaluationPurpose.LANDING_TRUE_QUENCH: 3,
        }
    )

    class FakeWalker:
        def __init__(self, *, calculator, config, softening_enabled):
            assert softening_enabled is True
            assert config.max_trials == 200
            self.calculator = SimpleNamespace(snapshot=lambda: counts)

        def run(self, state):
            return result

        def relaxation_diagnostics(self):
            return {
                "proposal_optimizer": "safe-lbfgs-total",
                "quench_optimizer": "scipy-lbfgsb",
                "proposal_relax_accepted_secants": 5,
            }

    writes: list[str] = []
    monkeypatch.setattr(runner, "preflight", lambda **_: _preflight("c60"))
    monkeypatch.setattr(runner, "load_state", lambda _: _state())
    monkeypatch.setattr(runner, "SurfaceWalker", FakeWalker)
    monkeypatch.setattr(runner, "ASECalculator", lambda calculator: calculator)
    monkeypatch.setattr(runner, "_calculator", lambda: object())
    monkeypatch.setattr(
        runner,
        "_write_json",
        lambda path, payload: (
            writes.append(path.name),
            path.write_text(json.dumps(payload), encoding="utf-8"),
        ),
    )

    summary = runner.run(
        system="c60",
        output_dir=tmp_path / "production",
        expected_git_commit="e" * 40,
    )
    assert summary["effective_config"]["max_trials"] == 200
    assert summary["effective_config"]["max_force_evals"] is None
    assert summary["purpose_counts"] == counts.as_dict()
    assert summary["force_evaluations"] == 17
    assert summary["optimizer_telemetry"]["proposal_relax_accepted_secants"] == 5
    assert summary["timing"]["per_trial_wall_time_s"] is None
    assert "not exposed" in summary["timing"]["per_trial_wall_time_reason"]
    assert summary["walk_records"] == [
        {
            "trial": 1,
            "seed_entry_id": 0,
            "discovered_entry_id": 1,
            "energy_eV": -11.5,
            "accepted_new_basin": True,
        }
    ]
    assert writes[-1] == "summary.json"


def test_run_refuses_existing_output_before_calculator(tmp_path, monkeypatch):
    runner = load_runner()
    output = tmp_path / "production"
    output.mkdir()
    calculator_calls = 0

    def calculator():
        nonlocal calculator_calls
        calculator_calls += 1
        return object()

    monkeypatch.setattr(runner, "preflight", lambda **_: _preflight("pdo"))
    monkeypatch.setattr(runner, "_calculator", calculator)
    with pytest.raises(FileExistsError):
        runner.run(
            system="pdo",
            output_dir=output,
            expected_git_commit="e" * 40,
        )
    assert calculator_calls == 0


def test_preflight_only_does_not_create_output_or_calculator(tmp_path, monkeypatch):
    runner = load_runner()
    output = tmp_path / "production"
    calculator_calls = 0

    def calculator():
        nonlocal calculator_calls
        calculator_calls += 1
        return object()

    monkeypatch.setattr(runner, "preflight", lambda **_: _preflight("pdo"))
    monkeypatch.setattr(runner, "_calculator", calculator)
    checked = runner.run(
        system="pdo",
        output_dir=output,
        expected_git_commit="e" * 40,
        preflight_only=True,
    )
    assert checked == _preflight("pdo")
    assert not output.exists()
    assert calculator_calls == 0


def test_cli_requires_system_output_and_execution_commit():
    runner = load_runner()
    with pytest.raises(SystemExit):
        runner._parse_args([])
    args = runner._parse_args(
        [
            "--system",
            "c60",
            "--output",
            "run-output",
            "--expected-git-commit",
            "e" * 40,
            "--preflight-only",
        ]
    )
    assert args.system == "c60"
    assert args.preflight_only is True
