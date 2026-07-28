"""Contracts for the paired fixed-total-FE native direction-budget runner."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pamssw import LSSSWConfig


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-direction-candidate-budget-gpu-ablation"
    / "run_ablation.py"
)


def load_runner(name: str = "direction_candidate_budget_gpu_ablation"):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FakeBaseRunner:
    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, object]] = []
        self.build_config_calls: list[tuple[str, Path]] = []

    def build_config(self, system: str, case_dir: Path) -> LSSSWConfig:
        self.build_config_calls.append((system, Path(case_dir)))
        return LSSSWConfig(
            max_trials=200,
            max_force_evals=None,
            rng_seed=42,
            oracle_candidates=4,
            proposal_optimizer="safe-lbfgs-total",
            proposal_fmax=0.05,
            local_softening_mode="active_neighbors",
            direction_probe_enabled=False,
            direction_synthesis_mode="none",
            direction_selection_mode="discrete",
            quench_optimizer="scipy-lbfgsb",
            quench_fallback_optimizer=None,
            quench_fmax=0.01 if system == "c60" else 0.03,
            quench_maxiter=400,
            direction_diagnostics_enabled=True,
            direction_diagnostics_path=str(Path(case_dir) / "direction_trace.jsonl"),
        )

    def preflight(self, *, system: str, expected_git_commit: str):
        self.preflight_calls.append(
            {"system": system, "expected_git_commit": expected_git_commit}
        )
        return {
            "execution_commit": expected_git_commit,
            "input_path": f"/input/{system}.xyz",
            "input_sha256": "a" * 64,
            "model_path": "/model/mace.model",
            "model_sha256": "b" * 64,
            "runtime_versions": {"python": "test"},
            "cuda": {"available": True, "device_name": "fake"},
            "calculator": {"device": "cuda"},
        }

    @staticmethod
    def load_state(system: str):
        return SimpleNamespace(numbers=[6], fixed_mask=None, pbc=(False, False, False))

    @staticmethod
    def _calculator():
        return object()


class FakeCounts:
    def __init__(self, total: int, unattributed: int = 0) -> None:
        self.total = total
        self.unattributed = unattributed

    def as_dict(self) -> dict[str, int]:
        return {
            "bootstrap_true_quench": 2,
            "direction_oracle": 12,
            "biased_proposal_relax": self.total - 14 - self.unattributed,
            "landing_true_quench": 0,
            "unattributed": self.unattributed,
        }


class FakeCounter:
    def __init__(self, total: int, unattributed: int = 0) -> None:
        self._counts = FakeCounts(total, unattributed)

    def snapshot(self) -> FakeCounts:
        return self._counts


class FakeWalker:
    force_total = 20
    unattributed = 0

    def __init__(self, *, calculator, config, softening_enabled: bool) -> None:
        self.input_calculator = calculator
        self.config = config
        self.softening_enabled = softening_enabled
        self.calculator = FakeCounter(self.force_total, self.unattributed)

    def run(self, state):
        direction_path = Path(self.config.direction_diagnostics_path)
        direction_path.parent.mkdir(parents=True, exist_ok=True)
        direction_path.write_text(
            "\n".join(
                json.dumps({"trial": 0, "step": step, "candidate_count": 3})
                for step in range(2)
            )
            + "\n",
            encoding="utf-8",
        )
        record = SimpleNamespace(
            seed_entry_id=0,
            discovered_entry_id=1,
            energy=-1.0,
            accepted_new_basin=True,
        )
        return SimpleNamespace(
            archive=SimpleNamespace(entries=[SimpleNamespace(energy=0.0)]),
            best_energy=-1.0,
            best_state=state,
            walk_history=[record, record, record],
            stats={
                "n_trials": 3,
                "force_evaluations": self.force_total,
                "budget_exhausted": 1,
                "direction_choices": 2,
                "direction_candidate_evaluations": 6,
                "fragment_rejections": 1,
            },
        )

    @staticmethod
    def relaxation_diagnostics():
        return {"proposal_relax_count": 2}


def fake_runtime(base: FakeBaseRunner, *, code_root: Path, walker=FakeWalker):
    return SimpleNamespace(
        code_root=code_root,
        base_runner=base,
        frozen_runner_path=code_root / "runs" / "frozen.py",
        calculator_wrapper=lambda calculator: calculator,
        walker_class=walker,
        write_state=lambda path, _state: Path(path).write_text("state\n", encoding="utf-8"),
    )


def checked_run(
    runner,
    runtime,
    code_root: Path,
    *,
    arm: str = "precap",
    system: str = "c60",
    total_force_budget: int = 20,
    output_dir: Path,
):
    return runner.preflight(
        arm=arm,
        system=system,
        seed=42,
        total_force_budget=total_force_budget,
        code_root=code_root,
        expected_code_commit=runner.ARM_COMMITS[arm],
        output_dir=output_dir,
        target_loader=lambda root: runtime,
        git_head=lambda root: runner.ARM_COMMITS[arm],
        tracked_clean=lambda root: True,
        arm_contract_checker=lambda target, checked_arm: {"arm": checked_arm},
    )


def test_projection_preserves_frozen_direction_protocol_and_only_applies_budget_seed_and_quench(
    tmp_path,
):
    runner = load_runner("direction_budget_projection")
    base = FakeBaseRunner()

    source, effective, diff, overrides = runner.config_projection(
        system="c60",
        case_dir=tmp_path / "case",
        total_force_budget=6000,
        seed=77,
        base_runner=base,
    )

    assert source["oracle_candidates"] == effective["oracle_candidates"] == 4
    assert source["proposal_optimizer"] == effective["proposal_optimizer"] == "safe-lbfgs-total"
    assert source["proposal_fmax"] == effective["proposal_fmax"] == pytest.approx(0.05)
    assert source["direction_probe_enabled"] is effective["direction_probe_enabled"] is False
    assert source["direction_synthesis_mode"] == effective["direction_synthesis_mode"] == "none"
    assert source["direction_selection_mode"] == effective["direction_selection_mode"] == "discrete"
    assert effective["max_trials"] == 200
    assert effective["max_force_evals"] == 6000
    assert effective["rng_seed"] == 77
    assert effective["quench_optimizer"] == "ase-lbfgs"
    assert effective["quench_fallback_optimizer"] == "ase-fire"
    assert effective["quench_fmax"] == pytest.approx(0.01)
    assert effective["quench_maxiter"] == 400
    assert diff == {
        "max_force_evals": [None, 6000],
        "quench_fallback_optimizer": [None, "ase-fire"],
        "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
        "rng_seed": [42, 77],
    }
    assert overrides == {
        "max_trials": 200,
        "max_force_evals": 6000,
        "rng_seed": 77,
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
    }


def test_preflight_pins_arm_to_its_exact_clean_target_checkout_and_checks_native_contract(
    tmp_path,
):
    runner = load_runner("direction_budget_preflight")
    base = FakeBaseRunner()
    code_root = tmp_path / "hardcap"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root=code_root)
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    calls: list[tuple[Path, str]] = []

    checked = runner.preflight(
        arm="hardcap",
        system="pdo",
        seed=42,
        total_force_budget=6000,
        code_root=code_root,
        expected_code_commit=runner.ARM_COMMITS["hardcap"],
        target_loader=lambda root: runtime,
        git_head=lambda root: runner.ARM_COMMITS["hardcap"],
        tracked_clean=lambda root: True,
        arm_contract_checker=lambda target, arm: calls.append((target.code_root, arm))
        or {"arm": arm, "native_contract": "checked"},
    )

    assert base.preflight_calls == [
        {"system": "pdo", "expected_git_commit": runner.ARM_COMMITS["hardcap"]}
    ]
    assert calls == [(code_root.resolve(), "hardcap")]
    assert checked["target"]["execution_commit"] == runner.ARM_COMMITS["hardcap"]
    assert checked["arm"] == "hardcap"
    assert checked["base_preflight"]["cuda"]["available"] is True
    assert checked["target"]["frozen_runner_sha256"] == runner._sha256(
        runtime.frozen_runner_path
    )

    with pytest.raises(RuntimeError, match="must use exact commit"):
        runner.preflight(
            arm="hardcap",
            system="pdo",
            seed=42,
            total_force_budget=6000,
            code_root=code_root,
            expected_code_commit="0" * 40,
            target_loader=lambda root: runtime,
            git_head=lambda root: runner.ARM_COMMITS["hardcap"],
            tracked_clean=lambda root: True,
            arm_contract_checker=lambda target, arm: {},
        )

    with pytest.raises(RuntimeError, match="not clean"):
        runner.preflight(
            arm="hardcap",
            system="pdo",
            seed=42,
            total_force_budget=6000,
            code_root=code_root,
            expected_code_commit=runner.ARM_COMMITS["hardcap"],
            target_loader=lambda root: runtime,
            git_head=lambda root: runner.ARM_COMMITS["hardcap"],
            tracked_clean=lambda root: False,
            arm_contract_checker=lambda target, arm: {},
        )


def test_run_emits_exact_budget_closed_direction_oracle_artifacts(tmp_path):
    runner = load_runner("direction_budget_run")
    base = FakeBaseRunner()
    code_root = tmp_path / "precap"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root=code_root)
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "output"

    checked = runner.preflight(
        arm="precap",
        system="c60",
        seed=42,
        total_force_budget=20,
        code_root=code_root,
        expected_code_commit=runner.ARM_COMMITS["precap"],
        output_dir=output,
        target_loader=lambda root: runtime,
        git_head=lambda root: runner.ARM_COMMITS["precap"],
        tracked_clean=lambda root: True,
        arm_contract_checker=lambda target, arm: {"arm": arm},
    )
    summary = runner.run(
        arm="precap",
        system="c60",
        seed=42,
        total_force_budget=20,
        code_root=code_root,
        output_dir=output,
        expected_code_commit=runner.ARM_COMMITS["precap"],
        target_loader=lambda root: runtime,
        preflight_fn=lambda **_: checked,
    )

    assert summary["force_evaluations"] == 20
    assert summary["stats"]["budget_exhausted"] == 1
    assert sum(summary["purpose_counts"].values()) == 20
    assert summary["purpose_counts"]["unattributed"] == 0
    assert summary["record_counts"] == {
        "n_trials": 3,
        "walk_records": 3,
        "unlogged_walk_trials": 0,
        "direction_records": 2,
        "unlogged_direction_choices": 0,
    }
    assert summary["direction_oracle_audit"] == {
        "candidate_count_sum": 6,
        "candidate_count_max": 3,
        "candidate_count_over_oracle_candidates": 0,
        "expected_force_evaluations": 12,
        "recorded_force_evaluations": 12,
    }
    assert summary["fragment_rejections"] == 1
    for name in runner.EXPECTED_OUTPUT_FILES:
        assert (output / name).is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


def test_run_fails_closed_when_direction_oracle_ledger_does_not_match_trace(tmp_path):
    runner = load_runner("direction_budget_fail_closed")
    base = FakeBaseRunner()
    code_root = tmp_path / "hardcap"
    code_root.mkdir()

    class UnclosedWalker(FakeWalker):
        force_total = 20

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.calculator = FakeCounter(20, unattributed=1)

    runtime = fake_runtime(base, code_root=code_root, walker=UnclosedWalker)
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    checked = runner.preflight(
        arm="hardcap",
        system="c60",
        seed=42,
        total_force_budget=20,
        code_root=code_root,
        expected_code_commit=runner.ARM_COMMITS["hardcap"],
        output_dir=tmp_path / "output",
        target_loader=lambda root: runtime,
        git_head=lambda root: runner.ARM_COMMITS["hardcap"],
        tracked_clean=lambda root: True,
        arm_contract_checker=lambda target, arm: {"arm": arm},
    )

    with pytest.raises(RuntimeError, match="unattributed"):
        runner.run(
            arm="hardcap",
            system="c60",
            seed=42,
            total_force_budget=20,
            code_root=code_root,
            output_dir=tmp_path / "output",
            expected_code_commit=runner.ARM_COMMITS["hardcap"],
            target_loader=lambda root: runtime,
            preflight_fn=lambda **_: checked,
        )
    assert not (tmp_path / "output" / "summary.json").exists()


def test_preflight_only_never_creates_an_output(tmp_path):
    runner = load_runner("direction_budget_preflight_only")
    base = FakeBaseRunner()
    code_root = tmp_path / "hardcap"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root=code_root)
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "output"

    checked = runner.run(
        arm="hardcap",
        system="pdo",
        seed=42,
        total_force_budget=6000,
        code_root=code_root,
        output_dir=output,
        expected_code_commit=runner.ARM_COMMITS["hardcap"],
        preflight_only=True,
        target_loader=lambda root: runtime,
        preflight_fn=lambda **kwargs: runner.preflight(
            **kwargs,
            target_loader=lambda root: runtime,
            git_head=lambda root: runner.ARM_COMMITS["hardcap"],
            tracked_clean=lambda root: True,
            arm_contract_checker=lambda target, arm: {"arm": arm},
        ),
    )

    assert checked["arm"] == "hardcap"
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("direction_curvature_source", "true"),
        ("choice_aligned_softening_enabled", True),
        ("plateau_evolution_enabled", True),
        ("archive_escape_momentum_enabled", True),
    ],
)
def test_projection_rejects_direction_oracle_accounting_gate_drift(tmp_path, field, value):
    runner = load_runner(f"direction_budget_gate_{field}")

    class GateDriftBaseRunner(FakeBaseRunner):
        def build_config(self, system: str, case_dir: Path) -> LSSSWConfig:
            return replace(super().build_config(system, case_dir), **{field: value})

    with pytest.raises(RuntimeError, match="frozen production direction protocol drifted"):
        runner.config_projection(
            system="c60",
            case_dir=tmp_path / "case",
            total_force_budget=6000,
            seed=42,
            base_runner=GateDriftBaseRunner(),
        )


def test_run_never_publishes_summary_before_all_non_summary_artifacts_exist(tmp_path):
    runner = load_runner("direction_budget_missing_best")
    base = FakeBaseRunner()
    code_root = tmp_path / "precap"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root=code_root)
    runtime.write_state = lambda _path, _state: None
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "output"
    checked = checked_run(runner, runtime, code_root, output_dir=output)

    with pytest.raises(RuntimeError, match="best_minimum.xyz"):
        runner.run(
            arm="precap",
            system="c60",
            seed=42,
            total_force_budget=20,
            code_root=code_root,
            output_dir=output,
            expected_code_commit=runner.ARM_COMMITS["precap"],
            target_loader=lambda root: runtime,
            preflight_fn=lambda **_: checked,
        )

    assert not (output / "summary.json").exists()


class FractionalForceTotalWalker(FakeWalker):
    force_total = 20.5


class StringBudgetExhaustedWalker(FakeWalker):
    def run(self, state):
        result = super().run(state)
        result.stats["budget_exhausted"] = "1"
        return result


class BoolBudgetExhaustedWalker(FakeWalker):
    def run(self, state):
        result = super().run(state)
        result.stats["budget_exhausted"] = True
        return result


class NegativePurposeCounts(FakeCounts):
    def as_dict(self) -> dict[str, int]:
        return {
            "bootstrap_true_quench": 2,
            "direction_oracle": 12,
            "biased_proposal_relax": -1,
            "landing_true_quench": 7,
            "unattributed": 0,
        }


class NegativePurposeWalker(FakeWalker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.calculator = SimpleNamespace(snapshot=lambda: NegativePurposeCounts(20))


class NaNEnergyWalker(FakeWalker):
    def run(self, state):
        result = super().run(state)
        result.walk_history[0].energy = float("nan")
        return result


@pytest.mark.parametrize(
    ("walker", "message"),
    [
        (FractionalForceTotalWalker, "force_evaluations"),
        (StringBudgetExhaustedWalker, "budget_exhausted"),
        (BoolBudgetExhaustedWalker, "budget_exhausted"),
        (NegativePurposeWalker, "purpose"),
    ],
)
def test_run_rejects_non_exact_or_negative_ledger_values_before_summary(
    tmp_path, walker, message
):
    runner = load_runner(f"direction_budget_bad_ledger_{walker.__name__}")
    base = FakeBaseRunner()
    code_root = tmp_path / "precap"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root=code_root, walker=walker)
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "output"
    checked = checked_run(runner, runtime, code_root, output_dir=output)

    with pytest.raises(ValueError, match=message):
        runner.run(
            arm="precap",
            system="c60",
            seed=42,
            total_force_budget=20,
            code_root=code_root,
            output_dir=output,
            expected_code_commit=runner.ARM_COMMITS["precap"],
            target_loader=lambda root: runtime,
            preflight_fn=lambda **_: checked,
        )

    assert not (output / "summary.json").exists()


def test_run_rejects_nonfinite_energy_before_writing_final_artifacts(tmp_path):
    runner = load_runner("direction_budget_nan_energy")
    base = FakeBaseRunner()
    code_root = tmp_path / "precap"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root=code_root, walker=NaNEnergyWalker)
    runtime.frozen_runner_path.parent.mkdir(parents=True)
    runtime.frozen_runner_path.write_text("frozen\n", encoding="utf-8")
    output = tmp_path / "output"
    checked = checked_run(runner, runtime, code_root, output_dir=output)

    with pytest.raises(ValueError, match="energy"):
        runner.run(
            arm="precap",
            system="c60",
            seed=42,
            total_force_budget=20,
            code_root=code_root,
            output_dir=output,
            expected_code_commit=runner.ARM_COMMITS["precap"],
            target_loader=lambda root: runtime,
            preflight_fn=lambda **_: checked,
        )

    assert not (output / "best_minimum.xyz").exists()
    assert not (output / "summary.json").exists()


def test_cli_requires_explicit_code_root_and_fixed_total_budget():
    runner = load_runner("direction_budget_cli")
    with pytest.raises(SystemExit):
        runner._parse_args([])
    args = runner._parse_args(
        [
            "--arm",
            "hardcap",
            "--system",
            "c60",
            "--seed",
            "42",
            "--code-root",
            "/clean/root",
            "--output",
            "output",
            "--expected-code-commit",
            runner.ARM_COMMITS["hardcap"],
            "--total-force-budget",
            "6000",
            "--preflight-only",
        ]
    )
    assert args.total_force_budget == 6000
    assert args.preflight_only is True
