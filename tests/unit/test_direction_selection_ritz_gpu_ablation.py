"""Contracts for the minimal discrete-versus-plain-Ritz GPU ablation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pamssw import LSSSWConfig


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-direction-selection-ritz-gpu-ablation"
)
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
ANALYZER_PATH = RUN_ROOT / "analyze_evidence.py"


def test_runner_source_exists_before_contracts_are_exercised():
    assert RUNNER_PATH.is_file()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_runner(name: str = "direction_selection_ritz_ablation"):
    return _load(RUNNER_PATH, name)


class FakeBaseRunner:
    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, object]] = []

    def build_config(self, system: str, case_dir: Path) -> LSSSWConfig:
        return LSSSWConfig(
            max_trials=200,
            max_force_evals=None,
            rng_seed=42,
            oracle_candidates=12 if system == "c60" else 8,
            proposal_optimizer="safe-lbfgs-total",
            proposal_fmax=0.05,
            target_uphill_energy=0.8,
            quench_optimizer="scipy-lbfgsb",
            quench_fallback_optimizer=None,
            quench_fmax=0.01 if system == "c60" else 0.03,
            quench_maxiter=400,
            local_softening_mode="active_neighbors",
            local_softening_strength=0.15,
            local_softening_penalty="buckingham_repulsive",
            local_softening_xi=0.3,
            direction_curvature_source="inner",
            direction_selection_mode="discrete",
            direction_synthesis_mode="none",
            direction_type_ucb_enabled=False,
            direction_archive_enabled=False,
            direction_probe_enabled=False,
            plateau_evolution_enabled=False,
            archive_escape_momentum_enabled=False,
            direction_diagnostics_enabled=True,
            direction_diagnostics_path=str(Path(case_dir) / "direction_trace.jsonl"),
            accepted_structures_log=str(Path(case_dir) / "accepted_structures.jsonl"),
            accepted_structures_dir=str(Path(case_dir) / "accepted_minima"),
        )

    def preflight(self, *, system: str, expected_git_commit: str):
        self.preflight_calls.append(
            {"system": system, "expected_git_commit": expected_git_commit}
        )
        return {
            "execution_commit": expected_git_commit,
            "input_path": f"/inputs/{system}.xyz",
            "input_sha256": "a" * 64,
            "model_path": "/models/mace.model",
            "model_sha256": "b" * 64,
            "runtime_versions": {"python": "test"},
            "cuda": {"available": True, "device_name": "fake-cuda"},
            "calculator": {"device": "cuda"},
        }

    @staticmethod
    def load_state(system: str):
        return SimpleNamespace(numbers=[6], fixed_mask=None, pbc=(False, False, False))

    @staticmethod
    def _calculator():
        return object()


class FakeCounts:
    def __init__(self, total: int) -> None:
        self.total = total

    def as_dict(self) -> dict[str, int]:
        return {
            "bootstrap_true_quench": 2,
            "direction_oracle": 6,
            "biased_proposal_relax": self.total - 8,
            "landing_true_quench": 0,
            "unattributed": 0,
        }


class FakeWalker:
    def __init__(self, *, calculator, config, softening_enabled: bool) -> None:
        assert softening_enabled is True
        self.config = config
        self.calculator = SimpleNamespace(snapshot=lambda: FakeCounts(20))

    def run(self, state):
        direction_path = Path(self.config.direction_diagnostics_path)
        direction_path.parent.mkdir(parents=True, exist_ok=True)
        # Ritz adds a zero-HVP synthetic selection candidate to this legacy field.
        candidate_count = 4 if self.config.direction_selection_mode == "rayleigh_ritz" else 3
        direction_path.write_text(
            json.dumps(
                {
                    "trial": 1,
                    "proposal": 0,
                    "step": 0,
                    "selected_kind": (
                        "ritz"
                        if self.config.direction_selection_mode == "rayleigh_ritz"
                        else "random"
                    ),
                    "candidate_count": candidate_count,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        record = SimpleNamespace(
            seed_entry_id=0,
            discovered_entry_id=1,
            energy=-2.0,
            accepted_new_basin=True,
        )
        return SimpleNamespace(
            archive=SimpleNamespace(entries=[SimpleNamespace(energy=-1.0)]),
            best_energy=-2.0,
            best_state=state,
            walk_history=[record],
            stats={
                "n_trials": 1,
                "n_minima": 2,
                "force_evaluations": 20,
                "budget_exhausted": 1,
                "direction_choices": 1,
                "direction_candidate_evaluations": candidate_count,
                "duplicate_rate": 0.0,
                "trust_damage_events": 0,
            },
        )

    @staticmethod
    def relaxation_diagnostics():
        return {"proposal_optimizer": "safe-lbfgs-total"}


def fake_runtime(base: FakeBaseRunner, code_root: Path):
    return SimpleNamespace(
        code_root=code_root,
        base_runner=base,
        calculator_wrapper=lambda calculator: calculator,
        walker_class=FakeWalker,
        write_state=lambda path, _state: Path(path).write_text("state\n", encoding="utf-8"),
    )


def test_pair_config_changes_only_selection_and_preserves_frozen_protocol(tmp_path):
    runner = load_runner("ritz_pair_projection")
    base = FakeBaseRunner()

    source, effective, source_diff, pair_diff, protocol = runner.config_projection(
        arm="rayleigh_ritz",
        system="c60",
        case_dir=tmp_path / "case",
        total_force_budget=6000,
        seed=77,
        base_runner=base,
    )

    assert source["oracle_candidates"] == effective["oracle_candidates"] == 12
    assert source["direction_selection_mode"] == "discrete"
    assert effective["direction_selection_mode"] == "rayleigh_ritz"
    assert pair_diff == {"direction_selection_mode": ["discrete", "rayleigh_ritz"]}
    assert source_diff == {
        "direction_selection_mode": ["discrete", "rayleigh_ritz"],
        "max_force_evals": [None, 6000],
        "rng_seed": [42, 77],
    }
    assert protocol["same_native_candidate_and_hvp_protocol"] is True
    for field in (
        "hvp_epsilon",
        "oracle_candidates",
        "proposal_optimizer",
        "target_uphill_energy",
        "quench_optimizer",
        "local_softening_mode",
    ):
        assert protocol["matched_fields"][field] == source[field]
    for config in (source, effective):
        assert config["direction_synthesis_mode"] == "none"
        assert config["direction_type_ucb_enabled"] is False
        assert config["direction_archive_enabled"] is False
        assert config["direction_probe_enabled"] is False
        assert config["plateau_evolution_enabled"] is False
        assert config["archive_escape_momentum_enabled"] is False

    _, pdo, _, pdo_pair_diff, _ = runner.config_projection(
        arm="discrete",
        system="pdo",
        case_dir=tmp_path / "pdo",
        total_force_budget=6000,
        seed=77,
        base_runner=base,
    )
    assert pdo["oracle_candidates"] == 8
    assert pdo_pair_diff == {"direction_selection_mode": ["discrete", "rayleigh_ritz"]}


def test_cpu_fake_run_records_direction_cost_without_equating_legacy_candidate_count_to_hvp(tmp_path):
    runner = load_runner("ritz_fake_run")
    base = FakeBaseRunner()
    code_root = tmp_path / "repo"
    code_root.mkdir()
    runtime = fake_runtime(base, code_root)
    output = tmp_path / "ritz-output"
    commit = "e" * 40

    checked = runner.preflight(
        arm="rayleigh_ritz",
        system="c60",
        seed=42,
        total_force_budget=20,
        code_root=code_root,
        output_dir=output,
        expected_git_commit=commit,
        target_loader=lambda root: runtime,
        git_head=lambda root: commit,
        tracked_clean=lambda root: True,
    )
    summary = runner.run(
        arm="rayleigh_ritz",
        system="c60",
        seed=42,
        total_force_budget=20,
        code_root=code_root,
        output_dir=output,
        expected_git_commit=commit,
        target_loader=lambda root: runtime,
        preflight_fn=lambda **_: checked,
    )

    assert summary["purpose_counts"]["direction_oracle"] == 6
    assert summary["direction_selection_audit"] == {
        "selection_count": 1,
        "reported_candidate_count_sum": 4,
        "reported_candidate_count_max": 4,
        "reported_candidate_count_semantics": "includes_zero_hvp_synthetic_ritz_when_constructed",
        "native_candidate_count": None,
        "native_candidate_count_reason": "unsupported_by_current_direction_trace",
        "direction_oracle_force_evaluations": 6,
        "selected_kind_counts": {"ritz": 1},
    }
    assert summary["plain_ritz_hvp_contract"]["runtime_cost_source"] == "purpose_counts.direction_oracle"
    assert sum(summary["purpose_counts"].values()) == 20
    assert summary["termination"]["reason"] == "force_budget_exhausted"
    assert (output / "energy_trace.json").is_file()
    assert (output / "direction_trace.jsonl").is_file()
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


def _summary(module, *, arm: str, system: str, seed: int, output: str) -> dict:
    source, effective, source_diff, pair_diff, protocol = module.config_projection(
        arm=arm,
        system=system,
        case_dir=Path(output),
        total_force_budget=20,
        seed=seed,
        base_runner=FakeBaseRunner(),
    )
    candidate_count = 4 if arm == "rayleigh_ritz" else 3
    return {
        "schema_version": 1,
        "arm": arm,
        "system": system,
        "seed": seed,
        "total_force_budget": 20,
        "force_evaluations": 20,
        "base_preflight": {
            "execution_commit": "e" * 40,
            "input_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "runtime_versions": {"python": "test"},
            "cuda": {"available": True},
            "calculator": {"device": "cuda"},
        },
        "source_config": source,
        "effective_config": effective,
        "source_to_effective_config_diff": source_diff,
        "paired_arm_config_diff": pair_diff,
        "plain_ritz_hvp_contract": protocol,
        "purpose_counts": {
            "bootstrap_true_quench": 2,
            "direction_oracle": 6,
            "biased_proposal_relax": 12,
            "landing_true_quench": 0,
            "unattributed": 0,
        },
        "initial_energy_eV": -1.0,
        "best_energy_eV": -2.0,
        "energy_drop_eV": 1.0,
        "stats": {
            "n_trials": 1,
            "n_minima": 2,
            "force_evaluations": 20,
            "budget_exhausted": 1,
            "direction_choices": 1,
            "direction_candidate_evaluations": candidate_count,
            "duplicate_rate": 0.0,
            "trust_damage_events": 0,
        },
        "record_counts": {
            "n_trials": 1,
            "walk_records": 1,
            "direction_records": 1,
            "unlogged_walk_trials": 0,
            "unlogged_direction_choices": 0,
        },
        "direction_selection_audit": {
            "selection_count": 1,
            "reported_candidate_count_sum": candidate_count,
            "reported_candidate_count_max": candidate_count,
            "reported_candidate_count_semantics": "includes_zero_hvp_synthetic_ritz_when_constructed",
            "native_candidate_count": None,
            "native_candidate_count_reason": "unsupported_by_current_direction_trace",
            "direction_oracle_force_evaluations": 6,
            "selected_kind_counts": {"ritz" if arm == "rayleigh_ritz" else "random": 1},
        },
        "timing": {"total_wall_time_s": 1.0},
        "termination": {"reason": "force_budget_exhausted"},
    }


def test_analyzer_pairs_case_summaries_and_marks_energy_auc_unsupported_without_cumulative_fe(tmp_path):
    runner = load_runner("ritz_analysis_runner")
    analyzer = _load(ANALYZER_PATH, "ritz_evidence")
    for system, seed in (("c60", 42), ("pdo", 42)):
        for arm in ("discrete", "rayleigh_ritz"):
            case = tmp_path / f"{arm}-{system}-{seed}"
            case.mkdir()
            (case / "summary.json").write_text(
                json.dumps(_summary(runner, arm=arm, system=system, seed=seed, output=str(case))),
                encoding="utf-8",
            )
            (case / "energy_trace.json").write_text(
                json.dumps(
                    [
                        {"trial": 0, "energy_eV": -1.0, "best_energy_eV": -1.0},
                        {"trial": 1, "energy_eV": -2.0, "best_energy_eV": -2.0},
                    ]
                ),
                encoding="utf-8",
            )

    evidence = analyzer.build_evidence(tmp_path)

    assert len(evidence["pairs"]) == 2
    assert evidence["pairs"][0]["arms"]["rayleigh_ritz"]["direction_fe_fraction"] == pytest.approx(0.3)
    assert evidence["pairs"][0]["arms"]["discrete"]["best_energy_auc_eV_force_evals"] is None
    assert evidence["pairs"][0]["arms"]["discrete"]["best_energy_auc_reason"] == "unsupported_no_cumulative_total_force_evaluations_in_energy_trace"
    assert evidence["pairs"][0]["arms"]["discrete"]["duplicates"] is None
    assert evidence["pairs"][0]["arms"]["discrete"]["failures"] is None
    assert "不证明平衡态无偏性" in analyzer.render_conclusion(evidence)

    output = tmp_path / "analysis"
    written = analyzer.write_evidence(input_dir=tmp_path, output_dir=output)
    assert json.loads((output / "evidence.json").read_text(encoding="utf-8")) == written
    assert "claim ceiling" in (output / "conclusion.md").read_text(encoding="utf-8")
