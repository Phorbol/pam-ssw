"""Preregistered contracts for the Stage-2 block-Krylov GPU ablation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


RUN_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-block-krylov-direction-gpu-ablation"
)
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
ANALYZER_PATH = RUN_ROOT / "analyze_evidence.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stage2_runner_and_analyzer_are_preregistered_before_terminal_execution():
    assert RUNNER_PATH.is_file(), "Stage-2 runner must be committed before GPU execution"
    assert ANALYZER_PATH.is_file(), "Stage-2 analyzer must be committed before GPU execution"


def test_stage2_arms_have_stable_amended_allocation_order():
    runner = _load(RUNNER_PATH, "block_krylov_arm_contract")

    assert tuple(runner.ARMS) == (
        "discrete",
        "variational_breadth",
        "balanced_refinement",
        "deep_refinement",
    )
    assert runner.ARMS["discrete"] == {"direction_selection_mode": "discrete"}
    assert runner.ARMS["variational_breadth"] == {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 6,
        "block_krylov_depth": 1,
    }
    assert runner.ARMS["balanced_refinement"] == {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    }
    assert runner.ARMS["deep_refinement"] == {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    }
    assert runner.SEEDS == (42, 43, 44)
    assert runner.TOTAL_FORCE_BUDGET == 6000


def test_block_trace_contract_requires_exact_hvp_and_force_ledger():
    runner = _load(RUNNER_PATH, "block_krylov_trace_contract")
    rows = [
        {
            "selected_kind": "block_ritz",
            "candidate_count": 0,
            "krylov_blocks": 2,
            "krylov_depth": 3,
            "krylov_initial_basis_columns": [2, 2],
            "krylov_hvp_requested": 12,
            "krylov_hvp_consumed": 12,
            "krylov_hvp_count": 12,
            "oracle_selection_force_evaluations_delta": 24,
        }
    ]

    audit = runner.validate_direction_trace(
        arm="balanced_refinement",
        direction_rows=rows,
    )

    assert audit["selection_count"] == 1
    assert audit["direction_oracle_force_evaluations"] == 24
    assert audit["selected_kind_counts"] == {"block_ritz": 1}
    bad = [dict(rows[0], oracle_selection_force_evaluations_delta=10)]
    with pytest.raises(RuntimeError, match=r"2 \* krylov_hvp_consumed"):
        runner.validate_direction_trace(arm="balanced_refinement", direction_rows=bad)


def test_discrete_trace_closes_nonzero_direction_oracle_force_ledger():
    runner = _load(RUNNER_PATH, "discrete_trace_contract")

    audit = runner.validate_direction_trace(
        arm="discrete",
        direction_rows=[
            {
                "selected_kind": "random",
                "candidate_count": 3,
                "oracle_selection_force_evaluations_delta": 6,
            }
        ],
    )

    assert audit["direction_oracle_force_evaluations"] == 6
    with pytest.raises(RuntimeError, match="contains block keys"):
        runner.validate_direction_trace(
            arm="discrete",
            direction_rows=[
                {
                    "selected_kind": "random",
                    "candidate_count": 3,
                    "oracle_selection_force_evaluations_delta": 6,
                    "krylov_blocks": 2,
                }
            ],
        )


def test_analyzer_accepts_realistically_shuffled_c60_case_order():
    analyzer = _load(ANALYZER_PATH, "block_krylov_evidence_contract")
    cases = []
    for seed in (42, 43, 44):
        cases.extend(
            [
                _case("discrete", "c60", seed, best=-10.0, auc=-50000.0, archive=10),
                _case("variational_breadth", "c60", seed, best=-10.2, auc=-51000.0, archive=9),
                _case("balanced_refinement", "c60", seed, best=-10.3, auc=-51050.0, archive=10),
                _case("deep_refinement", "c60", seed, best=-10.3, auc=-51100.0, archive=10),
            ]
        )
    shuffled = [cases[index] for index in (7, 0, 10, 5, 2, 9, 4, 11, 1, 8, 6, 3)]

    evidence = analyzer.build_evidence_from_cases(shuffled, system="c60")

    assert evidence["c60_survivors"] == [
        "variational_breadth",
        "balanced_refinement",
        "deep_refinement",
    ]
    assert evidence["selected_pdo_transfer_arm"] == "deep_refinement"
    assert evidence["pdo_status"] == "not_run_pending_selected_c60_survivor"
    assert [(row["seed"], row["arm"]) for row in evidence["runs"]] == [
        (seed, arm)
        for seed in (42, 43, 44)
        for arm in ("discrete", "variational_breadth", "balanced_refinement", "deep_refinement")
    ]


def test_analyzer_rejects_manual_pdo_arm():
    analyzer = _load(ANALYZER_PATH, "block_krylov_manual_pdo_contract")
    with pytest.raises(analyzer.EvidenceError, match="selected_pdo_transfer_arm"):
        analyzer.build_evidence_from_cases(
            [
                _case("discrete", "pdo", 42, best=-1.0, auc=-1.0, archive=1),
                _case("balanced_refinement", "pdo", 42, best=-1.1, auc=-1.1, archive=1),
            ],
            system="pdo",
            selected_pdo_transfer_arm="deep_refinement",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda case: case["_runtime_identity"].update(model_sha256="different"), "runtime identity"),
        (lambda case: case["_effective_config"].update(proposal_optimizer="fire"), "effective config"),
    ],
)
def test_analyzer_rejects_mixed_provenance_and_non_direction_config(mutate, message):
    analyzer = _load(ANALYZER_PATH, f"block_krylov_mixed_{message.replace(' ', '_')}")
    cases = [
        _case(arm, "c60", seed, best=-10.0, auc=-50000.0, archive=10)
        for seed in (42, 43, 44)
        for arm in ("discrete", "variational_breadth", "balanced_refinement", "deep_refinement")
    ]
    bad = deepcopy(cases)
    mutate(bad[-1])

    with pytest.raises(analyzer.EvidenceError, match=message):
        analyzer.build_evidence_from_cases(bad, system="c60")


def test_pdo_arm_requires_current_analyzer_c60_evidence(tmp_path):
    runner = _load(RUNNER_PATH, "block_krylov_pdo_gate")
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schema_id": runner.EVIDENCE_SCHEMA_ID,
                "system": "c60",
                "cohort": {
                    "arms": list(runner.ARMS),
                    "seeds": list(runner.SEEDS),
                    "completed_cases": 12,
                },
                "c60_survivors": ["deep_refinement"],
                "selected_pdo_transfer_arm": "deep_refinement",
                "pdo_status": "not_run_pending_selected_c60_survivor",
                "provenance": {
                    "analyzer_sha256": runner._sha256(ANALYZER_PATH),
                },
            }
        ),
        encoding="utf-8",
    )

    assert runner.validate_c60_evidence(evidence_path, arm="discrete") == "deep_refinement"
    assert runner.validate_c60_evidence(evidence_path, arm="deep_refinement") == "deep_refinement"
    with pytest.raises(RuntimeError, match="selected survivor"):
        runner.validate_c60_evidence(evidence_path, arm="balanced_refinement")


def test_failed_case_does_not_claim_final_output_and_can_retry(monkeypatch, tmp_path):
    runner = _load(RUNNER_PATH, "block_krylov_atomic_case")
    attempts = {"count": 0}

    @dataclass(frozen=True)
    class FakeConfig:
        max_force_evals: int = 6000
        rng_seed: int = 42
        direction_selection_mode: str = "discrete"
        block_krylov_blocks: int = 2
        block_krylov_depth: int = 3
        direction_diagnostics_path: str = ""
        accepted_structures_log: str | None = None
        accepted_structures_dir: str | None = None

    class FakeBase:
        @staticmethod
        def build_config(_system, case_dir):
            return FakeConfig(direction_diagnostics_path=str(Path(case_dir) / "direction_trace.jsonl"))

        @staticmethod
        def _calculator():
            return object()

        @staticmethod
        def load_state(_system):
            return object()

    class FakeCounts:
        total = 6000

        @staticmethod
        def as_dict():
            return {"direction_oracle": 6, "biased_proposal_relax": 5994, "unattributed": 0}

    class FakeWalker:
        def __init__(self, *, calculator, config, softening_enabled):
            self.config = config
            self.calculator = SimpleNamespace(snapshot=lambda: FakeCounts())

        def run(self, state):
            attempts["count"] += 1
            path = Path(self.config.direction_diagnostics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "selected_kind": "random",
                        "candidate_count": 3,
                        "oracle_selection_force_evaluations_delta": 6,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            if attempts["count"] == 1:
                raise RuntimeError("injected case failure")
            return SimpleNamespace(
                archive=SimpleNamespace(entries=[SimpleNamespace(energy=-1.0)]),
                best_energy=-2.0,
                best_state=state,
                walk_history=[],
                stats={
                    "force_evaluations": 6000,
                    "budget_exhausted": 1,
                    "direction_choices": 1,
                    "n_trials": 0,
                    "n_minima": 1,
                    "duplicate_rate": 0.0,
                },
            )

        @staticmethod
        def relaxation_diagnostics():
            return {}

    base = FakeBase()
    runtime = SimpleNamespace(
        code_root=tmp_path,
        base_runner=base,
        calculator_wrapper=lambda calculator: calculator,
        walker_class=FakeWalker,
        write_state=lambda path, state: Path(path).write_text("state\n", encoding="utf-8"),
    )
    final = tmp_path / "case"

    def projection(*, arm, system, case_dir, total_force_budget, seed, base_runner):
        source_config = base_runner.build_config(system, case_dir)
        effective_config = replace(
            source_config,
            max_force_evals=total_force_budget,
            rng_seed=seed,
            **runner.ARMS[arm],
        )
        return asdict(source_config), asdict(effective_config), {}, {}, {}

    monkeypatch.setattr(runner, "config_projection", projection)
    source, effective, source_diff, arm_diffs, _ = projection(
        arm="discrete",
        system="c60",
        case_dir=final,
        total_force_budget=6000,
        seed=42,
        base_runner=base,
    )
    checked = {
        "source_config": source,
        "effective_config": effective,
        "source_to_effective_config_diff": source_diff,
        "paired_arm_config_diffs": arm_diffs,
    }
    kwargs = dict(
        arm="discrete",
        system="c60",
        seed=42,
        total_force_budget=6000,
        code_root=tmp_path,
        output_dir=final,
        expected_git_commit="e" * 40,
        target_loader=lambda root: runtime,
        preflight_fn=lambda **values: checked,
    )

    with pytest.raises(RuntimeError, match="injected case failure"):
        runner.run(**kwargs)
    assert not final.exists()
    assert not final.with_name(f".{final.name}.tmp").exists()

    summary = runner.run(**kwargs)
    assert summary["force_evaluations"] == 6000
    assert (final / "summary.json").is_file()
    assert not final.with_name(f".{final.name}.tmp").exists()


def test_analyzer_replaces_fully_written_temporary_files(monkeypatch, tmp_path):
    analyzer = _load(ANALYZER_PATH, "block_krylov_atomic_evidence")
    evidence = {
        "schema_version": 1,
        "schema_id": analyzer.EVIDENCE_SCHEMA_ID,
        "system": "c60",
        "c60_survivors": [],
        "selected_pdo_transfer_arm": None,
        "pdo_status": "not_run_no_c60_survivor",
    }
    monkeypatch.setattr(analyzer, "build_evidence", lambda **kwargs: evidence)
    original_replace = Path.replace
    replacements = []

    def record_replace(path, target):
        replacements.append((path.name, Path(target).name))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", record_replace)
    written = analyzer.write_evidence(input_dir=tmp_path, output_dir=tmp_path, system="c60")

    assert written == evidence
    assert replacements == [
        (".evidence.json.tmp", "evidence.json"),
        (".conclusion.md.tmp", "conclusion.md"),
    ]
    assert not (tmp_path / ".evidence.json.tmp").exists()
    assert not (tmp_path / ".conclusion.md.tmp").exists()


def _case(arm: str, system: str, seed: int, *, best: float, auc: float, archive: int) -> dict[str, object]:
    overrides = {
        "discrete": {"direction_selection_mode": "discrete"},
        "variational_breadth": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 6,
            "block_krylov_depth": 1,
        },
        "balanced_refinement": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 2,
            "block_krylov_depth": 3,
        },
        "deep_refinement": {
            "direction_selection_mode": "block_krylov",
            "block_krylov_blocks": 1,
            "block_krylov_depth": 6,
        },
    }[arm]
    source = {
        "max_force_evals": None,
        "rng_seed": 42,
        "direction_selection_mode": "discrete",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
        "proposal_optimizer": "safe-lbfgs-total",
    }
    effective = source | {"max_force_evals": 6000, "rng_seed": seed} | overrides
    return {
        "arm": arm,
        "system": system,
        "seed": seed,
        "best_energy_eV": best,
        "best_energy_auc_eV_force_evals": auc,
        "archive_size": archive,
        "budget_closed": True,
        "total_force_evaluations": 6000,
        "_source_config": source,
        "_effective_config": effective,
        "_runtime_identity": {
            "execution_commit": "e" * 40,
            "input_sha256": "i" * 64,
            "model_sha256": "m" * 64,
            "runtime_versions": {"python": "test"},
            "cuda": {"available": True},
            "calculator": {"device": "cuda", "precision": "float32"},
        },
    }
