"""Fail-closed evidence contracts for the fixed-state direction audit."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
import sys

import pytest


AUDIT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-block-krylov-direction-audit"
)
ANALYZER_PATH = AUDIT_ROOT / "analyze_fixed_state_audit.py"
RUNNER_PATH = AUDIT_ROOT / "run_fixed_state_audit.py"


def _load_analyzer():
    spec = importlib.util.spec_from_file_location("block_krylov_fixed_state_audit", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner():
    spec = importlib.util.spec_from_file_location("block_krylov_fixed_state_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(name: str, arm: str) -> dict[str, object]:
    blocks, depth = {
        "variational_breadth": (6, 1),
        "shallow_refinement": (3, 2),
        "balanced_refinement": (2, 3),
        "deep_refinement": (1, 6),
    }[arm]
    hvp_count = 2 * blocks * depth
    return {
        "case": name,
        "kind": "analytic",
        "arm": arm,
        "force_evaluations": 2 * hvp_count,
        "krylov_hvp_count": hvp_count,
        "purpose_counts": {"direction_oracle": 2 * hvp_count, "unattributed": 0},
        "selection": {"curvature": 1.0, "true_curvature": 1.0},
        "state_sha256": "c" * 64,
        "wall_seconds": 0.1,
        "diagnostics": {
            "krylov_blocks": blocks,
            "krylov_selected_block": 0,
            "krylov_hvp_count": hvp_count,
            "krylov_dimensions": [2] * blocks,
            "krylov_initial_ranks": [2] * blocks,
            "krylov_residual_norm": 0.0,
            "krylov_initial_span_overlap": 1.0,
            "krylov_antisymmetry": 0.0,
            "krylov_termination": "depth_reached",
            "direction_participation_ratio": 1.0,
        },
    }


def _raw() -> dict[str, object]:
    allocations = {
        "variational_breadth": {"block_krylov_blocks": 6, "block_krylov_depth": 1},
        "shallow_refinement": {"block_krylov_blocks": 3, "block_krylov_depth": 2},
        "balanced_refinement": {"block_krylov_blocks": 2, "block_krylov_depth": 3},
        "deep_refinement": {"block_krylov_blocks": 1, "block_krylov_depth": 6},
    }
    return {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "dirty": False,
        "operator": "total_proposal_central_fd",
        "hvp_epsilon": 1.0e-3,
        "max_hvps": 12,
        "allocations": allocations,
        "runtime": {"mode": "analytic_only", "device": "cpu", "precision": "float64"},
        "rows": [_row("diagonal", arm) for arm in allocations],
        "fixed_state_registry": {"c60": [], "pdo": []},
        "wall_seconds": 0.4,
    }


def _fixed_registry() -> dict[str, list[dict[str, object]]]:
    """Minimal but complete locked cohort provenance for analyzer contracts."""

    states = (
        ("bootstrap_quenched", "bootstrap", "reconstruct_frozen_starter_true_quench"),
        ("intermediate_accepted", "intermediate", "locked_accepted_structure"),
        ("plateau_accepted", "plateau", "locked_accepted_structure"),
    )
    registry: dict[str, list[dict[str, object]]] = {}
    for system in ("c60", "pdo"):
        entries: list[dict[str, object]] = []
        for state_id, phase, source in states:
            entry: dict[str, object] = {
                "state_id": state_id,
                "phase": phase,
                "source": source,
                "origin_summary_path": f"runs/origin-{system}/summary.json",
                "origin_summary_sha256": f"{system[0]}" * 64,
                "origin_commit": "b" * 40,
                "model_path": "/models/mace.model",
                "model_sha256": "m" * 64,
            }
            if source == "locked_accepted_structure":
                entry.update(
                    structure_path=f"runs/origin-{system}/{state_id}.xyz",
                    structure_sha256=f"{state_id[0]}" * 64,
                )
            else:
                entry.update(
                    raw_input_path=f"inputs/{system}.xyz",
                    raw_input_sha256=f"{system[-1]}" * 64,
                    strict_wrapper_path="runs/strict/run_production.py",
                    strict_wrapper_sha256="w" * 64,
                    base_runner_path="runs/base/run_production.py",
                    base_runner_sha256="r" * 64,
                    strict_config_diff={
                        "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
                        "quench_fallback_optimizer": [None, "ase-fire"],
                    },
                )
            entries.append(entry)
        registry[system] = entries
    return registry


def _fixed_state_provenance(entry: dict[str, object], *, state_sha256: str) -> dict[str, object]:
    provenance = dict(entry)
    provenance["state_sha256"] = state_sha256
    if entry["source"] == "locked_accepted_structure":
        return provenance
    provenance.update(
        strict_wrapper_path="runs/strict/run_production.py",
        strict_wrapper_sha256="w" * 64,
        base_runner_path="runs/base/run_production.py",
        base_runner_sha256="r" * 64,
        effective_config={
            "quench_optimizer": "ase-lbfgs",
            "quench_fallback_optimizer": "ase-fire",
            "quench_fmax": 0.01,
            "block_krylov_blocks": 2,
            "block_krylov_depth": 3,
        },
        origin_effective_config={
            "quench_optimizer": "ase-lbfgs",
            "quench_fallback_optimizer": "ase-fire",
            "quench_fmax": 0.01,
        },
        effective_config_diff=entry["strict_config_diff"],
        current_config_schema_additions={"block_krylov_blocks": 2, "block_krylov_depth": 3},
        bootstrap_purpose_counts={"starter_true_quench": 2, "unattributed": 0},
        bootstrap_force_evaluations=2,
        bootstrap_wall_seconds=0.1,
        resulting_state_sha256=state_sha256,
    )
    return provenance


def _fixed_raw() -> dict[str, object]:
    raw = _raw()
    registry = _fixed_registry()
    raw["fixed_state_registry"] = registry
    raw["runtime"] = {
        "mode": "full_fixed_state",
        "device": "cuda",
        "precision": "float32",
        "dtype": "float32",
        "model_path": "/models/mace.model",
        "model_sha256": "m" * 64,
    }
    rows: list[dict[str, object]] = []
    for system, entries in registry.items():
        for entry in entries:
            for arm in raw["allocations"]:
                row = _row(f"{system}:{entry['state_id']}", arm)
                state_sha256 = ("f" if system == "c60" else "e") * 64
                row.update(
                    kind="fixed_state",
                    system=system,
                    state_id=entry["state_id"],
                    case_metadata={"phase": entry["phase"]},
                    state_sha256=state_sha256,
                    state_provenance=_fixed_state_provenance(entry, state_sha256=state_sha256),
                    calculator={
                        "kind": "mace_omat_0_small",
                        "model_path": "/models/mace.model",
                        "model_sha256": "m" * 64,
                        "device": "cuda",
                        "precision": "float32",
                        "dtype": "float32",
                    },
                )
                rows.append(row)
    raw["rows"] = rows
    return raw


def test_analyzer_projects_only_budget_closed_direction_evidence():
    assert ANALYZER_PATH.is_file()
    analyzer = _load_analyzer()

    evidence = analyzer.project_evidence(_raw())

    assert evidence["accounting_invariants"]["all_rows_exact_central_fd"] is True
    assert evidence["accounting_invariants"]["all_rows_within_hvp_budget"] is True
    assert evidence["analytic_cases"] == ["diagonal"]
    assert "does not rank allocations by terminal energy" in evidence["claim_ceiling"]


def test_analyzer_rejects_missing_required_direction_diagnostic():
    assert ANALYZER_PATH.is_file()
    analyzer = _load_analyzer()
    raw = deepcopy(_raw())
    del raw["rows"][0]["diagnostics"]["krylov_antisymmetry"]

    with pytest.raises(RuntimeError, match="krylov_antisymmetry"):
        analyzer.project_evidence(raw)


def test_analyzer_rejects_raw_evidence_without_runtime_provenance():
    analyzer = _load_analyzer()
    raw = deepcopy(_raw())
    del raw["runtime"]

    with pytest.raises(RuntimeError, match="runtime"):
        analyzer.project_evidence(raw)


def test_strict_bootstrap_config_uses_wrapper_effective_protocol_for_c60_and_pdo(tmp_path):
    """Regression: bootstrap must not fall back to scipy L-BFGS-B or PdO fmax=.03."""

    runner = _load_runner()
    strict_wrapper, base_runner = runner._load_frozen_runtime()
    for system in ("c60", "pdo"):
        summary_path = (
            runner.REPO_ROOT
            / f"runs/20260728-safe-lbfgs-strict-quench-200-production/output-{system}-seed42/summary.json"
        )
        summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        config, effective, config_diff = runner._strict_bootstrap_config(
            system=system,
            case_dir=tmp_path / system,
            strict_wrapper=strict_wrapper,
            base_runner=base_runner,
            origin_summary=summary,
        )

        assert config.quench_optimizer == "ase-lbfgs"
        assert config.quench_fallback_optimizer == "ase-fire"
        assert config.quench_fmax == pytest.approx(0.01)
        assert effective["quench_optimizer"] == "ase-lbfgs"
        assert config_diff == summary["strict_quench_wrapper"]["config_diff"]


def test_analyzer_accepts_only_exact_fixed_cohort_and_projects_provenance_summary():
    analyzer = _load_analyzer()

    evidence = analyzer.project_evidence(_fixed_raw())

    assert evidence["c60_states"] == ["bootstrap_quenched", "intermediate_accepted", "plateau_accepted"]
    assert evidence["pdo_states"] == ["bootstrap_quenched", "intermediate_accepted", "plateau_accepted"]
    assert set(evidence["fixed_state_provenance"]) == {"c60", "pdo"}


def test_analyzer_rejects_full_fixed_state_mode_without_the_exact_cohort():
    analyzer = _load_analyzer()
    raw = _raw()
    raw["runtime"]["mode"] = "full_fixed_state"

    with pytest.raises(RuntimeError, match="cohort"):
        analyzer.project_evidence(raw)


def test_analyzer_rejects_fixed_rows_when_runtime_mode_is_analytic_only():
    analyzer = _load_analyzer()
    raw = _fixed_raw()
    raw["runtime"]["mode"] = "analytic_only"

    with pytest.raises(RuntimeError, match="analytic_only"):
        analyzer.project_evidence(raw)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw: raw["rows"][0].update(state_id="forged_state"),
            "state_id",
        ),
        (
            lambda raw: raw["rows"][0].pop("state_provenance"),
            "state_provenance",
        ),
        (
            lambda raw: raw["rows"].pop(),
            "cohort",
        ),
        (
            lambda raw: raw["rows"][0]["state_provenance"].update(state_sha256="0" * 64),
            "state checksum",
        ),
        (
            lambda raw: raw["rows"][0]["state_provenance"]["effective_config"].update(
                quench_optimizer="scipy-lbfgsb"
            ),
            "strict config",
        ),
        (
            lambda raw: raw["rows"][0].pop("calculator"),
            "calculator",
        ),
    ],
)
def test_analyzer_rejects_forged_or_incomplete_fixed_state_provenance(mutate, message):
    analyzer = _load_analyzer()
    raw = _fixed_raw()
    mutate(raw)

    with pytest.raises(RuntimeError, match=message):
        analyzer.project_evidence(raw)


def test_analytic_diagonal_case_does_not_wrap_central_fd_probes_across_periodic_cell():
    """A positive analytic quadratic must not become negative at a cell boundary."""

    assert RUNNER_PATH.is_file()
    runner = _load_runner()
    case = runner._analytic_cases()[0]
    row, _ = runner._direction_row(
        case=case["case"],
        kind="analytic",
        system="analytic",
        state_id=case["case"],
        state=case["state"],
        state_provenance={"origin": "test"},
        calculator_factory=lambda: runner.AnalyticCalculator(case["potential"]),
        calculator_provenance={"kind": "analytic"},
        intent_seed=case["intent_seed"],
        arm="variational_breadth",
        force_rank_one=False,
        case_metadata=case["metadata"],
    )

    assert row["selection"]["curvature"] > 0.0


def test_full_rank_analytic_diagonal_uses_the_preregistered_twelve_hvps():
    runner = _load_runner()
    case = runner._analytic_cases()[0]
    rows = runner._run_case_allocations(
        case=case["case"],
        kind="analytic",
        system="analytic",
        state_id=case["case"],
        state=case["state"],
        state_provenance={"origin": "test"},
        calculator_factory=lambda: runner.AnalyticCalculator(case["potential"]),
        calculator_provenance={"kind": "analytic"},
        intent_seed=case["intent_seed"],
        case_metadata=case["metadata"],
    )

    assert {row["krylov_hvp_count"] for row in rows} == {12}
