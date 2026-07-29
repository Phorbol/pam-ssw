from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import pytest

from pamssw import LSSSWConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260729-staged-direction-efficiency-ablation"
)
RUNNER_PATH = RUN_ROOT / "run_stage.py"
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_protocol(name: str = "_staged_efficiency_runner_protocol"):
    return load_module(PROTOCOL_PATH, name)


def load_runner(name: str = "_staged_efficiency_runner"):
    return load_module(RUNNER_PATH, name)


class FakeBaseRunner:
    def __init__(self, overrides: dict[str, object] | None = None) -> None:
        self.overrides = overrides or {}

    def build_config(self, system: str, case_dir: Path) -> LSSSWConfig:
        assert system == "c60"
        config = LSSSWConfig(
            max_trials=200,
            max_force_evals=None,
            max_steps_per_walk=8,
            quench_fmax=0.01,
            quench_maxiter=400,
            quench_optimizer="scipy-lbfgsb",
            quench_fallback_optimizer=None,
            rng_seed=42,
            oracle_candidates=12,
            proposal_relax_steps=80,
            proposal_optimizer="safe-lbfgs-total",
            proposal_fmax=0.05,
            direction_selection_mode="discrete",
            direction_synthesis_mode="none",
            direction_type_ucb_enabled=False,
            archive_escape_momentum_enabled=False,
            choice_aligned_softening_enabled=False,
            direction_probe_enabled=False,
            plateau_evolution_enabled=False,
            direction_curvature_source="inner",
            direction_diagnostics_enabled=True,
            direction_diagnostics_path=str(
                Path(case_dir) / "direction_trace.jsonl"
            ),
        )
        return replace(config, **self.overrides)


def candidate_count_case(protocol):
    return protocol.CaseSpec(
        stage="candidate_count",
        state_id="intermediate_accepted",
        seed=42,
        arm="k4",
        repeat=0,
        settings=protocol.RetainedSettings(
            enable_momentum_candidate=True,
            oracle_candidates=4,
            max_steps_per_walk=8,
            proposal_relax_steps=80,
        ),
    )


def valid_trace():
    return [
        {
            "step": 0,
            "selected_kind": "bond",
            "candidate_count": 4,
            "evaluated_candidate_kind_counts": {
                "bond": 2,
                "random": 2,
            },
            "oracle_selection_force_evaluations_delta": 8,
            "oracle_direction_force_evaluations_delta": 8,
        },
        {
            "step": 1,
            "selected_kind": "momentum",
            "candidate_count": 4,
            "evaluated_candidate_kind_counts": {
                "bond": 2,
                "momentum": 1,
                "random": 1,
            },
            "oracle_selection_force_evaluations_delta": 8,
            "oracle_direction_force_evaluations_delta": 8,
        },
    ]


def test_config_projection_changes_only_preregistered_case_and_quench_fields(
    tmp_path,
):
    protocol = load_protocol("_staged_projection_protocol")
    runner = load_runner("_staged_projection_runner")

    source, effective, diff = runner.config_projection(
        case=candidate_count_case(protocol),
        case_dir=tmp_path,
        base_runner=FakeBaseRunner(),
    )

    assert source["oracle_candidates"] == 12
    assert effective["direction_selection_mode"] == "discrete"
    assert effective["direction_synthesis_mode"] == "none"
    assert effective["direction_type_ucb_enabled"] is False
    assert effective["archive_escape_momentum_enabled"] is False
    assert effective["proposal_optimizer"] == "safe-lbfgs-total"
    assert effective["proposal_fmax"] == pytest.approx(0.05)
    assert effective["enable_momentum_candidate"] is True
    assert effective["oracle_candidates"] == 4
    assert effective["max_steps_per_walk"] == 8
    assert effective["proposal_relax_steps"] == 80
    assert effective["quench_optimizer"] == "ase-lbfgs"
    assert effective["quench_fallback_optimizer"] == "ase-fire"
    assert effective["quench_fmax"] == pytest.approx(0.01)
    assert effective["quench_maxiter"] == 400
    assert set(diff) == {
        "max_trials",
        "oracle_candidates",
        "quench_fallback_optimizer",
        "quench_optimizer",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("direction_selection_mode", "rayleigh_ritz"),
        ("direction_synthesis_mode", "regularized_ritz"),
        ("direction_type_ucb_enabled", True),
        ("archive_escape_momentum_enabled", True),
        ("choice_aligned_softening_enabled", True),
        ("direction_probe_enabled", True),
        ("plateau_evolution_enabled", True),
        ("direction_curvature_source", "true"),
        ("proposal_optimizer", "ase-fire"),
        ("proposal_fmax", 0.04),
        ("quench_fmax", 0.02),
    ],
)
def test_config_projection_rejects_frozen_protocol_drift(
    tmp_path,
    field,
    value,
):
    protocol = load_protocol(f"_staged_drift_protocol_{field}")
    runner = load_runner(f"_staged_drift_runner_{field}")

    with pytest.raises(RuntimeError, match="frozen protocol drifted"):
        runner.config_projection(
            case=candidate_count_case(protocol),
            case_dir=tmp_path,
            base_runner=FakeBaseRunner({field: value}),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_momentum_candidate", False),
        ("oracle_candidates", 8),
        ("max_steps_per_walk", 6),
        ("proposal_relax_steps", 40),
    ],
)
def test_config_projection_rejects_frozen_baseline_drift(
    tmp_path,
    field,
    value,
):
    protocol = load_protocol(f"_staged_base_protocol_{field}")
    runner = load_runner(f"_staged_base_runner_{field}")

    with pytest.raises(RuntimeError, match="frozen baseline drifted"):
        runner.config_projection(
            case=candidate_count_case(protocol),
            case_dir=tmp_path,
            base_runner=FakeBaseRunner({field: value}),
        )


def test_direction_trace_closes_candidate_sources_and_hvp_cost():
    runner = load_runner("_staged_trace_runner")

    audit = runner.validate_direction_trace(
        valid_trace(),
        oracle_candidates=4,
    )

    assert audit == {
        "selection_count": 2,
        "candidate_count": 8,
        "candidate_kind_counts": {
            "bond": 4,
            "momentum": 1,
            "random": 3,
        },
        "selected_kind_counts": {"bond": 1, "momentum": 1},
        "direction_oracle_force_evaluations": 16,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows[0][
                "evaluated_candidate_kind_counts"
            ].__setitem__("random", 1),
            "source counts do not close",
        ),
        (
            lambda rows: rows[1].__setitem__("selected_kind", "anchor"),
            "selected source was not evaluated",
        ),
        (
            lambda rows: rows[0].__setitem__(
                "oracle_selection_force_evaluations_delta",
                6,
            ),
            "oracle_selection_force_evaluations_delta does not close",
        ),
        (
            lambda rows: rows[0].__setitem__(
                "oracle_direction_force_evaluations_delta",
                10,
            ),
            "oracle_direction_force_evaluations_delta does not close",
        ),
    ],
)
def test_direction_trace_rejects_unclosed_source_or_hvp_ledger(
    mutation,
    message,
):
    runner = load_runner(
        f"_staged_trace_failure_{message.replace(' ', '_')}"
    )
    rows = deepcopy(valid_trace())
    mutation(rows)

    with pytest.raises(RuntimeError, match=message):
        runner.validate_direction_trace(rows, oracle_candidates=4)
