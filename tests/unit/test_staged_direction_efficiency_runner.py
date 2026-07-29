from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pamssw import LSSSWConfig
from pamssw.accounting import EvaluationPurpose
from pamssw.state import State


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


class FakePurposeCounts:
    def __init__(self, values):
        self.values = values

    def as_dict(self):
        return dict(self.values)


class FakeCaseCounter:
    def __init__(self, *, unattributed=0, direction_oracle=16):
        self.values = {
            purpose.value: 0 for purpose in EvaluationPurpose
        }
        self.values["unattributed"] = unattributed
        self.values["direction_oracle"] = direction_oracle
        self.values["biased_proposal_relax"] = 20
        self.current = None

    class _Purpose:
        def __init__(self, counter, purpose):
            self.counter = counter
            self.purpose = purpose

        def __enter__(self):
            self.counter.current = self.purpose.value

        def __exit__(self, exc_type, exc, traceback):
            self.counter.current = None

    def purpose(self, purpose):
        return self._Purpose(self, purpose)

    def evaluate(self, state):
        assert self.current is not None
        self.values[self.current] += 1
        return SimpleNamespace(
            energy=float(np.sum(state.positions**2)),
            gradient=np.zeros_like(state.positions),
        )

    def snapshot(self):
        return FakePurposeCounts(self.values)


class FakeEntry:
    def __init__(self, entry_id, energy):
        self.entry_id = entry_id
        self.energy = energy


class FakeArchive:
    def __init__(self, **_kwargs):
        self.entries = []

    def add(self, state, energy, parent_id):
        entry = FakeEntry(len(self.entries), energy)
        self.entries.append(entry)
        return entry


class FakeStepTarget:
    @staticmethod
    def target(_archive):
        return 0.8


class FakeCaseWalker:
    unattributed = 0
    direction_oracle = 16

    def __init__(self, *, calculator, config, softening_enabled):
        self.input_calculator = calculator
        self.config = config
        self.softening_enabled = softening_enabled
        self.calculator = FakeCaseCounter(
            unattributed=self.unattributed,
            direction_oracle=self.direction_oracle,
        )
        self.step_target_controller = FakeStepTarget()

    def _reset_direction_diagnostics(self):
        return None

    def _proposal_pool(self, state, archive, **_kwargs):
        path = Path(self.config.direction_diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(json.dumps(row) for row in valid_trace()) + "\n",
            encoding="utf-8",
        )
        escape = State(
            numbers=state.numbers.copy(),
            positions=state.positions + np.array([[0.2, 0.0, 0.0]]),
            cell=None if state.cell is None else state.cell.copy(),
            pbc=state.pbc,
            fixed_mask=state.fixed_mask.copy(),
        )
        return [SimpleNamespace(state=escape)]

    def relax_true_minimum(self, state, trajectory_name):
        assert trajectory_name is not None
        self.calculator.values["landing_true_quench"] += 10
        landing = State(
            numbers=state.numbers.copy(),
            positions=state.positions - np.array([[0.4, 0.0, 0.0]]),
            cell=None if state.cell is None else state.cell.copy(),
            pbc=state.pbc,
            fixed_mask=state.fixed_mask.copy(),
        )
        return SimpleNamespace(
            state=landing,
            energy=-1.0,
            gradient_norm=0.001,
            n_iter=10,
            telemetry=SimpleNamespace(termination_reason="converged"),
        )

    @staticmethod
    def relaxation_diagnostics():
        return {
            "proposal_relax_count": 2,
            "proposal_relax_median_iterations": 10.0,
            "proposal_relax_p90_iterations": 12.0,
            "proposal_relax_max_iterations": 12.0,
            "proposal_relax_termination_converged": 2,
            "proposal_relax_termination_maxiter": 0,
            "quench_fallback_attempts": 0,
        }

    @staticmethod
    def _is_fragmented_cluster(starter, landing):
        return False


class FakeStateWriter(FakeBaseRunner):
    @staticmethod
    def write_state(path, state):
        values = np.asarray(state.positions, dtype=float)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            " ".join(f"{value:.16g}" for value in values.reshape(-1))
            + "\n",
            encoding="utf-8",
        )


def one_atom_state():
    return State(
        numbers=np.array([6]),
        positions=np.array([[0.5, 0.0, 0.0]]),
    )


def run_fake_case(
    runner,
    protocol,
    tmp_path,
    *,
    walker_factory=FakeCaseWalker,
    certificate=True,
):
    state = one_atom_state()
    case = protocol.CaseSpec(
        stage="momentum",
        state_id="intermediate_accepted",
        seed=42,
        arm="momentum_on",
        repeat=0,
        settings=protocol.RetainedSettings(
            oracle_candidates=4,
        ),
    )
    return runner._run_case(
        case=case,
        state=state,
        state_provenance={
            "state_sha256": runner._state_sha256(state),
        },
        shared_calculator=object(),
        base_runner=FakeStateWriter(),
        case_dir=tmp_path / "case",
        walker_factory=walker_factory,
        archive_factory=FakeArchive,
        certificate_checker=lambda _result, _fmax: certificate,
    )


def test_run_case_publishes_strict_closed_terminal_artifacts(tmp_path):
    protocol = load_protocol("_staged_case_protocol")
    runner = load_runner("_staged_case_runner")

    row = run_fake_case(runner, protocol, tmp_path)

    assert row["status"] == "completed"
    assert row["stage"] == "momentum"
    assert row["repeat"] == 0
    assert row["certificate"] is True
    assert row["is_new_basin"] is True
    assert row["meaningful"] is True
    assert row["selection_probability"] == pytest.approx(1.0)
    assert row["purpose_counts"]["unattributed"] == 0
    assert sum(row["purpose_counts"].values()) == row["force_evaluations"]
    assert row["direction_audit"][
        "direction_oracle_force_evaluations"
    ] == row["purpose_counts"]["direction_oracle"]
    assert row["effective_config"]["proposal_optimizer"] == (
        "safe-lbfgs-total"
    )
    assert row["optimizer_diagnostics"]["proposal_relax_count"] > 0
    assert row["proposal_trace"]["termination_reason"] == "early_exit"
    assert len(row["starter_file_sha256"]) == 64
    assert len(row["escape_sha256"]) == 64
    assert len(row["landing_sha256"]) == 64
    assert json.loads(
        (tmp_path / "case" / "summary.json").read_text(
            encoding="utf-8"
        )
    ) == row


@pytest.mark.parametrize(
    ("walker_factory", "certificate", "message"),
    [
        (
            type(
                "UnattributedWalker",
                (FakeCaseWalker,),
                {"unattributed": 1},
            ),
            True,
            "unattributed force evaluations",
        ),
        (
            FakeCaseWalker,
            False,
            "strict certificate",
        ),
        (
            type(
                "DirectionMismatchWalker",
                (FakeCaseWalker,),
                {"direction_oracle": 17},
            ),
            True,
            "direction purpose ledger",
        ),
    ],
)
def test_run_case_fails_before_summary_when_terminal_or_ledger_is_invalid(
    tmp_path,
    walker_factory,
    certificate,
    message,
):
    suffix = message.replace(" ", "_")
    protocol = load_protocol(f"_staged_bad_case_protocol_{suffix}")
    runner = load_runner(f"_staged_bad_case_runner_{suffix}")

    with pytest.raises(RuntimeError, match=message):
        run_fake_case(
            runner,
            protocol,
            tmp_path,
            walker_factory=walker_factory,
            certificate=certificate,
        )

    assert not (tmp_path / "case" / "summary.json").exists()


def _fake_closed_executor(runner, calls):
    def execute(
        *,
        case,
        state,
        state_provenance,
        shared_calculator,
        base_runner,
        case_dir,
        execution_commit,
    ):
        del state, state_provenance, shared_calculator, base_runner
        calls.append(case.key)
        case_dir = Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        artifact_fields = {}
        for name in ("starter", "escape", "landing"):
            path = case_dir / f"{name}.xyz"
            path.write_text(f"{case.key} {name}\n", encoding="utf-8")
            artifact_fields[f"{name}_path"] = str(path)
            hash_key = (
                "starter_file_sha256"
                if name == "starter"
                else f"{name}_sha256"
            )
            artifact_fields[hash_key] = runner._sha256(path)
        selections = 2
        direction_fe = (
            2 * case.settings.oracle_candidates * selections
        )
        purposes = {
            "bootstrap_true_quench": 0,
            "starter_true_quench": 0,
            "direction_oracle": direction_fe,
            "biased_proposal_relax": 60,
            "escape_true_pes_check": 2,
            "landing_true_quench": 10,
            "post_relax_validation": 0,
            "unattributed": 0,
        }
        row = {
            "status": "completed",
            "stage": case.stage,
            "state_id": case.state_id,
            "seed": case.seed,
            "arm": case.arm,
            "repeat": case.repeat,
            "settings": asdict(case.settings),
            "selection_probability": 1.0,
            "exact_starter_reference": True,
            "certificate": True,
            "is_new_basin": True,
            "meaningful": True,
            "landing_delta_eV": -1.0,
            "force_evaluations": sum(purposes.values()),
            "purpose_counts": purposes,
            "direction_trace_valid": True,
            "direction_audit": {
                "selection_count": selections,
                "candidate_count": (
                    case.settings.oracle_candidates * selections
                ),
                "candidate_kind_counts": {
                    "bond": selections,
                    "random": (
                        case.settings.oracle_candidates * selections
                        - selections
                    ),
                },
                "selected_kind_counts": {"bond": selections},
                "direction_oracle_force_evaluations": direction_fe,
            },
            "optimizer_diagnostics": {
                "proposal_relax_count": 5,
                "proposal_relax_termination_maxiter": 1,
            },
            "fragmented": False,
            "fallback_used": False,
            "generation_wall_time_s": 1.0,
            "quench_wall_time_s": 0.5,
            "execution_commit": execution_commit,
            "effective_config": asdict(case.settings),
            **artifact_fields,
        }
        runner._write_json(case_dir / "summary.json", row)
        return row

    return execute


def test_stage_run_is_atomic_resumable_and_revalidates_artifacts(
    tmp_path,
    monkeypatch,
):
    runner = load_runner("_staged_orchestration_runner")
    monkeypatch.setattr(runner, "_current_commit", lambda: "abc123")
    monkeypatch.setattr(runner, "_tracked_worktree_clean", lambda: True)
    state = one_atom_state()
    runtime = (
        {
            "intermediate_accepted": state,
            "plateau_accepted": state,
        },
        object(),
        FakeStateWriter(),
        {
            "intermediate_accepted": {"state_sha256": "state-a"},
            "plateau_accepted": {"state_sha256": "state-b"},
        },
        {"calculator": "fake"},
    )
    calls = []
    output = tmp_path / "momentum"

    evidence = runner.run_stage(
        stage="momentum",
        output_dir=output,
        expected_git_commit="abc123",
        runtime_loader=lambda: runtime,
        case_executor=_fake_closed_executor(runner, calls),
    )

    assert len(calls) == 24
    assert evidence["cohort"]["completed_cases"] == 24
    assert (output / "raw.json").is_file()
    assert (output / "evidence.json").is_file()
    assert (output / "conclusion.md").is_file()
    assert (output / "run_stage.py").is_file()
    assert (output / "protocol.py").is_file()

    resumed = runner.run_stage(
        stage="momentum",
        output_dir=output,
        expected_git_commit="abc123",
        runtime_loader=lambda: runtime,
        case_executor=_fake_closed_executor(runner, calls),
    )
    assert resumed == evidence
    assert len(calls) == 24

    first = next((output / "cases").iterdir())
    (first / "landing.xyz").unlink()
    with pytest.raises(RuntimeError, match="artifact"):
        runner.run_stage(
            stage="momentum",
            output_dir=output,
            expected_git_commit="abc123",
            runtime_loader=lambda: runtime,
            case_executor=_fake_closed_executor(runner, calls),
        )


def test_stage_context_revalidates_prior_evidence_and_stage_l_gate(
    tmp_path,
):
    protocol = load_protocol("_staged_prior_protocol")
    runner = load_runner("_staged_prior_runner")
    retained = protocol.RetainedSettings()
    evidence_rows = []
    for case in protocol.case_matrix("bias_steps", retained):
        selections = 1
        direction_fe = 2 * case.settings.oracle_candidates
        purposes = {
            "direction_oracle": direction_fe,
            "biased_proposal_relax": 10,
            "landing_true_quench": 1,
            "unattributed": 0,
        }
        evidence_rows.append(
            {
                "status": "completed",
                "stage": case.stage,
                "state_id": case.state_id,
                "seed": case.seed,
                "arm": case.arm,
                "repeat": case.repeat,
                "settings": asdict(case.settings),
                "selection_probability": 1.0,
                "exact_starter_reference": True,
                "certificate": True,
                "is_new_basin": True,
                "meaningful": True,
                "landing_delta_eV": -1.0,
                "force_evaluations": sum(purposes.values()),
                "purpose_counts": purposes,
                "direction_trace_valid": True,
                "direction_audit": {
                    "selection_count": selections,
                    "candidate_count": case.settings.oracle_candidates,
                    "candidate_kind_counts": {
                        "random": case.settings.oracle_candidates,
                    },
                    "selected_kind_counts": {"random": selections},
                    "direction_oracle_force_evaluations": direction_fe,
                },
                "optimizer_diagnostics": {
                    "proposal_relax_count": 1,
                    "proposal_relax_termination_maxiter": 0,
                },
                "fragmented": False,
                "fallback_used": False,
                "generation_wall_time_s": 0.1,
                "quench_wall_time_s": 0.1,
            }
        )
    prior = protocol.build_evidence("bias_steps", retained, evidence_rows)
    prior["input_retained_settings"] = asdict(retained)
    prior_path = tmp_path / "prior.json"
    runner._write_json(prior_path, prior)

    _, _, _, not_entered = runner._stage_context(
        protocol,
        stage="relax_cap",
        prior_evidence_path=prior_path,
        record_not_entered=True,
    )
    assert not_entered is True

    prior["cases"][-1] = deepcopy(prior["cases"][0])
    runner._write_json(prior_path, prior)
    with pytest.raises(ValueError, match="prior evidence"):
        runner._stage_context(
            protocol,
            stage="relax_cap",
            prior_evidence_path=prior_path,
            record_not_entered=True,
        )


def test_campaign_summary_keeps_posterior_gate_closed(tmp_path):
    runner = load_runner("_staged_campaign_summary_runner")

    def case(state_id):
        return {
            "state_id": state_id,
            "meaningful": True,
            "certificate": True,
            "selection_probability": 1.0,
            "force_evaluations": 12,
            "purpose_counts": {
                "direction_oracle": 8,
                "landing_true_quench": 4,
                "unattributed": 0,
            },
            "direction_audit": {
                "candidate_count": 4,
                "selection_count": 1,
                "direction_oracle_force_evaluations": 8,
            },
            "landing_delta_eV": -1.0,
            "arm": "fake",
            "seed": 42,
            "repeat": 0,
            "is_new_basin": True,
            "fragmented": False,
            "fallback_used": False,
        }

    paths = {}
    previous_path = None
    for stage, state_id in (
        ("momentum", "intermediate_accepted"),
        ("candidate_count", "plateau_accepted"),
        ("bias_steps", "plateau_accepted"),
    ):
        path = tmp_path / f"{stage}.json"
        payload = {
            "stage": stage,
            "cohort": {"completed_cases": 1},
            "decision": {"status": "fake"},
            "totals": {
                "force_evaluations": 12,
                "purpose_counts": case(state_id)["purpose_counts"],
                "generation_wall_time_s": 1.0,
                "quench_wall_time_s": 0.5,
            },
            "arm_results": {},
            "cases": [case(state_id)],
            "prior_evidence_sha256": (
                None
                if previous_path is None
                else runner._sha256(previous_path)
            ),
        }
        runner._write_json(path, payload)
        paths[stage] = path
        previous_path = path
    relax_path = tmp_path / "relax_cap.json"
    runner._write_json(
        relax_path,
        {
            "stage": "relax_cap",
            "cohort": {"completed_cases": 0},
            "decision": {"status": "not_entered"},
            "stage_l_entry": {"entered": False},
            "cases": [],
            "prior_evidence_sha256": runner._sha256(previous_path),
        },
    )

    gate = runner.summarize_campaign(
        momentum_evidence_path=paths["momentum"],
        candidate_count_evidence_path=paths["candidate_count"],
        bias_step_evidence_path=paths["bias_steps"],
        relax_cap_evidence_path=relax_path,
        output_root=tmp_path / "summary",
    )

    assert gate["checks"] == {
        "two_fixed_direction_families_with_five_meaningful_each": False,
        "two_starter_classes_with_positive_outcomes": True,
        "complete_action_context_cost_certificate_records": True,
        "held_out_residual_signal_demonstrated": False,
    }
    assert gate["posterior_ready"] is False
    assert "model" not in gate
    assert "scalar_reward" not in gate
    assert (tmp_path / "summary" / "final_report.md").is_file()
    assert json.loads(
        (tmp_path / "summary" / "posterior_gate.json").read_text()
    ) == gate
    report = (tmp_path / "summary" / "final_report.md").read_text()
    assert "Generation wall time: 3.000000 s" in report
    assert "Candidate-count arm total FE" in report
    assert "Bias-step arm total FE" in report
