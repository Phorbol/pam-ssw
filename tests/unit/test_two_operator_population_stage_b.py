from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.config import LSSSWConfig, SSWConfig
from pamssw.potentials import DoubleWell2D
from pamssw.result import RelaxResult, UphillStepRecord, UphillWalkTrace
from pamssw.state import State
from pamssw.walker import GeometryValidator


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "runs/20260803-two-operator-population-gate/run_stage_b.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("_two_operator_stage_b", PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_positions_are_exact_sigma_direction_displacement():
    runner = load_runner()
    positions = np.zeros((2, 3))
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    observed = runner.direct_positions(positions, direction, 0.2)
    np.testing.assert_allclose(observed.reshape(-1), 0.2 * direction)


def test_starter_artifacts_are_frozen_for_all_systems():
    runner = load_runner()
    specs = runner.starter_artifacts()
    assert len(specs) == 6
    assert {(item.system, item.context) for item in specs} == {
        (system, context)
        for system in ("c60", "pdo", "cuo")
        for context in ("bootstrap", "h8_best")
    }
    assert all(item.path.is_file() for item in specs)
    assert all(runner.file_sha256(item.path) == item.sha256 for item in specs)


def test_locked_starter_restores_the_original_constraint_mask():
    runner = load_runner()
    c60 = runner.load_locked_starter("c60", "bootstrap", cuo_resources=None)
    pdo = runner.load_locked_starter("pdo", "bootstrap", cuo_resources=None)
    assert int(c60.fixed_mask.sum()) == 0
    assert int(pdo.fixed_mask.sum()) == 40
    assert pdo.pbc == (True, True, False)


def test_build_config_preserves_frozen_per_system_direction_width(tmp_path):
    runner = load_runner()
    observed = {
        system: runner.build_config(system, tmp_path / system, seed=55)
        for system in ("c60", "pdo", "cuo")
    }
    assert {system: cfg.oracle_candidates for system, cfg in observed.items()} == {
        "c60": 4,
        "pdo": 8,
        "cuo": 8,
    }
    assert all(cfg.max_steps_per_walk == 8 for cfg in observed.values())
    assert all(cfg.local_softening_scope == "oracle" for cfg in observed.values())


def test_split_shared_direction_cost_does_not_duplicate_hvp_work():
    runner = load_runner()
    pair = runner.split_pair_cost(
        shared_direction_fe=8,
        direct_counts={"landing_true_quench": 30, "unattributed": 0},
        ssw_counts={
            "direction_oracle": 24,
            "biased_proposal_relax": 90,
            "landing_true_quench": 40,
            "unattributed": 0,
        },
    )
    assert pair["shared_direction_force_evaluations"] == 8
    assert pair["direct_force_evaluations"] == 30
    assert pair["ssw_force_evaluations"] == 146
    assert pair["pair_force_evaluations"] == 184


def test_pair_cap_reserves_the_complete_fixed_cohort():
    runner = load_runner()
    assert runner.protocol.PAIR_SUBMISSION_CAP == 3_300
    assert (
        len(runner.protocol.case_matrix()) * runner.protocol.PAIR_SUBMISSION_CAP
        <= runner.protocol.MAX_FORCE_EVALUATIONS
    )


def test_counts_delta_preserves_purpose_identity():
    runner = load_runner()
    before = EvaluationCounts.from_mapping(
        {EvaluationPurpose.STARTER_TRUE_QUENCH: 1}
    )
    after = before + EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.DIRECTION_ORACLE: 8,
            EvaluationPurpose.BIASED_PROPOSAL_RELAX: 12,
        }
    )
    delta = runner.counts_delta(after, before)
    assert delta["starter_true_quench"] == 0
    assert delta["direction_oracle"] == 8
    assert delta["biased_proposal_relax"] == 12
    assert sum(delta.values()) == 20


def test_serialize_pair_charges_shared_direction_once_globally():
    runner = load_runner()
    starter = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
    )
    landing = RelaxResult(
        state=starter.with_flat_positions(
            np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]).reshape(-1)
        ),
        energy=-1.0,
        gradient_norm=0.0,
        n_iter=2,
    )
    config = SSWConfig(quench_fmax=0.1, dedup_rmsd_tol=0.01)

    class Walker:
        geometry_validator = GeometryValidator()

        @staticmethod
        def _is_fragmented_cluster(_starter, _landing):
            return False

    step = UphillStepRecord(
        step_index=0,
        direction_kind="random",
        target_eV=0.8,
        true_energy_before_eV=0.0,
        true_energy_after_eV=0.1,
        requested_sigma=0.2,
        executed_sigma=0.2,
        base_bias_weight=1.0,
        final_bias_weight=1.0,
        true_curvature=1.0,
        inner_curvature=1.0,
        proposal_relax_iterations=1,
        proposal_relax_outcome="useful_progress",
        proposal_relax_termination="maxiter",
        direction_oracle_force_evaluations=8,
        biased_relax_force_evaluations=12,
        true_pes_check_force_evaluations=1,
        displacement_clipped=False,
        step_termination_reason="continue",
    )
    trace = UphillWalkTrace(target_eV=0.8, termination_reason="reached_step_cap", steps=(step,))
    artifact = runner.starter_artifacts()[0]
    pair = runner.serialize_pair(
        system="c60",
        starter_context="bootstrap",
        seed=55,
        starter_state=starter,
        starter_energy=0.0,
        direction=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0),
        sigma=0.2,
        ssw_trace=trace,
        ssw_landing=landing,
        ssw_walker=Walker(),
        ssw_counts={
            "direction_oracle": 8,
            "biased_proposal_relax": 12,
            "escape_true_pes_check": 1,
            "landing_true_quench": 9,
            "post_relax_validation": 1,
            "unattributed": 0,
        },
        direct_landing=landing,
        direct_walker=Walker(),
        direct_counts={"landing_true_quench": 5, "unattributed": 0},
        config=config,
        starter_artifact=artifact,
        starter_validation_counts={"starter_true_quench": 1, "unattributed": 0},
        starter_validation_wall_time_s=0.1,
        initial_direction_wall_time_s=0.2,
        ssw_wall_time_s=1.2,
        direct_wall_time_s=0.4,
    )
    direct, ssw = pair["actions"]
    assert direct["force_evaluations"] == 5
    assert ssw["force_evaluations"] == 23
    assert direct["fully_loaded_force_evaluations"] == 13
    assert ssw["fully_loaded_force_evaluations"] == 31
    assert pair["pair_force_evaluations"] == 37
    assert pair["pair_wall_time_s"] == pytest.approx(1.7)


def test_run_pair_executes_two_arms_from_one_analytic_action_input(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    starter = State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]]))
    config = LSSSWConfig(
        max_trials=1,
        max_steps_per_walk=1,
        target_uphill_energy=0.2,
        quench_fmax=1.0e-3,
        quench_maxiter=80,
        quench_optimizer="scipy-lbfgsb",
        oracle_candidates=1,
        proposal_relax_steps=4,
        proposal_fmax=0.05,
        proposal_optimizer="scipy-lbfgsb",
        rng_seed=55,
        max_force_evals=400,
        proposal_pool_size=1,
        local_softening_scope="oracle",
        local_softening_protocol="moving_reference",
    )
    monkeypatch.setattr(
        runner,
        "load_locked_starter",
        lambda system, context, cuo_resources: starter,
    )
    monkeypatch.setattr(
        runner,
        "make_calculator",
        lambda system, cuo_resources: AnalyticCalculator(DoubleWell2D()),
    )
    monkeypatch.setattr(
        runner,
        "build_config",
        lambda system, case_directory, seed: config,
    )
    pair = runner.run_pair(
        system="c60",
        starter_context="bootstrap",
        seed=55,
        output_directory=tmp_path,
        cuo_resources=None,
    )
    assert len(pair["actions"]) == 2
    assert {row["operator_family"] for row in pair["actions"]} == {"direct", "ssw"}
    assert pair["direction_sha256"] is not None
    assert pair["shared_direction_force_evaluations"] > 0
    assert pair["pair_force_evaluations"] == (
        sum(pair["starter_validation_purpose_counts"].values())
        + pair["shared_direction_force_evaluations"]
        + sum(row["force_evaluations"] for row in pair["actions"])
    )
    assert all(row["purpose_counts"].get("unattributed", 0) == 0 for row in pair["actions"])
    assert (tmp_path / "pair.json").is_file()


def test_selected_cases_preserve_preregistered_order():
    runner = load_runner()
    selected = runner.selected_cases(
        systems=("c60", "cuo"),
        starters=("h8_best",),
        seeds=(55, 57),
    )
    assert [
        (item.system, item.starter_context, item.seed)
        for item in selected
    ] == [
        ("c60", "h8_best", 55),
        ("c60", "h8_best", 57),
        ("cuo", "h8_best", 55),
        ("cuo", "h8_best", 57),
    ]


def test_validate_pair_record_rejects_broken_global_ledger():
    runner = load_runner()
    actions = []
    for family in ("direct", "ssw"):
        actions.append(
            {
                "system": "c60",
                "starter_context": "bootstrap",
                "seed": 55,
                "operator_family": family,
                "certified": False,
                "same_starter_basin": False,
                "geometry_valid": False,
                "fragmented": False,
                "budget_censored": True,
                "landing_delta_eV": None,
                "improved_global_best": False,
                "force_evaluations": 10,
                "fully_loaded_force_evaluations": 18,
                "wall_time_s": 1.0,
                "fully_loaded_wall_time_s": 1.2,
                "purpose_counts": {"landing_true_quench": 10, "unattributed": 0},
            }
        )
    pair = {
        "schema_version": 1,
        "system": "c60",
        "starter_context": "bootstrap",
        "seed": 55,
        "starter_sha256": runner.starter_artifacts()[0].sha256,
        "fixed_atom_count": 0,
        "direction_sha256": "a" * 64,
        "execution_sigma": 0.2,
        "shared_direction_force_evaluations": 8,
        "starter_validation_purpose_counts": {
            "starter_true_quench": 1,
            "unattributed": 0,
        },
        "actions": actions,
        "pair_force_evaluations": 30,
    }
    with pytest.raises(ValueError, match="pair force ledger"):
        runner.validate_pair_record(pair)


def test_campaign_resumes_only_validated_pair_records(tmp_path, monkeypatch):
    runner = load_runner()
    calls = []

    def fake_run_pair(**kwargs):
        calls.append((kwargs["system"], kwargs["starter_context"], kwargs["seed"]))
        actions = []
        for family in ("direct", "ssw"):
            actions.append(
                {
                    "system": "c60",
                    "starter_context": "bootstrap",
                    "seed": 55,
                    "operator_family": family,
                    "certified": False,
                    "same_starter_basin": False,
                    "geometry_valid": False,
                    "fragmented": False,
                    "budget_censored": True,
                    "landing_delta_eV": None,
                    "improved_global_best": False,
                    "force_evaluations": 10,
                    "fully_loaded_force_evaluations": 18,
                    "wall_time_s": 1.0,
                    "fully_loaded_wall_time_s": 1.2,
                    "purpose_counts": {
                        "landing_true_quench": 10,
                        "unattributed": 0,
                    },
                }
            )
        pair = {
            "schema_version": 1,
            "system": "c60",
            "starter_context": "bootstrap",
            "seed": 55,
            "starter_sha256": runner.starter_artifacts()[0].sha256,
            "fixed_atom_count": 0,
            "direction_sha256": "a" * 64,
            "execution_sigma": 0.2,
            "shared_direction_force_evaluations": 8,
            "starter_validation_purpose_counts": {
                "starter_true_quench": 1,
                "unattributed": 0,
            },
            "actions": actions,
            "pair_force_evaluations": 29,
        }
        runner.write_json(Path(kwargs["output_directory"]) / "pair.json", pair)
        return pair

    monkeypatch.setattr(runner, "run_pair", fake_run_pair)
    first = runner.run_campaign(
        output_directory=tmp_path,
        systems=("c60",),
        starters=("bootstrap",),
        seeds=(55,),
        max_force_evaluations=3_300,
        cuo_resources=None,
    )
    second = runner.run_campaign(
        output_directory=tmp_path,
        systems=("c60",),
        starters=("bootstrap",),
        seeds=(55,),
        max_force_evaluations=3_300,
        cuo_resources=None,
    )
    assert calls == [("c60", "bootstrap", 55)]
    assert first["force_evaluations"] == second["force_evaluations"] == 29
    assert first["completed_pairs"] == second["completed_pairs"] == 1


def test_cli_parser_accepts_one_pair_smoke_selection(tmp_path):
    runner = load_runner()
    args = runner.parse_args(
        [
            "--output",
            str(tmp_path),
            "--systems",
            "c60",
            "--starters",
            "bootstrap",
            "--seeds",
            "55",
            "--max-force-evaluations",
            "4000",
        ]
    )
    assert args.output == tmp_path
    assert args.systems == ["c60"]
    assert args.starters == ["bootstrap"]
    assert args.seeds == [55]
    assert args.max_force_evaluations == 4000


def test_stage_b_requires_stage_a_admission(tmp_path):
    runner = load_runner()
    path = tmp_path / "stage-a.json"
    path.write_text('{"decision": "CLOSE_TWO_OPERATOR_PORTFOLIO"}\n')
    with pytest.raises(RuntimeError, match="did not admit"):
        runner.require_stage_a_admission(path)
