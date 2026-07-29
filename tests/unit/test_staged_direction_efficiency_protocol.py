from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-staged-direction-efficiency-ablation"
    / "protocol.py"
)


def load_protocol(name: str = "_staged_direction_efficiency_protocol"):
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_complete_rows(module, stage: str):
    rows = []
    for case in module.case_matrix(stage, module.RetainedSettings()):
        rows.append(
            {
                "stage": stage,
                "state_id": case.state_id,
                "seed": case.seed,
                "arm": case.arm,
                "repeat": case.repeat,
                "certificate": True,
                "is_new_basin": True,
                "landing_delta_eV": -1.0,
                "force_evaluations": 100,
                "purpose_counts": {
                    "biased_proposal_relax": 60,
                },
                "optimizer_diagnostics": {
                    "proposal_relax_count": 5,
                    "proposal_relax_termination_maxiter": 1,
                },
            }
        )
    return rows


def test_stage_case_counts_and_alternating_repeat_order():
    module = load_protocol()
    retained = module.RetainedSettings()

    assert len(module.case_matrix("momentum", retained)) == 24
    assert len(module.case_matrix("candidate_count", retained)) == 36
    assert len(module.case_matrix("bias_steps", retained)) == 24
    assert len(module.case_matrix("relax_cap", retained)) == 24

    block = [
        case.arm
        for case in module.case_matrix("momentum", retained)
        if case.state_id == "intermediate_accepted"
        and case.seed == 42
        and case.repeat == 0
    ]
    reversed_block = [
        case.arm
        for case in module.case_matrix("momentum", retained)
        if case.state_id == "intermediate_accepted"
        and case.seed == 42
        and case.repeat == 1
    ]
    assert block == ["momentum_on", "momentum_off"]
    assert reversed_block == list(reversed(block))


def test_repeat_stable_set_requires_both_exact_repeats():
    module = load_protocol()
    rows = make_complete_rows(module, "momentum")
    rows_by_key = {
        (row["state_id"], row["seed"], row["arm"], row["repeat"]): row
        for row in rows
    }
    rows_by_key[
        ("intermediate_accepted", 42, "momentum_on", 1)
    ]["landing_delta_eV"] = 0.1

    stable = module.repeat_stable_meaningful_set(
        rows,
        "momentum_on",
    )

    assert ("intermediate_accepted", 42) not in stable
    assert len(stable) == 5


def test_momentum_decision_requires_stable_set_and_repeat_count_dominance():
    module = load_protocol()
    retained = module.RetainedSettings()
    rows = make_complete_rows(module, "momentum")
    for row in rows:
        if (
            row["arm"] == "momentum_off"
            and row["state_id"] == "plateau_accepted"
            and row["seed"] == 44
        ):
            row["landing_delta_eV"] = 0.1

    decision = module.decide_stage("momentum", rows, retained)

    assert decision["status"] == "positive"
    assert decision["retained_settings"][
        "enable_momentum_candidate"
    ] is True


def test_candidate_count_chooses_smallest_no_loss_lower_cost_arm():
    module = load_protocol()
    retained = module.RetainedSettings()
    rows = make_complete_rows(module, "candidate_count")
    costs = {"k4": 40, "k8": 60, "k12": 100}
    for row in rows:
        row["force_evaluations"] = costs[row["arm"]]

    decision = module.decide_stage("candidate_count", rows, retained)

    assert decision["status"] == "reduced"
    assert decision["retained_settings"]["oracle_candidates"] == 4


def test_relax_cap_requires_both_total_and_proposal_cost_reduction():
    module = load_protocol()
    retained = module.RetainedSettings()
    rows = make_complete_rows(module, "relax_cap")
    for row in rows:
        row["force_evaluations"] = 80 if row["arm"] == "l40" else 100
        row["purpose_counts"]["biased_proposal_relax"] = 60

    equal_proposal = module.decide_stage("relax_cap", rows, retained)

    assert equal_proposal["status"] == "baseline_retained"
    assert equal_proposal["retained_settings"][
        "proposal_relax_steps"
    ] == 80

    for row in rows:
        if row["arm"] == "l40":
            row["purpose_counts"]["biased_proposal_relax"] = 50

    lower_proposal = module.decide_stage("relax_cap", rows, retained)

    assert lower_proposal["status"] == "reduced"
    assert lower_proposal["retained_settings"][
        "proposal_relax_steps"
    ] == 40


def test_stage_l_entry_uses_measured_cost_and_cap_hit_fractions():
    module = load_protocol()
    rows = make_complete_rows(module, "bias_steps")

    gate = module.stage_l_entry(rows, retained_arm="b5")

    assert gate == {
        "entered": True,
        "proposal_relax_force_fraction": 0.6,
        "proposal_relax_cap_hit_fraction": 0.2,
        "proposal_relax_calls": 60,
        "proposal_relax_cap_hits": 12,
    }


def test_stage_l_entry_rejects_zero_physical_work():
    module = load_protocol()
    rows = make_complete_rows(module, "bias_steps")
    for row in rows:
        if row["arm"] == "b5":
            row["force_evaluations"] = 0
            row["optimizer_diagnostics"]["proposal_relax_count"] = 0

    with pytest.raises(ValueError, match="positive total FE"):
        module.stage_l_entry(rows, retained_arm="b5")


def make_evidence_rows(module, stage: str):
    rows = []
    for case in module.case_matrix(stage, module.RetainedSettings()):
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
        rows.append(
            {
                "status": "completed",
                "stage": stage,
                "state_id": case.state_id,
                "seed": case.seed,
                "arm": case.arm,
                "repeat": case.repeat,
                "settings": {
                    "enable_momentum_candidate": (
                        case.settings.enable_momentum_candidate
                    ),
                    "oracle_candidates": (
                        case.settings.oracle_candidates
                    ),
                    "max_steps_per_walk": (
                        case.settings.max_steps_per_walk
                    ),
                    "proposal_relax_steps": (
                        case.settings.proposal_relax_steps
                    ),
                },
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
            }
        )
    return rows


def test_build_evidence_requires_complete_closed_cohort_and_no_selector():
    module = load_protocol()
    retained = module.RetainedSettings()
    rows = make_evidence_rows(module, "momentum")

    evidence = module.build_evidence("momentum", retained, rows)

    assert evidence["cohort"]["completed_cases"] == 24
    assert evidence["decision"]["status"] == "unproven_retained"
    assert evidence["totals"]["unattributed_force_evaluations"] == 0
    assert evidence["totals"]["meaningful_outcomes"] == 24
    assert evidence["meaningful_energy_drop_threshold_eV"] == 0.001
    assert evidence["totals"]["fragmented_outcomes"] == 0
    assert evidence["totals"]["fallback_outcomes"] == 0
    assert evidence["totals"]["generation_wall_time_s"] == 24.0
    assert evidence["totals"]["quench_wall_time_s"] == 12.0
    assert "fixed-starter" in evidence["claim_ceiling"]
    assert evidence["production_default_changed"] is False
    assert "selector" not in evidence
    assert "posterior" not in evidence
    assert "reward" not in evidence

    with pytest.raises(ValueError, match="exact cohort"):
        module.build_evidence("momentum", retained, rows[:-1])


def test_build_bias_evidence_binds_stage_l_entry_to_retained_arm():
    module = load_protocol()
    retained = module.RetainedSettings()
    rows = make_evidence_rows(module, "bias_steps")
    for row in rows:
        if row["arm"] == "b5":
            row["force_evaluations"] -= 20
            row["purpose_counts"]["biased_proposal_relax"] -= 20

    evidence = module.build_evidence("bias_steps", retained, rows)

    assert evidence["decision"]["retained_settings"][
        "max_steps_per_walk"
    ] == 5
    assert evidence["stage_l_entry"]["retained_arm"] == "b5"
    assert evidence["stage_l_entry"]["entered"] is False
