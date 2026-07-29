from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-fixed-direction-family-terminal-labels"
    / "protocol.py"
)


def load_protocol(name: str = "_fixed_direction_family_protocol"):
    spec = importlib.util.spec_from_file_location(name, PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows(module, *, arm_signal: bool) -> list[dict]:
    rows = []
    for case in module.case_matrix():
        if arm_signal:
            meaningful = (
                case.arm == "bond_only"
                and case.state_id == "plateau_accepted"
            )
        else:
            meaningful = case.state_id == "plateau_accepted"
        expected_kind = module.ARM_SETTINGS[case.arm].expected_kind
        selections = 8
        direction_fe = 2 * 4 * selections
        purposes = {
            "bootstrap_true_quench": 0,
            "starter_true_quench": 0,
            "direction_oracle": direction_fe,
            "biased_proposal_relax": 100,
            "escape_true_pes_check": 2,
            "landing_true_quench": 20,
            "post_relax_validation": 1,
            "unattributed": 0,
        }
        rows.append(
            {
                "status": "completed",
                "stage": "fixed_direction_family",
                "state_id": case.state_id,
                "seed": case.seed,
                "arm": case.arm,
                "repeat": case.repeat,
                "settings": {
                    "enable_momentum_candidate": False,
                    "oracle_candidates": 4,
                    "max_steps_per_walk": 8,
                    "proposal_relax_steps": 80,
                    "n_bond_pairs": (
                        module.ARM_SETTINGS[case.arm].n_bond_pairs
                    ),
                    "stagnation_bond_pair_boost": 0,
                    "expected_kind": expected_kind,
                },
                "selection_probability": 1.0,
                "exact_starter_reference": True,
                "certificate": True,
                "is_new_basin": meaningful,
                "meaningful": meaningful,
                "landing_delta_eV": -1.0 if meaningful else 1.0,
                "force_evaluations": sum(purposes.values()),
                "purpose_counts": purposes,
                "direction_trace_valid": True,
                "direction_audit": {
                    "selection_count": selections,
                    "candidate_count": 4 * selections,
                    "candidate_kind_counts": {
                        expected_kind: 4 * selections,
                    },
                    "selected_kind_counts": {
                        expected_kind: selections,
                    },
                    "direction_oracle_force_evaluations": direction_fe,
                },
                "fragmented": False,
                "fallback_used": False,
                "generation_wall_time_s": 1.0,
                "quench_wall_time_s": 0.5,
            }
        )
    return rows


def test_case_matrix_is_paired_and_has_only_one_direction_family_per_arm():
    module = load_protocol()

    cases = module.case_matrix()

    assert len(cases) == 24
    assert cases[0].arm == "random_only"
    assert cases[1].arm == "bond_only"
    assert [
        case.arm
        for case in cases
        if case.state_id == "intermediate_accepted"
        and case.seed == 42
        and case.repeat == 1
    ] == ["bond_only", "random_only"]
    assert module.ARM_SETTINGS["random_only"].n_bond_pairs == 0
    assert module.ARM_SETTINGS["random_only"].expected_kind == "random"
    assert module.ARM_SETTINGS["bond_only"].n_bond_pairs == 4
    assert module.ARM_SETTINGS["bond_only"].expected_kind == "bond"
    assert all(
        case.settings.stagnation_bond_pair_boost == 0
        for case in cases
    )


def test_stable_labels_require_both_repeats_to_be_meaningful():
    module = load_protocol()
    rows = _rows(module, arm_signal=False)
    for row in rows:
        if (
            row["state_id"] == "plateau_accepted"
            and row["seed"] == 42
            and row["arm"] == "bond_only"
            and row["repeat"] == 1
        ):
            row["meaningful"] = False
            row["is_new_basin"] = False
            row["landing_delta_eV"] = 1.0

    labels = module.stable_labels(rows)

    label = next(
        item
        for item in labels
        if item["state_id"] == "plateau_accepted"
        and item["seed"] == 42
        and item["arm"] == "bond_only"
    )
    assert label["meaningful"] is False


def test_action_conditioned_gate_requires_held_out_gain_not_only_labels():
    module = load_protocol()

    signal = module.build_evidence(_rows(module, arm_signal=True))
    no_signal = module.build_evidence(_rows(module, arm_signal=False))

    assert signal["posterior_gate"]["held_out_residual_signal"] is True
    assert signal["posterior_gate"]["enter_posterior_stage"] is False
    assert signal["posterior_gate"]["reason"] == (
        "insufficient_stable_meaningful_labels_per_family"
    )
    assert no_signal["posterior_gate"]["held_out_residual_signal"] is False
    assert no_signal["posterior_gate"]["enter_posterior_stage"] is False


def test_evidence_fails_closed_on_incomplete_or_mixed_family_rows():
    module = load_protocol()
    rows = _rows(module, arm_signal=False)

    with pytest.raises(ValueError, match="exact cohort"):
        module.build_evidence(rows[:-1])

    rows[0]["direction_audit"]["candidate_kind_counts"] = {
        "bond": 1,
        "random": 31,
    }
    with pytest.raises(ValueError, match="single expected family"):
        module.build_evidence(rows)
