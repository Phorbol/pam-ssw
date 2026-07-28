from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    REPO_ROOT
    / "runs"
    / "20260728-direction-conditioned-checkpoint-shooting"
    / "run_audit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_direction_conditioned_checkpoint_shooting",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_matrix_is_the_preregistered_twelve_cases():
    module = _load_module()

    matrix = module.case_matrix()

    assert len(matrix) == 12
    assert matrix[0] == {
        "state_id": "intermediate_accepted",
        "seed": 42,
        "arm": "balanced_refinement",
    }
    assert matrix[-1] == {
        "state_id": "plateau_accepted",
        "seed": 44,
        "arm": "deep_refinement",
    }
    assert len(
        {
            (row["state_id"], row["seed"], row["arm"])
            for row in matrix
        }
    ) == 12


@pytest.mark.parametrize(
    ("productive", "expected"),
    [
        ([False, True], "productive_final"),
        ([True, False], "overshoot"),
        ([True, True], "productive_earlier_and_final"),
        ([False, False], "no_productive_checkpoint"),
    ],
)
def test_classification_uses_checkpoint_order(productive, expected):
    module = _load_module()
    checkpoints = [
        {"step_index": index, "productive": value}
        for index, value in enumerate(productive, start=1)
    ]

    assert module.classify_trajectory(checkpoints) == expected


def test_classification_rejects_empty_or_nonconsecutive_checkpoints():
    module = _load_module()

    with pytest.raises(ValueError, match="at least one"):
        module.classify_trajectory([])

    with pytest.raises(ValueError, match="consecutive"):
        module.classify_trajectory(
            [
                {"step_index": 1, "productive": False},
                {"step_index": 3, "productive": True},
            ]
        )


def _evidence_row(case):
    checkpoints = []
    for step_index, productive in ((1, True), (2, False)):
        checkpoints.append(
            {
                "step_index": step_index,
                "status": "completed",
                "certificate": True,
                "is_new_basin": productive,
                "landing_delta_eV": -1.0 if productive else 0.5,
                "productive": productive,
                "force_evaluations": 22,
                "purpose_counts": {
                    "direction_oracle": 0,
                    "biased_proposal_relax": 0,
                    "landing_true_quench": 20,
                    "escape_true_pes_check": 1,
                    "post_relax_validation": 1,
                    "starter_true_quench": 0,
                    "bootstrap_true_quench": 0,
                    "unattributed": 0,
                },
            }
        )
    return {
        **case,
        "status": "completed",
        "generation_force_evaluations": 77,
        "generation_purpose_counts": {
            "direction_oracle": 24,
            "biased_proposal_relax": 50,
            "landing_true_quench": 0,
            "escape_true_pes_check": 3,
            "post_relax_validation": 0,
            "starter_true_quench": 0,
            "bootstrap_true_quench": 0,
            "unattributed": 0,
        },
        "direction_selection_count": 1,
        "checkpoints": checkpoints,
        "classification": "overshoot",
    }


def test_evidence_closes_generation_and_checkpoint_ledgers():
    module = _load_module()
    rows = [_evidence_row(case) for case in module.case_matrix()]

    evidence = module.build_evidence(rows)

    assert evidence["cohort"]["completed_cases"] == 12
    assert evidence["cohort"]["checkpoint_count"] == 24
    assert evidence["classification_counts"]["overshoot"] == 12
    assert evidence["certificate_count"] == 24
    assert evidence["meaningful_checkpoint_count"] == 12
    assert evidence["production_default_changed"] is False
    assert evidence["meaningful_energy_drop_threshold_eV"] == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda rows: rows[0].update(generation_force_evaluations=76),
            "generation purpose ledger",
        ),
        (
            lambda rows: rows[0]["checkpoints"][0].update(force_evaluations=21),
            "checkpoint purpose ledger",
        ),
        (
            lambda rows: rows[0].update(direction_selection_count=2),
            "direction ledger",
        ),
        (
            lambda rows: rows[0]["checkpoints"][1].update(step_index=3),
            "consecutive",
        ),
    ],
)
def test_evidence_rejects_unclosed_or_misordered_rows(mutation, match):
    module = _load_module()
    rows = [_evidence_row(case) for case in module.case_matrix()]
    corrupted = deepcopy(rows)
    mutation(corrupted)

    with pytest.raises(ValueError, match=match):
        module.build_evidence(corrupted)
