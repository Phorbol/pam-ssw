from __future__ import annotations

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
