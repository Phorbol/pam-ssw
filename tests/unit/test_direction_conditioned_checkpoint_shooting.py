from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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


def test_checkpoint_discovery_returns_consecutive_macro_steps(tmp_path):
    module = _load_module()
    expected = []
    for step_index in (1, 2, 3):
        path = (
            tmp_path
            / (
                "trial0001_proposal001_"
                f"step{step_index:03d}_proposal_relax.xyz"
            )
        )
        path.write_text("", encoding="utf-8")
        expected.append(path)

    assert module.discover_checkpoint_paths(tmp_path) == expected


@pytest.mark.parametrize("failure", ["empty", "gap", "unexpected"])
def test_checkpoint_discovery_rejects_incomplete_or_wrong_files(
    tmp_path,
    failure,
):
    module = _load_module()
    if failure == "gap":
        for step_index in (1, 3):
            (
                tmp_path
                / (
                    "trial0001_proposal001_"
                    f"step{step_index:03d}_proposal_relax.xyz"
                )
            ).write_text("", encoding="utf-8")
    elif failure == "unexpected":
        (
            tmp_path
            / "trial0002_proposal001_step001_proposal_relax.xyz"
        ).write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        module.discover_checkpoint_paths(tmp_path)


def test_effective_checkpoint_delegates_to_core_walk_clip():
    module = _load_module()
    starter = object()
    raw_checkpoint = object()
    effective = object()
    calls = []

    def core_clipper(reference, candidate, max_displacement):
        calls.append((reference, candidate, max_displacement))
        return effective, True

    result, clipped = module.effective_checkpoint_state(
        starter,
        raw_checkpoint,
        max_displacement=5.0,
        _clipper=core_clipper,
    )

    assert clipped is True
    assert result is effective
    assert calls == [(starter, raw_checkpoint, 5.0)]


def test_accepted_checkpoint_prefix_stops_at_the_proposal_endpoint():
    module = _load_module()
    checkpoints = [
        SimpleNamespace(positions=[[1.0, 0.0, 0.0]]),
        SimpleNamespace(positions=[[2.0, 0.0, 0.0]]),
        SimpleNamespace(positions=[[3.0, 0.0, 0.0]]),
    ]
    endpoint = SimpleNamespace(positions=[[2.0, 0.0, 0.0]])

    accepted, errors = module.accepted_checkpoint_prefix(
        checkpoints,
        endpoint,
        tolerance=1.0e-8,
    )

    assert accepted == checkpoints[:2]
    assert errors == pytest.approx([1.0, 0.0, 1.0])
