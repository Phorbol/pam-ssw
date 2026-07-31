from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-feature-learnability-gate"
    / "protocol.py"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_direction_feature_learnability_protocol_test",
        PROTOCOL_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_repeat(protocol, *, repeat_shift: float = 0.0):
    rows = []
    softness = (3.0, 0.0, 2.0, 1.0)
    intent = (0.0, 4.0, 3.0, 1.0)
    quality = tuple(
        soft + anchor
        for soft, anchor in zip(softness, intent)
    )
    for system in protocol.SYSTEMS:
        for state_id in protocol.STATE_IDS:
            for seed in protocol.SEEDS:
                for candidate_index in protocol.CANDIDATE_INDICES:
                    rows.append(
                        {
                            "system": system,
                            "state_id": state_id,
                            "seed": seed,
                            "candidate_index": candidate_index,
                            "direction_sha256": (
                                f"{system}-{state_id}-{seed}-"
                                f"{candidate_index}"
                            ),
                            "static_rank": candidate_index + 1,
                            "static_score": float(
                                -candidate_index + repeat_shift
                            ),
                            "true_curvature": float(
                                -softness[candidate_index]
                                + repeat_shift
                            ),
                            "kind": "bond",
                            "landing_delta_eV": float(
                                -quality[candidate_index]
                                + repeat_shift
                            ),
                            "certificate": True,
                            "landing_geometry_valid": True,
                            "direction_trace": [
                                {
                                    "selected_to_anchor_abs_cosine": (
                                        intent[candidate_index]
                                    )
                                }
                            ],
                        }
                    )
    return rows


def test_repeat_average_produces_twelve_complete_k4_groups() -> None:
    protocol = _load_protocol()
    rows = protocol.build_dataset(
        _synthetic_repeat(protocol),
        _synthetic_repeat(protocol, repeat_shift=0.2),
    )

    assert len(rows) == 48
    assert len({row["group_id"] for row in rows}) == 12
    assert {
        row["candidate_index"]
        for row in rows
        if row["group_id"] == ("c60", "intermediate_accepted", 42)
    } == {0, 1, 2, 3}
    first = next(
        row
        for row in rows
        if row["group_id"] == ("c60", "intermediate_accepted", 42)
        and row["candidate_index"] == 0
    )
    assert first["landing_delta_eV"] == -2.9


def test_group_context_and_system_folds_are_disjoint() -> None:
    protocol = _load_protocol()
    rows = protocol.build_dataset(
        _synthetic_repeat(protocol),
        _synthetic_repeat(protocol),
    )

    expected_folds = {
        "group": 12,
        "context": 4,
        "system": 2,
    }
    for mode, count in expected_folds.items():
        folds = protocol.make_folds(rows, mode)
        assert len(folds) == count
        for fold in folds:
            assert set(fold["train_group_ids"]).isdisjoint(
                fold["test_group_ids"]
            )
            assert fold["train_group_ids"]
            assert fold["test_group_ids"]


def test_combined_fixed_ridge_recovers_complementary_signal() -> None:
    protocol = _load_protocol()
    rows = protocol.build_dataset(
        _synthetic_repeat(protocol),
        _synthetic_repeat(protocol),
    )

    result = protocol.cross_validate(rows, "system")

    assert result["combined"]["top1_accuracy"] == 1.0
    assert result["combined"]["mean_regret_eV"] == 0.0
    assert result["combined"]["mean_regret_eV"] < result["softness"][
        "mean_regret_eV"
    ]
    assert result["combined"]["mean_regret_eV"] < result["intent"][
        "mean_regret_eV"
    ]


def test_posterior_gate_rejects_failure_in_either_held_out_system() -> None:
    protocol = _load_protocol()
    passing = {
        "combined": {
            "mean_regret_eV": 0.1,
            "by_system": {
                "c60": {
                    "top1_accuracy": 4 / 6,
                    "median_regret_eV": 0.0,
                },
                "pdo": {
                    "top1_accuracy": 4 / 6,
                    "median_regret_eV": 0.0,
                },
            },
        },
        "softness": {"mean_regret_eV": 0.2},
        "intent": {"mean_regret_eV": 0.3},
        "static_score": {"mean_regret_eV": 0.1},
    }

    assert protocol.posterior_stage_allowed(passing) is True

    passing["combined"]["by_system"]["pdo"]["top1_accuracy"] = 3 / 6
    assert protocol.posterior_stage_allowed(passing) is False
