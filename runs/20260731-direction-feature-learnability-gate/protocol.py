"""Offline held-out learnability gate for already-paid direction features."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


SYSTEMS = ("c60", "pdo")
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
CANDIDATE_INDICES = (0, 1, 2, 3)
MODEL_FEATURES = {
    "softness": ("softness",),
    "intent": ("intent_overlap", "random_kind"),
    "combined": ("softness", "intent_overlap", "random_kind"),
}


def _case_key(
    row: Mapping[str, Any],
) -> tuple[str, str, int, int]:
    return (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        int(row["candidate_index"]),
    )


def _expected_keys() -> set[tuple[str, str, int, int]]:
    return {
        (system, state_id, seed, candidate_index)
        for system in SYSTEMS
        for state_id in STATE_IDS
        for seed in SEEDS
        for candidate_index in CANDIDATE_INDICES
    }


def _mean(left: float, right: float) -> float:
    return 0.5 * (float(left) + float(right))


def _group_standardize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    centered = array - float(np.mean(array))
    scale = float(np.sqrt(np.mean(centered * centered)))
    if scale <= 1.0e-12:
        return np.zeros_like(array)
    return centered / scale


def build_dataset(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    first = {_case_key(row): row for row in first_rows}
    second = {_case_key(row): row for row in second_rows}
    expected = _expected_keys()
    if set(first) != expected or set(second) != expected:
        raise ValueError("counterfactual repeats must contain the exact K4 matrix")
    rows: list[dict[str, Any]] = []
    for key in sorted(expected):
        left = first[key]
        right = second[key]
        if (
            not bool(left["certificate"])
            or not bool(right["certificate"])
            or not bool(left["landing_geometry_valid"])
            or not bool(right["landing_geometry_valid"])
        ):
            raise ValueError("counterfactual label lacks a valid force certificate")
        if left["direction_sha256"] != right["direction_sha256"]:
            raise ValueError("direction identity differs across repeats")
        if int(left["static_rank"]) != int(right["static_rank"]):
            raise ValueError("static rank differs across repeats")
        left_trace = left["direction_trace"][0]
        right_trace = right["direction_trace"][0]
        anchor_left = float(
            left_trace["selected_to_anchor_abs_cosine"]
        )
        anchor_right = float(
            right_trace["selected_to_anchor_abs_cosine"]
        )
        rows.append(
            {
                "system": key[0],
                "state_id": key[1],
                "seed": key[2],
                "candidate_index": key[3],
                "group_id": key[:3],
                "direction_sha256": left["direction_sha256"],
                "static_rank": int(left["static_rank"]),
                "static_score": _mean(
                    left["static_score"],
                    right["static_score"],
                ),
                "true_curvature": _mean(
                    left["true_curvature"],
                    right["true_curvature"],
                ),
                "anchor_abs_cosine": _mean(
                    anchor_left,
                    anchor_right,
                ),
                "kind": str(left["kind"]),
                "landing_delta_eV": _mean(
                    left["landing_delta_eV"],
                    right["landing_delta_eV"],
                ),
            }
        )

    grouped: dict[
        tuple[str, str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    for group_id, group in grouped.items():
        group.sort(key=lambda row: row["candidate_index"])
        if [row["candidate_index"] for row in group] != list(
            CANDIDATE_INDICES
        ):
            raise ValueError(f"incomplete K4 group: {group_id}")
        softness = _group_standardize(
            [-float(row["true_curvature"]) for row in group]
        )
        intent = _group_standardize(
            [float(row["anchor_abs_cosine"]) for row in group]
        )
        random_kind = _group_standardize(
            [float(row["kind"] == "random") for row in group]
        )
        static_score = _group_standardize(
            [float(row["static_score"]) for row in group]
        )
        quality = np.asarray(
            [-float(row["landing_delta_eV"]) for row in group],
            dtype=float,
        )
        quality -= float(np.mean(quality))
        for index, row in enumerate(group):
            row["features"] = {
                "softness": float(softness[index]),
                "intent_overlap": float(intent[index]),
                "random_kind": float(random_kind[index]),
                "static_score": float(static_score[index]),
            }
            row["target_quality"] = float(quality[index])
    return sorted(
        rows,
        key=lambda row: (
            row["system"],
            row["state_id"],
            row["seed"],
            row["candidate_index"],
        ),
    )


def make_folds(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    group_ids = sorted({tuple(row["group_id"]) for row in rows})
    if mode == "group":
        held_out_sets = [
            {group_id}
            for group_id in group_ids
        ]
    elif mode == "context":
        contexts = sorted({group_id[:2] for group_id in group_ids})
        held_out_sets = [
            {
                group_id
                for group_id in group_ids
                if group_id[:2] == context
            }
            for context in contexts
        ]
    elif mode == "system":
        held_out_sets = [
            {
                group_id
                for group_id in group_ids
                if group_id[0] == system
            }
            for system in SYSTEMS
        ]
    else:
        raise ValueError("fold mode must be group, context, or system")
    all_groups = set(group_ids)
    return [
        {
            "fold_id": (mode, index),
            "train_group_ids": tuple(
                sorted(all_groups - held_out)
            ),
            "test_group_ids": tuple(sorted(held_out)),
        }
        for index, held_out in enumerate(held_out_sets)
    ]


def _fit_ridge(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(
        [
            [float(row["features"][name]) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    target = np.asarray(
        [float(row["target_quality"]) for row in rows],
        dtype=float,
    )
    penalty = np.eye(matrix.shape[1], dtype=float)
    return np.linalg.solve(
        matrix.T @ matrix + penalty,
        matrix.T @ target,
    )


def _summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    regrets = [float(row["regret_eV"]) for row in predictions]
    result = {
        "group_count": len(predictions),
        "top1_accuracy": float(
            sum(bool(row["top1_correct"]) for row in predictions)
            / len(predictions)
        ),
        "mean_regret_eV": float(sum(regrets) / len(regrets)),
        "median_regret_eV": float(median(regrets)),
        "by_system": {},
    }
    for system in SYSTEMS:
        system_rows = [
            row for row in predictions if row["system"] == system
        ]
        system_regrets = [
            float(row["regret_eV"]) for row in system_rows
        ]
        result["by_system"][system] = {
            "group_count": len(system_rows),
            "top1_accuracy": float(
                sum(bool(row["top1_correct"]) for row in system_rows)
                / len(system_rows)
            ),
            "mean_regret_eV": float(
                sum(system_regrets) / len(system_regrets)
            ),
            "median_regret_eV": float(median(system_regrets)),
        }
    return result


def cross_validate(
    rows: Sequence[Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    folds = make_folds(rows, mode)
    grouped: dict[
        tuple[str, str, int], list[Mapping[str, Any]]
    ] = defaultdict(list)
    for row in rows:
        grouped[tuple(row["group_id"])].append(row)
    predictions: dict[str, list[dict[str, Any]]] = {
        "static_score": [],
        **{name: [] for name in MODEL_FEATURES},
    }
    for fold in folds:
        training = [
            row
            for row in rows
            if tuple(row["group_id"]) in fold["train_group_ids"]
        ]
        coefficients = {
            model: _fit_ridge(training, feature_names)
            for model, feature_names in MODEL_FEATURES.items()
        }
        for group_id in fold["test_group_ids"]:
            group = sorted(
                grouped[group_id],
                key=lambda row: int(row["candidate_index"]),
            )
            best = min(
                group,
                key=lambda row: (
                    float(row["landing_delta_eV"]),
                    int(row["candidate_index"]),
                ),
            )
            score_by_model: dict[str, list[float]] = {
                "static_score": [
                    float(row["features"]["static_score"])
                    for row in group
                ],
            }
            for model, feature_names in MODEL_FEATURES.items():
                beta = coefficients[model]
                score_by_model[model] = [
                    float(
                        np.dot(
                            beta,
                            [
                                float(row["features"][name])
                                for name in feature_names
                            ],
                        )
                    )
                    for row in group
                ]
            for model, scores in score_by_model.items():
                selected_index = max(
                    range(len(group)),
                    key=lambda index: (
                        scores[index],
                        -int(group[index]["candidate_index"]),
                    ),
                )
                selected = group[selected_index]
                predictions[model].append(
                    {
                        "system": group_id[0],
                        "state_id": group_id[1],
                        "seed": group_id[2],
                        "selected_candidate_index": int(
                            selected["candidate_index"]
                        ),
                        "best_candidate_index": int(
                            best["candidate_index"]
                        ),
                        "top1_correct": (
                            selected["candidate_index"]
                            == best["candidate_index"]
                        ),
                        "regret_eV": (
                            float(selected["landing_delta_eV"])
                            - float(best["landing_delta_eV"])
                        ),
                    }
                )
    return {
        model: {
            **_summarize_predictions(model_predictions),
            "predictions": model_predictions,
        }
        for model, model_predictions in predictions.items()
    }


def posterior_stage_allowed(
    leave_system_result: Mapping[str, Any],
) -> bool:
    combined = leave_system_result["combined"]
    return bool(
        all(
            combined["by_system"][system]["top1_accuracy"]
            >= (4.0 / 6.0)
            and combined["by_system"][system]["median_regret_eV"]
            <= 1.0e-12
            for system in SYSTEMS
        )
        and combined["mean_regret_eV"]
        < leave_system_result["softness"]["mean_regret_eV"]
        and combined["mean_regret_eV"]
        < leave_system_result["intent"]["mean_regret_eV"]
        and combined["mean_regret_eV"]
        <= leave_system_result["static_score"]["mean_regret_eV"]
    )


def summarize(
    first_rows: Sequence[Mapping[str, Any]],
    second_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dataset = build_dataset(first_rows, second_rows)
    validation = {
        mode: cross_validate(dataset, mode)
        for mode in ("group", "context", "system")
    }
    return {
        "dataset": {
            "repeat_rows": len(first_rows) + len(second_rows),
            "averaged_candidate_rows": len(dataset),
            "group_count": len(
                {tuple(row["group_id"]) for row in dataset}
            ),
            "feature_names": [
                "true_curvature",
                "anchor_abs_cosine",
                "kind",
                "static_score_baseline",
            ],
            "excluded_redundant_features": ["score_sigma"],
        },
        "validation": validation,
        "posterior_stage_allowed": posterior_stage_allowed(
            validation["system"]
        ),
    }
