#!/usr/bin/env python3
"""Fail-closed analysis of the fixed 4 x 3 strict-quench rescue replay."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "output"
CORPUS_PATH = RUN_ROOT / "corpus.json"
EXPECTED_ROWS_SHA256 = (
    "57d1e178feef072ccaa20c5e56e017ceba09c82a06bc4b564644476d9c4deb0a"
)
EXPECTED_SUMMARY_SHA256 = (
    "a7532e0117199b87ad8c65a3a2e705244612902e7db97bf2484e9283a6c77d6e"
)
EXPECTED_EXECUTION_COMMIT = "42054840dd58701e89005ef06a6f0eb082cd2603"
EXPECTED_CORPUS_SHA256 = (
    "82b475eb0bcf0a6fb631bff0ee47cdd0223401db0db6262090c6f721d8624e67"
)
EXPECTED_MODEL_SHA256 = (
    "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
)
SYSTEMS = ("c60", "pdo")
STRICT_FMAX = 0.01
MAXITER = 400
OBJECTIVE = "true_mace_pes_no_bias_no_softening"
ARMS = (
    {
        "arm_id": "ase-lbfgs-restart",
        "optimizer": "ase-lbfgs",
        "safe_history_limit": None,
    },
    {
        "arm_id": "safe-lbfgs-total",
        "optimizer": "safe-lbfgs-total",
        "safe_history_limit": 10,
    },
    {
        "arm_id": "ase-fire",
        "optimizer": "ase-fire",
        "safe_history_limit": None,
    },
)
PURPOSES = (
    "bootstrap_true_quench",
    "starter_true_quench",
    "direction_oracle",
    "escape_true_pes_check",
    "biased_proposal_relax",
    "landing_true_quench",
    "post_relax_validation",
    "unattributed",
)
EXPECTED_PRIMARY_COHORT = {
    "overall": {
        "task_count": 32,
        "certificate_success_count": 28,
        "trigger_count": 4,
        "total_force_evaluations": 5096,
        "total_wall_time_s": 98.84687816997757,
    },
    "by_system": {
        "c60": {
            "task_count": 16,
            "certificate_success_count": 14,
            "trigger_count": 2,
            "total_force_evaluations": 2490,
            "total_wall_time_s": 42.58624181896448,
        },
        "pdo": {
            "task_count": 16,
            "certificate_success_count": 14,
            "trigger_count": 2,
            "total_force_evaluations": 2606,
            "total_wall_time_s": 56.26063635101309,
        },
    },
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _display_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(Path(path))


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{label} must be finite")
    return number


def _nonnegative_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _positions_sha256(positions: Any) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("state positions must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("state positions must be finite")
    canonical = np.array(coordinates, dtype=np.dtype("<f8"), order="C", copy=True)
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _validate_state_hash(payload: Any, digest: Any, *, label: str) -> None:
    state = _mapping(payload, label=f"{label} state")
    if _positions_sha256(state.get("positions")) != digest:
        raise ValueError(f"{label} positions hash mismatch")
    cell = state.get("cell")
    if cell is not None and not np.all(np.isfinite(np.asarray(cell, dtype=float))):
        raise ValueError(f"{label} state cell must be finite")


def _validate_primary_cohort(value: Any) -> Mapping[str, Any]:
    cohort = _mapping(value, label="primary cohort")
    if cohort != EXPECTED_PRIMARY_COHORT:
        raise ValueError("primary cohort identity mismatch")
    for field in EXPECTED_PRIMARY_COHORT["overall"]:
        if cohort["overall"][field] != sum(
            cohort["by_system"][system][field] for system in SYSTEMS
        ):
            raise ValueError("primary cohort ledger does not close")
    return cohort


def _validate_corpus(path: Path) -> tuple[Mapping[str, Any], dict[tuple[str, int], Mapping[str, Any]]]:
    if _sha256(path) != EXPECTED_CORPUS_SHA256:
        raise ValueError("corpus SHA mismatch")
    corpus = _mapping(_load_json(path), label="corpus")
    if corpus.get("schema_version") != 1 or corpus.get("entry_count") != 4:
        raise ValueError("corpus identity mismatch")
    source = _mapping(corpus.get("source"), label="corpus source")
    source_model = _mapping(source.get("model"), label="corpus source model")
    if source_model.get("sha256") != EXPECTED_MODEL_SHA256:
        raise ValueError("corpus source model mismatch")
    if (
        source.get("raw_landing_corpus_sha256")
        != "100759e1871cdefb54972f91751c452763f74d0e34ee576f268a5808219a552b"
    ):
        raise ValueError("corpus source replay mismatch")
    selection = _mapping(corpus.get("selection"), label="corpus selection")
    _validate_primary_cohort(selection.get("primary_cohort"))
    if (
        selection.get("primary_arm_id") != "ase-lbfgs"
        or selection.get("strict_fmax_eV_per_A") != STRICT_FMAX
    ):
        raise ValueError("corpus selection mismatch")
    entries = corpus.get("entries")
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError("corpus entry count mismatch")
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry_value in entries:
        entry = _mapping(entry_value, label="corpus entry")
        key = (entry.get("system"), entry.get("task_index"))
        if key in by_key or key[0] not in SYSTEMS or type(key[1]) is not int:
            raise ValueError("corpus entry identity mismatch")
        start = _mapping(entry.get("fallback_start"), label="corpus fallback start")
        _validate_state_hash(
            start.get("state"),
            start.get("positions_sha256"),
            label="corpus fallback start",
        )
        by_key[key] = entry
    if set(by_key) != {("c60", 5), ("c60", 11), ("pdo", 3), ("pdo", 4)}:
        raise ValueError("corpus trigger set mismatch")
    return corpus, by_key


def _validate_summary(
    summary: Any,
    corpus: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    checked = _mapping(summary, label="summary")
    if checked.get("schema_version") != 1:
        raise ValueError("summary schema mismatch")
    if checked.get("execution_commit") != EXPECTED_EXECUTION_COMMIT:
        raise ValueError("execution commit mismatch")
    corpus_identity = _mapping(checked.get("corpus"), label="summary corpus")
    if corpus_identity != {
        "path": "runs/20260728-true-quench-sequential-rescue/corpus.json",
        "sha256": EXPECTED_CORPUS_SHA256,
        "entry_count": 4,
    }:
        raise ValueError("summary corpus identity mismatch")
    if checked.get("source_replay") != corpus["source"]:
        raise ValueError("summary source replay identity mismatch")
    model = _mapping(checked.get("model"), label="summary model")
    if model != corpus["source"]["model"]:
        raise ValueError("summary model identity mismatch")
    calculator = _mapping(checked.get("calculator"), label="calculator provenance")
    if calculator != corpus["source"]["calculator"]:
        raise ValueError("calculator provenance mismatch")
    runtime = _mapping(checked.get("runtime_versions"), label="runtime provenance")
    if runtime != corpus["source"]["runtime_versions"]:
        raise ValueError("runtime provenance mismatch")
    cuda = _mapping(checked.get("cuda"), label="CUDA provenance")
    if cuda != corpus["source"]["cuda"]:
        raise ValueError("CUDA provenance mismatch")
    if checked.get("safe_lbfgs_default_history_limit") != 10:
        raise ValueError("safe L-BFGS history identity mismatch")
    if checked.get("strict_protocol") != {
        "fmax_eV_per_A": STRICT_FMAX,
        "maxiter": MAXITER,
        "coordinate_trust_radius_A": None,
        "objective": OBJECTIVE,
    }:
        raise ValueError("strict protocol mismatch")
    if checked.get("arms") != list(ARMS):
        raise ValueError("arm matrix identity mismatch")
    if (
        checked.get("trigger_count") != 4
        or checked.get("row_count") != 12
        or checked.get("calculator_instances") != 6
        or checked.get("rows_file") != "rows.json"
    ):
        raise ValueError("summary matrix identity mismatch")
    primary = _validate_primary_cohort(checked.get("primary_cohort"))
    return checked, {
        "source_replay": corpus["source"],
        "model": model,
        "calculator": calculator,
        "runtime_versions": runtime,
        "cuda": cuda,
        "primary_cohort": primary,
    }


def _validate_purpose_counts(value: Any, *, label: str) -> dict[str, int]:
    counts = _mapping(value, label=f"{label} purpose counts")
    if set(counts) != set(PURPOSES):
        raise ValueError(f"{label} evaluation ledger purposes mismatch")
    return {
        purpose: _nonnegative_int(counts[purpose], label=f"{label} {purpose}")
        for purpose in PURPOSES
    }


def _validate_row(
    row_value: Any,
    entry: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    row = _mapping(row_value, label=label)
    identity_fields = (
        "system",
        "task_index",
        "trial_index",
        "proposal_index",
        "source_trajectory_path",
        "source_trajectory_sha256",
        "source_frame_index",
    )
    if any(row.get(field) != entry.get(field) for field in identity_fields):
        raise ValueError(f"{label} raw fallback start identity mismatch")
    if any(row.get(field) != arm[field] for field in arm):
        raise ValueError(f"{label} arm identity mismatch")
    if (
        row.get("fmax_eV_per_A") != STRICT_FMAX
        or row.get("maxiter") != MAXITER
        or row.get("coordinate_trust_radius_A") is not None
        or row.get("objective") != OBJECTIVE
    ):
        raise ValueError(f"{label} strict protocol mismatch")
    if row.get("primary") != entry["primary"]:
        raise ValueError(f"{label} primary identity mismatch")

    fallback = _mapping(row.get("fallback"), label=f"{label} fallback")
    initial = _mapping(fallback.get("initial"), label=f"{label} fallback initial")
    final = _mapping(fallback.get("final"), label=f"{label} fallback final")
    start = entry["fallback_start"]
    if (
        initial.get("positions_sha256") != start["positions_sha256"]
        or initial.get("state") != start["state"]
    ):
        raise ValueError(f"{label} raw fallback start identity mismatch")
    _validate_state_hash(
        initial.get("state"),
        initial.get("positions_sha256"),
        label=f"{label} fallback initial",
    )
    _validate_state_hash(
        final.get("state"),
        final.get("positions_sha256"),
        label=f"{label} fallback final",
    )
    initial_energy = _finite(
        initial.get("energy_eV"), label=f"{label} initial energy"
    )
    _finite(
        initial.get("max_active_force_eV_per_A"),
        label=f"{label} initial force",
        nonnegative=True,
    )
    final_energy = _finite(final.get("energy_eV"), label=f"{label} final energy")
    final_force = _finite(
        final.get("max_active_force_eV_per_A"),
        label=f"{label} final force",
        nonnegative=True,
    )
    energy_change = _finite(
        fallback.get("energy_change_eV"), label=f"{label} energy change"
    )
    if not math.isclose(
        energy_change, final_energy - initial_energy, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(f"{label} energy change does not close")
    _nonnegative_int(fallback.get("n_iter"), label=f"{label} n_iter")
    wall_time = _finite(
        fallback.get("wall_time_s"), label=f"{label} wall time", nonnegative=True
    )
    certificate = final_force <= STRICT_FMAX
    if (
        type(fallback.get("certificate_passed")) is not bool
        or fallback["certificate_passed"] != certificate
    ):
        raise ValueError(f"{label} certificate mismatch")
    telemetry = _mapping(fallback.get("telemetry"), label=f"{label} telemetry")
    if (
        telemetry.get("backend") != arm["optimizer"]
        or telemetry.get("gradient_measure") != "raw_active_max_force"
        or type(telemetry.get("converged")) is not bool
        or telemetry["converged"] != certificate
    ):
        raise ValueError(f"{label} telemetry convergence mismatch")
    termination = _nonempty_string(
        fallback.get("termination_reason"), label=f"{label} termination"
    )
    if (
        telemetry.get("termination_reason") != termination
        or (termination == "converged") != certificate
    ):
        raise ValueError(f"{label} termination/certificate mismatch")
    force_evaluations = _nonnegative_int(
        fallback.get("force_evaluations"), label=f"{label} force evaluations"
    )
    evaluator_calls = _nonnegative_int(
        fallback.get("evaluator_calls"), label=f"{label} evaluator calls"
    )
    if telemetry.get("evaluator_calls") != evaluator_calls:
        raise ValueError(f"{label} telemetry evaluator ledger mismatch")
    counts = _validate_purpose_counts(
        fallback.get("purpose_count_delta"), label=label
    )
    if (
        evaluator_calls != force_evaluations
        or sum(counts.values()) != force_evaluations
        or counts["landing_true_quench"] != force_evaluations
        or counts["unattributed"] != 0
        or any(
            counts[purpose] != 0
            for purpose in PURPOSES
            if purpose != "landing_true_quench"
        )
    ):
        raise ValueError(f"{label} evaluation ledger does not close")
    cost = _mapping(row.get("cost"), label=f"{label} cost")
    primary_fe = entry["primary"]["force_evaluations"]
    if cost != {
        "primary_force_evaluations": primary_fe,
        "offline_fallback_force_evaluations": force_evaluations,
        "offline_combined_replay_force_evaluations": (
            primary_fe + force_evaluations
        ),
        "offline_repeats_primary_terminal_evaluation": True,
        "continuous_implementation_avoidable_force_evaluations": 1,
    }:
        raise ValueError(f"{label} per-row cost ledger mismatch")
    return {
        "certificate": certificate,
        "force_evaluations": force_evaluations,
        "wall_time_s": wall_time,
        "termination_reason": termination,
        "energy_change_eV": energy_change,
        "final_force": final_force,
        "purpose_counts": counts,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def analyze(
    *,
    summary_path: Path = DEFAULT_OUTPUT_ROOT / "summary.json",
    rows_path: Path = DEFAULT_OUTPUT_ROOT / "rows.json",
    corpus_path: Path = CORPUS_PATH,
) -> dict[str, Any]:
    """Validate every identity and ledger before deriving fixed-corpus evidence."""

    summary_path = Path(summary_path)
    rows_path = Path(rows_path)
    corpus_path = Path(corpus_path)
    rows_hash = _sha256(rows_path)
    summary_hash = _sha256(summary_path)
    if rows_hash != EXPECTED_ROWS_SHA256:
        raise ValueError("source rows SHA mismatch")
    if summary_hash != EXPECTED_SUMMARY_SHA256:
        raise ValueError("source summary SHA mismatch")
    corpus, entries = _validate_corpus(corpus_path)
    summary, provenance = _validate_summary(_load_json(summary_path), corpus)
    rows = _load_json(rows_path)
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("row count mismatch: expected complete 12-row matrix")

    expected_keys = {
        (system, task_index, arm["arm_id"])
        for system, task_index in entries
        for arm in ARMS
    }
    seen: set[tuple[str, int, str]] = set()
    validated_by_arm: dict[str, list[dict[str, Any]]] = {
        arm["arm_id"]: [] for arm in ARMS
    }
    aggregate_counts = {purpose: 0 for purpose in PURPOSES}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        key = (row.get("system"), row.get("task_index"), row.get("arm_id"))
        if key in seen:
            raise ValueError("fallback matrix contains duplicate row")
        seen.add(key)
        if key not in expected_keys:
            raise ValueError("fallback matrix contains unexpected row")
        arm = next(item for item in ARMS if item["arm_id"] == key[2])
        result = _validate_row(
            row,
            entries[(key[0], key[1])],
            arm,
            label=f"row {index}",
        )
        validated_by_arm[key[2]].append(result)
        for purpose, count in result["purpose_counts"].items():
            aggregate_counts[purpose] += count
    if seen != expected_keys:
        raise ValueError("fallback matrix coverage mismatch")

    recorded_counts = _validate_purpose_counts(
        summary.get("fallback_evaluation_counts"), label="summary"
    )
    if recorded_counts != aggregate_counts:
        raise ValueError("summary evaluation ledger does not close")

    primary_total = EXPECTED_PRIMARY_COHORT["overall"]["total_force_evaluations"]
    primary_success = EXPECTED_PRIMARY_COHORT["overall"][
        "certificate_success_count"
    ]
    arm_evidence: dict[str, Any] = {}
    expected_summary_by_arm = _mapping(
        summary.get("pipeline_cost_by_fallback_arm"),
        label="summary per-arm costs",
    )
    if set(expected_summary_by_arm) != {arm["arm_id"] for arm in ARMS}:
        raise ValueError("summary per-arm matrix mismatch")
    for arm in ARMS:
        arm_id = arm["arm_id"]
        values = validated_by_arm[arm_id]
        fallback_success = sum(item["certificate"] for item in values)
        fallback_fe = sum(item["force_evaluations"] for item in values)
        offline_total = primary_total + fallback_fe
        continuous_total = offline_total - len(values)
        expected_summary = {
            "trigger_count": 4,
            "fallback_certificate_success_count": fallback_success,
            "offline_fallback_force_evaluations": fallback_fe,
            "offline_pipeline_total_force_evaluations": offline_total,
            "continuous_pipeline_projected_force_evaluations": continuous_total,
        }
        if expected_summary_by_arm[arm_id] != expected_summary:
            raise ValueError(f"summary per-arm formula mismatch for {arm_id}")
        continuous_increment = continuous_total - primary_total
        arm_evidence[arm_id] = {
            "optimizer": arm["optimizer"],
            "safe_history_limit": arm["safe_history_limit"],
            "fallback": {
                "task_count": len(values),
                "certificate_success_count": fallback_success,
                "certificate_failure_count": len(values) - fallback_success,
                "certificate_success_rate": fallback_success / len(values),
                "total_force_evaluations": fallback_fe,
                "total_wall_time_s": sum(
                    item["wall_time_s"] for item in values
                ),
                "termination_reason_counts": dict(
                    sorted(
                        Counter(
                            item["termination_reason"] for item in values
                        ).items()
                    )
                ),
                "energy_change_eV": _distribution(
                    [item["energy_change_eV"] for item in values]
                ),
                "final_max_force_eV_per_A": _distribution(
                    [item["final_force"] for item in values]
                ),
            },
            "pipeline": {
                "task_count": 32,
                "certificate_success_count": primary_success + fallback_success,
                "certificate_success_rate": (
                    primary_success + fallback_success
                )
                / 32,
                "primary_force_evaluations": primary_total,
                "offline_total_force_evaluations": offline_total,
                "offline_increment_force_evaluations": fallback_fe,
                "offline_increment_fraction_of_primary": fallback_fe
                / primary_total,
                "continuous_projected_total_force_evaluations": continuous_total,
                "continuous_projected_increment_force_evaluations": (
                    continuous_increment
                ),
                "continuous_projected_increment_fraction_of_primary": (
                    continuous_increment / primary_total
                ),
                "continuous_projection_avoided_terminal_evaluations": len(values),
            },
        }
    full_coverage = [
        arm_id
        for arm_id, result in arm_evidence.items()
        if result["fallback"]["certificate_success_count"] == 4
    ]
    return {
        "schema_version": 1,
        "validation": {
            "status": "passed",
            "row_count": 12,
            "matrix": "4_fixed_primary_certificate_failures_x_3_fallback_arms",
            "artifacts": {
                "rows": {
                    "path": _display_path(rows_path),
                    "sha256": rows_hash,
                },
                "summary": {
                    "path": _display_path(summary_path),
                    "sha256": summary_hash,
                },
                "corpus": {
                    "path": _display_path(corpus_path),
                    "sha256": EXPECTED_CORPUS_SHA256,
                },
            },
            "execution_commit": EXPECTED_EXECUTION_COMMIT,
            "primary_cohort": provenance["primary_cohort"],
            "fallback_evaluation_counts": aggregate_counts,
            "provenance": {
                key: provenance[key]
                for key in (
                    "source_replay",
                    "model",
                    "calculator",
                    "runtime_versions",
                    "cuda",
                )
            },
        },
        "arms": arm_evidence,
        "selection_statement": {
            "only_full_coverage_fallback_on_fixed_corpus": (
                full_coverage[0] if len(full_coverage) == 1 else None
            ),
            "production_default_selected": False,
        },
        "claim_boundary": {
            "continuous_cost": (
                "projection assuming cached terminal evaluation; not measured by "
                "this offline replay"
            ),
            "endpoint_energy_comparison": (
                "not a fair speed comparison because fallback endpoints differ"
            ),
            "global_search_performance": "not measured",
            "production_default": "not selected",
        },
        "next_step": (
            "minimally integrate ASE-LBFGS primary plus certificate-triggered "
            "FIRE fallback, then run fixed-budget end-to-end validation; add no "
            "extra fallback layer or parameter"
        ),
    }


def write_conclusion(evidence: Mapping[str, Any], path: Path) -> None:
    """Write the fixed-corpus result and its explicit claim ceiling in Chinese."""

    restart = evidence["arms"]["ase-lbfgs-restart"]
    safe = evidence["arms"]["safe-lbfgs-total"]
    fire = evidence["arms"]["ase-fire"]
    fire_fraction = (
        100
        * fire["pipeline"][
            "continuous_projected_increment_fraction_of_primary"
        ]
    )
    text = f"""# Sequential strict-quench rescue：固定 corpus 结论

验证通过：输入是 ASE-LBFGS primary 严格证书失败的固定 4 个 terminal states；primary cohort 共 32 个任务，原本通过 28/32，成本为 5096 次 force evaluations。

- ASE-LBFGS restart：3/4 个 fallback 通过，完整 pipeline 为 31/32；离线 replay 总成本 {restart["pipeline"]["offline_total_force_evaluations"]} FE，连续实现投影 {restart["pipeline"]["continuous_projected_total_force_evaluations"]} FE。
- safe L-BFGS：1/4 个 fallback 通过，完整 pipeline 为 29/32；离线 replay 总成本 {safe["pipeline"]["offline_total_force_evaluations"]} FE，连续实现投影 {safe["pipeline"]["continuous_projected_total_force_evaluations"]} FE。
- FIRE：4/4 个 fallback 通过，完整 pipeline 为 32/32；离线 replay 总成本 {fire["pipeline"]["offline_total_force_evaluations"]} FE，连续实现投影 {fire["pipeline"]["continuous_projected_total_force_evaluations"]} FE。相对 primary，连续投影额外 490 FE，即 {fire_fraction:.7f}%。

在这个固定 corpus 上，FIRE 是唯一覆盖全部 4 个 ASE-LBFGS 证书失败任务的 fallback。

## 解释边界

“连续实现投影”假设 primary 的 cached terminal evaluation 能直接交给 fallback，因此每个触发任务比当前离线 replay 少 1 FE；这是投影，不是当前实测。不同 fallback endpoint energy 不相同，因此这些 endpoint 的能量与 wall time 不作速度公平比较。本结果不声称 global-search 性能已经改善，也不声称生产默认优化器已经选定。

## 唯一下一步

最小化集成 `ASE-LBFGS primary + certificate-triggered FIRE fallback`，随后做固定预算 end-to-end 验证；不加入额外 fallback 层或参数。
"""
    Path(path).write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_OUTPUT_ROOT / "summary.json"
    )
    parser.add_argument(
        "--rows", type=Path, default=DEFAULT_OUTPUT_ROOT / "rows.json"
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--evidence", type=Path, default=RUN_ROOT / "evidence.json")
    parser.add_argument("--conclusion", type=Path, default=RUN_ROOT / "conclusion.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = analyze(
        summary_path=args.summary,
        rows_path=args.rows,
        corpus_path=args.corpus,
    )
    _write_json(args.evidence, evidence)
    write_conclusion(evidence, args.conclusion)
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
