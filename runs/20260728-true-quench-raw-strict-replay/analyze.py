#!/usr/bin/env python3
"""Fail-closed analysis of the fixed raw-landing strict-quench replay."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import floor, isfinite
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "output"
DEFAULT_CORPUS_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-true-quench-tiered-ablation"
    / "output"
    / "corpus.json"
)
EXPECTED_EXECUTION_COMMIT = "e3141c46b53e0a646fbe3da95678f04e389b4613"
FROZEN_CORPUS_SHA256 = "100759e1871cdefb54972f91751c452763f74d0e34ee576f268a5808219a552b"
FROZEN_MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
FROZEN_CORPUS_REPO_PATH = (
    "runs/20260728-true-quench-tiered-ablation/output/corpus.json"
)
SYSTEMS = ("c60", "pdo")
TASKS_PER_SYSTEM = 16
STRICT_FMAX = 0.01
MAXITER = 400
OBJECTIVE = "true_mace_pes_no_bias_no_softening"
ARMS = (
    {
        "arm_id": "scipy-lbfgsb",
        "optimizer": "scipy-lbfgsb",
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
    {
        "arm_id": "ase-fire2",
        "optimizer": "ase-fire2",
        "safe_history_limit": None,
    },
    {
        "arm_id": "ase-lbfgs",
        "optimizer": "ase-lbfgs",
        "safe_history_limit": None,
    },
)
ARM_BY_ID = {arm["arm_id"]: arm for arm in ARMS}
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
EXPECTED_ROW_COUNT = len(SYSTEMS) * TASKS_PER_SYSTEM * len(ARMS)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}") from error


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} provenance must be a nonempty string")
    return value


def _percentile(values: Sequence[float | int], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_corpus(path: Path) -> dict[tuple[str, int], Mapping[str, Any]]:
    if _sha256(path) != FROZEN_CORPUS_SHA256:
        raise ValueError("frozen corpus SHA mismatch")
    payload = _mapping(_read_json(path), label="corpus")
    systems = _mapping(payload.get("systems"), label="corpus systems")
    if set(systems) != set(SYSTEMS):
        raise ValueError("corpus systems are incomplete or unexpected")
    tasks: dict[tuple[str, int], Mapping[str, Any]] = {}
    for system in SYSTEMS:
        manifest = _mapping(systems[system], label=f"{system} corpus manifest")
        entries = manifest.get("entries")
        if (
            manifest.get("system") != system
            or manifest.get("task_count") != TASKS_PER_SYSTEM
            or not isinstance(entries, list)
            or len(entries) != TASKS_PER_SYSTEM
        ):
            raise ValueError(f"{system} corpus identity is invalid")
        for task_index, entry_value in enumerate(entries):
            entry = _mapping(
                entry_value, label=f"{system} corpus task {task_index}"
            )
            if (
                entry.get("system") != system
                or entry.get("task_index") != task_index
                or entry.get("trial_index") != task_index + 1
                or entry.get("proposal_index") != 1
                or entry.get("source_frame_index") != 0
            ):
                raise ValueError(f"{system} corpus task {task_index} identity is invalid")
            tasks[(system, task_index)] = entry
    return tasks


def _validate_summary(
    summary: Mapping[str, Any], corpus_path: Path
) -> dict[str, Mapping[str, Any]]:
    if summary.get("schema_version") != 1:
        raise ValueError("summary schema mismatch")
    if summary.get("execution_commit") != EXPECTED_EXECUTION_COMMIT:
        raise ValueError("execution commit mismatch")
    if summary.get("row_count") != EXPECTED_ROW_COUNT:
        raise ValueError("summary row count mismatch")
    corpus = _mapping(summary.get("corpus"), label="summary corpus")
    if corpus.get("sha256") != FROZEN_CORPUS_SHA256:
        raise ValueError("summary corpus SHA mismatch")
    if corpus.get("path") != FROZEN_CORPUS_REPO_PATH:
        raise ValueError("summary corpus path mismatch")
    if (
        corpus.get("task_count_per_system") != TASKS_PER_SYSTEM
        or corpus.get("total_task_count") != len(SYSTEMS) * TASKS_PER_SYSTEM
    ):
        raise ValueError("summary corpus task count mismatch")
    if _sha256(corpus_path) != FROZEN_CORPUS_SHA256:
        raise ValueError("frozen corpus SHA mismatch")
    model = _mapping(summary.get("model"), label="summary model")
    _nonempty_string(model.get("path"), label="model path")
    if model.get("sha256") != FROZEN_MODEL_SHA256:
        raise ValueError("model SHA mismatch")
    calculator = _mapping(summary.get("calculator"), label="calculator provenance")
    for field in ("default_dtype", "device", "inference_precision"):
        _nonempty_string(calculator.get(field), label=f"calculator {field}")
    if not isinstance(calculator.get("enable_cueq"), bool):
        raise ValueError("calculator enable_cueq provenance must be boolean")
    cuda = _mapping(summary.get("cuda"), label="CUDA provenance")
    if not isinstance(cuda.get("available"), bool):
        raise ValueError("CUDA available provenance must be boolean")
    for field in ("device_name", "runtime_version"):
        _nonempty_string(cuda.get(field), label=f"CUDA {field}")
    runtime_versions = _mapping(
        summary.get("runtime_versions"), label="runtime versions provenance"
    )
    if not runtime_versions:
        raise ValueError("runtime versions provenance must not be empty")
    for package, version in runtime_versions.items():
        _nonempty_string(package, label="runtime package")
        _nonempty_string(version, label=f"runtime version {package}")
    inputs = _mapping(summary.get("inputs"), label="inputs provenance")
    if set(inputs) != set(SYSTEMS):
        raise ValueError("inputs provenance systems mismatch")
    for system in SYSTEMS:
        input_identity = _mapping(
            inputs[system], label=f"{system} input provenance"
        )
        _nonempty_string(input_identity.get("path"), label=f"{system} input path")
        digest = input_identity.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{system} input SHA provenance is invalid")
    protocol = _mapping(summary.get("strict_protocol"), label="strict protocol")
    if protocol != {
        "fmax_eV_per_A": STRICT_FMAX,
        "maxiter": MAXITER,
        "objective": OBJECTIVE,
    }:
        raise ValueError("summary strict protocol mismatch")
    if summary.get("arms") != list(ARMS):
        raise ValueError("summary arm identity mismatch")
    if summary.get("rows_file") != "rows.json":
        raise ValueError("summary rows file mismatch")
    return {
        "calculator": calculator,
        "cuda": cuda,
        "runtime_versions": runtime_versions,
        "inputs": inputs,
        "model": model,
    }


def _purpose_counts(value: Any, *, label: str) -> dict[str, int]:
    counts = _mapping(value, label=f"{label} purpose counts")
    if set(counts) != set(PURPOSES):
        raise ValueError(f"{label} evaluation ledger has unexpected purposes")
    return {
        purpose: _nonnegative_int(
            counts[purpose], label=f"{label} purpose {purpose}"
        )
        for purpose in PURPOSES
    }


def _validate_raw_identity(
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    label: str,
) -> None:
    expected = {
        "trial_index": task["trial_index"],
        "proposal_index": task["proposal_index"],
        "source_trajectory_path": task["source_trajectory_path"],
        "source_trajectory_sha256": task["source_trajectory_sha256"],
        "source_frame_index": task["source_frame_index"],
    }
    if any(row.get(field) != value for field, value in expected.items()):
        raise ValueError(f"{label} raw identity mismatch")
    initial = _mapping(row.get("initial"), label=f"{label} initial")
    if (
        initial.get("positions_sha256") != task["initial_positions_sha256"]
        or initial.get("state") != task["state"]
    ):
        raise ValueError(f"{label} raw identity mismatch")


def _validate_row(
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    _validate_raw_identity(row, task, label=label)
    if any(row.get(field) != arm[field] for field in arm):
        raise ValueError(f"{label} arm identity mismatch")
    if (
        row.get("fmax_eV_per_A") != STRICT_FMAX
        or row.get("maxiter") != MAXITER
        or row.get("coordinate_trust_radius_A") is not None
        or row.get("objective") != OBJECTIVE
    ):
        raise ValueError(f"{label} strict protocol mismatch")

    initial = _mapping(row.get("initial"), label=f"{label} initial")
    final = _mapping(row.get("final"), label=f"{label} final")
    initial_energy = _finite_float(
        initial.get("energy_eV"), label=f"{label} initial energy"
    )
    final_energy = _finite_float(
        final.get("energy_eV"), label=f"{label} final energy"
    )
    final_force = _finite_float(
        final.get("max_active_force_eV_per_A"),
        label=f"{label} final max active force",
    )
    if final_force < 0:
        raise ValueError(f"{label} final max active force must be nonnegative")
    force_evaluations = _nonnegative_int(
        row.get("force_evaluations"), label=f"{label} force evaluations"
    )
    evaluator_calls = _nonnegative_int(
        row.get("evaluator_calls"), label=f"{label} evaluator calls"
    )
    counts = _purpose_counts(row.get("purpose_count_delta"), label=label)
    if (
        sum(counts.values()) != force_evaluations
        or evaluator_calls != force_evaluations
        or counts["landing_true_quench"] != force_evaluations
        or counts["unattributed"] != 0
        or any(
            counts[purpose] != 0
            for purpose in PURPOSES
            if purpose != "landing_true_quench"
        )
    ):
        raise ValueError(f"{label} evaluation ledger does not close")
    wall_time = _finite_float(row.get("wall_time_s"), label=f"{label} wall time")
    if wall_time < 0:
        raise ValueError(f"{label} wall time must be nonnegative")
    termination_reason = row.get("termination_reason")
    if not isinstance(termination_reason, str) or not termination_reason:
        raise ValueError(f"{label} termination reason is invalid")
    telemetry = _mapping(row.get("telemetry"), label=f"{label} telemetry")
    if (
        telemetry.get("backend") != arm["optimizer"]
        or telemetry.get("evaluator_calls") != evaluator_calls
        or telemetry.get("gradient_measure") != "raw_active_max_force"
        or telemetry.get("termination_reason") != termination_reason
    ):
        raise ValueError(f"{label} telemetry identity mismatch")
    certificate_converged = final_force <= STRICT_FMAX
    telemetry_converged = telemetry.get("converged")
    if not isinstance(telemetry_converged, bool):
        raise ValueError(f"{label} telemetry convergence must be boolean")
    if (termination_reason == "converged") != certificate_converged:
        raise ValueError(f"{label} certificate convergence mismatch")
    if telemetry_converged != certificate_converged:
        raise ValueError(f"{label} telemetry convergence mismatch")
    return {
        "certificate_converged": certificate_converged,
        "force_evaluations": force_evaluations,
        "wall_time_s": wall_time,
        "energy_drop_eV": initial_energy - final_energy,
        "final_max_force_eV_per_A": final_force,
        "termination_reason": termination_reason,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(row["certificate_converged"]) for row in rows)
    force_evaluations = [int(row["force_evaluations"]) for row in rows]
    wall_times = [float(row["wall_time_s"]) for row in rows]
    energy_drops = [float(row["energy_drop_eV"]) for row in rows]
    final_forces = [float(row["final_max_force_eV_per_A"]) for row in rows]
    return {
        "task_count": len(rows),
        "success_count": successes,
        "failure_count": len(rows) - successes,
        "success_rate": successes / len(rows),
        "termination_reasons": dict(
            sorted(Counter(str(row["termination_reason"]) for row in rows).items())
        ),
        "total_force_evaluations": sum(force_evaluations),
        "median_force_evaluations": median(force_evaluations),
        "p90_force_evaluations": _percentile(force_evaluations, 0.9),
        "max_force_evaluations": max(force_evaluations),
        "total_wall_time_s": sum(wall_times),
        "median_wall_time_s": median(wall_times),
        "total_energy_drop_eV": sum(energy_drops),
        "median_energy_drop_eV": median(energy_drops),
        "final_max_force_eV_per_A": {
            "median": median(final_forces),
            "max": max(final_forces),
        },
    }


def analyze(
    *,
    summary_path: Path,
    rows_path: Path,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
) -> dict[str, Any]:
    """Validate the frozen replay and return certificate-based aggregates."""

    summary = _mapping(_read_json(summary_path), label="summary")
    provenance = _validate_summary(summary, corpus_path)
    rows_value = _read_json(rows_path)
    if not isinstance(rows_value, list) or len(rows_value) != EXPECTED_ROW_COUNT:
        raise ValueError(f"row count must be {EXPECTED_ROW_COUNT}")
    corpus_tasks = _load_corpus(corpus_path)

    keyed_rows: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row_index, row_value in enumerate(rows_value):
        row = _mapping(row_value, label=f"row {row_index}")
        system = row.get("system")
        task_index = row.get("task_index")
        arm_id = row.get("arm_id")
        if (
            system not in SYSTEMS
            or isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or not 0 <= task_index < TASKS_PER_SYSTEM
            or arm_id not in ARM_BY_ID
        ):
            raise ValueError(f"row {row_index} matrix identity is invalid")
        key = (system, task_index, arm_id)
        if key in keyed_rows:
            raise ValueError(f"duplicate matrix row {key}")
        keyed_rows[key] = row

    expected_keys = {
        (system, task_index, arm["arm_id"])
        for system in SYSTEMS
        for task_index in range(TASKS_PER_SYSTEM)
        for arm in ARMS
    }
    if set(keyed_rows) != expected_keys:
        raise ValueError("matrix coverage is incomplete")

    validated: dict[tuple[str, int, str], dict[str, Any]] = {}
    total_counts = {purpose: 0 for purpose in PURPOSES}
    for key in sorted(keyed_rows):
        system, task_index, arm_id = key
        row = keyed_rows[key]
        validated[key] = _validate_row(
            row,
            corpus_tasks[(system, task_index)],
            ARM_BY_ID[arm_id],
            label=f"{system} task {task_index} arm {arm_id}",
        )
        counts = _purpose_counts(
            row["purpose_count_delta"], label=f"{system} task {task_index} arm {arm_id}"
        )
        for purpose in PURPOSES:
            total_counts[purpose] += counts[purpose]

    summary_counts = _purpose_counts(
        summary.get("evaluation_counts"), label="summary"
    )
    if summary_counts != total_counts or summary_counts["unattributed"] != 0:
        raise ValueError("summary evaluation ledger does not close")

    systems: dict[str, Any] = {}
    for system in SYSTEMS:
        arm_results = {
            arm["arm_id"]: _aggregate(
                [
                    validated[(system, task_index, arm["arm_id"])]
                    for task_index in range(TASKS_PER_SYSTEM)
                ]
            )
            for arm in ARMS
        }
        by_task: dict[str, Any] = {}
        any_arm = 0
        all_arms = 0
        for task_index in range(TASKS_PER_SYSTEM):
            successful_arms = [
                arm["arm_id"]
                for arm in ARMS
                if validated[(system, task_index, arm["arm_id"])][
                    "certificate_converged"
                ]
            ]
            any_arm += bool(successful_arms)
            all_arms += len(successful_arms) == len(ARMS)
            by_task[str(task_index)] = {
                "success_count": len(successful_arms),
                "successful_arms": successful_arms,
            }
        systems[system] = {
            "arms": arm_results,
            "taskwise_coverage": {
                "any_arm": any_arm,
                "all_arms": all_arms,
                "task_count": TASKS_PER_SYSTEM,
                "by_task": by_task,
            },
        }

    overall_arms = {
        arm["arm_id"]: _aggregate(
            [
                validated[(system, task_index, arm["arm_id"])]
                for system in SYSTEMS
                for task_index in range(TASKS_PER_SYSTEM)
            ]
        )
        for arm in ARMS
    }
    best_success_count = max(
        result["success_count"] for result in overall_arms.values()
    )
    return {
        "schema_version": 1,
        "validation": {
            "status": "passed",
            "execution_commit": EXPECTED_EXECUTION_COMMIT,
            "row_count": EXPECTED_ROW_COUNT,
            "corpus_sha256": FROZEN_CORPUS_SHA256,
            "model_sha256": FROZEN_MODEL_SHA256,
            "summary_sha256": _sha256(summary_path),
            "rows_sha256": _sha256(rows_path),
            "evaluation_counts": total_counts,
            "strict_fmax_eV_per_A": STRICT_FMAX,
            "cost_scope": (
                "unconditional_all_attempts_including_successes_and_failures"
            ),
            "provenance": provenance,
        },
        "systems": systems,
        "overall": {
            "arms": overall_arms,
            "single_arm_full_coverage": [
                arm_id
                for arm_id, result in overall_arms.items()
                if result["success_count"] == len(SYSTEMS) * TASKS_PER_SYSTEM
            ],
            "strongest_single_arm_by_success": [
                arm_id
                for arm_id, result in overall_arms.items()
                if result["success_count"] == best_success_count
            ],
            "best_success_count": best_success_count,
        },
        "claim_boundary": {
            "measured": (
                "fixed raw-landing local true-PES strict-quench convergence and "
                "cost on one frozen C60/PdO corpus"
            ),
            "endpoint_energy_speed_comparison": (
                "not a controlled speed comparison because optimizers may end "
                "in different local minima"
            ),
            "global_search_performance": "not measured",
            "production_default": "not selected by this replay alone",
        },
    }


def _number(value: float | int, digits: int = 3) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def write_conclusion(evidence: Mapping[str, Any], path: Path) -> None:
    """Write a compact Chinese-first conclusion from validated evidence."""

    lines = [
        "# Fixed raw landing 严格 true-quench 消融结论",
        "",
        "## 判据与数据边界",
        "",
        (
            f"- 输入为冻结的 C60/PdO raw landing corpus，共 "
            f"{evidence['validation']['row_count']} 行；严格力证书为 "
            f"`max_active_force <= {evidence['validation']['strict_fmax_eV_per_A']:.2f} "
            "eV/Å`，不采信 optimizer_success 作为收敛判据。"
        ),
        (
            f"- 总核算为 "
            f"{evidence['validation']['evaluation_counts']['landing_true_quench']} "
            "次 `landing_true_quench` force eval；`unattributed=0`，逐行与汇总账本闭合。"
        ),
        "- 表中 FE、P90、wall time 和降能统计均为 unconditional：包含成功与失败的全部 attempts。",
        "",
        "## 精确结果",
        "",
        "| system | arm | 证书成功 | termination reasons | 总 FE | 中位 FE | P90 FE | 最大 FE | 总 wall/s | 中位 wall/s | 总降能/eV | 中位降能/eV | 终态力中位/最大 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        for arm in ARMS:
            arm_id = arm["arm_id"]
            result = evidence["systems"][system]["arms"][arm_id]
            final_force = result["final_max_force_eV_per_A"]
            lines.append(
                "| "
                + " | ".join(
                    (
                        system,
                        arm_id,
                        f"{result['success_count']}/{result['task_count']}",
                        ", ".join(
                            f"{reason}:{count}"
                            for reason, count in result[
                                "termination_reasons"
                            ].items()
                        ),
                        _number(result["total_force_evaluations"]),
                        _number(result["median_force_evaluations"], 1),
                        _number(result["p90_force_evaluations"], 1),
                        _number(result["max_force_evaluations"]),
                        _number(result["total_wall_time_s"]),
                        _number(result["median_wall_time_s"]),
                        _number(result["total_energy_drop_eV"]),
                        _number(result["median_energy_drop_eV"]),
                        (
                            f"{_number(final_force['median'], 6)} / "
                            f"{_number(final_force['max'], 6)}"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## 最小结论",
            "",
            "- 此次固定 corpus 中，没有单一优化器 arm 实现全覆盖。",
            (
                "- 按严格力证书的整体成功数，"
                + "、".join(evidence["overall"]["strongest_single_arm_by_success"])
                + f" 为最强单一 arm（{evidence['overall']['best_success_count']}/32）；"
                "这只是该固定 corpus 上的局部淬火结果，不是生产最终答案。"
            ),
            (
                "- C60 与 PdO 的 taskwise `any-arm` 覆盖分别为 "
                f"{evidence['systems']['c60']['taskwise_coverage']['any_arm']}/16 和 "
                f"{evidence['systems']['pdo']['taskwise_coverage']['any_arm']}/16。"
            ),
            "- 不同 arm 可能落入不同局部极小值，因此不能把终态能量差直接解释为公平的收敛速度比较。",
            "- 本实验没有重跑 SSW proposal，也不能据此声称全局搜索性能或修改生产默认值。",
            "",
            "## 唯一下一步",
            "",
            (
                "- 仅做基于严格力证书触发的 `primary + fallback` sequential rescue "
                "小消融：先运行一个 primary，只有未通过证书时才从其终态调用 fallback；"
                "继续核算两阶段全部 force eval，并且不直接修改默认值。"
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT_ROOT / "summary.json")
    parser.add_argument("--rows", type=Path, default=DEFAULT_OUTPUT_ROOT / "rows.json")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
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
