#!/usr/bin/env python3
"""Validate and summarize the fixed-budget direction hard-cap evidence."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
TOTAL_FORCE_BUDGET = 6000
PAIRS = (("c60", 42), ("c60", 43), ("pdo", 42))
ARM_COMMITS = {
    "precap": "32980ad9154ec6481b310477dd7b85597cefd49a",
    "hardcap": "3b0dd65711d2155460e0ab6b1cd0c95602a8c422",
}
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
INPUT_SHA256 = {
    "c60": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
    "pdo": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
}
OUTPUT_CONFIG_SUFFIXES = {
    "accepted_structures_dir": "accepted_minima",
    "accepted_structures_log": "accepted_structures.jsonl",
    "direction_diagnostics_path": "direction_trace.jsonl",
}
METRICS = {
    "trials": ("record_counts", "n_trials"),
    "minima": ("stats", "n_minima"),
    "duplicate_rate": ("stats", "duplicate_rate"),
    "energy_drop_eV": ("energy_drop_eV",),
    "wall_time_s": ("timing", "total_wall_time_s"),
    "candidate_pool_mean": ("stats", "direction_mean_candidate_pool_size"),
    "direction_force_evaluations": ("purpose_counts", "direction_oracle"),
}


class EvidenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _nonnegative_exact_int(value: Any, label: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{label} must be a builtin exact nonnegative integer",
    )
    return value


def _finite_number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value),
        f"{label} must be a finite builtin number",
    )
    number = float(value)
    if minimum is not None:
        _require(number >= minimum, f"{label} must be >= {minimum}")
    if maximum is not None:
        _require(number <= maximum, f"{label} must be <= {maximum}")
    return number


def _value(summary: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = summary
    for key in path:
        value = value[key]
    return value


def _validate_output_paths(
    config: dict[str, Any], *, run_name: str, config_name: str
) -> None:
    for key, suffix in OUTPUT_CONFIG_SUFFIXES.items():
        expected_suffix = f"{run_name}/{suffix}"
        _require(
            str(config[key]).endswith(expected_suffix),
            f"{run_name}: {config_name}.{key} is not derived from its output path",
        )


def _without_output_paths(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in OUTPUT_CONFIG_SUFFIXES
    }


def _load_run(root: Path, *, system: str, seed: int, arm: str) -> dict[str, Any]:
    run_name = f"output-{arm}-{system}-seed{seed}"
    summary_path = root / run_name / "summary.json"
    _require(summary_path.is_file(), f"{run_name}: summary.json is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    commit = ARM_COMMITS[arm]
    _nonnegative_exact_int(summary["seed"], f"{run_name}: seed")
    _require(summary["arm"] == arm, f"{run_name}: arm mismatch")
    _require(summary["system"] == system, f"{run_name}: system mismatch")
    _require(summary["seed"] == seed, f"{run_name}: seed mismatch")
    _require(
        summary["target"]["execution_commit"] == commit
        and summary["base_preflight"]["execution_commit"] == commit,
        f"{run_name}: execution commit mismatch",
    )
    _require(
        summary["base_preflight"]["model_sha256"] == MODEL_SHA256,
        f"{run_name}: model mismatch",
    )
    _require(
        summary["base_preflight"]["input_sha256"] == INPUT_SHA256[system],
        f"{run_name}: input mismatch",
    )
    _require(
        summary["base_preflight"]["cuda"]["available"] is True
        and summary["base_preflight"]["calculator"]["device"] == "cuda",
        f"{run_name}: GPU runtime contract failed",
    )
    total_budget = _nonnegative_exact_int(
        summary["total_force_budget"], f"{run_name}: total_force_budget"
    )
    force_evaluations = _nonnegative_exact_int(
        summary["force_evaluations"], f"{run_name}: force_evaluations"
    )
    budget_exhausted = _nonnegative_exact_int(
        summary["stats"]["budget_exhausted"], f"{run_name}: budget_exhausted"
    )
    _require(
        total_budget == force_evaluations == TOTAL_FORCE_BUDGET
        and budget_exhausted == 1,
        f"{run_name}: fixed 6000-FE budget contract failed",
    )
    _finite_number(summary["initial_energy_eV"], f"{run_name}: initial_energy_eV")
    _finite_number(summary["best_energy_eV"], f"{run_name}: best_energy_eV")
    _finite_number(
        summary["energy_drop_eV"], f"{run_name}: energy_drop_eV", minimum=0.0
    )
    _finite_number(
        summary["timing"]["total_wall_time_s"],
        f"{run_name}: total_wall_time_s",
        minimum=0.0,
    )
    _finite_number(
        summary["stats"]["duplicate_rate"],
        f"{run_name}: duplicate_rate",
        minimum=0.0,
        maximum=1.0,
    )
    candidate_mean = _finite_number(
        summary["stats"]["direction_mean_candidate_pool_size"],
        f"{run_name}: candidate mean",
        minimum=0.0,
    )

    purpose_counts = summary["purpose_counts"]
    for purpose, count in purpose_counts.items():
        _nonnegative_exact_int(count, f"{run_name}: purpose_counts.{purpose}")
    _require(
        purpose_counts["unattributed"] == 0,
        f"{run_name}: unattributed force evaluations are nonzero",
    )
    _require(
        sum(purpose_counts.values()) == summary["force_evaluations"],
        f"{run_name}: purpose counts do not close to total force evaluations",
    )

    audit = summary["direction_oracle_audit"]
    for field in (
        "candidate_count_sum",
        "candidate_count_max",
        "expected_force_evaluations",
        "recorded_force_evaluations",
    ):
        _nonnegative_exact_int(audit[field], f"{run_name}: audit.{field}")
    expected_direction_fe = 2 * audit["candidate_count_sum"]
    _require(
        purpose_counts["direction_oracle"]
        == audit["recorded_force_evaluations"]
        == audit["expected_force_evaluations"]
        == expected_direction_fe,
        f"{run_name}: direction-oracle accounting is not 2 * candidate sum",
    )
    oracle_candidates = _nonnegative_exact_int(
        summary["effective_config"]["oracle_candidates"],
        f"{run_name}: oracle_candidates",
    )
    actual_overflow = max(0, audit["candidate_count_max"] - oracle_candidates)
    if arm == "precap":
        _require(actual_overflow > 0, f"{run_name}: precap overflow is not positive")
    else:
        _require(actual_overflow == 0, f"{run_name}: hardcap overflow is nonzero")

    native = summary["native_generator_contract"]
    candidate_budget = _nonnegative_exact_int(
        native["candidate_budget"], f"{run_name}: native candidate_budget"
    )
    nonfirst_count = _nonnegative_exact_int(
        native["nonfirst_candidate_count"],
        f"{run_name}: native nonfirst_candidate_count",
    )
    expected_native = {
        "precap": "nonfirst_native_candidates_exceed_budget",
        "hardcap": "native_candidates_do_not_exceed_budget",
    }[arm]
    native_count_valid = (
        nonfirst_count > candidate_budget
        if arm == "precap"
        else nonfirst_count <= candidate_budget
    )
    _require(
        candidate_budget == 2
        and native["expectation"] == expected_native
        and native_count_valid,
        f"{run_name}: native generator contract is invalid for {arm}",
    )

    records = summary["record_counts"]
    for field, count in records.items():
        _nonnegative_exact_int(count, f"{run_name}: record_counts.{field}")
    stats = summary["stats"]
    for field in (
        "n_trials",
        "n_minima",
        "direction_choices",
        "direction_candidate_evaluations",
    ):
        _nonnegative_exact_int(stats[field], f"{run_name}: stats.{field}")
    _require(
        records["n_trials"] == records["walk_records"] == stats["n_trials"]
        and records["direction_records"] == stats["direction_choices"]
        and records["unlogged_direction_choices"] == 0
        and records["unlogged_walk_trials"] == 0,
        f"{run_name}: record counts do not close",
    )
    _require(
        stats["direction_candidate_evaluations"] == audit["candidate_count_sum"],
        f"{run_name}: direction candidate evaluations do not match audit",
    )
    expected_candidate_mean = (
        audit["candidate_count_sum"] / stats["direction_choices"]
        if stats["direction_choices"]
        else 0.0
    )
    _require(
        math.isclose(candidate_mean, expected_candidate_mean, rel_tol=1e-12),
        f"{run_name}: candidate mean does not match candidate sum / choices",
    )

    for config_name in ("source_config", "effective_config"):
        _validate_output_paths(
            summary[config_name], run_name=run_name, config_name=config_name
        )

    return {
        "run": run_name,
        "system": system,
        "seed": seed,
        "arm": arm,
        "summary": str(summary_path.relative_to(root)),
        "summary_sha256": _sha256(summary_path),
        "execution_commit": commit,
        "model_sha256": MODEL_SHA256,
        "input_sha256": INPUT_SHA256[system],
        "force_evaluations": summary["force_evaluations"],
        "budget_exhausted": True,
        "purpose_counts_closed": True,
        "unattributed_force_evaluations": 0,
        "candidate_count_sum": audit["candidate_count_sum"],
        "candidate_count_max": audit["candidate_count_max"],
        "candidate_count_overflow": actual_overflow,
        "oracle_candidates": oracle_candidates,
        "metrics": {
            name: _value(summary, path) for name, path in METRICS.items()
        },
        "_configs": {
            name: _without_output_paths(summary[name])
            for name in ("source_config", "effective_config")
        },
        "_runtime_identity": {
            key: summary["base_preflight"][key]
            for key in (
                "runtime_versions",
                "cuda",
                "calculator",
                "input_sha256",
                "model_sha256",
            )
        }
        | {"frozen_runner_sha256": summary["target"]["frozen_runner_sha256"]},
    }


def build_evidence(root: Path = RUN_ROOT) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for system, seed in PAIRS:
        by_arm = {
            arm: _load_run(root, system=system, seed=seed, arm=arm)
            for arm in ("precap", "hardcap")
        }
        for config_name in ("source_config", "effective_config"):
            _require(
                by_arm["precap"]["_configs"][config_name]
                == by_arm["hardcap"]["_configs"][config_name],
                f"{system} seed {seed}: paired {config_name} differs beyond output paths",
            )
        _require(
            by_arm["precap"]["_runtime_identity"]
            == by_arm["hardcap"]["_runtime_identity"],
            f"{system} seed {seed}: paired runtime identity differs",
        )

        delta = {
            name: by_arm["hardcap"]["metrics"][name]
            - by_arm["precap"]["metrics"][name]
            for name in METRICS
        }
        pairs.append(
            {
                "system": system,
                "seed": seed,
                "precap": by_arm["precap"]["metrics"],
                "hardcap": by_arm["hardcap"]["metrics"],
                "delta_hardcap_minus_precap": delta,
            }
        )
        for arm in ("precap", "hardcap"):
            by_arm[arm].pop("_configs")
            by_arm[arm].pop("_runtime_identity")
            runs.append(by_arm[arm])

    return {
        "schema_version": 1,
        "provenance": {"analyzer_sha256": _sha256(Path(__file__).resolve())},
        "scope": {
            "included_runs": 6,
            "pairs": [
                {"system": system, "seed": seed} for system, seed in PAIRS
            ],
            "excluded_glob": "output-*.partial-failed",
            "claim": "budget_semantic_correctness_only",
        },
        "validated_contract": {
            "total_force_budget": TOTAL_FORCE_BUDGET,
            "budget_exhausted": True,
            "purpose_counts_closed": True,
            "unattributed_force_evaluations": 0,
            "direction_oracle_formula": "2 * candidate_count_sum",
            "precap_overflow_positive_and_hardcap_overflow_zero": True,
            "native_generator_contracts_valid": True,
            "record_counts_and_direction_stats_closed": True,
            "paired_configs_equal_except_derived_output_paths": True,
            "paired_runtime_identity_equal": True,
        },
        "runs": runs,
        "pairs": pairs,
    }


def render_conclusion(evidence: dict[str, Any]) -> str:
    rows = []
    for pair in evidence["pairs"]:
        delta = pair["delta_hardcap_minus_precap"]
        rows.append(
            "| {system} | {seed} | {trials:+d} | {minima:+d} | "
            "{duplicate:+.6f} | {drop:+.6f} | {wall:+.6f} | "
            "{candidate:+.6f} | {direction_fe:+d} |".format(
                system=pair["system"].upper(),
                seed=pair["seed"],
                trials=delta["trials"],
                minima=delta["minima"],
                duplicate=delta["duplicate_rate"],
                drop=delta["energy_drop_eV"],
                wall=delta["wall_time_s"],
                candidate=delta["candidate_pool_mean"],
                direction_fe=delta["direction_force_evaluations"],
            )
        )

    return "\n".join(
        [
            "# Direction candidate hard-cap: fixed-budget GPU evidence",
            "",
            "纳入且仅纳入 6 个完成的 GPU run：C60 seeds 42/43 与 PdO seed 42 "
            "的 precap/hardcap 配对；`*.partial-failed` 明确排除。",
            "",
            "六个 run 均耗尽精确的 6000 次 force evaluation；purpose 计数闭合、"
            "`unattributed=0`，且 `direction_oracle = 2 × candidate_count_sum`。"
            "arm-specific native contract 成立，实际 precap overflow 为正、hardcap "
            "overflow 为零；record/direction 统计闭合。配对配置除派生输出路径外"
            "一致，runtime/CUDA/calculator/input/model/frozen runner 运行环境身份一致。"
            "因此本实验支持的是 **hard-cap 预算语义正确**。",
            "",
            "下表全部为 `hardcap - precap`：",
            "",
            "| system | seed | trials | minima | duplicate rate | energy drop (eV) | "
            "wall (s) | candidate mean | direction FE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "性能结果不一致：C60 seed 42 的 energy drop 增加，seed 43 减少；"
            "PdO seed 42 也减少。候选池均值下降并不保证总 direction FE 下降，"
            "因为完成的 direction choices/trials 数会变化。",
            "",
            "结论止于预算语义正确性；不宣称搜索质量或性能提升，也不为这一语义"
            "验证继续扩展 seeds。",
            "",
        ]
    )


def main() -> None:
    evidence = build_evidence(RUN_ROOT)
    (RUN_ROOT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "conclusion.md").write_text(
        render_conclusion(evidence), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
