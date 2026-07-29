#!/usr/bin/env python3
"""Build paired evidence for discrete versus plain Rayleigh--Ritz selection."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
ARMS = ("discrete", "rayleigh_ritz")
PRODUCTION_TOTAL_FORCE_BUDGET = 6000
OUTPUT_PATH_FIELDS = {
    "accepted_structures_dir",
    "accepted_structures_log",
    "direction_diagnostics_path",
}
PAIR_DIFF = {"direction_selection_mode": ["discrete", "rayleigh_ritz"]}


class EvidenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an exact integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    return float(value)


def _normalise_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in OUTPUT_PATH_FIELDS}


def _optional_exact_int(mapping: Mapping[str, Any], key: str) -> tuple[int | None, str | None]:
    if key not in mapping:
        return None, f"unsupported_missing_stats.{key}"
    return _exact_int(mapping[key], f"stats.{key}"), None


def _energy_auc(trace: list[Any], label: str) -> tuple[float | None, str | None, float]:
    _require(trace, f"{label}: energy trace is empty")
    energies: list[float] = []
    cumulative: list[int] = []
    has_cumulative = True
    for index, point in enumerate(trace):
        _require(isinstance(point, dict), f"{label}: energy trace point {index} is not a mapping")
        energies.append(_finite(point.get("energy_eV"), f"{label}: energy trace energy {index}"))
        if "cumulative_total_force_evaluations" not in point:
            has_cumulative = False
            continue
        cumulative.append(
            _exact_int(
                point["cumulative_total_force_evaluations"],
                f"{label}: cumulative total FE {index}",
            )
        )
    final_energy = energies[-1]
    if not has_cumulative:
        return None, "unsupported_no_cumulative_total_force_evaluations_in_energy_trace", final_energy
    _require(len(cumulative) == len(energies), f"{label}: partial cumulative FE trace")
    _require(
        all(right >= left for left, right in zip(cumulative, cumulative[1:])),
        f"{label}: cumulative total FE is not monotonic",
    )
    best = energies[0]
    best_energies: list[float] = []
    for energy in energies:
        best = min(best, energy)
        best_energies.append(best)
    auc = sum(
        0.5 * (best_energies[index] + best_energies[index + 1])
        * (cumulative[index + 1] - cumulative[index])
        for index in range(len(best_energies) - 1)
    )
    return float(auc), None, final_energy


def _load_case(summary_path: Path) -> dict[str, Any]:
    label = str(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(isinstance(summary, dict), f"{label}: summary is not a mapping")
    arm = summary.get("arm")
    system = summary.get("system")
    _require(arm in ARMS, f"{label}: unsupported arm")
    _require(isinstance(system, str) and system, f"{label}: system is missing")
    seed = _exact_int(summary.get("seed"), f"{label}: seed")
    total_fe = _exact_int(summary.get("force_evaluations"), f"{label}: force_evaluations", minimum=1)
    _require(
        total_fe == _exact_int(summary.get("total_force_budget"), f"{label}: total_force_budget", minimum=1),
        f"{label}: force-evaluation budget does not close",
    )
    _require(
        total_fe == PRODUCTION_TOTAL_FORCE_BUDGET,
        f"{label}: formal evidence requires exactly {PRODUCTION_TOTAL_FORCE_BUDGET} force evaluations",
    )
    purpose_counts = summary.get("purpose_counts")
    _require(isinstance(purpose_counts, dict), f"{label}: purpose_counts is missing")
    for purpose, count in purpose_counts.items():
        _exact_int(count, f"{label}: purpose_counts.{purpose}")
    _require(sum(purpose_counts.values()) == total_fe, f"{label}: purpose counts do not close")
    _require(purpose_counts.get("unattributed", 0) == 0, f"{label}: unattributed force evaluations")
    _require("direction_oracle" in purpose_counts, f"{label}: direction_oracle is missing")
    stats = summary.get("stats")
    _require(isinstance(stats, dict), f"{label}: stats is missing")
    trials = _exact_int(stats.get("n_trials"), f"{label}: n_trials")
    selections = _exact_int(stats.get("direction_choices"), f"{label}: direction_choices")
    unique_minima = _exact_int(stats.get("n_minima"), f"{label}: n_minima", minimum=1)
    _require(
        _exact_int(stats.get("budget_exhausted"), f"{label}: budget_exhausted") == 1,
        f"{label}: did not terminate at the force budget",
    )
    audit = summary.get("direction_selection_audit")
    _require(isinstance(audit, dict), f"{label}: direction_selection_audit is missing")
    _require(
        _exact_int(audit.get("selection_count"), f"{label}: selection_count") == selections,
        f"{label}: selection count does not close",
    )
    _require(
        _exact_int(audit.get("direction_oracle_force_evaluations"), f"{label}: direction FE")
        == purpose_counts["direction_oracle"],
        f"{label}: direction-oracle audit does not match purpose ledger",
    )
    selected_kind_counts = audit.get("selected_kind_counts")
    _require(
        isinstance(selected_kind_counts, dict),
        f"{label}: selected_kind_counts is missing",
    )
    for kind, count in selected_kind_counts.items():
        _exact_int(count, f"{label}: selected_kind_counts.{kind}")
    if arm == "rayleigh_ritz":
        _require(
            _exact_int(selected_kind_counts.get("ritz", 0), f"{label}: selected_kind_counts.ritz") > 0,
            f"mode_not_exercised: {label}: rayleigh_ritz arm selected no ritz direction",
        )
    _require(summary.get("paired_arm_config_diff") == PAIR_DIFF, f"{label}: paired config diff is not selection-only")
    protocol = summary.get("plain_ritz_hvp_contract")
    _require(isinstance(protocol, dict), f"{label}: plain Ritz HVP contract is missing")
    _require(protocol.get("same_native_candidate_and_hvp_protocol") is True, f"{label}: native HVP protocol is not paired")
    _require(protocol.get("runtime_cost_source") == "purpose_counts.direction_oracle", f"{label}: direction FE cost source is invalid")
    _require(summary.get("termination", {}).get("reason") == "force_budget_exhausted", f"{label}: termination reason is invalid")
    trace_path = summary_path.with_name("energy_trace.json")
    _require(trace_path.is_file(), f"{label}: energy_trace.json is missing")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    _require(isinstance(trace, list), f"{label}: energy trace is not a list")
    auc, auc_reason, final_energy = _energy_auc(trace, label)
    initial = _finite(summary.get("initial_energy_eV"), f"{label}: initial_energy_eV")
    best = _finite(summary.get("best_energy_eV"), f"{label}: best_energy_eV")
    drop = _finite(summary.get("energy_drop_eV"), f"{label}: energy_drop_eV")
    _require(math.isclose(drop, initial - best, rel_tol=1e-12, abs_tol=1e-12), f"{label}: energy drop does not close")
    duplicate_rate = None
    if "duplicate_rate" in stats:
        duplicate_rate = _finite(stats["duplicate_rate"], f"{label}: duplicate_rate")
    duplicates, duplicates_reason = _optional_exact_int(stats, "duplicate_count")
    failures, failures_reason = _optional_exact_int(stats, "failure_count")
    damage, damage_reason = _optional_exact_int(stats, "trust_damage_events")
    escape = purpose_counts.get("escape_true_pes_check")
    if escape is not None:
        escape = _exact_int(escape, f"{label}: escape_true_pes_check")
    base = summary.get("base_preflight")
    _require(isinstance(base, dict), f"{label}: base_preflight is missing")
    return {
        "run": str(summary_path.parent),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "energy_trace": str(trace_path),
        "energy_trace_sha256": _sha256(trace_path),
        "arm": arm,
        "system": system,
        "seed": seed,
        "initial_energy_eV": initial,
        "best_energy_eV": best,
        "final_energy_eV": final_energy,
        "energy_drop_eV": drop,
        "best_energy_auc_eV_force_evals": auc,
        "best_energy_auc_reason": auc_reason,
        "total_force_evaluations": total_fe,
        "purpose_counts": purpose_counts,
        "direction_force_evaluations": purpose_counts["direction_oracle"],
        "direction_fe_fraction": purpose_counts["direction_oracle"] / total_fe,
        "escape_true_pes_check_force_evaluations": escape,
        "trials": trials,
        "selection_count": selections,
        "unique_minima": unique_minima,
        "duplicate_rate": duplicate_rate,
        "duplicates": duplicates,
        "duplicates_reason": duplicates_reason,
        "failures": failures,
        "failures_reason": failures_reason,
        "damage_events": damage,
        "damage_events_reason": damage_reason,
        "wall_time_s": _finite(summary.get("timing", {}).get("total_wall_time_s"), f"{label}: wall time"),
        "termination_reason": summary["termination"]["reason"],
        "_effective_config": _normalise_config(summary["effective_config"]),
        "_source_config": _normalise_config(summary["source_config"]),
        "_runtime_identity": {
            "execution_commit": base.get("execution_commit"),
            "input_sha256": base.get("input_sha256"),
            "model_sha256": base.get("model_sha256"),
            "runtime_versions": base.get("runtime_versions"),
            "cuda": base.get("cuda"),
            "calculator": base.get("calculator"),
        },
        "_protocol": protocol,
    }


def build_evidence(input_dir: Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    summaries = sorted(input_dir.rglob("summary.json"))
    _require(summaries, f"no completed case summaries under {input_dir}")
    cases = [_load_case(path) for path in summaries]
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for case in cases:
        key = (case["system"], case["seed"])
        arms = grouped.setdefault(key, {})
        _require(case["arm"] not in arms, f"duplicate completed arm for {key}: {case['arm']}")
        arms[case["arm"]] = case
    pairs: list[dict[str, Any]] = []
    public_runs: list[dict[str, Any]] = []
    for (system, seed), arms in sorted(grouped.items()):
        _require(set(arms) == set(ARMS), f"incomplete pair for {system} seed {seed}")
        discrete = arms["discrete"]
        ritz = arms["rayleigh_ritz"]
        _require(discrete["_source_config"] == ritz["_source_config"], f"{system} seed {seed}: source config mismatch")
        effective_diff = {
            key: [discrete["_effective_config"][key], ritz["_effective_config"][key]]
            for key in sorted(discrete["_effective_config"])
            if discrete["_effective_config"][key] != ritz["_effective_config"][key]
        }
        _require(effective_diff == PAIR_DIFF, f"{system} seed {seed}: effective pair differs beyond selection mode")
        _require(discrete["_runtime_identity"] == ritz["_runtime_identity"], f"{system} seed {seed}: runtime/model/input identity mismatch")
        _require(discrete["total_force_evaluations"] == ritz["total_force_evaluations"], f"{system} seed {seed}: total FE mismatch")
        _require(discrete["_protocol"]["matched_fields"] == ritz["_protocol"]["matched_fields"], f"{system} seed {seed}: native candidate/HVP protocol mismatch")
        public_arms = {}
        for arm, case in arms.items():
            public = {key: value for key, value in case.items() if not key.startswith("_")}
            public_arms[arm] = public
            public_runs.append(public)
        pairs.append(
            {
                "system": system,
                "seed": seed,
                "paired_total_force_budget": discrete["total_force_evaluations"],
                "effective_config_diff": effective_diff,
                "arms": public_arms,
                "delta_rayleigh_ritz_minus_discrete": {
                    key: ritz[key] - discrete[key]
                    for key in (
                        "best_energy_eV",
                        "final_energy_eV",
                        "energy_drop_eV",
                        "direction_force_evaluations",
                        "trials",
                        "selection_count",
                        "unique_minima",
                        "wall_time_s",
                    )
                },
            }
        )
    return {
        "schema_version": 1,
        "provenance": {"analyzer_sha256": _sha256(Path(__file__).resolve())},
        "scope": {
            "input_dir": str(input_dir),
            "completed_case_summaries": len(cases),
            "paired_system_seed_cases": len(pairs),
            "claim_ceiling": "selection_representation_under_matched_native_candidate_and_hvp_protocol_only_with_projected_native_true_hvp_curvature",
        },
        "validated_contract": {
            "paired_effective_config_diff": PAIR_DIFF,
            "formal_total_force_budget": PRODUCTION_TOTAL_FORCE_BUDGET,
            "same_native_candidate_and_hvp_protocol": True,
            "direction_cost_source": "purpose_counts.direction_oracle",
            "legacy_candidate_count_includes_zero_hvp_synthetic_ritz": True,
            "escape_true_pes_check_reported_when_present": True,
        },
        "runs": public_runs,
        "pairs": pairs,
    }


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    rows = []
    for pair in evidence["pairs"]:
        delta = pair["delta_rayleigh_ritz_minus_discrete"]
        discrete = pair["arms"]["discrete"]
        ritz = pair["arms"]["rayleigh_ritz"]
        rows.append(
            "| {system} | {seed} | {best:+.6f} | {drop:+.6f} | {direction_fe:+d} | {trials:+d} | {wall:+.3f} | {escape_d} / {escape_r} |".format(
                system=pair["system"].upper(),
                seed=pair["seed"],
                best=delta["best_energy_eV"],
                drop=delta["energy_drop_eV"],
                direction_fe=delta["direction_force_evaluations"],
                trials=delta["trials"],
                wall=delta["wall_time_s"],
                escape_d=discrete["escape_true_pes_check_force_evaluations"],
                escape_r=ritz["escape_true_pes_check_force_evaluations"],
            )
        )
    return "\n".join(
        [
            "# Direction-selection representation evidence",
            "",
            "每一对仅在 `direction_selection_mode` 上不同（`discrete` 对 `rayleigh_ritz`）；"
            "native candidate generation、central HVP 协议、oracle_candidates、starter selector、"
            "safe-LBFGS proposal、uphill policy、true quench、LS softening 与总 FE budget 均经配置比对。",
            "Ritz true curvature 是复用 native central-FD true-HVP 的子空间投影；在非线性 PES 上，"
            "它与 direct mixed-direction central-FD stencil 相差 `O(hvp_epsilon^2)`。真实方向成本取"
            "purpose-resolved `direction_oracle` FE；legacy `candidate_count` 在 Ritz 成功时包含零-HVP "
            "synthetic candidate，不能据此推断 Ritz 的 HVP 成本。",
            "",
            "| system | seed | delta best energy (eV) | delta energy drop (eV) | delta direction FE | delta trials | delta wall (s) | escape true-PES FE (D / RR) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "claim ceiling：本实验只比较相同方向候选/HVP 协议、并使用 projected native true-HVP curvature 的选择表示；"
            "不证明平衡态无偏性，也不证明一般系统优越性。"
            "能量 AUC 仅在 energy trace 提供 cumulative total FE 时报告；缺失时明确标记 unsupported。",
            "",
        ]
    )


def write_evidence(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    evidence = build_evidence(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "conclusion.md").write_text(render_conclusion(evidence), encoding="utf-8")
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = write_evidence(input_dir=args.input_dir, output_dir=args.output_dir)
    print(json.dumps(evidence["scope"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
