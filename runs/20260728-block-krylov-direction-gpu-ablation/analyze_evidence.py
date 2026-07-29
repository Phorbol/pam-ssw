#!/usr/bin/env python3
"""Fail-closed evidence projection for the Stage-2 block-Krylov ablation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
ARMS = ("discrete", "variational_breadth", "balanced_refinement", "deep_refinement")
BLOCK_ARMS = ARMS[1:]
SEEDS = (42, 43, 44)
TOTAL_FORCE_BUDGET = 6000
EVIDENCE_SCHEMA_ID = "block_krylov_stage2_v1"
TIE_BREAK_ORDER = ("deep_refinement", "balanced_refinement", "variational_breadth")


class EvidenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(type(value) is int and value >= minimum, f"{label} must be an exact integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite")
    return float(value)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runner():
    spec = importlib.util.spec_from_file_location("_block_krylov_ablation_runner", RUNNER_PATH)
    _require(spec is not None and spec.loader is not None, f"cannot load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _energy_auc(trace: Sequence[Mapping[str, Any]], label: str) -> float:
    _require(bool(trace), f"{label}: energy trace is empty")
    cumulative: list[int] = []
    best: list[float] = []
    running_best: float | None = None
    for index, point in enumerate(trace):
        _require(isinstance(point, Mapping), f"{label}: energy trace row {index} is not a mapping")
        cumulative.append(_exact_int(point.get("cumulative_total_force_evaluations"), f"{label}: cumulative FE {index}"))
        energy = _finite(point.get("energy_eV"), f"{label}: energy {index}")
        reported_best = _finite(point.get("best_energy_eV"), f"{label}: best energy {index}")
        running_best = energy if running_best is None else min(running_best, energy)
        _require(math.isclose(reported_best, running_best, rel_tol=1e-12, abs_tol=1e-12), f"{label}: best-energy trace does not close")
        best.append(running_best)
    _require(cumulative[0] == 0, f"{label}: energy trace must start at cumulative FE 0")
    _require(all(right >= left for left, right in zip(cumulative, cumulative[1:])), f"{label}: cumulative FE is not monotonic")
    return float(sum(
        0.5 * (best[index] + best[index + 1]) * (cumulative[index + 1] - cumulative[index])
        for index in range(len(best) - 1)
    ))


def _load_case(summary_path: Path) -> dict[str, Any]:
    label = str(summary_path)
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), f"{label}: summary is not a mapping")
    arm = raw.get("arm")
    system = raw.get("system")
    _require(arm in ARMS, f"{label}: unsupported arm")
    _require(system in ("c60", "pdo"), f"{label}: unsupported system")
    seed = _exact_int(raw.get("seed"), f"{label}: seed")
    budget = _exact_int(raw.get("total_force_budget"), f"{label}: total force budget", minimum=1)
    _require(budget == TOTAL_FORCE_BUDGET, f"{label}: total force budget is not {TOTAL_FORCE_BUDGET}")
    total = _exact_int(raw.get("force_evaluations"), f"{label}: force evaluations", minimum=1)
    _require(total <= budget, f"{label}: force evaluations exceed budget")
    purposes = raw.get("purpose_counts")
    _require(isinstance(purposes, dict), f"{label}: purpose_counts is missing")
    for purpose, value in purposes.items():
        _exact_int(value, f"{label}: purpose count {purpose!r}")
    _require(sum(purposes.values()) == total, f"{label}: purpose ledger does not close")
    _require(purposes.get("unattributed", 0) == 0, f"{label}: unattributed evaluations")
    _require("direction_oracle" in purposes, f"{label}: direction_oracle is missing")
    trace_path = summary_path.with_name("energy_trace.json")
    _require(trace_path.is_file(), f"{label}: energy_trace.json is missing")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    _require(isinstance(trace, list), f"{label}: energy trace is not a list")
    auc = _energy_auc(trace, label)
    _require(trace[-1]["cumulative_total_force_evaluations"] == total, f"{label}: energy trace does not end at total FE")
    direction_path = summary_path.with_name("direction_trace.jsonl")
    _require(direction_path.is_file(), f"{label}: direction_trace.jsonl is missing")
    rows = []
    for line_number, line in enumerate(direction_path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(f"{label}: invalid direction JSON line {line_number}") from error
            _require(isinstance(row, dict), f"{label}: direction row {line_number} is not a mapping")
            rows.append(row)
    runner = _load_runner()
    try:
        direction_audit = runner.validate_direction_trace(arm=arm, direction_rows=rows)
    except RuntimeError as error:
        raise EvidenceError(f"{label}: {error}") from error
    terminal_untraced = (
        purposes["direction_oracle"]
        - direction_audit["direction_oracle_force_evaluations"]
    )
    _require(
        terminal_untraced >= 0,
        f"{label}: terminal untraced direction FE must be non-negative",
    )
    if terminal_untraced:
        stats = raw.get("stats")
        termination = raw.get("termination")
        _require(total == budget, f"{label}: terminal partial selection requires exact budget exhaustion")
        _require(
            isinstance(stats, dict) and stats.get("budget_exhausted") == 1,
            f"{label}: terminal partial selection requires budget_exhausted stats",
        )
        _require(
            isinstance(termination, dict) and termination.get("budget_exhausted") is True,
            f"{label}: terminal partial selection requires budget exhaustion termination",
        )
        _require(
            terminal_untraced
            < direction_audit["maximum_complete_selection_force_evaluations"],
            f"{label}: terminal partial selection is not smaller than one complete selection",
        )
    direction_audit["terminal_untraced_direction_oracle_force_evaluations"] = terminal_untraced
    _require(
        direction_audit["direction_oracle_force_evaluations"] + terminal_untraced
        == purposes["direction_oracle"],
        f"{label}: completed trace plus terminal partial direction FE does not close",
    )
    effective = raw.get("effective_config")
    source = raw.get("source_config")
    _require(isinstance(effective, dict) and isinstance(source, dict), f"{label}: configs are missing")
    expected = runner.ARMS[arm]
    for key, value in expected.items():
        _require(effective.get(key) == value, f"{label}: effective config missed {key}={value!r}")
    initial = _finite(raw.get("initial_energy_eV"), f"{label}: initial energy")
    best = _finite(raw.get("best_energy_eV"), f"{label}: best energy")
    drop = _finite(raw.get("energy_drop_eV"), f"{label}: energy drop")
    _require(math.isclose(drop, initial - best, rel_tol=1e-12, abs_tol=1e-12), f"{label}: energy drop does not close")
    archive_size = _exact_int(raw.get("archive_size"), f"{label}: archive size", minimum=1)
    unique_minima = _exact_int(raw.get("unique_minima"), f"{label}: unique minima", minimum=1)
    duplicate_fraction = _finite(raw.get("duplicate_fraction"), f"{label}: duplicate fraction")
    _require(0.0 <= duplicate_fraction <= 1.0, f"{label}: duplicate fraction outside [0, 1]")
    base = raw.get("base_preflight")
    _require(isinstance(base, dict), f"{label}: runtime provenance is missing")
    return {
        "arm": arm,
        "system": system,
        "seed": seed,
        "run": str(summary_path.parent),
        "summary_sha256": _sha256(summary_path),
        "energy_trace_sha256": _sha256(trace_path),
        "initial_energy_eV": initial,
        "best_energy_eV": best,
        "energy_drop_eV": drop,
        "best_energy_auc_eV_force_evals": auc,
        "archive_size": archive_size,
        "unique_minima": unique_minima,
        "duplicate_fraction": duplicate_fraction,
        "terminal_failure_count": raw.get("terminal_failure_count"),
        "terminal_failure_count_reason": raw.get("terminal_failure_count_reason"),
        "total_force_evaluations": total,
        "budget_closed": True,
        "purpose_counts": purposes,
        "terminal_untraced_direction_oracle_force_evaluations": terminal_untraced,
        "direction_selection_audit": direction_audit,
        "wall_time_s": _finite(raw.get("timing", {}).get("total_wall_time_s"), f"{label}: wall time"),
        "_effective_config": runner._normalise_config(effective),
        "_source_config": runner._normalise_config(source),
        "_runtime_identity": {
            "execution_commit": base.get("execution_commit"),
            "input_sha256": base.get("input_sha256"),
            "model_sha256": base.get("model_sha256"),
            "runtime_versions": base.get("runtime_versions"),
            "cuda": base.get("cuda"),
            "calculator": base.get("calculator"),
        },
    }


def _public(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in case.items() if not key.startswith("_")}


def _validate_case_shape(case: Mapping[str, Any], *, system: str) -> None:
    _require(case.get("system") == system, "case system does not match requested cohort")
    _require(case.get("arm") in ARMS, "case arm is unsupported")
    _exact_int(case.get("seed"), "case seed")
    _finite(case.get("best_energy_eV"), "case best energy")
    _finite(case.get("best_energy_auc_eV_force_evals"), "case best-energy AUC")
    _exact_int(case.get("archive_size"), "case archive size", minimum=1)
    _require(case.get("budget_closed") is True, "case budget is not closed")
    _require(case.get("total_force_evaluations") == TOTAL_FORCE_BUDGET, "case total FE is not preregistered budget")
    for key in ("_source_config", "_effective_config", "_runtime_identity"):
        _require(isinstance(case.get(key), Mapping), f"case {key} is missing")


def _validate_cohort_contract(cases: Sequence[Mapping[str, Any]]) -> None:
    runner = _load_runner()
    reference_runtime = cases[0]["_runtime_identity"]
    reference_source = cases[0]["_source_config"]
    reference_effective = cases[0]["_effective_config"]
    allowed_effective = {
        "direction_selection_mode",
        "block_krylov_blocks",
        "block_krylov_depth",
        "rng_seed",
    }
    for case in cases:
        _require(case["_runtime_identity"] == reference_runtime, "cohort runtime identity mismatch")
        _require(case["_source_config"] == reference_source, "cohort source config mismatch")
        effective = case["_effective_config"]
        _require(effective.keys() == reference_effective.keys(), "cohort effective config fields mismatch")
        changed = {
            key
            for key in effective
            if effective[key] != reference_effective[key]
        }
        _require(changed <= allowed_effective, "cohort effective config differs beyond preregistered direction fields")
        expected = runner.ARMS[str(case["arm"])]
        for key, value in expected.items():
            _require(effective.get(key) == value, f"effective config missed {key}={value!r}")
        _require(effective.get("rng_seed") == case["seed"], "effective config rng_seed mismatch")
        _require(effective.get("max_force_evals") == TOTAL_FORCE_BUDGET, "effective config force budget mismatch")
        source = case["_source_config"]
        _require(source.keys() == effective.keys(), "source/effective config fields mismatch")
        source_changed = {key for key in source if source[key] != effective[key]}
        _require(
            source_changed <= allowed_effective | {"max_force_evals"},
            "source/effective config differs beyond preregistered fields",
        )


def build_evidence_from_cases(
    cases: Sequence[Mapping[str, Any]], *, system: str, selected_pdo_transfer_arm: str | None = None
) -> dict[str, Any]:
    """Build the survivor decision from already-validated public case records."""

    _require(system in ("c60", "pdo"), "unsupported cohort")
    for case in cases:
        _validate_case_shape(case, system=system)
    _require(bool(cases), "cohort is empty")
    _validate_cohort_contract(cases)
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for case in cases:
        seed = int(case["seed"])
        arms = grouped.setdefault(seed, {})
        _require(case["arm"] not in arms, f"duplicate arm for seed {seed}")
        arms[str(case["arm"])] = case
    if system == "c60":
        _require(set(grouped) == set(SEEDS), "C60 cohort must contain exactly seeds 42, 43, 44")
        for seed, arms in grouped.items():
            _require(set(arms) == set(ARMS), f"C60 seed {seed} must contain exactly the four preregistered arms")
        arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in BLOCK_ARMS}
        for seed in SEEDS:
            discrete = grouped[seed]["discrete"]
            for arm in BLOCK_ARMS:
                block = grouped[seed][arm]
                delta = _finite(block["best_energy_eV"], "block best") - _finite(discrete["best_energy_eV"], "discrete best")
                auc_improvement = _finite(discrete["best_energy_auc_eV_force_evals"], "discrete auc") - _finite(block["best_energy_auc_eV_force_evals"], "block auc")
                coverage_ratio = _exact_int(block["archive_size"], "block archive") / max(_exact_int(discrete["archive_size"], "discrete archive"), 1)
                arm_rows[arm].append(
                    {
                        "seed": seed,
                        "delta_block_minus_discrete_best_energy_eV": delta,
                        "improved": delta < 0.0,
                        "paired_best_energy_auc_improvement_eV_force_evals": auc_improvement,
                        "coverage_ratio": coverage_ratio,
                        "coverage_ok": coverage_ratio >= 0.8,
                        "budget_closed": bool(block["budget_closed"] and discrete["budget_closed"]),
                    }
                )
        survivors: list[str] = []
        arm_summary: dict[str, dict[str, Any]] = {}
        for arm in BLOCK_ARMS:
            rows = arm_rows[arm]
            survives = sum(row["improved"] for row in rows) >= 2 and all(row["coverage_ok"] for row in rows) and all(row["budget_closed"] for row in rows)
            arm_summary[arm] = {
                "paired_rows": rows,
                "improvements": sum(row["improved"] for row in rows),
                "survives_c60": survives,
                "median_paired_best_energy_delta_eV": statistics.median(row["delta_block_minus_discrete_best_energy_eV"] for row in rows),
                "median_paired_best_energy_auc_improvement_eV_force_evals": statistics.median(row["paired_best_energy_auc_improvement_eV_force_evals"] for row in rows),
            }
            if survives:
                survivors.append(arm)
        order = {arm: index for index, arm in enumerate(TIE_BREAK_ORDER)}
        selected = min(
            survivors,
            key=lambda arm: (
                arm_summary[arm]["median_paired_best_energy_delta_eV"],
                -arm_summary[arm]["median_paired_best_energy_auc_improvement_eV_force_evals"],
                order[arm],
            ),
        ) if survivors else None
        return {
            "schema_version": 1,
            "schema_id": EVIDENCE_SCHEMA_ID,
            "system": "c60",
            "cohort": {"arms": list(ARMS), "seeds": list(SEEDS), "completed_cases": len(cases)},
            "c60_arm_results": arm_summary,
            "c60_survivors": survivors,
            "selected_pdo_transfer_arm": selected,
            "pdo_status": "not_run_pending_selected_c60_survivor" if selected else "not_run_no_c60_survivor",
            "claim_ceiling": "three seeds are a preregistered survivor gate, not a significance claim; production default remains discrete",
            "runs": [_public(grouped[seed][arm]) for seed in SEEDS for arm in ARMS],
        }
    _require(selected_pdo_transfer_arm in BLOCK_ARMS, "PDO analysis requires analyzer-produced selected_pdo_transfer_arm")
    expected_arms = {"discrete", selected_pdo_transfer_arm}
    for seed, arms in grouped.items():
        _require(set(arms) == expected_arms, f"unexpected PdO arm; expected selected_pdo_transfer_arm {selected_pdo_transfer_arm}")
    _require(set(grouped) == set(SEEDS), "PdO cohort must contain exactly seeds 42, 43, 44")
    return {
        "schema_version": 1,
        "schema_id": EVIDENCE_SCHEMA_ID,
        "system": "pdo",
        "cohort": {"arms": ["discrete", selected_pdo_transfer_arm], "seeds": list(SEEDS), "completed_cases": len(cases)},
        "selected_pdo_transfer_arm": selected_pdo_transfer_arm,
        "pdo_status": "completed_selected_c60_survivor",
        "runs": [_public(grouped[seed][arm]) for seed in SEEDS for arm in ("discrete", selected_pdo_transfer_arm)],
    }


def build_evidence(*, input_dir: Path, system: str, selected_pdo_transfer_arm: str | None = None) -> dict[str, Any]:
    summaries = sorted(Path(input_dir).rglob("summary.json"))
    _require(summaries, f"no completed case summaries under {input_dir}")
    cases = [case for path in summaries if (case := _load_case(path))["system"] == system]
    _require(cases, f"no completed {system} case summaries under {input_dir}")
    if system == "c60":
        _require(selected_pdo_transfer_arm is None, "manual PDO arm selection is forbidden")
        return build_evidence_from_cases(cases, system="c60") | {
            "provenance": {"analyzer_sha256": _sha256(Path(__file__).resolve())},
        }
    _require(selected_pdo_transfer_arm in BLOCK_ARMS, "PDO requires selected_pdo_transfer_arm from C60 evidence")
    return build_evidence_from_cases(cases, system="pdo", selected_pdo_transfer_arm=selected_pdo_transfer_arm) | {
        "provenance": {"analyzer_sha256": _sha256(Path(__file__).resolve())}
    }


def write_evidence(*, input_dir: Path, output_dir: Path, system: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if system == "c60":
        evidence = build_evidence(input_dir=input_dir, system="c60")
    else:
        existing_path = output_dir / "evidence.json"
        _require(existing_path.is_file(), "PdO analysis requires committed C60 evidence.json")
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        _require(existing.get("schema_id") == EVIDENCE_SCHEMA_ID, "PdO analysis requires current C60 evidence schema")
        _require(
            existing.get("provenance", {}).get("analyzer_sha256")
            == _sha256(Path(__file__).resolve()),
            "PdO analysis requires evidence from the current analyzer",
        )
        selected = existing.get("selected_pdo_transfer_arm")
        _require(selected in BLOCK_ARMS, "PdO analysis has no analyzer-produced C60 survivor")
        _require(existing.get("pdo_status") == "not_run_pending_selected_c60_survivor", "PdO transfer has already been resolved or was not allowed")
        pdo = build_evidence(input_dir=input_dir, system="pdo", selected_pdo_transfer_arm=selected)
        evidence = dict(existing)
        evidence["pdo_transfer"] = pdo
        evidence["pdo_status"] = pdo["pdo_status"]
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_temporary = output_dir / ".evidence.json.tmp"
    conclusion_temporary = output_dir / ".conclusion.md.tmp"
    try:
        evidence_temporary.write_text(
            json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        conclusion_temporary.write_text(render_conclusion(evidence), encoding="utf-8")
        evidence_temporary.replace(output_dir / "evidence.json")
        conclusion_temporary.replace(output_dir / "conclusion.md")
    except BaseException:
        evidence_temporary.unlink(missing_ok=True)
        conclusion_temporary.unlink(missing_ok=True)
        raise
    return evidence


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    selected = evidence.get("selected_pdo_transfer_arm")
    status = evidence.get("pdo_status")
    return "\n".join(
        [
            "# Block-Krylov direction ablation evidence",
            "",
            f"C60 survivors: {', '.join(evidence.get('c60_survivors', [])) or 'none'}.",
            f"Selected PdO transfer arm: {selected or 'none'}. PdO status: {status}.",
            "The C60 3-seed rule is a preregistered survivor gate, not a significance claim.",
            "Production default remains `discrete` regardless of this experimental result.",
            "",
        ]
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=("c60", "pdo"))
    parser.add_argument("--input-dir", type=Path, default=RUN_ROOT / "cases")
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = write_evidence(input_dir=args.input_dir, output_dir=args.output_dir, system=args.system)
    print(json.dumps({"system": args.system, "pdo_status": evidence.get("pdo_status")}, sort_keys=True))


if __name__ == "__main__":
    main()
