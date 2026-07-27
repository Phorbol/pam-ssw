#!/usr/bin/env python3
"""Audit a five-trial no-loss run without hiding numerical non-reproducibility."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_PATH = RUN_ROOT / "reference.json"
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
MAX_TRIALS = 5
_TRACE_ARTIFACTS = ("energy_trace", "walk_records", "direction_trace")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_reference(reference_path: Path) -> dict[str, Any]:
    reference = _read_json(reference_path)
    if reference.get("schema_version") != 1 or reference.get("trial_count") != MAX_TRIALS:
        raise ValueError("reference schema or trial count mismatch")
    if set(reference.get("systems", ())) != set(SYSTEMS):
        raise ValueError("reference systems mismatch")
    for system in SYSTEMS:
        source = reference["systems"][system]
        if not isinstance(source.get("old_execution_commit"), str):
            raise ValueError(f"{system} reference lacks old execution commit")
        if set(source.get("old_artifact_sha256", ())) != {
            "summary",
            "energy_trace",
            "walk_records",
            "direction_trace",
        }:
            raise ValueError(f"{system} reference lacks old artifact hashes")
        if any(not isinstance(value, str) or len(value) != 64 for value in source["old_artifact_sha256"].values()):
            raise ValueError(f"{system} reference contains malformed old artifact hashes")
    return reference


def _consecutive_step_edges(direction_trace: list[dict[str, Any]]) -> int:
    last_step: dict[tuple[int, int], int] = {}
    edges = 0
    for row in direction_trace:
        key = (int(row["trial"]), int(row["proposal"]))
        step = int(row["step"])
        if last_step.get(key) == step - 1:
            edges += 1
        last_step[key] = step
    return edges


def _validate_native_candidate_only(summary: dict[str, Any]) -> None:
    config = summary["effective_config"]
    required = {
        "max_trials": MAX_TRIALS,
        "direction_selection_mode": "discrete",
        "direction_synthesis_mode": "none",
        "direction_probe_enabled": False,
        "plateau_evolution_enabled": False,
    }
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"validation config mismatch: {key}")


def _case_paths(output_root: Path, system: str) -> dict[str, Path]:
    case_dir = output_root / system
    return {
        "summary": case_dir / "summary.json",
        "energy_trace": case_dir / "energy_trace.json",
        "walk_records": case_dir / "walk_records.json",
        "direction_trace": case_dir / "direction_trace.jsonl",
    }


def _load_run(output_root: Path, system: str) -> dict[str, Any]:
    paths = _case_paths(output_root, system)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {
        "paths": paths,
        "summary": _read_json(paths["summary"]),
        "energy_trace": _read_json(paths["energy_trace"]),
        "walk_records": _read_json(paths["walk_records"]),
        "direction_trace": _read_jsonl(paths["direction_trace"]),
    }


def _energy(row: dict[str, Any], field: str, *, label: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} lacks finite {field}") from error
    if not isfinite(value):
        raise ValueError(f"{label} lacks finite {field}")
    return value


def _first_sequence_mismatch(
    artifact: str,
    reference_rows: list[dict[str, Any]],
    current_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for index, (reference_row, current_row) in enumerate(zip(reference_rows, current_rows)):
        if reference_row != current_row:
            return {
                "artifact": artifact,
                "index": index,
                "reference": reference_row,
                "current": current_row,
            }
    if len(reference_rows) != len(current_rows):
        index = min(len(reference_rows), len(current_rows))
        return {
            "artifact": artifact,
            "index": index,
            "reference": reference_rows[index] if index < len(reference_rows) else None,
            "current": current_rows[index] if index < len(current_rows) else None,
        }
    return None


def _energy_trace_summary(
    reference_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not reference_rows or not current_rows:
        raise ValueError("energy trace is empty")
    energy_deltas: list[float] = []
    best_energy_deltas: list[float] = []
    different_rows = abs(len(reference_rows) - len(current_rows))
    for index, (reference_row, current_row) in enumerate(zip(reference_rows, current_rows)):
        reference_energy = _energy(reference_row, "energy_eV", label=f"reference energy trace row {index}")
        current_energy = _energy(current_row, "energy_eV", label=f"current energy trace row {index}")
        reference_best = _energy(reference_row, "best_energy_eV", label=f"reference energy trace row {index}")
        current_best = _energy(current_row, "best_energy_eV", label=f"current energy trace row {index}")
        energy_deltas.append(current_energy - reference_energy)
        best_energy_deltas.append(current_best - reference_best)
        if reference_row != current_row:
            different_rows += 1
    initial_reference = _energy(reference_rows[0], "energy_eV", label="reference initial energy")
    initial_current = _energy(current_rows[0], "energy_eV", label="current initial energy")
    return {
        "exact": reference_rows == current_rows,
        "reference_rows": len(reference_rows),
        "current_rows": len(current_rows),
        "different_rows": different_rows,
        "first_difference": _first_sequence_mismatch("energy_trace", reference_rows, current_rows),
        "initial_reference_energy_eV": initial_reference,
        "initial_current_energy_eV": initial_current,
        "initial_energy_delta_eV": initial_current - initial_reference,
        "final_energy_delta_eV": energy_deltas[-1],
        "final_best_energy_delta_eV": best_energy_deltas[-1],
        "max_abs_energy_delta_eV": max(abs(delta) for delta in energy_deltas),
        "max_abs_best_energy_delta_eV": max(abs(delta) for delta in best_energy_deltas),
    }


def _first_direction_summary(
    reference_rows: list[dict[str, Any]], current_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not reference_rows or not current_rows:
        return {
            "available": False,
            "reference_available": bool(reference_rows),
            "current_available": bool(current_rows),
        }
    reference = reference_rows[0]
    current = current_rows[0]
    reference_curvature = _energy(reference, "curvature", label="reference first direction")
    current_curvature = _energy(current, "curvature", label="current first direction")
    return {
        "available": True,
        "selected_kind_match": reference.get("selected_kind") == current.get("selected_kind"),
        "candidate_count_match": reference.get("candidate_count") == current.get("candidate_count"),
        "reference_curvature": reference_curvature,
        "current_curvature": current_curvature,
        "curvature_delta": current_curvature - reference_curvature,
    }


def _comparison(reference: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    energy_trace = _energy_trace_summary(reference["energy_trace"], current["energy_trace"])
    first_mismatch = energy_trace["first_difference"]
    for artifact in ("walk_records", "direction_trace"):
        if first_mismatch is None:
            first_mismatch = _first_sequence_mismatch(
                artifact,
                reference[artifact],
                current[artifact],
            )
    trajectory_exact = all(reference[artifact] == current[artifact] for artifact in _TRACE_ARTIFACTS)
    return {
        "trajectory_exact": trajectory_exact,
        "initial_energy_delta_eV": energy_trace["initial_energy_delta_eV"],
        "first_mismatch": first_mismatch,
        "energy_trace": energy_trace,
        "first_direction": _first_direction_summary(
            reference["direction_trace"], current["direction_trace"]
        ),
    }


def _validate_current_run(
    run: dict[str, Any], expected: dict[str, Any], system: str
) -> dict[str, Any]:
    summary = run["summary"]
    if summary.get("system") != system:
        raise ValueError(f"{system} summary identity mismatch")
    if summary.get("stats", {}).get("n_trials") != MAX_TRIALS:
        raise ValueError(f"{system} did not complete five trials")
    if summary.get("old_execution_commit") != expected["old_execution_commit"]:
        raise ValueError(f"{system} old execution provenance mismatch")
    if summary.get("old_artifact_sha256") != expected["old_artifact_sha256"]:
        raise ValueError(f"{system} old artifact provenance mismatch")
    if not isinstance(summary.get("execution_commit"), str):
        raise ValueError(f"{system} lacks execution commit")
    _validate_native_candidate_only(summary)
    if len(run["energy_trace"]) != MAX_TRIALS + 1:
        raise ValueError(f"{system} energy trace does not contain five trials")
    if len(run["walk_records"]) != MAX_TRIALS:
        raise ValueError(f"{system} walk records do not contain five trials")

    purpose_counts = {key: int(value) for key, value in summary["purpose_counts"].items()}
    force_evaluations = int(summary["force_evaluations"])
    if sum(purpose_counts.values()) != force_evaluations:
        raise ValueError(f"{system} purpose accounting does not close")
    if purpose_counts.get("unattributed") != 0:
        raise ValueError(f"{system} purpose accounting contains unattributed evaluations")
    candidate_count = sum(int(row["candidate_count"]) for row in run["direction_trace"])
    if purpose_counts.get("direction_oracle") != 2 * candidate_count:
        raise ValueError(f"{system} direction oracle accounting does not match its central-HVP trace")
    mechanism_savings = {
        "true_curvature_hvp": 2 * len(run["direction_trace"]),
        "true_after_carry": _consecutive_step_edges(run["direction_trace"]),
    }
    escape_savings = sum(mechanism_savings.values())
    baseline_counts = dict(purpose_counts)
    baseline_counts["escape_true_pes_check"] += escape_savings
    savings = {key: baseline_counts[key] - value for key, value in purpose_counts.items()}
    if any(value != 0 for key, value in savings.items() if key != "escape_true_pes_check"):
        raise ValueError(f"{system} no-loss accounting claims a non-escape saving")
    if savings["escape_true_pes_check"] != escape_savings:
        raise ValueError(f"{system} escape saving does not match trace-derived mechanisms")
    return {
        "execution_commit": summary["execution_commit"],
        "new_raw_sha256": {name: _sha256(path) for name, path in run["paths"].items()},
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "counterfactual_baseline_purpose_counts": baseline_counts,
        "mechanism_savings": mechanism_savings,
        "savings": savings,
    }


def _analyze_system(
    reference: dict[str, Any], output_root: Path, repeat_root: Path | None, system: str
) -> dict[str, Any]:
    expected = reference["systems"][system]
    current = _load_run(output_root, system)
    current_validation = _validate_current_run(current, expected, system)
    old_vs_current = _comparison(expected, current)

    repeat_validation = None
    current_vs_repeat = None
    if repeat_root is not None and (repeat_root / system).is_dir():
        repeat = _load_run(repeat_root, system)
        repeat_validation = _validate_current_run(repeat, expected, system)
        if repeat_validation["execution_commit"] != current_validation["execution_commit"]:
            raise ValueError(f"{system} repeat execution commit differs from current run")
        current_vs_repeat = _comparison(current, repeat)

    exact_gpu_trajectory_claim_supported = bool(
        old_vs_current["trajectory_exact"]
        and current_vs_repeat is not None
        and current_vs_repeat["trajectory_exact"]
    )
    return {
        "trajectory_exact": old_vs_current["trajectory_exact"],
        "exact_gpu_trajectory_claim_supported": exact_gpu_trajectory_claim_supported,
        "old_execution_commit": expected["old_execution_commit"],
        "old_artifact_sha256": expected["old_artifact_sha256"],
        "comparisons": {
            "old_vs_current": old_vs_current,
            "current_vs_repeat": current_vs_repeat,
        },
        "current_run": current_validation,
        "repeat_run": repeat_validation,
        # Preserve the cost fields at their previous location for concise consumers.
        **{key: current_validation[key] for key in (
            "new_raw_sha256",
            "force_evaluations",
            "purpose_counts",
            "counterfactual_baseline_purpose_counts",
            "mechanism_savings",
            "savings",
        )},
    }


def analyze(
    reference_path: Path,
    output_root: Path,
    repeat_root: Path | None = None,
) -> dict[str, Any]:
    reference = _load_reference(Path(reference_path))
    output_root = Path(output_root)
    if repeat_root is not None and not Path(repeat_root).is_dir():
        raise FileNotFoundError(repeat_root)
    systems = {
        system: _analyze_system(reference, output_root, None if repeat_root is None else Path(repeat_root), system)
        for system in SYSTEMS
    }
    return {
        "schema_version": 2,
        "trial_count": MAX_TRIALS,
        "reference_sha256": _sha256(Path(reference_path)),
        "output_root": str(output_root),
        "repeat_root": None if repeat_root is None else str(repeat_root),
        "trajectory_exact": all(payload["trajectory_exact"] for payload in systems.values()),
        "exact_gpu_trajectory_claim_supported": all(
            payload["exact_gpu_trajectory_claim_supported"] for payload in systems.values()
        ),
        "cost_baseline_scope": (
            "counterfactual reconstruction from each current run's exact direction trace; "
            "the old 200-trial output did not persist per-trial purpose counts"
        ),
        "systems": systems,
    }


def write_conclusion(evidence: dict[str, Any], path: Path) -> None:
    """Write a compact claim boundary next to the machine-readable evidence."""

    lines = [
        "# Five-trial GPU no-loss validation",
        "",
        f"trajectory_exact: {str(evidence['trajectory_exact']).lower()}",
        "exact_gpu_trajectory_claim_supported: "
        f"{str(evidence['exact_gpu_trajectory_claim_supported']).lower()}",
        "",
        "数值差异按 JSON 中记录的 Python float 值逐项比较；没有引入容差。"
        "因此轨迹不一致是测量结果，而不是 analyzer failure。"
        "配置、provenance、purpose closure、unattributed=0 和 central-HVP 核算仍 fail closed。",
        "",
        "| System | old vs current exact | current vs repeat exact | initial delta (old-current, eV) | "
        "initial delta (current-repeat, eV) | escape-only counterfactual FE saving |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        payload = evidence["systems"][system]
        old_vs_current = payload["comparisons"]["old_vs_current"]
        current_vs_repeat = payload["comparisons"]["current_vs_repeat"]
        repeat_exact = "not available" if current_vs_repeat is None else str(current_vs_repeat["trajectory_exact"]).lower()
        repeat_delta = "not available" if current_vs_repeat is None else f"{current_vs_repeat['initial_energy_delta_eV']:.16g}"
        lines.append(
            f"| {'C60' if system == 'c60' else 'PdO'} | {str(old_vs_current['trajectory_exact']).lower()} | {repeat_exact} | "
            f"{old_vs_current['initial_energy_delta_eV']:.16g} | {repeat_delta} | "
            f"{payload['savings']['escape_true_pes_check']} |"
        )
    lines.extend(
        [
            "",
            "结论：这份运行可以支持‘当前实现的 purpose 账本闭合，且按 trace 反事实重建只节省 "
            "`ESCAPE_TRUE_PES_CHECK`’；不能支持 exact GPU trajectory 的声明。",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repeat-root", type=Path)
    parser.add_argument("--evidence", type=Path, default=RUN_ROOT / "evidence.json")
    parser.add_argument("--conclusion", type=Path, default=RUN_ROOT / "conclusion.md")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = analyze(args.reference, args.output_root, args.repeat_root)
    args.evidence.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_conclusion(evidence, args.conclusion)
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
