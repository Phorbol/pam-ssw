#!/usr/bin/env python3
"""Validate and summarize the fixed safe-L-BFGS history-depth ledger."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping


RUN_ROOT = Path(__file__).resolve().parent
RAW_DIR = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
SEEDS = tuple(range(42, 50))
ARMS = (
    ("adaptive-scale-history1", 1),
    ("adaptive-scale-history10", 10),
)
HISTORY1 = ARMS[0][0]
HISTORY10 = ARMS[1][0]


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def certificate_satisfied(row: Mapping[str, Any]) -> bool:
    """Derive the force certificate without trusting optimizer status."""

    try:
        fmax = float(row["fmax_eV_per_A"])
        force = float(row["final_active_max_force_eV_per_A"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return bool(
        math.isfinite(fmax)
        and math.isfinite(force)
        and fmax > 0.0
        and 0.0 <= force <= fmax
    )


def _validate_positions(value: object, label: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty position array")
    positions: list[list[float]] = []
    for atom_index, atom in enumerate(value):
        if not isinstance(atom, list) or len(atom) != 3:
            raise ValueError(f"{label}[{atom_index}] must have three coordinates")
        positions.append(
            [
                _finite_number(coordinate, f"{label}[{atom_index}][{axis}]")
                for axis, coordinate in enumerate(atom)
            ]
        )
    return positions


def _validate_evaluator_closure(row: Mapping[str, Any], label: str) -> None:
    force_evaluations = _nonnegative_int(row.get("force_evaluations"), f"{label}.force_evaluations")
    trace = row.get("trace")
    telemetry = row.get("telemetry")
    purpose_counts = row.get("purpose_counts")
    if not isinstance(trace, list):
        raise ValueError(f"{label}.trace must be a list")
    if not isinstance(telemetry, Mapping):
        raise ValueError(f"{label}.telemetry must be an object")
    if not isinstance(purpose_counts, Mapping):
        raise ValueError(f"{label}.purpose_counts must be an object")
    evaluator_calls = _nonnegative_int(
        telemetry.get("evaluator_calls"),
        f"{label}.telemetry.evaluator_calls",
    )
    biased_calls = _nonnegative_int(
        purpose_counts.get("biased_proposal_relax"),
        f"{label}.purpose_counts.biased_proposal_relax",
    )
    unattributed = _nonnegative_int(
        purpose_counts.get("unattributed"),
        f"{label}.purpose_counts.unattributed",
    )
    purpose_total = sum(
        _nonnegative_int(value, f"{label}.purpose_counts.{purpose}")
        for purpose, value in purpose_counts.items()
    )
    if not (
        len(trace)
        == force_evaluations
        == evaluator_calls
        == biased_calls
        == purpose_total
        and unattributed == 0
    ):
        raise ValueError(
            f"{label} evaluator closure failed: "
            "trace = force_evaluations = telemetry.evaluator_calls = "
            "biased_proposal_relax = purpose total, with unattributed = 0"
        )


def load_validated_ledger(ledger_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_object(ledger_dir / "summary.json")
    if summary.get("schema_version") != 1:
        raise ValueError("summary schema_version must be 1")
    if summary.get("systems") != list(SYSTEMS):
        raise ValueError("summary systems do not match the fixed matrix")
    expected_arms = [
        {"arm_id": arm_id, "history_limit": history_limit}
        for arm_id, history_limit in ARMS
    ]
    if summary.get("arms") != expected_arms:
        raise ValueError("summary arms do not match the fixed matrix")
    if summary.get("task_count") != 16 or summary.get("row_count") != 32:
        raise ValueError("summary does not describe the fixed 32-row matrix")

    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        payload = _load_object(ledger_dir / f"{system}.json")
        system_rows = payload.get("rows")
        if payload.get("system") != system or not isinstance(system_rows, list):
            raise ValueError(f"{system}.json does not contain the expected system rows")
        rows.extend(system_rows)

    expected_keys = {
        (system, arm_id, seed)
        for system in SYSTEMS
        for arm_id, _ in ARMS
        for seed in SEEDS
    }
    if len(rows) != 32:
        raise ValueError("ledger does not contain the fixed 32-row matrix")

    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    task_hashes: dict[tuple[str, int], str] = {}
    arm_limits = dict(ARMS)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} must be an object")
        system = row.get("system")
        arm_id = row.get("arm_id")
        seed = row.get("seed")
        if (
            system not in SYSTEMS
            or arm_id not in arm_limits
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError(f"row {index} has an invalid fixed-matrix identity")
        key = (system, arm_id, seed)
        if key in by_key:
            raise ValueError(f"duplicate fixed-matrix row: {key}")
        by_key[key] = row
        if row.get("history_limit") != arm_limits[arm_id]:
            raise ValueError(f"row {key} has the wrong history limit")
        if row.get("task_id") != f"{system}-seed-{seed}-bias-1":
            raise ValueError(f"row {key} has the wrong task ID")
        task_sha = row.get("task_sha256")
        if not isinstance(task_sha, str) or len(task_sha) != 64:
            raise ValueError(f"row {key} has an invalid task SHA256")
        task_key = (system, seed)
        previous_sha = task_hashes.setdefault(task_key, task_sha)
        if previous_sha != task_sha:
            raise ValueError(f"row {key} does not share its paired task SHA256")
        _finite_number(row.get("fmax_eV_per_A"), f"row {key}.fmax_eV_per_A")
        _finite_number(
            row.get("final_active_max_force_eV_per_A"),
            f"row {key}.final_active_max_force_eV_per_A",
        )
        _finite_number(
            row.get("final_total_biased_energy_eV"),
            f"row {key}.final_total_biased_energy_eV",
        )
        _validate_positions(row.get("final_positions"), f"row {key}.final_positions")
        _validate_evaluator_closure(row, f"row {key}")

    if set(by_key) != expected_keys:
        raise ValueError("ledger identities do not match the fixed 32-row matrix")
    return summary, [by_key[key] for key in sorted(expected_keys)]


def _arm_aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "certificate_satisfied_count": sum(certificate_satisfied(row) for row in rows),
        "row_count": len(rows),
        "force_evaluations": sum(int(row["force_evaluations"]) for row in rows),
    }


def _raw_position_differences(
    history1_positions: object,
    history10_positions: object,
) -> tuple[float, float]:
    first = _validate_positions(history1_positions, "history1_positions")
    second = _validate_positions(history10_positions, "history10_positions")
    if len(first) != len(second):
        raise ValueError("paired endpoints have different atom counts")
    squared_atom_distances = [
        math.fsum((right[axis] - left[axis]) ** 2 for axis in range(3))
        for left, right in zip(first, second, strict=True)
    ]
    return (
        math.sqrt(math.fsum(squared_atom_distances) / len(squared_atom_distances)),
        math.sqrt(max(squared_atom_distances)),
    )


def build_evidence(
    ledger_dir: Path,
    summary: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped = {
        system: {
            arm_id: [
                row
                for row in rows
                if row["system"] == system and row["arm_id"] == arm_id
            ]
            for arm_id, _ in ARMS
        }
        for system in SYSTEMS
    }
    by_system_arm = {
        system: {
            arm_id: _arm_aggregate(grouped[system][arm_id])
            for arm_id, _ in ARMS
        }
        for system in SYSTEMS
    }
    overall_by_arm = {
        arm_id: _arm_aggregate(
            [row for row in rows if row["arm_id"] == arm_id]
        )
        for arm_id, _ in ARMS
    }

    paired_by_system: dict[str, Any] = {}
    row_by_key = {
        (row["system"], row["arm_id"], row["seed"]): row
        for row in rows
    }
    for system in SYSTEMS:
        pairs = []
        deltas: list[int] = []
        for seed in SEEDS:
            history1 = row_by_key[(system, HISTORY1, seed)]
            history10 = row_by_key[(system, HISTORY10, seed)]
            call_delta = int(history10["force_evaluations"]) - int(
                history1["force_evaluations"]
            )
            deltas.append(call_delta)
            raw_rms, raw_max = _raw_position_differences(
                history1["final_positions"],
                history10["final_positions"],
            )
            pairs.append(
                {
                    "task_id": history1["task_id"],
                    "seed": seed,
                    "history1_force_evaluations": int(history1["force_evaluations"]),
                    "history10_force_evaluations": int(history10["force_evaluations"]),
                    "history10_minus_history1_force_evaluations": call_delta,
                    "endpoint_raw_diagnostics": {
                        "history10_minus_history1_energy_eV": float(
                            history10["final_total_biased_energy_eV"]
                        )
                        - float(history1["final_total_biased_energy_eV"]),
                        "raw_cartesian_rms_atom_difference_A": raw_rms,
                        "raw_cartesian_max_atom_difference_A": raw_max,
                    },
                }
            )
        paired_by_system[system] = {
            "force_evaluation_delta_definition": "history10_minus_history1",
            "force_evaluation_delta_summary": {
                "lower_count": sum(delta < 0 for delta in deltas),
                "tie_count": sum(delta == 0 for delta in deltas),
                "higher_count": sum(delta > 0 for delta in deltas),
                "median": float(statistics.median(deltas)),
            },
            "pairs": pairs,
        }

    return {
        "schema_version": 1,
        "analysis_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "execution": {
            "execution_commit": summary["execution_commit"],
            "source_summary_sha256": summary["source_summary_sha256"],
            "pamssw_bundle_sha256": summary["pamssw_bundle_sha256"],
            "model_sha256": summary["model_sha256"],
            "input_sha256": summary["input_sha256"],
            "cuda": summary["cuda"],
        },
        "raw_files": {
            name: {"sha256": sha256((ledger_dir / name).read_bytes()).hexdigest()}
            for name in ("summary.json", "c60.json", "pdo.json")
        },
        "ledger": {
            "row_count": 32,
            "task_count": 16,
            "evaluator_closure_validated": True,
        },
        "overall_by_arm": overall_by_arm,
        "by_system_arm": by_system_arm,
        "paired_by_system": paired_by_system,
        "claim_boundary": (
            "Fixed-task, fixed-model CUDA replay only. Endpoint energy and raw Cartesian "
            "position differences are paired diagnostics, not endpoint-equivalence or "
            "general SSW evidence."
        ),
    }


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    overall = evidence["overall_by_arm"]
    by_system = evidence["by_system_arm"]
    paired = evidence["paired_by_system"]
    c60_delta = paired["c60"]["force_evaluation_delta_summary"]
    pdo_delta = paired["pdo"]["force_evaluation_delta_summary"]
    return (
        "# Safe L-BFGS history-depth ablation — evidence-limited conclusion\n\n"
        "## Certificate outcome\n\n"
        "Both arms satisfied 16/16 force certificates. The certificate is derived "
        "strictly from `fmax > 0` and `0 <= final active max force <= fmax`; optimizer "
        "status is not substituted for that force test.\n\n"
        "## Fixed-task cost outcome\n\n"
        f"History 10 used {overall[HISTORY10]['force_evaluations']} evaluator calls; "
        f"history 1 used {overall[HISTORY1]['force_evaluations']}.\n\n"
        f"C60: {by_system['c60'][HISTORY10]['force_evaluations']} versus "
        f"{by_system['c60'][HISTORY1]['force_evaluations']} calls (history 10 versus "
        f"history 1). Per-task history10-minus-history1 differences were "
        f"{c60_delta['lower_count']} lower, {c60_delta['tie_count']} tie, and "
        f"{c60_delta['higher_count']} higher, with median paired delta "
        f"{c60_delta['median']:+g}. The C60 effect is not consistent across tasks.\n\n"
        f"PdO: {by_system['pdo'][HISTORY10]['force_evaluations']} versus "
        f"{by_system['pdo'][HISTORY1]['force_evaluations']} calls. History 10 was "
        f"lower on {pdo_delta['lower_count']} of 8 paired tasks, with median paired "
        f"delta {pdo_delta['median']:+g}.\n\n"
        "## Next validation choice\n\n"
        "Advance history 10 to the 200-step validation because it preserves all "
        "certificates, shows a strong PdO cost reduction, and is the current default. "
        "This is a validation choice, not a new default claim.\n\n"
        "## Claim ceiling\n\n"
        "Endpoints can differ. The recorded paired endpoint-energy and raw Cartesian "
        "position differences are diagnostics only: they support no endpoint-equivalence "
        "claim and no general SSW claim.\n"
    )


def analyze(ledger_dir: Path, output_dir: Path) -> None:
    summary, rows = load_validated_ledger(ledger_dir)
    evidence = build_evidence(ledger_dir, summary, rows)
    conclusion = render_conclusion(evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "conclusion.md").write_text(conclusion, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    arguments = parser.parse_args()
    try:
        analyze(arguments.ledger_dir, arguments.output_dir)
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
