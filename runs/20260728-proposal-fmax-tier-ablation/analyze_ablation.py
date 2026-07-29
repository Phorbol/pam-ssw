#!/usr/bin/env python3
"""Validate and analyze the fixed one-bias proposal-fmax ablation."""

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

from pamssw.accounting import EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
RAW_DIR = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
CANDIDATE_SEEDS = tuple(range(42, 58))
TARGET_ELIGIBLE_TASKS = 8
ARMS = {"fmax-0.05": 0.05, "fmax-0.10": 0.10}
STRICT_ARM = "fmax-0.05"
LOOSE_ARM = "fmax-0.10"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(
        coordinates, dtype=np.dtype("<f8"), order="C", copy=True
    )
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _state(payload: Mapping[str, Any]) -> State:
    return State(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=(
            None
            if payload.get("cell") is None
            else np.asarray(payload["cell"], dtype=float)
        ),
        pbc=tuple(bool(value) for value in payload["pbc"]),
        fixed_mask=np.asarray(payload["fixed_mask"], dtype=bool),
    )


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _counts(
    payload: object,
    *,
    label: str,
    allowed: set[str],
    expected_total: int,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} purpose accounting is missing")
    expected_keys = {purpose.value for purpose in EvaluationPurpose}
    if set(payload) != expected_keys:
        raise ValueError(f"{label} purpose accounting has unknown keys")
    values = {
        key: int(value)
        for key, value in payload.items()
    }
    if (
        any(value < 0 for value in values.values())
        or sum(values.values()) != expected_total
        or values[EvaluationPurpose.UNATTRIBUTED.value] != 0
        or any(value for key, value in values.items() if key not in allowed)
    ):
        raise ValueError(f"{label} purpose accounting does not close")
    return values


def _validate_protocol(summary: object) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be an object")
    if (
        summary.get("schema_version") != 1
        or summary.get("task_count") != 16
        or summary.get("row_count") != 32
    ):
        raise ValueError("summary does not describe the fixed matrix")
    protocol = summary.get("protocol")
    if protocol != {
        "systems": ["c60", "pdo"],
        "candidate_seeds": list(CANDIDATE_SEEDS),
        "target_capture_eligible_tasks_per_system": 8,
        "capture_selection_rule": (
            "strict first 8 capture-eligible tasks from the deterministic "
            "candidate-seed prefix, defined before any arm execution"
        ),
        "target_bias_count": 1,
        "softening_enabled": False,
        "proposal_optimizer": "safe-lbfgs-total",
        "safe_history_limit": 10,
        "proposal_arms": [
            {"arm_id": "fmax-0.05", "fmax_eV_per_A": 0.05},
            {"arm_id": "fmax-0.10", "fmax_eV_per_A": 0.10},
        ],
        "proposal_maxiter": "unchanged from current production config",
        "bootstrap_optimizer": "scipy-lbfgsb",
        "bootstrap_fmax_eV_per_A": 0.05,
        "bootstrap_maxiter": 400,
        "bootstrap_shared_by_all_seeds_and_arms": True,
        "landing_optimizer": "scipy-lbfgsb",
        "landing_fmax_eV_per_A": 0.05,
        "landing_maxiter": 400,
        "landing_objective": "true_mace_pes_no_bias_no_softening",
    }:
        raise ValueError("summary protocol violates the fixed ablation")
    systems = summary.get("systems")
    if not isinstance(systems, Mapping) or set(systems) != set(SYSTEMS):
        raise ValueError("summary systems are incomplete")
    return summary


def _validate_task(
    item: object,
    *,
    system: str,
) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError("captured task must be an object")
    seed = item.get("seed")
    if item.get("system") != system or seed not in CANDIDATE_SEEDS:
        raise ValueError("captured task identity is invalid")
    task = item.get("task")
    if not isinstance(task, Mapping):
        raise ValueError("captured task payload is missing")
    if (
        task.get("schema_version") != 1
        or len(task.get("biases", [])) != 1
        or float(task.get("fmax")) != 0.05
    ):
        raise ValueError("captured task is not the fixed one-bias task")
    if item.get("source_task_sha256") != _canonical_sha256(task):
        raise ValueError("captured source task hash does not match payload")
    fixed_payload = dict(task)
    del fixed_payload["fmax"]
    if item.get("fixed_biased_pes_sha256") != _canonical_sha256(
        fixed_payload
    ):
        raise ValueError("fixed biased-PES hash does not match payload")
    capture_counts = item.get("capture_evaluation_counts")
    if not isinstance(capture_counts, Mapping):
        raise ValueError("capture purpose accounting is missing")
    capture_total = sum(int(value) for value in capture_counts.values())
    _counts(
        capture_counts,
        label="capture",
        allowed={
            purpose.value
            for purpose in EvaluationPurpose
            if purpose
            not in {
                EvaluationPurpose.BIASED_PROPOSAL_RELAX,
                EvaluationPurpose.LANDING_TRUE_QUENCH,
                EvaluationPurpose.UNATTRIBUTED,
            }
        },
        expected_total=capture_total,
    )
    return item


def _validate_position(
    payload: Mapping[str, Any],
    *,
    hash_key: str = "positions_sha256",
) -> State:
    state = _state(payload["state"])
    if payload.get(hash_key) != _position_sha256(state.positions):
        raise ValueError("position hash does not match coordinates")
    return state


def _validate_bootstrap(system_data: Mapping[str, Any]) -> None:
    bootstrap = system_data.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("bootstrap evidence is incomplete")
    if (
        bootstrap.get("optimizer") != "scipy-lbfgsb"
        or bootstrap.get("fmax_eV_per_A") != 0.05
        or bootstrap.get("maxiter") != 400
        or bootstrap.get("objective")
        != "true_mace_pes_no_bias_no_softening"
    ):
        raise ValueError("bootstrap protocol is inconsistent")
    state = _validate_position(bootstrap)
    energy = _finite(bootstrap.get("energy_eV"), "bootstrap energy")
    force = _finite(
        bootstrap.get("max_active_force_eV_per_A"), "bootstrap force"
    )
    total = int(bootstrap.get("force_evaluations"))
    _counts(
        bootstrap.get("purpose_counts"),
        label="bootstrap",
        allowed={
            EvaluationPurpose.STARTER_TRUE_QUENCH.value,
            EvaluationPurpose.POST_RELAX_VALIDATION.value,
        },
        expected_total=total,
    )
    certificate = bool(
        math.isfinite(energy)
        and force <= float(bootstrap["fmax_eV_per_A"])
        and np.all(np.isfinite(state.positions))
    )
    if bootstrap.get("certificate_satisfied") is not certificate:
        raise ValueError("bootstrap certificate is inconsistent")


def _validate_capture_attempts(
    system_data: Mapping[str, Any],
    *,
    system: str,
) -> tuple[int, ...]:
    attempts = system_data.get("capture_attempts")
    if (
        not isinstance(attempts, list)
        or not TARGET_ELIGIBLE_TASKS <= len(attempts) <= len(CANDIDATE_SEEDS)
    ):
        raise ValueError("capture attempt prefix is incomplete")
    attempted_seeds = tuple(int(item.get("seed")) for item in attempts)
    if attempted_seeds != CANDIDATE_SEEDS[: len(attempts)]:
        raise ValueError("capture attempt prefix is not deterministic")
    eligible = []
    for item in attempts:
        if not isinstance(item, Mapping):
            raise ValueError("capture attempt must be an object")
        seed = int(item["seed"])
        total = int(item.get("force_evaluations"))
        counts = _counts(
            item.get("purpose_counts"),
            label="capture",
            allowed={
                EvaluationPurpose.DIRECTION_ORACLE.value,
                EvaluationPurpose.ESCAPE_TRUE_PES_CHECK.value,
            },
            expected_total=total,
        )
        status = item.get("status")
        reason = item.get("reason")
        selected = item.get("selected_for_arms")
        direction_kind = item.get("selected_direction_kind")
        if not isinstance(direction_kind, str) or not direction_kind:
            raise ValueError("capture attempt lacks selected direction kind")
        if status == "eligible":
            if reason is not None or selected is not True:
                raise ValueError("eligible capture attempt is inconsistent")
            eligible.append(seed)
        elif status == "ineligible":
            if (
                not isinstance(reason, str)
                or not reason
                or selected is not False
            ):
                raise ValueError("ineligible capture attempt lacks reason")
        else:
            raise ValueError("capture attempt has unknown status")
        expected_trace = f"{system}/seed-{seed}/direction_trace.jsonl"
        if item.get("direction_trace_path") != expected_trace:
            raise ValueError("capture attempt trace path is inconsistent")
        if sum(counts.values()) != total:
            raise ValueError("capture purpose accounting does not close")
    if (
        len(eligible) != TARGET_ELIGIBLE_TASKS
        or attempts[-1].get("status") != "eligible"
    ):
        raise ValueError("capture attempt prefix does not end at eighth eligible")
    expected_file = f"{system}/capture_attempts.json"
    if system_data.get("capture_attempts_file") != expected_file:
        raise ValueError("capture attempt ledger path is inconsistent")
    return tuple(eligible)


def _validate_row(
    row: object,
    *,
    system: str,
    tasks: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("row must be an object")
    seed = row.get("seed")
    arm_id = row.get("arm_id")
    if (
        row.get("system") != system
        or seed not in tasks
        or arm_id not in ARMS
    ):
        raise ValueError("row identity is invalid")
    source = tasks[int(seed)]
    if (
        row.get("source_task_sha256")
        != source["source_task_sha256"]
        or row.get("fixed_biased_pes_sha256")
        != source["fixed_biased_pes_sha256"]
    ):
        raise ValueError("row does not reference its fixed task")

    proposal = row.get("proposal")
    landing = row.get("landing")
    if not isinstance(proposal, Mapping) or not isinstance(landing, Mapping):
        raise ValueError("row stages are incomplete")
    source_task = source["task"]
    if (
        proposal.get("optimizer") != "safe-lbfgs-total"
        or proposal.get("safe_history_limit") != 10
        or proposal.get("fmax_eV_per_A") != ARMS[str(arm_id)]
        or proposal.get("maxiter") != int(source_task["maxiter"])
        or proposal.get("coordinate_trust_radius_A")
        != source_task["coordinate_trust_radius"]
    ):
        raise ValueError("proposal stage violates the fixed protocol")
    initial_hash = _position_sha256(
        source_task["initial_state"]["positions"]
    )
    if proposal.get("initial_positions_sha256") != initial_hash:
        raise ValueError("proposal does not start from the fixed task")
    proposal_final = proposal.get("final")
    if not isinstance(proposal_final, Mapping):
        raise ValueError("proposal final result is missing")
    proposal_state = _validate_position(proposal_final)
    proposal_force = _finite(
        proposal_final.get("max_active_force_eV_per_A"),
        "proposal force",
    )
    _finite(proposal_final.get("biased_energy_eV"), "biased energy")
    proposal_calls = int(proposal.get("force_evaluations"))
    _finite(proposal.get("wall_time_s"), "proposal wall time")
    proposal_counts = _counts(
        proposal.get("purpose_counts"),
        label="proposal",
        allowed={EvaluationPurpose.BIASED_PROPOSAL_RELAX.value},
        expected_total=proposal_calls,
    )
    telemetry = proposal.get("telemetry")
    if (
        not isinstance(telemetry, Mapping)
        or int(telemetry.get("evaluator_calls")) != proposal_calls
        or proposal_counts[
            EvaluationPurpose.BIASED_PROPOSAL_RELAX.value
        ]
        != proposal_calls
    ):
        raise ValueError("proposal purpose accounting does not close")
    calculated_proposal_certificate = bool(
        proposal_force <= ARMS[str(arm_id)]
        and np.all(np.isfinite(proposal_state.positions))
    )
    if proposal.get("certificate_satisfied") is not (
        calculated_proposal_certificate
    ):
        raise ValueError("proposal certificate is inconsistent")

    if (
        landing.get("optimizer") != "scipy-lbfgsb"
        or landing.get("fmax_eV_per_A") != 0.05
        or landing.get("maxiter") != 400
        or landing.get("objective")
        != "true_mace_pes_no_bias_no_softening"
    ):
        raise ValueError("landing stage violates the fixed protocol")
    landing_initial = landing.get("initial")
    landing_final = landing.get("final")
    if not isinstance(landing_initial, Mapping) or not isinstance(
        landing_final, Mapping
    ):
        raise ValueError("landing states are incomplete")
    initial_state = _validate_position(landing_initial)
    if (
        landing_initial["positions_sha256"]
        != proposal_final["positions_sha256"]
        or not np.array_equal(initial_state.numbers, proposal_state.numbers)
    ):
        raise ValueError("landing does not start from proposal endpoint")
    final_state = _validate_position(landing_final)
    final_energy = _finite(
        landing_final.get("energy_eV"), "landing energy"
    )
    final_force = _finite(
        landing_final.get("max_active_force_eV_per_A"),
        "landing force",
    )
    landing_calls = int(landing.get("force_evaluations"))
    _finite(landing.get("wall_time_s"), "landing wall time")
    landing_counts = _counts(
        landing.get("purpose_counts"),
        label="landing",
        allowed={
            EvaluationPurpose.LANDING_TRUE_QUENCH.value,
            EvaluationPurpose.POST_RELAX_VALIDATION.value,
        },
        expected_total=landing_calls,
    )
    landing_telemetry = landing.get("telemetry")
    if (
        not isinstance(landing_telemetry, Mapping)
        or int(landing_telemetry.get("evaluator_calls"))
        != landing_counts[EvaluationPurpose.LANDING_TRUE_QUENCH.value]
    ):
        raise ValueError("landing purpose accounting does not close")
    calculated_landing_certificate = bool(
        final_force <= 0.05
        and math.isfinite(final_energy)
        and np.all(np.isfinite(final_state.positions))
    )
    if landing.get("certificate_satisfied") is not (
        calculated_landing_certificate
    ):
        raise ValueError("landing certificate is inconsistent")
    return row


def _same_basin(
    strict: Mapping[str, Any],
    loose: Mapping[str, Any],
    *,
    energy_tol: float,
    rmsd_tol: float,
) -> tuple[bool, float, float | None, str]:
    strict_final = strict["landing"]["final"]
    loose_final = loose["landing"]["final"]
    energy_delta = abs(
        float(strict_final["energy_eV"])
        - float(loose_final["energy_eV"])
    )
    rmsd = MinimaArchive._rmsd(
        _state(strict_final["state"]),
        _state(loose_final["state"]),
    )
    if not math.isfinite(rmsd):
        return False, energy_delta, None, "nonfinite_rejected"
    return (
        energy_delta <= energy_tol and rmsd <= rmsd_tol,
        energy_delta,
        rmsd,
        "finite",
    )


def analyze(raw_dir: Path) -> dict[str, Any]:
    summary = _validate_protocol(
        _read_json(Path(raw_dir) / "summary.json")
    )
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "source_execution_commit": summary.get("execution_commit"),
        "claim_boundary": {
            "fixed_one_bias_local_ablation": True,
            "softening_disabled": True,
            "shared_bootstrap_true_pes_fmax_eV_per_A": 0.05,
            "capture_eligibility_conditioned": True,
            "capture_eligibility_may_filter_direction_kinds": True,
            "full_ssw_superiority_supported": False,
        },
        "systems": {},
    }
    for system in SYSTEMS:
        system_data = summary["systems"][system]
        _validate_bootstrap(system_data)
        eligible_seeds = _validate_capture_attempts(
            system_data, system=system
        )
        task_items = system_data.get("tasks")
        rows = system_data.get("rows")
        if (
            not isinstance(task_items, list)
            or len(task_items) != TARGET_ELIGIBLE_TASKS
            or not isinstance(rows, list)
            or len(rows) != TARGET_ELIGIBLE_TASKS * len(ARMS)
        ):
            raise ValueError(f"{system} matrix is incomplete")
        tasks: dict[int, Mapping[str, Any]] = {}
        for item in task_items:
            validated = _validate_task(item, system=system)
            seed = int(validated["seed"])
            if seed in tasks:
                raise ValueError("duplicate captured task")
            tasks[seed] = validated
        if tuple(tasks) != eligible_seeds:
            raise ValueError(
                f"{system} tasks are not the first capture-eligible seeds"
            )
        attempts_by_seed = {
            int(item["seed"]): item
            for item in system_data["capture_attempts"]
        }
        for seed, task in tasks.items():
            if (
                task["capture_evaluation_counts"]
                != attempts_by_seed[seed]["purpose_counts"]
            ):
                raise ValueError(
                    "selected task capture counts differ from attempt ledger"
                )

        by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
        for item in rows:
            row = _validate_row(item, system=system, tasks=tasks)
            key = (int(row["seed"]), str(row["arm_id"]))
            if key in by_key:
                raise ValueError("duplicate result row")
            by_key[key] = row
        expected = {
            (seed, arm_id) for seed in eligible_seeds for arm_id in ARMS
        }
        if set(by_key) != expected:
            raise ValueError(f"{system} rows do not cover fixed matrix")

        energy_tol = _finite(
            system_data.get("dedup_energy_tol_eV"), "dedup energy tolerance"
        )
        rmsd_tol = _finite(
            system_data.get("dedup_rmsd_tol_A"), "dedup RMSD tolerance"
        )
        comparisons = []
        for seed in eligible_seeds:
            strict = by_key[(seed, STRICT_ARM)]
            loose = by_key[(seed, LOOSE_ARM)]
            if (
                strict["source_task_sha256"]
                != loose["source_task_sha256"]
                or strict["fixed_biased_pes_sha256"]
                != loose["fixed_biased_pes_sha256"]
            ):
                raise ValueError("paired arms do not share the fixed task")
            same, energy_delta, rmsd, rmsd_status = _same_basin(
                strict,
                loose,
                energy_tol=energy_tol,
                rmsd_tol=rmsd_tol,
            )
            strict_calls = int(strict["proposal"]["force_evaluations"])
            loose_calls = int(loose["proposal"]["force_evaluations"])
            comparisons.append(
                {
                    "seed": seed,
                    "strict_proposal_calls": strict_calls,
                    "loose_proposal_calls": loose_calls,
                    "proposal_call_savings": strict_calls - loose_calls,
                    "strict_proposal_certificate": bool(
                        strict["proposal"]["certificate_satisfied"]
                    ),
                    "loose_proposal_certificate": bool(
                        loose["proposal"]["certificate_satisfied"]
                    ),
                    "strict_landing_certificate": bool(
                        strict["landing"]["certificate_satisfied"]
                    ),
                    "loose_landing_certificate": bool(
                        loose["landing"]["certificate_satisfied"]
                    ),
                    "landing_same_basin": same,
                    "landing_energy_abs_delta_eV": energy_delta,
                    "landing_rmsd_A": rmsd,
                    "landing_rmsd_status": rmsd_status,
                }
            )
        savings = [
            int(item["proposal_call_savings"]) for item in comparisons
        ]
        same_count = sum(
            bool(item["landing_same_basin"]) for item in comparisons
        )
        arm_costs = {}
        for arm_id in (STRICT_ARM, LOOSE_ARM):
            arm_rows = [
                by_key[(seed, arm_id)] for seed in eligible_seeds
            ]
            proposal_calls = sum(
                int(row["proposal"]["force_evaluations"])
                for row in arm_rows
            )
            proposal_wall = sum(
                float(row["proposal"]["wall_time_s"])
                for row in arm_rows
            )
            landing_calls = sum(
                int(row["landing"]["force_evaluations"])
                for row in arm_rows
            )
            landing_wall = sum(
                float(row["landing"]["wall_time_s"])
                for row in arm_rows
            )
            arm_costs[arm_id] = {
                "proposal_force_evaluations": proposal_calls,
                "proposal_wall_time_s": proposal_wall,
                "landing_force_evaluations": landing_calls,
                "landing_wall_time_s": landing_wall,
                "combined_force_evaluations": (
                    proposal_calls + landing_calls
                ),
                "combined_wall_time_s": proposal_wall + landing_wall,
            }
        strict_combined_calls = int(
            arm_costs[STRICT_ARM]["combined_force_evaluations"]
        )
        loose_combined_calls = int(
            arm_costs[LOOSE_ARM]["combined_force_evaluations"]
        )
        combined_savings = strict_combined_calls - loose_combined_calls
        bootstrap_calls = int(
            system_data["bootstrap"]["force_evaluations"]
        )
        capture_calls = sum(
            int(item["force_evaluations"])
            for item in system_data["capture_attempts"]
        )
        evidence["systems"][system] = {
            "shared_pre_arm_costs": {
                "bootstrap_force_evaluations": bootstrap_calls,
                "bootstrap_wall_time_s": float(
                    system_data["bootstrap"]["wall_time_s"]
                ),
                "capture_force_evaluations": capture_calls,
                "combined_force_evaluations": (
                    bootstrap_calls + capture_calls
                ),
            },
            "capture_selection": {
                "candidate_seed_prefix": [
                    int(item["seed"])
                    for item in system_data["capture_attempts"]
                ],
                "selected_seeds": list(eligible_seeds),
                "ineligible_attempts": [
                    {
                        "seed": int(item["seed"]),
                        "reason": item["reason"],
                        "selected_direction_kind": item[
                            "selected_direction_kind"
                        ],
                        "force_evaluations": int(
                            item["force_evaluations"]
                        ),
                        "purpose_counts": dict(item["purpose_counts"]),
                    }
                    for item in system_data["capture_attempts"]
                    if item["status"] == "ineligible"
                ],
                "eligible_direction_kind_counts": dict(
                    sorted(
                        Counter(
                            item["selected_direction_kind"]
                            for item in system_data["capture_attempts"]
                            if item["status"] == "eligible"
                        ).items()
                    )
                ),
                "ineligible_direction_kind_counts": dict(
                    sorted(
                        Counter(
                            item["selected_direction_kind"]
                            for item in system_data["capture_attempts"]
                            if item["status"] == "ineligible"
                        ).items()
                    )
                ),
            },
            "archive_matcher": {
                "definition": "current MinimaArchive energy-plus-RMSD rule",
                "dedup_energy_tol_eV": energy_tol,
                "dedup_rmsd_tol_A": rmsd_tol,
            },
            "comparisons": comparisons,
            "paired": {
                "task_count": len(comparisons),
                "loose_lower_proposal_calls_count": sum(
                    value > 0 for value in savings
                ),
                "loose_equal_proposal_calls_count": sum(
                    value == 0 for value in savings
                ),
                "loose_higher_proposal_calls_count": sum(
                    value < 0 for value in savings
                ),
                "proposal_call_savings_total": sum(savings),
                "proposal_call_savings_median": float(median(savings)),
                "both_proposal_certified_count": sum(
                    bool(item["strict_proposal_certificate"])
                    and bool(item["loose_proposal_certificate"])
                    for item in comparisons
                ),
                "both_landing_certified_count": sum(
                    bool(item["strict_landing_certificate"])
                    and bool(item["loose_landing_certificate"])
                    for item in comparisons
                ),
                "landing_same_basin_count": same_count,
                "landing_equivalence_rate": same_count / len(comparisons),
                "arm_costs": arm_costs,
                "combined_force_evaluation_savings": combined_savings,
                "combined_force_evaluation_savings_fraction": (
                    combined_savings / strict_combined_calls
                ),
            },
        }
    return evidence


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "evidence.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = analyze(args.raw_dir)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
