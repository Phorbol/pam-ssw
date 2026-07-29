#!/usr/bin/env python3
"""Independently analyze the locked direction-continuation cohort."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = RUN_ROOT / "run_ablation.py"
MEANINGFUL_ENERGY_DROP_EV = 0.001


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "_direction_continuation_protocol",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row["certificate"]
        and row["is_new_basin"]
        and float(row["landing_delta_eV"])
        <= -MEANINGFUL_ENERGY_DROP_EV
    )


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(statistics.median(values))


def _condition(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["state_id"]), int(row["seed"])


def validate_shared_initial_directions(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    expected = {
        (state_id, seed)
        for state_id in (
            "intermediate_accepted",
            "plateau_accepted",
        )
        for seed in (42, 43, 44)
    }
    observed = {
        (str(record.get("state_id")), int(record.get("seed")))
        for record in records
    }
    if len(records) != 6 or observed != expected:
        raise ValueError(
            "shared initial ledger requires six paired selections"
        )
    by_condition = {
        (str(record["state_id"]), int(record["seed"])): record
        for record in records
    }
    for condition in expected:
        record = by_condition[condition]
        purposes = record.get("purpose_counts")
        hashes = {
            str(row["direction_trace"][0]["selected_direction_sha256"])
            for row in rows
            if _condition(row) == condition
        }
        if (
            record.get("force_evaluations") != 24
            or not isinstance(purposes, Mapping)
            or purposes.get("direction_oracle") != 24
            or purposes.get("unattributed") != 0
            or sum(int(value) for value in purposes.values()) != 24
            or hashes != {str(record["direction_sha256"])}
        ):
            raise ValueError(
                "shared initial direction ledger does not close"
            )
    return {
        "selection_count": 6,
        "force_evaluations": 144,
    }


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    protocol = _load_runner()
    expected = {
        (case["state_id"], case["seed"], case["arm"])
        for case in protocol.case_matrix()
    }
    observed = {
        (row.get("state_id"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if (
        len(rows) != 18
        or len(observed) != 18
        or observed != expected
    ):
        raise ValueError("analysis requires the exact 18-case cohort")

    first_hashes: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        purposes = row.get("purpose_counts")
        trace = row.get("direction_trace")
        if (
            row.get("status") != "completed"
            or row.get("exact_starter_reference") is not True
            or row.get("certificate") is not True
            or row.get("direction_trace_valid") is not True
            or row.get("landing_geometry_valid") is not True
            or not isinstance(purposes, Mapping)
            or not isinstance(trace, list)
            or not trace
            or sum(int(value) for value in purposes.values())
            != row.get("force_evaluations")
            or purposes.get("unattributed") != 0
            or purposes.get("bootstrap_true_quench") != 0
        ):
            raise ValueError("case completion or purpose ledger does not close")
        key = _condition(row)
        first_hashes.setdefault(key, set()).add(
            str(trace[0]["selected_direction_sha256"])
        )
    if any(len(hashes) != 1 for hashes in first_hashes.values()):
        raise ValueError("paired arms do not share the same step-zero direction")

    by_key = {
        (str(row["state_id"]), int(row["seed"]), str(row["arm"])): row
        for row in rows
    }
    arm_results: dict[str, dict[str, Any]] = {}
    meaningful_conditions: dict[str, set[tuple[str, int]]] = {}
    for arm in protocol.ARMS:
        arm_rows = [
            by_key[(state_id, seed, arm)]
            for state_id in protocol.STATE_IDS
            for seed in protocol.SEEDS
        ]
        continuity = [
            float(trace["selected_to_previous_selected_abs_cosine"])
            for row in arm_rows
            for trace in row["direction_trace"][1:]
            if trace.get(
                "selected_to_previous_selected_abs_cosine"
            )
            is not None
        ]
        meaningful = {
            _condition(row) for row in arm_rows if _meaningful(row)
        }
        meaningful_conditions[arm] = meaningful
        arm_results[arm] = {
            "completed_cases": len(arm_rows),
            "certificate_count": sum(
                bool(row["certificate"]) for row in arm_rows
            ),
            "geometry_invalid_count": sum(
                not bool(row["landing_geometry_valid"])
                for row in arm_rows
            ),
            "fragmented_count": sum(
                bool(row["fragmented"]) for row in arm_rows
            ),
            "fallback_count": sum(
                bool(row["fallback_used"]) for row in arm_rows
            ),
            "continuation_projection_degenerate_count": sum(
                int(row["continuation_projection_degenerate"])
                for row in arm_rows
            ),
            "new_basin_count": sum(
                bool(row["is_new_basin"]) for row in arm_rows
            ),
            "meaningful_outcome_count": len(meaningful),
            "meaningful_conditions": [
                {"state_id": state_id, "seed": seed}
                for state_id, seed in sorted(meaningful)
            ],
            "median_abs_consecutive_mode_cosine": _median(continuity),
            "median_landing_delta_eV": float(
                statistics.median(
                    float(row["landing_delta_eV"])
                    for row in arm_rows
                )
            ),
            "direction_force_evaluations": sum(
                int(row["purpose_counts"]["direction_oracle"])
                for row in arm_rows
            ),
            "total_force_evaluations": sum(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "median_force_evaluations": float(
                statistics.median(
                    int(row["force_evaluations"])
                    for row in arm_rows
                )
            ),
            "generation_wall_time_s": sum(
                float(row["generation_wall_time_s"])
                for row in arm_rows
            ),
            "quench_wall_time_s": sum(
                float(row["quench_wall_time_s"])
                for row in arm_rows
            ),
        }

    control = arm_results["fixed_intent_ritz"]
    control_events = meaningful_conditions["fixed_intent_ritz"]
    survivors: list[str] = []
    gates: dict[str, dict[str, bool]] = {}
    for arm in ("transported_direction", "continuation_lanczos"):
        result = arm_results[arm]
        events = meaningful_conditions[arm]
        control_continuity = control[
            "median_abs_consecutive_mode_cosine"
        ]
        arm_continuity = result[
            "median_abs_consecutive_mode_cosine"
        ]
        continuity_improved = bool(
            control_continuity is not None
            and arm_continuity is not None
            and arm_continuity > control_continuity
        )
        no_validity_regression = bool(
            result["certificate_count"] == control["certificate_count"]
            and result["geometry_invalid_count"]
            <= control["geometry_invalid_count"]
            and result["fragmented_count"]
            <= control["fragmented_count"]
            and result["continuation_projection_degenerate_count"] == 0
        )
        extra_event = bool(events - control_events)
        cheaper_reproduction = bool(
            control_events
            and control_events <= events
            and all(
                int(
                    by_key[
                        (state_id, seed, arm)
                    ]["force_evaluations"]
                )
                < int(
                    by_key[
                        (
                            state_id,
                            seed,
                            "fixed_intent_ritz",
                        )
                    ]["force_evaluations"]
                )
                for state_id, seed in control_events
            )
        )
        terminal_gain = extra_event or cheaper_reproduction
        gates[arm] = {
            "continuity_improved": continuity_improved,
            "no_validity_regression": no_validity_regression,
            "extra_meaningful_paired_event": extra_event,
            "cheaper_control_event_reproduction": cheaper_reproduction,
            "terminal_gain": terminal_gain,
        }
        if continuity_improved and no_validity_regression and terminal_gain:
            survivors.append(arm)

    if not survivors:
        decision = "no_survivor"
    elif survivors == ["transported_direction"]:
        decision = "transported_direction_survives"
    elif survivors == ["continuation_lanczos"]:
        decision = "continuation_lanczos_survives"
    else:
        decision = "multiple_survivors"

    purpose_totals = {
        purpose: sum(
            int(row["purpose_counts"][purpose]) for row in rows
        )
        for purpose in rows[0]["purpose_counts"]
    }
    return {
        "schema_version": 1,
        "decision": decision,
        "survivors": survivors,
        "survivor_gates": gates,
        "cohort": {
            "states": list(protocol.STATE_IDS),
            "seeds": list(protocol.SEEDS),
            "arms": list(protocol.ARMS),
            "completed_cases": len(rows),
        },
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "bootstrap_force_evaluations": 0,
        "certificate_count": sum(
            bool(row["certificate"]) for row in rows
        ),
        "arm_results": arm_results,
        "totals": {
            "force_evaluations": sum(
                int(row["force_evaluations"]) for row in rows
            ),
            "purpose_counts": purpose_totals,
            "unattributed_force_evaluations": purpose_totals[
                "unattributed"
            ],
            "generation_wall_time_s": sum(
                float(row["generation_wall_time_s"])
                for row in rows
            ),
            "quench_wall_time_s": sum(
                float(row["quench_wall_time_s"])
                for row in rows
            ),
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "paired three-seed C60 direction-mechanism ablation; "
            "no cross-system or long-run superiority claim"
        ),
    }


def _conclusion_markdown(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Direction-mode continuation C60 conclusion",
        "",
        f"Decision: `{evidence['decision']}`.",
        "",
        "| arm | meaningful | median mode cosine | direction FE | total FE |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, result in evidence["arm_results"].items():
        cosine = result["median_abs_consecutive_mode_cosine"]
        cosine_text = "n/a" if cosine is None else f"{cosine:.6f}"
        lines.append(
            f"| {arm} | {result['meaningful_outcome_count']} | "
            f"{cosine_text} | {result['direction_force_evaluations']} | "
            f"{result['total_force_evaluations']} |"
        )
    lines.extend(
        [
            "",
            "The decision applies the preregistered vector gate: mode "
            "continuity, validity, and paired terminal outcome/cost are "
            "reported separately; no weighted score is used.",
            "",
            "If no arm survives, this route stops. The next experiment must "
            "change physical event content rather than add selector or "
            "eigensolver heuristics.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_output(output_dir: Path) -> dict[str, Any]:
    raw_path = output_dir / "raw.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    evidence = analyze_rows(raw["cases"])
    shared_audit = validate_shared_initial_directions(
        raw["shared_initial_directions"],
        raw["cases"],
    )
    evidence["shared_initial_direction"] = shared_audit
    evidence["totals"]["case_force_evaluations"] = evidence[
        "totals"
    ]["force_evaluations"]
    evidence["totals"]["force_evaluations"] += shared_audit[
        "force_evaluations"
    ]
    evidence["totals"]["purpose_counts"]["direction_oracle"] += (
        shared_audit["force_evaluations"]
    )
    evidence["execution_commit"] = raw["execution_commit"]
    evidence["shared_provenance"] = raw["shared_provenance"]
    evidence["state_provenance"] = raw["state_provenance"]
    _write_json(RUN_ROOT / "evidence.json", evidence)
    (RUN_ROOT / "conclusion.md").write_text(
        _conclusion_markdown(evidence),
        encoding="utf-8",
    )
    return evidence


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = analyze_output(args.output)
    print(
        json.dumps(
            {
                "decision": evidence["decision"],
                "survivors": evidence["survivors"],
                "arm_results": evidence["arm_results"],
                "totals": evidence["totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
