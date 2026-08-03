#!/usr/bin/env python3
"""Run the bounded C60/PdO U3 bias-shape GPU screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

from pamssw.calculators import ASECalculator
from pamssw.proposal_replay import (
    ProposalTaskNotCaptured,
    capture_proposal_task,
)


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
U01_ROOT = (
    REPO_ROOT / "runs" / "20260730-uphill-propagation-u0-u1"
)
SYSTEM_SPECS = {
    "c60": (
        (2002, 3),
        (2003, 5),
        (2004, 8),
        (2006, 3),
        (2007, 5),
        (2008, 8),
    ),
    "pdo": (
        (2001, 1),
        (2005, 1),
    ),
}

sys.path.insert(0, str(U01_ROOT))
sys.path.insert(0, str(RUN_ROOT))

from analyze_shape_ablation import analyze_rows  # noqa: E402
from run_shape_ablation import ARM_IDS, run_shape_arms  # noqa: E402
from run_gpu_screen import (  # noqa: E402
    _base_runner,
    _bootstrap,
    _capture_record,
    _config,
    _config_projection,
    _runtime_provenance,
    _sha256,
)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sum_force_evaluations(payload: dict[str, Any]) -> int:
    total = sum(
        int(row["force_evaluations"]) for row in payload["rows"]
    )
    for system in payload["systems"].values():
        total += int(system["bootstrap"]["force_evaluations"])
        total += sum(
            int(record["force_evaluations"])
            for record in system["evaluation_tasks"]
        )
        total += sum(
            int(record["force_evaluations"])
            for record in system["capture_failures"]
        )
    return total


def _sum_unattributed(payload: dict[str, Any]) -> int:
    records = list(payload["rows"])
    for system in payload["systems"].values():
        records.append(system["bootstrap"])
        records.extend(system["evaluation_tasks"])
        records.extend(system["capture_failures"])
    return sum(
        int(record["purpose_counts"]["unattributed"])
        for record in records
    )


def run(
    *,
    systems: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    invalid = set(systems) - set(SYSTEM_SPECS)
    if invalid:
        raise ValueError(f"unknown systems: {sorted(invalid)}")
    started = perf_counter()
    base = _base_runner()
    provenance = _runtime_provenance(base)
    provenance["u01_gpu_runner_sha256"] = provenance[
        "script_sha256"
    ]
    provenance["script_sha256"] = _sha256(Path(__file__))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": (
            "conditional frozen-prefix test of Gaussian finite-tail "
            "shape against a center-curvature-matched quadratic"
        ),
        "protocol": {
            "systems": list(systems),
            "system_specs": {
                system: [
                    {
                        "seed": seed,
                        "target_bias_count": bias_count,
                    }
                    for seed, bias_count in SYSTEM_SPECS[system]
                ]
                for system in systems
            },
            "arms": list(ARM_IDS),
            "newest_bias_only": True,
            "right_censor_rule": (
                "terminated prefixes retain exact costs and are not "
                "replaced"
            ),
            "parameter_sweep": False,
        },
        "provenance": provenance,
        "systems": {},
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    for system in systems:
        calculator = ASECalculator(base._calculator())
        bootstrap_config = _config(system, 0, base)
        print(f"[{system}] bootstrap", flush=True)
        seed_state, bootstrap = _bootstrap(
            system,
            calculator,
            bootstrap_config,
            base,
        )
        system_payload = {
            "config": _config_projection(bootstrap_config),
            "bootstrap": bootstrap,
            "evaluation_tasks": [],
            "capture_failures": [],
        }
        payload["systems"][system] = system_payload
        _write(output, payload)

        for seed, bias_count in SYSTEM_SPECS[system]:
            task_id = f"{system}-u3-{seed}-bias{bias_count}"
            config = _config(system, seed, base)
            print(f"[{system}] capture {task_id}", flush=True)
            capture_started = perf_counter()
            try:
                captured = capture_proposal_task(
                    seed_state,
                    calculator,
                    config,
                    target_bias_count=bias_count,
                )
            except ProposalTaskNotCaptured as error:
                failure = {
                    "task_id": task_id,
                    "seed": seed,
                    "target_bias_count": bias_count,
                    "error": str(error),
                    "force_evaluations": (
                        error.evaluation_counts.total
                    ),
                    "purpose_counts": (
                        error.evaluation_counts.as_dict()
                    ),
                    "wall_time_s": (
                        perf_counter() - capture_started
                    ),
                }
                system_payload["capture_failures"].append(failure)
                print(
                    f"[{system}] right-censored {task_id}",
                    flush=True,
                )
                _write(output, payload)
                continue

            capture = _capture_record(
                system=system,
                task_id=task_id,
                seed=seed,
                bias_count=bias_count,
                captured=captured,
                wall_time_s=perf_counter() - capture_started,
            )
            rows = run_shape_arms(
                captured.task,
                system=system,
                task_id=task_id,
                calculator_factory=lambda: calculator,
                optimizer=config.proposal_optimizer,
            )
            system_payload["evaluation_tasks"].append(capture)
            payload["rows"].extend(rows)
            print(
                f"[{system}] replayed {task_id}: "
                + ", ".join(
                    f"{row['arm_id']}={row['force_evaluations']}"
                    for row in rows
                ),
                flush=True,
            )
            _write(output, payload)

    payload["summary"] = analyze_rows(payload["rows"])
    payload["validation"] = {
        "attempted_task_count": sum(
            len(SYSTEM_SPECS[system]) for system in systems
        ),
        "completed_task_count": (
            len(payload["rows"]) // len(ARM_IDS)
        ),
        "row_count": len(payload["rows"]),
        "observer_only_force_evaluations": sum(
            int(row["observer_only_force_evaluations"])
            for row in payload["rows"]
        ),
        "unattributed_force_evaluations": _sum_unattributed(
            payload
        ),
        "accounted_force_evaluations": _sum_force_evaluations(
            payload
        ),
    }
    payload["total_wall_time_s"] = perf_counter() - started
    _write(output, payload)
    return payload


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(SYSTEM_SPECS),
        default=list(SYSTEM_SPECS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "gpu_shape.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run(systems=args.systems, output=args.output)
    print(
        json.dumps(
            result["validation"],
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
