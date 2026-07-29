#!/usr/bin/env python3
"""Run the pre-registered C60 late-prefix U2 feedback factorial."""

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
SYSTEM = "c60"
EVALUATION_SPECS = (
    (2002, 3),
    (2003, 5),
    (2004, 8),
    (2006, 3),
    (2007, 5),
    (2008, 8),
)

sys.path.insert(0, str(U01_ROOT))
sys.path.insert(0, str(RUN_ROOT))

from analyze_factorial import analyze_rows  # noqa: E402
from run_factorial import ARM_IDS, run_factorial_arms  # noqa: E402
from run_ablation import (  # noqa: E402
    base_sigma_from_task,
    measure_inner_curvature,
    task_sha256,
)
from run_gpu_screen import (  # noqa: E402
    _accounted_force_evaluations,
    _accounted_unattributed_force_evaluations,
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
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(*, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    base = _base_runner()
    provenance = _runtime_provenance(base)
    provenance["u01_gpu_runner_sha256"] = provenance["script_sha256"]
    provenance["script_sha256"] = _sha256(Path(__file__))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": (
            "C60 conditional late-prefix 2x2 decomposition of existing "
            "sigma and effective-bias-curvature feedback"
        ),
        "protocol": {
            "system": SYSTEM,
            "evaluation_specs": [
                {"seed": seed, "target_bias_count": count}
                for seed, count in EVALUATION_SPECS
            ],
            "arms": list(ARM_IDS),
            "right_censor_rule": (
                "terminated prefixes are recorded with exact costs and "
                "are not replaced"
            ),
            "production_bias_weight_bounds": True,
        },
        "provenance": provenance,
        "systems": {},
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    calculator = ASECalculator(base._calculator())
    config = _config(SYSTEM, 0, base)
    print("[c60] bootstrap", flush=True)
    seed_state, bootstrap = _bootstrap(
        SYSTEM,
        calculator,
        config,
        base,
    )
    system_payload: dict[str, Any] = {
        "config": _config_projection(config),
        "bootstrap": bootstrap,
        "calibration_tasks": [],
        "calibration_failures": [],
        "evaluation_tasks": [],
        "capture_failures": [],
    }
    payload["systems"][SYSTEM] = system_payload
    _write(output, payload)

    for seed, bias_count in EVALUATION_SPECS:
        task_id = f"c60-u2-{seed}-bias{bias_count}"
        task_config = _config(SYSTEM, seed, base)
        print(f"[c60] capture {task_id}", flush=True)
        capture_started = perf_counter()
        try:
            captured = capture_proposal_task(
                seed_state,
                calculator,
                task_config,
                target_bias_count=bias_count,
            )
        except ProposalTaskNotCaptured as error:
            failure = {
                "task_id": task_id,
                "seed": seed,
                "target_bias_count": bias_count,
                "error": str(error),
                "force_evaluations": error.evaluation_counts.total,
                "purpose_counts": error.evaluation_counts.as_dict(),
                "wall_time_s": perf_counter() - capture_started,
            }
            system_payload["capture_failures"].append(failure)
            print(f"[c60] right-censored {task_id}", flush=True)
            _write(output, payload)
            continue
        capture_wall_time_s = perf_counter() - capture_started

        task = captured.task
        capture = _capture_record(
            system=SYSTEM,
            task_id=task_id,
            seed=seed,
            bias_count=bias_count,
            captured=captured,
            wall_time_s=capture_wall_time_s,
        )
        base_sigma = base_sigma_from_task(
            task,
            target_step_rms=task_config.target_step_rms,
            max_step_rms=task_config.max_step_rms,
            step_rms_scope=task_config.step_rms_scope,
            active_threshold=task_config.step_active_threshold,
        )
        inner_curvature, curvature_counts = measure_inner_curvature(
            task,
            calculator,
            hvp_epsilon=task_config.hvp_epsilon,
        )
        characterization = {
            "source_task_sha256": task_sha256(task),
            "base_sigma": base_sigma,
            "inner_curvature": inner_curvature,
            "target_negative_curvature": (
                task_config.target_negative_curvature
            ),
            "force_evaluations": curvature_counts.total,
            "purpose_counts": curvature_counts.as_dict(),
        }
        rows = run_factorial_arms(
            task,
            system=SYSTEM,
            task_id=task_id,
            calculator_factory=lambda: calculator,
            optimizer=task_config.proposal_optimizer,
            base_sigma=base_sigma,
            inner_curvature=inner_curvature,
            target_negative_curvature=(
                task_config.target_negative_curvature
            ),
            bias_weight_min=task_config.bias_weight_min,
            bias_weight_max=task_config.bias_weight_max,
        )
        system_payload["evaluation_tasks"].append(
            {**capture, "characterization": characterization}
        )
        payload["rows"].extend(rows)
        print(
            "[c60] replayed "
            + task_id
            + ": "
            + ", ".join(
                f"{row['arm_id']}={row['force_evaluations']}"
                for row in rows
            ),
            flush=True,
        )
        _write(output, payload)

    payload["summary"] = analyze_rows(payload["rows"])
    payload["validation"] = {
        "attempted_task_count": len(EVALUATION_SPECS),
        "completed_task_count": len(payload["rows"]) // len(ARM_IDS),
        "row_count": len(payload["rows"]),
        "accounted_force_evaluations": (
            _accounted_force_evaluations(payload)
        ),
        "unattributed_force_evaluations": (
            _accounted_unattributed_force_evaluations(payload)
        ),
        "observer_only_force_evaluations": sum(
            int(row["observer_only_force_evaluations"])
            for row in payload["rows"]
        ),
    }
    payload["total_wall_time_s"] = perf_counter() - started
    _write(output, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "gpu_factorial.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run(output=args.output)
    print(
        json.dumps(result["validation"], indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
