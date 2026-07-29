"""Execute the bounded analytic U0/U1 mechanism smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.potentials import DoubleWell2D
from pamssw.proposal_replay import capture_proposal_task
from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_ROOT))

from analyze_results import analyze_rows  # noqa: E402
from run_ablation import (  # noqa: E402
    base_sigma_from_task,
    calibrate_fixed_parameters,
    measure_inner_curvature,
    run_task_arms,
    task_sha256,
)


def _calculator() -> AnalyticCalculator:
    return AnalyticCalculator(DoubleWell2D())


def _state() -> State:
    return State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )


def _config(seed: int) -> SSWConfig:
    return SSWConfig(
        max_trials=1,
        max_steps_per_walk=5,
        oracle_candidates=1,
        proposal_relax_steps=20,
        proposal_fmax=0.05,
        rng_seed=seed,
    )


def main() -> None:
    calibration_rows = []
    for seed in (101, 102):
        task = capture_proposal_task(
            _state(),
            _calculator(),
            _config(seed),
            target_bias_count=1,
        ).task
        calibration_rows.append(
            {
                "system": "double_well_2d",
                "task_id": f"cal-{seed}",
                "sigma": task.biases[-1].sigma,
                "weight": task.biases[-1].weight,
                "source_task_sha256": task_sha256(task),
            }
        )
    calibration = calibrate_fixed_parameters(calibration_rows)
    fixed = calibration["double_well_2d"]

    rows = []
    characterizations = []
    for seed in (0, 1):
        for bias_count in (1, 3, 5):
            task_id = f"eval-seed{seed}-bias{bias_count}"
            task = capture_proposal_task(
                _state(),
                _calculator(),
                _config(seed),
                target_bias_count=bias_count,
            ).task
            base_sigma = base_sigma_from_task(
                task,
                target_step_rms=0.15,
                max_step_rms=0.35,
                step_rms_scope="all_atoms",
                active_threshold=1.0e-4,
            )
            inner_curvature, counts = measure_inner_curvature(
                task,
                _calculator(),
                hvp_epsilon=1.0e-3,
            )
            characterizations.append(
                {
                    "task_id": task_id,
                    "source_task_sha256": task_sha256(task),
                    "base_sigma": base_sigma,
                    "inner_curvature": inner_curvature,
                    "evaluation_counts": counts.as_dict(),
                }
            )
            rows.extend(
                run_task_arms(
                    task,
                    system="double_well_2d",
                    task_id=task_id,
                    calculator_factory=_calculator,
                    optimizer="ase-fire",
                    base_sigma=base_sigma,
                    inner_curvature=inner_curvature,
                    target_negative_curvature=0.05,
                    fixed_sigma=float(fixed["sigma"]),
                    fixed_weight=float(fixed["weight"]),
                )
            )

    summary = analyze_rows(rows)
    payload = {
        "schema_version": 1,
        "scope": "conditional fixed-prefix analytic mechanism smoke",
        "calibration": calibration,
        "characterization": characterizations,
        "rows": rows,
        "summary": summary,
        "validation": {
            "expected_row_count": 18,
            "matrix_closed": summary["row_count"] == 18,
            "observer_only_force_evaluations": sum(
                int(row["observer_only_force_evaluations"])
                for row in rows
            ),
        },
    }
    output = RUN_ROOT / "analytic_smoke.json"
    output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()

