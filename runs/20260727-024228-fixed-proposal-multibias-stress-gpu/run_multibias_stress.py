from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
import time
from typing import Any

import torch

from pamssw.proposal_replay import capture_proposal_task, replay_proposal_task


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
BASE_RUN_ROOT = RUN_ROOT.parent / "20260727-023234-fixed-proposal-replay-gpu"
BASE_SUMMARY = BASE_RUN_ROOT / "output" / "summary.json"
BASE_DRIVER = BASE_RUN_ROOT / "run_fixed_replay.py"
FROZEN_CODE_COMMIT = "a64816b25c91f93b7d5a3e9c882fc036853c03d9"
BACKENDS = ("ase-fire", "ase-fire2", "safe-lbfgs-total")
SEEDS = (42, 43, 44, 45)
TARGET_BIAS_COUNTS = (2, 4)
OBSERVATION_MAXITER = 400


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")

    base = json.loads(BASE_SUMMARY.read_text(encoding="utf-8"))
    helpers = runpy.run_path(str(BASE_DRIVER))
    source = helpers["_source"]()
    started = time.perf_counter()
    systems = []
    for base_system in base["systems"]:
        system = base_system["system"]
        raw_state = source["_state"](system)
        bootstrap_state = raw_state.with_flat_positions(
            base_system["bootstrap"]["positions"]
        )
        base_config = source["_ssw_config"](system, "ase-fire")
        capture_calculator = source["_calculator"]()
        capture_calculator.evaluate(bootstrap_state)
        captured_items = []
        capture_failures = []
        for seed in SEEDS:
            for target_bias_count in TARGET_BIAS_COUNTS:
                config = replace(
                    base_config,
                    rng_seed=seed,
                    max_steps_per_walk=target_bias_count,
                    proposal_optimizer="ase-fire",
                )
                try:
                    captured = capture_proposal_task(
                        bootstrap_state,
                        capture_calculator,
                        config,
                        target_bias_count=target_bias_count,
                    )
                except RuntimeError as error:
                    capture_failures.append(
                        {
                            "seed": seed,
                            "target_bias_count": target_bias_count,
                            "error": str(error),
                        }
                    )
                    continue
                captured_items.append(
                    {
                        "task_id": (
                            f"{system}-seed-{seed}-bias-{target_bias_count}"
                        ),
                        "seed": seed,
                        "target_bias_count": target_bias_count,
                        "task": replace(
                            captured.task,
                            maxiter=OBSERVATION_MAXITER,
                        ),
                        "generation_evaluation_counts": (
                            captured.evaluation_counts.as_dict()
                        ),
                    }
                )

        calculators = {backend: source["_calculator"]() for backend in BACKENDS}
        for calculator in calculators.values():
            calculator.evaluate(bootstrap_state)
        rows = []
        for task_index, item in enumerate(captured_items):
            task = item["task"]
            task_rows: dict[str, dict[str, Any]] = {}
            rotated = (
                BACKENDS[task_index % len(BACKENDS) :]
                + BACKENDS[: task_index % len(BACKENDS)]
            )
            for backend in rotated:
                replay = replay_proposal_task(
                    task,
                    calculators[backend],
                    optimizer=backend,
                )
                row = {
                    "system": system,
                    "task_id": item["task_id"],
                    "seed": item["seed"],
                    "bias_count": len(task.biases),
                    "reference_path_backend": "ase-fire",
                    "observation_maxiter": OBSERVATION_MAXITER,
                    "backend": backend,
                    **helpers["_result_payload"](replay),
                }
                task_rows[backend] = row
            fire = task_rows["ase-fire"]
            for backend in BACKENDS:
                row = task_rows[backend]
                row.update(helpers["_endpoint_comparison"](row, fire, task))
                row["fire_force_evaluations"] = fire["force_evaluations"]
                rows.append(row)

        result = {
            "system": system,
            "reference_path_backend": "ase-fire",
            "captured_tasks": [
                {
                    "task_id": item["task_id"],
                    "seed": item["seed"],
                    "bias_count": item["target_bias_count"],
                    "generation_evaluation_counts": (
                        item["generation_evaluation_counts"]
                    ),
                }
                for item in captured_items
            ],
            "capture_failures": capture_failures,
            "rows": rows,
            "aggregate": helpers["_aggregate"](rows),
        }
        systems.append(result)
        _write_json(OUTPUT_ROOT / f"{system}.json", result)

    payload = {
        "schema_version": 1,
        "frozen_code_commit": FROZEN_CODE_COMMIT,
        "base_summary": str(BASE_SUMMARY),
        "base_summary_sha256": _sha256(BASE_SUMMARY),
        "reference_path_backend": "ase-fire",
        "seeds": list(SEEDS),
        "target_bias_counts": list(TARGET_BIAS_COUNTS),
        "observation_maxiter": OBSERVATION_MAXITER,
        "device": "cuda",
        "cuda_device": torch.cuda.get_device_name(0),
        "systems": systems,
        "wall_time_total_s": time.perf_counter() - started,
    }
    _write_json(OUTPUT_ROOT / "summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
