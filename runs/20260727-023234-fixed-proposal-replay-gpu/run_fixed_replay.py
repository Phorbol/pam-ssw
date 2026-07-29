from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import runpy
import time
from typing import Any

import numpy as np
import torch

from pamssw.exploration.runner import _bootstrap_minimum
from pamssw.pbc import mic_displacement
from pamssw.proposal_replay import (
    capture_proposal_task,
    proposal_task_to_payload,
    replay_proposal_task,
)


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
SOURCE_DRIVER = (
    RUN_ROOT.parent
    / "20260727-014407-bias-relaxation-gpu-g1"
    / "run_g1.py"
)
FROZEN_COMMIT = "a64816b25c91f93b7d5a3e9c882fc036853c03d9"
SYSTEMS = ("c60", "pdo")
BACKENDS = ("ase-fire", "ase-fire2", "safe-lbfgs-total")
SEEDS = tuple(range(42, 50))


def _source() -> dict[str, Any]:
    return runpy.run_path(str(SOURCE_DRIVER))


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


def _result_payload(replay) -> dict[str, Any]:
    result = replay.result
    telemetry = result.telemetry
    return {
        "certificate_satisfied": replay.certificate_satisfied,
        "force_evaluations": replay.evaluation_counts.total,
        "purpose_counts": replay.evaluation_counts.as_dict(),
        "wall_time_s": replay.wall_time_s,
        "final_biased_energy_eV": float(result.energy),
        "final_max_active_atom_force_eV_per_A": float(result.gradient_norm),
        "iterations": int(result.n_iter),
        "termination_reason": telemetry.termination_reason,
        "backend_evaluations": telemetry.backend_evaluations,
        "evaluator_calls": telemetry.evaluator_calls,
        "reporting_evaluator_calls": telemetry.reporting_evaluator_calls,
        "accepted_steps": telemetry.accepted_steps,
        "rejected_steps": telemetry.rejected_steps,
        "accepted_secants": telemetry.accepted_secants,
        "rejected_secants": telemetry.rejected_secants,
        "line_search_evaluations": telemetry.line_search_evaluations,
        "final_positions": result.state.positions.tolist(),
    }


def _endpoint_comparison(row: dict[str, Any], fire: dict[str, Any], task) -> dict[str, Any]:
    positions = np.asarray(row["final_positions"], dtype=float)
    fire_positions = np.asarray(fire["final_positions"], dtype=float)
    displacement = mic_displacement(
        positions,
        fire_positions,
        task.initial_state.cell,
        task.initial_state.pbc,
    )
    norms = np.linalg.norm(displacement, axis=1)
    movable = task.initial_state.movable_mask
    active_norms = norms[movable]
    energy_delta = (
        float(row["final_biased_energy_eV"])
        - float(fire["final_biased_energy_eV"])
    )
    return {
        "relative_to_fire_energy_eV": energy_delta,
        "relative_to_fire_active_rms_A": float(
            np.sqrt(np.mean(active_norms**2))
        ),
        "relative_to_fire_active_max_A": float(np.max(active_norms)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, Any] = {}
    for backend in BACKENDS:
        selected = [row for row in rows if row["backend"] == backend]
        calls = np.asarray([row["force_evaluations"] for row in selected], dtype=float)
        by_backend[backend] = {
            "tasks": len(selected),
            "certificate_count": sum(
                bool(row["certificate_satisfied"]) for row in selected
            ),
            "force_evaluations_total": int(np.sum(calls)),
            "force_evaluations_median": float(np.median(calls)),
            "force_evaluations_min": int(np.min(calls)),
            "force_evaluations_max": int(np.max(calls)),
            "wall_time_total_s": float(
                sum(float(row["wall_time_s"]) for row in selected)
            ),
        }
    for backend in BACKENDS[1:]:
        compared = [row for row in rows if row["backend"] == backend]
        by_backend[backend]["paired_vs_fire"] = {
            "lower_calls": sum(
                row["force_evaluations"] < row["fire_force_evaluations"]
                for row in compared
            ),
            "equal_calls": sum(
                row["force_evaluations"] == row["fire_force_evaluations"]
                for row in compared
            ),
            "higher_calls": sum(
                row["force_evaluations"] > row["fire_force_evaluations"]
                for row in compared
            ),
            "median_call_ratio": float(
                np.median(
                    [
                        row["force_evaluations"] / row["fire_force_evaluations"]
                        for row in compared
                    ]
                )
            ),
        }
    return by_backend


def run_system(system: str, source: dict[str, Any]) -> dict[str, Any]:
    state = source["_state"](system)
    base_config = source["_ssw_config"](system, "ase-fire")
    bootstrap_started = time.perf_counter()
    bootstrap_state, bootstrap_energy, bootstrap_counts = _bootstrap_minimum(
        state,
        source["_calculator"],
        base_config,
        total_force_budget=1000,
    )
    bootstrap_wall = time.perf_counter() - bootstrap_started

    capture_calculator = source["_calculator"]()
    capture_calculator.evaluate(bootstrap_state)
    tasks = []
    task_records = []
    for seed in SEEDS:
        config = replace(
            base_config,
            rng_seed=seed,
            max_steps_per_walk=1,
            proposal_optimizer="ase-fire",
        )
        captured = capture_proposal_task(
            bootstrap_state,
            capture_calculator,
            config,
            target_bias_count=1,
        )
        tasks.append(captured.task)
        task_records.append(
            {
                "task_id": f"{system}-seed-{seed}-bias-1",
                "seed": seed,
                "generation_evaluation_counts": captured.evaluation_counts.as_dict(),
                "task": proposal_task_to_payload(captured.task),
            }
        )

    calculators = {backend: source["_calculator"]() for backend in BACKENDS}
    for calculator in calculators.values():
        calculator.evaluate(bootstrap_state)

    rows = []
    for task_index, (seed, task) in enumerate(zip(SEEDS, tasks)):
        task_rows: dict[str, dict[str, Any]] = {}
        rotated = BACKENDS[task_index % len(BACKENDS) :] + BACKENDS[: task_index % len(BACKENDS)]
        for backend in rotated:
            replay = replay_proposal_task(
                task,
                calculators[backend],
                optimizer=backend,
            )
            row = {
                "system": system,
                "task_id": f"{system}-seed-{seed}-bias-1",
                "seed": seed,
                "bias_count": len(task.biases),
                "backend": backend,
                **_result_payload(replay),
            }
            task_rows[backend] = row
        fire = task_rows["ase-fire"]
        for backend in BACKENDS:
            row = task_rows[backend]
            row.update(_endpoint_comparison(row, fire, task))
            row["fire_force_evaluations"] = fire["force_evaluations"]
            rows.append(row)

    return {
        "system": system,
        "bootstrap": {
            "energy_eV": bootstrap_energy,
            "force_evaluations": bootstrap_counts.total,
            "purpose_counts": bootstrap_counts.as_dict(),
            "wall_time_s": bootstrap_wall,
            "positions": bootstrap_state.positions.tolist(),
        },
        "tasks": task_records,
        "rows": rows,
        "aggregate": _aggregate(rows),
    }


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    source = _source()
    model = source["MODEL"]
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    if _sha256(model) != source["MODEL_SHA256"]:
        raise ValueError("model hash mismatch")
    for spec in source["SYSTEMS"].values():
        if _sha256(spec["input"]) != spec["sha256"]:
            raise ValueError(f"input hash mismatch: {spec['input']}")

    started = time.perf_counter()
    systems = []
    for system in SYSTEMS:
        result = run_system(system, source)
        systems.append(result)
        _write_json(OUTPUT_ROOT / f"{system}.json", result)
    payload = {
        "schema_version": 1,
        "frozen_commit": FROZEN_COMMIT,
        "device": "cuda",
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "model": str(model),
        "model_sha256": source["MODEL_SHA256"],
        "systems": systems,
        "wall_time_total_s": time.perf_counter() - started,
    }
    _write_json(OUTPUT_ROOT / "summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
