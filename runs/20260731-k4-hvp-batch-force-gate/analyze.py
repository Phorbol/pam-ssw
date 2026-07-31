#!/usr/bin/env python3
"""Analyze equivalence and speed for the K4 HVP GPU batch micro-gate."""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
OUTPUT_DIR = RUN_ROOT / "output"
THRESHOLDS = {
    "max_abs_energy_error_eV": 0.005,
    "max_abs_force_error_eV_per_A": 0.0005,
    "relative_hvp_norm_error": 0.005,
    "max_abs_curvature_error_eV_per_A2": 0.1,
}
MINIMUM_SPEEDUP = 1.3


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    RUN_ROOT / "protocol.py",
    "_k4_hvp_batch_force_protocol_analysis",
)


def _arrays(result: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(result[key], dtype=float)
        for key in ("energies", "forces", "hvps", "curvatures")
    }


def analyze() -> dict[str, Any]:
    summary_path = OUTPUT_DIR / "summary.json"
    rows_path = OUTPUT_DIR / "rows.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if (
        summary["evaluation_counts"]["direction_oracle"] != 384
        or summary["evaluation_counts"]["unattributed"] != 0
        or len(rows) != 40
        or sum(int(row["force_evaluations"]) for row in rows) != 320
    ):
        raise RuntimeError("batch micro-gate FE ledger does not close")

    indexed = {
        (row["system"], int(row["repetition"]), row["mode"]): row
        for row in rows
    }
    mode_rows = []
    for system in ("c60", "pdo"):
        serial_times = {
            repetition: float(
                indexed[(system, repetition, "serial")]["result"][
                    "wall_time_s"
                ]
            )
            for repetition in range(5)
        }
        for mode, batch_size in (("batch2", 2), ("batch4", 4), ("batch8", 8)):
            metrics = []
            speedups = []
            times = []
            for repetition in range(5):
                reference = indexed[(system, repetition, "serial")]["result"]
                candidate = indexed[(system, repetition, mode)]["result"]
                metrics.append(
                    protocol.equivalence_metrics(
                        _arrays(reference),
                        _arrays(candidate),
                    )
                )
                candidate_time = float(candidate["wall_time_s"])
                times.append(candidate_time)
                speedups.append(
                    serial_times[repetition] / candidate_time
                )
            worst = {
                key: max(float(metric[key]) for metric in metrics)
                for key in THRESHOLDS
            }
            equivalent = all(
                worst[key] <= threshold
                for key, threshold in THRESHOLDS.items()
            )
            mode_rows.append(
                {
                    "system": system,
                    "mode": mode,
                    "batch_size": batch_size,
                    "equivalence": worst,
                    "thresholds": THRESHOLDS,
                    "equivalent": equivalent,
                    "median_serial_wall_time_s": median(
                        serial_times.values()
                    ),
                    "median_batch_wall_time_s": median(times),
                    "median_speedup": median(speedups),
                    "speedups": speedups,
                }
            )
    gate = protocol.evaluate_batch_gate(
        mode_rows,
        minimum_speedup=MINIMUM_SPEEDUP,
    )
    evidence = {
        "schema_version": 1,
        "execution_commit": summary["execution_commit"],
        "model_sha256": summary["model_sha256"],
        "cuda_device": summary["cuda_device"],
        "raw_summary_path": str(summary_path.relative_to(REPO_ROOT)),
        "raw_summary_sha256": sha256(summary_path.read_bytes()).hexdigest(),
        "raw_rows_path": str(rows_path.relative_to(REPO_ROOT)),
        "raw_rows_sha256": sha256(rows_path.read_bytes()).hexdigest(),
        "new_force_evaluations": 384,
        "evaluation_counts": summary["evaluation_counts"],
        "mode_rows": mode_rows,
        "gate": gate,
    }
    (RUN_ROOT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# K4 central-HVP GPU batch result",
        "",
        (
            "- ForceService gate allowed: "
            f"**{gate['force_service_gate_allowed']}**."
        ),
        (
            "- Surviving batch sizes: "
            f"**{gate['surviving_batch_sizes']}**."
        ),
        "- Force evaluations: **384** (`unattributed=0`).",
        "- Production change allowed: **False**.",
        "",
        "| System | Batch | Equivalent | Median speedup | Force err | HVP rel err |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in mode_rows:
        lines.append(
            f"| {row['system']} | {row['batch_size']} | "
            f"{row['equivalent']} | {row['median_speedup']:.3f} | "
            f"{row['equivalence']['max_abs_force_error_eV_per_A']:.3e} | "
            f"{row['equivalence']['relative_hvp_norm_error']:.3e} |"
        )
    lines.extend(
        [
            "",
            "This is a fixed HVP execution microbenchmark. It does not reduce "
            "FE or establish terminal-search improvement.",
            "",
        ]
    )
    (RUN_ROOT / "conclusion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return evidence


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
