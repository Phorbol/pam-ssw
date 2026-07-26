from __future__ import annotations

import json
from pathlib import Path
import runpy

import torch


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
MULTISEED_DRIVER = (
    RUN_ROOT.parent
    / "20260727-024648-safe-lbfgs-multiseed-gpu"
    / "run_multiseed.py"
)
FROZEN_CODE_COMMIT = "61f32ef70585adfd78b1484bd6f0cfcac8317352"


def _write_json(path: Path, payload) -> None:
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

    driver = runpy.run_path(str(MULTISEED_DRIVER))
    source = driver["_source"]()
    base_replay_path = driver["BASE_REPLAY_SUMMARY"]
    base_replay = json.loads(base_replay_path.read_text(encoding="utf-8"))
    pdo = next(
        item for item in base_replay["systems"] if item["system"] == "pdo"
    )
    certified_state = source["_state"]("pdo").with_flat_positions(
        pdo["bootstrap"]["positions"]
    )
    arm_function = driver["_run_arm"]
    arm_function.__globals__["OUTPUT_ROOT"] = OUTPUT_ROOT
    arm_function.__globals__["FROZEN_CODE_COMMIT"] = FROZEN_CODE_COMMIT
    result = arm_function(
        source,
        system="pdo",
        backend="safe-lbfgs-total",
        seed=44,
        certified_state=certified_state,
    )
    event_path = (
        OUTPUT_ROOT
        / "pdo__safe-lbfgs-total__seed44"
        / "campaign"
        / "events.jsonl"
    )
    attempts = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["record_type"] == "attempt"
    ]
    payload = {
        "schema_version": 1,
        "frozen_code_commit": FROZEN_CODE_COMMIT,
        "system": "pdo",
        "backend": "safe-lbfgs-total",
        "seed": 44,
        "benchmark_eligible": result["benchmark_eligible"],
        "benchmark_ineligibility_reasons": (
            result["benchmark_ineligibility_reasons"]
        ),
        "attempt_count": len(attempts),
        "worker_error_count": sum(
            item["status"] == "worker_error" for item in attempts
        ),
        "posterior_observed_count": sum(
            item["posterior_observed"] for item in attempts
        ),
        "completed_attempts": result["completed_attempts"],
        "failed_attempts": result["failed_attempts"],
        "total_evaluations": result["total_evaluations"],
        "unused_force_budget": result["unused_force_budget"],
        "purpose_counts": result["purpose_counts"],
        "best_energy_drop_from_bootstrap_eV": (
            result["best_energy_drop_from_bootstrap_eV"]
        ),
    }
    _write_json(OUTPUT_ROOT / "verification.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
