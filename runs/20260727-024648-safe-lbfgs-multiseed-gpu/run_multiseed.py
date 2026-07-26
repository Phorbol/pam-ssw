from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import runpy
import time
from typing import Any

import numpy as np
import torch

from pamssw import run_posterior_ssw


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
SOURCE_DRIVER = (
    RUN_ROOT.parent
    / "20260727-014407-bias-relaxation-gpu-g1"
    / "run_g1.py"
)
BASE_REPLAY_SUMMARY = (
    RUN_ROOT.parent
    / "20260727-023234-fixed-proposal-replay-gpu"
    / "output"
    / "summary.json"
)
FROZEN_CODE_COMMIT = "a64816b25c91f93b7d5a3e9c882fc036853c03d9"
SYSTEMS = ("c60", "pdo")
BACKENDS = ("ase-fire", "safe-lbfgs-total")
SEEDS = (42, 43, 44)


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


def _run_arm(
    source: dict[str, Any],
    *,
    system: str,
    backend: str,
    seed: int,
    certified_state,
) -> dict[str, Any]:
    case_root = OUTPUT_ROOT / f"{system}__{backend}__seed{seed}"
    case_root.mkdir()
    ssw_config = replace(
        source["_ssw_config"](system, backend),
        rng_seed=seed,
    )
    exploration = replace(
        source["_exploration_config"](case_root),
        master_seed=seed,
    )
    started = time.perf_counter()
    result = run_posterior_ssw(
        certified_state,
        source["_calculator"],
        ssw_config,
        exploration,
    )
    wall_time = time.perf_counter() - started
    entries = sorted(result.archive.entries, key=lambda entry: entry.energy)
    bootstrap_entry = next(
        entry for entry in result.archive.entries if entry.entry_id == 0
    )
    diagnostics = source["_diagnostic_facts"](
        exploration.run_directory / "optimizer_diagnostics.json"
    )
    payload = {
        "schema_version": 1,
        "frozen_code_commit": FROZEN_CODE_COMMIT,
        "system": system,
        "backend": backend,
        "seed": seed,
        "input": str(source["SYSTEMS"][system]["input"]),
        "input_sha256": source["SYSTEMS"][system]["sha256"],
        "certified_state_source": str(BASE_REPLAY_SUMMARY),
        "certified_state_source_sha256": _sha256(BASE_REPLAY_SUMMARY),
        "model": str(source["MODEL"]),
        "model_sha256": source["MODEL_SHA256"],
        "device": "cuda",
        "dtype": "float32",
        "policy_name": exploration.policy_name,
        "batch_size": exploration.batch_size,
        "max_workers": exploration.max_workers,
        "action_force_budget": exploration.action_force_budget,
        "total_force_budget": exploration.total_force_budget,
        "bootstrap_evaluations": result.bootstrap_evaluations,
        "action_evaluations": result.action_evaluations,
        "total_evaluations": result.total_evaluations,
        "unused_force_budget": result.unused_force_budget,
        "purpose_counts": result.purpose_counts.as_dict(),
        "completed_batches": result.completed_batches,
        "completed_attempts": result.completed_attempts,
        "failed_attempts": result.failed_attempts,
        "posterior_observed_attempts": result.posterior_observed_attempts,
        "stop_reason": result.stop_reason.value,
        "benchmark_eligible": result.benchmark_eligible,
        "benchmark_ineligibility_reasons": list(
            result.benchmark_ineligibility_reasons
        ),
        "bootstrap_energy_eV": float(bootstrap_entry.energy),
        "best_energy_eV": float(entries[0].energy),
        "best_energy_drop_from_bootstrap_eV": float(
            bootstrap_entry.energy - entries[0].energy
        ),
        "unique_minima": len(entries),
        "archive_energies_eV": [float(entry.energy) for entry in entries],
        "optimizer_diagnostics": diagnostics,
        "wall_time_s": wall_time,
    }
    _write_json(case_root / "result.json", payload)
    return payload


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    systems = []
    for system in SYSTEMS:
        backend_rows = {}
        for backend in BACKENDS:
            selected = [
                row
                for row in rows
                if row["system"] == system and row["backend"] == backend
            ]
            drops = np.asarray(
                [row["best_energy_drop_from_bootstrap_eV"] for row in selected],
                dtype=float,
            )
            backend_rows[backend] = {
                "seeds": [row["seed"] for row in selected],
                "benchmark_eligible_count": sum(
                    row["benchmark_eligible"] for row in selected
                ),
                "best_energy_drop_median_eV": float(np.median(drops)),
                "best_energy_drop_min_eV": float(np.min(drops)),
                "best_energy_drop_max_eV": float(np.max(drops)),
                "unique_minima": [row["unique_minima"] for row in selected],
                "completed_attempts": [
                    row["completed_attempts"] for row in selected
                ],
                "action_evaluations": [
                    row["action_evaluations"] for row in selected
                ],
                "biased_proposal_evaluations": [
                    row["purpose_counts"]["biased_proposal_relax"]
                    for row in selected
                ],
                "proposal_relax_count": [
                    int(
                        row["optimizer_diagnostics"]["sums"][
                            "proposal_relax_count"
                        ]
                    )
                    for row in selected
                ],
                "proposal_relax_converged": [
                    int(
                        row["optimizer_diagnostics"]["sums"][
                            "proposal_relax_termination_converged"
                        ]
                    )
                    for row in selected
                ],
                "wall_time_s": [row["wall_time_s"] for row in selected],
            }
        paired = []
        for seed in SEEDS:
            fire = next(
                row
                for row in rows
                if row["system"] == system
                and row["backend"] == "ase-fire"
                and row["seed"] == seed
            )
            safe = next(
                row
                for row in rows
                if row["system"] == system
                and row["backend"] == "safe-lbfgs-total"
                and row["seed"] == seed
            )
            paired.append(
                {
                    "seed": seed,
                    "safe_minus_fire_best_energy_drop_eV": (
                        safe["best_energy_drop_from_bootstrap_eV"]
                        - fire["best_energy_drop_from_bootstrap_eV"]
                    ),
                    "safe_minus_fire_unique_minima": (
                        safe["unique_minima"] - fire["unique_minima"]
                    ),
                    "safe_minus_fire_completed_attempts": (
                        safe["completed_attempts"] - fire["completed_attempts"]
                    ),
                    "safe_minus_fire_action_evaluations": (
                        safe["action_evaluations"] - fire["action_evaluations"]
                    ),
                    "safe_minus_fire_biased_proposal_evaluations": (
                        safe["purpose_counts"]["biased_proposal_relax"]
                        - fire["purpose_counts"]["biased_proposal_relax"]
                    ),
                }
            )
        systems.append(
            {
                "system": system,
                "backends": backend_rows,
                "paired": paired,
            }
        )
    return {"systems": systems}


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    source = _source()
    base_replay = json.loads(BASE_REPLAY_SUMMARY.read_text(encoding="utf-8"))
    certified_states = {}
    for system_data in base_replay["systems"]:
        system = system_data["system"]
        certified_states[system] = source["_state"](system).with_flat_positions(
            system_data["bootstrap"]["positions"]
        )
    if _sha256(source["MODEL"]) != source["MODEL_SHA256"]:
        raise ValueError("model hash mismatch")
    for spec in source["SYSTEMS"].values():
        if _sha256(spec["input"]) != spec["sha256"]:
            raise ValueError(f"input hash mismatch: {spec['input']}")

    started = time.perf_counter()
    rows = []
    for system in SYSTEMS:
        for seed in SEEDS:
            for backend in BACKENDS:
                arm = _run_arm(
                    source,
                    system=system,
                    backend=backend,
                    seed=seed,
                    certified_state=certified_states[system],
                )
                rows.append(arm)
                print(
                    json.dumps(
                        {
                            "system": system,
                            "seed": seed,
                            "backend": backend,
                            "best_energy_drop_eV": (
                                arm["best_energy_drop_from_bootstrap_eV"]
                            ),
                            "total_evaluations": arm["total_evaluations"],
                            "wall_time_s": arm["wall_time_s"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    payload = {
        "schema_version": 1,
        "frozen_code_commit": FROZEN_CODE_COMMIT,
        "systems": list(SYSTEMS),
        "backends": list(BACKENDS),
        "seeds": list(SEEDS),
        "policy_name": "uniform",
        "batch_size": 2,
        "max_workers": 2,
        "action_force_budget": 1000,
        "total_force_budget": 3000,
        "certified_state_source": str(BASE_REPLAY_SUMMARY),
        "certified_state_source_sha256": _sha256(BASE_REPLAY_SUMMARY),
        "rows": rows,
        "aggregate": _aggregate(rows),
        "wall_time_total_s": time.perf_counter() - started,
    }
    _write_json(OUTPUT_ROOT / "summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
