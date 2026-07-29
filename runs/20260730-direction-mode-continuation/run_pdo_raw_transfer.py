#!/usr/bin/env python3
"""Run the exact raw-PdO direction-continuation transfer ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "run_ablation.py"
FIXED_AUDIT_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-block-krylov-direction-audit"
    / "run_fixed_state_audit.py"
)
SYSTEM = "pdo"
STATE_ID = "raw_bootstrap"
SEEDS = (42, 43, 44)
ARMS = ("fixed_intent_ritz", "transported_direction")
RAW_INPUT_SHA256 = (
    "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0"
)
MAX_TOTAL_FORCE_EVALUATIONS = 30_000


def case_matrix() -> list[dict[str, Any]]:
    return [
        {"state_id": STATE_ID, "seed": seed, "arm": arm}
        for seed in SEEDS
        for arm in ARMS
    ]


def _load_runtime():
    protocol = _load_module(PROTOCOL_PATH, "_pdo_raw_shared_protocol")
    audit = _load_module(FIXED_AUDIT_PATH, "_pdo_raw_fixed_audit")
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    return protocol, base_runner


def _load_module(path: Path, name: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _strict_pdo_config(base_runner, case_dir: Path, *, seed: int):
    return replace(
        base_runner.build_config(SYSTEM, case_dir),
        max_trials=1,
        max_force_evals=None,
        rng_seed=seed,
        quench_optimizer="ase-lbfgs",
        quench_fallback_optimizer="ase-fire",
        quench_fmax=0.01,
    )


def _bootstrap(raw_state, shared_calculator, base_runner, protocol, output_dir: Path):
    from pamssw.accounting import EvaluationPurpose
    from pamssw.relax import has_force_convergence_certificate
    from pamssw.walker import SurfaceWalker

    config = _strict_pdo_config(
        base_runner,
        output_dir / "bootstrap",
        seed=SEEDS[0],
    )
    walker = SurfaceWalker(
        calculator=shared_calculator,
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH):
        raw_evaluation = walker.calculator.evaluate(raw_state)
    relaxed = walker.relax_true_minimum(
        raw_state,
        trajectory_name="pdo-raw-bootstrap",
        quench_purpose=EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH,
    )
    wall_time = float(perf_counter() - started)
    certificate = bool(
        has_force_convergence_certificate(relaxed, config.quench_fmax)
    )
    geometry_valid = bool(
        walker.geometry_validator.is_valid_state(relaxed.state)
    )
    purposes = walker.calculator.snapshot().as_dict()
    force_evaluations = int(sum(purposes.values()))
    if (
        not certificate
        or not geometry_valid
        or purposes["unattributed"] != 0
        or purposes["bootstrap_true_quench"] <= 0
        or any(
            value != 0
            for purpose, value in purposes.items()
            if purpose
            not in {"bootstrap_true_quench", "post_relax_validation"}
        )
    ):
        raise RuntimeError("raw PdO bootstrap certificate or ledger failed")

    raw_path = output_dir / "raw_input.xyz"
    minimum_path = output_dir / "bootstrap_minimum.xyz"
    base_runner.write_state(raw_path, raw_state)
    base_runner.write_state(minimum_path, relaxed.state)
    record = {
        "certificate": certificate,
        "geometry_valid": geometry_valid,
        "raw_energy_eV": float(raw_evaluation.energy),
        "bootstrap_energy_eV": float(relaxed.energy),
        "energy_drop_eV": float(raw_evaluation.energy) - float(relaxed.energy),
        "final_max_force_eV_per_A": float(relaxed.gradient_norm),
        "iterations": int(relaxed.n_iter),
        "termination_reason": relaxed.telemetry.termination_reason,
        "fallback_used": bool(
            walker.relaxation_diagnostics()["quench_fallback_attempts"]
        ),
        "force_evaluations": force_evaluations,
        "purpose_counts": purposes,
        "wall_time_s": wall_time,
        "raw_state_sha256": protocol._state_sha256(raw_state),
        "bootstrap_state_sha256": protocol._state_sha256(relaxed.state),
        "raw_path": str(raw_path),
        "raw_file_sha256": protocol._sha256(raw_path),
        "bootstrap_path": str(minimum_path),
        "bootstrap_file_sha256": protocol._sha256(minimum_path),
        "effective_config": asdict(config),
    }
    return relaxed.state, record


def run(output_dir: Path) -> dict[str, Any]:
    from pamssw.calculators import ASECalculator

    output_dir = Path(output_dir)
    protocol, base_runner = _load_runtime()
    execution_commit = protocol._current_commit()
    if not protocol._tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    raw_input_path = Path(base_runner.INPUT_PATHS[SYSTEM])
    if protocol._sha256(raw_input_path) != RAW_INPUT_SHA256:
        raise RuntimeError("original PdO input checksum drifted")
    output_dir.mkdir(parents=True)

    raw_state = base_runner.load_state(SYSTEM)
    shared_calculator = ASECalculator(base_runner._calculator())
    bootstrap_state, bootstrap = _bootstrap(
        raw_state,
        shared_calculator,
        base_runner,
        protocol,
        output_dir,
    )
    state_provenance = {
        "state_id": STATE_ID,
        "state_sha256": protocol._state_sha256(bootstrap_state),
        "source": "single_strict_true_quench_of_original_pdo",
        "raw_input_path": str(raw_input_path),
        "raw_input_sha256": RAW_INPUT_SHA256,
        "bootstrap_force_evaluations": bootstrap["force_evaluations"],
    }

    rows: list[dict[str, Any]] = []
    shared_records: list[dict[str, Any]] = []
    total_force_evaluations = int(bootstrap["force_evaluations"])
    for seed in SEEDS:
        initial_choice, shared_record = (
            protocol._precompute_shared_initial_direction(
                state=bootstrap_state,
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                state_id=STATE_ID,
                seed=seed,
                shared_dir=(
                    output_dir / "shared-initial" / f"{STATE_ID}-seed{seed}"
                ),
                system=SYSTEM,
            )
        )
        shared_records.append(shared_record)
        total_force_evaluations += int(shared_record["force_evaluations"])
        for arm in ARMS:
            print(
                f"[pdo-raw-transfer] seed={seed} arm={arm}",
                flush=True,
            )
            case_dir = output_dir / "cases" / f"{STATE_ID}-seed{seed}-{arm}"
            row = protocol._run_case(
                state=bootstrap_state,
                state_provenance=state_provenance,
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                state_id=STATE_ID,
                seed=seed,
                arm=arm,
                case_dir=case_dir,
                initial_direction_choice=initial_choice,
                system=SYSTEM,
                fragmentation_applicable=False,
            )
            rows.append(row)
            total_force_evaluations += int(row["force_evaluations"])
            if total_force_evaluations > MAX_TOTAL_FORCE_EVALUATIONS:
                raise RuntimeError("raw PdO transfer exceeded its 30000-FE stop")
            protocol._write_json(
                output_dir / "raw.json",
                {
                    "schema_version": 1,
                    "execution_commit": execution_commit,
                    "system": SYSTEM,
                    "state_id": STATE_ID,
                    "raw_input_path": str(raw_input_path),
                    "raw_input_sha256": RAW_INPUT_SHA256,
                    "model_path": str(base_runner.MODEL_PATH),
                    "model_sha256": protocol._sha256(base_runner.MODEL_PATH),
                    "bootstrap": bootstrap,
                    "state_provenance": state_provenance,
                    "shared_initial_directions": shared_records,
                    "cases": rows,
                },
            )
    return {
        "execution_commit": execution_commit,
        "completed_cases": len(rows),
        "bootstrap_force_evaluations": bootstrap["force_evaluations"],
        "shared_initial_direction_force_evaluations": sum(
            int(item["force_evaluations"]) for item in shared_records
        ),
        "action_force_evaluations": sum(
            int(row["force_evaluations"]) for row in rows
        ),
        "total_force_evaluations": total_force_evaluations,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUN_ROOT / "pdo-raw-output",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
