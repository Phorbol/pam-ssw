#!/usr/bin/env python3
"""Run a five-trial GPU check that exact direction-oracle reuse is trajectory-neutral."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

from pamssw.accounting import EvaluationPurpose


RUN_ROOT = Path(__file__).resolve().parent
PRODUCTION_ROOT = RUN_ROOT.parents[0] / "20260728-safe-lbfgs-200-production"
PRODUCTION_RUNNER_PATH = PRODUCTION_ROOT / "run_production.py"
OLD_OUTPUT_ROOT = PRODUCTION_ROOT / "output"
DEFAULT_REFERENCE_PATH = RUN_ROOT / "reference.json"
SYSTEMS = ("c60", "pdo")
MAX_TRIALS = 5


def _load_production_runner():
    spec = importlib.util.spec_from_file_location(
        "_safe_lbfgs_200_production_for_no_loss_validation",
        PRODUCTION_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {PRODUCTION_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRODUCTION_RUNNER = _load_production_runner()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _old_case_reference(case_dir: Path, system: str) -> dict[str, Any]:
    paths = {
        "summary": case_dir / "summary.json",
        "energy_trace": case_dir / "energy_trace.json",
        "walk_records": case_dir / "walk_records.json",
        "direction_trace": case_dir / "direction_trace.jsonl",
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [str(path) for path in paths.values() if not path.is_file()]
        raise FileNotFoundError(", ".join(missing))
    summary = _read_json(paths["summary"])
    energy_trace = _read_json(paths["energy_trace"])
    walk_records = _read_json(paths["walk_records"])
    direction_trace = _read_jsonl(paths["direction_trace"])
    if summary.get("system") != system:
        raise ValueError(f"{system} reference summary identity mismatch")
    if len(energy_trace) < MAX_TRIALS + 1 or len(walk_records) < MAX_TRIALS:
        raise ValueError(f"{system} old output does not contain {MAX_TRIALS} trials")
    energy_prefix = energy_trace[: MAX_TRIALS + 1]
    walk_prefix = walk_records[:MAX_TRIALS]
    if [row.get("trial") for row in energy_prefix] != list(range(MAX_TRIALS + 1)):
        raise ValueError(f"{system} energy-trace prefix is not ordered")
    if [row.get("trial") for row in walk_prefix] != list(range(1, MAX_TRIALS + 1)):
        raise ValueError(f"{system} walk-record prefix is not ordered")
    direction_prefix = [row for row in direction_trace if int(row["trial"]) < MAX_TRIALS]
    if not direction_prefix:
        raise ValueError(f"{system} direction trace has no first-five records")
    return {
        "old_execution_commit": summary["execution_commit"],
        "old_artifact_sha256": {name: _sha256(path) for name, path in paths.items()},
        "energy_trace": energy_prefix,
        "walk_records": walk_prefix,
        "direction_trace": direction_prefix,
    }


def freeze_reference(old_output_root: Path, reference_path: Path) -> dict[str, Any]:
    """Persist the old 200-trial artifacts required for the exact five-trial comparison."""

    old_output_root = Path(old_output_root)
    payload = {
        "schema_version": 1,
        "trial_count": MAX_TRIALS,
        "old_output_root": str(old_output_root),
        "systems": {
            system: {"system": system, **_old_case_reference(old_output_root / system, system)}
            for system in SYSTEMS
        },
    }
    _write_json(Path(reference_path), payload)
    return payload


def _load_reference(reference_path: Path) -> dict[str, Any]:
    reference = _read_json(Path(reference_path))
    if reference.get("schema_version") != 1 or reference.get("trial_count") != MAX_TRIALS:
        raise ValueError("reference schema or trial count mismatch")
    if set(reference.get("systems", ())) != set(SYSTEMS):
        raise ValueError("reference systems mismatch")
    return reference


def build_config(system: str, case_dir: Path):
    """Use the production configuration unchanged except for the trial horizon."""

    return replace(PRODUCTION_RUNNER.build_config(system, case_dir), max_trials=MAX_TRIALS)


def preflight(*, system: str, expected_git_commit: str, reference_path: Path) -> dict[str, Any]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    reference = _load_reference(reference_path)
    source = reference["systems"][system]
    if not source.get("old_execution_commit") or not source.get("old_artifact_sha256"):
        raise ValueError("reference lacks old execution provenance")
    return {
        **PRODUCTION_RUNNER.preflight(system=system, expected_git_commit=expected_git_commit),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "old_execution_commit": source["old_execution_commit"],
        "old_artifact_sha256": source["old_artifact_sha256"],
    }


def run(
    *,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    preflight_only: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    provenance = preflight(
        system=system,
        expected_git_commit=expected_git_commit,
        reference_path=Path(reference_path),
    )
    if preflight_only:
        return provenance

    output_dir.mkdir(parents=True)
    state = PRODUCTION_RUNNER.load_state(system)
    config = build_config(system, output_dir)
    walker = PRODUCTION_RUNNER.SurfaceWalker(
        calculator=PRODUCTION_RUNNER.ASECalculator(PRODUCTION_RUNNER._calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    wall_time_s = perf_counter() - started
    counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    force_evaluations = int(result.stats["force_evaluations"])
    if counts.total != force_evaluations:
        raise RuntimeError("force-evaluation total does not close")
    if purpose_counts[EvaluationPurpose.UNATTRIBUTED.value] != 0:
        raise RuntimeError("purpose accounting contains unattributed evaluations")
    if int(result.stats["n_trials"]) != MAX_TRIALS:
        raise RuntimeError("validation run did not complete all five trials")

    energy_trace = PRODUCTION_RUNNER._energy_trace(result)
    walk_records = PRODUCTION_RUNNER._walk_records(result)
    optimizer_telemetry = walker.relaxation_diagnostics()
    summary = {
        **provenance,
        "effective_config": asdict(config),
        "initial_energy_eV": float(energy_trace[0]["energy_eV"]),
        "best_energy_eV": float(result.best_energy),
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "optimizer_telemetry": optimizer_telemetry,
        "stats": result.stats,
        "timing": {"total_wall_time_s": wall_time_s},
        "walk_records": walk_records,
    }
    PRODUCTION_RUNNER.write_state(output_dir / "best_minimum.xyz", result.best_state)
    PRODUCTION_RUNNER._write_json(output_dir / "energy_trace.json", energy_trace)
    PRODUCTION_RUNNER._write_json(output_dir / "walk_records.json", walk_records)
    PRODUCTION_RUNNER._write_json(output_dir / "optimizer_diagnostics.json", optimizer_telemetry)
    PRODUCTION_RUNNER._write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-reference", action="store_true")
    parser.add_argument("--old-output-root", type=Path, default=OLD_OUTPUT_ROOT)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--system", choices=SYSTEMS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.freeze_reference:
        payload = freeze_reference(args.old_output_root, args.reference)
    else:
        if args.system is None or args.output is None or args.expected_git_commit is None:
            raise SystemExit("--system, --output, and --expected-git-commit are required to run validation")
        payload = run(
            system=args.system,
            output_dir=args.output,
            expected_git_commit=args.expected_git_commit,
            reference_path=args.reference,
            preflight_only=args.preflight_only,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
