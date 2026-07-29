#!/usr/bin/env python3
"""Run the public validated C60 profile for exactly 200 macro steps."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from pamssw import validated_ls_ssw_config, validated_profile_metadata


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FROZEN_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
PROFILE = "c60_direction_efficient_validated_20260729"
MAX_TRIALS = 200
SEED = 42
EXPECTED_OUTPUT_FILES = (
    "best_minimum.xyz",
    "energy_trace.json",
    "walk_records.json",
    "optimizer_diagnostics.json",
    "summary.json",
)
_BASE_RUN_LOCK = threading.Lock()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_c60_k4_frozen_production_runner", FROZEN_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen C60 production runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(mapping), allow_nan=False))


def _profile_config(output_dir: Path):
    return validated_ls_ssw_config(
        PROFILE,
        output_dir=output_dir,
        max_trials=MAX_TRIALS,
        rng_seed=SEED,
        max_force_evals=None,
    )


def config_projection(output_dir: Path) -> dict[str, Any]:
    """Return the exact JSON form persisted by the production runner."""
    projected = _json_mapping(asdict(_profile_config(Path(output_dir))))
    required = {
        "max_trials": MAX_TRIALS,
        "rng_seed": SEED,
        "max_force_evals": None,
        "oracle_candidates": 4,
        "max_steps_per_walk": 8,
        "proposal_relax_steps": 80,
        "proposal_optimizer": "safe-lbfgs-total",
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "direction_type_ucb_enabled": False,
        "direction_probe_enabled": False,
        "plateau_evolution_enabled": False,
    }
    drift = {
        field: [expected, projected.get(field)]
        for field, expected in required.items()
        if projected.get(field) != expected
    }
    if drift:
        raise RuntimeError(f"validated profile drifted from approved protocol: {drift}")
    return projected


def preflight(
    *,
    output_dir: Path,
    expected_git_commit: str,
    base_runner: ModuleType | None = None,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if expected_git_commit != actual_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not FROZEN_RUNNER_PATH.is_file():
        raise FileNotFoundError(FROZEN_RUNNER_PATH)

    base = _base_runner() if base_runner is None else base_runner
    base_preflight = dict(
        base.preflight(system="c60", expected_git_commit=expected_git_commit)
    )
    if base_preflight.get("execution_commit") != actual_commit:
        raise RuntimeError("frozen runner preflight did not pin execution commit")
    if base_preflight.get("system") != "c60":
        raise RuntimeError("frozen runner preflight returned the wrong system")

    metadata = validated_profile_metadata(PROFILE)
    if metadata.get("model_sha256") != base_preflight.get("model_sha256"):
        raise RuntimeError("validated profile model does not match runtime model")

    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "profile": PROFILE,
        "profile_metadata": metadata,
        "base_runner": {
            "path": str(FROZEN_RUNNER_PATH),
            "sha256": _sha256(FROZEN_RUNNER_PATH),
        },
        "base_preflight": base_preflight,
        "effective_config": config_projection(Path(output_dir)),
    }


def _injected_base_run(
    *,
    base_runner: ModuleType,
    output_dir: Path,
    expected_git_commit: str,
    expected_config: Mapping[str, Any],
) -> None:
    with _BASE_RUN_LOCK:
        original_build_config = base_runner.build_config

        def build_config(system: str, case_dir: Path):
            if system != "c60":
                raise RuntimeError("frozen runner requested an unexpected system")
            config = _profile_config(Path(case_dir))
            if _json_mapping(asdict(config)) != dict(expected_config):
                raise RuntimeError("runtime profile config differs from preflight")
            return config

        base_runner.build_config = build_config
        try:
            base_runner.run(
                system="c60",
                output_dir=output_dir,
                expected_git_commit=expected_git_commit,
                preflight_only=False,
            )
        finally:
            base_runner.build_config = original_build_config


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{field} must be a non-negative integer")
    return value


def _validate_output(
    output_dir: Path, expected_config: Mapping[str, Any]
) -> dict[str, Any]:
    for name in EXPECTED_OUTPUT_FILES:
        if not (output_dir / name).is_file():
            raise RuntimeError(f"production runner did not produce {name}")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise RuntimeError("production summary is not an object")
    if summary.get("effective_config") != dict(expected_config):
        raise RuntimeError("production summary does not contain the exact profile")
    stats = summary.get("stats")
    if not isinstance(stats, dict) or stats.get("n_trials") != MAX_TRIALS:
        raise RuntimeError("production run did not complete all 200 macro steps")
    total = _nonnegative_int(
        summary.get("force_evaluations"), "force_evaluations"
    )
    if stats.get("force_evaluations") != total:
        raise RuntimeError("summary and stats force-evaluation totals differ")
    purpose_counts = summary.get("purpose_counts")
    if not isinstance(purpose_counts, dict):
        raise RuntimeError("purpose accounting is absent")
    purpose_total = sum(
        _nonnegative_int(value, f"purpose_counts[{name!r}]")
        for name, value in purpose_counts.items()
    )
    if purpose_total != total:
        raise RuntimeError("purpose accounting does not close")
    if purpose_counts.get("unattributed") != 0:
        raise RuntimeError("purpose accounting contains unattributed evaluations")
    return summary


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
    base_runner_loader: Callable[[], ModuleType] = _base_runner,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    base_runner = base_runner_loader()
    checked = preflight(
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
        base_runner=base_runner,
    )
    if preflight_only:
        return checked

    _injected_base_run(
        base_runner=base_runner,
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
        expected_config=checked["effective_config"],
    )
    summary = _validate_output(output_dir, checked["effective_config"])
    if "validated_profile_run" in summary:
        raise RuntimeError("production summary already has profile provenance")
    summary["validated_profile_run"] = checked
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
