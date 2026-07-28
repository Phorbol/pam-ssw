#!/usr/bin/env python3
"""Run the frozen 200-trial LS-SSW searches with strict ASE true quenches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FROZEN_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
STRICT_QUENCH_PROTOCOL = {
    "quench_optimizer": "ase-lbfgs",
    "quench_fallback_optimizer": "ase-fire",
    "quench_fmax": 0.01,
    "quench_maxiter": 400,
}
EXPECTED_CONFIG_DIFFS = {
    "c60": {
        "quench_fallback_optimizer": [None, "ase-fire"],
        "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
    },
    "pdo": {
        "quench_fallback_optimizer": [None, "ase-fire"],
        "quench_fmax": [0.03, 0.01],
        "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
    },
}
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
    """Load the frozen production runner without copying its runtime behavior."""

    spec = importlib.util.spec_from_file_location(
        "_safe_lbfgs_strict_quench_frozen_runner", FROZEN_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen 200-trial production runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _effective_config(source_config: object) -> object:
    try:
        return replace(source_config, **STRICT_QUENCH_PROTOCOL)
    except TypeError as error:
        raise RuntimeError(
            "frozen runner did not return a replaceable config"
        ) from error


def _config_mapping(config: object) -> dict[str, Any]:
    try:
        mapping = asdict(config)
    except TypeError as error:
        raise RuntimeError("frozen runner config is not a dataclass") from error
    if not isinstance(mapping, dict):
        raise RuntimeError("frozen runner config is not a mapping dataclass")
    return mapping


def _json_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Use the exact JSON representation that the frozen summary will persist."""

    return json.loads(json.dumps(dict(mapping), allow_nan=False))


def config_projection(
    system: str, case_dir: Path, base_runner: ModuleType
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]]]:
    """Return the full frozen and strict configs plus their exact allowed delta."""

    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    source_config = base_runner.build_config(system, Path(case_dir))
    source = _json_mapping(_config_mapping(source_config))
    effective = _json_mapping(_config_mapping(_effective_config(source_config)))
    if set(source) != set(effective):
        raise RuntimeError("strict true-quench config changed its field set")
    diff = {
        field: [source[field], effective[field]]
        for field in source
        if source[field] != effective[field]
    }
    if diff != EXPECTED_CONFIG_DIFFS[system]:
        raise RuntimeError(
            f"frozen {system} configuration no longer has the approved "
            "strict-quench delta"
        )
    if source.get("max_trials") != 200 or effective.get("max_trials") != 200:
        raise RuntimeError("strict production must retain exactly 200 macro steps")
    if source.get("rng_seed") != 42 or effective.get("rng_seed") != 42:
        raise RuntimeError("strict production must retain seed 42")
    if source.get("proposal_optimizer") != "safe-lbfgs-total":
        raise RuntimeError(
            "strict production must retain safe-LBFGS proposal relaxation"
        )
    if source.get("proposal_fmax") != 0.05:
        raise RuntimeError("strict production must retain proposal fmax 0.05")
    for field, expected in STRICT_QUENCH_PROTOCOL.items():
        if effective.get(field) != expected:
            raise RuntimeError(f"strict production did not set {field} to {expected!r}")
    return source, effective, diff


def preflight(
    *,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    base_runner: ModuleType | None = None,
) -> dict[str, Any]:
    """Pin wrapper/base identity and delegate all MACE/CUDA runtime checks."""

    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
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
        base.preflight(system=system, expected_git_commit=expected_git_commit)
    )
    if base_preflight.get("execution_commit") != actual_commit:
        raise RuntimeError("frozen runner preflight did not pin the wrapper commit")
    if base_preflight.get("system") != system:
        raise RuntimeError("frozen runner preflight returned the wrong system")
    source, effective, diff = config_projection(system, Path(output_dir), base)
    return {
        "schema_version": 1,
        "execution_commit": actual_commit,
        "system": system,
        "base_runner": {
            "path": str(FROZEN_RUNNER_PATH),
            "sha256": _sha256(FROZEN_RUNNER_PATH),
        },
        "base_preflight": base_preflight,
        "source_config": source,
        "effective_config": effective,
        "overrides": dict(STRICT_QUENCH_PROTOCOL),
        "config_diff": diff,
    }


def _injected_base_run(
    *,
    base_runner: ModuleType,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    checked: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Run the frozen implementation once with a temporary exact config override."""

    expected_source = dict(checked["source_config"])
    expected_effective = dict(checked["effective_config"])
    with _BASE_RUN_LOCK:
        original_build_config = base_runner.build_config

        def build_config(injected_system: str, case_dir: Path) -> object:
            if injected_system != system:
                raise RuntimeError("frozen runner requested an unexpected system")
            source_config = original_build_config(injected_system, case_dir)
            source = _json_mapping(_config_mapping(source_config))
            if source != expected_source:
                raise RuntimeError(
                    "frozen runner config differs from preflight source config"
                )
            effective_config = _effective_config(source_config)
            if _json_mapping(_config_mapping(effective_config)) != expected_effective:
                raise RuntimeError(
                    "frozen runner strict config differs from preflight "
                    "effective config"
                )
            return effective_config

        base_runner.build_config = build_config
        try:
            return base_runner.run(
                system=system,
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
    output_dir: Path, expected_effective_config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    for name in EXPECTED_OUTPUT_FILES:
        path = output_dir / name
        if not path.is_file():
            raise RuntimeError(f"frozen runner did not produce {path}")
    summary_path = output_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("frozen runner summary is not an object")
    if payload.get("effective_config") != dict(expected_effective_config):
        raise RuntimeError("frozen runner summary effective config is not exact")
    stats = payload.get("stats")
    if not isinstance(stats, dict) or stats.get("n_trials") != 200:
        raise RuntimeError("strict production did not complete all 200 macro steps")
    force_evaluations = _nonnegative_int(
        payload.get("force_evaluations"), "force_evaluations"
    )
    if stats.get("force_evaluations") != force_evaluations:
        raise RuntimeError("summary and statistics force-evaluation totals differ")
    purpose_counts = payload.get("purpose_counts")
    if not isinstance(purpose_counts, dict):
        raise RuntimeError("summary purpose counts are absent")
    purpose_total = sum(
        _nonnegative_int(value, f"purpose_counts[{name!r}]")
        for name, value in purpose_counts.items()
    )
    if purpose_total != force_evaluations:
        raise RuntimeError("purpose counts do not close against force evaluations")
    if purpose_counts.get("unattributed") != 0:
        raise RuntimeError("purpose accounting contains unattributed evaluations")
    telemetry = payload.get("optimizer_telemetry")
    if not isinstance(telemetry, dict):
        raise RuntimeError("summary optimizer telemetry is absent")
    attempts = _nonnegative_int(
        telemetry.get("quench_fallback_attempts", 0), "quench fallback attempts"
    )
    converged = _nonnegative_int(
        telemetry.get("quench_fallback_converged", 0), "quench fallback converged"
    )
    if converged > attempts:
        raise RuntimeError("quench fallback converged count exceeds attempts")
    return payload, {"attempts": attempts, "converged": converged}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def run(
    *,
    system: str,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
    base_runner_loader: Callable[[], ModuleType] = _base_runner,
    preflight_fn: Callable[..., Mapping[str, Any]] = preflight,
) -> dict[str, Any]:
    """Delegate one frozen production search with only strict quench changes."""

    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    base_runner = base_runner_loader()
    checked = dict(
        preflight_fn(
            system=system,
            output_dir=output_dir,
            expected_git_commit=expected_git_commit,
            base_runner=base_runner,
        )
    )
    if preflight_only:
        return checked

    _injected_base_run(
        base_runner=base_runner,
        system=system,
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
        checked=checked,
    )
    summary, fallback_counts = _validate_output(
        output_dir, checked["effective_config"]
    )
    if "strict_quench_wrapper" in summary:
        raise RuntimeError("frozen runner summary already has wrapper provenance")
    summary["strict_quench_wrapper"] = {
        "schema_version": 1,
        "base_runner": checked["base_runner"],
        "base_preflight": checked["base_preflight"],
        "source_config": checked["source_config"],
        "effective_config": checked["effective_config"],
        "overrides": checked["overrides"],
        "config_diff": checked["config_diff"],
        "fallback_counts": fallback_counts,
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(
        system=args.system,
        output_dir=args.output,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
