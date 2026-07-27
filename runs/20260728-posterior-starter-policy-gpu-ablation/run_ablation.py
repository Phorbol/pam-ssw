#!/usr/bin/env python3
"""Serial, exact-accounting C60/PdO posterior starter-policy GPU ablation.

This harness intentionally compares only the existing outer starter policies
under one production-derived *unsoftened* SSW action.  Proposal relaxation is
unchanged; true quenching uses one fixed, explicitly recorded ASE-LBFGS with
FIRE fallback protocol shared by every policy.  The policy, credit, direction,
and archive algorithms are unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from hashlib import sha256
import importlib.metadata
import importlib.util
import json
from math import isfinite
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pamssw.calculators import ASECalculator
from pamssw.config import SSWConfig
from pamssw.exploration import (
    ExplorationEventLog,
    PosteriorExplorationConfig,
    run_posterior_ssw,
)
from pamssw.exploration.policies import SUPPORTED_POLICIES
from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PAMSSW_SOURCE_ROOT = REPO_ROOT / "pamssw"
PRODUCTION_RUNNER_PATH = (
    RUN_ROOT.parent / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
SYSTEMS = ("c60", "pdo")
POLICIES = ("uniform", "posterior_proportional", "minimal_ucb")
SERIAL_BATCH_SIZE = 1
SERIAL_MAX_WORKERS = 1
SCHEMA_VERSION = 1
RUNTIME_PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "ase": "ase",
    "torch": "torch",
    "mace": "mace-torch",
}
_PRODUCTION_MODULE_NAME = "_posterior_starter_policy_production_runner"


def _load_production_runner():
    module = sys.modules.get(_PRODUCTION_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(
        _PRODUCTION_MODULE_NAME,
        PRODUCTION_RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen production runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PRODUCTION_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pamssw_bundle_sha256(source_root: Path) -> str:
    paths = sorted(
        path
        for path in source_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )
    if not paths:
        raise ValueError("pamssw source bundle is empty")
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
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


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for key, distribution in RUNTIME_PACKAGES.items():
        versions[key] = importlib.metadata.version(distribution)
    return versions


def _cuda_info() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "requested_device": "cuda",
        "available": True,
        "runtime_version": str(torch.version.cuda),
        "device_name": str(torch.cuda.get_device_name(0)),
    }


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validated_systems(systems: object) -> tuple[str, ...]:
    try:
        values = tuple(systems)
    except TypeError as exc:
        raise TypeError("systems must be iterable") from exc
    if not values:
        raise ValueError("systems must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("systems must be unique")
    if any(value not in SYSTEMS for value in values):
        raise ValueError(f"systems must be drawn from {SYSTEMS!r}")
    return values


def _validated_seeds(seeds: object) -> tuple[int, ...]:
    try:
        values = tuple(seeds)
    except TypeError as exc:
        raise TypeError("master seeds must be iterable") from exc
    if not values:
        raise ValueError("master seeds must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("master seeds must be unique")
    return tuple(_nonnegative_int("master_seed", value) for value in values)


def _experiment_overrides() -> dict[str, object]:
    return {
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "accepted_structures_log": None,
        "accepted_structures_dir": None,
        "write_proposal_minima": False,
        "proposal_minima_dir": None,
        "write_relaxation_trajectories": False,
        "relaxation_trajectory_dir": None,
        "direction_diagnostics_enabled": False,
        "direction_diagnostics_path": None,
        "direction_archive_enabled": False,
        "direction_archive_path": None,
        "proposal_pool_size": 1,
        "proposal_duplicate_rescue_optimizer": None,
    }


def build_ssw_config(system: str, case_directory: Path) -> tuple[SSWConfig, dict[str, object]]:
    """Project the frozen LS-SSW production settings onto plain SSW exactly once."""
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    production = _load_production_runner()
    source = production.build_config(system, Path(case_directory))
    source_values = asdict(source)
    common_names = {item.name for item in fields(SSWConfig)}
    values = {name: source_values[name] for name in common_names}
    overrides = _experiment_overrides()
    values.update(overrides)
    config = SSWConfig(**values)
    if config.proposal_pool_size != 1:
        raise RuntimeError("posterior worker requires proposal_pool_size=1")
    if config.proposal_duplicate_rescue_optimizer is not None:
        raise RuntimeError("posterior worker requires duplicate rescue disabled")
    projection = {
        "source_config_type": type(source).__name__,
        "source_config": source_values,
        "effective_ssw_config": asdict(config),
        "softening_enabled": False,
        "removed_ls_fields": sorted(set(source_values) - common_names),
        "overrides": overrides,
    }
    return config, projection


def build_exploration_config(
    *,
    policy_name: str,
    run_directory: Path,
    master_seed: int,
    action_force_budget: int,
    total_force_budget: int,
    batch_size: int = SERIAL_BATCH_SIZE,
    max_workers: int = SERIAL_MAX_WORKERS,
) -> PosteriorExplorationConfig:
    """Build one deliberately serial campaign configuration."""
    if policy_name not in POLICIES or policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {policy_name}")
    if batch_size != SERIAL_BATCH_SIZE:
        raise ValueError("this ablation requires batch_size=1")
    if max_workers != SERIAL_MAX_WORKERS:
        raise ValueError("this ablation requires max_workers=1")
    return PosteriorExplorationConfig(
        policy_name=policy_name,
        batch_size=batch_size,
        max_workers=max_workers,
        action_force_budget=_positive_int("action_force_budget", action_force_budget),
        total_force_budget=_positive_int("total_force_budget", total_force_budget),
        master_seed=_nonnegative_int("master_seed", master_seed),
        run_directory=Path(run_directory),
    )


class ThreadOwnedCalculatorFactory:
    """Avoid calculator sharing between bootstrap and the sole action worker.

    The posterior runner invokes its bootstrap on the caller thread, then all
    actions through one persistent ``ThreadPoolExecutor`` worker.  Returning a
    calculator from another thread is rejected rather than assumed safe.
    """

    def __init__(self, builder: Callable[[], object]) -> None:
        if not callable(builder):
            raise TypeError("builder must be callable")
        self._builder = builder
        self._main_thread_id = threading.get_ident()
        self._lock = threading.Lock()
        self._bootstrap_instances = 0
        self._action_instances = 0
        self._action_thread_id: int | None = None
        self._action_calculator: object | None = None

    def __call__(self) -> object:
        current_thread_id = threading.get_ident()
        with self._lock:
            if current_thread_id == self._main_thread_id:
                if self._bootstrap_instances != 0:
                    raise RuntimeError("bootstrap calculator factory was called more than once")
                calculator = self._builder()
                self._bootstrap_instances = 1
                return calculator
            if self._action_thread_id is None:
                self._action_thread_id = current_thread_id
            elif self._action_thread_id != current_thread_id:
                raise RuntimeError("serial action calculator factory observed multiple worker threads")
            if self._action_calculator is None:
                self._action_calculator = self._builder()
                self._action_instances = 1
            return self._action_calculator

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "bootstrap_instances": self._bootstrap_instances,
                "action_instances": self._action_instances,
                "action_thread_count": int(self._action_thread_id is not None),
            }


class _TimedCalculator:
    """One factory-call-local timing adapter with no effect on PES results."""

    def __init__(self, calculator: object, record: dict[str, object]) -> None:
        self._calculator = calculator
        self._record = record

    def evaluate(self, state: State):
        return self._measure(self._calculator.evaluate, state)

    def evaluate_flat(self, flat_positions, template: State):
        return self._measure(self._calculator.evaluate_flat, flat_positions, template)

    def _measure(self, evaluator, *args):
        started = perf_counter()
        try:
            return evaluator(*args)
        finally:
            elapsed = perf_counter() - started
            self._record["evaluator_calls"] = int(self._record["evaluator_calls"]) + 1
            self._record["evaluator_wall_time_s"] = float(
                self._record["evaluator_wall_time_s"]
            ) + elapsed


class InstrumentedCalculatorFactory:
    """Create one timing adapter per bootstrap/action while reusing its base model."""

    def __init__(self, owner: ThreadOwnedCalculatorFactory) -> None:
        if not isinstance(owner, ThreadOwnedCalculatorFactory):
            raise TypeError("owner must be a ThreadOwnedCalculatorFactory")
        self._owner = owner
        self._main_thread_id = threading.get_ident()
        self._lock = threading.Lock()
        self._bootstrap_records: list[dict[str, object]] = []
        self._action_records: list[dict[str, object]] = []

    def __call__(self):
        calculator = self._owner()
        phase = "bootstrap" if threading.get_ident() == self._main_thread_id else "action"
        record: dict[str, object] = {
            "phase": phase,
            "evaluator_calls": 0,
            "evaluator_wall_time_s": 0.0,
        }
        with self._lock:
            if phase == "bootstrap":
                self._bootstrap_records.append(record)
            else:
                self._action_records.append(record)
        return _TimedCalculator(calculator, record)

    def snapshot(self) -> dict[str, int]:
        return self._owner.snapshot()

    def action_records(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(record) for record in self._action_records)

    def bootstrap_records(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(record) for record in self._bootstrap_records)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False))
            stream.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _event_attempts(event_path: Path) -> list[dict[str, Any]]:
    """Recover action facts only after the authoritative log has replayed."""
    replayed = ExplorationEventLog(event_path).reconstruct_posterior()
    del replayed
    active_snapshot: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    for row in read_jsonl(event_path):
        record_type = row.get("record_type")
        if record_type == "policy_snapshot":
            active_snapshot = row
        elif record_type == "attempt":
            if active_snapshot is None:
                raise RuntimeError("attempt appeared without an active policy snapshot")
            attempts.append({"attempt": row, "snapshot": dict(active_snapshot)})
        elif record_type == "batch_commit":
            if active_snapshot is None:
                raise RuntimeError("batch commit appeared without a policy snapshot")
            if len(row.get("action_ids", [])) != 1:
                raise RuntimeError("serial harness observed a non-singleton committed batch")
            active_snapshot = None
    if active_snapshot is not None:
        raise RuntimeError("event log ended with an uncommitted batch")
    return attempts


def _posterior_mean(successes: int, failures: int) -> float:
    return float((1 + successes) / (2 + successes + failures))


def _action_metrics(
    *,
    event_path: Path,
    bootstrap_energy_eV: float,
    action_timing: tuple[dict[str, object], ...],
) -> list[dict[str, Any]]:
    attempts = _event_attempts(event_path)
    if any(int(record["evaluator_calls"]) == 0 for record in action_timing):
        raise RuntimeError("zero-FE action timing cannot be unambiguously aligned")
    timing_index = 0
    cumulative_force_evaluations = 0
    best_landing_energy_eV = float(bootstrap_energy_eV)
    unique_minima = 1
    posterior: dict[int, list[int]] = {}
    metrics: list[dict[str, Any]] = []

    for ordinal, item in enumerate(attempts):
        attempt = item["attempt"]
        snapshot = item["snapshot"]
        counts = attempt["evaluation_counts"]
        if not isinstance(counts, dict):
            raise RuntimeError("attempt evaluation_counts must be a dictionary")
        force_evaluations = int(attempt["force_evaluations"])
        if sum(int(value) for value in counts.values()) != force_evaluations:
            raise RuntimeError("attempt force-evaluation ledger does not close")
        if int(counts["unattributed"]) != 0:
            raise RuntimeError("attempt contains unattributed evaluations")
        if attempt["cost_is_exact"] is not True:
            raise RuntimeError("attempt cost is not exact")

        starter_id = int(attempt["starter_id"])
        successes, failures = posterior.setdefault(starter_id, [0, 0])
        timing: dict[str, object] | None = None
        if force_evaluations > 0:
            if timing_index >= len(action_timing):
                raise RuntimeError("nonzero-cost action has no evaluator timing record")
            timing = action_timing[timing_index]
            timing_index += 1
            if int(timing["evaluator_calls"]) != force_evaluations:
                raise RuntimeError("action evaluator-call timing does not equal charged FE")

        landing_energy = attempt["landing_energy"]
        if landing_energy is not None:
            landing_energy = float(landing_energy)
            if not isfinite(landing_energy):
                raise RuntimeError("landing energy must be finite")
            best_landing_energy_eV = min(best_landing_energy_eV, landing_energy)
        if attempt["inserted_into_archive"]:
            unique_minima += 1
        cumulative_force_evaluations += force_evaluations

        metrics.append(
            {
                "schema_version": SCHEMA_VERSION,
                "ordinal": ordinal,
                "action_id": attempt["action_id"],
                "batch_id": attempt["batch_id"],
                "starter_id": starter_id,
                "selection_probability": attempt["selection_probability"],
                "policy_name": attempt["policy_name"],
                "policy_support_complete": snapshot["support_complete"],
                "eligible_starter_ids": snapshot["eligible_starter_ids"],
                "probabilities": snapshot["probabilities"],
                "posterior_before": {
                    "successes": successes,
                    "failures": failures,
                    "mean": _posterior_mean(successes, failures),
                },
                "status": attempt["status"],
                "failure_reason": attempt["failure_reason"],
                "posterior_observed": attempt["posterior_observed"],
                "discovered_against_snapshot": attempt["discovered_against_snapshot"],
                "inserted_into_archive": attempt["inserted_into_archive"],
                "landing_energy_eV": landing_energy,
                "force_evaluations": force_evaluations,
                "evaluation_counts": counts,
                "cumulative_action_force_evaluations": cumulative_force_evaluations,
                "best_landing_energy_eV": best_landing_energy_eV,
                "unique_minima": unique_minima,
                "evaluator_wall_time_s": None if timing is None else timing["evaluator_wall_time_s"],
                "evaluator_calls": None if timing is None else timing["evaluator_calls"],
            }
        )
        if attempt["posterior_observed"]:
            if attempt["discovered_against_snapshot"]:
                posterior[starter_id][0] += 1
            else:
                posterior[starter_id][1] += 1

    if timing_index != len(action_timing):
        raise RuntimeError("evaluator timing records are not matched to actions")
    return metrics


def run_campaign(
    *,
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    exploration_config: PosteriorExplorationConfig,
    run_posterior: Callable[..., object] = run_posterior_ssw,
) -> dict[str, Any]:
    """Execute one already-preflighted campaign and publish closed facts only."""
    if type(ssw_config) is not SSWConfig:
        raise TypeError("run_campaign requires exactly an SSWConfig")
    if exploration_config.batch_size != 1 or exploration_config.max_workers != 1:
        raise ValueError("run_campaign requires serial batch_size=max_workers=1")
    if not isinstance(calculator_factory, InstrumentedCalculatorFactory):
        raise TypeError("run_campaign requires an InstrumentedCalculatorFactory")
    if not callable(run_posterior):
        raise TypeError("run_posterior must be callable")
    started = perf_counter()
    result = run_posterior(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
    )
    campaign_wall_time_s = perf_counter() - started
    event_path = exploration_config.run_directory / "events.jsonl"
    if not event_path.is_file():
        raise RuntimeError("posterior runner did not write its event log")
    purpose_counts = result.purpose_counts.as_dict()
    if purpose_counts["unattributed"] != 0:
        raise RuntimeError("campaign contains unattributed evaluations")
    if result.total_evaluations + result.unused_force_budget != result.total_force_budget:
        raise RuntimeError("campaign budget does not close")
    if not result.benchmark_eligible:
        raise RuntimeError(
            f"campaign is not benchmark eligible: {result.benchmark_ineligibility_reasons!r}"
        )

    timing_records = calculator_factory.action_records()
    factory_snapshot = calculator_factory.snapshot()
    bootstrap_timing = calculator_factory.bootstrap_records()
    if len(bootstrap_timing) != 1:
        raise RuntimeError("bootstrap timing must contain exactly one calculator factory call")
    if int(bootstrap_timing[0]["evaluator_calls"]) != result.bootstrap_evaluations:
        raise RuntimeError("bootstrap evaluator-call timing does not equal charged FE")

    metrics = _action_metrics(
        event_path=event_path,
        bootstrap_energy_eV=float(result.archive.entries[0].energy),
        action_timing=timing_records,
    )
    expected_attempts = result.completed_attempts + result.failed_attempts
    if len(metrics) != expected_attempts:
        raise RuntimeError("action metric count does not equal terminal attempts")
    if sum(metric["force_evaluations"] for metric in metrics) != result.action_evaluations:
        raise RuntimeError("action metric FE does not equal campaign action FE")
    action_metrics_path = exploration_config.run_directory / "action_metrics.jsonl"
    _write_jsonl_exclusive(action_metrics_path, metrics)

    final_posterior = [
        {
            "starter_id": entry.entry_id,
            "energy_eV": float(entry.energy),
            "successes": result.posterior.counts(entry.entry_id)[0],
            "failures": result.posterior.counts(entry.entry_id)[1],
            "mean": result.posterior.mean(entry.entry_id),
        }
        for entry in result.archive.entries
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_name": result.policy_name,
        "batch_size": exploration_config.batch_size,
        "max_workers": exploration_config.max_workers,
        "master_seed": exploration_config.master_seed,
        "action_force_budget": exploration_config.action_force_budget,
        "total_force_budget": result.total_force_budget,
        "bootstrap_evaluations": result.bootstrap_evaluations,
        "action_evaluations": result.action_evaluations,
        "total_evaluations": result.total_evaluations,
        "unused_force_budget": result.unused_force_budget,
        "purpose_counts": purpose_counts,
        "completed_batches": result.completed_batches,
        "completed_attempts": result.completed_attempts,
        "failed_attempts": result.failed_attempts,
        "posterior_observed_attempts": result.posterior_observed_attempts,
        "benchmark_eligible": result.benchmark_eligible,
        "benchmark_ineligibility_reasons": list(result.benchmark_ineligibility_reasons),
        "event_log_replayed": True,
        "archive_entries": len(result.archive.entries),
        "best_archive_energy_eV": min(float(entry.energy) for entry in result.archive.entries),
        "bootstrap_energy_eV": float(result.archive.entries[0].energy),
        "duplicate_rate": result.archive.duplicate_rate(),
        "final_posterior": final_posterior,
        "campaign_wall_time_s": campaign_wall_time_s,
        "bootstrap_evaluator_wall_time_s": bootstrap_timing[0]["evaluator_wall_time_s"],
        "action_metrics_path": str(action_metrics_path),
        "event_log_path": str(event_path),
        "calculator_factory": factory_snapshot,
    }
    _write_json_exclusive(exploration_config.run_directory / "campaign_summary.json", summary)
    return summary


def _preflight_output_root(output_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output root already exists: {output_root}")
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"output root parent does not exist: {output_root.parent}")


def preflight(
    *,
    expected_git_commit: str,
    systems: Sequence[str],
    master_seeds: Sequence[int],
    action_force_budget: int,
    total_force_budget: int,
) -> dict[str, Any]:
    """Validate immutable execution identity without creating outputs or a calculator."""
    selected_systems = _validated_systems(systems)
    selected_seeds = _validated_seeds(master_seeds)
    _positive_int("action_force_budget", action_force_budget)
    _positive_int("total_force_budget", total_force_budget)
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    production = _load_production_runner()
    projections: dict[str, object] = {}
    inputs: dict[str, object] = {}
    for system in selected_systems:
        config, projection = build_ssw_config(system, RUN_ROOT / ".preflight" / system)
        if type(config) is not SSWConfig or projection["softening_enabled"] is not False:
            raise RuntimeError("projected configuration is not unsoftened SSW")
        projections[system] = projection
        input_path = Path(production.INPUT_PATHS[system])
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        inputs[system] = {"path": str(input_path), "sha256": _sha256(input_path)}
    model_path = Path(production.MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    cuda = _cuda_info()
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_commit": actual_commit,
        "pamssw_bundle_sha256": _pamssw_bundle_sha256(PAMSSW_SOURCE_ROOT),
        "systems": list(selected_systems),
        "master_seeds": list(selected_seeds),
        "policies": list(POLICIES),
        "batch_size": SERIAL_BATCH_SIZE,
        "max_workers": SERIAL_MAX_WORKERS,
        "action_force_budget": action_force_budget,
        "total_force_budget": total_force_budget,
        "inputs": inputs,
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "calculator": dict(production.CALCULATOR_CONFIG),
        "runtime_versions": _runtime_versions(),
        "cuda": cuda,
        "calculator_reuse": {
            "bootstrap": "one uncached caller-thread calculator",
            "actions": "one worker-thread-local calculator reused serially",
            "cross_thread_calculator_sharing": False,
            "accounting": "each bootstrap/action is wrapped by its own EvalCounter",
        },
        "projections": projections,
    }


def _mace_calculator_factory() -> object:
    production = _load_production_runner()
    from mace.calculators import MACECalculator

    calculator = MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )
    return ASECalculator(calculator)


def run_ablation(
    *,
    output_root: Path,
    expected_git_commit: str,
    systems: Sequence[str],
    master_seeds: Sequence[int],
    action_force_budget: int,
    total_force_budget: int,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run caller-parameterized smoke or decision matrices; no hard-coded matrix."""
    output_root = Path(output_root)
    _preflight_output_root(output_root)
    manifest = preflight(
        expected_git_commit=expected_git_commit,
        systems=systems,
        master_seeds=master_seeds,
        action_force_budget=action_force_budget,
        total_force_budget=total_force_budget,
    )
    if preflight_only:
        return manifest

    production = _load_production_runner()
    output_root.mkdir()
    _write_json_exclusive(output_root / "manifest.json", manifest)
    campaigns: list[dict[str, Any]] = []
    for system in manifest["systems"]:
        for master_seed in manifest["master_seeds"]:
            for policy_name in POLICIES:
                case_directory = output_root / system / f"seed-{master_seed:08d}" / policy_name
                case_directory.parent.mkdir(parents=True, exist_ok=True)
                config, projection = build_ssw_config(system, case_directory)
                exploration = build_exploration_config(
                    policy_name=policy_name,
                    run_directory=case_directory,
                    master_seed=master_seed,
                    action_force_budget=action_force_budget,
                    total_force_budget=total_force_budget,
                )
                owner = ThreadOwnedCalculatorFactory(_mace_calculator_factory)
                factory = InstrumentedCalculatorFactory(owner)
                summary = run_campaign(
                    initial_state=production.load_state(system),
                    calculator_factory=factory,
                    ssw_config=config,
                    exploration_config=exploration,
                )
                campaigns.append(
                    {
                        "system": system,
                        "master_seed": master_seed,
                        "policy_name": policy_name,
                        "projection": projection,
                        "campaign_summary_path": str(case_directory / "campaign_summary.json"),
                    }
                )
    index = {"schema_version": SCHEMA_VERSION, "manifest": manifest, "campaigns": campaigns}
    _write_json_exclusive(output_root / "index.json", index)
    return index


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    parser.add_argument("--master-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--action-force-budget", type=int, required=True)
    parser.add_argument("--total-force-budget", type=int, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_ablation(
        output_root=args.output,
        expected_git_commit=args.expected_git_commit,
        systems=tuple(args.systems),
        master_seeds=tuple(args.master_seeds),
        action_force_budget=args.action_force_budget,
        total_force_budget=args.total_force_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
