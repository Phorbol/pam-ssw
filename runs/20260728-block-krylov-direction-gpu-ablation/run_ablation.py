#!/usr/bin/env python3
"""Run one preregistered fixed-budget block-Krylov LS-SSW case.

This is a deliberately thin adaptation of the checked-in direction-selection
Ritz runner.  It keeps the frozen production runner for structure/model/CUDA
preflight and only changes the Stage-2 direction-selection allocation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
FROZEN_RUNNER_RELATIVE = Path("runs/20260728-safe-lbfgs-200-production/run_production.py")
SYSTEMS = ("c60", "pdo")
# Insertion order is the preregistered Stage-2 arm order.
ARMS: dict[str, dict[str, object]] = {
    "discrete": {"direction_selection_mode": "discrete"},
    "variational_breadth": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 6,
        "block_krylov_depth": 1,
    },
    "balanced_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 2,
        "block_krylov_depth": 3,
    },
    "deep_refinement": {
        "direction_selection_mode": "block_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 6,
    },
}
SEEDS = (42, 43, 44)
TOTAL_FORCE_BUDGET = 6000
MAX_TRIALS = 200
EXPECTED_ORACLE_CANDIDATES = {"c60": 12, "pdo": 8}
OUTPUT_PATH_FIELDS = {
    "accepted_structures_dir",
    "accepted_structures_log",
    "direction_diagnostics_path",
}
ALLOWED_DIRECTION_DIFF = {
    "direction_selection_mode",
    "block_krylov_blocks",
    "block_krylov_depth",
    "rng_seed",
    "direction_diagnostics_path",
}
BLOCK_TRACE_KEYS = {
    "selected_kind",
    "candidate_count",
    "krylov_blocks",
    "krylov_hvp_requested",
    "krylov_hvp_consumed",
    "krylov_hvp_count",
    "oracle_selection_force_evaluations_delta",
}
REQUIRED_OUTPUTS = (
    "best_minimum.xyz",
    "energy_trace.json",
    "walk_records.json",
    "optimizer_diagnostics.json",
    "direction_trace.jsonl",
    "summary.json",
)


@dataclass(frozen=True)
class TargetRuntime:
    code_root: Path
    base_runner: Any
    calculator_wrapper: Callable[[Any], Any]
    walker_class: type
    write_state: Callable[[Path, Any], None]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _current_commit(code_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code_root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _tracked_worktree_clean(code_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _assert_module_under_root(module: Any, root: Path, label: str) -> None:
    source_path = getattr(module, "__file__", None)
    if source_path is None:
        raise RuntimeError(f"{label} has no source path")
    try:
        Path(source_path).resolve().relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} was not imported from {root}: {source_path}") from error


def _load_target_runtime(code_root: Path) -> TargetRuntime:
    root = Path(code_root).resolve()
    runner_path = root / FROZEN_RUNNER_RELATIVE
    if not runner_path.is_file():
        raise FileNotFoundError(runner_path)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    importlib.invalidate_caches()
    pamssw_module = importlib.import_module("pamssw")
    calculators_module = importlib.import_module("pamssw.calculators")
    walker_module = importlib.import_module("pamssw.walker")
    for module, label in (
        (pamssw_module, "pamssw"),
        (calculators_module, "pamssw.calculators"),
        (walker_module, "pamssw.walker"),
    ):
        _assert_module_under_root(module, root, label)
    spec = importlib.util.spec_from_file_location("_block_krylov_frozen_production", runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen production runner: {runner_path}")
    base_runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base_runner
    spec.loader.exec_module(base_runner)
    _assert_module_under_root(base_runner, root, "frozen production runner")
    return TargetRuntime(
        code_root=root,
        base_runner=base_runner,
        calculator_wrapper=calculators_module.ASECalculator,
        walker_class=walker_module.SurfaceWalker,
        write_state=pamssw_module.write_state,
    )


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeError(f"{label} must be an exact integer >= {minimum}")
    return value


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RuntimeError(f"{label} must be a finite builtin number")
    return float(value)


def _json_config(config: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))


def _config_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, list[Any]]:
    if left.keys() != right.keys():
        raise RuntimeError("configuration field set drifted")
    return {key: [left[key], right[key]] for key in sorted(left) if left[key] != right[key]}


def _normalise_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in OUTPUT_PATH_FIELDS}


def _arm_expected(arm: str) -> Mapping[str, object]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return ARMS[arm]


def _validate_arm_protocol(
    *, system: str, configurations: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, list[Any]]], dict[str, Any]]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    if tuple(configurations) != tuple(ARMS):
        raise RuntimeError("arm configurations must use the preregistered stable order")
    discrete = configurations["discrete"]
    if discrete["oracle_candidates"] != EXPECTED_ORACLE_CANDIDATES[system]:
        raise RuntimeError("frozen production oracle_candidates drifted")
    diffs: dict[str, dict[str, list[Any]]] = {}
    expected_common = {
        "max_trials": MAX_TRIALS,
        "max_force_evals": TOTAL_FORCE_BUDGET,
        "proposal_optimizer": "safe-lbfgs-total",
        "use_archive_acquisition": True,
        "seed_selection_mode": "archive_ucb",
        "direction_synthesis_mode": "none",
        "direction_type_ucb_enabled": False,
        "direction_archive_enabled": False,
        "direction_probe_enabled": False,
        "plateau_evolution_enabled": False,
        "archive_escape_momentum_enabled": False,
        "direction_diagnostics_enabled": True,
    }
    for arm, config in configurations.items():
        for key, expected in expected_common.items():
            if config.get(key) != expected:
                raise RuntimeError(f"{arm} frozen protocol drifted: {key}={config.get(key)!r}")
        expected_arm = _arm_expected(arm)
        for key, expected in expected_arm.items():
            if config.get(key) != expected:
                raise RuntimeError(f"{arm} did not receive preregistered {key}={expected!r}")
        diff = _config_diff(_normalise_config(discrete), _normalise_config(config))
        unexpected = set(diff) - (ALLOWED_DIRECTION_DIFF - {"direction_diagnostics_path", "rng_seed"})
        if unexpected:
            raise RuntimeError(f"{arm} changed frozen settings: {sorted(unexpected)}")
        diffs[arm] = diff
    return diffs, {
        "allowed_direction_diff": sorted(ALLOWED_DIRECTION_DIFF),
        "same_frozen_production_config": True,
        "block_krylov_hvp_contract": {
            "selected_kind": "block_ritz",
            "candidate_count": 0,
            "maximum_requested_or_consumed_hvps": 12,
            "oracle_force_evaluations_per_hvp": 2,
            "strict_diagnostic_keys": sorted(BLOCK_TRACE_KEYS),
            "runtime_cost_source": "purpose_counts.direction_oracle",
        },
    }


def config_projection(
    *, arm: str, system: str, case_dir: Path, total_force_budget: int, seed: int, base_runner: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]], dict[str, dict[str, list[Any]]], dict[str, Any]]:
    _arm_expected(arm)
    if total_force_budget != TOTAL_FORCE_BUDGET:
        raise ValueError(f"Stage-2 execution requires total force budget {TOTAL_FORCE_BUDGET}")
    _exact_int(seed, "seed")
    source_config = base_runner.build_config(system, Path(case_dir))
    source = _json_config(source_config)
    configurations = {
        name: _json_config(
            replace(
                source_config,
                max_force_evals=total_force_budget,
                rng_seed=seed,
                **overrides,
            )
        )
        for name, overrides in ARMS.items()
    }
    arm_diffs, protocol = _validate_arm_protocol(system=system, configurations=configurations)
    effective = configurations[arm]
    source_diff = _config_diff(source, effective)
    unexpected_source = set(source_diff) - (ALLOWED_DIRECTION_DIFF | {"max_force_evals"})
    if unexpected_source:
        raise RuntimeError(f"unexpected source-to-effective changes: {sorted(unexpected_source)}")
    return source, effective, source_diff, arm_diffs, protocol


def preflight(
    *,
    arm: str,
    system: str,
    seed: int,
    total_force_budget: int,
    code_root: Path,
    output_dir: Path | None,
    expected_git_commit: str,
    target_loader: Callable[[Path], TargetRuntime] = _load_target_runtime,
    git_head: Callable[[Path], str] = _current_commit,
    tracked_clean: Callable[[Path], bool] = _tracked_worktree_clean,
) -> dict[str, Any]:
    root = Path(code_root).resolve()
    actual_commit = git_head(root)
    if expected_git_commit != actual_commit:
        raise RuntimeError(f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}")
    if not tracked_clean(root):
        raise RuntimeError(f"target worktree is not clean: {root}")
    target = target_loader(root)
    source, effective, source_diff, arm_diffs, protocol = config_projection(
        arm=arm,
        system=system,
        case_dir=Path(output_dir) if output_dir is not None else root / ".block-krylov-preflight",
        total_force_budget=total_force_budget,
        seed=seed,
        base_runner=target.base_runner,
    )
    base_preflight = dict(target.base_runner.preflight(system=system, expected_git_commit=expected_git_commit))
    if base_preflight.get("execution_commit") != actual_commit:
        raise RuntimeError("frozen production preflight commit mismatch")
    if not isinstance(base_preflight.get("cuda"), Mapping) or base_preflight["cuda"].get("available") is not True:
        raise RuntimeError("frozen production preflight did not prove CUDA availability")
    for key in ("input_sha256", "model_sha256", "input_path", "model_path"):
        if not isinstance(base_preflight.get(key), str) or not base_preflight[key]:
            raise RuntimeError(f"frozen production preflight omitted {key}")
    return {
        "schema_version": 1,
        "arm": arm,
        "system": system,
        "seed": seed,
        "total_force_budget": total_force_budget,
        "target": {
            "code_root": str(root),
            "execution_commit": actual_commit,
            "tracked_worktree_clean": True,
            "frozen_runner_path": str(root / FROZEN_RUNNER_RELATIVE),
            "frozen_runner_sha256": _sha256(root / FROZEN_RUNNER_RELATIVE),
            "ablation_runner_path": str(Path(__file__).resolve()),
            "ablation_runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "base_preflight": base_preflight,
        "source_config": source,
        "effective_config": effective,
        "source_to_effective_config_diff": source_diff,
        "paired_arm_config_diffs": arm_diffs,
        "block_krylov_hvp_contract": protocol["block_krylov_hvp_contract"],
        "protocol": protocol,
    }


def _direction_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"direction diagnostics file was not written: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid direction diagnostics JSON at line {line_number}") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"direction diagnostics row {line_number} is not a mapping")
        rows.append(row)
    return rows


def validate_direction_trace(*, arm: str, direction_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check the exact per-selection contract without relying on MACE/GPU."""

    _arm_expected(arm)
    if not direction_rows:
        raise RuntimeError("direction diagnostics contain no selections")
    selected_kind_counts: dict[str, int] = {}
    direction_force_evaluations = 0
    expected = ARMS[arm]
    for index, row in enumerate(direction_rows, start=1):
        kind = row.get("selected_kind")
        if not isinstance(kind, str):
            raise RuntimeError(f"direction row {index} has no selected_kind")
        selected_kind_counts[kind] = selected_kind_counts.get(kind, 0) + 1
        if arm == "discrete":
            present = BLOCK_TRACE_KEYS - {"selected_kind", "candidate_count"}
            forbidden = sorted(key for key in present if key in row)
            if forbidden:
                raise RuntimeError(f"discrete direction row {index} contains block keys: {forbidden}")
            continue
        missing = sorted(key for key in BLOCK_TRACE_KEYS if key not in row)
        if missing:
            raise RuntimeError(f"block direction row {index} is missing strict keys: {missing}")
        if kind != "block_ritz":
            raise RuntimeError(f"block direction row {index} selected_kind is not block_ritz")
        if _exact_int(row["candidate_count"], f"direction row {index} candidate_count") != 0:
            raise RuntimeError(f"block direction row {index} candidate_count is not zero")
        blocks = _exact_int(row["krylov_blocks"], f"direction row {index} krylov_blocks", minimum=1)
        if blocks != expected["block_krylov_blocks"]:
            raise RuntimeError(f"block direction row {index} has unexpected krylov_blocks")
        requested = _exact_int(row["krylov_hvp_requested"], f"direction row {index} krylov_hvp_requested")
        consumed = _exact_int(row["krylov_hvp_consumed"], f"direction row {index} krylov_hvp_consumed")
        count = _exact_int(row["krylov_hvp_count"], f"direction row {index} krylov_hvp_count")
        expected_requested = int(expected["block_krylov_blocks"]) * int(expected["block_krylov_depth"])
        if requested != expected_requested or requested > 12 or consumed > 12 or consumed > requested or count != consumed:
            raise RuntimeError(f"block direction row {index} violates requested/consumed HVP contract")
        oracle_delta = _exact_int(
            row["oracle_selection_force_evaluations_delta"],
            f"direction row {index} oracle selection FE delta",
        )
        if oracle_delta != 2 * consumed:
            raise RuntimeError("oracle_selection_force_evaluations_delta must equal 2 * krylov_hvp_consumed")
        direction_force_evaluations += oracle_delta
    return {
        "selection_count": len(direction_rows),
        "selected_kind_counts": selected_kind_counts,
        "direction_oracle_force_evaluations": direction_force_evaluations,
        "strict_block_trace_validated": arm != "discrete",
    }


def _energy_trace(result: Any, total_force_evaluations: int) -> list[dict[str, Any]]:
    """Persist the known FE endpoints; no unobserved per-trial FE is invented."""

    initial = _finite(result.archive.entries[0].energy, "initial energy")
    best = _finite(result.best_energy, "best energy")
    return [
        {"trial": 0, "energy_eV": initial, "best_energy_eV": initial, "cumulative_total_force_evaluations": 0},
        {
            "trial": _exact_int(result.stats["n_trials"], "n_trials"),
            "energy_eV": best,
            "best_energy_eV": best,
            "cumulative_total_force_evaluations": total_force_evaluations,
        },
    ]


def _walk_records(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "trial": index,
            "seed_entry_id": _exact_int(record.seed_entry_id, "seed_entry_id"),
            "discovered_entry_id": _exact_int(record.discovered_entry_id, "discovered_entry_id"),
            "energy_eV": _finite(record.energy, "walk record energy"),
            "accepted_new_basin": bool(record.accepted_new_basin),
        }
        for index, record in enumerate(result.walk_history, start=1)
    ]


def _validate_run_closure(*, result: Any, walker: Any, arm: str, direction_path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    total = _exact_int(result.stats["force_evaluations"], "force_evaluations", minimum=1)
    if total != TOTAL_FORCE_BUDGET:
        raise RuntimeError(f"fixed total force budget was not reached exactly: {total} != {TOTAL_FORCE_BUDGET}")
    if _exact_int(result.stats.get("budget_exhausted", 0), "budget_exhausted") != 1:
        raise RuntimeError("execution did not terminate by budget exhaustion")
    counts = walker.calculator.snapshot()
    if _exact_int(counts.total, "counter total") != total:
        raise RuntimeError("force-evaluation total does not close")
    purposes = {str(key): _exact_int(value, f"purpose count {key!r}") for key, value in counts.as_dict().items()}
    if sum(purposes.values()) != total or purposes.get("unattributed", 0) != 0:
        raise RuntimeError("purpose-resolved force ledger does not close")
    if "direction_oracle" not in purposes:
        raise RuntimeError("purpose ledger omits direction_oracle")
    rows = _direction_rows(direction_path)
    audit = validate_direction_trace(arm=arm, direction_rows=rows)
    choices = _exact_int(result.stats.get("direction_choices", 0), "direction_choices")
    if choices != audit["selection_count"]:
        raise RuntimeError("direction diagnostics do not close against direction choices")
    if arm != "discrete" and purposes["direction_oracle"] != audit["direction_oracle_force_evaluations"]:
        raise RuntimeError("direction-oracle purpose ledger does not close against block trace")
    audit["direction_rows"] = rows
    return purposes, audit


def run(
    *,
    arm: str,
    system: str,
    seed: int,
    total_force_budget: int,
    code_root: Path,
    output_dir: Path,
    expected_git_commit: str,
    preflight_only: bool = False,
    target_loader: Callable[[Path], TargetRuntime] = _load_target_runtime,
    preflight_fn: Callable[..., dict[str, Any]] = preflight,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    checked = preflight_fn(
        arm=arm, system=system, seed=seed, total_force_budget=total_force_budget,
        code_root=code_root, output_dir=output_dir, expected_git_commit=expected_git_commit,
    )
    if preflight_only:
        return checked
    target = target_loader(Path(code_root).resolve())
    source, effective, source_diff, arm_diffs, protocol = config_projection(
        arm=arm, system=system, case_dir=output_dir, total_force_budget=total_force_budget,
        seed=seed, base_runner=target.base_runner,
    )
    for key, value in {
        "source_config": source,
        "effective_config": effective,
        "source_to_effective_config_diff": source_diff,
        "paired_arm_config_diffs": arm_diffs,
    }.items():
        if checked[key] != value:
            raise RuntimeError(f"{key} changed after preflight")
    config = replace(target.base_runner.build_config(system, output_dir), max_force_evals=total_force_budget, rng_seed=seed, **ARMS[arm])
    if _json_config(config) != effective:
        raise RuntimeError("walker config does not equal preflight effective config")
    output_dir.mkdir(parents=True)
    walker = target.walker_class(
        calculator=target.calculator_wrapper(target.base_runner._calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(target.base_runner.load_state(system))
    wall_time_s = perf_counter() - started
    purposes, direction_audit = _validate_run_closure(
        result=result, walker=walker, arm=arm, direction_path=Path(config.direction_diagnostics_path)
    )
    force_evaluations = _exact_int(result.stats["force_evaluations"], "force_evaluations")
    energy_trace = _energy_trace(result, force_evaluations)
    initial = _finite(energy_trace[0]["energy_eV"], "initial energy")
    best = _finite(result.best_energy, "best energy")
    summary = {
        **checked,
        "effective_config": effective,
        "force_evaluations": force_evaluations,
        "purpose_counts": purposes,
        "stats": result.stats,
        "archive_size": len(result.archive.entries),
        "unique_minima": _exact_int(result.stats["n_minima"], "n_minima", minimum=1),
        "duplicate_fraction": _finite(result.stats.get("duplicate_rate", 0.0), "duplicate_rate"),
        "terminal_failure_count": result.stats.get("failure_count"),
        "terminal_failure_count_reason": "unsupported_by_current_result_stats" if "failure_count" not in result.stats else None,
        "direction_selection_audit": direction_audit,
        "initial_energy_eV": initial,
        "best_energy_eV": best,
        "energy_drop_eV": initial - best,
        "optimizer_diagnostics": walker.relaxation_diagnostics(),
        "timing": {"total_wall_time_s": wall_time_s},
        "termination": {"reason": "force_budget_exhausted", "budget_exhausted": True},
        "walk_records": _walk_records(result),
    }
    target.write_state(output_dir / "best_minimum.xyz", result.best_state)
    _write_json(output_dir / "energy_trace.json", energy_trace)
    _write_json(output_dir / "walk_records.json", summary["walk_records"])
    _write_json(output_dir / "optimizer_diagnostics.json", summary["optimizer_diagnostics"])
    _write_json(output_dir / "summary.json", summary)
    for filename in REQUIRED_OUTPUTS:
        if not (output_dir / filename).is_file():
            raise RuntimeError(f"required output was not written: {filename}")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--arm", required=True, choices=tuple(ARMS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--force-budget", type=int, default=TOTAL_FORCE_BUDGET)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.force_budget != TOTAL_FORCE_BUDGET:
        raise ValueError(f"production execution requires --force-budget {TOTAL_FORCE_BUDGET}")
    summary = run(
        arm=args.arm, system=args.system, seed=args.seed, total_force_budget=args.force_budget,
        code_root=REPO_ROOT, output_dir=args.output_dir, expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
