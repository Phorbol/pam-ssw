#!/usr/bin/env python3
"""Run one paired, fixed-total-FE native direction-candidate-budget ablation arm.

The two arms are *source identities*, not configuration toggles: ``precap``
loads the frozen kernel before native portfolio capping, whereas ``hardcap``
loads the commit which caps the generator itself.  Each CLI process therefore
loads exactly one explicit, clean target checkout and refuses to switch roots.
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


RUNNER_PATH = Path(__file__).resolve()
FROZEN_RUNNER_RELATIVE = Path(
    "runs/20260728-safe-lbfgs-200-production/run_production.py"
)
PRODUCTION_TOTAL_FORCE_BUDGET = 6000
MAX_TRIALS = 200

ARM_COMMITS = {
    "precap": "32980ad9154ec6481b310477dd7b85597cefd49a",
    "hardcap": "3b0dd65711d2155460e0ab6b1cd0c95602a8c422",
}
SYSTEMS = ("c60", "pdo")
EXPECTED_OUTPUT_FILES = (
    "best_minimum.xyz",
    "energy_trace.json",
    "walk_records.json",
    "optimizer_diagnostics.json",
    "direction_trace.jsonl",
    "summary.json",
)
REQUIRED_NON_SUMMARY_OUTPUT_FILES = tuple(
    filename for filename in EXPECTED_OUTPUT_FILES if filename != "summary.json"
)

_loaded_target_root: Path | None = None
_loaded_target_runtime: "TargetRuntime | None" = None


@dataclass(frozen=True)
class TargetRuntime:
    """One canonical import namespace for a single target checkout."""

    code_root: Path
    base_runner: Any
    frozen_runner_path: Path
    calculator_wrapper: Callable[[Any], Any]
    walker_class: type
    write_state: Callable[[Path, Any], None]
    pamssw_module: Any | None = None
    walker_module: Any | None = None


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _current_commit(code_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
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


def _assert_module_under_root(module: Any, code_root: Path, label: str) -> None:
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        raise RuntimeError(f"target {label} has no source path")
    try:
        Path(module_path).resolve().relative_to(code_root)
    except ValueError as error:
        raise RuntimeError(
            f"target {label} was not loaded from explicit code root {code_root}: {module_path}"
        ) from error


def _load_target_runtime(code_root: Path) -> TargetRuntime:
    """Load target modules once under their canonical names in this process."""

    global _loaded_target_root, _loaded_target_runtime
    root = Path(code_root).resolve()
    if _loaded_target_root is not None:
        if root != _loaded_target_root:
            raise RuntimeError(
                "one process cannot load two code roots; start a fresh process per arm"
            )
        assert _loaded_target_runtime is not None
        return _loaded_target_runtime

    package_path = root / "pamssw" / "__init__.py"
    frozen_runner_path = root / FROZEN_RUNNER_RELATIVE
    if not package_path.is_file():
        raise FileNotFoundError(package_path)
    if not frozen_runner_path.is_file():
        raise FileNotFoundError(frozen_runner_path)

    existing = sys.modules.get("pamssw")
    if existing is not None:
        _assert_module_under_root(existing, root, "pamssw")
    else:
        sys.path.insert(0, str(root))
        importlib.invalidate_caches()
        importlib.import_module("pamssw")

    pamssw_module = importlib.import_module("pamssw")
    calculators_module = importlib.import_module("pamssw.calculators")
    walker_module = importlib.import_module("pamssw.walker")
    _assert_module_under_root(pamssw_module, root, "pamssw")
    _assert_module_under_root(calculators_module, root, "pamssw.calculators")
    _assert_module_under_root(walker_module, root, "pamssw.walker")

    module_name = "_direction_candidate_budget_frozen_production"
    spec = importlib.util.spec_from_file_location(module_name, frozen_runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen production runner: {frozen_runner_path}")
    base_runner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = base_runner
    spec.loader.exec_module(base_runner)
    _assert_module_under_root(base_runner, root, "frozen production runner")

    runtime = TargetRuntime(
        code_root=root,
        base_runner=base_runner,
        frozen_runner_path=frozen_runner_path,
        calculator_wrapper=calculators_module.ASECalculator,
        walker_class=walker_module.SurfaceWalker,
        write_state=pamssw_module.write_state,
        pamssw_module=pamssw_module,
        walker_module=walker_module,
    )
    _loaded_target_root = root
    _loaded_target_runtime = runtime
    return runtime


def _mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value[key] for key in value}


def _nonnegative_exact_int(value: Any, label: str) -> int:
    """Accept only built-in non-negative integers, never coercions."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative exact integer")
    return value


def _positive_exact_int(value: Any, label: str) -> int:
    result = _nonnegative_exact_int(value, label)
    if result == 0:
        raise ValueError(f"{label} must be positive")
    return result


def _finite_energy(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite energy")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite energy")
    return result


def _json_mapping(config: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))


def _config_diff(source: Mapping[str, Any], effective: Mapping[str, Any]) -> dict[str, list[Any]]:
    if source.keys() != effective.keys():
        raise RuntimeError("source and effective config fields differ")
    return {
        key: [source[key], effective[key]]
        for key in sorted(source)
        if source[key] != effective[key]
    }


def _validate_frozen_direction_protocol(config: Mapping[str, Any]) -> None:
    expected = {
        "max_trials": MAX_TRIALS,
        "rng_seed": 42,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "local_softening_mode": "active_neighbors",
        "direction_probe_enabled": False,
        "direction_synthesis_mode": "none",
        "direction_selection_mode": "discrete",
        "direction_diagnostics_enabled": True,
        "direction_curvature_source": "inner",
        "choice_aligned_softening_enabled": False,
        "plateau_evolution_enabled": False,
        "archive_escape_momentum_enabled": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"frozen production direction protocol drifted: {key}={config.get(key)!r}"
            )


def _quench_overrides(system: str) -> dict[str, Any]:
    if system == "c60":
        return {
            "quench_optimizer": "ase-lbfgs",
            "quench_fallback_optimizer": "ase-fire",
            "quench_fmax": 0.01,
            "quench_maxiter": 400,
        }
    if system == "pdo":
        return {
            "quench_optimizer": "scipy-lbfgsb",
            "quench_fallback_optimizer": None,
            "quench_fmax": 0.03,
            "quench_maxiter": 400,
        }
    raise ValueError(f"unknown system: {system}")


def config_projection(
    *,
    system: str,
    case_dir: Path,
    total_force_budget: int,
    seed: int,
    base_runner: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]], dict[str, Any]]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    total_force_budget = _positive_exact_int(total_force_budget, "total_force_budget")
    seed = _nonnegative_exact_int(seed, "seed")

    source_config = base_runner.build_config(system, Path(case_dir))
    source = _json_mapping(source_config)
    _validate_frozen_direction_protocol(source)
    overrides = {
        "max_trials": MAX_TRIALS,
        "max_force_evals": total_force_budget,
        "rng_seed": seed,
        **_quench_overrides(system),
    }
    effective_config = replace(source_config, **overrides)
    effective = _json_mapping(effective_config)
    diff = _config_diff(source, effective)
    allowed_changes = {
        "max_force_evals",
        "rng_seed",
        "quench_optimizer",
        "quench_fallback_optimizer",
        "quench_fmax",
        "quench_maxiter",
    }
    unexpected = set(diff) - allowed_changes
    if unexpected:
        raise RuntimeError(
            f"direction ablation has unexpected config changes: {sorted(unexpected)}"
        )
    return source, effective, diff, overrides


def _native_generator_contract(target: TargetRuntime, arm: str) -> dict[str, Any]:
    """Exercise the native non-first generator path without a PES evaluation."""

    if target.pamssw_module is None or target.walker_module is None:
        raise RuntimeError("target runtime does not expose native generator modules")
    import numpy as np

    state = target.pamssw_module.State(
        numbers=np.asarray([1, 1, 1, 1], dtype=int),
        positions=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 1.0, 0.0]],
            dtype=float,
        ),
    )
    candidate_budget = 2
    generator = target.walker_module.CandidateDirectionGenerator(
        np.random.default_rng(2),
        n_random=candidate_budget,
        bond_pairs=[(0, 1)],
        n_bond_pairs=1,
        bond_distance_threshold=2.0,
    )
    candidates = generator.generate(
        state,
        previous_direction=np.ones(state.positions.size, dtype=float),
    )
    candidate_count = len(candidates)
    if arm == "precap":
        if candidate_count <= candidate_budget:
            raise RuntimeError(
                "precap target no longer permits a non-first native portfolio beyond oracle_candidates"
            )
        expectation = "nonfirst_native_candidates_exceed_budget"
    elif arm == "hardcap":
        if candidate_count > candidate_budget:
            raise RuntimeError(
                "hardcap target does not cap non-first native candidates at oracle_candidates"
            )
        expectation = "native_candidates_do_not_exceed_budget"
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "candidate_budget": candidate_budget,
        "nonfirst_candidate_count": candidate_count,
        "expectation": expectation,
    }


def _validate_arm_identity(
    *, arm: str, expected_code_commit: str, actual_code_commit: str
) -> None:
    if arm not in ARM_COMMITS:
        raise ValueError(f"unknown arm: {arm}")
    expected_arm_commit = ARM_COMMITS[arm]
    if expected_code_commit != expected_arm_commit:
        raise RuntimeError(
            f"arm {arm} must use exact commit {expected_arm_commit}, got {expected_code_commit}"
        )
    if actual_code_commit != expected_arm_commit:
        raise RuntimeError(
            f"target commit mismatch for {arm}: expected {expected_arm_commit}, got {actual_code_commit}"
        )


def preflight(
    *,
    arm: str,
    system: str,
    seed: int,
    total_force_budget: int,
    code_root: Path,
    expected_code_commit: str,
    output_dir: Path | None = None,
    target_loader: Callable[[Path], TargetRuntime] = _load_target_runtime,
    git_head: Callable[[Path], str] = _current_commit,
    tracked_clean: Callable[[Path], bool] = _tracked_worktree_clean,
    arm_contract_checker: Callable[[TargetRuntime, str], Mapping[str, Any]] = _native_generator_contract,
) -> dict[str, Any]:
    root = Path(code_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    actual_code_commit = git_head(root)
    _validate_arm_identity(
        arm=arm,
        expected_code_commit=expected_code_commit,
        actual_code_commit=actual_code_commit,
    )
    if not tracked_clean(root):
        raise RuntimeError(f"target code root is not clean: {root}")
    target = target_loader(root)
    if Path(target.code_root).resolve() != root:
        raise RuntimeError("target loader returned a runtime for a different code root")
    source_config, effective_config, config_diff, overrides = config_projection(
        system=system,
        case_dir=(
            Path(output_dir)
            if output_dir is not None
            else Path("/direction-candidate-budget-preflight") / system
        ),
        total_force_budget=total_force_budget,
        seed=seed,
        base_runner=target.base_runner,
    )
    base_preflight = _mapping(
        target.base_runner.preflight(
            system=system,
            expected_git_commit=expected_code_commit,
        )
    )
    if base_preflight.get("execution_commit") != expected_code_commit:
        raise RuntimeError("frozen runner preflight returned a mismatched execution commit")
    cuda = base_preflight.get("cuda", {})
    if not isinstance(cuda, Mapping) or cuda.get("available") is not True:
        raise RuntimeError("target preflight did not prove CUDA availability")
    frozen_runner_path = Path(target.frozen_runner_path)
    if not frozen_runner_path.is_file():
        raise FileNotFoundError(frozen_runner_path)
    return {
        "schema_version": 1,
        "arm": arm,
        "system": system,
        "seed": int(seed),
        "total_force_budget": int(total_force_budget),
        "target": {
            "code_root": str(root),
            "execution_commit": actual_code_commit,
            "tracked_worktree_clean": True,
            "frozen_runner_path": str(frozen_runner_path),
            "frozen_runner_sha256": _sha256(frozen_runner_path),
            "ablation_runner_path": str(RUNNER_PATH),
            "ablation_runner_sha256": _sha256(RUNNER_PATH),
        },
        "base_preflight": base_preflight,
        "source_config": source_config,
        "effective_config": effective_config,
        "config_diff": config_diff,
        "overrides": overrides,
        "native_generator_contract": dict(arm_contract_checker(target, arm)),
    }


def _energy_trace(result: Any) -> list[dict[str, Any]]:
    initial_energy = _finite_energy(result.archive.entries[0].energy, "initial energy")
    best_energy = initial_energy
    points = [
        {
            "trial": 0,
            "energy_eV": initial_energy,
            "best_energy_eV": initial_energy,
            "accepted_new_basin": True,
        }
    ]
    for trial, record in enumerate(result.walk_history, start=1):
        energy = _finite_energy(record.energy, f"walk record {trial} energy")
        best_energy = min(best_energy, energy)
        points.append(
            {
                "trial": trial,
                "energy_eV": energy,
                "best_energy_eV": best_energy,
                "accepted_new_basin": bool(record.accepted_new_basin),
            }
        )
    return points


def _walk_records(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "trial": trial,
            "seed_entry_id": _nonnegative_exact_int(
                record.seed_entry_id, f"walk record {trial} seed_entry_id"
            ),
            "discovered_entry_id": _nonnegative_exact_int(
                record.discovered_entry_id, f"walk record {trial} discovered_entry_id"
            ),
            "energy_eV": _finite_energy(record.energy, f"walk record {trial} energy"),
            "accepted_new_basin": bool(record.accepted_new_basin),
        }
        for trial, record in enumerate(result.walk_history, start=1)
    ]


def _direction_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"direction diagnostics file was not written: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid direction diagnostics JSON at line {line_number}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"direction diagnostics record {line_number} is not a mapping")
        candidate_count = record.get("candidate_count")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count <= 0:
            raise RuntimeError(
                f"direction diagnostics record {line_number} has invalid candidate_count"
            )
        records.append(record)
    return records


def _validate_run_closure(
    *,
    result: Any,
    walker: Any,
    config: Mapping[str, Any],
    total_force_budget: int,
    direction_path: Path,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    total_force_budget = _positive_exact_int(total_force_budget, "total_force_budget")
    force_evaluations = _nonnegative_exact_int(
        result.stats["force_evaluations"], "force_evaluations"
    )
    if force_evaluations != total_force_budget:
        raise RuntimeError(
            f"fixed total force budget not reached exactly: {force_evaluations} != {total_force_budget}"
        )
    if _nonnegative_exact_int(
        result.stats.get("budget_exhausted", 0), "budget_exhausted"
    ) != 1:
        raise RuntimeError("fixed-budget run did not terminate by budget exhaustion")
    counts = walker.calculator.snapshot()
    if _nonnegative_exact_int(counts.total, "counter total") != force_evaluations:
        raise RuntimeError("force-evaluation total does not close")
    purpose_counts = {
        str(key): _nonnegative_exact_int(value, f"purpose count {key!r}")
        for key, value in counts.as_dict().items()
    }
    if sum(purpose_counts.values()) != force_evaluations:
        raise RuntimeError("purpose accounting does not close")
    if purpose_counts.get("unattributed", 0) != 0:
        raise RuntimeError("purpose accounting contains unattributed evaluations")

    direction_records = _direction_records(direction_path)
    candidate_sum = sum(
        _nonnegative_exact_int(record["candidate_count"], "direction candidate_count")
        for record in direction_records
    )
    direction_choices = _nonnegative_exact_int(
        result.stats.get("direction_choices", 0), "direction_choices"
    )
    candidate_evaluations = _nonnegative_exact_int(
        result.stats.get("direction_candidate_evaluations", 0),
        "direction_candidate_evaluations",
    )
    if direction_choices != len(direction_records):
        raise RuntimeError("direction diagnostics omit one or more direction choices")
    if candidate_evaluations != candidate_sum:
        raise RuntimeError("direction candidate statistics do not match direction diagnostics")
    expected_direction_evaluations = 2 * candidate_sum
    recorded_direction_evaluations = purpose_counts.get("direction_oracle", 0)
    if recorded_direction_evaluations != expected_direction_evaluations:
        raise RuntimeError(
            "direction-oracle force ledger does not equal two central-HVP evaluations per candidate"
        )
    candidate_budget = _positive_exact_int(config["oracle_candidates"], "oracle_candidates")
    candidate_max = max(
        (
            _nonnegative_exact_int(
                record["candidate_count"], "direction candidate_count"
            )
            for record in direction_records
        ),
        default=0,
    )
    n_trials = _nonnegative_exact_int(result.stats["n_trials"], "n_trials")
    record_counts = {
        "n_trials": n_trials,
        "walk_records": len(result.walk_history),
        "unlogged_walk_trials": n_trials - len(result.walk_history),
        "direction_records": len(direction_records),
        "unlogged_direction_choices": direction_choices - len(direction_records),
    }
    if record_counts["unlogged_walk_trials"] != 0:
        raise RuntimeError("walk records do not close against n_trials")
    if record_counts["unlogged_direction_choices"] != 0:
        raise RuntimeError("direction records do not close against direction choices")
    direction_audit = {
        "candidate_count_sum": candidate_sum,
        "candidate_count_max": candidate_max,
        "candidate_count_over_oracle_candidates": sum(
            _nonnegative_exact_int(
                record["candidate_count"], "direction candidate_count"
            )
            > candidate_budget
            for record in direction_records
        ),
        "expected_force_evaluations": expected_direction_evaluations,
        "recorded_force_evaluations": recorded_direction_evaluations,
    }
    return purpose_counts, record_counts, direction_audit


def run(
    *,
    arm: str,
    system: str,
    seed: int,
    total_force_budget: int,
    code_root: Path,
    output_dir: Path,
    expected_code_commit: str,
    preflight_only: bool = False,
    target_loader: Callable[[Path], TargetRuntime] = _load_target_runtime,
    preflight_fn: Callable[..., dict[str, Any]] = preflight,
) -> dict[str, Any]:
    seed = _nonnegative_exact_int(seed, "seed")
    total_force_budget = _positive_exact_int(total_force_budget, "total_force_budget")
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    checked = preflight_fn(
        arm=arm,
        system=system,
        seed=seed,
        total_force_budget=total_force_budget,
        code_root=code_root,
        expected_code_commit=expected_code_commit,
        output_dir=output_dir,
    )
    if preflight_only:
        return checked

    target = target_loader(Path(code_root).resolve())
    source_config, effective_config, config_diff, overrides = config_projection(
        system=system,
        case_dir=output_dir,
        total_force_budget=total_force_budget,
        seed=seed,
        base_runner=target.base_runner,
    )
    if source_config != checked["source_config"]:
        raise RuntimeError("source config changed after preflight")
    if effective_config != checked["effective_config"]:
        raise RuntimeError("effective config changed after preflight")
    if config_diff != checked["config_diff"] or overrides != checked["overrides"]:
        raise RuntimeError("config projection changed after preflight")

    config = target.base_runner.build_config(system, output_dir)
    config = replace(
        config,
        **checked["overrides"],
    )
    if _json_mapping(config) != effective_config:
        raise RuntimeError("walker configuration does not equal preflight projection")

    output_dir.mkdir(parents=True)
    state = target.base_runner.load_state(system)
    walker = target.walker_class(
        calculator=target.calculator_wrapper(target.base_runner._calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    total_wall_time_s = perf_counter() - started
    direction_path = Path(config.direction_diagnostics_path)
    purpose_counts, record_counts, direction_audit = _validate_run_closure(
        result=result,
        walker=walker,
        config=effective_config,
        total_force_budget=total_force_budget,
        direction_path=direction_path,
    )
    if arm == "hardcap" and direction_audit["candidate_count_over_oracle_candidates"] != 0:
        raise RuntimeError("hardcap arm emitted a candidate pool larger than oracle_candidates")

    energy_trace = _energy_trace(result)
    walk_records = _walk_records(result)
    best_energy = _finite_energy(result.best_energy, "best energy")
    initial_energy = _finite_energy(energy_trace[0]["energy_eV"], "initial energy")
    optimizer_diagnostics = walker.relaxation_diagnostics()
    summary = {
        **checked,
        "effective_config": effective_config,
        "force_evaluations": _nonnegative_exact_int(
            result.stats["force_evaluations"], "force_evaluations"
        ),
        "purpose_counts": purpose_counts,
        "stats": result.stats,
        "record_counts": record_counts,
        "direction_oracle_audit": direction_audit,
        "fragment_rejections": _nonnegative_exact_int(
            result.stats.get("fragment_rejections", 0), "fragment_rejections"
        ),
        "initial_energy_eV": initial_energy,
        "best_energy_eV": best_energy,
        "energy_drop_eV": initial_energy - best_energy,
        "optimizer_diagnostics": optimizer_diagnostics,
        "timing": {"total_wall_time_s": total_wall_time_s},
        "walk_records": walk_records,
    }
    target.write_state(output_dir / "best_minimum.xyz", result.best_state)
    _write_json(output_dir / "energy_trace.json", energy_trace)
    _write_json(output_dir / "walk_records.json", walk_records)
    _write_json(output_dir / "optimizer_diagnostics.json", optimizer_diagnostics)
    for filename in REQUIRED_NON_SUMMARY_OUTPUT_FILES:
        if not (output_dir / filename).is_file():
            raise RuntimeError(f"required output was not written: {filename}")
    _write_json(output_dir / "summary.json", summary)
    if not (output_dir / "summary.json").is_file():
        raise RuntimeError("required output was not written: summary.json")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=tuple(ARM_COMMITS))
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument(
        "--total-force-budget",
        type=int,
        default=PRODUCTION_TOTAL_FORCE_BUDGET,
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.preflight_only and args.total_force_budget != PRODUCTION_TOTAL_FORCE_BUDGET:
        raise ValueError(
            "production execution requires --total-force-budget "
            f"{PRODUCTION_TOTAL_FORCE_BUDGET}"
        )
    summary = run(
        arm=args.arm,
        system=args.system,
        seed=args.seed,
        total_force_budget=args.total_force_budget,
        code_root=args.code_root,
        output_dir=args.output,
        expected_code_commit=args.expected_code_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
