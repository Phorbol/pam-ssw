#!/usr/bin/env python3
"""Run one fixed-budget discrete-versus-plain-Rayleigh--Ritz LS-SSW arm.

The two arms share a frozen production configuration.  ``rayleigh_ritz`` only
changes how already-evaluated native directions are represented for selection;
it does not enable Ritz synthesis or any extra candidate-generation feature.
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
ARMS = ("discrete", "rayleigh_ritz")
MAX_TRIALS = 200
PRODUCTION_TOTAL_FORCE_BUDGET = 6000
EXPECTED_ORACLE_CANDIDATES = {"c60": 12, "pdo": 8}
OUTPUT_PATH_FIELDS = {
    "accepted_structures_dir",
    "accepted_structures_log",
    "direction_diagnostics_path",
}
NATIVE_DIRECTION_AND_HVP_FIELDS = (
    "oracle_candidates",
    "hvp_epsilon",
    "direction_curvature_source",
    "n_bond_pairs",
    "enable_momentum_candidate",
    "enable_anchor_candidate",
    "random_direction_distribution",
    "enable_bond_form_break_split",
    "n_bond_formation_pairs",
    "n_bond_breaking_pairs",
    "bond_formation_max_distance",
    "bond_breaking_max_distance",
    "bond_distance_threshold",
    "use_archive_acquisition",
    "seed_selection_mode",
    "proposal_optimizer",
    "proposal_fmax",
    "target_uphill_energy",
    "quench_optimizer",
    "quench_fallback_optimizer",
    "quench_fmax",
    "quench_maxiter",
    "local_softening_mode",
    "local_softening_strength",
    "local_softening_penalty",
    "local_softening_xi",
)
REQUIRED_OUTPUTS = (
    "best_minimum.xyz",
    "energy_trace.json",
    "walk_records.json",
    "optimizer_diagnostics.json",
    "direction_trace.jsonl",
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


def _assert_module_under_root(module: Any, root: Path, label: str) -> None:
    source_path = getattr(module, "__file__", None)
    if source_path is None:
        raise RuntimeError(f"{label} has no source path")
    try:
        Path(source_path).resolve().relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"{label} was not imported from the explicit target worktree {root}: {source_path}"
        ) from error


def _load_target_runtime(code_root: Path) -> TargetRuntime:
    """Import the production runner and pamssw from exactly one clean worktree."""

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
    _assert_module_under_root(pamssw_module, root, "pamssw")
    _assert_module_under_root(calculators_module, root, "pamssw.calculators")
    _assert_module_under_root(walker_module, root, "pamssw.walker")

    module_name = "_direction_selection_ritz_frozen_production"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import frozen production runner: {runner_path}")
    base_runner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = base_runner
    spec.loader.exec_module(base_runner)
    _assert_module_under_root(base_runner, root, "frozen production runner")
    return TargetRuntime(
        code_root=root,
        base_runner=base_runner,
        calculator_wrapper=calculators_module.ASECalculator,
        walker_class=walker_module.SurfaceWalker,
        write_state=pamssw_module.write_state,
    )


def _nonnegative_exact_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a builtin exact non-negative integer")
    return value


def _positive_exact_int(value: Any, label: str) -> int:
    value = _nonnegative_exact_int(value, label)
    if value == 0:
        raise ValueError(f"{label} must be positive")
    return value


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite builtin number")
    return float(value)


def _json_config(config: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True, allow_nan=False))


def _config_diff(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, list[Any]]:
    if left.keys() != right.keys():
        raise RuntimeError("configuration fields differ")
    return {
        key: [left[key], right[key]]
        for key in sorted(left)
        if left[key] != right[key]
    }


def _validate_pair_protocol(
    *, system: str, discrete: Mapping[str, Any], ritz: Mapping[str, Any]
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    pair_diff = _config_diff(discrete, ritz)
    expected_diff = {"direction_selection_mode": ["discrete", "rayleigh_ritz"]}
    if pair_diff != expected_diff:
        raise RuntimeError(
            "paired arms must differ only in direction_selection_mode: "
            f"{pair_diff}"
        )
    expected_oracle = EXPECTED_ORACLE_CANDIDATES[system]
    if discrete["oracle_candidates"] != expected_oracle:
        raise RuntimeError(
            f"frozen production {system} oracle_candidates drifted: "
            f"expected {expected_oracle}, got {discrete['oracle_candidates']}"
        )
    for config_name, config in (("discrete", discrete), ("rayleigh_ritz", ritz)):
        expected = {
            "max_trials": MAX_TRIALS,
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
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(
                    f"{config_name} frozen protocol drifted: {key}={config.get(key)!r}"
                )
    matched_fields = {key: discrete[key] for key in NATIVE_DIRECTION_AND_HVP_FIELDS}
    return pair_diff, {
        "same_native_candidate_and_hvp_protocol": True,
        "matched_fields": matched_fields,
        "plain_ritz": {
            "direction_selection_mode": "rayleigh_ritz",
            "direction_synthesis_mode": "none",
            "regularized_ritz_enabled": False,
            "direction_probe_enabled": False,
            "direction_type_ucb_enabled": False,
            "direction_archive_enabled": False,
            "plateau_evolution_enabled": False,
            "archive_escape_momentum_enabled": False,
        },
        "runtime_cost_source": "purpose_counts.direction_oracle",
        "true_curvature_definition": "native_central_fd_true_hvp_subspace_projection",
        "direct_mixed_direction_stencil_note": (
            "on a nonlinear PES this reused projection differs from a direct "
            "mixed-direction central-FD stencil by O(hvp_epsilon**2)"
        ),
        "legacy_candidate_count_note": (
            "direction_trace.candidate_count includes a zero-HVP synthetic Ritz "
            "candidate when plain Rayleigh-Ritz is constructed"
        ),
    }


def config_projection(
    *,
    arm: str,
    system: str,
    case_dir: Path,
    total_force_budget: int,
    seed: int,
    base_runner: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]], dict[str, list[Any]], dict[str, Any]]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if system not in SYSTEMS:
        raise ValueError(f"unknown system: {system}")
    total_force_budget = _positive_exact_int(total_force_budget, "total_force_budget")
    seed = _nonnegative_exact_int(seed, "seed")
    source_config = base_runner.build_config(system, Path(case_dir))
    source = _json_config(source_config)
    shared_overrides = {
        "max_force_evals": total_force_budget,
        "rng_seed": seed,
    }
    discrete = _json_config(
        replace(source_config, **shared_overrides, direction_selection_mode="discrete")
    )
    ritz = _json_config(
        replace(source_config, **shared_overrides, direction_selection_mode="rayleigh_ritz")
    )
    pair_diff, protocol = _validate_pair_protocol(
        system=system, discrete=discrete, ritz=ritz
    )
    effective = discrete if arm == "discrete" else ritz
    source_diff = _config_diff(source, effective)
    allowed_source_changes = {"max_force_evals", "rng_seed", "direction_selection_mode"}
    unexpected = set(source_diff) - allowed_source_changes
    if unexpected:
        raise RuntimeError(
            f"focused direction-selection runner has unexpected source changes: {sorted(unexpected)}"
        )
    return source, effective, source_diff, pair_diff, protocol


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
    if not root.is_dir():
        raise FileNotFoundError(root)
    actual_commit = git_head(root)
    if expected_git_commit != actual_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, got {actual_commit}"
        )
    if not tracked_clean(root):
        raise RuntimeError(f"target worktree is not clean: {root}")
    target = target_loader(root)
    if Path(target.code_root).resolve() != root:
        raise RuntimeError("target loader returned a different code root")
    source, effective, source_diff, pair_diff, protocol = config_projection(
        arm=arm,
        system=system,
        case_dir=(Path(output_dir) if output_dir is not None else root / ".ritz-preflight"),
        total_force_budget=total_force_budget,
        seed=seed,
        base_runner=target.base_runner,
    )
    base_preflight = dict(
        target.base_runner.preflight(
            system=system,
            expected_git_commit=expected_git_commit,
        )
    )
    if base_preflight.get("execution_commit") != actual_commit:
        raise RuntimeError("frozen production preflight returned a mismatched commit")
    cuda = base_preflight.get("cuda")
    if not isinstance(cuda, Mapping) or cuda.get("available") is not True:
        raise RuntimeError("frozen production preflight did not prove CUDA availability")
    for field in ("input_sha256", "model_sha256", "input_path", "model_path"):
        if not isinstance(base_preflight.get(field), str) or not base_preflight[field]:
            raise RuntimeError(f"frozen production preflight omitted {field}")
    return {
        "schema_version": 1,
        "arm": arm,
        "system": system,
        "seed": int(seed),
        "total_force_budget": int(total_force_budget),
        "target": {
            "code_root": str(root),
            "execution_commit": actual_commit,
            "tracked_worktree_clean": True,
            "frozen_runner_path": str(root / FROZEN_RUNNER_RELATIVE),
            "frozen_runner_sha256": _sha256(root / FROZEN_RUNNER_RELATIVE)
            if (root / FROZEN_RUNNER_RELATIVE).is_file()
            else None,
            "ablation_runner_path": str(Path(__file__).resolve()),
            "ablation_runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "base_preflight": base_preflight,
        "source_config": source,
        "effective_config": effective,
        "source_to_effective_config_diff": source_diff,
        "paired_arm_config_diff": pair_diff,
        "plain_ritz_hvp_contract": protocol,
    }


def _energy_trace(result: Any) -> list[dict[str, Any]]:
    initial = _finite_number(result.archive.entries[0].energy, "initial energy")
    best = initial
    trace = [
        {
            "trial": 0,
            "energy_eV": initial,
            "best_energy_eV": initial,
            "accepted_new_basin": True,
        }
    ]
    for trial, record in enumerate(result.walk_history, start=1):
        energy = _finite_number(record.energy, f"walk record {trial} energy")
        best = min(best, energy)
        trace.append(
            {
                "trial": trial,
                "energy_eV": energy,
                "best_energy_eV": best,
                "accepted_new_basin": bool(record.accepted_new_basin),
            }
        )
    return trace


def _walk_records(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "trial": trial,
            "seed_entry_id": _nonnegative_exact_int(record.seed_entry_id, "seed_entry_id"),
            "discovered_entry_id": _nonnegative_exact_int(
                record.discovered_entry_id, "discovered_entry_id"
            ),
            "energy_eV": _finite_number(record.energy, "walk record energy"),
            "accepted_new_basin": bool(record.accepted_new_basin),
        }
        for trial, record in enumerate(result.walk_history, start=1)
    ]


def _direction_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"direction diagnostics file was not written: {path}")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid direction diagnostics JSON at line {number}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"direction diagnostics record {number} is not a mapping")
        _positive_exact_int(record.get("candidate_count"), f"direction record {number} candidate_count")
        if not isinstance(record.get("selected_kind"), str):
            raise RuntimeError(f"direction diagnostics record {number} has no selected_kind")
        records.append(record)
    return records


def _validate_run_closure(
    *, result: Any, walker: Any, total_force_budget: int, direction_path: Path
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    total_force_budget = _positive_exact_int(total_force_budget, "total_force_budget")
    force_evaluations = _nonnegative_exact_int(
        result.stats["force_evaluations"], "force_evaluations"
    )
    if force_evaluations != total_force_budget:
        raise RuntimeError(
            f"fixed total force budget was not reached exactly: {force_evaluations} != {total_force_budget}"
        )
    if _nonnegative_exact_int(result.stats.get("budget_exhausted", 0), "budget_exhausted") != 1:
        raise RuntimeError("fixed-budget execution did not terminate by budget exhaustion")
    counts = walker.calculator.snapshot()
    if _nonnegative_exact_int(counts.total, "counter total") != force_evaluations:
        raise RuntimeError("force-evaluation total does not close")
    purpose_counts = {
        str(key): _nonnegative_exact_int(value, f"purpose count {key!r}")
        for key, value in counts.as_dict().items()
    }
    if sum(purpose_counts.values()) != force_evaluations:
        raise RuntimeError("purpose-resolved force ledger does not close")
    if purpose_counts.get("unattributed", 0) != 0:
        raise RuntimeError("purpose accounting contains unattributed evaluations")

    direction_records = _direction_records(direction_path)
    choices = _nonnegative_exact_int(result.stats.get("direction_choices", 0), "direction_choices")
    if choices != len(direction_records):
        raise RuntimeError("direction diagnostics do not close against direction choices")
    candidate_count_sum = sum(
        _positive_exact_int(record["candidate_count"], "direction candidate_count")
        for record in direction_records
    )
    legacy_candidate_stat = _nonnegative_exact_int(
        result.stats.get("direction_candidate_evaluations", 0),
        "direction_candidate_evaluations",
    )
    if legacy_candidate_stat != candidate_count_sum:
        raise RuntimeError("legacy direction candidate statistic does not match diagnostics")
    record_counts = {
        "n_trials": _nonnegative_exact_int(result.stats["n_trials"], "n_trials"),
        "walk_records": len(result.walk_history),
        "direction_records": len(direction_records),
        "unlogged_walk_trials": _nonnegative_exact_int(result.stats["n_trials"], "n_trials")
        - len(result.walk_history),
        "unlogged_direction_choices": choices - len(direction_records),
    }
    if record_counts["unlogged_walk_trials"] != 0:
        raise RuntimeError("walk records do not close against n_trials")
    selected_kind_counts: dict[str, int] = {}
    for record in direction_records:
        kind = record["selected_kind"]
        selected_kind_counts[kind] = selected_kind_counts.get(kind, 0) + 1
    direction_audit = {
        "selection_count": choices,
        "reported_candidate_count_sum": candidate_count_sum,
        "reported_candidate_count_max": max(
            (record["candidate_count"] for record in direction_records), default=0
        ),
        "reported_candidate_count_semantics": "includes_zero_hvp_synthetic_ritz_when_constructed",
        "native_candidate_count": None,
        "native_candidate_count_reason": "unsupported_by_current_direction_trace",
        "direction_oracle_force_evaluations": purpose_counts.get("direction_oracle", 0),
        "selected_kind_counts": selected_kind_counts,
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
    expected_git_commit: str,
    preflight_only: bool = False,
    target_loader: Callable[[Path], TargetRuntime] = _load_target_runtime,
    preflight_fn: Callable[..., dict[str, Any]] = preflight,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not preflight_only and output_dir.exists():
        raise FileExistsError(output_dir)
    checked = preflight_fn(
        arm=arm,
        system=system,
        seed=seed,
        total_force_budget=total_force_budget,
        code_root=code_root,
        output_dir=output_dir,
        expected_git_commit=expected_git_commit,
    )
    if preflight_only:
        return checked

    target = target_loader(Path(code_root).resolve())
    source, effective, source_diff, pair_diff, protocol = config_projection(
        arm=arm,
        system=system,
        case_dir=output_dir,
        total_force_budget=total_force_budget,
        seed=seed,
        base_runner=target.base_runner,
    )
    for key, value in {
        "source_config": source,
        "effective_config": effective,
        "source_to_effective_config_diff": source_diff,
        "paired_arm_config_diff": pair_diff,
        "plain_ritz_hvp_contract": protocol,
    }.items():
        if checked[key] != value:
            raise RuntimeError(f"{key} changed after preflight")
    config = replace(target.base_runner.build_config(system, output_dir), **{
        "max_force_evals": total_force_budget,
        "rng_seed": seed,
        "direction_selection_mode": arm,
    })
    if _json_config(config) != effective:
        raise RuntimeError("walker config does not equal the preflight effective config")

    output_dir.mkdir(parents=True)
    state = target.base_runner.load_state(system)
    walker = target.walker_class(
        calculator=target.calculator_wrapper(target.base_runner._calculator()),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(state)
    wall_time_s = perf_counter() - started
    purpose_counts, record_counts, direction_audit = _validate_run_closure(
        result=result,
        walker=walker,
        total_force_budget=total_force_budget,
        direction_path=Path(config.direction_diagnostics_path),
    )
    energy_trace = _energy_trace(result)
    initial_energy = _finite_number(energy_trace[0]["energy_eV"], "initial energy")
    best_energy = _finite_number(result.best_energy, "best energy")
    walk_records = _walk_records(result)
    termination = {
        "reason": "force_budget_exhausted",
        "budget_exhausted": True,
        "stats_termination_counts": result.stats.get("termination_counts"),
    }
    summary = {
        **checked,
        "effective_config": effective,
        "force_evaluations": _nonnegative_exact_int(
            result.stats["force_evaluations"], "force_evaluations"
        ),
        "purpose_counts": purpose_counts,
        "stats": result.stats,
        "record_counts": record_counts,
        "direction_selection_audit": direction_audit,
        "initial_energy_eV": initial_energy,
        "best_energy_eV": best_energy,
        "energy_drop_eV": initial_energy - best_energy,
        "optimizer_diagnostics": walker.relaxation_diagnostics(),
        "timing": {"total_wall_time_s": wall_time_s},
        "termination": termination,
        "walk_records": walk_records,
    }
    target.write_state(output_dir / "best_minimum.xyz", result.best_state)
    _write_json(output_dir / "energy_trace.json", energy_trace)
    _write_json(output_dir / "walk_records.json", walk_records)
    _write_json(output_dir / "optimizer_diagnostics.json", summary["optimizer_diagnostics"])
    for filename in REQUIRED_OUTPUTS:
        if not (output_dir / filename).is_file():
            raise RuntimeError(f"required output was not written: {filename}")
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--total-force-budget", type=int, default=PRODUCTION_TOTAL_FORCE_BUDGET)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if (
        not args.preflight_only
        and args.total_force_budget != PRODUCTION_TOTAL_FORCE_BUDGET
    ):
        raise ValueError(
            "production execution requires --total-force-budget "
            f"{PRODUCTION_TOTAL_FORCE_BUDGET}"
        )
    summary = run(
        arm=args.arm,
        system=args.system,
        seed=args.seed,
        total_force_budget=args.total_force_budget,
        code_root=REPO_ROOT,
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
