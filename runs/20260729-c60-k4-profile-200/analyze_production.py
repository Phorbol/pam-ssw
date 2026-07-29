#!/usr/bin/env python3
"""Validate and compare the public K4 C60 run with historical K12 evidence."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
DEFAULT_K4_DIR = RUN_ROOT / "output-c60-seed42"
DEFAULT_K12_DIR = (
    RUN_ROOT.parent
    / "20260728-safe-lbfgs-strict-quench-200-production"
    / "output-c60-seed42"
)
OUTPUT_CONFIG_FIELDS = (
    "accepted_structures_dir",
    "accepted_structures_log",
    "direction_diagnostics_path",
)
K4_SCHEMA_ONLY_FIELDS = ("block_krylov_blocks", "block_krylov_depth")
EXPECTED_SCIENTIFIC_DIFF = {"oracle_candidates": [12, 4]}
IDENTITY_FIELDS = (
    "input_sha256",
    "model_sha256",
    "runtime_versions",
    "calculator",
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing raw artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    for field in OUTPUT_CONFIG_FIELDS:
        normalized.pop(field, None)
    return normalized


def _validate_case(case_dir: Path, label: str) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    trace_path = case_dir / "energy_trace.json"
    summary = _mapping(_read_json(summary_path), f"{label} summary")
    trace = _read_json(trace_path)
    if not isinstance(trace, list) or len(trace) != 201:
        raise ValueError(f"{label} energy trace must contain trials 0 through 200")

    stats = _mapping(summary.get("stats"), f"{label} stats")
    if stats.get("n_trials") != 200:
        raise ValueError(f"{label} did not complete exactly 200 trials")
    config = _mapping(summary.get("effective_config"), f"{label} config")
    required_config = {
        "max_trials": 200,
        "rng_seed": 42,
        "max_steps_per_walk": 8,
        "proposal_relax_steps": 80,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
        "direction_type_ucb_enabled": False,
    }
    drift = {
        field: [expected, config.get(field)]
        for field, expected in required_config.items()
        if config.get(field) != expected
    }
    if drift:
        raise ValueError(f"{label} production protocol drift: {drift}")

    purpose = _mapping(summary.get("purpose_counts"), f"{label} purpose counts")
    total = _nonnegative_int(
        summary.get("force_evaluations"), f"{label} force evaluations"
    )
    purpose_total = sum(
        _nonnegative_int(value, f"{label} purpose {name}")
        for name, value in purpose.items()
    )
    if purpose_total != total:
        raise ValueError(f"{label} purpose ledger does not close")
    if purpose.get("unattributed") != 0:
        raise ValueError(f"{label} purpose ledger contains unattributed calls")

    initial = _finite(summary.get("initial_energy_eV"), f"{label} initial energy")
    best = _finite(summary.get("best_energy_eV"), f"{label} best energy")
    drop = _finite(summary.get("energy_drop_eV"), f"{label} energy drop")
    if not math.isclose(initial - best, drop, abs_tol=1.0e-10):
        raise ValueError(f"{label} energy drop does not close")
    best_trace = [
        _finite(
            _mapping(row, f"{label} trace row").get("best_energy_eV"),
            f"{label} trace best energy",
        )
        for row in trace
    ]
    if not math.isclose(best_trace[0], initial, abs_tol=1.0e-10):
        raise ValueError(f"{label} trace initial energy does not close")
    if not math.isclose(min(best_trace), best, abs_tol=1.0e-10):
        raise ValueError(f"{label} trace best energy does not close")

    optimizer = _mapping(
        summary.get("optimizer_telemetry"), f"{label} optimizer telemetry"
    )
    wall = _finite(
        _mapping(summary.get("timing"), f"{label} timing").get(
            "total_wall_time_s"
        ),
        f"{label} wall time",
    )
    improvements = [initial - energy for energy in best_trace]
    trapezoid_auc = (
        sum(improvements[1:-1])
        + 0.5 * improvements[0]
        + 0.5 * improvements[-1]
    )

    def first_hit(threshold: float) -> int | None:
        return next(
            (
                trial
                for trial, improvement in enumerate(improvements)
                if improvement >= threshold
            ),
            None,
        )

    last_best_trial = max(
        (
            trial
            for trial in range(1, len(best_trace))
            if best_trace[trial] < best_trace[trial - 1]
        ),
        default=0,
    )
    direction_total = sum(
        _nonnegative_int(
            stats.get(f"direction_selected_{kind}", 0),
            f"{label} selected {kind}",
        )
        for kind in ("random", "bond", "momentum")
    )
    direction_selection = {
        kind: {
            "count": int(stats.get(f"direction_selected_{kind}", 0)),
            "fraction": (
                float(stats.get(f"direction_selected_{kind}", 0))
                / direction_total
                if direction_total
                else 0.0
            ),
        }
        for kind in ("random", "bond", "momentum")
    }
    minima = _nonnegative_int(stats.get("n_minima"), f"{label} minima")
    return {
        "execution_commit": summary.get("execution_commit"),
        "identity": {field: summary.get(field) for field in IDENTITY_FIELDS},
        "config": _normalized_config(config),
        "terminal": {
            "initial_energy_eV": initial,
            "best_energy_eV": best,
            "energy_drop_eV": drop,
            "n_trials": 200,
            "n_minima": minima,
            "duplicate_rate": _finite(
                stats.get("duplicate_rate"), f"{label} duplicate rate"
            ),
            "last_best_trial": last_best_trial,
        },
        "progress": {
            "best_drop_auc_eV_trial": trapezoid_auc,
            "mean_best_drop_eV": sum(improvements) / len(improvements),
            "first_drop_10_eV_trial": first_hit(10.0),
            "first_drop_20_eV_trial": first_hit(20.0),
            "first_drop_30_eV_trial": first_hit(30.0),
        },
        "cost": {
            "force_evaluations": total,
            "wall_time_s": wall,
            "wall_time_per_trial_s": wall / 200,
            "energy_drop_per_1000_force_evaluations_eV": drop * 1000 / total,
            "minima_per_1000_force_evaluations": minima * 1000 / total,
            "purpose_counts": dict(purpose),
            "purpose_fractions": {
                name: value / total for name, value in purpose.items()
            },
        },
        "direction": {
            "bias_steps": int(stats.get("bias_steps", 0)),
            "candidate_evaluations": int(
                stats.get("direction_candidate_evaluations", 0)
            ),
            "selection": direction_selection,
        },
        "optimizer": {
            field: optimizer.get(field)
            for field in (
                "proposal_relax_mean_iterations",
                "proposal_relax_median_iterations",
                "proposal_relax_p90_iterations",
                "proposal_relax_termination_maxiter",
                "true_quench_mean_iterations",
                "true_quench_median_iterations",
                "true_quench_p90_iterations",
                "quench_fallback_attempts",
                "quench_fallback_converged",
                "true_quench_termination_converged",
                "true_quench_termination_unconverged",
            )
        },
        "raw_initial_quench_ledger": {
            "bootstrap_true_quench": purpose.get("bootstrap_true_quench", 0),
            "starter_true_quench": purpose.get("starter_true_quench", 0),
            "semantic_note": (
                "SurfaceWalker.run labels its one initial raw-State quench as "
                "starter_true_quench in this execution; it is budgeted but the "
                "ledger name is not the intended bootstrap_true_quench label."
            ),
        },
        "artifact_sha256": {
            "summary": _sha256(summary_path),
            "energy_trace": _sha256(trace_path),
        },
    }


def analyze(k4_dir: Path, k12_dir: Path) -> dict[str, Any]:
    k4 = _validate_case(Path(k4_dir), "K4")
    k12 = _validate_case(Path(k12_dir), "K12")
    for field in IDENTITY_FIELDS:
        if k4["identity"][field] != k12["identity"][field]:
            raise ValueError(f"K4 and K12 differ in runtime identity: {field}")

    k4_config = dict(k4.pop("config"))
    k12_config = dict(k12.pop("config"))
    schema_only = {
        field: k4_config.pop(field)
        for field in K4_SCHEMA_ONLY_FIELDS
        if field in k4_config and field not in k12_config
    }
    unexpected_fields = set(k4_config) ^ set(k12_config)
    if unexpected_fields:
        raise ValueError(
            f"unexpected config schema drift: {sorted(unexpected_fields)}"
        )
    scientific_diff = {
        field: [k12_config[field], k4_config[field]]
        for field in sorted(k4_config)
        if k4_config[field] != k12_config[field]
    }
    if scientific_diff != EXPECTED_SCIENTIFIC_DIFF:
        raise ValueError(
            f"unexpected scientific config drift: {scientific_diff}"
        )

    k4_cost = k4["cost"]
    k12_cost = k12["cost"]
    k4_terminal = k4["terminal"]
    k12_terminal = k12["terminal"]
    k4_direction_fe = k4_cost["purpose_counts"]["direction_oracle"]
    k12_direction_fe = k12_cost["purpose_counts"]["direction_oracle"]
    comparison = {
        "best_energy_delta_eV": (
            k4_terminal["best_energy_eV"] - k12_terminal["best_energy_eV"]
        ),
        "energy_drop_delta_eV": (
            k4_terminal["energy_drop_eV"] - k12_terminal["energy_drop_eV"]
        ),
        "best_drop_auc_delta_eV_trial": (
            k4["progress"]["best_drop_auc_eV_trial"]
            - k12["progress"]["best_drop_auc_eV_trial"]
        ),
        "total_force_evaluations_saved": (
            k12_cost["force_evaluations"] - k4_cost["force_evaluations"]
        ),
        "direction_force_evaluations_saved": k12_direction_fe - k4_direction_fe,
        "non_direction_force_evaluations_delta": (
            (k4_cost["force_evaluations"] - k4_direction_fe)
            - (k12_cost["force_evaluations"] - k12_direction_fe)
        ),
        "wall_time_delta_s": k4_cost["wall_time_s"] - k12_cost["wall_time_s"],
        "minima_delta": k4_terminal["n_minima"] - k12_terminal["n_minima"],
        "duplicate_rate_delta": (
            k4_terminal["duplicate_rate"] - k12_terminal["duplicate_rate"]
        ),
    }
    return {
        "schema_version": 1,
        "protocol": {
            "same_input_model_runtime": True,
            "same_execution_commit": (
                k4["execution_commit"] == k12["execution_commit"]
            ),
            "scientific_config_diff": scientific_diff,
            "schema_only_k4_fields": schema_only,
            "comparison_status": (
                "descriptive historical comparison, not a same-commit paired "
                "single-variable causal ablation"
            ),
        },
        "k4_public_profile": k4,
        "k12_historical": k12,
        "comparison": comparison,
        "decision": {
            "profile_completed_200_trials": k4_terminal["n_trials"] == 200,
            "purpose_ledger_closed": True,
            "unattributed_force_evaluations": 0,
            "k4_reduced_total_force_evaluations": (
                comparison["total_force_evaluations_saved"] > 0
            ),
            "k4_improved_final_best_energy": (
                comparison["best_energy_delta_eV"] < 0
            ),
            "k4_improved_wall_time": comparison["wall_time_delta_s"] < 0,
            "promote_direction_posterior_now": False,
            "next_algorithmic_target": (
                "landing-quality-aware direction construction/selection under "
                "a frozen cost budget; do not add a posterior selector until "
                "out-of-sample prediction beats the fixed physics baseline"
            ),
        },
    }


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k4-dir", type=Path, default=DEFAULT_K4_DIR)
    parser.add_argument("--k12-dir", type=Path, default=DEFAULT_K12_DIR)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = analyze(args.k4_dir, args.k12_dir)
    _atomic_write(args.output, evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
