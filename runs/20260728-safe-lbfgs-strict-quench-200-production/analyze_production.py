#!/usr/bin/env python3
"""Validate the strict-quench 200-step runs and their frozen baseline.

The comparison is deliberately descriptive.  A strict quench changes the
relaxed landing point, so this script first proves ledger/certificate identity
and then reports the resulting, potentially divergent, SSW trajectories.
"""

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
REPO_ROOT = RUN_ROOT.parents[1]
DEFAULT_STRICT_ROOT = RUN_ROOT
DEFAULT_BASELINE_ROOT = RUN_ROOT
SYSTEMS = ("c60", "pdo")
INPUT_PATHS_REPO_RELATIVE = {
    "c60": Path("runs/20260428-c60-mace-production/prerelaxed_c60.xyz"),
    "pdo": Path("PdO.xyz"),
}
BASE_RUNNER_PATH_REPO_RELATIVE = Path(
    "runs/20260728-safe-lbfgs-200-production/run_production.py"
)
STRICT_EXECUTION_COMMIT = "32980ad9154ec6481b310477dd7b85597cefd49a"
STRICT_CONFIG_DIFFS = {
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
REQUIRED_RAW_FILES = (
    "summary.json",
    "energy_trace.json",
    "walk_records.json",
    "optimizer_diagnostics.json",
)
IDENTITY_KEYS = (
    "input_path",
    "input_sha256",
    "model_path",
    "model_sha256",
    "runtime_versions",
    "cuda",
    "calculator",
    "safe_lbfgs_default_history_limit",
)
DERIVED_OUTPUT_CONFIG_FIELDS = (
    "accepted_structures_dir",
    "accepted_structures_log",
    "direction_diagnostics_path",
)
EXPECTED_TRANSITION_COMPLETENESS = {
    "strict": {
        "c60": {
            "trials": 200,
            "recorded_trials": 200,
            "unlogged_trials": 0,
            "fragment_rejections": 0,
        },
        "pdo": {
            "trials": 200,
            "recorded_trials": 200,
            "unlogged_trials": 0,
            "fragment_rejections": 0,
        },
    },
    "baseline": {
        "c60": {
            "trials": 200,
            "recorded_trials": 199,
            "unlogged_trials": 1,
            "fragment_rejections": 1,
        },
        "pdo": {
            "trials": 200,
            "recorded_trials": 200,
            "unlogged_trials": 0,
            "fragment_rejections": 0,
        },
    },
}


def sha256_file(path: Path) -> str:
    """Return a stable hash for a persisted artifact."""

    return sha256(Path(path).read_bytes()).hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def content_sha256_excluding_self(payload: Mapping[str, Any]) -> str:
    """Hash evidence after removing its recursively defined self-hash field."""

    copied = json.loads(json.dumps(payload, allow_nan=False))
    provenance = copied.get("artifact_provenance")
    if isinstance(provenance, dict):
        provenance.pop("content_sha256_excluding_self", None)
    return sha256(_canonical_json(copied)).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(text)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


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


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _raw_case(case_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[Any], list[Any], dict[str, str]]:
    paths = {name: case_dir / name for name in REQUIRED_RAW_FILES}
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"missing raw artifact: {path}")
    summary = _mapping(_read_json(paths["summary.json"]), "summary")
    trace = _list(_read_json(paths["energy_trace.json"]), "energy trace")
    walk_records = _list(_read_json(paths["walk_records.json"]), "walk records")
    diagnostics = _mapping(
        _read_json(paths["optimizer_diagnostics.json"]), "optimizer diagnostics"
    )
    return summary, diagnostics, trace, walk_records, {
        name.removesuffix(".json"): sha256_file(path) for name, path in paths.items()
    }


def _validate_trial_artifacts(
    summary: Mapping[str, Any],
    trace: list[Any],
    walk_records: list[Any],
    system: str,
    *,
    expected_completeness: Mapping[str, int],
) -> dict[str, Any]:
    stats = _mapping(summary.get("stats"), f"{system} stats")
    if stats.get("n_trials") != 200 or stats.get("configured_max_trials") != 200:
        raise ValueError(f"{system} does not contain exactly 200 trials")
    recorded_trials = len(walk_records)
    unlogged_trials = 200 - recorded_trials
    if unlogged_trials < 0:
        raise ValueError(f"{system} contains more walk records than completed trials")
    fragment_rejections = _nonnegative_int(
        stats.get("fragment_rejections", 0), f"{system} fragment rejections"
    )
    if unlogged_trials and fragment_rejections != unlogged_trials:
        raise ValueError(f"{system} has unexplained unlogged transition records")
    if len(trace) != recorded_trials + 1:
        raise ValueError(f"{system} energy trace does not match recorded transitions")
    if summary.get("walk_records") != walk_records:
        raise ValueError(f"{system} embedded walk records differ from raw records")
    walk_trials = [
        _mapping(row, f"{system} walk record").get("trial") for row in walk_records
    ]
    trace_trials = [
        _mapping(row, f"{system} energy trace record").get("trial") for row in trace
    ]
    if walk_trials != list(range(1, recorded_trials + 1)):
        raise ValueError(f"{system} walk records are not an ordered recorded prefix")
    if trace_trials != list(range(recorded_trials + 1)):
        raise ValueError(f"{system} energy trace is not an ordered recorded prefix")

    initial = _finite_float(summary.get("initial_energy_eV"), f"{system} initial energy")
    best = _finite_float(summary.get("best_energy_eV"), f"{system} best energy")
    drop = _finite_float(summary.get("energy_drop_eV"), f"{system} energy drop")
    trace_energies = [
        _finite_float(_mapping(row, f"{system} energy trace").get("best_energy_eV"), f"{system} trace energy")
        for row in trace
    ]
    if not math.isclose(trace_energies[0], initial, abs_tol=1.0e-12):
        raise ValueError(f"{system} initial energy disagrees with the trace")
    if not math.isclose(min(trace_energies), best, abs_tol=1.0e-12):
        raise ValueError(f"{system} best energy disagrees with the trace")
    if not math.isclose(initial - best, drop, abs_tol=1.0e-12):
        raise ValueError(f"{system} energy drop does not close")
    observed_completeness = {
        "trials": 200,
        "recorded_trials": recorded_trials,
        "unlogged_trials": unlogged_trials,
        "fragment_rejections": fragment_rejections,
    }
    if set(expected_completeness) != set(observed_completeness):
        raise ValueError(f"{system} transition completeness expectation has wrong fields")
    for key, observed in observed_completeness.items():
        if expected_completeness[key] != observed:
            raise ValueError(f"{system} transition completeness mismatch: {key}")
    return {
        **observed_completeness,
        "unlogged_accounted_by": (
            {"fragment_rejections": fragment_rejections} if unlogged_trials else {}
        ),
    }


def _derived_config_diff(
    source: Mapping[str, Any], effective: Mapping[str, Any], system: str
) -> dict[str, list[Any]]:
    """Recompute the complete source-to-effective delta, never trust metadata."""

    if set(source) != set(effective):
        raise ValueError(f"{system} derived strict config diff has unequal field sets")
    return {
        key: [source[key], effective[key]]
        for key in sorted(source)
        if source[key] != effective[key]
    }


def _normalized_config_for_cross_run_comparison(
    config: Mapping[str, Any], system: str, label: str
) -> dict[str, Any]:
    """Remove only output locations before comparing the complete run config."""

    normalized = dict(config)
    for field in DERIVED_OUTPUT_CONFIG_FIELDS:
        value = normalized.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{system} {label} config has invalid derived output path: {field}")
        normalized[field] = f"<derived-output:{field}>"
    return normalized


def _assert_baseline_matches_strict_source_config(
    strict_source_config: Mapping[str, Any],
    baseline_effective_config: Mapping[str, Any],
    system: str,
) -> None:
    """Fail closed unless all non-output fields match the frozen source config."""

    normalized_source = _normalized_config_for_cross_run_comparison(
        strict_source_config, system, "strict source"
    )
    normalized_baseline = _normalized_config_for_cross_run_comparison(
        baseline_effective_config, system, "baseline"
    )
    if normalized_source != normalized_baseline:
        raise ValueError(
            f"{system} baseline config differs from strict source config outside "
            "derived output paths"
        )


def _cost(summary: Mapping[str, Any], system: str) -> dict[str, Any]:
    force_evaluations = _nonnegative_int(
        summary.get("force_evaluations"), f"{system} force_evaluations"
    )
    stats = _mapping(summary.get("stats"), f"{system} stats")
    if stats.get("force_evaluations") != force_evaluations:
        raise ValueError(f"{system} summary and stats force totals differ")
    purpose_counts = _mapping(summary.get("purpose_counts"), f"{system} purpose counts")
    normalized = {
        name: _nonnegative_int(value, f"{system} purpose_counts[{name!r}]")
        for name, value in purpose_counts.items()
    }
    purpose_total = sum(normalized.values())
    if purpose_total != force_evaluations:
        raise ValueError(f"{system} purpose counts do not close against force evaluations")
    if normalized.get("unattributed") != 0:
        raise ValueError(f"{system} has unattributed force evaluations")
    if force_evaluations == 0:
        raise ValueError(f"{system} has no force evaluations")
    return {
        "purpose_counts": normalized,
        "purpose_total": purpose_total,
        "unattributed": normalized["unattributed"],
        "purpose_fractions": {
            name: count / force_evaluations for name, count in normalized.items()
        },
    }


def _termination_counts(diagnostics: Mapping[str, Any], prefix: str) -> dict[str, int]:
    marker = f"{prefix}_termination_"
    return {
        key.removeprefix(marker): _nonnegative_int(value, key)
        for key, value in diagnostics.items()
        if key.startswith(marker)
    }


def _true_quench(
    diagnostics: Mapping[str, Any], system: str, *, strict: bool, fmax: float
) -> dict[str, Any]:
    count = _nonnegative_int(diagnostics.get("true_quench_count"), f"{system} true quench count")
    termination = _termination_counts(diagnostics, "true_quench")
    if sum(termination.values()) != count:
        if strict:
            raise ValueError(f"{system} lacks a full strict true-quench certificate")
        raise ValueError(f"{system} true-quench termination counts do not close")
    certified = termination.get("converged", 0)
    unconverged = _nonnegative_int(
        diagnostics.get("true_quench_unconverged"), f"{system} true quench unconverged"
    )
    if unconverged != count - certified:
        if strict:
            raise ValueError(f"{system} lacks a full strict true-quench certificate")
        raise ValueError(f"{system} true-quench certificate count does not close")
    max_gradient = _finite_float(
        diagnostics.get("true_quench_max_gradient"), f"{system} true quench max gradient"
    )
    attempts = _nonnegative_int(
        diagnostics.get("quench_fallback_attempts", 0), f"{system} fallback attempts"
    )
    fallback_converged = _nonnegative_int(
        diagnostics.get("quench_fallback_converged", 0), f"{system} fallback converged"
    )
    if fallback_converged > attempts:
        raise ValueError(f"{system} fallback convergence exceeds fallback attempts")
    if strict:
        if count != 201 or certified != 201 or unconverged != 0:
            raise ValueError(f"{system} lacks a full strict true-quench certificate")
        if max_gradient > fmax + 1.0e-12:
            raise ValueError(f"{system} strict true-quench certificate exceeds fmax")
    return {
        "count": count,
        "certified": certified,
        "unconverged": unconverged,
        "termination_counts": termination,
        "max_gradient_eV_per_angstrom": max_gradient,
        "fallback_counts": {"attempts": attempts, "converged": fallback_converged},
    }


def _identity(summary: Mapping[str, Any], system: str) -> dict[str, Any]:
    identity = {}
    for key in IDENTITY_KEYS:
        if key not in summary:
            raise ValueError(f"{system} is missing provenance key {key}")
        if key != "input_path":
            identity[key] = summary[key]
    identity["input_path_repo_relative"] = _known_repo_relative_path(
        summary["input_path"],
        INPUT_PATHS_REPO_RELATIVE[system],
        f"{system} input",
    )
    return identity


def _known_repo_relative_path(value: Any, expected: Path, label: str) -> str:
    """Normalize a recorded checkout path only when its stable suffix matches."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path must be a non-empty string")
    recorded_parts = Path(value).parts
    expected_parts = expected.parts
    if len(recorded_parts) < len(expected_parts) or tuple(
        recorded_parts[-len(expected_parts) :]
    ) != tuple(expected_parts):
        raise ValueError(f"{label} path does not match expected repository location")
    return expected.as_posix()


def _normalized_base_runner(value: Any, system: str) -> dict[str, str]:
    base_runner = _mapping(value, f"{system} strict base runner")
    sha = base_runner.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise ValueError(f"{system} strict base runner SHA is invalid")
    return {
        "path_repo_relative": _known_repo_relative_path(
            base_runner.get("path"),
            BASE_RUNNER_PATH_REPO_RELATIVE,
            f"{system} strict base runner",
        ),
        "sha256": sha,
    }


def _case_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    stats = _mapping(summary["stats"], "stats")
    return {
        "initial_energy_eV": _finite_float(summary["initial_energy_eV"], "initial energy"),
        "best_energy_eV": _finite_float(summary["best_energy_eV"], "best energy"),
        "energy_drop_eV": _finite_float(summary["energy_drop_eV"], "energy drop"),
        "force_evaluations": _nonnegative_int(summary["force_evaluations"], "force evaluations"),
        "wall_time_s": _finite_float(
            _mapping(summary["timing"], "timing").get("total_wall_time_s"), "wall time"
        ),
        "archive_entries": _nonnegative_int(stats.get("n_minima"), "archive entries"),
        "duplicate_rate": _finite_float(stats.get("duplicate_rate"), "duplicate rate"),
    }


def _strict_case(strict_root: Path, system: str) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = strict_root / f"output-{system}-seed42"
    summary, diagnostics, trace, walk_records, raw_sha256 = _raw_case(case_dir)
    if summary.get("system") != system:
        raise ValueError(f"{system} strict summary identity mismatch")
    if summary.get("execution_commit") != STRICT_EXECUTION_COMMIT:
        raise ValueError(f"{system} strict execution commit mismatch")
    if summary.get("optimizer_telemetry") != diagnostics:
        raise ValueError(f"{system} strict optimizer telemetry differs from raw diagnostics")
    trial_metadata = _validate_trial_artifacts(
        summary,
        trace,
        walk_records,
        system,
        expected_completeness=EXPECTED_TRANSITION_COMPLETENESS["strict"][system],
    )
    cost = _cost(summary, system)
    config = _mapping(summary.get("effective_config"), f"{system} strict effective config")
    required_config = {
        "max_trials": 200,
        "rng_seed": 42,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "quench_optimizer": "ase-lbfgs",
        "quench_fmax": 0.01,
        "quench_maxiter": 400,
        "quench_fallback_optimizer": "ase-fire",
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise ValueError(f"{system} strict effective config mismatch: {key}")
    quench = _true_quench(diagnostics, system, strict=True, fmax=0.01)
    wrapper = _mapping(summary.get("strict_quench_wrapper"), f"{system} strict wrapper")
    if wrapper.get("schema_version") != 1:
        raise ValueError(f"{system} strict wrapper schema mismatch")
    source_config = _mapping(wrapper.get("source_config"), f"{system} source config")
    wrapper_effective = _mapping(
        wrapper.get("effective_config"), f"{system} wrapper effective config"
    )
    if wrapper_effective != config:
        raise ValueError(f"{system} wrapper effective config differs from summary")
    derived_diff = _derived_config_diff(source_config, wrapper_effective, system)
    if derived_diff != wrapper.get("config_diff"):
        raise ValueError(f"{system} derived strict config diff differs from stored diff")
    if derived_diff != STRICT_CONFIG_DIFFS[system]:
        raise ValueError(f"{system} derived strict config diff exceeds allowed diff")
    expected_fallback = quench["fallback_counts"]
    if wrapper.get("fallback_counts") != expected_fallback:
        raise ValueError(f"{system} strict wrapper fallback counts mismatch")
    base_preflight = _mapping(wrapper.get("base_preflight"), f"{system} base preflight")
    for key in IDENTITY_KEYS:
        if key not in base_preflight or summary.get(key) != base_preflight[key]:
            raise ValueError(f"{system} strict wrapper provenance mismatch: {key}")
    if base_preflight.get("execution_commit") != STRICT_EXECUTION_COMMIT:
        raise ValueError(f"{system} strict base preflight commit mismatch")
    return {
        "case_dir": case_dir.name,
        "raw_sha256": raw_sha256,
        "execution_commit": summary["execution_commit"],
        "identity": _identity(summary, system),
        "effective_config": {
            key: config[key] for key in required_config
        },
        "config_diff": derived_diff,
        "strict_quench_wrapper": {
            "base_runner": _normalized_base_runner(wrapper.get("base_runner"), system),
            "fallback_counts": wrapper["fallback_counts"],
            "overrides": wrapper.get("overrides"),
        },
        **trial_metadata,
        "walk_records": trial_metadata["recorded_trials"],
        "cost": cost,
        "true_quench": quench,
        **_case_metrics(summary),
    }, source_config


def _baseline_case(
    baseline_root: Path, system: str, *, strict_source_config: Mapping[str, Any]
) -> dict[str, Any]:
    case_dir = baseline_root / f"output-baseline-current-{system}-seed42"
    summary, diagnostics, trace, walk_records, raw_sha256 = _raw_case(case_dir)
    if summary.get("system") != system:
        raise ValueError(f"{system} baseline summary identity mismatch")
    if summary.get("execution_commit") != STRICT_EXECUTION_COMMIT:
        raise ValueError(f"{system} baseline execution commit mismatch")
    if summary.get("optimizer_telemetry") != diagnostics:
        raise ValueError(f"{system} baseline optimizer telemetry differs from raw diagnostics")
    trial_metadata = _validate_trial_artifacts(
        summary,
        trace,
        walk_records,
        system,
        expected_completeness=EXPECTED_TRANSITION_COMPLETENESS["baseline"][system],
    )
    cost = _cost(summary, system)
    config = _mapping(summary.get("effective_config"), f"{system} baseline config")
    _assert_baseline_matches_strict_source_config(strict_source_config, config, system)
    expected = {
        "max_trials": 200,
        "rng_seed": 42,
        "proposal_optimizer": "safe-lbfgs-total",
        "proposal_fmax": 0.05,
        "quench_optimizer": "scipy-lbfgsb",
        "quench_fmax": 0.01 if system == "c60" else 0.03,
        "quench_maxiter": 400,
        "quench_fallback_optimizer": None,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{system} baseline config mismatch: {key}")
    quench = _true_quench(diagnostics, system, strict=False, fmax=expected["quench_fmax"])
    if quench["fallback_counts"] != {"attempts": 0, "converged": 0}:
        raise ValueError(f"{system} baseline unexpectedly has fallback telemetry")
    return {
        "case_dir": case_dir.name,
        "raw_sha256": raw_sha256,
        "execution_commit": summary.get("execution_commit"),
        "identity": _identity(summary, system),
        "effective_config": expected,
        **trial_metadata,
        "walk_records": trial_metadata["recorded_trials"],
        "cost": cost,
        "true_quench": quench,
        **_case_metrics(summary),
    }


def _compare_case(strict: Mapping[str, Any], baseline: Mapping[str, Any], system: str) -> dict[str, Any]:
    if strict["identity"] != baseline["identity"]:
        raise ValueError(f"{system} strict/baseline input-model-runtime identity mismatch")
    delta_keys = (
        "initial_energy_eV",
        "best_energy_eV",
        "energy_drop_eV",
        "force_evaluations",
        "wall_time_s",
        "archive_entries",
        "duplicate_rate",
    )
    delta = {key: strict[key] - baseline[key] for key in delta_keys}
    scope = (
        "same quench_fmax=0.01; the intended config delta is the true-quench "
        "optimizer plus certificate-triggered FIRE fallback"
        if system == "c60"
        else "quench optimizer and quench_fmax both changed (0.03 to 0.01); "
        "this is not an optimizer-only comparison"
    )
    return {
        "strict_minus_baseline": True,
        "config_scope": scope,
        "trajectory_boundary": "SSW trajectories diverge after changed quenches; one seed is descriptive, not a causal performance estimate.",
        "configuration_identity": {
            "strict_summary_matches_wrapper_effective": True,
            "baseline_matches_strict_source_after_normalizing": list(
                DERIVED_OUTPUT_CONFIG_FIELDS
            ),
        },
        "delta": delta,
        "true_quench_certificate_delta": strict["true_quench"]["certified"]
        - baseline["true_quench"]["certified"],
        "fallback_attempt_delta": strict["true_quench"]["fallback_counts"]["attempts"]
        - baseline["true_quench"]["fallback_counts"]["attempts"],
    }


def analyze(strict_root: Path = DEFAULT_STRICT_ROOT, baseline_root: Path = DEFAULT_BASELINE_ROOT) -> dict[str, Any]:
    """Return validated strict/baseline evidence without modifying raw outputs."""

    strict_results = {
        system: _strict_case(Path(strict_root), system) for system in SYSTEMS
    }
    strict = {system: result[0] for system, result in strict_results.items()}
    baseline = {
        system: _baseline_case(
            Path(baseline_root),
            system,
            strict_source_config=strict_results[system][1],
        )
        for system in SYSTEMS
    }
    return {
        "schema_version": 1,
        "comparison_scope": {
            "systems": list(SYSTEMS),
            "seed": 42,
            "strict_execution_commit": STRICT_EXECUTION_COMMIT,
            "baseline_execution_commit": STRICT_EXECUTION_COMMIT,
            "same_commit": True,
            "interpretation": "All costs are counted force evaluations; wall time is recorded separately and is not a substitute for FE accounting.",
        },
        "strict_production": strict,
        "baseline_production": baseline,
        "comparison": {
            system: _compare_case(strict[system], baseline[system], system)
            for system in SYSTEMS
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    """Render a decision note that preserves the causal boundary."""

    lines = [
        "# Strict-quench 200-step production conclusion",
        "",
        "严格组与 baseline 均来自同一 frozen base execution commit `32980ad9154ec6481b310477dd7b85597cefd49a`、同一输入/模型/runtime；每条 CUDA/MACE 轨迹均执行 200 个 macro trials 和 201 次 true quench。purpose ledger 均精确闭合到总 force evaluations，且 `unattributed=0`。严格证书指终止计数为 `converged=201`、`unconverged=0`，并满足 `max |F| <= 0.01 eV/Å`。",
        "",
        "| System | Strict drop eV | Strict FE | Strict wall s | Baseline drop eV | Baseline FE | Baseline wall s | Strict-baseline drop eV | FE delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        strict = evidence["strict_production"][system]
        baseline = evidence["baseline_production"][system]
        delta = evidence["comparison"][system]["delta"]
        lines.append(
            f"| {system.upper()} | {strict['energy_drop_eV']:.6f} | "
            f"{strict['force_evaluations']} | {strict['wall_time_s']:.1f} | "
            f"{baseline['energy_drop_eV']:.6f} | {baseline['force_evaluations']} | "
            f"{baseline['wall_time_s']:.1f} | {delta['energy_drop_eV']:.6f} | "
            f"{delta['force_evaluations']:+.0f} |"
        )
    lines.extend(["", "## Cost structure and certificates", ""])
    for system in SYSTEMS:
        strict = evidence["strict_production"][system]
        fractions = strict["cost"]["purpose_fractions"]
        quench = strict["true_quench"]
        lines.append(
            f"- {system.upper()}: certificate {quench['certified']}/{quench['count']}; "
            f"ASE-LBFGS→FIRE fallback {quench['fallback_counts']['converged']}/"
            f"{quench['fallback_counts']['attempts']}. FE shares: proposal "
            f"{_percent(fractions['biased_proposal_relax'])}, direction oracle "
            f"{_percent(fractions['direction_oracle'])}, landing true quench "
            f"{_percent(fractions['landing_true_quench'])}."
        )
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            "- **C60:** `quench_fmax=0.01` 与 baseline 相同；严格配置仅替换 true-quench optimizer 为 ASE-LBFGS，并在它未给出证书时以 FIRE 补救。新运行把证书从 40/201 提升为 201/201（6 次 fallback，均收敛），总 FE 少 1,217、wall time 少 26.1 s；但 energy drop 少 6.441956 eV、archive 少 20 个 minima。故采用 **ASE-LBFGS+certificate FIRE** 作为 certified-data/validation candidate，而不是 global-search 默认。结合 `pamssw/walker.py` 的代码审计，当前框架允许未获证书的 landing 进入 archive，导致优化器终止语义与后续搜索状态直接耦合；该判断不是由本次 JSON 证据单独推出。严格组成本最大项仍是 proposal relaxation，其次为 direction oracle。",
            "- **PdO:** baseline 在 `quench_fmax=0.03` 已经是 201/201 证书。严格运行同时改变 optimizer 和 `quench_fmax: 0.03→0.01`，FE 增加 10,872、energy drop 少 0.878601 eV；因此不能纯归因于 optimizer，拒绝把 0.01 设为 production default。干净策略是 **0.03 用于 search，只有需要严格数据的结果再 selective refine 到 0.01**；若要隔离阈值效应，再做同一 ASE-LBFGS+certificate FIRE 下 0.03 vs 0.01 的 paired threshold ablation。当前 PdO 的 proposal 与 landing true quench 是并列主要 FE 瓶颈。",
            f"- **P0 transition-dataset completeness:** C60 baseline 的 `stats.n_trials={evidence['baseline_production']['c60']['trials']}`，但只持久化了 {evidence['baseline_production']['c60']['recorded_trials']} 条 walk records 和 {evidence['baseline_production']['c60']['recorded_trials'] + 1} 个 energy-trace states；缺失的 trial 由 `fragment_rejections={evidence['baseline_production']['c60']['fragment_rejections']}` 精确解释。这次分析保留 `recorded_trials={evidence['baseline_production']['c60']['recorded_trials']}`、`unlogged_trials={evidence['baseline_production']['c60']['unlogged_trials']}`，没有伪造记录。对 posterior/credit 学习而言，失败 transition 也必须写入 action/outcome log。PdO 为 {evidence['baseline_production']['pdo']['recorded_trials']}/{evidence['baseline_production']['pdo']['trials']}，`fragment_rejections={evidence['baseline_production']['pdo']['fragment_rejections']}`。",
            "- 两个系统在更换 true-quench 方案后观测到的能量与归档指标不同，后续 SSW 状态序列不能继续视为 paired trajectory comparison；且每个条件只有 seed 42。上述差分是严格核算下的描述性结果，不是 basin identity、optimizer 一般性或因果 superiority 声明。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    strict_root: Path = DEFAULT_STRICT_ROOT,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    output_root: Path = RUN_ROOT,
) -> dict[str, Any]:
    """Persist compact evidence and conclusion, with a non-recursive self-hash."""

    evidence = analyze(strict_root, baseline_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    evidence["artifact_provenance"] = {
        "analyzer_path_repo_relative": Path(__file__)
        .resolve()
        .relative_to(REPO_ROOT)
        .as_posix(),
        "analyzer_sha256": sha256_file(Path(__file__)),
    }
    evidence["artifact_provenance"]["content_sha256_excluding_self"] = (
        content_sha256_excluding_self(evidence)
    )
    evidence_path = output_root / "production_evidence.json"
    conclusion_path = output_root / "production_conclusion.md"
    _atomic_write_text(
        evidence_path,
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_text(conclusion_path, render_conclusion(evidence))
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-root", type=Path, default=DEFAULT_STRICT_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    write_outputs(args.strict_root, args.baseline_root, args.output_root)


if __name__ == "__main__":
    main()
