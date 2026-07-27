#!/usr/bin/env python3
"""Validate and summarize the two safe-LBFGS 200-trial production runs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
DEFAULT_PRODUCTION_ROOT = RUN_ROOT / "output"
DEFAULT_HISTORICAL_ROOT = Path(
    "/mnt/d/download/trae-research-code/ssw/"
    "runs/20260504-c60-pdo-short4-100t-force-budget-multiseed/output"
)
SYSTEMS = ("c60", "pdo")
PROVENANCE_KEYS = (
    "execution_commit",
    "model_sha256",
    "runtime_versions",
    "cuda",
    "calculator",
    "safe_lbfgs_default_history_limit",
)
SYSTEM_CONFIG_DIFFERENCES = {
    "accepted_structures_dir",
    "accepted_structures_log",
    "dedup_rmsd_tol",
    "direction_diagnostics_path",
    "local_softening_active_count",
    "local_softening_cutoff_scale",
    "max_step_rms",
    "min_step_scale",
    "oracle_candidates",
    "proposal_relax_steps",
    "quench_fmax",
    "target_step_rms",
    "walk_trust_radius",
}
CORE_CONFIG = {
    "max_trials": 200,
    "max_force_evals": None,
    "rng_seed": 42,
    "max_steps_per_walk": 8,
    "proposal_optimizer": "safe-lbfgs-total",
    "quench_optimizer": "scipy-lbfgsb",
    "target_uphill_energy": 0.8,
    "proposal_fmax": 0.05,
    "local_softening_strength": 0.15,
    "local_softening_penalty": "buckingham_repulsive",
    "local_softening_xi": 0.3,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _termination_counts(stats: Mapping[str, Any], prefix: str) -> dict[str, int]:
    marker = f"{prefix}_termination_"
    return {
        key.removeprefix(marker): int(value)
        for key, value in sorted(stats.items())
        if key.startswith(marker) and int(value) != 0
    }


def _outcome_counts(stats: Mapping[str, Any], prefix: str) -> dict[str, int]:
    marker = f"{prefix}_outcome_"
    return {
        key.removeprefix(marker): int(value)
        for key, value in sorted(stats.items())
        if key.startswith(marker)
        and not key.endswith("_rate")
        and int(value) != 0
    }


def _best_improvements(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    improvements = []
    previous = float(trace[0]["best_energy_eV"])
    for row in trace[1:]:
        current = float(row["best_energy_eV"])
        if current < previous:
            improvements.append(
                {
                    "trial": int(row["trial"]),
                    "best_energy_eV": current,
                    "improvement_eV": previous - current,
                }
            )
        previous = current
    return improvements


def _validate_config(config: Mapping[str, Any], system: str) -> None:
    for key, expected in CORE_CONFIG.items():
        if config.get(key) != expected:
            raise ValueError(f"{system} effective config mismatch: {key}")
    expected_system = {
        "c60": {
            "dedup_rmsd_tol": 0.15,
            "local_softening_active_count": 3,
            "local_softening_cutoff_scale": 1.3,
            "max_step_rms": 0.15,
            "min_step_scale": 0.1,
            "oracle_candidates": 12,
            "proposal_relax_steps": 80,
            "quench_fmax": 0.01,
            "target_step_rms": 0.08,
            "walk_trust_radius": 5.0,
        },
        "pdo": {
            "dedup_rmsd_tol": 0.4,
            "local_softening_active_count": 5,
            "local_softening_cutoff_scale": 1.15,
            "max_step_rms": 0.35,
            "min_step_scale": 0.05,
            "oracle_candidates": 8,
            "proposal_relax_steps": 300,
            "quench_fmax": 0.03,
            "target_step_rms": 0.15,
            "walk_trust_radius": 4.0,
        },
    }[system]
    for key, expected in expected_system.items():
        if config.get(key) != expected:
            raise ValueError(f"{system} effective config mismatch: {key}")


def _analyze_production_case(case_dir: Path, system: str) -> dict[str, Any]:
    paths = {
        "summary": case_dir / "summary.json",
        "energy_trace": case_dir / "energy_trace.json",
        "walk_records": case_dir / "walk_records.json",
        "optimizer_diagnostics": case_dir / "optimizer_diagnostics.json",
    }
    summary = _read_json(paths["summary"])
    trace = _read_json(paths["energy_trace"])
    walk_records = _read_json(paths["walk_records"])
    diagnostics = _read_json(paths["optimizer_diagnostics"])
    stats = summary["stats"]

    if summary.get("system") != system:
        raise ValueError(f"{system} summary identity mismatch")
    if (
        stats.get("n_trials") != 200
        or stats.get("configured_max_trials") != 200
        or len(walk_records) != 200
        or len(trace) != 201
    ):
        raise ValueError(f"{system} did not provide exactly 200 trials")
    if [row.get("trial") for row in walk_records] != list(range(1, 201)):
        raise ValueError(f"{system} walk records are not 200 trials in order")
    if [row.get("trial") for row in trace] != list(range(201)):
        raise ValueError(f"{system} energy trace is not 200 trials in order")
    if summary.get("walk_records") != walk_records:
        raise ValueError(f"{system} embedded walk records differ from raw records")
    if summary.get("optimizer_telemetry") != diagnostics:
        raise ValueError(f"{system} optimizer diagnostics differ from summary")

    force_evaluations = int(summary["force_evaluations"])
    purpose_counts = {
        key: int(value) for key, value in summary["purpose_counts"].items()
    }
    if sum(purpose_counts.values()) != force_evaluations:
        raise ValueError(f"{system} purpose accounting does not close")
    if purpose_counts.get("unattributed") != 0:
        raise ValueError(f"{system} purpose accounting contains unattributed cost")
    if int(stats.get("force_evaluations")) != force_evaluations:
        raise ValueError(f"{system} force-evaluation totals disagree")
    _validate_config(summary["effective_config"], system)

    initial_energy = float(summary["initial_energy_eV"])
    best_energy = float(summary["best_energy_eV"])
    if (
        float(trace[0]["best_energy_eV"]) != initial_energy
        or min(float(row["best_energy_eV"]) for row in trace) != best_energy
        or float(summary["energy_drop_eV"]) != initial_energy - best_energy
    ):
        raise ValueError(f"{system} energy summary does not match trace")

    proposal_count = int(stats["proposal_relax_count"])
    proposal_termination = _termination_counts(stats, "proposal_relax")
    proposal_converged = proposal_termination.get("converged", 0)
    if sum(proposal_termination.values()) != proposal_count:
        raise ValueError(f"{system} proposal termination counts do not close")
    true_count = int(stats["true_quench_count"])
    true_termination = _termination_counts(stats, "true_quench")
    true_converged = true_termination.get("converged", 0)
    if sum(true_termination.values()) != true_count:
        raise ValueError(f"{system} true-quench termination counts do not close")

    phase_counts = {
        "proposal_relax": purpose_counts["biased_proposal_relax"],
        "true_quench_and_validation": (
            purpose_counts["starter_true_quench"]
            + purpose_counts["landing_true_quench"]
            + purpose_counts["post_relax_validation"]
        ),
        "direction_oracle": purpose_counts["direction_oracle"],
        "escape_true_pes_check": purpose_counts["escape_true_pes_check"],
        "bootstrap_true_quench": purpose_counts["bootstrap_true_quench"],
    }
    if sum(phase_counts.values()) != force_evaluations:
        raise ValueError(f"{system} phase cost accounting does not close")
    improvements = _best_improvements(trace)
    return {
        "raw_sha256": {name: _sha256(path) for name, path in paths.items()},
        "trials": 200,
        "initial_energy_eV": initial_energy,
        "best_energy_eV": best_energy,
        "energy_drop_eV": initial_energy - best_energy,
        "force_evaluations": force_evaluations,
        "force_evaluations_per_trial": force_evaluations / 200.0,
        "wall_time_s": float(summary["timing"]["total_wall_time_s"]),
        "archive_entries": int(stats["n_minima"]),
        "duplicate_rate": float(stats["duplicate_rate"]),
        "cost": {
            "purpose_counts": purpose_counts,
            "purpose_shares": {
                key: value / force_evaluations
                for key, value in purpose_counts.items()
            },
            "phase_counts": phase_counts,
            "phase_shares": {
                key: value / force_evaluations
                for key, value in phase_counts.items()
            },
        },
        "proposal_relaxation": {
            "count": proposal_count,
            "converged": proposal_converged,
            "failures": proposal_count - proposal_converged,
            "termination_counts": proposal_termination,
            "outcome_counts": _outcome_counts(stats, "proposal_relax"),
        },
        "true_quench": {
            "count": true_count,
            "converged": true_converged,
            "unconverged": true_count - true_converged,
            "termination_counts": true_termination,
            "outcome_counts": _outcome_counts(stats, "true_quench"),
        },
        "best_improvements": improvements,
        "final_best_first_reached_trial": improvements[-1]["trial"],
    }


def _historical_fire_case(root: Path, system: str) -> dict[str, Any]:
    case_dir = root / f"{system}_seed42_default8"
    summary_path = case_dir / "ssw_summary.json"
    trace_path = case_dir / "energy_trace.json"
    summary = _read_json(summary_path)
    trace = _read_json(trace_path)
    stats = summary["stats"]
    if (
        summary.get("system") != system
        or summary.get("seed") != 42
        or summary.get("variant") != "default8"
        or summary["config"].get("proposal_optimizer") != "ase-fire"
        or summary["config"].get("max_force_evals") != 58000
        or stats.get("force_evaluations") != 58000
        or stats.get("budget_exhausted") != 1
        or len(trace) != int(stats["n_trials"]) + 1
    ):
        raise ValueError(f"{system} historical FIRE boundary is not the frozen partial run")
    return {
        "raw_sha256": {
            "summary": _sha256(summary_path),
            "energy_trace": _sha256(trace_path),
        },
        "optimizer": "ase-fire",
        "force_cap": 58000,
        "completed_trials": int(stats["n_trials"]),
        "initial_energy_eV": float(summary["initial_energy"]),
        "best_energy_eV": float(summary["best_energy"]),
        "energy_drop_eV": float(summary["energy_drop"]),
        "wall_time_s": float(summary["elapsed_s"]),
        "archive_entries": int(summary["n_minima"]),
        "duplicate_rate": float(stats["duplicate_rate"]),
    }


def analyze(production_root: Path, historical_root: Path) -> dict[str, Any]:
    production_root = Path(production_root)
    production = {
        system: _analyze_production_case(production_root / system, system)
        for system in SYSTEMS
    }
    summaries = {
        system: _read_json(production_root / system / "summary.json")
        for system in SYSTEMS
    }
    for key in PROVENANCE_KEYS:
        if summaries["c60"].get(key) != summaries["pdo"].get(key):
            raise ValueError(f"production provenance mismatch: {key}")
    c60_config = summaries["c60"]["effective_config"]
    pdo_config = summaries["pdo"]["effective_config"]
    differences = {
        key
        for key in set(c60_config) | set(pdo_config)
        if c60_config.get(key) != pdo_config.get(key)
    }
    if differences != SYSTEM_CONFIG_DIFFERENCES:
        raise ValueError("cross-system effective config differences are not frozen")

    historical = {
        system: _historical_fire_case(Path(historical_root), system)
        for system in SYSTEMS
    }
    return {
        "schema_version": 1,
        "production_provenance": {
            key: summaries["c60"][key] for key in PROVENANCE_KEYS
        },
        "production": production,
        "historical_fire_boundary": {
            "comparison_scope": (
                "same named inputs, seed-42 default8, but unequal completed "
                "trials and budgets"
            ),
            "strict_superiority_supported": False,
            "reason": (
                "historical FIRE stopped at the 58k force cap and lacks the "
                "current commit, runtime, and artifact-hash provenance"
            ),
            "systems": historical,
        },
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Safe-LBFGS 200-step production conclusion",
        "",
        "两套任务均完成 200 个 SSW macro trials，且 purpose ledger 求和等于总 force evaluations，`unattributed=0`。",
        "",
        "| System | Initial eV | Best eV | Drop eV | Force evals | Wall s | Entries | Duplicate | Final best trial |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in SYSTEMS:
        row = evidence["production"][system]
        lines.append(
            f"| {system.upper()} | {row['initial_energy_eV']:.6f} | "
            f"{row['best_energy_eV']:.6f} | {row['energy_drop_eV']:.6f} | "
            f"{row['force_evaluations']} | {row['wall_time_s']:.1f} | "
            f"{row['archive_entries']} | {row['duplicate_rate']:.3f} | "
            f"{row['final_best_first_reached_trial']} |"
        )
    lines.extend(["", "## Optimizer and cost evidence", ""])
    for system in SYSTEMS:
        row = evidence["production"][system]
        proposal = row["proposal_relaxation"]
        quench = row["true_quench"]
        shares = row["cost"]["phase_shares"]
        trials = ", ".join(
            str(item["trial"]) for item in row["best_improvements"]
        )
        lines.extend(
            [
                f"- {system.upper()}: proposal relaxation "
                f"{proposal['converged']}/{proposal['count']} converged, "
                f"{proposal['failures']} failed; true quench "
                f"{quench['converged']}/{quench['count']} converged, "
                f"{quench['unconverged']} unconverged.",
                f"  Cost shares: proposal {_percent(shares['proposal_relax'])}, "
                f"true-quench/validation {_percent(shares['true_quench_and_validation'])}, "
                f"direction oracle {_percent(shares['direction_oracle'])}, "
                f"escape checks {_percent(shares['escape_true_pes_check'])}.",
                f"  Best-energy improvement trials: {trials}.",
            ]
        )
    lines.extend(
        [
            "",
            "C60 的 true quench 为 35/201 converged、166 unconverged，这是当前长任务最明确的瓶颈；PdO 为 201/201 converged。",
            "",
            "## Historical FIRE boundary",
            "",
        ]
    )
    for system in SYSTEMS:
        old = evidence["historical_fire_boundary"]["systems"][system]
        lines.append(
            f"- {system.upper()} FIRE default8: 58,000 force evaluations 后 "
            f"完成 {old['completed_trials']} trials，best={old['best_energy_eV']:.6f} eV，"
            f"drop={old['energy_drop_eV']:.6f} eV。"
        )
    lines.extend(
        [
            "",
            "这些 FIRE 结果是同名输入、seed-42、default8 的 58k-cap 部分轨迹，但完成 trial 数、总预算和 provenance 不完全匹配。因此它们只提供描述性边界，**不支持严格的 200-step superiority 结论**。",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    production_root: Path, historical_root: Path, output_root: Path
) -> dict[str, Any]:
    evidence = analyze(production_root, historical_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "conclusion.md").write_text(
        render_conclusion(evidence), encoding="utf-8"
    )
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--production-root", type=Path, default=DEFAULT_PRODUCTION_ROOT
    )
    parser.add_argument(
        "--historical-root", type=Path, default=DEFAULT_HISTORICAL_ROOT
    )
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    write_outputs(
        args.production_root, args.historical_root, args.output_root
    )


if __name__ == "__main__":
    main()
