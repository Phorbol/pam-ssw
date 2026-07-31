#!/usr/bin/env python3
"""Equal-budget CuO gate for current both-scope LS versus oracle-only LS."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from pamssw.accounting import EvaluationPurpose
from pamssw.io import write_state
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_GATE = (
    RUN_ROOT.parent / "20260731-starter-selection-mechanism-gate" / "run_gate.py"
)
SCOPES = ("both", "oracle")
SEED = 42
STARTER_MODE = "metropolis_chain"
DEFAULT_FORCE_BUDGET = 20_000


def _load_source_gate():
    name = "_cuo_oracle_scope_source_gate"
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, SOURCE_GATE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source gate: {SOURCE_GATE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_scope(
    *, gate, resources, shared_bootstrap, scope: str, case_directory: Path, force_budget: int
) -> dict:
    remaining = force_budget - shared_bootstrap.counts.total
    config = replace(
        gate.build_config(
            "cuo",
            case_directory,
            seed=SEED,
            starter_mode=STARTER_MODE,
            force_budget=remaining,
        ),
        local_softening_scope=scope,
    )
    walker = SurfaceWalker(
        calculator=gate._calculator("cuo", resources),
        config=config,
        softening_enabled=True,
    )
    started = perf_counter()
    result = walker.run(
        shared_bootstrap.result.state,
        prequenched_initial=shared_bootstrap.result,
    )
    search_wall_time_s = perf_counter() - started
    search_counts = walker.calculator.snapshot()
    counts = shared_bootstrap.counts + search_counts
    if search_counts.total != int(result.stats["force_evaluations"]):
        raise RuntimeError("search force-evaluation ledger does not close")
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("unattributed force evaluations are forbidden")
    if counts.total > force_budget:
        raise RuntimeError("scope arm exceeded its force-evaluation budget")

    stats = dict(result.stats)
    proposal_count = int(stats["proposal_relax_count"])
    line_failures = int(
        stats.get("proposal_relax_termination_line_search_failed", 0)
    )
    summary = {
        "schema_version": 1,
        "scope": scope,
        "system": "cuo",
        "seed": SEED,
        "starter_mode": STARTER_MODE,
        "campaign_force_budget": force_budget,
        "shared_bootstrap_force_evaluations": shared_bootstrap.counts.total,
        "search_force_evaluations": search_counts.total,
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "initial_energy_eV": float(shared_bootstrap.result.energy),
        "best_energy_eV": float(result.best_energy),
        "energy_drop_eV": float(shared_bootstrap.result.energy - result.best_energy),
        "completed_trials": int(stats["n_trials"]),
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.archive.duplicate_rate()),
        "proposal_relax_count": proposal_count,
        "proposal_line_search_failures": line_failures,
        "proposal_line_search_failure_rate": (
            line_failures / proposal_count if proposal_count else 0.0
        ),
        "proposal_force_evaluations": counts.count(
            EvaluationPurpose.BIASED_PROPOSAL_RELAX
        ),
        "proposal_force_evaluation_share": counts.count(
            EvaluationPurpose.BIASED_PROPOSAL_RELAX
        ) / counts.total,
        "proposal_rejected_line_trials": int(stats["proposal_relax_rejected_steps"]),
        "proposal_line_search_evaluations": int(
            stats["proposal_relax_line_search_evaluations"]
        ),
        "shared_bootstrap_wall_time_s": shared_bootstrap.wall_time_s,
        "search_wall_time_s": search_wall_time_s,
        "wall_time_s": shared_bootstrap.wall_time_s + search_wall_time_s,
        "effective_config": asdict(config),
        "stats": stats,
    }
    case_directory.mkdir(parents=True, exist_ok=False)
    write_state(case_directory / "best_minimum.xyz", result.best_state)
    _write_json(case_directory / "summary.json", summary)
    return summary


def run(*, output: Path, expected_commit: str, force_budget: int) -> None:
    import torch

    if _git_commit() != expected_commit:
        raise RuntimeError("execution commit does not match --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    gate = _load_source_gate()
    resources = gate._materialize_cuo_resources(
        gate.CUO_ARCHIVE_PATH,
        output / "cuo-input",
    )
    shared_bootstrap = gate._bootstrap_case(
        system="cuo",
        seed=SEED,
        bootstrap_directory=output / "bootstrap",
        force_budget=force_budget,
        cuo_resources=resources,
    )
    manifest = {
        "schema_version": 1,
        "execution_commit": expected_commit,
        "system": "cuo",
        "seed": SEED,
        "starter_mode": STARTER_MODE,
        "scopes": list(SCOPES),
        "force_budget_per_scope": force_budget,
        "bootstrap_protocol": "one exact true-PES minimum reused and charged to both scope arms",
        "single_change": "local_softening_scope: both -> oracle",
        "primary_metrics": [
            "proposal line-search failure rate",
            "proposal force-evaluation share",
            "completed trials",
            "best energy at equal total force budget",
        ],
    }
    _write_json(output / "manifest.json", manifest)
    rows = []
    configs = []
    for scope in SCOPES:
        summary = _run_scope(
            gate=gate,
            resources=resources,
            shared_bootstrap=shared_bootstrap,
            scope=scope,
            case_directory=output / scope,
            force_budget=force_budget,
        )
        rows.append(summary)
        configs.append(summary["effective_config"])
        _write_json(output / "evidence.json", {"manifest": manifest, "arms": rows})
        print(
            f"scope={scope} best={summary['best_energy_eV']:.9f} "
            f"trials={summary['completed_trials']} "
            f"line_fail_rate={summary['proposal_line_search_failure_rate']:.6f} "
            f"proposal_FE={summary['proposal_force_evaluations']}"
        )
    config_differences = {
        key: [configs[0][key], configs[1][key]]
        for key in configs[0]
        if configs[0][key] != configs[1][key]
    }
    artifact_fields = {
        key
        for key in config_differences
        if key.endswith(("_path", "_dir", "_log"))
    }
    algorithmic_differences = {
        key: value
        for key, value in config_differences.items()
        if key not in artifact_fields
    }
    if algorithmic_differences != {
        "local_softening_scope": ["both", "oracle"]
    }:
        raise RuntimeError(
            f"scope arms have confounded algorithm configs: {algorithmic_differences}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--force-budget", type=int, default=DEFAULT_FORCE_BUDGET)
    args = parser.parse_args()
    run(
        output=args.output,
        expected_commit=args.expected_commit,
        force_budget=args.force_budget,
    )


if __name__ == "__main__":
    main()
