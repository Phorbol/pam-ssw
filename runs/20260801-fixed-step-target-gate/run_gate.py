#!/usr/bin/env python3
"""Run the shared-bootstrap archive-scaled versus fixed-target GPU gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260801-paired-continuation-restart-gate"
    / "run_gate.py"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
SYSTEMS = ("c60", "pdo", "cuo")
TARGET_MODES = ("archive_scaled", "fixed_reference")
STARTER_MODE = "metropolis_chain"
FIXED_REFERENCE_EV = 0.8
DEFAULT_FORCE_BUDGET = 20_000


def _load_module(path: Path, name: str):
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = _load_module(SOURCE_RUNNER_PATH, "_fixed_target_source_runner")
protocol = _load_module(PROTOCOL_PATH, "_fixed_target_protocol_runner")


class TargetModeController:
    """Run-local macro-target switch retaining the wrapped trial bookkeeping."""

    def __init__(self, wrapped, *, mode: str, reference_eV: float):
        if mode not in TARGET_MODES:
            raise ValueError("invalid target mode")
        if not math.isfinite(float(reference_eV)) or reference_eV <= 0.0:
            raise ValueError("reference_eV must be finite and positive")
        self.wrapped = wrapped
        self.mode = mode
        self.reference_eV = float(reference_eV)
        self.history_eV: list[float] = []

    def target(self, archive=None) -> float:
        value = (
            float(self.wrapped.target(archive))
            if self.mode == "archive_scaled"
            else self.reference_eV
        )
        self.history_eV.append(value)
        return value

    def record_trial(self, **kwargs) -> None:
        self.wrapped.record_trial(**kwargs)

    def stats(self) -> dict[str, float | int | str]:
        values = dict(self.wrapped.stats())
        values["step_target_mode"] = self.mode
        if self.mode == "fixed_reference":
            values["adaptive_step_target"] = self.reference_eV
        return values


def _sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_config(
    system: str,
    case_directory: Path,
    *,
    seed: int,
    target_mode: str,
    force_budget: int,
):
    if system not in SYSTEMS:
        raise ValueError("invalid system")
    if target_mode not in TARGET_MODES:
        raise ValueError("invalid target mode")
    config = source.build_config(
        system,
        Path(case_directory),
        seed=seed,
        starter_mode=STARTER_MODE,
        force_budget=force_budget,
    )
    if not math.isclose(
        float(config.target_uphill_energy),
        FIXED_REFERENCE_EV,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError("frozen profile no longer uses the 0.8 eV reference")
    return config


def _run_case(
    *,
    system: str,
    seed: int,
    target_mode: str,
    case_directory: Path,
    force_budget: int,
    shared_bootstrap,
    bootstrap_state_sha256: str,
    cuo_resources=None,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.io import write_state
    from pamssw.walker import SurfaceWalker

    remaining = force_budget - shared_bootstrap.counts.total
    if remaining <= 0:
        raise RuntimeError("bootstrap exhausted campaign budget")
    case_directory.mkdir(parents=True, exist_ok=False)
    config = build_config(
        system,
        case_directory,
        seed=seed,
        target_mode=target_mode,
        force_budget=remaining,
    )
    walker = SurfaceWalker(
        calculator=source.source._calculator(system, cuo_resources),
        config=config,
        softening_enabled=True,
    )
    target_controller = TargetModeController(
        walker.step_target_controller,
        mode=target_mode,
        reference_eV=FIXED_REFERENCE_EV,
    )
    walker.step_target_controller = target_controller
    started = perf_counter()
    result = walker.run(
        shared_bootstrap.result.state,
        prequenched_initial=shared_bootstrap.result,
    )
    search_wall = perf_counter() - started
    search_counts = walker.calculator.snapshot()
    counts = shared_bootstrap.counts + search_counts
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("force ledger contains unattributed work")
    if counts.total > force_budget:
        raise RuntimeError("case exceeded force budget")
    accepted_path = Path(config.accepted_structures_log)
    accepted_rows = source._campaign_accepted_rows(
        accepted_path,
        bootstrap_force_evaluations=shared_bootstrap.counts.total,
    )
    initial_energy = float(shared_bootstrap.result.energy)
    gain_auc = protocol.gain_auc(
        initial_energy_eV=initial_energy,
        bootstrap_force_evaluations=shared_bootstrap.counts.total,
        accepted_rows=accepted_rows,
        total_force_budget=force_budget,
    )
    completed_trials = int(result.stats["n_trials"])
    target_requests = list(target_controller.history_eV)
    if len(target_requests) < completed_trials:
        raise RuntimeError("target request history is shorter than completed trials")
    summary = {
        "schema_version": 1,
        "system": system,
        "seed": seed,
        "target_mode": target_mode,
        "starter_mode": STARTER_MODE,
        "campaign_force_budget": force_budget,
        "bootstrap_energy_eV": initial_energy,
        "bootstrap_state_sha256": bootstrap_state_sha256,
        "shared_bootstrap_force_evaluations": shared_bootstrap.counts.total,
        "search_force_evaluations": search_counts.total,
        "force_evaluations": counts.total,
        "purpose_counts": counts.as_dict(),
        "initial_energy_eV": initial_energy,
        "best_energy_eV": float(result.best_energy),
        "energy_drop_eV": initial_energy - float(result.best_energy),
        "gain_auc_eV": float(gain_auc),
        "completed_trials": completed_trials,
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.archive.duplicate_rate()),
        "budget_exhausted": bool(result.stats["budget_exhausted"]),
        "search_wall_time_s": search_wall,
        "wall_time_s": shared_bootstrap.wall_time_s + search_wall,
        "target_reference_eV": FIXED_REFERENCE_EV,
        "completed_trial_target_history_eV": target_requests[:completed_trials],
        "all_target_requests_eV": target_requests,
        "effective_config": asdict(config),
        "stats": {
            **dict(result.stats),
            "force_evaluations": counts.total,
            "max_force_evals": force_budget,
            "step_target_mode": target_mode,
        },
    }
    write_state(case_directory / "best_minimum.xyz", result.best_state)
    _write_json(case_directory / "accepted_trace.json", accepted_rows)
    _write_json(case_directory / "energy_trace.json", source.source._energy_trace(result))
    _write_json(case_directory / "summary.json", summary)
    return summary


def validate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(evidence["cases"])
    expected = protocol.case_matrix(
        systems=evidence["cohort"]["systems"],
        seeds=evidence["cohort"]["seeds"],
        target_modes=evidence["cohort"]["target_modes"],
    )
    keys = [
        {
            "system": row["system"],
            "seed": int(row["seed"]),
            "target_mode": row["target_mode"],
        }
        for row in cases
    ]
    if keys != expected:
        raise ValueError("case matrix does not close")
    bootstrap_by_block = {}
    for row in cases:
        total = int(row["force_evaluations"])
        budget = int(row["campaign_force_budget"])
        if total != budget or budget != int(
            evidence["cohort"]["force_budget_per_case"]
        ):
            raise ValueError("campaign force budget does not close")
        if sum(int(value) for value in row["purpose_counts"].values()) != total:
            raise ValueError("purpose ledger does not close")
        if int(row["purpose_counts"].get("unattributed", -1)) != 0:
            raise ValueError("unattributed force evaluations present")
        block = (row["system"], int(row["seed"]))
        bootstrap = (
            row["bootstrap_state_sha256"],
            float(row["bootstrap_energy_eV"]),
            int(row["shared_bootstrap_force_evaluations"]),
        )
        previous = bootstrap_by_block.setdefault(block, bootstrap)
        if bootstrap != previous:
            raise ValueError("shared bootstrap differs within block")
    decision = protocol.cohort_decision(cases)
    if decision != evidence["decision"]:
        raise ValueError("recorded decision does not match protocol")
    return {
        "case_count": len(cases),
        "unattributed": 0,
        "shared_bootstrap_block_count": len(bootstrap_by_block),
        **decision,
    }


def run_gate(
    *,
    output_directory: Path,
    expected_commit: str,
    systems: Sequence[str],
    seeds: Sequence[int],
    target_modes: Sequence[str],
    force_budget: int,
    preflight_only: bool = False,
) -> dict[str, Any]:
    systems = tuple(systems)
    seeds = tuple(int(seed) for seed in seeds)
    target_modes = tuple(target_modes)
    protocol.case_matrix(
        systems=systems,
        seeds=seeds,
        target_modes=target_modes,
    )
    provenance = source.source._preflight(expected_commit, systems)
    provenance["target_modes"] = list(target_modes)
    provenance["starter_mode"] = STARTER_MODE
    provenance["target_gate"] = (
        "only the macro target source differs; within-walk trust, cumulative "
        "Gaussian propagation, proposal relaxation and true quench are frozen"
    )
    if preflight_only:
        return provenance
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    cuo_resources = None
    if "cuo" in systems:
        cuo_resources = source.source._materialize_cuo_resources(
            source.source.CUO_ARCHIVE_PATH,
            output_directory / "cuo-input",
        )
    cases = []
    total_started = perf_counter()
    for system in systems:
        for seed in seeds:
            bootstrap_directory = (
                output_directory / system / f"seed-{seed:08d}" / "bootstrap"
            )
            shared = source.source._bootstrap_case(
                system=system,
                seed=seed,
                bootstrap_directory=bootstrap_directory,
                force_budget=force_budget,
                cuo_resources=cuo_resources,
            )
            bootstrap_sha = _sha256(bootstrap_directory / "minimum.xyz")
            for mode in target_modes:
                summary = _run_case(
                    system=system,
                    seed=seed,
                    target_mode=mode,
                    case_directory=(
                        output_directory
                        / system
                        / f"seed-{seed:08d}"
                        / mode
                    ),
                    force_budget=force_budget,
                    shared_bootstrap=shared,
                    bootstrap_state_sha256=bootstrap_sha,
                    cuo_resources=cuo_resources,
                )
                cases.append(summary)
                _write_json(output_directory / "partial_cases.json", cases)
    decision = protocol.cohort_decision(cases)
    evidence = {
        "schema_version": 1,
        "execution_commit": _git_commit(),
        "provenance": provenance,
        "cohort": {
            "systems": list(systems),
            "seeds": list(seeds),
            "target_modes": list(target_modes),
            "starter_mode": STARTER_MODE,
            "fixed_reference_eV": FIXED_REFERENCE_EV,
            "force_budget_per_case": force_budget,
        },
        "cases": cases,
        "decision": decision,
        "aggregate": {
            "case_count": len(cases),
            "new_force_evaluations": sum(
                int(row["force_evaluations"]) for row in cases
            ),
            "wall_time_s": perf_counter() - total_started,
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "U-T1 shared-bootstrap seed-46 C60/PdO/CuO fixed-budget macro "
            "target gate; not a production or universal target result"
        ),
    }
    validate_evidence(evidence)
    _write_json(output_directory / "evidence.json", evidence)
    return evidence


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--target-modes", nargs="+", choices=TARGET_MODES)
    parser.add_argument("--force-budget", type=int, default=DEFAULT_FORCE_BUDGET)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-evidence", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check_evidence is not None:
        evidence = json.loads(args.check_evidence.read_text(encoding="utf-8"))
        print(json.dumps(validate_evidence(evidence), indent=2, sort_keys=True))
        return 0
    if args.output is None or args.expected_commit is None:
        raise SystemExit("--output and --expected-commit are required")
    evidence = run_gate(
        output_directory=args.output,
        expected_commit=args.expected_commit,
        systems=SYSTEMS if args.systems is None else args.systems,
        seeds=(46,) if args.seeds is None else args.seeds,
        target_modes=TARGET_MODES if args.target_modes is None else args.target_modes,
        force_budget=args.force_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
