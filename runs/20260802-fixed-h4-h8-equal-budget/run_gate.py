#!/usr/bin/env python3
"""Run the shared-bootstrap fixed H4 versus H8 equal-budget GPU gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
OBSERVABILITY_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260801-uphill-action-observability" / "run_gate.py"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
SYSTEMS = ("c60", "pdo", "cuo")
HORIZONS = (4, 8)
SEED = 49
STARTER_MODE = "metropolis_chain"
TARGET_MODE = "archive_scaled"
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


observability = _load_module(
    OBSERVABILITY_RUNNER_PATH,
    "_fixed_h4_h8_observability_runner",
)
protocol = _load_module(PROTOCOL_PATH, "_fixed_h4_h8_protocol_runner")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_config(
    system: str,
    case_directory: Path,
    *,
    seed: int,
    horizon: int,
    force_budget: int,
):
    if horizon not in HORIZONS:
        raise ValueError("horizon must be 4 or 8")
    base = observability.build_config(
        system,
        Path(case_directory),
        seed=seed,
        force_budget=force_budget,
    )
    return replace(base, max_steps_per_walk=int(horizon))


def _strict_certificate_rate(actions: Sequence[Mapping[str, Any]]) -> float:
    attempted = [row for row in actions if row["landing_converged"] is not None]
    if not attempted:
        return 0.0
    return float(
        sum(row["landing_converged"] is True for row in attempted) / len(attempted)
    )


def _run_case(
    *,
    system: str,
    seed: int,
    horizon: int,
    case_directory: Path,
    force_budget: int,
    shared_bootstrap,
    bootstrap_state_sha256: str,
    cuo_resources=None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
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
        horizon=horizon,
        force_budget=remaining,
    )
    walker = SurfaceWalker(
        calculator=observability.base.source.source._calculator(
            system,
            cuo_resources,
        ),
        config=config,
        softening_enabled=True,
    )
    walker.step_target_controller = observability.base.TargetModeController(
        walker.step_target_controller,
        mode=TARGET_MODE,
        reference_eV=observability.base.FIXED_REFERENCE_EV,
    )
    started = perf_counter()
    result = walker.run(
        shared_bootstrap.result.state,
        prequenched_initial=shared_bootstrap.result,
    )
    search_wall_time_s = perf_counter() - started
    search_counts = walker.calculator.snapshot()
    counts = shared_bootstrap.counts + search_counts
    if counts.count(EvaluationPurpose.UNATTRIBUTED) != 0:
        raise RuntimeError("force ledger contains unattributed work")
    if counts.total > force_budget:
        raise RuntimeError("case exceeded force budget")

    action_rows = [
        observability.analysis.serialize_action(row)
        for row in result.action_history
    ]
    if not action_rows:
        raise RuntimeError("case produced no completed action record")
    action_path = case_directory / "action_history.jsonl"
    _write_jsonl(action_path, action_rows)
    action_sha256 = _sha256(action_path)
    accepted_rows = observability.base.source._campaign_accepted_rows(
        Path(config.accepted_structures_log),
        bootstrap_force_evaluations=shared_bootstrap.counts.total,
    )
    initial_energy = float(shared_bootstrap.result.energy)
    summary = {
        "schema_version": 1,
        "system": system,
        "seed": seed,
        "horizon": horizon,
        "starter_mode": STARTER_MODE,
        "target_mode": TARGET_MODE,
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
        "gain_auc_eV": protocol.gain_auc(
            initial_energy_eV=initial_energy,
            accepted_rows=accepted_rows,
            total_force_budget=force_budget,
        ),
        "completed_trials": int(result.stats["n_trials"]),
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.archive.duplicate_rate()),
        "budget_exhausted": bool(result.stats["budget_exhausted"]),
        "action_count": len(action_rows),
        "strict_landing_certificate_rate": _strict_certificate_rate(action_rows),
        "action_history_sha256": action_sha256,
        "search_wall_time_s": search_wall_time_s,
        "wall_time_s": shared_bootstrap.wall_time_s + search_wall_time_s,
        "effective_config": asdict(config),
        "stats": {
            **dict(result.stats),
            "force_evaluations": counts.total,
            "max_force_evals": force_budget,
            "step_target_mode": TARGET_MODE,
        },
    }
    summary["action_analysis"] = observability.analysis.analyze_case(
        summary,
        action_rows,
    )
    write_state(case_directory / "best_minimum.xyz", result.best_state)
    _write_json(case_directory / "accepted_trace.json", accepted_rows)
    _write_json(
        case_directory / "energy_trace.json",
        observability.base.source.source._energy_trace(result),
    )
    _write_json(case_directory / "summary.json", summary)
    return summary, action_rows, action_sha256


def validate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(evidence["cases"])
    expected = protocol.case_matrix(
        systems=evidence["cohort"]["systems"],
        seeds=evidence["cohort"]["seeds"],
        horizons=evidence["cohort"]["horizons"],
    )
    observed = [
        {
            "system": str(row["system"]),
            "seed": int(row["seed"]),
            "horizon": int(row["horizon"]),
        }
        for row in cases
    ]
    if observed != expected:
        raise ValueError("case matrix does not close")
    bootstrap_by_block: dict[tuple[str, int], tuple[Any, ...]] = {}
    config_by_block: dict[tuple[str, int], dict[int, Mapping[str, Any]]] = {}
    unused = 0
    maximum_residual = 0
    for row in cases:
        total = int(row["force_evaluations"])
        budget = int(row["campaign_force_budget"])
        counts = {str(key): int(value) for key, value in row["purpose_counts"].items()}
        if sum(counts.values()) != total:
            raise ValueError("purpose ledger does not close")
        if counts.get("unattributed", -1) != 0:
            raise ValueError("unattributed force evaluations present")
        residual = budget - total
        if residual < 0:
            raise ValueError("campaign exceeded force budget")
        if residual and not bool(row["budget_exhausted"]):
            raise ValueError("campaign stopped before budget exhaustion")
        maximum_atomic_batch = 2 * int(row["effective_config"]["oracle_candidates"])
        if residual >= maximum_atomic_batch:
            raise ValueError("budget residual exceeds one atomic HVP batch")
        unused += residual
        maximum_residual = max(maximum_residual, residual)
        block = (str(row["system"]), int(row["seed"]))
        bootstrap = (
            str(row["bootstrap_state_sha256"]),
            float(row["bootstrap_energy_eV"]),
            int(row["shared_bootstrap_force_evaluations"]),
        )
        if bootstrap_by_block.setdefault(block, bootstrap) != bootstrap:
            raise ValueError("shared bootstrap differs within block")
        config_by_block.setdefault(block, {})[int(row["horizon"])] = row[
            "effective_config"
        ]
    for configs in config_by_block.values():
        if set(configs) != set(HORIZONS):
            raise ValueError("horizon pair is incomplete")
        protocol.scientific_config_differences(configs[4], configs[8])
    decision = protocol.cohort_decision(cases)
    if decision != evidence["decision"]:
        raise ValueError("recorded decision does not match protocol")
    return {
        "case_count": len(cases),
        "shared_bootstrap_block_count": len(bootstrap_by_block),
        "unattributed": 0,
        "unused_force_evaluations": unused,
        "maximum_case_budget_residual": maximum_residual,
        **decision,
    }


def run_gate(
    *,
    output_directory: Path,
    expected_commit: str,
    systems: Sequence[str],
    seeds: Sequence[int],
    horizons: Sequence[int],
    force_budget: int,
    preflight_only: bool = False,
) -> dict[str, Any]:
    systems = tuple(systems)
    seeds = tuple(int(seed) for seed in seeds)
    horizons = tuple(int(horizon) for horizon in horizons)
    protocol.case_matrix(systems=systems, seeds=seeds, horizons=horizons)
    provenance = observability.base.source.source._preflight(
        expected_commit,
        systems,
    )
    provenance["starter_mode"] = STARTER_MODE
    provenance["target_mode"] = TARGET_MODE
    provenance["horizons"] = list(horizons)
    provenance["gate_boundary"] = (
        "only max_steps_per_walk differs; all direction, bias, trust, proposal, "
        "quench, matcher and starter mechanisms are frozen"
    )
    if preflight_only:
        return provenance
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    cuo_resources = None
    if "cuo" in systems:
        cuo_resources = observability.base.source.source._materialize_cuo_resources(
            observability.base.source.source.CUO_ARCHIVE_PATH,
            output_directory / "cuo-input",
        )
    cases = []
    total_started = perf_counter()
    for system in systems:
        for seed in seeds:
            block = output_directory / system / f"seed-{seed:08d}"
            bootstrap_directory = block / "bootstrap"
            shared = observability.base.source.source._bootstrap_case(
                system=system,
                seed=seed,
                bootstrap_directory=bootstrap_directory,
                force_budget=force_budget,
                cuo_resources=cuo_resources,
            )
            bootstrap_sha256 = _sha256(bootstrap_directory / "minimum.xyz")
            for horizon in horizons:
                summary, _, _ = _run_case(
                    system=system,
                    seed=seed,
                    horizon=horizon,
                    case_directory=block / f"h{horizon}",
                    force_budget=force_budget,
                    shared_bootstrap=shared,
                    bootstrap_state_sha256=bootstrap_sha256,
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
            "horizons": list(horizons),
            "starter_mode": STARTER_MODE,
            "target_mode": TARGET_MODE,
            "force_budget_per_case": force_budget,
        },
        "cases": cases,
        "decision": decision,
        "aggregate": {
            "case_count": len(cases),
            "new_force_evaluations": sum(int(row["force_evaluations"]) for row in cases),
            "wall_time_s": perf_counter() - total_started,
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "seed-49 C60/PdO/CuO shared-bootstrap equal-budget H4/H8 survivor "
            "gate; not statistical significance or a production-default result"
        ),
    }
    validate_evidence(evidence)
    _write_json(output_directory / "evidence.json", evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS)
    parser.add_argument("--force-budget", type=int, default=DEFAULT_FORCE_BUDGET)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-evidence", type=Path)
    args = parser.parse_args(argv)
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
        seeds=(SEED,) if args.seeds is None else args.seeds,
        horizons=HORIZONS if args.horizons is None else args.horizons,
        force_budget=args.force_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
