#!/usr/bin/env python3
"""Run the S-CR1 shared-bootstrap four-selector GPU gate."""

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
SOURCE_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-starter-selection-mechanism-gate"
    / "run_gate.py"
)
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
SYSTEMS = ("c60", "pdo", "cuo")
STARTER_MODES = (
    "uniform_archive",
    "archive_ucb",
    "metropolis_chain",
    "paired_best_uniform",
)
DEFAULT_FORCE_BUDGET = 20_000
MAX_TRIALS = 10_000


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


source = _load_module(SOURCE_RUNNER_PATH, "_paired_selector_source_runner")
protocol = _load_module(PROTOCOL_PATH, "_paired_selector_protocol_runner")


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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
    starter_mode: str,
    force_budget: int,
):
    if system not in SYSTEMS:
        raise ValueError("invalid system")
    if starter_mode not in STARTER_MODES:
        raise ValueError("invalid starter mode")
    base = source.build_config(
        system,
        Path(case_directory),
        seed=seed,
        starter_mode="uniform_archive",
        force_budget=force_budget,
    )
    updates = {
        "seed_selection_mode": starter_mode,
        "accepted_structures_log": str(
            Path(case_directory) / "accepted_structures.jsonl"
        ),
    }
    if system == "cuo":
        updates["local_softening_scope"] = "oracle"
    return replace(base, **updates)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _campaign_accepted_rows(
    path: Path,
    *,
    bootstrap_force_evaluations: int,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    projected = []
    previous = -1
    for row in rows:
        search_fe = int(row["force_evaluations"])
        if search_fe < previous:
            raise RuntimeError("accepted-structure force counts are not ordered")
        previous = search_fe
        projected.append(
            {
                **row,
                "search_force_evaluations": search_fe,
                "force_evaluations": bootstrap_force_evaluations + search_fe,
            }
        )
    return projected


def _run_case(
    *,
    system: str,
    seed: int,
    starter_mode: str,
    case_directory: Path,
    force_budget: int,
    shared_bootstrap,
    bootstrap_state_sha256: str,
    cuo_resources=None,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose
    from pamssw.walker import SurfaceWalker
    from pamssw.io import write_state

    remaining = force_budget - shared_bootstrap.counts.total
    if remaining <= 0:
        raise RuntimeError("bootstrap exhausted campaign budget")
    case_directory.mkdir(parents=True, exist_ok=False)
    config = build_config(
        system,
        case_directory,
        seed=seed,
        starter_mode=starter_mode,
        force_budget=remaining,
    )
    walker = SurfaceWalker(
        calculator=source._calculator(system, cuo_resources),
        config=config,
        softening_enabled=True,
    )
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
    accepted_rows = _campaign_accepted_rows(
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
    summary = {
        "schema_version": 1,
        "system": system,
        "seed": seed,
        "starter_mode": starter_mode,
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
        "completed_trials": int(result.stats["n_trials"]),
        "archive_entries": len(result.archive.entries),
        "duplicate_rate": float(result.archive.duplicate_rate()),
        "budget_exhausted": bool(result.stats["budget_exhausted"]),
        "search_wall_time_s": search_wall,
        "wall_time_s": shared_bootstrap.wall_time_s + search_wall,
        "effective_config": asdict(config),
        "stats": {
            **dict(result.stats),
            "force_evaluations": counts.total,
            "max_force_evals": force_budget,
        },
    }
    write_state(case_directory / "best_minimum.xyz", result.best_state)
    _write_json(case_directory / "accepted_trace.json", accepted_rows)
    _write_json(case_directory / "energy_trace.json", source._energy_trace(result))
    _write_json(case_directory / "summary.json", summary)
    return summary


def validate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    cases = list(evidence["cases"])
    expected = protocol.case_matrix(
        systems=evidence["cohort"]["systems"],
        seeds=evidence["cohort"]["seeds"],
        starter_modes=evidence["cohort"]["starter_modes"],
    )
    keys = [
        {
            "system": row["system"],
            "seed": int(row["seed"]),
            "starter_mode": row["starter_mode"],
        }
        for row in cases
    ]
    if keys != expected:
        raise ValueError("case matrix does not close")
    bootstrap_by_block = {}
    for row in cases:
        if sum(int(value) for value in row["purpose_counts"].values()) != int(
            row["force_evaluations"]
        ):
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
    decision = protocol.scr1_decision(cases)
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
    starter_modes: Sequence[str],
    force_budget: int,
    preflight_only: bool = False,
) -> dict[str, Any]:
    systems = tuple(systems)
    seeds = tuple(int(seed) for seed in seeds)
    starter_modes = tuple(starter_modes)
    protocol.case_matrix(
        systems=systems,
        seeds=seeds,
        starter_modes=starter_modes,
    )
    provenance = source._preflight(expected_commit, systems)
    provenance["starter_modes"] = list(starter_modes)
    provenance["selector_gate"] = (
        "only seed_selection_mode differs within a system/seed block; CuO "
        "uses the previously admitted oracle-only LS scope in every arm"
    )
    if preflight_only:
        return provenance
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(output_directory)
    output_directory.mkdir(parents=True)
    cuo_resources = None
    if "cuo" in systems:
        cuo_resources = source._materialize_cuo_resources(
            source.CUO_ARCHIVE_PATH,
            output_directory / "cuo-input",
        )
    cases = []
    total_started = perf_counter()
    for system in systems:
        for seed in seeds:
            bootstrap_directory = (
                output_directory / system / f"seed-{seed:08d}" / "bootstrap"
            )
            shared = source._bootstrap_case(
                system=system,
                seed=seed,
                bootstrap_directory=bootstrap_directory,
                force_budget=force_budget,
                cuo_resources=cuo_resources,
            )
            bootstrap_sha = _sha256(bootstrap_directory / "minimum.xyz")
            for mode in starter_modes:
                summary = _run_case(
                    system=system,
                    seed=seed,
                    starter_mode=mode,
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
    decision = protocol.scr1_decision(cases)
    evidence = {
        "schema_version": 1,
        "execution_commit": _git_commit(),
        "provenance": provenance,
        "cohort": {
            "systems": list(systems),
            "seeds": list(seeds),
            "starter_modes": list(starter_modes),
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
            "S-CR1 shared-bootstrap seed-45 C60/PdO/CuO fixed-budget gate; "
            "not a production or posterior-learning result"
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
    parser.add_argument("--starter-modes", nargs="+", choices=STARTER_MODES)
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
        seeds=(45,) if args.seeds is None else args.seeds,
        starter_modes=(
            STARTER_MODES if args.starter_modes is None else args.starter_modes
        ),
        force_budget=args.force_budget,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
