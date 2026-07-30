#!/usr/bin/env python3
"""Run the preregistered choice-aligned LS survivor gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence


RUN_ROOT = Path(__file__).resolve().parent
BASE_RUNNER = RUN_ROOT / "run_gate.py"
BASE_EVIDENCE = RUN_ROOT / "output" / "evidence.json"


def _load_base():
    spec = importlib.util.spec_from_file_location("_ls_scope_gate_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AlignedConfigBuilder:
    def __init__(self, base_builder) -> None:
        self.base_builder = base_builder

    def build_production_config(self, system, case_directory, *, master_seed):
        return replace(
            self.base_builder.build_production_config(
                system,
                case_directory,
                master_seed=master_seed,
            ),
            choice_aligned_softening_enabled=True,
        )


def summarize_alignment(baseline_rows, aligned_rows) -> dict[str, Any]:
    baseline = {
        (row["system"], row["state_id"], row["seed"]): row
        for row in baseline_rows
        if row["scope"] == "both"
    }
    aligned = {
        (row["system"], row["state_id"], row["seed"]): row
        for row in aligned_rows
    }
    if baseline.keys() != aligned.keys() or len(aligned) != 18:
        raise ValueError("alignment cohort does not match the 18 baseline blocks")
    paired = []
    for key in sorted(aligned):
        current = baseline[key]
        candidate = aligned[key]
        if (
            current["state_sha256"] != candidate["state_sha256"]
            or current["directions"][0]["direction_sha256"]
            != candidate["directions"][0]["direction_sha256"]
        ):
            raise ValueError("starter or first-direction pairing drifted")
        direction_steps = len(candidate["directions"])
        builds = len(candidate["softening_builds"])
        paired.append(
            {
                "system": key[0],
                "state_id": key[1],
                "seed": key[2],
                "landing_effect_eV": (
                    float(candidate["landing_delta_eV"])
                    - float(current["landing_delta_eV"])
                ),
                "force_evaluation_effect": (
                    int(candidate["force_evaluations"])
                    - int(current["force_evaluations"])
                ),
                "force_evaluation_ratio": (
                    float(candidate["force_evaluations"])
                    / float(current["force_evaluations"])
                ),
                "direction_steps": direction_steps,
                "softening_builds": builds,
                "choice_aligned_rebuilds": builds - direction_steps,
            }
        )
    systems = {}
    for system in ("c60", "pdo"):
        rows = [row for row in paired if row["system"] == system]
        landing = [float(row["landing_effect_eV"]) for row in rows]
        ratios = [float(row["force_evaluation_ratio"]) for row in rows]
        systems[system] = {
            "pairs": len(rows),
            "improved_pairs": sum(value < 0.0 for value in landing),
            "tied_pairs": sum(abs(value) <= 1.0e-8 for value in landing),
            "mean_landing_effect_eV": statistics.mean(landing),
            "median_landing_effect_eV": statistics.median(landing),
            "median_force_evaluation_ratio": statistics.median(ratios),
            "total_choice_aligned_rebuilds": sum(
                int(row["choice_aligned_rebuilds"]) for row in rows
            ),
        }
    survives = all(
        result["median_landing_effect_eV"] <= 0.0
        and result["improved_pairs"] >= 5
        and result["median_force_evaluation_ratio"] <= 1.10
        for result in systems.values()
    )
    return {
        "schema_version": 1,
        "systems": systems,
        "survives": survives,
        "paired_cases": paired,
        "claim_ceiling": (
            "fixed-starter single-change survivor gate; no production default "
            "changes without a later equal-budget full-search experiment"
        ),
    }


def run(output: Path, expected_git_commit: str) -> dict[str, Any]:
    base = _load_base()
    manifest = base.preflight(expected_git_commit)
    if not BASE_EVIDENCE.is_file():
        raise FileNotFoundError(BASE_EVIDENCE)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest.update(
        {
            "arm": "both_choice_aligned",
            "single_change": {
                "choice_aligned_softening_enabled": [False, True],
            },
            "baseline_evidence": str(BASE_EVIDENCE),
        }
    )
    base._write_json(output / "manifest.json", manifest)
    baseline_evidence = json.loads(BASE_EVIDENCE.read_text(encoding="utf-8"))
    calculator, _ = base._calculator()
    config_builder = AlignedConfigBuilder(
        base._load_module(base.CONFIG_RUNNER, "_ls_alignment_config_builder")
    )
    rows = []
    for system in base.SYSTEMS:
        for state_id, state_path in base.STATE_FILES[system].items():
            state = base.read_state(state_path)
            for seed in base.SEEDS:
                case_directory = output / system / state_id / f"seed-{seed}"
                row = base._run_case(
                    system=system,
                    state_id=state_id,
                    state=state,
                    seed=seed,
                    scope="both",
                    calculator=calculator,
                    config_builder=config_builder,
                    case_directory=case_directory,
                )
                row["scope"] = "both_choice_aligned"
                base._write_json(case_directory / "summary.json", row)
                rows.append(row)
                print(
                    f"{system} {state_id} seed={seed} "
                    f"FE={row['force_evaluations']} "
                    f"dE={row['landing_delta_eV']:.6f}"
                )
    evidence = summarize_alignment(baseline_evidence["cases"], rows)
    evidence["execution_commit"] = expected_git_commit
    evidence["cases"] = rows
    base._write_json(output / "evidence.json", evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args(argv)
    evidence = run(args.output, args.expected_git_commit)
    print(json.dumps(evidence["systems"], indent=2, sort_keys=True))
    print(f"survives={evidence['survives']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

