#!/usr/bin/env python3
"""Offline paired readout for completed LJ38 direction-target run folders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (25092501, 25092502)
ARMS = ("rotation", "full_per_atom")


def read_json(path: Path):
    return json.loads(path.read_text())


def rows_for(run_dir: Path):
    result = read_json(run_dir / "result.json")
    progress_path = run_dir / "progress.jsonl"
    observations = []
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            row = json.loads(line)
            cost = row.get("evaluation_requests") if row.get("kind") == "initial" else row.get("cumulative_requests")
            if cost is None:
                continue
            gate = row.get("minimum_gate") or {}
            if row.get("kind") == "initial":
                e, fmax = row.get("energy_eV"), row.get("fmax_eV_A")
                qualified = (row.get("converged") and fmax is not None and fmax <= 0.05
                             and row.get("connected_component_sizes_1p3sigma") == [38])
            else:
                e, fmax = gate.get("energy_eV"), gate.get("fmax_eV_A")
                qualified = bool(gate.get("converged") and gate.get("force_gate_0p05")
                                 and row.get("minimum_connected_component_sizes_1p3sigma") == [38])
            if qualified and e is not None:
                observations.append((int(cost), float(e)))
    return result, observations


def prefix_best(observations, budget):
    eligible = [energy for cost, energy in observations if cost <= budget]
    return min(eligible) if eligible else None


def analyze(runs_root: Path):
    discovered = {}
    for run_dir in sorted(runs_root.glob("run-*")):
        result_path = run_dir / "result.json"
        if not result_path.is_file():
            continue
        result, observations = rows_for(run_dir)
        key = (int(result["seed"]), str(result["arm"]))
        if key in discovered:
            raise ValueError(f"duplicate result for seed/arm {key}: {run_dir}")
        discovered[key] = (run_dir, result, observations)

    pairs = []
    for seed in SEEDS:
        arms = {arm: discovered.get((seed, arm)) for arm in ARMS}
        if not all(arms.values()):
            pairs.append({"seed": seed, "status": "incomplete_pair",
                          "present": {arm: arms[arm] is not None for arm in ARMS}})
            continue
        budgets = [int(arms[arm][1]["search_requests_total"]) for arm in ARMS]
        prefix = min(budgets)
        pair = {"seed": seed, "status": "paired_readout", "common_paid_request_prefix": prefix,
                "arms": {}}
        for arm in ARMS:
            path, result, observations = arms[arm]
            fresh_target = next((row for row in result.get("fresh", [])
                                 if row.get("role") == "target_candidate"), None)
            pair["arms"][arm] = {
                "run": str(path), "status": result.get("status"),
                "search_requests": result.get("search_requests_total"),
                "fresh_requests": result.get("fresh_requests"),
                "first_hit": result.get("status") == "first_hit",
                "candidate_requests": (result.get("first_candidate") or {}).get("evaluation_requests"),
                "first_hit_requests": ((result.get("first_candidate") or {}).get("evaluation_requests")
                                       if result.get("status") == "first_hit" else None),
                "candidate_energy_eV": (result.get("first_candidate") or {}).get("energy_eV"),
                "candidate_fresh_passed": None if fresh_target is None else fresh_target.get("passed"),
                "best_qualified_energy_at_shared_prefix_eV": prefix_best(observations, prefix),
                "best_energy_eV": result.get("best_energy_eV"),
                "search_wall_seconds": result.get("search_wall_seconds"),
            }
        pairs.append(pair)
    return {"status": "complete" if all(row["status"] == "paired_readout" for row in pairs) else "partial",
            "scope": "paired development results on two reused LJ38 starts; not a success-rate estimate",
            "runs_root": str(runs_root.resolve()), "pairs": pairs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.runs.resolve())
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
