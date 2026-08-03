#!/usr/bin/env python3
"""Analyze the paired cumulative/newest-only Gaussian-history gate."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


def load_module(path: Path, name: str):
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


protocol = load_module(PROTOCOL_PATH, "_bias_history_analysis_protocol")


def _escaped(row) -> bool:
    return bool(
        row["certified"]
        and not row["same_starter_basin"]
        and not row.get("budget_censored", False)
    )


def pairing_summary(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["system"], row["starter_context"], int(row["seed"]))
        grouped[key][row["operator_family"]] = row
    result = Counter(
        {
            "cumulative_only_escape": 0,
            "newest_only_escape": 0,
            "both_escape": 0,
            "neither_escape": 0,
            "cumulative_lower_energy": 0,
            "newest_lower_energy": 0,
            "energy_tie": 0,
        }
    )
    for arms in grouped.values():
        if set(arms) != set(protocol.FAMILIES):
            raise ValueError("paired action is incomplete")
        cumulative = arms["cumulative"]
        newest = arms["newest_only"]
        cumulative_escape = _escaped(cumulative)
        newest_escape = _escaped(newest)
        if cumulative_escape and newest_escape:
            result["both_escape"] += 1
        elif cumulative_escape:
            result["cumulative_only_escape"] += 1
        elif newest_escape:
            result["newest_only_escape"] += 1
        else:
            result["neither_escape"] += 1
        left = cumulative.get("landing_delta_eV")
        right = newest.get("landing_delta_eV")
        if left is None or right is None:
            continue
        if float(left) < float(right) - 1.0e-6:
            result["cumulative_lower_energy"] += 1
        elif float(right) < float(left) - 1.0e-6:
            result["newest_lower_energy"] += 1
        else:
            result["energy_tie"] += 1
    return dict(result)


def load_pairs(output: Path):
    paths = sorted(Path(output).glob("*/*/seed-*/pair.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _flatten_actions(pairs):
    return [row for pair in pairs for row in pair["actions"]]


def _purpose_ledger(pairs):
    counts = Counter()
    for pair in pairs:
        counts.update(
            {
                str(key): int(value)
                for key, value in pair["starter_validation_purpose_counts"].items()
            }
        )
        counts.update(
            {
                str(key): int(value)
                for key, value in pair["shared_prefix_purpose_counts"].items()
            }
        )
        for action in pair["actions"]:
            counts.update(
                {str(key): int(value) for key, value in action["purpose_counts"].items()}
            )
    return dict(sorted(counts.items()))


def family_purpose_counts(rows):
    result = {family: Counter() for family in protocol.FAMILIES}
    for row in rows:
        result[row["operator_family"]].update(
            {str(key): int(value) for key, value in row["purpose_counts"].items()}
        )
    return {
        family: dict(sorted(counts.items()))
        for family, counts in result.items()
    }


def family_wall_times(rows):
    return {
        family: {
            "exclusive_wall_time_s": sum(
                float(row["wall_time_s"])
                for row in rows
                if row["operator_family"] == family
            ),
            "fully_loaded_wall_time_s": sum(
                float(row["fully_loaded_wall_time_s"])
                for row in rows
                if row["operator_family"] == family
            ),
        }
        for family in protocol.FAMILIES
    }


def _system_summary(rows):
    result = {}
    for system in protocol.SYSTEMS:
        result[system] = {}
        for family in protocol.FAMILIES:
            selected = [
                row
                for row in rows
                if row["system"] == system and row["operator_family"] == family
            ]
            deltas = [
                float(row["landing_delta_eV"])
                for row in selected
                if row.get("landing_delta_eV") is not None
            ]
            costs = [
                int(row["fully_loaded_force_evaluations"])
                for row in selected
            ]
            result[system][family] = {
                "actions": len(selected),
                "certified": sum(bool(row["certified"]) for row in selected),
                "geometric_escapes": sum(_escaped(row) for row in selected),
                "median_fully_loaded_force_evaluations": median(costs) if costs else None,
                "total_fully_loaded_force_evaluations": sum(costs),
                "median_landing_delta_eV": median(deltas) if deltas else None,
                "history_reduction_active_actions": sum(
                    bool(row.get("trace", {}).get("history_reduction_active"))
                    for row in selected
                ),
            }
    return result


def _reference_comparison(pairs, reference: Path | None):
    if reference is None:
        return None
    reference_pairs = load_pairs(reference)
    reference_map = {
        (pair["system"], pair["starter_context"], int(pair["seed"])): pair
        for pair in reference_pairs
    }
    counts = Counter(
        {
            "pairs": 0,
            "starter_sha256_equal": 0,
            "direction_sha256_equal": 0,
            "execution_sigma_equal": 0,
        }
    )
    mismatches = []
    for pair in pairs:
        key = (pair["system"], pair["starter_context"], int(pair["seed"]))
        old = reference_map.get(key)
        if old is None:
            mismatches.append({"key": list(key), "reason": "missing_reference"})
            continue
        counts["pairs"] += 1
        for field in ("starter_sha256", "direction_sha256"):
            equal = pair[field] == old[field]
            counts[f"{field}_equal"] += int(equal)
            if not equal:
                mismatches.append({"key": list(key), "field": field})
        sigma_equal = abs(float(pair["execution_sigma"]) - float(old["execution_sigma"])) <= 1.0e-12
        counts["execution_sigma_equal"] += int(sigma_equal)
        if not sigma_equal:
            mismatches.append({"key": list(key), "field": "execution_sigma"})
    return {"counts": dict(counts), "mismatches": mismatches}


def analyze(output: Path, *, reference: Path | None = None):
    pairs = load_pairs(output)
    if len(pairs) != len(protocol.case_matrix()):
        raise ValueError(
            f"expected {len(protocol.case_matrix())} pairs, observed {len(pairs)}"
        )
    actions = _flatten_actions(pairs)
    decision = protocol.decide(actions)
    ledger = _purpose_ledger(pairs)
    pair_total = sum(int(pair["pair_force_evaluations"]) for pair in pairs)
    if sum(ledger.values()) != pair_total:
        raise ValueError("global purpose ledger does not close")
    if ledger.get("unattributed", 0) != 0:
        raise ValueError("global ledger contains unattributed work")
    return {
        "schema_version": 1,
        "decision": decision,
        "pairing_summary": pairing_summary(actions),
        "system_summary": _system_summary(actions),
        "purpose_counts": ledger,
        "family_exclusive_purpose_counts": family_purpose_counts(actions),
        "family_wall_times": family_wall_times(actions),
        "force_evaluations": pair_total,
        "pair_wall_time_s": sum(float(pair["pair_wall_time_s"]) for pair in pairs),
        "completed_pairs": len(pairs),
        "completed_actions": len(actions),
        "starter_preparation_mode": "shared_true_quench",
        "basin_label_mode": "geometry_primary",
        "reference_input_comparison": _reference_comparison(pairs, reference),
        "production_default_changed": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RUN_ROOT / "output")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--evidence", type=Path, default=RUN_ROOT / "evidence.json")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    evidence = analyze(args.output, reference=args.reference)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.evidence.with_name(f".{args.evidence.name}.tmp")
    temporary.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.evidence)
    print(json.dumps(evidence["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
