#!/usr/bin/env python3
"""Analyze the paired two-vector intent-refresh direction screen."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = RUN_ROOT / "run_intent_refresh_ablation.py"
BASE_ANALYZER_PATH = RUN_ROOT / "analyze_residual_ritz_ablation.py"
CANDIDATE_ARM = "continuation_intent_ritz2"
BASE_CANDIDATE_ARM = "residual_ritz2"
SYSTEMS = ("c60", "pdo")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validate_matrix(rows: Sequence[Mapping[str, Any]]) -> None:
    runner = _load(RUNNER_PATH, "_intent_refresh_protocol")
    expected = {
        (
            str(case["system"]),
            str(case["state_id"]),
            int(case["seed"]),
            str(case["arm"]),
        )
        for system in SYSTEMS
        for case in runner.case_matrix(system)
    }
    observed = {
        (
            str(row["system"]),
            str(row["state_id"]),
            int(row["seed"]),
            str(row["arm"]),
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("paired 90-action matrix is incomplete or duplicated")


def _rename_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "arm": (
            BASE_CANDIDATE_ARM
            if row["arm"] == CANDIDATE_ARM
            else row["arm"]
        ),
    }


def _relabel_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    for system in SYSTEMS:
        system_result = evidence["systems"][system]
        arm_results = system_result["arm_results"]
        arm_results[CANDIDATE_ARM] = arm_results.pop(
            BASE_CANDIDATE_ARM
        )
        system_result["candidate_vs_transport"] = system_result.pop(
            "residual_vs_transport"
        )
        system_result["candidate_vs_control"] = system_result.pop(
            "residual_vs_control"
        )
    evidence["hypothesis"] = (
        "a two-HVP Ritz solve in the span of the transported mode and "
        "the original random/bond intent supplies enough independent "
        "direction renewal to preserve landing quality"
    )
    evidence["decision"] = (
        "advance_continuation_intent_ritz2"
        if all(
            evidence["systems"][system]["advance_gate"]
            for system in SYSTEMS
        )
        else "reject_continuation_intent_ritz2"
    )
    return evidence


def analyze_cases(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_matrix(rows)
    base = _load(BASE_ANALYZER_PATH, "_intent_refresh_base_analyzer")
    return _relabel_evidence(
        base.analyze_cases([_rename_row(row) for row in rows])
    )


def analyze_raw(
    c60_raw: Mapping[str, Any],
    pdo_raw: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        {**row, "system": system}
        for system, raw in (("c60", c60_raw), ("pdo", pdo_raw))
        for row in raw.get("cases", [])
    ]
    _validate_matrix(rows)
    transformed = []
    for raw in (c60_raw, pdo_raw):
        transformed.append(
            {
                **raw,
                "cases": [
                    _rename_row(row) for row in raw.get("cases", [])
                ],
            }
        )
    base = _load(BASE_ANALYZER_PATH, "_intent_refresh_raw_analyzer")
    return _relabel_evidence(
        base.analyze_raw(transformed[0], transformed[1])
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c60-raw", type=Path, required=True)
    parser.add_argument("--pdo-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = analyze_raw(
        json.loads(args.c60_raw.read_text(encoding="utf-8")),
        json.loads(args.pdo_raw.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
