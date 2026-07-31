#!/usr/bin/env python3
"""Run the zero-FE direction-feature held-out learnability gate."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = RUN_ROOT / "protocol.py"


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_direction_feature_learnability_protocol_analysis",
        PROTOCOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load learnability protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = _load(first_path)
    second = _load(second_path)
    if first["case_count"] != 48 or second["case_count"] != 48:
        raise RuntimeError("learnability gate requires two complete K4 campaigns")
    analysis = protocol.summarize(first["rows"], second["rows"])
    return {
        "schema_version": 1,
        "source_campaigns": [
            {
                "path": str(first_path),
                "execution_commit": first["execution_commit"],
                "case_count": first["case_count"],
                "force_evaluations": first[
                    "total_force_evaluations"
                ],
            },
            {
                "path": str(second_path),
                "execution_commit": second["execution_commit"],
                "case_count": second["case_count"],
                "force_evaluations": second[
                    "total_force_evaluations"
                ],
            },
        ],
        "new_force_evaluations": 0,
        "analysis": analysis,
    }


def _render(payload: Mapping[str, Any]) -> str:
    analysis = payload["analysis"]
    lines = [
        "# Direction feature learnability result",
        "",
        "## Decision",
        "",
        (
            "- Allow posterior-selector stage: "
            f"**{analysis['posterior_stage_allowed']}**."
        ),
        f"- New force evaluations: {payload['new_force_evaluations']}.",
        (
            "- Dataset: "
            f"{analysis['dataset']['averaged_candidate_rows']} candidates in "
            f"{analysis['dataset']['group_count']} shared K4 groups."
        ),
        "",
        "## Held-out results",
        "",
        "| Split | Model | Accuracy | Mean regret (eV) | Median regret (eV) |",
        "|---|---|---:|---:|---:|",
    ]
    for mode in ("group", "context", "system"):
        for model in (
            "static_score",
            "softness",
            "intent",
            "combined",
        ):
            result = analysis["validation"][mode][model]
            lines.append(
                f"| {mode} | {model} | "
                f"{result['top1_accuracy']:.3f} | "
                f"{result['mean_regret_eV']:.6f} | "
                f"{result['median_regret_eV']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Leave-system combined model",
            "",
            "| Held-out system | Accuracy | Mean regret (eV) | Median regret (eV) |",
            "|---|---:|---:|---:|",
        ]
    )
    combined = analysis["validation"]["system"]["combined"]
    for system in protocol.SYSTEMS:
        result = combined["by_system"][system]
        lines.append(
            f"| {system} | {result['top1_accuracy']:.3f} | "
            f"{result['mean_regret_eV']:.6f} | "
            f"{result['median_regret_eV']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The promotion decision is determined only by the fixed combined",
            "ridge model under leave-system-out validation. Group and context",
            "splits are diagnostics and cannot override a failed system holdout.",
            "No hyperparameter, feature, or model family was selected after",
            "observing the result.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--conclusion", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = run(args.first, args.second)
    _write_json(args.evidence, payload)
    args.conclusion.write_text(_render(payload), encoding="utf-8")
    print(json.dumps(payload["analysis"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
