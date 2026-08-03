#!/usr/bin/env python3
"""Analyze the preregistered zero-new-FE initial K4 HVP replay."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
DEFAULT_REPEAT_EVIDENCE = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-candidate-counterfactual-gate"
    / "repeat_evidence.json"
)
DEFAULT_ACTION_EVIDENCE = (
    REPO_ROOT
    / "runs"
    / "20260731-action-family-transfer-gate"
    / "output"
    / "evidence.json"
)


def _load_protocol():
    spec = importlib.util.spec_from_file_location(
        "_hvp_value_of_information_protocol_analysis",
        PROTOCOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load protocol from {PROTOCOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()


def _read_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _render(evidence: Mapping[str, Any]) -> str:
    analysis = evidence["analysis"]
    gate = analysis["gate"]
    lines = [
        "# Initial K4 HVP value-of-information result",
        "",
        "## Decision",
        "",
        (
            "- Live selected-only-HVP gate allowed: "
            f"**{gate['live_selected_only_hvp_gate_allowed']}**."
        ),
        (
            "- Initial all-candidate HVP deletion supported: "
            f"**{gate['initial_all_candidate_hvp_deletion_supported']}**."
        ),
        "- New force evaluations: **0**.",
        f"- Scope: {analysis['scope']}.",
        "",
        "## Paired shared-pool readout",
        "",
        "| System | Rule | Median regret (eV) | Median projected FE | Δ regret vs K4 (eV) | Δ FE vs K4 | Pass |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for system, strategies in analysis["by_system"].items():
        baseline = strategies["static_k4"]
        lines.append(
            f"| {system} | static_k4 | "
            f"{baseline['median_terminal_regret_eV']:.6f} | "
            f"{baseline['median_projected_force_evaluations']:.1f} | "
            "0.000000 | 0.0 | baseline |"
        )
        for strategy in protocol.NO_HVP_STRATEGIES:
            row = strategies[strategy]
            decision = gate[strategy]["by_system"][system]
            lines.append(
                f"| {system} | {strategy} | "
                f"{row['median_terminal_regret_eV']:.6f} | "
                f"{row['median_projected_force_evaluations']:.1f} | "
                f"{decision['quality_delta_eV']:.6f} | "
                f"{decision['projected_force_evaluation_delta']:.1f} | "
                f"{decision['pass']} |"
            )
    lines.extend(
        [
            "",
            "## Independent live D0/K4 anchor",
            "",
            "| System | Pairs | Median D0−K4 landing ΔE (eV) | Median D0−K4 FE |",
            "|---|---:|---:|---:|",
        ]
    )
    for system, row in evidence["d0_k4_live_anchor"].items():
        lines.append(
            f"| {system} | {row['pair_count']} | "
            f"{row['median_d0_minus_k4_landing_delta_eV']:.6f} | "
            f"{row['median_d0_minus_k4_force_evaluations']:.1f} |"
        )
    lines.extend(
        [
            "",
            "The D0 comparison is kept separate because its exact-anchor action is",
            "not one of the four shared K4 candidates. The replay changes only the",
            "initial candidate-selection rule; all continuation direction searches,",
            "biased relaxations, and true quenches are inherited unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    repeat_evidence_path: Path,
    action_evidence_path: Path,
) -> dict[str, Any]:
    repeat = _read_json(repeat_evidence_path)
    action = _read_json(action_evidence_path)
    consolidated = protocol.consolidate_repeats(repeat["candidate_repeats"])
    analysis = protocol.analyze_candidate_repeats(consolidated)
    evidence = {
        "schema_version": 1,
        "new_force_evaluations": 0,
        "repeat_evidence_path": str(
            repeat_evidence_path.relative_to(REPO_ROOT)
        ),
        "repeat_evidence_sha256": _digest(repeat_evidence_path),
        "action_evidence_path": str(
            action_evidence_path.relative_to(REPO_ROOT)
        ),
        "action_evidence_sha256": _digest(action_evidence_path),
        "analysis": analysis,
        "d0_k4_live_anchor": protocol.summarize_d0_k4_live_cases(
            action["cases"]
        ),
    }
    _write_json(RUN_ROOT / "evidence.json", evidence)
    (RUN_ROOT / "conclusion.md").write_text(
        _render(evidence),
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat-evidence",
        type=Path,
        default=DEFAULT_REPEAT_EVIDENCE,
    )
    parser.add_argument(
        "--action-evidence",
        type=Path,
        default=DEFAULT_ACTION_EVIDENCE,
    )
    args = parser.parse_args()
    evidence = run(
        repeat_evidence_path=args.repeat_evidence.resolve(),
        action_evidence_path=args.action_evidence.resolve(),
    )
    print(json.dumps(evidence["analysis"]["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
