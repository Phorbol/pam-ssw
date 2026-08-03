#!/usr/bin/env python3
"""Combine two complete counterfactual campaigns and enforce the repeat gate."""

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
        "_direction_candidate_repeat_protocol",
        PROTOCOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load counterfactual protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _render(payload: Mapping[str, Any]) -> str:
    repeat = payload["repeat_analysis"]
    rankers = payload["offline_ranker_gate"]
    first = repeat["first_analysis"]["by_system"]
    second = repeat["second_analysis"]["by_system"]
    lines = [
        "# Repeated shared-candidate direction gate",
        "",
        "## Result",
        "",
        (
            "- Static selector inadequacy supported: "
            f"**{repeat['static_selector_inadequacy_supported']}**."
        ),
        (
            "- Posterior selector promotion allowed: "
            f"**{repeat['posterior_promotion_allowed']}**."
        ),
        (
            f"- Stable static-winner misses: "
            f"{repeat['stable_static_winner_miss_count']}/"
            f"{repeat['group_count']} shared pools."
        ),
        (
            f"- Stable best-candidate identity: "
            f"{repeat['best_candidate_identity_stable_count']}/"
            f"{repeat['group_count']} pools."
        ),
        (
            "- Median absolute repeat difference in landing delta: "
            f"{repeat['median_landing_delta_repeat_difference_eV']:.6f} eV."
        ),
        (
            "- Maximum absolute repeat difference in landing delta: "
            f"{repeat['max_landing_delta_repeat_difference_eV']:.6f} eV."
        ),
        (
            "- Prospective true-curvature ablation allowed: "
            f"**{rankers['prospective_true_curvature_ablation_allowed']}**."
        ),
        (
            "- Direction-family posterior promotion allowed: "
            f"**{rankers['family_posterior_promotion_allowed']}**."
        ),
        (
            "- Adaptive quadratic score cost range: "
            f"{rankers['adaptive_score_energy_cost']['range_eV']:.3e} eV "
            f"around "
            f"{rankers['adaptive_score_energy_cost']['minimum_eV']:.6f} eV."
        ),
        "",
        "## Per-run system gate",
        "",
        "| System | Run 1 | Run 2 | Stable winner misses |",
        "|---|---|---|---:|",
    ]
    for system in sorted(first):
        lines.append(
            f"| {system} | {first[system]['classification']} | "
            f"{second[system]['classification']} | "
            f"{repeat['stable_static_winner_miss_by_system'][system]} |"
        )
    lines.extend(
        [
            "",
            "## Zero-extra-FE offline rankers",
            "",
            "| Ranker | Top-1 hits | Median regret (eV) | Mean regret (eV) |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "static_score",
        "inner_curvature",
        "true_curvature",
        "random_then_static",
        "loo_beta_family",
    ):
        row = rankers["overall"][name]
        lines.append(
            f"| {name} | {row['top1_hits']}/{row['group_count']} | "
            f"{row['median_regret_eV']:.6f} | "
            f"{row['mean_regret_eV']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The two mechanisms must be kept separate. Better candidates are",
            "already present in the K4 pool often enough to reject candidate",
            "generation as the sole bottleneck. However, PdO's score-ordering",
            "sign is not repeat-stable, so the preregistered cross-system gate",
            "does not yet authorize an online UCB/TS or learned selector.",
            "The leave-one-group-out beta rule selected the random family in",
            "every held group, so its apparent aggregate gain is a fixed global",
            "family prior rather than context-sensitive posterior learning.",
            "",
            "The next bounded step is therefore a prospective, fixed-budget",
            "ablation of the already-paid true-curvature ranker against the",
            "current static score. A new force probe or online posterior is not",
            "justified before that simpler physical baseline is tested.",
            "",
        ]
    )
    return "\n".join(lines)


def run(first_path: Path, second_path: Path) -> dict[str, Any]:
    first = _load(first_path)
    second = _load(second_path)
    if first["execution_commit"] != second["execution_commit"]:
        raise RuntimeError("repeat campaigns used different commits")
    if first["case_count"] != 48 or second["case_count"] != 48:
        raise RuntimeError("repeat gate requires two complete 48-case campaigns")
    repeat = protocol.summarize_repeats(
        first["rows"],
        second["rows"],
    )
    if not repeat["direction_identity_stable"]:
        raise RuntimeError("candidate direction identity changed across repeats")
    if not repeat["static_rank_stable"]:
        raise RuntimeError("static candidate ranks changed across repeats")
    key = lambda row: (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        int(row["candidate_index"]),
    )
    first_rows = {key(row): row for row in first["rows"]}
    second_rows = {key(row): row for row in second["rows"]}
    candidate_repeats = []
    for case_key in sorted(first_rows):
        left = first_rows[case_key]
        right = second_rows[case_key]
        candidate_repeats.append(
            {
                "system": case_key[0],
                "state_id": case_key[1],
                "seed": case_key[2],
                "candidate_index": case_key[3],
                "kind": left["kind"],
                "static_rank": left["static_rank"],
                "direction_sha256": left["direction_sha256"],
                "static_score_first": left["static_score"],
                "static_score_second": right["static_score"],
                "curvature_first": left["curvature"],
                "curvature_second": right["curvature"],
                "true_curvature_first": left["true_curvature"],
                "true_curvature_second": right["true_curvature"],
                "score_sigma_first": left["score_sigma"],
                "score_sigma_second": right["score_sigma"],
                "landing_delta_eV_first": left["landing_delta_eV"],
                "landing_delta_eV_second": right["landing_delta_eV"],
                "certificate_first": left["certificate"],
                "certificate_second": right["certificate"],
                "landing_geometry_valid_first": (
                    left["landing_geometry_valid"]
                ),
                "landing_geometry_valid_second": (
                    right["landing_geometry_valid"]
                ),
                "is_new_basin_first": left["is_new_basin"],
                "is_new_basin_second": right["is_new_basin"],
                "walk_termination_first": left[
                    "walk_termination_reason"
                ],
                "walk_termination_second": right[
                    "walk_termination_reason"
                ],
                "force_evaluations_first": left["force_evaluations"],
                "force_evaluations_second": right["force_evaluations"],
            }
        )
    offline_ranker_gate = protocol.evaluate_repeat_rankers(
        first["rows"],
        second["rows"],
    )
    return {
        "schema_version": 1,
        "execution_commit": first["execution_commit"],
        "campaigns": [
            {
                "path": str(first_path),
                "force_evaluations": first["total_force_evaluations"],
                "wall_time_s": first["wall_time_s"],
            },
            {
                "path": str(second_path),
                "force_evaluations": second["total_force_evaluations"],
                "wall_time_s": second["wall_time_s"],
            },
        ],
        "total_force_evaluations": (
            first["total_force_evaluations"]
            + second["total_force_evaluations"]
        ),
        "total_wall_time_s": (
            first["wall_time_s"] + second["wall_time_s"]
        ),
        "quality": {
            "certified_cases": sum(
                bool(row["certificate"])
                for row in first["rows"] + second["rows"]
            ),
            "geometry_valid_cases": sum(
                bool(row["landing_geometry_valid"])
                for row in first["rows"] + second["rows"]
            ),
            "fragmented_cases": sum(
                bool(row["fragmented"])
                for row in first["rows"] + second["rows"]
            ),
            "total_cases": len(first["rows"]) + len(second["rows"]),
        },
        "candidate_repeats": candidate_repeats,
        "repeat_analysis": repeat,
        "offline_ranker_gate": offline_ranker_gate,
    }


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
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
