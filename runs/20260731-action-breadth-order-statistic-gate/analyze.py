#!/usr/bin/env python3
"""Run the exact zero-FE K4 action-breadth order-statistic gate."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any, Mapping


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-direction-candidate-counterfactual-gate"
    / "repeat_evidence.json"
)
SOURCE_SHA256 = (
    "cb19e5b4b9a8839ad917f287177251d179e6b0e662c4be6fa56a8d12bd347d5d"
)
SHARED_POOL_FORCE_EVALUATIONS = 8
CAMPAIGNS = ("first", "second")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    RUN_ROOT / "protocol.py",
    "_action_breadth_order_statistic_protocol_analysis",
)


def _source_hash() -> str:
    return sha256(SOURCE_PATH.read_bytes()).hexdigest()


def _candidate(
    row: Mapping[str, Any],
    *,
    campaign: str,
) -> dict[str, Any]:
    return {
        "candidate_index": int(row["candidate_index"]),
        "landing_delta_eV": float(row[f"landing_delta_eV_{campaign}"]),
        "force_evaluations": int(
            row[f"force_evaluations_{campaign}"]
        ),
        "valid": (
            bool(row[f"certificate_{campaign}"])
            and bool(row[f"landing_geometry_valid_{campaign}"])
        ),
        "static_rank": int(row["static_rank"]),
        "kind": str(row["kind"]),
    }


def _aggregate(
    groups: list[Mapping[str, Any]],
    *,
    campaign: str,
    system: str,
) -> dict[str, Any]:
    rows = [
        group
        for group in groups
        if group["campaign"] == campaign and group["system"] == system
    ]
    if len(rows) != 6:
        raise RuntimeError(
            f"{campaign}/{system} does not contain six shared pools"
        )
    return {
        "campaign": campaign,
        "system": system,
        "group_count": len(rows),
        "static_median_regret_eV": median(
            row["static_b1"]["regret_eV"] for row in rows
        ),
        "static_median_force_evaluations": median(
            row["static_b1"]["force_evaluations"] for row in rows
        ),
        "static_valid_probability": median(
            row["static_b1"]["valid_subset_probability"] for row in rows
        ),
        "b2_median_regret_eV": median(
            row["uniform_b2"]["expected_regret_eV"] for row in rows
        ),
        "b2_median_force_evaluations": median(
            row["uniform_b2"]["expected_force_evaluations"] for row in rows
        ),
        "b2_valid_probability": median(
            row["uniform_b2"]["valid_subset_probability"] for row in rows
        ),
        "frontier": {
            f"uniform_b{breadth}": {
                "median_expected_regret_eV": median(
                    row[f"uniform_b{breadth}"]["expected_regret_eV"]
                    for row in rows
                ),
                "median_expected_force_evaluations": median(
                    row[f"uniform_b{breadth}"][
                        "expected_force_evaluations"
                    ]
                    for row in rows
                ),
                "median_valid_subset_probability": median(
                    row[f"uniform_b{breadth}"][
                        "valid_subset_probability"
                    ]
                    for row in rows
                ),
            }
            for breadth in range(1, 5)
        },
    }


def analyze() -> dict[str, Any]:
    if _source_hash() != SOURCE_SHA256:
        raise RuntimeError("repeat evidence checksum drifted")
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = (
        defaultdict(list)
    )
    for row in source["candidate_repeats"]:
        for campaign in CAMPAIGNS:
            key = (
                campaign,
                str(row["system"]),
                str(row["state_id"]),
                int(row["seed"]),
            )
            grouped[key].append(_candidate(row, campaign=campaign))

    groups = []
    campaign_candidate_fe = {campaign: 0 for campaign in CAMPAIGNS}
    for (campaign, system, state_id, seed), candidates in sorted(
        grouped.items()
    ):
        if len(candidates) != 4:
            raise RuntimeError("shared K4 pool does not contain four candidates")
        if sorted(candidate["candidate_index"] for candidate in candidates) != [
            0,
            1,
            2,
            3,
        ]:
            raise RuntimeError("shared K4 candidate indices are incomplete")
        campaign_candidate_fe[campaign] += sum(
            candidate["force_evaluations"] for candidate in candidates
        )
        group = {
            "campaign": campaign,
            "system": system,
            "state_id": state_id,
            "seed": seed,
            "static_b1": protocol.static_b1_summary(
                candidates,
                shared_pool_force_evaluations=(
                    SHARED_POOL_FORCE_EVALUATIONS
                ),
            ),
        }
        for breadth in range(1, 5):
            group[f"uniform_b{breadth}"] = (
                protocol.uniform_subset_summary(
                    candidates,
                    breadth=breadth,
                    shared_pool_force_evaluations=(
                        SHARED_POOL_FORCE_EVALUATIONS
                    ),
                )
            )
        groups.append(group)

    campaign_ledger = {}
    for index, campaign in enumerate(CAMPAIGNS):
        reported = int(source["campaigns"][index]["force_evaluations"])
        projected = (
            campaign_candidate_fe[campaign]
            + 12 * SHARED_POOL_FORCE_EVALUATIONS
        )
        if reported != projected:
            raise RuntimeError(f"{campaign} campaign FE ledger does not close")
        campaign_ledger[campaign] = {
            "candidate_force_evaluations": campaign_candidate_fe[campaign],
            "shared_pool_force_evaluations": (
                12 * SHARED_POOL_FORCE_EVALUATIONS
            ),
            "total_force_evaluations": projected,
        }

    strata = [
        _aggregate(groups, campaign=campaign, system=system)
        for campaign in CAMPAIGNS
        for system in ("c60", "pdo")
    ]
    gate = protocol.evaluate_live_b2_gate(strata)
    evidence = {
        "schema_version": 1,
        "source_path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "source_sha256": SOURCE_SHA256,
        "new_force_evaluations": 0,
        "campaign_ledger": campaign_ledger,
        "shared_pool_count": len(groups),
        "candidate_outcome_count": len(source["candidate_repeats"]) * 2,
        "group_summaries": groups,
        "strata": strata,
        "gate": gate,
    }
    (RUN_ROOT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# K4 action-breadth order-statistic result",
        "",
        f"- Live B2 gate allowed: **{gate['live_b2_gate_allowed']}**.",
        "- New force evaluations: **0**.",
        "- Production change allowed: **False**.",
        "",
        "| Campaign | System | Static regret / FE | B2 regret / FE | Elasticity | Pass |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in gate["strata"]:
        lines.append(
            f"| {row['campaign']} | {row['system']} | "
            f"{row['static_median_regret_eV']:.6f} / "
            f"{row['static_median_force_evaluations']:.1f} | "
            f"{row['b2_median_regret_eV']:.6f} / "
            f"{row['b2_median_force_evaluations']:.1f} | "
            f"{row['benefit_cost_elasticity']:.3f} | "
            f"{row['passed']} |"
        )
    lines.extend(
        [
            "",
            "B3/B4 are diagnostics only. This gate does not claim batched GPU "
            "speedup and does not change any production default.",
            "",
        ]
    )
    (RUN_ROOT / "conclusion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return evidence


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
