#!/usr/bin/env python3
"""Audit whether low-energy starters are associated with global improvements.

This is a read-only analysis of completed uniform-starter campaigns.  It does
not estimate counterfactual outcomes for starters that were not selected.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).with_name("evidence.json")
CAMPAIGN_ROOTS = (
    ROOT / "runs/20260730-starter-cell-online-gate/production-20k-seed42-output",
    ROOT / "runs/20260730-starter-cell-online-gate/production-20k-seeds43-44-output",
)
SYSTEMS = ("c60", "pdo")
SEEDS = (42, 43, 44)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _poisson_binomial_upper_tail(probabilities: list[float], observed: int) -> float:
    distribution = [1.0]
    for probability in probabilities:
        updated = [0.0] * (len(distribution) + 1)
        for successes, mass in enumerate(distribution):
            updated[successes] += mass * (1.0 - probability)
            updated[successes + 1] += mass * probability
        distribution = updated
    return sum(distribution[observed:])


def _parse_campaign(event_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = event_path.with_name("campaign_summary.json")
    summary = json.loads(summary_path.read_text())
    system = event_path.parts[-4]
    seed = int(event_path.parts[-3].split("-")[-1])
    energies = {0: float(summary["bootstrap_energy_eV"])}
    creation_attempt = {0: -1}
    visits: Counter[int] = Counter()
    previous_landing_id = 0
    rows: list[dict[str, Any]] = []
    snapshot: dict[str, Any] | None = None

    for line in event_path.read_text().splitlines():
        record = json.loads(line)
        if record["record_type"] == "policy_snapshot":
            snapshot = record
            probabilities = record["probabilities"]
            if record["policy_name"] != "uniform" or not record["support_complete"]:
                raise ValueError(f"non-uniform policy snapshot in {event_path}")
            if probabilities and max(probabilities) - min(probabilities) > 1e-15:
                raise ValueError(f"unequal uniform probabilities in {event_path}")
            continue
        if record["record_type"] != "attempt":
            continue
        if snapshot is None or snapshot["batch_id"] != record["batch_id"]:
            raise ValueError(f"attempt without matching snapshot in {event_path}")
        if record["status"] != "completed" or record["landing_energy"] is None:
            raise ValueError(f"non-completed attempt in benchmark-eligible {event_path}")
        if not record["cost_is_exact"] or record["evaluation_counts"]["unattributed"] != 0:
            raise ValueError(f"inexact force accounting in {event_path}")

        starter_id = int(record["starter_id"])
        if starter_id not in energies:
            raise ValueError(f"unknown starter {starter_id} in {event_path}")
        ordered = sorted(energies, key=lambda entry_id: (energies[entry_id], entry_id))
        archive_size = len(ordered)
        rank = ordered.index(starter_id)
        rank_fraction = rank / max(1, archive_size - 1)
        pre_action_best = energies[ordered[0]]
        landing_energy = float(record["landing_energy"])
        attempt_index = len(rows)
        row = {
            "system": system,
            "seed": seed,
            "attempt_index": attempt_index,
            "archive_size": archive_size,
            "starter_id": starter_id,
            "starter_energy_eV": energies[starter_id],
            "pre_action_best_energy_eV": pre_action_best,
            "starter_energy_gap_eV": energies[starter_id] - pre_action_best,
            "starter_energy_rank": rank,
            "starter_energy_rank_fraction": rank_fraction,
            "starter_is_current_best": rank == 0,
            "starter_is_top_energy_quartile": rank <= math.floor((archive_size - 1) * 0.25),
            "starter_is_previous_landing": starter_id == previous_landing_id,
            "starter_is_latest_new_entry": creation_attempt[starter_id] == attempt_index - 1,
            "starter_prior_visits": visits[starter_id],
            "landing_energy_eV": landing_energy,
            "global_improvement": landing_energy < pre_action_best - 1e-6,
            "global_improvement_eV": max(0.0, pre_action_best - landing_energy),
            "starter_improvement": landing_energy < energies[starter_id] - 1e-6,
            "inserted_into_archive": bool(record["inserted_into_archive"]),
            "force_evaluations": int(record["force_evaluations"]),
        }
        rows.append(row)
        visits[starter_id] += 1

        landing_id = record["landing_entry_id"]
        if record["inserted_into_archive"]:
            if landing_id is None:
                raise ValueError(f"new entry lacks landing ID in {event_path}")
            landing_id = int(landing_id)
            energies[landing_id] = landing_energy
            creation_attempt[landing_id] = attempt_index
        if landing_id is not None:
            previous_landing_id = int(landing_id)

    if len(rows) != int(summary["completed_attempts"]):
        raise ValueError(f"attempt count mismatch in {event_path}")
    if sum(row["force_evaluations"] for row in rows) != int(summary["action_evaluations"]):
        raise ValueError(f"action force count mismatch in {event_path}")

    provenance = {
        "system": system,
        "seed": seed,
        "events": str(event_path.relative_to(ROOT)),
        "events_sha256": _sha256(event_path),
        "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": _sha256(summary_path),
        "completed_attempts": len(rows),
        "action_force_evaluations": sum(row["force_evaluations"] for row in rows),
    }
    return rows, provenance


def _rate(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row[condition]]
    successes = sum(row["global_improvement"] for row in selected)
    return {
        "attempts": len(selected),
        "global_improvements": successes,
        "rate": successes / len(selected) if selected else None,
    }


def _system_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    improvements = [row for row in rows if row["global_improvement"]]
    non_improvements = [row for row in rows if not row["global_improvement"]]
    nontrivial_improvements = [row for row in improvements if row["archive_size"] > 1]

    observed_rank_mean = _mean(
        [row["starter_energy_rank_fraction"] for row in nontrivial_improvements]
    )
    rank_mean_variance = (
        sum(
            (row["archive_size"] + 1) / (12 * (row["archive_size"] - 1))
            for row in nontrivial_improvements
        )
        / len(nontrivial_improvements) ** 2
        if nontrivial_improvements
        else None
    )
    rank_z = (
        (observed_rank_mean - 0.5) / math.sqrt(rank_mean_variance)
        if observed_rank_mean is not None and rank_mean_variance
        else None
    )
    best_count = sum(row["starter_is_current_best"] for row in nontrivial_improvements)
    best_null_probabilities = [
        1.0 / row["archive_size"] for row in nontrivial_improvements
    ]

    late = [row for row in rows if row["attempt_index"] >= 10]
    late_top = [row for row in late if row["starter_is_top_energy_quartile"]]
    late_bottom = [row for row in late if not row["starter_is_top_energy_quartile"]]

    return {
        "attempts": len(rows),
        "force_evaluations": sum(row["force_evaluations"] for row in rows),
        "new_archive_entries": sum(row["inserted_into_archive"] for row in rows),
        "starter_improvements": sum(row["starter_improvement"] for row in rows),
        "global_improvements": len(improvements),
        "mean_successful_starter_rank_fraction": _mean(
            [row["starter_energy_rank_fraction"] for row in improvements]
        ),
        "mean_other_starter_rank_fraction": _mean(
            [row["starter_energy_rank_fraction"] for row in non_improvements]
        ),
        "current_best": _rate(rows, "starter_is_current_best"),
        "not_current_best": {
            "attempts": sum(not row["starter_is_current_best"] for row in rows),
            "global_improvements": sum(
                row["global_improvement"] and not row["starter_is_current_best"]
                for row in rows
            ),
        },
        "top_energy_quartile": _rate(rows, "starter_is_top_energy_quartile"),
        "not_top_energy_quartile": {
            "attempts": sum(not row["starter_is_top_energy_quartile"] for row in rows),
            "global_improvements": sum(
                row["global_improvement"]
                and not row["starter_is_top_energy_quartile"]
                for row in rows
            ),
        },
        "late_attempts_from_index_10": {
            "attempts": len(late),
            "global_improvements": sum(row["global_improvement"] for row in late),
            "top_energy_quartile": {
                "attempts": len(late_top),
                "global_improvements": sum(row["global_improvement"] for row in late_top),
            },
            "not_top_energy_quartile": {
                "attempts": len(late_bottom),
                "global_improvements": sum(
                    row["global_improvement"] for row in late_bottom
                ),
            },
        },
        "conditional_rank_null": {
            "conditioning": (
                "global-improvement times and archive sizes; selected rank is uniform "
                "under the no-rank-effect null"
            ),
            "nontrivial_improvements": len(nontrivial_improvements),
            "observed_mean_rank_fraction": observed_rank_mean,
            "null_mean_rank_fraction": 0.5,
            "normal_z": rank_z,
            "observed_current_best_count": best_count,
            "current_best_count_upper_tail_probability": (
                _poisson_binomial_upper_tail(best_null_probabilities, best_count)
                if best_null_probabilities
                else None
            ),
        },
    }


def main() -> None:
    event_paths = []
    for campaign_root in CAMPAIGN_ROOTS:
        event_paths.extend(sorted(campaign_root.glob("*/seed-*/uniform/events.jsonl")))
    if len(event_paths) != len(SYSTEMS) * len(SEEDS):
        raise ValueError(f"expected six production event logs, found {len(event_paths)}")

    all_rows: list[dict[str, Any]] = []
    provenance = []
    per_campaign = {}
    for event_path in event_paths:
        rows, source = _parse_campaign(event_path)
        all_rows.extend(rows)
        provenance.append(source)
        per_campaign[f"{source['system']}-seed{source['seed']}"] = {
            "attempts": len(rows),
            "force_evaluations": sum(row["force_evaluations"] for row in rows),
            "new_archive_entries": sum(row["inserted_into_archive"] for row in rows),
            "global_improvements": sum(row["global_improvement"] for row in rows),
        }

    evidence = {
        "schema_version": 1,
        "claim_boundary": (
            "Observational association under uniformly selected starters; no "
            "counterfactual selector performance is inferred."
        ),
        "force_evaluations_added_by_this_audit": 0,
        "provenance": provenance,
        "per_campaign": per_campaign,
        "systems": {
            system: _system_summary(
                [row for row in all_rows if row["system"] == system]
            )
            for system in SYSTEMS
        },
        "global_improvement_events": [
            row for row in all_rows if row["global_improvement"]
        ],
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence["systems"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
