#!/usr/bin/env python3
"""Build bounded evidence and a human-readable report for the PdO K4/K8 run."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping


RUN_ROOT = Path(__file__).resolve().parent
ARMS = ("k4", "k8")
SEEDS = (42, 43)
EXPECTED_FORCE_BUDGET = 20_000


def arm_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    experiment = summary["experiment"]
    stats = summary["stats"]
    telemetry = summary["optimizer_telemetry"]
    purpose_counts = dict(summary["purpose_counts"])
    count = int(telemetry["true_quench_count"])
    converged = int(telemetry["true_quench_termination_converged"])
    unconverged = int(telemetry["true_quench_unconverged"])
    if converged + unconverged != count:
        raise ValueError("true-quench certificate counts do not close")
    total = int(summary["force_evaluations"])
    if sum(int(value) for value in purpose_counts.values()) != total:
        raise ValueError("purpose-level force counts do not close")
    if int(purpose_counts.get("unattributed", 0)) != 0:
        raise ValueError("unattributed force evaluations are not allowed")
    if total != EXPECTED_FORCE_BUDGET:
        raise ValueError(f"expected {EXPECTED_FORCE_BUDGET} force evaluations")
    best_energy = float(summary["best_energy_eV"])
    best_trials = [
        int(record["trial"])
        for record in summary["walk_records"]
        if abs(float(record["energy_eV"]) - best_energy) <= 1e-9
    ]
    first_best_trial = 0 if abs(float(summary["initial_energy_eV"]) - best_energy) <= 1e-9 else min(best_trials)
    return {
        "arm": str(experiment["arm"]),
        "seed": int(experiment["seed"]),
        "execution_commit": str(summary["execution_commit"]),
        "energy": {
            "initial_eV": float(summary["initial_energy_eV"]),
            "best_eV": float(summary["best_energy_eV"]),
            "drop_eV": float(summary["energy_drop_eV"]),
        },
        "search_output": {
            "completed_trials": int(stats["n_trials"]),
            "archive_minima": int(stats["n_minima"]),
            "duplicate_rate": float(stats["duplicate_rate"]),
            "budget_exhausted": bool(stats["budget_exhausted"]),
            "first_best_trial": first_best_trial,
        },
        "force_evaluations": {
            "total": total,
            "by_purpose": purpose_counts,
        },
        "wall_time_s": float(summary["timing"]["total_wall_time_s"]),
        "directions": {
            "choices": int(stats["direction_choices"]),
            "candidate_evaluations": int(stats["direction_candidate_evaluations"]),
            "selected": {
                "random": int(stats["direction_selected_random"]),
                "bond": int(stats["direction_selected_bond"]),
                "momentum": int(stats["direction_selected_momentum"]),
            },
        },
        "proposal_relaxation": {
            "count": int(telemetry["proposal_relax_count"]),
            "mean_iterations": float(telemetry["proposal_relax_mean_iterations"]),
            "median_iterations": float(telemetry["proposal_relax_median_iterations"]),
            "p90_iterations": float(telemetry["proposal_relax_p90_iterations"]),
            "unconverged": int(telemetry["proposal_relax_unconverged"]),
        },
        "true_quench": {
            "mean_iterations": float(telemetry["true_quench_mean_iterations"]),
            "median_iterations": float(telemetry["true_quench_median_iterations"]),
            "p90_iterations": float(telemetry["true_quench_p90_iterations"]),
            "max_gradient": float(telemetry["true_quench_max_gradient"]),
        },
        "quench_certificates": {
            "count": count,
            "converged": converged,
            "unconverged": unconverged,
            "complete": unconverged == 0,
            "fraction": float(converged / count) if count else 1.0,
        },
    }


def paired_evidence(k4: Mapping[str, Any], k8: Mapping[str, Any]) -> dict[str, Any]:
    if int(k4["seed"]) != int(k8["seed"]):
        raise ValueError("paired arms must have the same seed")
    p4 = k4["force_evaluations"]["by_purpose"]
    p8 = k8["force_evaluations"]["by_purpose"]
    return {
        "seed": int(k4["seed"]),
        "delta_k4_minus_k8": {
            "energy_drop_eV": float(k4["energy"]["drop_eV"] - k8["energy"]["drop_eV"]),
            "completed_trials": int(
                k4["search_output"]["completed_trials"]
                - k8["search_output"]["completed_trials"]
            ),
            "archive_minima": int(
                k4["search_output"]["archive_minima"]
                - k8["search_output"]["archive_minima"]
            ),
            "duplicate_rate": float(
                k4["search_output"]["duplicate_rate"]
                - k8["search_output"]["duplicate_rate"]
            ),
            "wall_time_s": float(k4["wall_time_s"] - k8["wall_time_s"]),
            "direction_force_evaluations": int(
                p4["direction_oracle"] - p8["direction_oracle"]
            ),
            "proposal_force_evaluations": int(
                p4["biased_proposal_relax"] - p8["biased_proposal_relax"]
            ),
            "landing_force_evaluations": int(
                p4["landing_true_quench"] - p8["landing_true_quench"]
            ),
        },
    }


def promotion_decision(pairs: list[Mapping[str, Any]]) -> dict[str, Any]:
    energy_gate = {
        str(pair["seed"]): pair["delta_k4_minus_k8"]["energy_drop_eV"] >= 0.0
        for pair in pairs
    }
    trial_gate = {
        str(pair["seed"]): pair["delta_k4_minus_k8"]["completed_trials"] >= 0
        for pair in pairs
    }
    return {
        "promote_k4": all(energy_gate.values()) and all(trial_gate.values()),
        "energy_gate_by_seed": energy_gate,
        "trial_gate_by_seed": trial_gate,
        "rule": (
            "K4 must match or beat K8 energy drop and complete at least as many "
            "trials in both paired seeds."
        ),
    }


def build_evidence(run_root: Path = RUN_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = run_root / f"output-{arm}-seed{seed}" / "summary.json"
            rows.append(arm_evidence(json.loads(path.read_text(encoding="utf-8"))))
    by_key = {(row["arm"], row["seed"]): row for row in rows}
    pairs = [
        paired_evidence(by_key[("k4", seed)], by_key[("k8", seed)])
        for seed in SEEDS
    ]
    aggregate = {
        arm: {
            "mean_energy_drop_eV": mean(
                by_key[(arm, seed)]["energy"]["drop_eV"] for seed in SEEDS
            ),
            "mean_completed_trials": mean(
                by_key[(arm, seed)]["search_output"]["completed_trials"]
                for seed in SEEDS
            ),
            "mean_archive_minima": mean(
                by_key[(arm, seed)]["search_output"]["archive_minima"]
                for seed in SEEDS
            ),
            "mean_duplicate_rate": mean(
                by_key[(arm, seed)]["search_output"]["duplicate_rate"]
                for seed in SEEDS
            ),
        }
        for arm in ARMS
    }
    return {
        "schema_version": 1,
        "experiment": "PdO K4/K8 fixed-total-force-budget ablation",
        "arms": rows,
        "paired_differences": pairs,
        "aggregate_descriptive_only": aggregate,
        "promotion": promotion_decision(pairs),
        "claim_boundary": {
            "paired_seed_count": len(SEEDS),
            "statistical_inference": False,
            "note": (
                "Two paired seeds diagnose mechanism and reject/retain the "
                "pre-registered promotion gate; they do not establish a "
                "population-level performance difference."
            ),
        },
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def render_report(evidence: Mapping[str, Any]) -> str:
    by_key = {
        (row["arm"], row["seed"]): row
        for row in evidence["arms"]
    }
    lines = [
        "# PdO K4/K8 fixed-budget result",
        "",
        "## Outcome",
        "",
        (
            "**Do not promote K4 as the PdO default.** K4 bought more completed "
            "macro steps by cutting direction-oracle cost, but the paired "
            "best-energy result split 1–1 and therefore failed the "
            "pre-registered gate."
        ),
        "",
        "| Seed | Arm | ΔE best (eV) | Trials | Minima | Duplicate | Direction FE | Proposal FE | Landing FE | Wall (s) | Quench cert. |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in SEEDS:
        for arm in ("k8", "k4"):
            row = by_key[(arm, seed)]
            purpose = row["force_evaluations"]["by_purpose"]
            cert = row["quench_certificates"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(seed),
                        arm.upper(),
                        _fmt(row["energy"]["drop_eV"], 6),
                        str(row["search_output"]["completed_trials"]),
                        str(row["search_output"]["archive_minima"]),
                        _fmt(row["search_output"]["duplicate_rate"], 3),
                        str(purpose["direction_oracle"]),
                        str(purpose["biased_proposal_relax"]),
                        str(purpose["landing_true_quench"]),
                        _fmt(row["wall_time_s"], 1),
                        f'{cert["converged"]}/{cert["count"]}',
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Every arm used exactly 20,000 force evaluations; bootstrap cost was "
            "37 FE, starter-quench cost was 0, and unattributed cost was 0.",
            "",
            "## Paired interpretation",
            "",
        ]
    )
    for pair in evidence["paired_differences"]:
        d = pair["delta_k4_minus_k8"]
        lines.append(
            f'- Seed {pair["seed"]}: K4−K8 = '
            f'{d["energy_drop_eV"]:+.6f} eV energy drop, '
            f'{d["completed_trials"]:+d} trials, '
            f'{d["archive_minima"]:+d} minima, '
            f'{d["direction_force_evaluations"]:+d} direction FE.'
        )
    lines.extend(
        [
            "",
            (
                "The clean mechanism is therefore supported: reducing K from 8 "
                "to 4 approximately halves candidate/HVP work and converts that "
                "budget into more macro attempts. It does not yet preserve final "
                "energy reliably; seed 43 lost 0.519775 eV despite 32 extra trials."
            ),
            "",
            "## Certificate boundary",
            "",
            (
                "Seed 42 had complete true-quench force certificates in both "
                "arms. Seed 43 had one non-certified true quench in each arm "
                "(K8 73/74; K4 105/106). The current walker records such a "
                "returned state in the archive, so archive counts are search "
                "outputs, not a guarantee that every stored state satisfies "
                "`fmax=0.03 eV/Å`."
            ),
            "",
            (
                "The best seed-43 energies were reached before the final reported "
                "states "
                f'(K8 trial {by_key[("k8", 43)]["search_output"]["first_best_trial"]}; '
                f'K4 trial {by_key[("k4", 43)]["search_output"]["first_best_trial"]}), '
                "so this qualification does "
                "not change the paired best-energy decision. Per-quench "
                "termination identity was not persisted, so no stronger claim "
                "about which archive entry lacked the certificate is made."
            ),
            "",
            "## Decision",
            "",
            "- Retain K8 in the current PdO production profile; do not promote K4.",
            "- Keep K4 as a throughput-oriented experimental arm.",
            (
                "- Next direction experiment should improve candidate quality at "
                "K4-scale cost, rather than restoring candidate count blindly."
            ),
            (
                "- Add an explicit true-quench certificate gate/log before using "
                "archive-minima counts as scientifically certified minima."
            ),
            "",
            "This is a two-seed mechanism ablation, not a statistical performance proof.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    evidence = build_evidence()
    (RUN_ROOT / "evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (RUN_ROOT / "final_report.md").write_text(
        render_report(evidence),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
