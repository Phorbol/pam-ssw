from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
OPTIMIZERS = (
    "ase-fire",
    "ase-fire2",
    "safe-lbfgs-total",
    "bias-separated-lbfgs",
)
FROZEN_COMMIT = "c598096170d40625b6d127668935a166cc1507c1"


def _load(system: str, optimizer: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / f"{system}__{optimizer}" / "result.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["frozen_commit"] != FROZEN_COMMIT:
        raise ValueError(f"commit mismatch: {path}")
    if not payload["benchmark_eligible"]:
        raise ValueError(f"ineligible result: {path}")
    if payload["purpose_counts"]["unattributed"] != 0:
        raise ValueError(f"unattributed calls: {path}")
    if sum(payload["purpose_counts"].values()) != payload["total_evaluations"]:
        raise ValueError(f"open purpose ledger: {path}")
    attempts = payload["completed_attempts"] + payload["failed_attempts"]
    if payload["optimizer_diagnostics"]["attempt_count"] != attempts:
        raise ValueError(f"diagnostic/action mismatch: {path}")
    return payload


def _row(payload: dict[str, Any]) -> dict[str, Any]:
    sums = payload["optimizer_diagnostics"]["sums"]
    return {
        "system": payload["system"],
        "optimizer": payload["optimizer"],
        "bootstrap_evaluations": payload["bootstrap_evaluations"],
        "action_evaluations": payload["action_evaluations"],
        "total_evaluations": payload["total_evaluations"],
        "unused_force_budget": payload["unused_force_budget"],
        "biased_proposal_relax_evaluations": payload["purpose_counts"][
            "biased_proposal_relax"
        ],
        "proposal_relax_backend_evaluations": sums[
            "proposal_relax_backend_evaluations"
        ],
        "proposal_relax_rejected_steps": sums["proposal_relax_rejected_steps"],
        "proposal_relax_rejected_secants": sums[
            "proposal_relax_rejected_secants"
        ],
        "proposal_relax_converged": sums["proposal_relax_termination_converged"],
        "proposal_relax_maxiter": sums["proposal_relax_termination_maxiter"],
        "proposal_relax_line_search_failed": sums[
            "proposal_relax_termination_line_search_failed"
        ],
        "proposal_relax_bias_secant_curvature_sum": sums[
            "proposal_relax_bias_secant_curvature_sum"
        ],
        "completed_attempts": payload["completed_attempts"],
        "failed_attempts": payload["failed_attempts"],
        "unique_minima": payload["unique_minima"],
        "best_energy_eV": payload["best_energy_eV"],
        "best_energy_drop_from_bootstrap_eV": payload[
            "best_energy_drop_from_bootstrap_eV"
        ],
        "wall_time_s": payload["wall_time_s"],
    }


def _difference(separated: dict[str, Any], total: dict[str, Any]) -> dict[str, Any]:
    names = (
        "action_evaluations",
        "total_evaluations",
        "biased_proposal_relax_evaluations",
        "proposal_relax_backend_evaluations",
        "proposal_relax_rejected_steps",
        "proposal_relax_rejected_secants",
        "proposal_relax_converged",
        "proposal_relax_line_search_failed",
        "completed_attempts",
        "unique_minima",
        "best_energy_drop_from_bootstrap_eV",
        "wall_time_s",
    )
    return {
        f"bias_separated_minus_total_{name}": separated[name] - total[name]
        for name in names
    }


def main() -> int:
    preflight = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    if preflight["frozen_commit"] != FROZEN_COMMIT:
        raise ValueError("preflight commit mismatch")
    rows = [
        _row(_load(system, optimizer))
        for system in SYSTEMS
        for optimizer in OPTIMIZERS
    ]
    by_key = {(row["system"], row["optimizer"]): row for row in rows}
    paired = {
        system: _difference(
            by_key[(system, "bias-separated-lbfgs")],
            by_key[(system, "safe-lbfgs-total")],
        )
        for system in SYSTEMS
    }
    primary_cost_improved_both = all(
        paired[system][
            "bias_separated_minus_total_biased_proposal_relax_evaluations"
        ]
        < 0
        for system in SYSTEMS
    )
    payload = {
        "schema_version": 1,
        "frozen_commit": FROZEN_COMMIT,
        "matrix_complete": len(rows) == len(SYSTEMS) * len(OPTIMIZERS),
        "rows": rows,
        "bias_separated_minus_total": paired,
        "gate_decision": {
            "advance_bias_separation_to_g2": primary_cost_improved_both,
            "primary_metric": "charged biased_proposal_relax evaluations",
            "reason": (
                "primary cost improved on both systems"
                if primary_cost_improved_both
                else "bias separation increased primary proposal cost on at least one system"
            ),
        },
        "claim_ceiling": (
            "One GPU seed establishes execution validity and rejects advancement "
            "of this bias-secant mechanism; it does not rank optimizers statistically."
        ),
    }
    with (RUN_ROOT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
