from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from math import fsum, isclose, isfinite
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.exploration import (
    AttemptResult,
    AttemptStatus,
    ExplorationController,
    ExplorationEventLog,
    PosteriorExplorationConfig,
    SCHEMA_VERSION,
    run_posterior_ssw,
)
from pamssw.potentials import DoubleWell2D
from pamssw.state import State


EXPECTED_SNAPSHOT_FIELDS = {
    "archive_version",
    "batch_id",
    "eligible_starter_ids",
    "policy_name",
    "policy_version",
    "probabilities",
    "record_type",
    "schema_version",
    "support_complete",
}
EXPECTED_ATTEMPT_FIELDS = {
    "action_id",
    "archive_version",
    "batch_id",
    "cost_is_exact",
    "discovered_against_snapshot",
    "evaluation_counts",
    "failure_reason",
    "force_budget",
    "force_evaluations",
    "inserted_into_archive",
    "landing_energy",
    "landing_entry_id",
    "policy_name",
    "policy_version",
    "posterior_observed",
    "random_seed",
    "record_type",
    "schema_version",
    "selection_probability",
    "slot_id",
    "starter_id",
    "status",
    "within_batch_collision",
}
EXPECTED_COMMIT_FIELDS = {
    "action_ids",
    "batch_id",
    "record_type",
    "schema_version",
}


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _nonnegative_int(value: object, message: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        message,
    )
    return int(value)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse event log: {path}") from exc
    _require(bool(rows), "event log must not be empty")
    _require(all(isinstance(row, dict) for row in rows), "event rows must be objects")
    return rows


def _committed_batches(path: Path) -> list[dict[str, Any]]:
    """Parse schema-v2 committed batches without trusting the runner result."""
    batches: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    seen_action_ids: set[str] = set()
    expected_batch_id = 0
    for row in _read_rows(path):
        kind = row.get("record_type")
        _require(row.get("schema_version") == SCHEMA_VERSION, "schema version drift")
        if kind == "policy_snapshot":
            _require(set(row) == EXPECTED_SNAPSHOT_FIELDS, "snapshot field drift")
            _require(active is None, "nested policy snapshot")
            batch_id = _nonnegative_int(row["batch_id"], "invalid snapshot batch id")
            _require(batch_id == expected_batch_id, "nonsequential snapshot batch id")
            _require(
                _nonnegative_int(row["policy_version"], "invalid policy version")
                == expected_batch_id,
                "nonsequential policy version",
            )
            _require(
                _nonnegative_int(row["archive_version"], "invalid archive version")
                == expected_batch_id,
                "nonsequential archive version",
            )
            _require(
                isinstance(row["policy_name"], str) and bool(row["policy_name"]),
                "invalid policy name",
            )
            probabilities = row["probabilities"]
            starters = row["eligible_starter_ids"]
            _require(row["support_complete"] is True, "incomplete policy support")
            _require(
                isinstance(probabilities, list) and isinstance(starters, list),
                "policy support must be JSON arrays",
            )
            _require(len(probabilities) == len(starters) > 0, "policy support mismatch")
            _require(
                len(set(starters)) == len(starters)
                and all(
                    isinstance(starter, int)
                    and not isinstance(starter, bool)
                    and starter >= 0
                    for starter in starters
                ),
                "invalid policy starter support",
            )
            _require(
                all(
                    not isinstance(value, bool)
                    and isinstance(value, (int, float))
                    and isfinite(float(value))
                    and float(value) > 0.0
                    for value in probabilities
                ),
                "invalid policy probability",
            )
            _require(
                isclose(fsum(float(value) for value in probabilities), 1.0, abs_tol=1e-12),
                "policy probabilities do not normalize",
            )
            active = {"snapshot": row, "attempts": []}
        elif kind == "attempt":
            _require(set(row) == EXPECTED_ATTEMPT_FIELDS, "attempt field drift")
            _require(active is not None, "attempt outside a batch")
            action_id = row["action_id"]
            _require(
                isinstance(action_id, str) and bool(action_id),
                "invalid action identifier",
            )
            _require(action_id not in seen_action_ids, "duplicate action identifier")
            seen_action_ids.add(action_id)
            snapshot = active["snapshot"]
            _require(row["batch_id"] == snapshot["batch_id"], "batch mismatch")
            _require(row["policy_name"] == snapshot["policy_name"], "policy mismatch")
            _require(
                row["policy_version"] == snapshot["policy_version"],
                "policy version mismatch",
            )
            _require(
                row["archive_version"] == snapshot["archive_version"],
                "archive version mismatch",
            )
            slot_id = _nonnegative_int(row["slot_id"], "invalid attempt slot")
            _require(slot_id == len(active["attempts"]), "attempts are not in slot order")
            _nonnegative_int(row["random_seed"], "invalid action seed")
            force_budget = _nonnegative_int(row["force_budget"], "invalid action budget")
            _require(force_budget > 0, "action budget must be positive")
            force_evaluations = _nonnegative_int(
                row["force_evaluations"], "invalid force evaluation count"
            )
            _require(force_evaluations <= force_budget, "action exceeded budget")
            _require(row["cost_is_exact"] is True, "attempt cost is not exact")
            _require(
                isinstance(row["posterior_observed"], bool),
                "invalid posterior observation flag",
            )
            _require(isinstance(row["status"], str), "invalid terminal status")
            _require(
                row["status"] in {status.value for status in AttemptStatus},
                "invalid terminal status",
            )
            status = AttemptStatus(row["status"])
            starter_id = _nonnegative_int(row["starter_id"], "invalid attempt starter")
            selection_probability = row["selection_probability"]
            _require(
                not isinstance(selection_probability, bool)
                and isinstance(selection_probability, (int, float))
                and isfinite(float(selection_probability))
                and float(selection_probability) > 0.0,
                "invalid saved selection probability",
            )
            probability_by_starter = dict(
                zip(snapshot["eligible_starter_ids"], snapshot["probabilities"])
            )
            _require(starter_id in probability_by_starter, "starter outside support")
            _require(
                isclose(
                    float(selection_probability),
                    float(probability_by_starter[starter_id]),
                    abs_tol=1e-15,
                ),
                "saved selection probability mismatch",
            )
            counts = row["evaluation_counts"]
            _require(isinstance(counts, dict), "attempt counts must be an object")
            _require(
                tuple(counts) == tuple(purpose.value for purpose in EvaluationPurpose),
                "attempt counts must use canonical purpose ordering",
            )
            _require(
                all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in counts.values()
                ),
                "attempt counts must be nonnegative integers",
            )
            _require(
                sum(counts.values()) == force_evaluations,
                "attempt evaluation counts do not close",
            )
            _require(counts["unattributed"] == 0, "unattributed action call")
            for name in (
                "discovered_against_snapshot",
                "inserted_into_archive",
                "within_batch_collision",
            ):
                _require(isinstance(row[name], bool), f"invalid {name}")
            failure_reason = row["failure_reason"]
            if status is AttemptStatus.COMPLETED:
                _require(failure_reason is None, "completed attempt has a failure reason")
                _nonnegative_int(row["landing_entry_id"], "completed attempt lacks entry id")
                landing_energy = row["landing_energy"]
                _require(
                    not isinstance(landing_energy, bool)
                    and isinstance(landing_energy, (int, float))
                    and isfinite(float(landing_energy)),
                    "completed attempt has invalid landing energy",
                )
                _require(
                    not row["inserted_into_archive"]
                    or row["discovered_against_snapshot"],
                    "inserted attempt was not discovered",
                )
                _require(
                    row["within_batch_collision"]
                    == (
                        row["discovered_against_snapshot"]
                        and not row["inserted_into_archive"]
                    ),
                    "invalid within-batch collision provenance",
                )
            else:
                _require(
                    isinstance(failure_reason, str) and bool(failure_reason),
                    "failed attempt lacks failure reason",
                )
                _require(
                    row["landing_entry_id"] is None and row["landing_energy"] is None,
                    "failed attempt contains landing data",
                )
                _require(
                    not row["discovered_against_snapshot"]
                    and not row["inserted_into_archive"]
                    and not row["within_batch_collision"],
                    "failed attempt contains discovery credit",
                )
            _require(
                row["posterior_observed"]
                == (status is not AttemptStatus.WORKER_ERROR and force_evaluations > 0),
                "terminal posterior observation rule mismatch",
            )
            active["attempts"].append(row)
        elif kind == "batch_commit":
            _require(set(row) == EXPECTED_COMMIT_FIELDS, "commit field drift")
            _require(active is not None, "commit outside a batch")
            attempts = active["attempts"]
            _require(bool(attempts), "empty committed batch")
            _nonnegative_int(row["batch_id"], "invalid commit batch id")
            _require(row["batch_id"] == active["snapshot"]["batch_id"], "commit batch mismatch")
            _require(
                isinstance(row["action_ids"], list)
                and row["action_ids"] == [attempt["action_id"] for attempt in attempts],
                "commit action list mismatch",
            )
            batches.append({**active, "commit": row})
            active = None
            expected_batch_id += 1
        else:
            raise ValidationError(f"unknown record type: {kind!r}")
    _require(active is None, "uncommitted terminal batch")
    return batches


def _sum_action_counts(attempts: list[dict[str, Any]]) -> dict[str, int]:
    return {
        purpose.value: sum(
            int(attempt["evaluation_counts"][purpose.value]) for attempt in attempts
        )
        for purpose in EvaluationPurpose
    }


def _posterior_counts(posterior, entry_ids: list[int]) -> dict[str, list[int]]:
    return {str(entry_id): list(posterior.counts(entry_id)) for entry_id in entry_ids}


def _initial_state() -> State:
    return State(
        numbers=np.ones(4, dtype=int),
        positions=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def _real_campaign(root: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    run_directory = root / "real-campaign"
    ssw_config = SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=2,
        proposal_pool_size=1,
        rng_seed=19,
    )
    exploration_config = PosteriorExplorationConfig(
        policy_name="uniform",
        batch_size=3,
        max_workers=2,
        action_force_budget=30,
        total_force_budget=130,
        master_seed=37,
        run_directory=run_directory,
    )
    result = run_posterior_ssw(
        _initial_state(),
        lambda: AnalyticCalculator(DoubleWell2D()),
        ssw_config,
        exploration_config,
    )
    event_path = run_directory / "events.jsonl"
    batches = _committed_batches(event_path)
    attempts = [attempt for batch in batches for attempt in batch["attempts"]]
    diagnostics = json.loads(
        (run_directory / "optimizer_diagnostics.json").read_text(encoding="utf-8")
    )
    _require(
        diagnostics.get("schema_version") == 1
        and isinstance(diagnostics.get("attempts"), list),
        "optimizer diagnostics schema drift",
    )
    diagnostic_ids = sorted(row["action_id"] for row in diagnostics["attempts"])
    event_ids = sorted(row["action_id"] for row in attempts)
    action_counts = _sum_action_counts(attempts)
    campaign_counts = result.purpose_counts.as_dict()
    bootstrap_counts = {
        purpose.value: campaign_counts[purpose.value] - action_counts[purpose.value]
        for purpose in EvaluationPurpose
    }
    entry_ids = [entry.entry_id for entry in result.archive.entries]
    live_posterior = _posterior_counts(result.posterior, entry_ids)
    replayed = ExplorationEventLog(event_path).reconstruct_posterior()
    replayed_posterior = _posterior_counts(replayed, entry_ids)
    observed = sum(bool(row["posterior_observed"]) for row in attempts)
    invariants = {
        "real_batches_committed": len(batches) == result.completed_batches,
        "real_worker_actions_equal_event_actions": diagnostic_ids == event_ids,
        "real_attempt_partition_closes": (
            len(attempts) == result.completed_attempts + result.failed_attempts
        ),
        "real_action_counts_close": sum(action_counts.values()) == result.action_evaluations,
        "real_bootstrap_counts_nonnegative": all(
            value >= 0 for value in bootstrap_counts.values()
        ),
        "real_bootstrap_counts_close": (
            sum(bootstrap_counts.values()) == result.bootstrap_evaluations
        ),
        "real_total_counts_close": (
            result.bootstrap_evaluations + result.action_evaluations
            == result.total_evaluations
            == result.purpose_counts.total
        ),
        "real_budget_closes": (
            result.total_evaluations + result.unused_force_budget
            == result.total_force_budget
        ),
        "real_unattributed_zero": (
            campaign_counts[EvaluationPurpose.UNATTRIBUTED.value] == 0
        ),
        "real_posterior_replays": live_posterior == replayed_posterior,
        "real_observation_count_replays": (
            observed == result.posterior_observed_attempts == replayed.completed_attempts
        ),
    }
    _require(all(invariants.values()), "real campaign invariant failed")
    payload = {
        "configuration": {
            "potential": "DoubleWell2D",
            "policy_name": exploration_config.policy_name,
            "proposal_pool_size": ssw_config.proposal_pool_size,
            "batch_size": exploration_config.batch_size,
            "max_workers": exploration_config.max_workers,
            "action_force_budget": exploration_config.action_force_budget,
            "total_force_budget": exploration_config.total_force_budget,
            "master_seed": exploration_config.master_seed,
            "ssw_rng_seed": ssw_config.rng_seed,
        },
        "batch_widths": [len(batch["attempts"]) for batch in batches],
        "action_ids": event_ids,
        "action_costs": [row["force_evaluations"] for row in attempts],
        "terminal_status_counts": dict(
            sorted(Counter(row["status"] for row in attempts).items())
        ),
        "action_purpose_counts": action_counts,
        "derived_bootstrap_purpose_counts": bootstrap_counts,
        "campaign_purpose_counts": campaign_counts,
        "bootstrap_evaluations": result.bootstrap_evaluations,
        "action_evaluations": result.action_evaluations,
        "total_evaluations": result.total_evaluations,
        "unused_force_budget": result.unused_force_budget,
        "stop_reason": result.stop_reason.value,
        "benchmark_eligible": result.benchmark_eligible,
        "benchmark_ineligibility_reasons": list(result.benchmark_ineligibility_reasons),
        "posterior_observed_attempts": result.posterior_observed_attempts,
        "live_posterior": live_posterior,
        "replayed_posterior": replayed_posterior,
    }
    return payload, invariants


def _matrix_archive() -> MinimaArchive:
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.05)
    archive.add(
        State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])),
        0.0,
        parent_id=None,
    )
    return archive


def _counts(purpose: EvaluationPurpose, total: int) -> EvaluationCounts:
    return EvaluationCounts.from_mapping({purpose: total})


def _terminal_matrix(root: Path) -> tuple[dict[str, Any], dict[str, bool]]:
    event_path = root / "terminal-matrix.jsonl"
    controller = ExplorationController(
        _matrix_archive(),
        "uniform",
        71,
        ExplorationEventLog(event_path),
        require_exact_cost=True,
    )

    cases = (
        (
            "completed",
            AttemptStatus.COMPLETED,
            _counts(EvaluationPurpose.DIRECTION_ORACLE, 3),
            True,
        ),
        (
            "invalid_after_pes",
            AttemptStatus.INVALID,
            _counts(EvaluationPurpose.BIASED_PROPOSAL_RELAX, 2),
            True,
        ),
        (
            "fragmented_after_pes",
            AttemptStatus.FRAGMENTED,
            _counts(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK, 1),
            True,
        ),
        (
            "budget_exhausted",
            AttemptStatus.BUDGET_EXHAUSTED,
            _counts(EvaluationPurpose.DIRECTION_ORACLE, 2),
            True,
        ),
        ("invalid_before_pes", AttemptStatus.INVALID, EvaluationCounts.zero(), False),
        ("worker_error", AttemptStatus.WORKER_ERROR, EvaluationCounts.zero(), False),
    )

    def worker(action, starter_state) -> AttemptResult:
        del starter_state
        name, status, counts, _ = cases[action.slot_id]
        if status is AttemptStatus.COMPLETED:
            return AttemptResult(
                action=action,
                landing_state=State(
                    numbers=np.array([1]),
                    positions=np.array([[1.0, 0.0, 0.0]]),
                ),
                landing_energy=-1.0,
                force_evaluations=counts.total,
                status=status,
                failure_reason=None,
                evaluation_counts=counts,
                cost_is_exact=True,
            )
        return AttemptResult(
            action=action,
            landing_state=None,
            landing_energy=None,
            force_evaluations=counts.total,
            status=status,
            failure_reason=name,
            evaluation_counts=counts,
            cost_is_exact=True,
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        outcomes = controller.run_batch(
            executor,
            worker,
            batch_size=len(cases),
            force_budget=8,
        )

    (batch,) = _committed_batches(event_path)
    attempts = batch["attempts"]
    replayed = ExplorationEventLog(event_path).reconstruct_posterior()
    entry_ids = [entry.entry_id for entry in controller.archive.entries]
    live_posterior = _posterior_counts(controller.posterior, entry_ids)
    replayed_posterior = _posterior_counts(replayed, entry_ids)
    rows = [
        {
            "case": cases[index][0],
            "status": row["status"],
            "force_evaluations": row["force_evaluations"],
            "cost_is_exact": row["cost_is_exact"],
            "posterior_observed": row["posterior_observed"],
            "failure_reason": row["failure_reason"],
            "evaluation_counts": row["evaluation_counts"],
        }
        for index, row in enumerate(attempts)
    ]
    expected_observed = [case[3] for case in cases]
    invariants = {
        "matrix_one_attempt_per_outcome": len(attempts) == len(outcomes) == len(cases),
        "matrix_action_ids_close": (
            [row["action_id"] for row in attempts]
            == [outcome.action_id for outcome in outcomes]
        ),
        "matrix_statuses_preserved": (
            [row["status"] for row in attempts] == [case[1].value for case in cases]
        ),
        "matrix_terminal_fields_preserved": all(
            row["force_evaluations"] == case[2].total
            and row["evaluation_counts"] == case[2].as_dict()
            and row["cost_is_exact"] is True
            and row["failure_reason"]
            == (None if case[1] is AttemptStatus.COMPLETED else case[0])
            for row, case in zip(attempts, cases)
        ),
        "matrix_observation_rule_preserved": (
            [row["posterior_observed"] for row in attempts] == expected_observed
        ),
        "matrix_live_replay_match": live_posterior == replayed_posterior,
        "matrix_replay_observation_count": (
            replayed.completed_attempts == sum(expected_observed)
        ),
        "matrix_zero_cost_not_observed": all(
            row["posterior_observed"] is False
            for row in attempts
            if row["force_evaluations"] == 0
        ),
    }
    _require(all(invariants.values()), "terminal matrix invariant failed")
    return {
        "rows": rows,
        "live_posterior": live_posterior,
        "replayed_posterior": replayed_posterior,
    }, invariants


CLAIM_CEILING = (
    "The tested analytic one-action ThreadPool path persisted one terminal "
    "outcome per worker action, closed force-evaluation accounting, and "
    "replayed the live starter-productivity posterior. It does not compare "
    "starter-policy performance or establish equilibrium-unbiased sampling."
)


def _conclusion(evidence: dict[str, Any]) -> str:
    campaign = evidence["real_campaign"]
    return f"""# Posterior terminal-outcome validation

Source commit: `{evidence["git_commit"]}`

The analytic ThreadPool campaign committed {len(campaign["action_ids"])} actions
in batch widths {campaign["batch_widths"]}. Bootstrap, action, and total
force-evaluation counts were {campaign["bootstrap_evaluations"]},
{campaign["action_evaluations"]}, and {campaign["total_evaluations"]};
unattributed evaluations were zero.

All real-campaign and terminal-matrix invariants passed. The committed event
log reconstructed the same starter-productivity posterior as the live
controller.

This validation does not compare starter-policy performance, establish
thermodynamic or stationary-distribution unbiasedness, validate GPU/MLIP
execution, or make a C60/PdO performance claim.
"""


def _write_exclusive(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()


def run_validation(output_directory: Path, *, source_commit: str) -> dict[str, Any]:
    output_directory = Path(output_directory)
    _require(output_directory.is_dir(), "output directory must exist")
    _require(
        len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit),
        "source commit must be a lowercase 40-character Git SHA",
    )
    evidence_path = output_directory / "evidence.json"
    conclusion_path = output_directory / "conclusion.md"
    _require(not evidence_path.exists(), "evidence output already exists")
    _require(not conclusion_path.exists(), "conclusion output already exists")

    with TemporaryDirectory(prefix="pamssw-posterior-validation-") as temporary:
        root = Path(temporary)
        real_campaign, real_invariants = _real_campaign(root)
        terminal_matrix, matrix_invariants = _terminal_matrix(root)

    invariants = {**real_invariants, **matrix_invariants}
    _require(all(invariants.values()), "not all validation invariants passed")
    evidence = {
        "schema_version": 1,
        "git_commit": source_commit,
        "validation_scope": (
            "analytic one-proposal fixed-budget ThreadPool execution and "
            "terminal credit/replay semantics"
        ),
        "real_campaign": real_campaign,
        "terminal_matrix": terminal_matrix,
        "invariants": invariants,
        "claim_ceiling": CLAIM_CEILING,
    }
    encoded = json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_exclusive(evidence_path, encoded)
    _write_exclusive(conclusion_path, _conclusion(evidence))
    return evidence


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--require-clean", action="store_true")
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    actual_commit = _git_output(repo_root, "rev-parse", "HEAD")
    _require(
        actual_commit == arguments.expected_git_commit,
        "execution commit does not match --expected-git-commit",
    )
    if arguments.require_clean:
        _require(
            not _git_output(repo_root, "status", "--porcelain", "--untracked-files=no"),
            "tracked worktree must be clean",
        )
    run_validation(arguments.output_directory, source_commit=actual_commit)


if __name__ == "__main__":
    main()
