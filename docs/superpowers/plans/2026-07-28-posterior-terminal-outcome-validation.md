# Posterior Terminal-Outcome Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate reproducible analytic evidence that the existing one-proposal ThreadPool exploration path commits exactly one terminal result per action with closed evaluation accounting and replayable posterior credit.

**Architecture:** Add no production abstraction and change no exploration algorithm. A standalone validation script runs the public posterior SSW runner plus a deterministic terminal-status matrix, independently audits schema-v2 event rows, and writes compact evidence only after every invariant passes; one end-to-end test protects the validator contract.

**Tech Stack:** Python 3, NumPy, pytest, ASE-independent analytic `DoubleWell2D`, `ThreadPoolExecutor`, existing `pamssw.exploration` APIs, JSONL.

---

## File map

- Create
  `runs/20260728-posterior-terminal-outcome-validation/run_validation.py`:
  execute both validation layers, audit event logs, fail closed, and write
  deterministic evidence.
- Create
  `tests/integration/test_posterior_terminal_outcome_validation.py`:
  one end-to-end contract test for the validation artifact.
- Create
  `runs/20260728-posterior-terminal-outcome-validation/evidence.json`:
  reviewed output generated from the committed validator.
- Create
  `runs/20260728-posterior-terminal-outcome-validation/conclusion.md`:
  human-readable claim ceiling generated from the same evidence.
- Do not modify `pamssw/`, existing optimizer code, policies, walker code, or
  numerical configuration defaults.

### Task 1: Build the fail-closed validation artifact

**Files:**

- Create:
  `tests/integration/test_posterior_terminal_outcome_validation.py`
- Create:
  `runs/20260728-posterior-terminal-outcome-validation/run_validation.py`

- [ ] **Step 1: Write the single end-to-end failing test**

Create
`tests/integration/test_posterior_terminal_outcome_validation.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260728-posterior-terminal-outcome-validation"
    / "run_validation.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "posterior_terminal_outcome_validation",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load validation script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_writes_only_closed_replayable_evidence(tmp_path: Path) -> None:
    validator = _load_validator()
    evidence = validator.run_validation(tmp_path, source_commit="0" * 40)

    persisted = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    conclusion = (tmp_path / "conclusion.md").read_text(encoding="utf-8")

    assert persisted == evidence
    assert evidence["schema_version"] == 1
    assert evidence["git_commit"] == "0" * 40
    assert evidence["real_campaign"]["configuration"]["proposal_pool_size"] == 1
    assert evidence["real_campaign"]["configuration"]["batch_size"] > (
        evidence["real_campaign"]["configuration"]["max_workers"]
    ) > 1
    assert all(evidence["invariants"].values())
    assert [
        row["posterior_observed"] for row in evidence["terminal_matrix"]["rows"]
    ] == [True, True, True, True, False, False]
    assert evidence["terminal_matrix"]["live_posterior"] == (
        evidence["terminal_matrix"]["replayed_posterior"]
    )
    assert "does not compare starter-policy performance" in conclusion
```

- [ ] **Step 2: Run the test and verify the artifact is absent**

Run:

```bash
pytest -q tests/integration/test_posterior_terminal_outcome_validation.py
```

Expected: fail while loading
`runs/20260728-posterior-terminal-outcome-validation/run_validation.py`
because the script does not exist.

- [ ] **Step 3: Implement strict event parsing and common helpers**

Create
`runs/20260728-posterior-terminal-outcome-validation/run_validation.py`
with these imports and strict helpers:

```python
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
    "archive_version", "batch_id", "eligible_starter_ids", "policy_name",
    "policy_version", "probabilities", "record_type", "schema_version",
    "support_complete",
}
EXPECTED_ATTEMPT_FIELDS = {
    "action_id", "archive_version", "batch_id", "cost_is_exact",
    "discovered_against_snapshot", "evaluation_counts", "failure_reason",
    "force_budget", "force_evaluations", "inserted_into_archive",
    "landing_energy", "landing_entry_id", "policy_name", "policy_version",
    "posterior_observed", "random_seed", "record_type", "schema_version",
    "selection_probability", "slot_id", "starter_id",
    "status", "within_batch_collision",
}
EXPECTED_COMMIT_FIELDS = {
    "action_ids", "batch_id", "record_type", "schema_version",
}


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _require(bool(rows), "event log must not be empty")
    _require(all(isinstance(row, dict) for row in rows), "event rows must be objects")
    return rows


def _committed_batches(path: Path) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    seen_action_ids: set[str] = set()
    for row in _read_rows(path):
        kind = row.get("record_type")
        _require(row.get("schema_version") == SCHEMA_VERSION, "schema version drift")
        if kind == "policy_snapshot":
            _require(set(row) == EXPECTED_SNAPSHOT_FIELDS, "snapshot field drift")
            _require(active is None, "nested policy snapshot")
            probabilities = row["probabilities"]
            starters = row["eligible_starter_ids"]
            _require(row["support_complete"] is True, "incomplete policy support")
            _require(len(probabilities) == len(starters) > 0, "policy support mismatch")
            _require(
                all(isfinite(float(value)) and float(value) > 0 for value in probabilities),
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
            _require(action_id not in seen_action_ids, "duplicate action identifier")
            seen_action_ids.add(action_id)
            snapshot = active["snapshot"]
            _require(row["batch_id"] == snapshot["batch_id"], "batch mismatch")
            _require(row["policy_name"] == snapshot["policy_name"], "policy mismatch")
            _require(row["policy_version"] == snapshot["policy_version"], "policy version mismatch")
            _require(row["archive_version"] == snapshot["archive_version"], "archive version mismatch")
            probability_by_starter = dict(
                zip(snapshot["eligible_starter_ids"], snapshot["probabilities"])
            )
            _require(row["starter_id"] in probability_by_starter, "starter outside support")
            _require(
                isclose(
                    float(row["selection_probability"]),
                    float(probability_by_starter[row["starter_id"]]),
                    abs_tol=1e-15,
                ),
                "saved selection probability mismatch",
            )
            _require(
                sum(row["evaluation_counts"].values()) == row["force_evaluations"],
                "attempt evaluation counts do not close",
            )
            _require(row["cost_is_exact"] is True, "attempt cost is not exact")
            _require(row["evaluation_counts"]["unattributed"] == 0, "unattributed action call")
            active["attempts"].append(row)
        elif kind == "batch_commit":
            _require(set(row) == EXPECTED_COMMIT_FIELDS, "commit field drift")
            _require(active is not None, "commit outside a batch")
            attempts = active["attempts"]
            _require(bool(attempts), "empty committed batch")
            _require(row["batch_id"] == active["snapshot"]["batch_id"], "commit batch mismatch")
            _require(
                row["action_ids"] == [attempt["action_id"] for attempt in attempts],
                "commit action list mismatch",
            )
            batches.append({**active, "commit": row})
            active = None
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
    return {
        str(entry_id): list(posterior.counts(entry_id))
        for entry_id in entry_ids
    }
```

- [ ] **Step 4: Implement the real analytic campaign audit**

Add the following functions to the same script:

```python
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
    attempts = [
        attempt for batch in batches for attempt in batch["attempts"]
    ]
    diagnostics = json.loads(
        (run_directory / "optimizer_diagnostics.json").read_text(encoding="utf-8")
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
        "real_bootstrap_counts_nonnegative": all(value >= 0 for value in bootstrap_counts.values()),
        "real_bootstrap_counts_close": sum(bootstrap_counts.values()) == result.bootstrap_evaluations,
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
            observed
            == result.posterior_observed_attempts
            == replayed.completed_attempts
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
        "terminal_status_counts": dict(sorted(Counter(row["status"] for row in attempts).items())),
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
```

- [ ] **Step 5: Implement the deterministic terminal matrix**

Add these functions to the same script:

```python
def _matrix_archive() -> MinimaArchive:
    archive = MinimaArchive()
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
        ("completed", AttemptStatus.COMPLETED, _counts(EvaluationPurpose.DIRECTION_ORACLE, 3), True),
        ("invalid_after_pes", AttemptStatus.INVALID, _counts(EvaluationPurpose.BIASED_PROPOSAL_RELAX, 2), True),
        ("fragmented_after_pes", AttemptStatus.FRAGMENTED, _counts(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK, 1), True),
        ("budget_exhausted", AttemptStatus.BUDGET_EXHAUSTED, _counts(EvaluationPurpose.DIRECTION_ORACLE, 2), True),
        ("invalid_before_pes", AttemptStatus.INVALID, EvaluationCounts.zero(), False),
        ("worker_error", AttemptStatus.WORKER_ERROR, EvaluationCounts.zero(), False),
    )

    def worker(action, starter_state) -> AttemptResult:
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
        "matrix_one_attempt_per_outcome": (
            len(attempts) == len(outcomes) == len(cases)
        ),
        "matrix_action_ids_close": (
            [row["action_id"] for row in attempts]
            == [outcome.action_id for outcome in outcomes]
        ),
        "matrix_statuses_preserved": (
            [row["status"] for row in attempts]
            == [case[1].value for case in cases]
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
```

- [ ] **Step 6: Implement deterministic artifact writing and the CLI**

Complete the same script with:

```python
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
    encoded = json.dumps(
        evidence,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    evidence_path.write_text(encoded, encoding="utf-8")
    conclusion_path.write_text(_conclusion(evidence), encoding="utf-8")
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
```

- [ ] **Step 7: Run the focused test**

Run:

```bash
pytest -q tests/integration/test_posterior_terminal_outcome_validation.py
```

Expected: `1 passed`.

- [ ] **Step 8: Run the exploration integration tests**

Run:

```bash
pytest -q \
  tests/unit/test_ssw_attempt_worker.py \
  tests/integration/test_exploration_controller.py \
  tests/integration/test_posterior_exploration_runner.py \
  tests/integration/test_posterior_terminal_outcome_validation.py
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit the validator implementation**

Run:

```bash
git add \
  runs/20260728-posterior-terminal-outcome-validation/run_validation.py \
  tests/integration/test_posterior_terminal_outcome_validation.py
git commit -m "test: validate posterior terminal outcome closure"
```

Expected: one commit containing only the validator and its end-to-end test.

### Task 2: Generate and review commit-pinned evidence

**Files:**

- Create:
  `runs/20260728-posterior-terminal-outcome-validation/evidence.json`
- Create:
  `runs/20260728-posterior-terminal-outcome-validation/conclusion.md`

- [ ] **Step 1: Confirm the implementation worktree is clean**

Run:

```bash
git status --short
validation_source_commit="$(git rev-parse HEAD)"
printf '%s\n' "$validation_source_commit"
```

Expected: no status output, followed by the 40-character implementation
commit stored in the task-specific `validation_source_commit` shell variable.

- [ ] **Step 2: Execute the validator against that exact commit**

Run in the same shell used in Step 1:

```bash
python runs/20260728-posterior-terminal-outcome-validation/run_validation.py \
  --output-directory runs/20260728-posterior-terminal-outcome-validation \
  --expected-git-commit "$validation_source_commit" \
  --require-clean
```

Expected: exit code 0 and creation of only `evidence.json` and
`conclusion.md`. `evidence.json.git_commit` must equal the implementation
commit, not the later evidence commit.

- [ ] **Step 3: Review the generated evidence mechanically**

Run:

```bash
python -m json.tool \
  runs/20260728-posterior-terminal-outcome-validation/evidence.json
rg -n \
  \"does not compare starter-policy performance|unbiased|C60|PdO|git_commit\" \
  runs/20260728-posterior-terminal-outcome-validation/conclusion.md \
  runs/20260728-posterior-terminal-outcome-validation/evidence.json
```

Expected:

- valid JSON;
- every `invariants` value is `true`;
- the real campaign reports `proposal_pool_size: 1`;
- the terminal matrix reports `[true, true, true, true, false, false]`;
- conclusion and claim ceiling contain no policy-superiority, GPU, C60, or
  PdO claim.

- [ ] **Step 4: Re-run the focused and full suites**

Run:

```bash
pytest -q tests/integration/test_posterior_terminal_outcome_validation.py
pytest -q
```

Expected: the focused test passes and the complete repository suite passes
with no regression from the clean baseline.

- [ ] **Step 5: Commit only the reviewed evidence**

Run:

```bash
git add \
  runs/20260728-posterior-terminal-outcome-validation/evidence.json \
  runs/20260728-posterior-terminal-outcome-validation/conclusion.md
git commit -m "docs: record posterior terminal outcome evidence"
```

Expected: one evidence-only commit whose parent is the exact
`evidence.json.git_commit`.

### Task 3: Independent review and Pull Request handoff

**Files:**

- Review all files added by Tasks 1 and 2.
- Modify only those files if a review identifies a concrete defect.

- [ ] **Step 1: Run specification review**

Use a fresh review agent to compare:

```text
docs/superpowers/specs/2026-07-28-posterior-terminal-outcome-validation-design.md
```

against the two implementation commits. Require a finding list with exact
file and line references, or an explicit `APPROVED`.

- [ ] **Step 2: Run code-quality and scientific-claim review**

Use a second fresh review agent. Require it to check:

- no `pamssw/` production file changed;
- no numerical or policy parameter was added outside the finite validator;
- strict parser checks exact schema and action/cost closure;
- the matrix is labelled as controller semantics rather than PES physics;
- evidence commit ancestry is correct;
- no result is described as policy superiority or equilibrium-unbiased
  sampling.

- [ ] **Step 3: Fix only concrete findings and rerun verification**

For each accepted finding, first add or tighten the single end-to-end test,
confirm it fails, make the smallest validator/artifact change, and run:

```bash
pytest -q tests/integration/test_posterior_terminal_outcome_validation.py
pytest -q
git diff --check
```

Expected: all tests pass and `git diff --check` is silent. If evidence-affecting
code changes, repeat Task 2 from a clean implementation commit so the evidence
never points to stale code.

- [ ] **Step 4: Push the branch and create or update the Pull Request**

Run:

```bash
git push -u origin feature/posterior-terminal-outcome-validation
gh pr create \
  --base experiment/safe-lbfgs-history-depth-ablation \
  --head feature/posterior-terminal-outcome-validation \
  --title "Validate posterior terminal outcome accounting" \
  --body-file /tmp/posterior-terminal-outcome-validation-pr.md
```

The PR body must state:

- this is an analytic evidence milestone, not an algorithm change;
- production code is unchanged;
- exact focused/full test results;
- the implementation commit named by the evidence;
- the claim ceiling;
- the next planned experiment is the fixed-budget uniform versus current
  UCB-like starter-policy ablation.

- [ ] **Step 5: Report the handoff**

Report:

- PR URL;
- implementation and evidence commit SHAs;
- focused and full-suite results;
- real campaign bootstrap/action/total force-evaluation counts;
- terminal status distribution;
- whether live and replayed posteriors matched;
- explicitly unproven policy-performance and GPU-production questions.
