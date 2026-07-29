#!/usr/bin/env python3
"""Run the paired C60 energy-bounded-anchor terminal ablation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PRIOR_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-anchor-consistent-direction-ablation"
    / "run_ablation.py"
)
STATE_IDS = ("intermediate_accepted", "plateau_accepted")
SEEDS = (42, 43, 44)
ARMS: dict[str, dict[str, object]] = {
    "anchor_lanczos": {
        "direction_selection_mode": "anchor_krylov",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
    "energy_bounded_anchor": {
        "direction_selection_mode": "energy_bounded_anchor",
        "block_krylov_blocks": 1,
        "block_krylov_depth": 12,
    },
}
EXPECTED_HVP_PER_SELECTION = {
    "anchor_lanczos": 12,
    "energy_bounded_anchor": 12,
}
FORBIDDEN_PURPOSES = {
    "starter_true_quench",
    "bootstrap_true_quench",
    "unattributed",
}
MEANINGFUL_ENERGY_DROP_EV = 0.001


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def case_matrix() -> list[dict[str, Any]]:
    return [
        {"state_id": state_id, "seed": seed, "arm": arm}
        for state_id in STATE_IDS
        for seed in SEEDS
        for arm in ARMS
    ]


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _meaningful(row: Mapping[str, Any]) -> bool:
    return bool(
        row["certificate"]
        and row["is_new_basin"]
        and _finite(row["landing_delta_eV"], "landing delta")
        < -MEANINGFUL_ENERGY_DROP_EV
    )


def _validate_direction_trace(
    *,
    arm: str,
    direction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if not direction_rows:
        raise RuntimeError("direction diagnostics contain no selections")
    expected_hvp = EXPECTED_HVP_PER_SELECTION[arm]
    expected_kind = (
        "energy_bounded_anchor"
        if arm == "energy_bounded_anchor"
        else "block_ritz"
    )
    direction_force_evaluations = 0
    for index, row in enumerate(direction_rows, start=1):
        selection_delta = int(
            row.get("oracle_selection_force_evaluations_delta", -1)
        )
        if (
            selection_delta != 2 * expected_hvp
            or row.get("selected_kind") != expected_kind
            or row.get("candidate_count") != 0
            or row.get("krylov_blocks") != 1
            or row.get("krylov_depth") != 12
            or row.get("krylov_initial_basis_columns") != [1]
            or row.get("krylov_hvp_requested") != expected_hvp
            or row.get("krylov_hvp_consumed") != expected_hvp
            or row.get("krylov_hvp_count") != expected_hvp
        ):
            raise RuntimeError(
                f"direction row {index} violates the fixed-HVP contract"
            )
        _finite(row.get("anchor_cosine"), "anchor cosine")
        _finite(row.get("selected_curvature"), "selected curvature")
        _finite(row.get("true_curvature"), "true curvature")
        if arm == "energy_bounded_anchor":
            for key in (
                "energy_bounded_anchor_overlap",
                "energy_bounded_anchor_curvature_limit",
                "energy_bounded_anchor_quadratic_energy",
                "energy_bounded_anchor_true_curvature",
                "energy_bounded_anchor_exact_curvature",
                "energy_bounded_anchor_step_scale",
                "energy_bounded_anchor_energy_target",
            ):
                _finite(row.get(key), key)
            if row.get("energy_bounded_anchor_feasible") not in {
                True,
                False,
            }:
                raise RuntimeError("energy-bound feasibility is missing")
            if row.get("energy_bounded_anchor_active") not in {
                True,
                False,
            }:
                raise RuntimeError("energy-bound activity is missing")
            if (
                abs(
                    _finite(row["anchor_cosine"], "anchor cosine")
                    - _finite(
                        row["energy_bounded_anchor_overlap"],
                        "bounded overlap",
                    )
                )
                > 1.0e-10
            ):
                raise RuntimeError("recorded anchor overlaps disagree")
            if (
                abs(
                    _finite(row["true_curvature"], "true curvature")
                    - _finite(
                        row["energy_bounded_anchor_true_curvature"],
                        "bounded true curvature",
                    )
                )
                > 1.0e-9
            ):
                raise RuntimeError("recorded true curvatures disagree")
            if (
                row["energy_bounded_anchor_feasible"]
                and _finite(
                    row["energy_bounded_anchor_quadratic_energy"],
                    "quadratic energy",
                )
                > _finite(
                    row["energy_bounded_anchor_energy_target"],
                    "energy target",
                )
                + 1.0e-8
            ):
                raise RuntimeError("feasible direction exceeds energy target")
        direction_force_evaluations += selection_delta
    return {
        "selection_count": len(direction_rows),
        "direction_oracle_force_evaluations": (
            direction_force_evaluations
        ),
        "hvp_count": len(direction_rows) * expected_hvp,
    }


def _configured_prior_runner():
    prior = _load_module(
        PRIOR_RUNNER_PATH,
        "_energy_bounded_anchor_prior_runner",
    )
    prior.ARMS = ARMS
    prior.EXPECTED_HVP_PER_SELECTION = EXPECTED_HVP_PER_SELECTION
    prior._validate_direction_trace = _validate_direction_trace
    return prior


def _validate_row(row: Mapping[str, Any]) -> None:
    arm = str(row.get("arm"))
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if row.get("status") != "completed":
        raise ValueError("case is not complete")
    if row.get("certificate") is not True:
        raise ValueError("terminal quench certificate is missing")
    purposes = row.get("purpose_counts")
    if not isinstance(purposes, Mapping):
        raise ValueError("purpose ledger is missing")
    if any(
        int(purposes.get(purpose, -1)) != 0
        for purpose in FORBIDDEN_PURPOSES
    ):
        raise ValueError("forbidden purpose has nonzero cost")
    if (
        sum(int(value) for value in purposes.values())
        != row.get("force_evaluations")
    ):
        raise ValueError("purpose ledger does not close")
    selections = int(row.get("direction_selection_count", -1))
    if (
        selections <= 0
        or row.get("direction_hvp_count") != selections * 12
        or purposes.get("direction_oracle") != selections * 24
    ):
        raise ValueError("direction ledger does not close")
    trace = row.get("direction_trace")
    if not isinstance(trace, list) or len(trace) != selections:
        raise ValueError("direction trace count does not close")
    _validate_direction_trace(arm=arm, direction_rows=trace)


def build_evidence(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = {
        (case["state_id"], case["seed"], case["arm"])
        for case in case_matrix()
    }
    observed = {
        (row.get("state_id"), row.get("seed"), row.get("arm"))
        for row in rows
    }
    if (
        len(rows) != 12
        or len(observed) != len(rows)
        or observed != expected
    ):
        raise ValueError("evidence requires the exact 12-case cohort")
    for row in rows:
        _validate_row(row)

    arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        traces = [
            trace
            for row in arm_rows
            for trace in row["direction_trace"]
        ]
        result: dict[str, Any] = {
            "completed_cases": len(arm_rows),
            "certificate_count": sum(
                bool(row["certificate"]) for row in arm_rows
            ),
            "fallback_count": sum(
                bool(row["fallback_used"]) for row in arm_rows
            ),
            "new_basin_count": sum(
                bool(row["is_new_basin"]) for row in arm_rows
            ),
            "meaningful_outcome_count": sum(
                _meaningful(row) for row in arm_rows
            ),
            "direction_selection_count": len(traces),
            "direction_hvp_count": len(traces) * 12,
            "total_force_evaluations": sum(
                int(row["force_evaluations"]) for row in arm_rows
            ),
            "direction_force_evaluations": sum(
                int(row["purpose_counts"]["direction_oracle"])
                for row in arm_rows
            ),
            "median_landing_delta_eV": _median(
                [
                    _finite(
                        row["landing_delta_eV"],
                        "landing delta",
                    )
                    for row in arm_rows
                ]
            ),
            "median_anchor_overlap": _median(
                [
                    _finite(trace["anchor_cosine"], "anchor cosine")
                    for trace in traces
                ]
            ),
            "median_true_curvature": _median(
                [
                    _finite(trace["true_curvature"], "true curvature")
                    for trace in traces
                ]
            ),
            "state_results": {
                state_id: {
                    "meaningful_outcome_count": sum(
                        _meaningful(row)
                        for row in arm_rows
                        if row["state_id"] == state_id
                    ),
                    "new_basin_count": sum(
                        bool(row["is_new_basin"])
                        for row in arm_rows
                        if row["state_id"] == state_id
                    ),
                }
                for state_id in STATE_IDS
            },
        }
        if arm == "energy_bounded_anchor":
            result.update(
                {
                    "feasible_selection_count": sum(
                        bool(
                            trace[
                                "energy_bounded_anchor_feasible"
                            ]
                        )
                        for trace in traces
                    ),
                    "active_selection_count": sum(
                        bool(
                            trace["energy_bounded_anchor_active"]
                        )
                        for trace in traces
                    ),
                    "median_quadratic_energy_eV": _median(
                        [
                            _finite(
                                trace[
                                    "energy_bounded_anchor_quadratic_energy"
                                ],
                                "quadratic energy",
                            )
                            for trace in traces
                        ]
                    ),
                    "median_exact_anchor_curvature": _median(
                        [
                            _finite(
                                trace[
                                    "energy_bounded_anchor_exact_curvature"
                                ],
                                "exact anchor curvature",
                            )
                            for trace in traces
                        ]
                    ),
                    "maximum_feasible_quadratic_energy_eV": max(
                        _finite(
                            trace[
                                "energy_bounded_anchor_quadratic_energy"
                            ],
                            "quadratic energy",
                        )
                        for trace in traces
                        if trace[
                            "energy_bounded_anchor_feasible"
                        ]
                    ),
                }
            )
        arm_results[arm] = result

    purpose_totals = {
        purpose: sum(
            int(row["purpose_counts"][purpose]) for row in rows
        )
        for purpose in rows[0]["purpose_counts"]
    }
    return {
        "schema_version": 1,
        "cohort": {
            "states": list(STATE_IDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "completed_cases": len(rows),
        },
        "certificate_count": sum(
            bool(row["certificate"]) for row in rows
        ),
        "meaningful_outcome_count": sum(
            _meaningful(row) for row in rows
        ),
        "meaningful_energy_drop_threshold_eV": (
            MEANINGFUL_ENERGY_DROP_EV
        ),
        "arm_results": arm_results,
        "totals": {
            "force_evaluations": sum(
                int(row["force_evaluations"]) for row in rows
            ),
            "purpose_counts": purpose_totals,
            "generation_wall_time_s": sum(
                _finite(
                    row["generation_wall_time_s"],
                    "generation wall time",
                )
                for row in rows
            ),
            "quench_wall_time_s": sum(
                _finite(
                    row["quench_wall_time_s"],
                    "quench wall time",
                )
                for row in rows
            ),
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "descriptive paired three-seed direction-mechanism audit; "
            "no direction mode or selector is promoted"
        ),
        "cases": list(rows),
    }


def load_and_validate_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rebuilt = build_evidence(payload["cases"])
    for key in (
        "cohort",
        "certificate_count",
        "meaningful_outcome_count",
        "arm_results",
        "totals",
        "production_default_changed",
    ):
        if payload.get(key) != rebuilt[key]:
            raise ValueError(f"stored evidence differs at {key}")
    for row in payload["cases"]:
        for path_key, hash_key in (
            ("escape_path", "escape_sha256"),
            ("landing_path", "landing_sha256"),
        ):
            artifact = REPO_ROOT / str(row[path_key])
            if not artifact.is_file():
                raise ValueError(f"missing artifact: {artifact}")
            if _sha256(artifact) != row[hash_key]:
                raise ValueError(f"artifact hash mismatch: {artifact}")
    return payload


def run(
    *,
    output_dir: Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    prior = _configured_prior_runner()
    (
        states,
        shared_calculator,
        base_runner,
        state_provenance,
        shared_provenance,
    ) = prior._load_locked_runtime()

    rows: list[dict[str, Any]] = []
    for case in case_matrix():
        state_id = str(case["state_id"])
        seed = int(case["seed"])
        arm = str(case["arm"])
        print(
            f"[energy-bounded-anchor] {state_id} seed={seed} "
            f"arm={arm}",
            flush=True,
        )
        case_dir = (
            output_dir
            / "cases"
            / f"{state_id}-seed{seed}-{arm}"
        )
        row = prior._run_case(
            state=states[state_id],
            state_provenance=state_provenance[state_id],
            shared_calculator=shared_calculator,
            base_runner=base_runner,
            state_id=state_id,
            seed=seed,
            arm=arm,
            case_dir=case_dir,
        )
        _validate_row(row)
        rows.append(row)
        _write_json(
            output_dir / "raw.json",
            {
                "schema_version": 1,
                "execution_commit": actual_commit,
                "shared_provenance": shared_provenance,
                "state_provenance": state_provenance,
                "cases": rows,
            },
        )

    evidence = build_evidence(rows)
    evidence.update(
        {
            "execution_commit": actual_commit,
            "shared_provenance": shared_provenance,
            "state_provenance": state_provenance,
        }
    )
    _write_json(output_dir / "evidence.json", evidence)
    load_and_validate_evidence(output_dir / "evidence.json")
    return evidence


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "output",
    )
    parser.add_argument("--expected-git-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = run(
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
    )
    print(
        json.dumps(
            {
                "cohort": evidence["cohort"],
                "arm_results": evidence["arm_results"],
                "totals": evidence["totals"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
