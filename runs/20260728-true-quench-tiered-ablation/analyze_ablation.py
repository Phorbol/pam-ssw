#!/usr/bin/env python3
"""Analyze the fixed raw-landing tiered true-quench ablation."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pamssw.archive import MinimaArchive
from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
RAW_DIR = RUN_ROOT / "output"
SYSTEMS = ("c60", "pdo")
TASKS_PER_SYSTEM = 16
STAGES = (("loose", 0.05), ("refine", 0.01))
ARMS = (
    ("scipy-lbfgsb", "scipy-lbfgsb", None),
    ("safe-lbfgs-total", "safe-lbfgs-total", 10),
    ("ase-fire2", "ase-fire2", None),
)
SCIPY_ARM = "scipy-lbfgsb"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _state(payload: Mapping[str, Any]) -> State:
    return State(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=(
            None
            if payload.get("cell") is None
            else np.asarray(payload["cell"], dtype=float)
        ),
        pbc=tuple(bool(value) for value in payload["pbc"]),
        fixed_mask=np.asarray(payload["fixed_mask"], dtype=bool),
    )


def _position_sha256(positions: object) -> str:
    coordinates = np.asarray(positions, dtype=np.dtype("<f8"))
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3)")
    canonical = np.array(
        coordinates, dtype=np.dtype("<f8"), order="C", copy=True
    )
    digest = sha256()
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _validate_summary(summary: object) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError("summary must be an object")
    if (
        summary.get("schema_version") != 1
        or summary.get("stage_a_task_count") != 32
        or summary.get("stage_b_row_count") != 96
        or summary.get("stage_c_row_count") != 96
    ):
        raise ValueError("summary does not describe the fixed tiered matrix")
    expected_arms = [
        {
            "arm_id": arm_id,
            "optimizer": optimizer,
            "safe_history_limit": history,
        }
        for arm_id, optimizer, history in ARMS
    ]
    if summary.get("arms") != expected_arms:
        raise ValueError("summary arms violate the fixed optimizer contract")
    protocol = summary.get("protocol")
    if not isinstance(protocol, Mapping) or protocol != {
        "capture_trials_per_system": 16,
        "capture_policy_conditioned": True,
        "objective": "true_mace_pes_no_bias_no_softening",
        "maxiter": 400,
        "stages": [
            {"stage": "loose", "fmax_eV_per_A": 0.05},
            {"stage": "refine", "fmax_eV_per_A": 0.01},
        ],
        "refine_common_start": "per-task loose scipy-lbfgsb endpoint",
    }:
        raise ValueError("summary protocol violates the fixed tiered contract")
    stage_a = summary.get("stage_a")
    if not isinstance(stage_a, Mapping) or set(stage_a) != set(SYSTEMS):
        raise ValueError("Stage A summary is incomplete")
    for system in SYSTEMS:
        stats = stage_a[system]["stats"]
        counts = stage_a[system]["purpose_counts"]
        if (
            stats.get("n_trials") != TASKS_PER_SYSTEM
            or sum(int(value) for value in counts.values())
            != int(stats["force_evaluations"])
            or int(counts.get("unattributed", -1)) != 0
        ):
            raise ValueError(f"{system} Stage A purpose accounting is open")
    return summary


def _validate_corpus(corpus: object) -> dict[tuple[str, int], Mapping[str, Any]]:
    if not isinstance(corpus, Mapping) or corpus.get(
        "capture_policy_conditioned"
    ) is not True:
        raise ValueError("raw-landing corpus does not state its capture policy")
    systems = corpus.get("systems")
    if not isinstance(systems, Mapping) or set(systems) != set(SYSTEMS):
        raise ValueError("raw-landing corpus systems are incomplete")
    entries = {}
    for system in SYSTEMS:
        manifest = systems[system]
        if (
            manifest.get("task_count") != TASKS_PER_SYSTEM
            or manifest.get("capture_policy_conditioned") is not True
            or len(manifest.get("entries", [])) != TASKS_PER_SYSTEM
        ):
            raise ValueError(f"{system} raw-landing corpus is incomplete")
        for item in manifest["entries"]:
            key = (system, int(item["task_index"]))
            if key in entries:
                raise ValueError(f"duplicate corpus task: {key}")
            state = _state(item["state"])
            if item.get("initial_positions_sha256") != _position_sha256(
                state.positions
            ):
                raise ValueError("corpus position hash does not match coordinates")
            entries[key] = item
    expected = {
        (system, task_index)
        for system in SYSTEMS
        for task_index in range(TASKS_PER_SYSTEM)
    }
    if set(entries) != expected:
        raise ValueError("raw-landing corpus does not cover the fixed tasks")
    return entries


def _validate_row(
    row: object,
    stage_contract: Mapping[str, float],
    arm_contract: Mapping[str, tuple[str, int | None]],
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("row must be an object")
    stage = row.get("stage")
    arm_id = row.get("arm_id")
    if stage not in stage_contract or arm_id not in arm_contract:
        raise ValueError("row has an unknown stage or arm")
    optimizer, history = arm_contract[str(arm_id)]
    if (
        row.get("optimizer") != optimizer
        or row.get("safe_history_limit") != history
        or row.get("fmax_eV_per_A") != stage_contract[str(stage)]
        or row.get("maxiter") != 400
        or row.get("objective") != "true_mace_pes_no_bias_no_softening"
    ):
        raise ValueError("row violates the fixed true-quench protocol")
    task_index = row.get("task_index")
    if (
        row.get("system") not in SYSTEMS
        or isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or not 0 <= task_index < TASKS_PER_SYSTEM
    ):
        raise ValueError("row task identity is invalid")
    initial = row.get("initial")
    final = row.get("final")
    telemetry = row.get("telemetry")
    counts = row.get("purpose_count_delta")
    if not all(
        isinstance(value, Mapping)
        for value in (initial, final, telemetry, counts)
    ):
        raise ValueError("row result components are incomplete")
    _finite(initial.get("energy_eV"), "initial energy")
    _finite(initial.get("max_active_force_eV_per_A"), "initial force")
    _finite(final.get("energy_eV"), "final energy")
    _finite(final.get("max_active_force_eV_per_A"), "final force")
    initial_state = _state(initial["state"])
    final_state = _state(final["state"])
    if (
        initial.get("positions_sha256")
        != _position_sha256(initial_state.positions)
        or final.get("positions_sha256")
        != _position_sha256(final_state.positions)
    ):
        raise ValueError("row position hash does not match coordinates")
    if (
        initial_state.n_atoms != final_state.n_atoms
        or not np.array_equal(initial_state.numbers, final_state.numbers)
    ):
        raise ValueError("row initial and final states are incompatible")
    evaluator_calls = int(row.get("evaluator_calls"))
    if not (
        evaluator_calls == int(telemetry.get("evaluator_calls"))
        == int(counts.get("landing_true_quench"))
        == sum(int(value) for value in counts.values())
        and int(counts.get("unattributed")) == 0
    ):
        raise ValueError("row purpose accounting does not close")
    return row


def _validated_rows(
    rows: object,
    corpus: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], Mapping[str, Any]]:
    if not isinstance(rows, list) or len(rows) != 192:
        raise ValueError("rows file does not contain the fixed 192-row matrix")
    stage_contract = dict(STAGES)
    arm_contract = {
        arm_id: (optimizer, history)
        for arm_id, optimizer, history in ARMS
    }
    by_key = {}
    for item in rows:
        row = _validate_row(item, stage_contract, arm_contract)
        key = (
            str(row["stage"]),
            str(row["system"]),
            str(row["arm_id"]),
            int(row["task_index"]),
        )
        if key in by_key:
            raise ValueError(f"duplicate row: {key}")
        by_key[key] = row
    expected = {
        (stage, system, arm_id, task_index)
        for stage, _ in STAGES
        for system in SYSTEMS
        for arm_id, _, _ in ARMS
        for task_index in range(TASKS_PER_SYSTEM)
    }
    if set(by_key) != expected:
        raise ValueError("rows do not cover the fixed tiered matrix")

    for system in SYSTEMS:
        for task_index in range(TASKS_PER_SYSTEM):
            corpus_hash = corpus[(system, task_index)][
                "initial_positions_sha256"
            ]
            loose_initials = {
                by_key[("loose", system, arm_id, task_index)]["initial"][
                    "positions_sha256"
                ]
                for arm_id, _, _ in ARMS
            }
            if loose_initials != {corpus_hash}:
                raise ValueError("loose arms do not share the fixed raw landing")
            scipy_final = by_key[
                ("loose", system, SCIPY_ARM, task_index)
            ]["final"]["positions_sha256"]
            refine_initials = {
                by_key[("refine", system, arm_id, task_index)]["initial"][
                    "positions_sha256"
                ]
                for arm_id, _, _ in ARMS
            }
            if refine_initials != {scipy_final}:
                raise ValueError(
                    "refine arms do not share the fixed SciPy loose endpoint"
                )
    return by_key


def _same_basin(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    energy_tol: float,
    rmsd_tol: float,
) -> tuple[bool, float, float]:
    energy_delta = abs(
        float(first["final"]["energy_eV"])
        - float(second["final"]["energy_eV"])
    )
    rmsd = MinimaArchive._rmsd(
        _state(first["final"]["state"]), _state(second["final"]["state"])
    )
    return energy_delta <= energy_tol and rmsd <= rmsd_tol, energy_delta, rmsd


def analyze(raw_dir: Path) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    summary = _validate_summary(_read_json(raw_dir / "summary.json"))
    corpus = _validate_corpus(_read_json(raw_dir / "corpus.json"))
    rows = _validated_rows(_read_json(raw_dir / "rows.json"), corpus)
    by_stage: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for stage, fmax in STAGES:
        by_stage[stage] = {}
        paired[stage] = {}
        for system in SYSTEMS:
            by_stage[stage][system] = {}
            paired[stage][system] = {}
            config = summary["stage_a"][system]["effective_config"]
            energy_tol = float(config["dedup_energy_tol"])
            rmsd_tol = float(config["dedup_rmsd_tol"])
            for arm_id, _, _ in ARMS:
                arm_rows = [
                    rows[(stage, system, arm_id, task_index)]
                    for task_index in range(TASKS_PER_SYSTEM)
                ]
                by_stage[stage][system][arm_id] = {
                    "task_count": TASKS_PER_SYSTEM,
                    "certificate_count": sum(
                        0.0
                        <= float(row["final"]["max_active_force_eV_per_A"])
                        <= fmax
                        for row in arm_rows
                    ),
                    "evaluator_calls": sum(
                        int(row["evaluator_calls"]) for row in arm_rows
                    ),
                    "wall_time_s": sum(
                        float(row["wall_time_s"]) for row in arm_rows
                    ),
                    "termination_reasons": dict(
                        sorted(
                            Counter(
                                str(row["termination_reason"])
                                for row in arm_rows
                            ).items()
                        )
                    ),
                }
            scipy_rows = [
                rows[(stage, system, SCIPY_ARM, task_index)]
                for task_index in range(TASKS_PER_SYSTEM)
            ]
            for arm_id, _, _ in ARMS:
                if arm_id == SCIPY_ARM:
                    continue
                pairs = []
                for task_index, scipy_row in enumerate(scipy_rows):
                    arm_row = rows[(stage, system, arm_id, task_index)]
                    same, energy_delta, rmsd = _same_basin(
                        scipy_row,
                        arm_row,
                        energy_tol=energy_tol,
                        rmsd_tol=rmsd_tol,
                    )
                    pairs.append(
                        {
                            "task_index": task_index,
                            "same_basin_current_archive_semantics": same,
                            "absolute_terminal_energy_delta_eV": energy_delta,
                            "terminal_rmsd_A": rmsd,
                            "evaluator_call_delta_vs_scipy": (
                                int(arm_row["evaluator_calls"])
                                - int(scipy_row["evaluator_calls"])
                            ),
                        }
                    )
                paired[stage][system][arm_id] = {
                    "same_basin_count": sum(
                        bool(item["same_basin_current_archive_semantics"])
                        for item in pairs
                    ),
                    "different_basin_count": sum(
                        not bool(item["same_basin_current_archive_semantics"])
                        for item in pairs
                    ),
                    "pairs": pairs,
                }
    return {
        "schema_version": 1,
        "ledger": {
            "stage_a_tasks": 32,
            "stage_b_rows": 96,
            "stage_c_rows": 96,
            "purpose_closure_validated": True,
        },
        "by_stage_system_arm": by_stage,
        "paired": paired,
        "same_basin_scope": (
            "current per-system archive energy and RMSD semantics only"
        ),
        "claim_boundary": (
            "Stage A raw landings are conditioned on the fixed capture policy "
            "(seed42, safe-LBFGS proposal, first 16 macro trials), not an "
            "unconditional PES landing distribution. Stage B is a fixed-start "
            "local true-quench comparison. Stage C is conditional on each "
            "task's SciPy loose endpoint and is not an independent global-search "
            "comparison."
        ),
    }


def render_conclusion(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# Fixed raw-landing tiered true-quench ablation",
        "",
        "Stage A captures are conditioned on the fixed seed-42, safe-LBFGS proposal, first-16-trial capture policy.",
        "",
        "| Stage | System | Arm | Certificates | Evaluator calls | Wall s |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for stage, _ in STAGES:
        for system in SYSTEMS:
            for arm_id, _, _ in ARMS:
                row = evidence["by_stage_system_arm"][stage][system][arm_id]
                lines.append(
                    f"| {stage} | {system} | {arm_id} | "
                    f"{row['certificate_count']}/16 | "
                    f"{row['evaluator_calls']} | {row['wall_time_s']:.2f} |"
                )
    lines.extend(
        [
            "",
            "Same-basin labels use only the current per-system archive energy and RMSD semantics.",
            "",
            "Stage B is a fixed raw-landing local true-quench comparison. Stage C shares each task's SciPy loose endpoint, so it measures conditional refinement rather than an independent optimizer or global-search outcome.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = analyze(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "evidence.json", evidence)
    (args.output_dir / "conclusion.md").write_text(
        render_conclusion(evidence), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
