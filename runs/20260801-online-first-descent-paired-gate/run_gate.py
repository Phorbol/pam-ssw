#!/usr/bin/env python3
"""Run the G-E1 online first-descent paired action gate."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL_PATH = RUN_ROOT / "protocol.py"
SOURCE_RUNNER_PATH = (
    REPO_ROOT / "runs" / "20260731-current-action-first-passage" / "run_gate.py"
)
GUP0_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-uphill-relax-counterfactual-gate"
    / "run_gate.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_module(
    PROTOCOL_PATH,
    "_online_first_descent_protocol_runner",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _current_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tracked_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _zero_counts() -> dict[str, int]:
    from pamssw.accounting import EvaluationPurpose

    return {purpose.value: 0 for purpose in EvaluationPurpose}


def _add_counts(*counts: Mapping[str, int]) -> dict[str, int]:
    result = _zero_counts()
    for row in counts:
        for name, value in row.items():
            result[name] += int(value)
    return result


def compose_arm_cost(
    *,
    generation_counts: Mapping[str, int],
    quench_counts: Mapping[str, int],
    quench_reused: bool,
) -> dict[str, Any]:
    complete_counts = _add_counts(generation_counts, quench_counts)
    executed_counts = (
        _add_counts(generation_counts)
        if quench_reused
        else dict(complete_counts)
    )
    return {
        "new_executed_purpose_counts": executed_counts,
        "new_executed_force_evaluations": sum(executed_counts.values()),
        "complete_action_purpose_counts": complete_counts,
        "complete_action_force_evaluations": sum(complete_counts.values()),
        "quench_reused": bool(quench_reused),
    }


def build_prefix_trace(
    generation: Mapping[str, Any],
    *,
    state_hashes: Mapping[int, str],
) -> list[dict[str, Any]]:
    rows = []
    directions = generation["direction_trace"]
    for observed in generation["walk_step_trace"]:
        step = int(observed["step"])
        direction = directions[step - 1]
        rows.append(
            {
                "step": step,
                "selected_direction_sha256": direction[
                    "selected_direction_sha256"
                ],
                "executed_step_scale": direction["executed_step_scale"],
                "uphill_final_bias_weight": direction[
                    "uphill_final_bias_weight"
                ],
                "true_energy_eV": observed["true_energy_eV"],
                "state_sha256": state_hashes[step],
            }
        )
    return rows


def _case_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["system"]),
        str(row["state_id"]),
        int(row["seed"]),
        str(row["arm"]),
    )


def _runtime(*, state_source_root: Path):
    source = _load_module(SOURCE_RUNNER_PATH, "_g_e1_source_runner")
    audit = _load_module(source.FIXED_AUDIT_PATH, "_g_e1_fixed_audit")
    config_gate = _load_module(source.CONFIG_GATE_PATH, "_g_e1_config_gate")
    action_runner = _load_module(
        source.ACTION_RUNNER_PATH,
        "_g_e1_action_runner",
    )
    shooting_runner = _load_module(
        source.SHOOTING_RUNNER_PATH,
        "_g_e1_shooting_runner",
    )
    gup0 = _load_module(GUP0_RUNNER_PATH, "_g_e1_gup0_runner")
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    states = {}
    for system in protocol.SYSTEMS:
        for state_id in protocol.STATE_IDS:
            states[(system, state_id)] = source._load_starter(
                state_source_root=state_source_root,
                system=system,
                state_id=state_id,
                audit=audit,
                base_runner=base_runner,
            )

    from pamssw.calculators import ASECalculator

    calculator = ASECalculator(base_runner._calculator())
    return {
        "source": source,
        "audit": audit,
        "config_gate": config_gate,
        "action_runner": action_runner,
        "shooting_runner": shooting_runner,
        "gup0": gup0,
        "base_runner": base_runner,
        "calculator": calculator,
        "states": states,
    }


def _checkpoint_state_hashes(
    *,
    generation: Mapping[str, Any],
    case_dir: Path,
    starter_state,
    audit,
) -> dict[int, str]:
    from pamssw.io import read_state

    hashes = {}
    for step in range(1, int(generation["reached_macro_steps"]) + 1):
        state = read_state(
            case_dir
            / "macro_checkpoints"
            / f"step{step:03d}_checkpoint.xyz",
            fixed_mask=starter_state.fixed_mask,
        )
        hashes[step] = audit._state_sha256(state)
    return hashes


def _terminal_checkpoint(
    *,
    generation: Mapping[str, Any],
    case_dir: Path,
    starter_state,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    reached = int(generation["reached_macro_steps"])
    if reached:
        path = (
            case_dir
            / "macro_checkpoints"
            / f"step{reached:03d}_checkpoint.xyz"
        )
    else:
        path = case_dir / "macro_checkpoints" / "step000_starter.xyz"
        path.parent.mkdir(parents=True, exist_ok=True)
        runtime["base_runner"].write_state(path, starter_state)
    return {
        "horizon": reached,
        "checkpoint_path": str(path),
        "checkpoint_sha256": _sha256(path),
    }


def _run_generation(
    *,
    case: Mapping[str, Any],
    arm_dir: Path,
    starter_state,
    provenance: Mapping[str, Any],
    runtime: Mapping[str, Any],
    remaining_budget: int,
    observer=None,
) -> tuple[dict[str, Any], Any]:
    base_config = runtime["action_runner"]._base_config(
        runtime["config_gate"],
        str(case["system"]),
        int(case["seed"]),
        arm_dir,
    )
    generation = runtime["source"]._generate_action_path(
        state=starter_state,
        provenance=provenance,
        calculator=runtime["calculator"],
        base_config=base_config,
        system=str(case["system"]),
        state_id=str(case["state_id"]),
        seed=int(case["seed"]),
        arm=str(case["arm"]),
        case_dir=arm_dir,
        remaining_budget=remaining_budget,
        action_runner=runtime["action_runner"],
        shooting_runner=runtime["shooting_runner"],
        base_runner=runtime["base_runner"],
        walk_step_observer=observer,
    )
    return generation, base_config


def _run_terminal_quench(
    *,
    checkpoint: Mapping[str, Any],
    generation: Mapping[str, Any],
    starter_state,
    config,
    system: str,
    quench_dir: Path,
    remaining_budget: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return runtime["source"]._quench_checkpoint(
        source=checkpoint,
        starter_state=starter_state,
        starter_energy=float(generation["starter_energy_eV"]),
        calculator=runtime["calculator"],
        config=config,
        system=system,
        case_dir=quench_dir,
        remaining_budget=remaining_budget,
        base_runner=runtime["base_runner"],
    )


def _reused_quench(reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **reference,
        "force_evaluations": 0,
        "purpose_counts": _zero_counts(),
        "wall_time_s": 0.0,
        "evidence_origin": "reused_paired_reference",
        "algorithmic_quench_force_evaluations": int(
            reference["force_evaluations"]
        ),
        "algorithmic_quench_purpose_counts": dict(
            reference["purpose_counts"]
        ),
    }


def _landing_relation(
    *,
    early_quench: Mapping[str, Any],
    reference_quench: Mapping[str, Any],
    starter_state,
    config,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if early_quench.get("evidence_origin") == "reused_paired_reference":
        return "SAME_LANDING", {
            "landing_matcher_same": True,
            "landing_descriptor_same": True,
            "landing_descriptor_delta": 0.0,
        }
    return runtime["gup0"]._landing_relation(
        explicit=early_quench,
        relaxed=reference_quench,
        starter_state=starter_state,
        config=config,
    )


def _pair_record(
    *,
    case: Mapping[str, Any],
    output_dir: Path,
    runtime: Mapping[str, Any],
    total_new_fe: int,
    started: float,
    max_new_fe: int,
    max_wall_s: float,
) -> tuple[dict[str, Any], int]:
    system, state_id, seed, arm = _case_key(case)
    starter_state, provenance = runtime["states"][(system, state_id)]
    pair_dir = (
        output_dir
        / "pairs"
        / system
        / state_id
        / f"seed-{seed:08d}"
        / arm
    )
    reference_dir = pair_dir / "reference"
    early_dir = pair_dir / "first_descent"

    if perf_counter() - started > max_wall_s:
        raise RuntimeError("G-E1 kernel wall budget exhausted")
    reference, reference_config = _run_generation(
        case=case,
        arm_dir=reference_dir,
        starter_state=starter_state,
        provenance=provenance,
        runtime=runtime,
        remaining_budget=max_new_fe - total_new_fe,
    )
    total_new_fe += int(reference["generation_force_evaluations"])
    reference_checkpoint = _terminal_checkpoint(
        generation=reference,
        case_dir=reference_dir,
        starter_state=starter_state,
        runtime=runtime,
    )
    reference_quench = _run_terminal_quench(
        checkpoint=reference_checkpoint,
        generation=reference,
        starter_state=starter_state,
        config=reference_config,
        system=system,
        quench_dir=reference_dir / "terminal_quench",
        remaining_budget=max_new_fe - total_new_fe,
        runtime=runtime,
    )
    total_new_fe += int(reference_quench["force_evaluations"])

    observer = protocol.FirstDescentObserver(
        starter_energy_eV=float(reference["starter_energy_eV"]),
        tolerance_eV=float(reference_config.dedup_energy_tol),
    )
    early, early_config = _run_generation(
        case=case,
        arm_dir=early_dir,
        starter_state=starter_state,
        provenance=provenance,
        runtime=runtime,
        remaining_budget=max_new_fe - total_new_fe,
        observer=observer,
    )
    total_new_fe += int(early["generation_force_evaluations"])
    reference_hashes = _checkpoint_state_hashes(
        generation=reference,
        case_dir=reference_dir,
        starter_state=starter_state,
        audit=runtime["audit"],
    )
    early_hashes = _checkpoint_state_hashes(
        generation=early,
        case_dir=early_dir,
        starter_state=starter_state,
        audit=runtime["audit"],
    )
    reference_trace = build_prefix_trace(
        reference,
        state_hashes=reference_hashes,
    )
    early_trace = build_prefix_trace(early, state_hashes=early_hashes)
    prefix = protocol.compare_prefix(reference_trace, early_trace)
    for field, left, right in (
        (
            "starter_energy_eV",
            reference["starter_energy_eV"],
            early["starter_energy_eV"],
        ),
        ("anchor_sha256", reference["anchor_sha256"], early["anchor_sha256"]),
    ):
        if left != right:
            prefix["mismatches"].append(
                {
                    "step": None,
                    "field": field,
                    "reference": left,
                    "early": right,
                }
            )
    prefix["prefix_valid"] = not prefix["mismatches"]

    early_checkpoint = _terminal_checkpoint(
        generation=early,
        case_dir=early_dir,
        starter_state=starter_state,
        runtime=runtime,
    )
    no_trigger_same_terminal = bool(
        observer.trigger_step is None
        and reference["walk_termination_reason"]
        == early["walk_termination_reason"]
        and reference_hashes.get(int(reference["reached_macro_steps"]), provenance["state_sha256"])
        == early_hashes.get(int(early["reached_macro_steps"]), provenance["state_sha256"])
    )
    if observer.trigger_step is None and not no_trigger_same_terminal:
        prefix["prefix_valid"] = False
        prefix["mismatches"].append(
            {
                "step": None,
                "field": "untriggered_terminal",
                "reference": reference["walk_termination_reason"],
                "early": early["walk_termination_reason"],
            }
        )
    if not prefix["prefix_valid"]:
        _write_json(pair_dir / "prefix_failure.json", prefix)
        raise RuntimeError(f"G-E1 paired prefix drifted: {_case_key(case)}")

    if no_trigger_same_terminal:
        early_quench = _reused_quench(reference_quench)
        early_quench_counts = dict(reference_quench["purpose_counts"])
        early_quench_reused = True
    else:
        early_quench = _run_terminal_quench(
            checkpoint=early_checkpoint,
            generation=early,
            starter_state=starter_state,
            config=early_config,
            system=system,
            quench_dir=early_dir / "terminal_quench",
            remaining_budget=max_new_fe - total_new_fe,
            runtime=runtime,
        )
        total_new_fe += int(early_quench["force_evaluations"])
        early_quench_counts = dict(early_quench["purpose_counts"])
        early_quench_reused = False

    reference_cost = compose_arm_cost(
        generation_counts=reference["generation_purpose_counts"],
        quench_counts=reference_quench["purpose_counts"],
        quench_reused=False,
    )
    early_cost = compose_arm_cost(
        generation_counts=early["generation_purpose_counts"],
        quench_counts=early_quench_counts,
        quench_reused=early_quench_reused,
    )
    relation, relation_metrics = _landing_relation(
        early_quench=early_quench,
        reference_quench=reference_quench,
        starter_state=starter_state,
        config=reference_config,
        runtime=runtime,
    )
    tolerance = float(reference_config.dedup_energy_tol)
    early_delta = early_quench.get("landing_delta_eV")
    early_certified = bool(
        early_quench.get("label") == "ESCAPED_CERTIFIED"
        and early_delta is not None
        and float(early_delta) < -tolerance
    )
    pair_counts = _add_counts(
        reference_cost["new_executed_purpose_counts"],
        early_cost["new_executed_purpose_counts"],
    )
    record = {
        **case,
        "state_sha256": provenance["state_sha256"],
        "dedup_energy_tol_eV": tolerance,
        "triggered": observer.trigger_step is not None,
        "trigger_step": observer.trigger_step,
        "prefix_valid": prefix["prefix_valid"],
        "prefix_comparison": prefix,
        "reference_trace": reference_trace,
        "early_trace": early_trace,
        "reference_generation": reference,
        "early_generation": early,
        "reference_quench": reference_quench,
        "early_quench": early_quench,
        "reference_cost": reference_cost,
        "early_cost": early_cost,
        "reference_complete_action_fe": reference_cost[
            "complete_action_force_evaluations"
        ],
        "early_complete_action_fe": early_cost[
            "complete_action_force_evaluations"
        ],
        "complete_action_fe_saving": (
            reference_cost["complete_action_force_evaluations"]
            - early_cost["complete_action_force_evaluations"]
        ),
        "early_landing_certified": early_certified,
        "early_landing_delta_eV": early_delta,
        "reference_landing_delta_eV": reference_quench.get(
            "landing_delta_eV"
        ),
        "tradeoff_class": protocol.classify_tradeoff(
            early_delta,
            reference_quench.get("landing_delta_eV"),
            tolerance,
        ),
        "landing_relation": relation,
        **relation_metrics,
        "new_executed_purpose_counts": pair_counts,
        "new_executed_force_evaluations": sum(pair_counts.values()),
    }
    _write_json(pair_dir / "pair.json", record)
    print(
        f"[G-E1] {system} {state_id} seed={seed} arm={arm} "
        f"trigger={observer.trigger_step} prefix={record['prefix_valid']} "
        f"saving={record['complete_action_fe_saving']} "
        f"tradeoff={record['tradeoff_class']} "
        f"new_fe={record['new_executed_force_evaluations']}",
        flush=True,
    )
    return record, total_new_fe


def _aggregate(
    *,
    pairs,
    commit: str,
    wall_time_s: float,
    max_new_fe: int,
    max_wall_s: float,
) -> dict[str, Any]:
    decision = protocol.build_decision(pairs)
    counts = _zero_counts()
    for pair in pairs:
        counts = _add_counts(counts, pair["new_executed_purpose_counts"])
    tradeoffs = Counter(
        str(pair["tradeoff_class"])
        for pair in pairs
        if pair["triggered"]
    )
    return {
        "schema_version": 1,
        "execution_commit": commit,
        "cohort": {
            "pair_count": len(pairs),
            "seeds": sorted({int(row["seed"]) for row in pairs}),
            "systems": sorted({str(row["system"]) for row in pairs}),
            "state_ids": sorted({str(row["state_id"]) for row in pairs}),
            "arms": sorted({str(row["arm"]) for row in pairs}),
        },
        "pairs": pairs,
        "aggregate": {
            **decision,
            "new_executed_force_evaluations": sum(counts.values()),
            "new_executed_purpose_counts": counts,
            "triggered_tradeoff_counts": dict(sorted(tradeoffs.items())),
            "wall_time_s": float(wall_time_s),
        },
        "budgets": {
            "max_new_force_evaluations": int(max_new_fe),
            "max_kernel_wall_time_s": float(max_wall_s),
        },
        "production_default_changed": False,
        "claim_ceiling": (
            "fresh paired complete-action first-descent gate on 24 C60/PdO "
            "D0/K4 actions; no long-search or production claim"
        ),
    }


def run(
    *,
    output_dir: Path,
    expected_commit: str | None,
    state_source_root: Path,
    limit_pairs: int | None,
    max_new_fe: int,
    max_wall_s: float,
) -> dict[str, Any]:
    commit = _current_commit()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if limit_pairs is None and not _tracked_clean():
        raise RuntimeError("full G-E1 gate requires a clean tracked worktree")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cases = protocol.case_matrix()
    if limit_pairs is not None:
        if limit_pairs <= 0:
            raise ValueError("--limit-pairs must be positive")
        cases = cases[:limit_pairs]
    runtime = _runtime(state_source_root=state_source_root.resolve())
    output_dir.mkdir(parents=True)
    started = perf_counter()
    total_new_fe = 0
    pairs = []
    for case in cases:
        pair, total_new_fe = _pair_record(
            case=case,
            output_dir=output_dir,
            runtime=runtime,
            total_new_fe=total_new_fe,
            started=started,
            max_new_fe=max_new_fe,
            max_wall_s=max_wall_s,
        )
        pairs.append(pair)
        partial = _aggregate(
            pairs=pairs,
            commit=commit,
            wall_time_s=perf_counter() - started,
            max_new_fe=max_new_fe,
            max_wall_s=max_wall_s,
        )
        _write_json(output_dir / "partial.json", partial)
        if total_new_fe > max_new_fe:
            raise RuntimeError("G-E1 force-evaluation budget exhausted")
    evidence = _aggregate(
        pairs=pairs,
        commit=commit,
        wall_time_s=perf_counter() - started,
        max_new_fe=max_new_fe,
        max_wall_s=max_wall_s,
    )
    if evidence["aggregate"]["new_executed_force_evaluations"] != total_new_fe:
        raise RuntimeError("G-E1 executed force-evaluation ledger does not close")
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def check_evidence(path: Path, *, require_full: bool = True) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    pairs = evidence["pairs"]
    if require_full and {_case_key(row) for row in pairs} != {
        _case_key(row) for row in protocol.case_matrix()
    }:
        raise RuntimeError("G-E1 evidence does not contain the fresh cohort")
    decision = protocol.build_decision(pairs)
    for name, value in decision.items():
        if evidence["aggregate"][name] != value:
            raise RuntimeError(f"G-E1 decision aggregate drifted: {name}")
    counts = _zero_counts()
    for pair in pairs:
        if not pair["prefix_valid"]:
            raise RuntimeError("G-E1 contains an invalid paired prefix")
        counts = _add_counts(counts, pair["new_executed_purpose_counts"])
    if counts != evidence["aggregate"]["new_executed_purpose_counts"]:
        raise RuntimeError("G-E1 purpose aggregate drifted")
    if sum(counts.values()) != evidence["aggregate"][
        "new_executed_force_evaluations"
    ]:
        raise RuntimeError("G-E1 executed force-evaluation total drifted")
    if counts["unattributed"] != 0:
        raise RuntimeError("G-E1 contains unattributed force evaluations")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUN_ROOT / "output")
    parser.add_argument("--expected-commit")
    parser.add_argument("--state-source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument(
        "--max-new-fe",
        type=int,
        default=protocol.MAX_NEW_FORCE_EVALUATIONS,
    )
    parser.add_argument(
        "--max-wall-s",
        type=float,
        default=protocol.MAX_KERNEL_WALL_TIME_S,
    )
    parser.add_argument("--check-evidence", type=Path)
    args = parser.parse_args()
    if args.check_evidence is not None:
        checked = check_evidence(
            args.check_evidence,
            require_full=args.limit_pairs is None,
        )
        print(json.dumps(checked["aggregate"], indent=2, sort_keys=True))
        return
    evidence = run(
        output_dir=args.output_dir,
        expected_commit=args.expected_commit,
        state_source_root=args.state_source_root,
        limit_pairs=args.limit_pairs,
        max_new_fe=args.max_new_fe,
        max_wall_s=args.max_wall_s,
    )
    print(json.dumps(evidence["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
