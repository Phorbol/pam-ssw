#!/usr/bin/env python3
"""True-quench accepted optimizer frames for G-UP1 first passage."""

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
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL_PATH = RUN_ROOT / "protocol.py"
GUP0_RUNNER_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-uphill-relax-counterfactual-gate"
    / "run_gate.py"
)
GUP0_SUMMARY_PATH = GUP0_RUNNER_PATH.with_name("evidence.json")
FIRST_PASSAGE_SUMMARY_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-current-action-first-passage"
    / "evidence.json"
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
    "_uphill_relax_first_passage_protocol_runner",
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


def extract_frame(path: Path, frame_index: int, template):
    from pamssw.io import read_state

    if frame_index < 0:
        raise ValueError("frame index must be nonnegative")
    return read_state(
        path,
        index=frame_index,
        fixed_mask=template.fixed_mask,
        metadata={
            "first_passage_source": str(path),
            "accepted_step": int(frame_index),
        },
    )


def _zero_counts() -> dict[str, int]:
    return {
        "bootstrap_true_quench": 0,
        "starter_true_quench": 0,
        "local_softening_pre_relax": 0,
        "direction_oracle": 0,
        "escape_true_pes_check": 0,
        "biased_proposal_relax": 0,
        "landing_true_quench": 0,
        "post_relax_validation": 0,
        "unattributed": 0,
    }


def reused_endpoint_row(
    source: Mapping[str, Any],
    *,
    frame_index: int,
    final_frame_index: int,
) -> dict[str, Any]:
    if frame_index == 0:
        prefix = "explicit"
        relation = str(source["landing_relation"])
    elif frame_index == final_frame_index:
        prefix = "relaxed"
        relation = "SAME_LANDING"
    else:
        raise ValueError("only frame zero and final frame can be reused")
    return {
        "frame_index": int(frame_index),
        "label": str(source[f"{prefix}_label"]),
        "landing_energy_eV": source.get(f"{prefix}_landing_energy_eV"),
        "landing_path": source.get(f"{prefix}_landing_path"),
        "landing_sha256": source.get(f"{prefix}_landing_sha256"),
        "landing_relation_to_final": relation,
        "new_force_evaluations": 0,
        "purpose_counts": _zero_counts(),
        "wall_time_s": 0.0,
        "evidence_origin": "reused_g_up0",
    }


def _resolve_raw_evidence(
    summary_path: Path,
    *,
    raw_path_key: str = "raw_evidence_path",
    raw_hash_key: str = "raw_evidence_sha256",
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_path = Path(summary[raw_path_key])
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path
    if not raw_path.is_file() or _sha256(raw_path) != summary[raw_hash_key]:
        raise RuntimeError(f"raw evidence SHA256 drifted: {raw_path}")
    return summary, raw_path, json.loads(raw_path.read_text(encoding="utf-8"))


def _source_context():
    gup0_summary, gup0_raw_path, gup0 = _resolve_raw_evidence(
        GUP0_SUMMARY_PATH
    )
    fp_summary, fp_raw_path, first_passage_raw = _resolve_raw_evidence(
        FIRST_PASSAGE_SUMMARY_PATH
    )
    pairs = list(gup0["pairs"])
    discovery = protocol.select_pairs(pairs, protocol.DISCOVERY_KEYS)
    holdout = protocol.select_pairs(pairs, protocol.HOLDOUT_KEYS)
    fp_cases = {
        (str(case["state_id"]), int(case["seed"]), str(case["arm"])): case
        for case in first_passage_raw["cases"]
        if case["system"] == "c60"
    }
    for pair in discovery + holdout:
        key = (pair["state_id"], int(pair["seed"]), pair["arm"])
        case = fp_cases[key]
        config = case["effective_config"]
        if config["proposal_optimizer"] != "safe-lbfgs-total":
            raise RuntimeError("source did not use safe-lbfgs-total")
        if int(config["relaxation_trajectory_stride"]) != 1:
            raise RuntimeError("source trajectory stride is not one")
        checkpoint = next(
            row
            for row in case["checkpoints"]
            if int(row["horizon"]) == int(pair["horizon"])
        )
        if checkpoint["walk_trust_radius_clipped"]:
            raise RuntimeError("clipped checkpoints are excluded from G-UP1")
        path = Path(pair["source_optimizer_trajectory_path"])
        if not path.is_file() or _sha256(path) != pair[
            "source_optimizer_trajectory_sha256"
        ]:
            raise RuntimeError(f"optimizer trajectory SHA256 drifted: {path}")
    return {
        "gup0_summary": gup0_summary,
        "gup0_raw_path": gup0_raw_path,
        "first_passage_summary": fp_summary,
        "first_passage_raw_path": fp_raw_path,
        "first_passage_cases": fp_cases,
        "discovery_pairs": discovery,
        "holdout_pairs": holdout,
    }


def _frame_count(path: Path) -> int:
    from ase.io import read

    return len(read(path, index=":"))


def _runtime():
    gup0 = _load_module(
        GUP0_RUNNER_PATH,
        "_uphill_relax_first_passage_gup0_runner",
    )
    first_passage = _load_module(
        gup0.FIRST_PASSAGE_PATH,
        "_uphill_relax_first_passage_source_runner",
    )
    audit = _load_module(
        first_passage.FIXED_AUDIT_PATH,
        "_uphill_relax_first_passage_fixed_audit",
    )
    config_gate = _load_module(
        first_passage.CONFIG_GATE_PATH,
        "_uphill_relax_first_passage_config_gate",
    )
    action_runner = _load_module(
        first_passage.ACTION_RUNNER_PATH,
        "_uphill_relax_first_passage_action_runner",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    from pamssw.calculators import ASECalculator

    calculator = ASECalculator(base_runner._calculator())
    states = {}
    for state_id in ("intermediate_accepted", "plateau_accepted"):
        states[state_id] = first_passage._load_starter(
            state_source_root=REPO_ROOT,
            system="c60",
            state_id=state_id,
            audit=audit,
            base_runner=base_runner,
        )
    return {
        "gup0": gup0,
        "first_passage": first_passage,
        "config_gate": config_gate,
        "action_runner": action_runner,
        "base_runner": base_runner,
        "calculator": calculator,
        "states": states,
    }


def _final_mapping(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": pair["relaxed_label"],
        "landing_energy_eV": pair["relaxed_landing_energy_eV"],
        "landing_path": pair["relaxed_landing_path"],
        "landing_sha256": pair["relaxed_landing_sha256"],
    }


def _quench_intermediate_frame(
    *,
    pair: Mapping[str, Any],
    frame_index: int,
    final_frame_index: int,
    case_dir: Path,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    remaining_budget: int,
) -> dict[str, Any]:
    from pamssw.io import write_state

    state_id = str(pair["state_id"])
    starter_state, provenance = runtime["states"][state_id]
    trajectory_path = Path(pair["source_optimizer_trajectory_path"])
    state = extract_frame(trajectory_path, frame_index, starter_state)
    case_dir.mkdir(parents=True, exist_ok=False)
    state_path = case_dir / f"frame-{frame_index:03d}.xyz"
    write_state(state_path, state)
    config = runtime["action_runner"]._base_config(
        runtime["config_gate"],
        "c60",
        int(pair["seed"]),
        case_dir,
    )
    source_case = context["first_passage_cases"][
        (state_id, int(pair["seed"]), str(pair["arm"]))
    ]
    quenched = runtime["first_passage"]._quench_checkpoint(
        source={
            "horizon": int(pair["horizon"]),
            "checkpoint_path": str(state_path),
            "checkpoint_sha256": _sha256(state_path),
        },
        starter_state=starter_state,
        starter_energy=float(source_case["starter_energy_eV"]),
        calculator=runtime["calculator"],
        config=config,
        system="c60",
        case_dir=case_dir / f"frame-{frame_index:03d}-quench",
        remaining_budget=remaining_budget,
        base_runner=runtime["base_runner"],
    )
    relation, metrics = runtime["gup0"]._landing_relation(
        explicit=quenched,
        relaxed=_final_mapping(pair),
        starter_state=starter_state,
        config=config,
    )
    return {
        "frame_index": int(frame_index),
        "label": str(quenched["label"]),
        "landing_energy_eV": quenched.get("landing_energy_eV"),
        "landing_path": quenched.get("landing_path"),
        "landing_sha256": quenched.get("landing_sha256"),
        "landing_relation_to_final": relation,
        "new_force_evaluations": int(quenched["force_evaluations"]),
        "purpose_counts": dict(quenched["purpose_counts"]),
        "wall_time_s": float(quenched["wall_time_s"]),
        "evidence_origin": "new_true_quench",
        "starter_state_sha256": provenance["state_sha256"],
        **metrics,
    }


def _aggregate_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(
            {name: int(value) for name, value in row["purpose_counts"].items()}
        )
    if counts["direction_oracle"] or counts["biased_proposal_relax"] or counts["unattributed"]:
        raise RuntimeError("G-UP1 replay ledger contains forbidden evaluations")
    if sum(counts.values()) != sum(int(row["new_force_evaluations"]) for row in rows):
        raise RuntimeError("G-UP1 replay ledger does not close")
    return dict(sorted(counts.items()))


def _trajectory_record(
    *,
    pair: Mapping[str, Any],
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    output_dir: Path,
    total_new_fe: int,
    started: float,
    max_new_fe: int,
    max_wall_s: float,
) -> tuple[dict[str, Any], int]:
    path = Path(pair["source_optimizer_trajectory_path"])
    frame_count = _frame_count(path)
    final_index = frame_count - 1
    case_dir = (
        output_dir
        / "discovery"
        / str(pair["state_id"])
        / f"seed-{int(pair['seed']):08d}"
        / str(pair["arm"])
        / f"h{int(pair['horizon']):02d}"
    )
    case_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for frame_index in range(frame_count):
        if perf_counter() - started > max_wall_s:
            raise RuntimeError("G-UP1 kernel wall budget exhausted")
        if frame_index in {0, final_index}:
            row = reused_endpoint_row(
                pair,
                frame_index=frame_index,
                final_frame_index=final_index,
            )
        else:
            if total_new_fe >= max_new_fe:
                raise RuntimeError("G-UP1 force-evaluation budget exhausted")
            row = _quench_intermediate_frame(
                pair=pair,
                frame_index=frame_index,
                final_frame_index=final_index,
                case_dir=case_dir,
                context=context,
                runtime=runtime,
                remaining_budget=max_new_fe - total_new_fe,
            )
            total_new_fe += int(row["new_force_evaluations"])
        rows.append(row)
        _write_json(case_dir / "partial.json", {"frames": rows})
        print(
            f"[G-UP1] discovery {protocol.trajectory_key(pair)} "
            f"frame={frame_index}/{final_index} label={row['label']} "
            f"relation={row['landing_relation_to_final']} "
            f"new_fe={row['new_force_evaluations']}",
            flush=True,
        )
    summary = protocol.summarize_trajectory(rows)
    return {
        "key": list(protocol.trajectory_key(pair)),
        **{
            name: pair[name]
            for name in ("state_id", "seed", "arm", "horizon")
        },
        "source_optimizer_trajectory_path": str(path),
        "source_optimizer_trajectory_sha256": pair[
            "source_optimizer_trajectory_sha256"
        ],
        "frames": rows,
        "summary": summary,
    }, total_new_fe


def run_discovery(
    *,
    output_dir: Path,
    expected_commit: str | None,
    max_new_fe: int,
    max_wall_s: float,
) -> dict[str, Any]:
    commit = _current_commit()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("discovery requires a clean tracked worktree")
    discovery_path = output_dir / "discovery.json"
    if discovery_path.exists():
        raise FileExistsError(discovery_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _source_context()
    runtime = _runtime()
    started = perf_counter()
    total_new_fe = 0
    trajectories = []
    for pair in context["discovery_pairs"]:
        record, total_new_fe = _trajectory_record(
            pair=pair,
            context=context,
            runtime=runtime,
            output_dir=output_dir,
            total_new_fe=total_new_fe,
            started=started,
            max_new_fe=max_new_fe,
            max_wall_s=max_wall_s,
        )
        trajectories.append(record)
        _write_json(
            output_dir / "discovery-partial.json",
            {"trajectories": trajectories, "new_force_evaluations": total_new_fe},
        )
    cutoff = protocol.derive_cutoff(
        [record["summary"] for record in trajectories]
    )
    all_rows = [row for record in trajectories for row in record["frames"]]
    evidence = {
        "schema_version": 1,
        "stage": "discovery",
        "execution_commit": commit,
        "trajectory_count": len(trajectories),
        "frame_count": len(all_rows),
        "fixed_cutoff": cutoff,
        "common_shorter_cutoff": all(
            cutoff < record["summary"]["final_frame_index"]
            for record in trajectories
        ),
        "new_force_evaluations": total_new_fe,
        "max_new_force_evaluations": max_new_fe,
        "purpose_counts": _aggregate_counts(all_rows),
        "wall_time_s": float(perf_counter() - started),
        "trajectories": trajectories,
    }
    _write_json(discovery_path, evidence)
    return evidence


def _holdout_row(
    *,
    pair: Mapping[str, Any],
    cutoff: int,
    context: Mapping[str, Any],
    runtime: Mapping[str, Any],
    output_dir: Path,
    remaining_budget: int,
) -> dict[str, Any]:
    path = Path(pair["source_optimizer_trajectory_path"])
    final_index = _frame_count(path) - 1
    has_headroom = cutoff < final_index
    executed_index = cutoff if has_headroom else final_index
    if executed_index in {0, final_index}:
        frame = reused_endpoint_row(
            pair,
            frame_index=executed_index,
            final_frame_index=final_index,
        )
    else:
        case_dir = (
            output_dir
            / "holdout"
            / str(pair["state_id"])
            / f"seed-{int(pair['seed']):08d}"
            / str(pair["arm"])
            / f"h{int(pair['horizon']):02d}"
        )
        case_dir.mkdir(parents=True, exist_ok=False)
        frame = _quench_intermediate_frame(
            pair=pair,
            frame_index=executed_index,
            final_frame_index=final_index,
            case_dir=case_dir,
            context=context,
            runtime=runtime,
            remaining_budget=remaining_budget,
        )
    return {
        **{
            name: pair[name]
            for name in ("state_id", "seed", "arm", "horizon")
        },
        "status": "completed",
        "fixed_cutoff": cutoff,
        "final_frame_index": final_index,
        "executed_frame_index": executed_index,
        "has_headroom": has_headroom,
        "cutoff_label": frame["label"],
        "cutoff_relation_to_final": frame["landing_relation_to_final"],
        "new_force_evaluations": frame["new_force_evaluations"],
        "purpose_counts": frame["purpose_counts"],
        "frame": frame,
    }


def run_holdout(
    *,
    output_dir: Path,
    expected_commit: str | None,
    max_new_fe: int,
    max_wall_s: float,
) -> dict[str, Any]:
    commit = _current_commit()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("holdout requires a clean tracked worktree")
    discovery_path = output_dir / "discovery.json"
    evidence_path = output_dir / "evidence.json"
    if not discovery_path.is_file():
        raise FileNotFoundError(discovery_path)
    if evidence_path.exists():
        raise FileExistsError(evidence_path)
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    cutoff = int(discovery["fixed_cutoff"])
    if not discovery["common_shorter_cutoff"]:
        raise RuntimeError("discovery found no common shorter cutoff")
    context = _source_context()
    runtime = _runtime()
    started = perf_counter()
    used = int(discovery["new_force_evaluations"])
    rows = []
    for pair in context["holdout_pairs"]:
        if used >= max_new_fe or discovery["wall_time_s"] + perf_counter() - started > max_wall_s:
            raise RuntimeError("G-UP1 total budget exhausted before holdout closure")
        row = _holdout_row(
            pair=pair,
            cutoff=cutoff,
            context=context,
            runtime=runtime,
            output_dir=output_dir,
            remaining_budget=max_new_fe - used,
        )
        used += int(row["new_force_evaluations"])
        rows.append(row)
        print(
            f"[G-UP1] holdout {protocol.trajectory_key(pair)} "
            f"frame={row['executed_frame_index']}/{row['final_frame_index']} "
            f"relation={row['cutoff_relation_to_final']}",
            flush=True,
        )
    decision = protocol.decide_holdout(rows)
    counts = _aggregate_counts(
        [row["frame"] for row in rows]
    )
    total_counts = Counter(discovery["purpose_counts"])
    total_counts.update(counts)
    evidence = {
        "schema_version": 1,
        "stage": "complete",
        "execution_commits": {
            "discovery": discovery["execution_commit"],
            "holdout": commit,
        },
        "fixed_cutoff": cutoff,
        "discovery": discovery,
        "holdout": rows,
        **decision,
        "new_force_evaluations": used,
        "max_new_force_evaluations": max_new_fe,
        "purpose_counts": dict(sorted(total_counts.items())),
        "wall_time_s": float(discovery["wall_time_s"] + perf_counter() - started),
        "production_default_changed": False,
        "claim_ceiling": (
            "C60 recorded Safe-LBFGS accepted-frame discovery plus four-pair "
            "mechanism holdout; not a full-action or cross-system promotion"
        ),
    }
    _write_json(evidence_path, evidence)
    return evidence


def run_smoke(
    *,
    output_dir: Path,
    key: tuple[str, int, str, int],
    frame_index: int,
    max_new_fe: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    context = _source_context()
    pair = protocol.select_pairs(
        context["discovery_pairs"] + context["holdout_pairs"],
        [key],
    )[0]
    final_index = _frame_count(Path(pair["source_optimizer_trajectory_path"])) - 1
    if frame_index <= 0 or frame_index >= final_index:
        raise ValueError("smoke frame must be an intermediate accepted step")
    runtime = _runtime()
    frame = _quench_intermediate_frame(
        pair=pair,
        frame_index=frame_index,
        final_frame_index=final_index,
        case_dir=output_dir / "case",
        context=context,
        runtime=runtime,
        remaining_budget=max_new_fe,
    )
    evidence = {
        "schema_version": 1,
        "smoke_only": True,
        "key": list(key),
        "frame": frame,
        "new_force_evaluations": frame["new_force_evaluations"],
        "purpose_counts": _aggregate_counts([frame]),
    }
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def check_discovery(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence["trajectory_count"] != 4 or evidence["frame_count"] != 238:
        raise RuntimeError("discovery cohort drifted")
    summaries = [row["summary"] for row in evidence["trajectories"]]
    if protocol.derive_cutoff(summaries) != evidence["fixed_cutoff"]:
        raise RuntimeError("discovery cutoff drifted")
    all_rows = [row for trajectory in evidence["trajectories"] for row in trajectory["frames"]]
    if _aggregate_counts(all_rows) != evidence["purpose_counts"]:
        raise RuntimeError("discovery ledger drifted")
    return {
        "status": "valid",
        "frame_count": evidence["frame_count"],
        "fixed_cutoff": evidence["fixed_cutoff"],
        "new_force_evaluations": evidence["new_force_evaluations"],
    }


def check_evidence(path: Path) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    check_discovery(path.parent / "discovery.json")
    decision = protocol.decide_holdout(evidence["holdout"])
    if decision["decision"] != evidence["decision"]:
        raise RuntimeError("holdout decision drifted")
    if sum(evidence["purpose_counts"].values()) != evidence["new_force_evaluations"]:
        raise RuntimeError("complete evidence ledger drifted")
    if evidence["purpose_counts"]["unattributed"] != 0:
        raise RuntimeError("complete evidence has unattributed evaluations")
    return {
        "status": "valid",
        "fixed_cutoff": evidence["fixed_cutoff"],
        "decision": evidence["decision"],
        "new_force_evaluations": evidence["new_force_evaluations"],
    }


def _parse_key(value: str) -> tuple[str, int, str, int]:
    state_id, seed, arm, horizon = value.split(",")
    return state_id, int(seed), arm, int(horizon)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("discovery", "holdout"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--max-new-force-evaluations", type=int, default=protocol.MAX_NEW_FORCE_EVALUATIONS)
    parser.add_argument("--max-kernel-wall-time-s", type=float, default=protocol.MAX_KERNEL_WALL_TIME_S)
    parser.add_argument("--smoke-key", type=_parse_key)
    parser.add_argument("--smoke-frame", type=int)
    parser.add_argument("--check-discovery", type=Path)
    parser.add_argument("--check-evidence", type=Path)
    args = parser.parse_args()
    if args.check_discovery is not None:
        print(json.dumps(check_discovery(args.check_discovery.resolve()), indent=2, sort_keys=True))
        return 0
    if args.check_evidence is not None:
        print(json.dumps(check_evidence(args.check_evidence.resolve()), indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.smoke_key is not None:
        if args.smoke_frame is None:
            parser.error("--smoke-frame is required with --smoke-key")
        evidence = run_smoke(
            output_dir=args.output_dir.resolve(),
            key=args.smoke_key,
            frame_index=args.smoke_frame,
            max_new_fe=args.max_new_force_evaluations,
        )
    elif args.stage == "discovery":
        evidence = run_discovery(
            output_dir=args.output_dir.resolve(),
            expected_commit=args.expected_commit,
            max_new_fe=args.max_new_force_evaluations,
            max_wall_s=args.max_kernel_wall_time_s,
        )
    elif args.stage == "holdout":
        evidence = run_holdout(
            output_dir=args.output_dir.resolve(),
            expected_commit=args.expected_commit,
            max_new_fe=args.max_new_force_evaluations,
            max_wall_s=args.max_kernel_wall_time_s,
        )
    else:
        parser.error("choose --stage or --smoke-key")
    print(json.dumps({key: evidence.get(key) for key in ("stage", "smoke_only", "fixed_cutoff", "decision", "new_force_evaluations", "wall_time_s")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
