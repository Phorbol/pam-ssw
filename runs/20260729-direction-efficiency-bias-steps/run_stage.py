#!/usr/bin/env python3
"""Run preregistered fixed-starter direction-efficiency stages."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PROTOCOL_PATH = RUN_ROOT / "protocol.py"
LOCKED_RUNTIME_PATH = (
    REPO_ROOT
    / "runs"
    / "20260729-anchor-consistent-direction-ablation"
    / "run_ablation.py"
)

FROZEN_FIELDS = {
    "direction_selection_mode": "discrete",
    "direction_synthesis_mode": "none",
    "direction_type_ucb_enabled": False,
    "archive_escape_momentum_enabled": False,
    "choice_aligned_softening_enabled": False,
    "direction_probe_enabled": False,
    "plateau_evolution_enabled": False,
    "direction_curvature_source": "inner",
    "proposal_optimizer": "safe-lbfgs-total",
    "proposal_fmax": 0.05,
    "quench_fmax": 0.01,
}

BASELINE_FIELDS = {
    "enable_momentum_candidate": True,
    "oracle_candidates": 12,
    "max_steps_per_walk": 8,
    "proposal_relax_steps": 80,
}

COMMON_OVERRIDES = {
    "max_trials": 1,
    "max_force_evals": None,
    "quench_optimizer": "ase-lbfgs",
    "quench_fallback_optimizer": "ase-fire",
    "quench_fmax": 0.01,
    "quench_maxiter": 400,
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_protocol():
    return _load_module(
        PROTOCOL_PATH,
        "_staged_direction_efficiency_protocol_runtime",
    )


def _load_locked_runtime():
    runner = _load_module(
        LOCKED_RUNTIME_PATH,
        "_staged_direction_efficiency_locked_runtime",
    )
    return runner._load_locked_runtime()


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


def _effective_config(*, case, case_dir: Path, base_runner):
    source_config = base_runner.build_config("c60", case_dir)
    source = asdict(source_config)
    for field, expected in FROZEN_FIELDS.items():
        if source[field] != expected:
            raise RuntimeError(f"frozen protocol drifted: {field}")
    for field, expected in BASELINE_FIELDS.items():
        if source[field] != expected:
            raise RuntimeError(f"frozen baseline drifted: {field}")
    effective_config = replace(
        source_config,
        **COMMON_OVERRIDES,
        rng_seed=case.seed,
        enable_momentum_candidate=(
            case.settings.enable_momentum_candidate
        ),
        oracle_candidates=case.settings.oracle_candidates,
        max_steps_per_walk=case.settings.max_steps_per_walk,
        proposal_relax_steps=case.settings.proposal_relax_steps,
        direction_diagnostics_enabled=True,
        direction_diagnostics_path=str(
            Path(case_dir) / "direction_trace.jsonl"
        ),
    )
    return source_config, effective_config


def config_projection(
    *,
    case,
    case_dir: Path,
    base_runner,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[Any]]]:
    source_config, effective_config = _effective_config(
        case=case,
        case_dir=case_dir,
        base_runner=base_runner,
    )
    source = asdict(source_config)
    effective = asdict(effective_config)
    diff = {
        key: [source[key], effective[key]]
        for key in source
        if source[key] != effective[key]
    }
    return source, effective, diff


def validate_direction_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    oracle_candidates: int,
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("direction trace is empty")
    candidate_kinds: Counter[str] = Counter()
    selected_kinds: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if int(row["candidate_count"]) != oracle_candidates:
            raise RuntimeError(
                f"direction row {index} candidate count drifted"
            )
        counts = row.get("evaluated_candidate_kind_counts")
        if not isinstance(counts, dict):
            raise RuntimeError(
                f"direction row {index} lacks source counts"
            )
        exact_counts = {
            str(key): int(value) for key, value in counts.items()
        }
        if sum(exact_counts.values()) != oracle_candidates:
            raise RuntimeError(
                f"direction row {index} source counts do not close"
            )
        selected = str(row["selected_kind"])
        if exact_counts.get(selected, 0) <= 0:
            raise RuntimeError(
                f"direction row {index} selected source was not evaluated"
            )
        expected_fe = 2 * oracle_candidates
        for field in (
            "oracle_selection_force_evaluations_delta",
            "oracle_direction_force_evaluations_delta",
        ):
            if int(row[field]) != expected_fe:
                raise RuntimeError(
                    f"direction row {index} {field} does not close"
                )
        candidate_kinds.update(exact_counts)
        selected_kinds[selected] += 1
    return {
        "selection_count": len(rows),
        "candidate_count": len(rows) * oracle_candidates,
        "candidate_kind_counts": dict(sorted(candidate_kinds.items())),
        "selected_kind_counts": dict(sorted(selected_kinds.items())),
        "direction_oracle_force_evaluations": (
            2 * oracle_candidates * len(rows)
        ),
    }


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
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
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state) -> str:
    digest = sha256()
    for values in (
        np.asarray(state.numbers, dtype="<i8"),
        np.asarray(state.positions, dtype="<f8"),
        np.asarray(state.fixed_mask, dtype=np.uint8),
        np.asarray(
            (
                state.cell
                if state.cell is not None
                else np.zeros((3, 3))
            ),
            dtype="<f8",
        ),
        np.asarray(state.pbc, dtype=np.uint8),
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


def _read_direction_rows(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"direction diagnostics file was not written: {path}"
        )
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"invalid direction JSON at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise RuntimeError(
                f"direction row {line_number} is not a mapping"
            )
        rows.append(row)
    return rows


def _run_case(
    *,
    case,
    state,
    state_provenance: Mapping[str, Any],
    shared_calculator,
    base_runner,
    case_dir: Path,
    execution_commit: str | None = None,
    walker_factory=None,
    archive_factory=None,
    certificate_checker=None,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose

    if walker_factory is None:
        from pamssw.walker import SurfaceWalker

        walker_factory = SurfaceWalker
    if archive_factory is None:
        from pamssw.archive import MinimaArchive

        archive_factory = MinimaArchive
    if certificate_checker is None:
        from pamssw.relax import has_force_convergence_certificate

        certificate_checker = has_force_convergence_certificate

    case_dir = Path(case_dir)
    source_config, config = _effective_config(
        case=case,
        case_dir=case_dir,
        base_runner=base_runner,
    )
    source = asdict(source_config)
    effective = asdict(config)
    diff = {
        key: [source[key], effective[key]]
        for key in source
        if source[key] != effective[key]
    }
    walker = walker_factory(
        calculator=shared_calculator,
        config=config,
        softening_enabled=True,
    )
    walker._reset_direction_diagnostics()
    archive = archive_factory(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )

    generation_started = perf_counter()
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        starter_evaluation = walker.calculator.evaluate(state)
    starter_energy = float(starter_evaluation.energy)
    seed_entry = archive.add(
        state,
        starter_energy,
        parent_id=None,
    )
    proposal = walker._proposal_pool(
        state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=seed_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    escape_state = proposal.state
    with walker.calculator.purpose(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    generation_wall_time = float(perf_counter() - generation_started)

    quench_started = perf_counter()
    landing = walker.relax_true_minimum(
        escape_state,
        trajectory_name=f"{case.key}-landing",
    )
    quench_wall_time = float(perf_counter() - quench_started)
    before_count = len(archive.entries)
    landing_entry = archive.add(
        landing.state,
        float(landing.energy),
        parent_id=seed_entry.entry_id,
    )
    certificate = bool(
        certificate_checker(landing, config.quench_fmax)
    )
    direction_rows = _read_direction_rows(
        Path(config.direction_diagnostics_path)
    )
    direction_audit = validate_direction_trace(
        direction_rows,
        oracle_candidates=config.oracle_candidates,
    )
    purpose_counts = walker.calculator.snapshot().as_dict()
    force_evaluations = sum(
        int(value) for value in purpose_counts.values()
    )
    if purpose_counts["unattributed"] != 0:
        raise RuntimeError(
            "case contains unattributed force evaluations"
        )
    if (
        purpose_counts["direction_oracle"]
        != direction_audit["direction_oracle_force_evaluations"]
    ):
        raise RuntimeError("direction purpose ledger does not close")
    if purpose_counts["landing_true_quench"] <= 0:
        raise RuntimeError("strict terminal quench did no physical work")
    if not certificate:
        raise RuntimeError("terminal quench lacks a strict certificate")

    state_hash = _state_sha256(state)
    if state_hash != state_provenance["state_sha256"]:
        raise RuntimeError("starter state hash differs from provenance")
    case_dir.mkdir(parents=True, exist_ok=True)
    starter_path = case_dir / "starter.xyz"
    escape_path = case_dir / "escape.xyz"
    landing_path = case_dir / "landing.xyz"
    base_runner.write_state(starter_path, state)
    base_runner.write_state(escape_path, escape_state)
    base_runner.write_state(landing_path, landing.state)
    optimizer_diagnostics = walker.relaxation_diagnostics()
    proposal_relax_count = int(
        optimizer_diagnostics["proposal_relax_count"]
    )
    proposal_trace = {
        "direction_steps": direction_rows,
        "attempted_bias_steps": proposal_relax_count,
        "configured_bias_step_cap": config.max_steps_per_walk,
        "termination_reason": (
            "reached_bias_step_cap"
            if proposal_relax_count == config.max_steps_per_walk
            else "early_exit"
        ),
        "optimizer_termination_counts": {
            key.removeprefix("proposal_relax_termination_"): int(
                value
            )
            for key, value in optimizer_diagnostics.items()
            if key.startswith("proposal_relax_termination_")
        },
    }
    is_new_basin = len(archive.entries) > before_count
    landing_delta = float(landing.energy) - starter_energy
    meaningful = bool(
        certificate
        and is_new_basin
        and landing_delta <= -_load_protocol().MEANINGFUL_ENERGY_DROP_EV
    )
    row = {
        "status": "completed",
        "stage": case.stage,
        "state_id": case.state_id,
        "state_sha256": state_hash,
        "exact_starter_reference": True,
        "seed": case.seed,
        "arm": case.arm,
        "repeat": case.repeat,
        "settings": asdict(case.settings),
        "selection_probability": 1.0,
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": landing_delta,
        "certificate": certificate,
        "final_max_force_eV_per_A": float(landing.gradient_norm),
        "is_new_basin": is_new_basin,
        "meaningful": meaningful,
        "landing_entry_id": int(landing_entry.entry_id),
        "fragmented": bool(
            walker._is_fragmented_cluster(state, landing.state)
        ),
        "fallback_used": bool(
            optimizer_diagnostics["quench_fallback_attempts"]
        ),
        "quench_iterations": int(landing.n_iter),
        "termination_reason": landing.telemetry.termination_reason,
        "force_evaluations": force_evaluations,
        "purpose_counts": purpose_counts,
        "direction_audit": direction_audit,
        "direction_trace_valid": True,
        "optimizer_diagnostics": optimizer_diagnostics,
        "proposal_trace": proposal_trace,
        "generation_wall_time_s": generation_wall_time,
        "quench_wall_time_s": quench_wall_time,
        "starter_path": str(starter_path),
        "starter_file_sha256": _sha256(starter_path),
        "escape_path": str(escape_path),
        "escape_sha256": _sha256(escape_path),
        "landing_path": str(landing_path),
        "landing_sha256": _sha256(landing_path),
        "effective_config": effective,
        "source_to_effective_config_diff": diff,
    }
    if execution_commit is not None:
        row["execution_commit"] = execution_commit
    row = json.loads(
        json.dumps(row, sort_keys=True, allow_nan=False)
    )
    _write_json(case_dir / "summary.json", row)
    return row


def _stage_context(
    protocol,
    *,
    stage: str,
    prior_evidence_path: Path | None,
    record_not_entered: bool,
) -> tuple[Any, dict[str, Any] | None, str | None, bool]:
    if stage == "momentum":
        if prior_evidence_path is not None:
            raise ValueError("momentum forbids prior evidence")
        if record_not_entered:
            raise ValueError("record-not-entered is only valid for relax_cap")
        return protocol.RetainedSettings(), None, None, False

    expected_prior = {
        "candidate_count": "momentum",
        "bias_steps": "candidate_count",
        "relax_cap": "bias_steps",
    }.get(stage)
    if expected_prior is None:
        raise ValueError(f"unknown stage: {stage}")
    if prior_evidence_path is None:
        raise ValueError(f"{stage} requires prior evidence")
    prior_evidence_path = Path(prior_evidence_path)
    prior = json.loads(prior_evidence_path.read_text(encoding="utf-8"))
    try:
        prior_input = protocol.RetainedSettings(
            **prior["input_retained_settings"]
        )
        revalidated = protocol.build_evidence(
            expected_prior,
            prior_input,
            prior["cases"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prior evidence does not revalidate") from error
    if (
        prior.get("stage") != expected_prior
        or prior.get("cohort") != revalidated["cohort"]
        or prior.get("decision") != revalidated["decision"]
        or (
            expected_prior == "bias_steps"
            and prior.get("stage_l_entry")
            != revalidated["stage_l_entry"]
        )
    ):
        raise ValueError("prior evidence is not the required complete stage")
    retained = protocol.RetainedSettings(
        **prior["decision"]["retained_settings"]
    )
    not_entered = False
    if stage == "relax_cap":
        entered = prior.get("stage_l_entry", {}).get("entered")
        if entered is not True:
            if not record_not_entered:
                raise RuntimeError("relax_cap Stage-L entry gate is false")
            not_entered = True
        elif record_not_entered:
            raise ValueError(
                "record-not-entered requires a false Stage-L entry gate"
            )
    elif record_not_entered:
        raise ValueError("record-not-entered is only valid for relax_cap")
    return retained, prior, _sha256(prior_evidence_path), not_entered


def _pin_sources(output_dir: Path) -> dict[str, str]:
    sources = {
        "run_stage.py": Path(__file__).resolve(),
        "protocol.py": PROTOCOL_PATH.resolve(),
    }
    hashes = {}
    for name, source in sources.items():
        target = output_dir / name
        source_hash = _sha256(source)
        if target.exists():
            if _sha256(target) != source_hash:
                raise RuntimeError(f"pinned source drifted: {name}")
        else:
            shutil.copy2(source, target)
        hashes[name] = source_hash
    return hashes


def _validate_saved_case(
    protocol,
    *,
    row: Mapping[str, Any],
    case,
    execution_commit: str,
    config_path: Path,
) -> None:
    if (
        row.get("status") != "completed"
        or row.get("stage") != case.stage
        or row.get("state_id") != case.state_id
        or int(row.get("seed", -1)) != case.seed
        or row.get("arm") != case.arm
        or int(row.get("repeat", -1)) != case.repeat
        or row.get("settings") != asdict(case.settings)
        or row.get("execution_commit") != execution_commit
        or row.get("certificate") is not True
        or row.get("exact_starter_reference") is not True
        or row.get("direction_trace_valid") is not True
        or float(row.get("selection_probability", -1.0)) != 1.0
        or bool(row.get("meaningful")) != protocol.is_meaningful(row)
    ):
        raise RuntimeError(f"saved case does not revalidate: {case.key}")
    purposes = row["purpose_counts"]
    audit = row["direction_audit"]
    if (
        sum(int(value) for value in purposes.values())
        != int(row["force_evaluations"])
        or int(purposes.get("unattributed", -1)) != 0
        or int(purposes["direction_oracle"])
        != int(audit["direction_oracle_force_evaluations"])
        or int(audit["candidate_count"])
        != int(audit["selection_count"])
        * case.settings.oracle_candidates
        or int(audit["direction_oracle_force_evaluations"])
        != 2 * int(audit["candidate_count"])
    ):
        raise RuntimeError(f"saved case ledger does not revalidate: {case.key}")
    for path_field, hash_field in (
        ("starter_path", "starter_file_sha256"),
        ("escape_path", "escape_sha256"),
        ("landing_path", "landing_sha256"),
    ):
        path = Path(row[path_field])
        if not path.is_file() or _sha256(path) != row[hash_field]:
            raise RuntimeError(
                f"saved case artifact does not revalidate: {case.key}"
            )
    if (
        not config_path.is_file()
        or json.loads(config_path.read_text(encoding="utf-8"))
        != row["effective_config"]
    ):
        raise RuntimeError(
            f"saved case config does not revalidate: {case.key}"
        )


def _conclusion(evidence: Mapping[str, Any]) -> str:
    decision = evidence.get("decision", {})
    return (
        f"# Direction-efficiency stage: {evidence['stage']}\n\n"
        f"Status: `{decision.get('status', 'not_entered')}`.\n\n"
        f"Completed cases: {evidence['cohort']['completed_cases']}.\n\n"
        "This is a fixed-starter, one-proposal paired ablation. "
        "It does not change production defaults and does not validate a "
        "posterior selector.\n"
    )


def run_stage(
    *,
    stage: str,
    output_dir: Path,
    expected_git_commit: str,
    prior_evidence_path: Path | None = None,
    record_not_entered: bool = False,
    runtime_loader=None,
    case_executor=None,
) -> dict[str, Any]:
    actual_commit = _current_commit()
    if actual_commit != expected_git_commit:
        raise RuntimeError(
            f"execution commit mismatch: expected {expected_git_commit}, "
            f"got {actual_commit}"
        )
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree is not clean")
    protocol = _load_protocol()
    retained, prior, prior_hash, not_entered = _stage_context(
        protocol,
        stage=stage,
        prior_evidence_path=prior_evidence_path,
        record_not_entered=record_not_entered,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = _pin_sources(output_dir)
    common = {
        "schema_version": 1,
        "stage": stage,
        "execution_commit": actual_commit,
        "input_retained_settings": asdict(retained),
        "prior_evidence_sha256": prior_hash,
        "runner_source_sha256": source_hashes,
    }
    if not_entered:
        evidence = {
            **common,
            "cohort": {
                "state_ids": list(protocol.STATE_IDS),
                "seeds": list(protocol.SEEDS),
                "repeats": list(protocol.REPEATS),
                "arms": list(protocol.STAGE_ARMS[stage]),
                "completed_cases": 0,
            },
            "decision": {"stage": stage, "status": "not_entered"},
            "stage_l_entry": prior["stage_l_entry"],
            "production_default_changed": False,
            "cases": [],
        }
        _write_json(output_dir / "raw.json", evidence)
        _write_json(output_dir / "evidence.json", evidence)
        (output_dir / "conclusion.md").write_text(
            _conclusion(evidence),
            encoding="utf-8",
        )
        return evidence

    if runtime_loader is None:
        runtime_loader = _load_locked_runtime
    if case_executor is None:
        case_executor = _run_case
    (
        states,
        shared_calculator,
        base_runner,
        state_provenance,
        shared_provenance,
    ) = runtime_loader()
    cases = protocol.case_matrix(stage, retained)
    rows = []
    for case in cases:
        case_dir = output_dir / "cases" / case.key
        summary_path = case_dir / "summary.json"
        config_path = (
            output_dir / "effective_configs" / f"{case.key}.json"
        )
        if summary_path.exists():
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            _validate_saved_case(
                protocol,
                row=row,
                case=case,
                execution_commit=actual_commit,
                config_path=config_path,
            )
        else:
            print(
                f"[direction-efficiency] stage={stage} case={case.key}",
                flush=True,
            )
            row = case_executor(
                case=case,
                state=states[case.state_id],
                state_provenance=state_provenance[case.state_id],
                shared_calculator=shared_calculator,
                base_runner=base_runner,
                case_dir=case_dir,
                execution_commit=actual_commit,
            )
            _write_json(config_path, row["effective_config"])
        rows.append(row)
        _write_json(
            output_dir / "raw.json",
            {
                **common,
                "shared_provenance": shared_provenance,
                "state_provenance": state_provenance,
                "cases": rows,
            },
        )

    evidence = protocol.build_evidence(stage, retained, rows)
    evidence.update(common)
    evidence["shared_provenance"] = shared_provenance
    evidence["state_provenance"] = state_provenance
    _write_json(output_dir / "evidence.json", evidence)
    (output_dir / "conclusion.md").write_text(
        _conclusion(evidence),
        encoding="utf-8",
    )
    return evidence


def _parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("momentum", "candidate_count", "bias_steps", "relax_cap"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--prior-evidence", type=Path)
    parser.add_argument("--record-not-entered", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    evidence = run_stage(
        stage=args.stage,
        output_dir=args.output_dir,
        expected_git_commit=args.expected_git_commit,
        prior_evidence_path=args.prior_evidence,
        record_not_entered=args.record_not_entered,
    )
    print(
        json.dumps(
            {
                "stage": evidence["stage"],
                "cohort": evidence["cohort"],
                "decision": evidence["decision"],
                "totals": evidence.get("totals"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
