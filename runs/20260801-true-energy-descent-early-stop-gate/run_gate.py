#!/usr/bin/env python3
"""Run the G-E0 true-PES descent early-stop audit."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import math
from hashlib import sha256
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
MANIFEST_PATH = RUN_ROOT / "manifest.json"
SOURCE_SUMMARY_PATH = (
    REPO_ROOT / "runs" / "20260731-current-action-first-passage" / "evidence.json"
)
SOURCE_RUNNER_PATH = SOURCE_SUMMARY_PATH.with_name("run_gate.py")
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


protocol = _load_module(PROTOCOL_PATH, "_true_energy_descent_protocol_runner")


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


def _case_key(case: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(case["system"]),
        str(case["state_id"]),
        int(case["seed"]),
        str(case["arm"]),
    )


def validate_source_inputs(
    *,
    summary_path: Path = SOURCE_SUMMARY_PATH,
    manifest_path: Path = MANIFEST_PATH,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate the immutable cohort without importing the MACE runtime."""

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_path = Path(summary["raw_evidence_path"])
    if not raw_path.is_absolute():
        raw_path = repo_root / raw_path
    raw_sha256 = _sha256(raw_path)
    if raw_sha256 != summary["raw_evidence_sha256"]:
        raise RuntimeError("source raw evidence SHA256 drifted")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source_raw_evidence_sha256"] != raw_sha256:
        raise RuntimeError("manifest source evidence SHA256 drifted")

    cases_by_key = {_case_key(case): case for case in raw["cases"]}
    if len(cases_by_key) != len(raw["cases"]):
        raise RuntimeError("source contains duplicate case keys")
    manifest_keys: set[tuple[str, str, int, str]] = set()
    accepted = 0
    attempted = 0
    for manifest_case in manifest["cases"]:
        key_values = manifest_case["key"]
        key = (
            str(key_values[0]),
            str(key_values[1]),
            int(key_values[2]),
            str(key_values[3]),
        )
        if key in manifest_keys:
            raise RuntimeError("manifest contains duplicate case keys")
        manifest_keys.add(key)
        source_case = cases_by_key.get(key)
        if source_case is None:
            raise RuntimeError(f"manifest case missing from source: {key}")
        reached = int(manifest_case["reached_macro_steps"])
        attempted_case = int(manifest_case["attempted_macro_steps"])
        if reached != int(source_case["reached_macro_steps"]):
            raise RuntimeError(f"reached-step drift: {key}")
        if attempted_case != int(source_case["attempted_macro_steps"]):
            raise RuntimeError(f"attempted-step drift: {key}")
        if tuple(range(1, reached + 1)) != tuple(
            int(row["step"]) for row in manifest_case["endpoints"]
        ):
            raise RuntimeError(f"manifest endpoints are not consecutive: {key}")
        source_checkpoints = {
            int(row["horizon"]): row for row in source_case["checkpoints"]
        }
        for endpoint in manifest_case["endpoints"]:
            path = repo_root / endpoint["checkpoint_path"]
            if _sha256(path) != endpoint["checkpoint_sha256"]:
                raise RuntimeError(f"endpoint SHA256 drifted: {path}")
            source_checkpoint = source_checkpoints.get(int(endpoint["step"]))
            if source_checkpoint is not None:
                if source_checkpoint["checkpoint_sha256"] != endpoint[
                    "checkpoint_sha256"
                ]:
                    raise RuntimeError(f"source checkpoint SHA256 drifted: {key}")
                landing_path = source_checkpoint.get("landing_path")
                landing_hash = source_checkpoint.get("landing_sha256")
                if landing_path is not None and (
                    landing_hash is None
                    or _sha256(Path(landing_path)) != landing_hash
                ):
                    raise RuntimeError(f"source landing SHA256 drifted: {key}")
        accepted += reached
        attempted += attempted_case

    if set(cases_by_key) != manifest_keys:
        raise RuntimeError("source and manifest case cohorts differ")
    expected = (
        int(manifest["case_count"]),
        int(manifest["accepted_endpoint_count"]),
        int(manifest["attempted_endpoint_count"]),
    )
    actual = (len(manifest_keys), accepted, attempted)
    if actual != expected:
        raise RuntimeError(f"manifest cohort does not close: {actual} != {expected}")
    return {
        "summary": summary,
        "raw": raw,
        "raw_path": raw_path,
        "manifest": manifest,
        "cases_by_key": cases_by_key,
        "source_raw_sha256": raw_sha256,
        "accepted_endpoint_count": accepted,
        "attempted_endpoint_count": attempted,
    }


def _zero_counts() -> dict[str, int]:
    from pamssw.accounting import EvaluationPurpose

    return {purpose.value: 0 for purpose in EvaluationPurpose}


def reused_energy_row(source: Mapping[str, Any]) -> dict[str, Any]:
    energy = float(source["checkpoint_energy_eV"])
    delta = float(source["checkpoint_delta_eV"])
    if not math.isfinite(energy) or not math.isfinite(delta):
        raise ValueError("reused checkpoint energy must be finite")
    return {
        "step": int(source["horizon"]),
        "checkpoint_energy_eV": energy,
        "checkpoint_delta_eV": delta,
        "new_force_evaluations": 0,
        "purpose_counts": _zero_counts(),
        "evidence_origin": "reused_first_passage",
    }


def reused_quench_row(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": int(source["horizon"]),
        "label": str(source["label"]),
        "landing_energy_eV": source.get("landing_energy_eV"),
        "landing_delta_eV": source.get("landing_delta_eV"),
        "landing_path": source.get("landing_path"),
        "landing_sha256": source.get("landing_sha256"),
        "certificate": bool(source.get("certificate", False)),
        "geometry_valid": source.get("geometry_valid"),
        "fragmented": source.get("fragmented"),
        "matcher_same": source.get("matcher_same"),
        "descriptor_same": source.get("descriptor_same"),
        "descriptor_delta": source.get("descriptor_delta"),
        "new_force_evaluations": 0,
        "purpose_counts": _zero_counts(),
        "wall_time_s": 0.0,
        "evidence_origin": "reused_first_passage",
    }


def required_quench_steps(
    *,
    first_crossing: int | None,
    terminal: int,
    reusable: set[int],
) -> tuple[int, ...]:
    if first_crossing is None:
        return ()
    candidates = {int(first_crossing), int(terminal)}
    return tuple(sorted(step for step in candidates if step not in reusable))


def summarize_case_outcome(
    *,
    energy_rows,
    quench_rows: Mapping[int, Mapping[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    crossing = protocol.first_descent_crossing(
        energy_rows,
        tolerance=tolerance,
    )
    if crossing is None:
        return {
            "first_crossing_step": None,
            "saved_outer_micro_steps": 0,
            "crossing_is_certified_lower_basin": False,
            "tradeoff_class": "UNLEARNABLE",
        }
    crossing_step = int(crossing["step"])
    terminal_step = int(energy_rows[-1]["step"])
    crossing_quench = quench_rows[crossing_step]
    terminal_quench = quench_rows[terminal_step]
    crossing_delta = crossing_quench.get("landing_delta_eV")
    certified_lower = bool(
        crossing_quench.get("label") == "ESCAPED_CERTIFIED"
        and crossing_delta is not None
        and float(crossing_delta) < -float(tolerance)
    )
    return {
        "first_crossing_step": crossing_step,
        "saved_outer_micro_steps": terminal_step - crossing_step,
        "crossing_is_certified_lower_basin": certified_lower,
        "tradeoff_class": protocol.classify_tradeoff(
            crossing_delta,
            terminal_quench.get("landing_delta_eV"),
            tolerance,
        ),
    }


def evaluate_missing_energy(
    state,
    *,
    starter_energy: float,
    calculator,
) -> dict[str, Any]:
    from pamssw.accounting import EvaluationPurpose

    before = calculator.snapshot().as_dict()
    with calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        energy = float(calculator.evaluate(state).energy)
    after = calculator.snapshot().as_dict()
    counts = {name: int(after[name] - before[name]) for name in before}
    if sum(counts.values()) != 1:
        raise RuntimeError("missing-energy ledger must contain exactly one evaluation")
    finite = math.isfinite(energy)
    return {
        "checkpoint_energy_eV": energy if finite else None,
        "checkpoint_delta_eV": energy - float(starter_energy) if finite else None,
        "status": "completed" if finite else "unlearnable_nonfinite_energy",
        "new_force_evaluations": sum(counts.values()),
        "purpose_counts": counts,
        "evidence_origin": "new_true_pes_energy",
    }


def _aggregate_counts(rows) -> dict[str, int]:
    counts: Counter[str] = Counter()
    new_force_evaluations = 0
    for row in rows:
        counts.update(
            {
                str(name): int(value)
                for name, value in row["purpose_counts"].items()
            }
        )
        new_force_evaluations += int(row["new_force_evaluations"])
    for name in _zero_counts():
        counts[name] += 0
    if sum(counts.values()) != new_force_evaluations:
        raise RuntimeError("G-E0 purpose ledger does not close")
    forbidden = (
        "direction_oracle",
        "biased_proposal_relax",
        "unattributed",
    )
    if any(counts[name] for name in forbidden):
        raise RuntimeError("G-E0 contains forbidden force evaluations")
    return dict(sorted(counts.items()))


def _runtime(context: Mapping[str, Any], *, state_source_root: Path):
    source_runner = _load_module(
        SOURCE_RUNNER_PATH,
        "_true_energy_descent_source_runner",
    )
    audit = _load_module(
        source_runner.FIXED_AUDIT_PATH,
        "_true_energy_descent_fixed_audit",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()

    states = {}
    for system, state_id, _seed, _arm in context["cases_by_key"]:
        state_key = (system, state_id)
        if state_key in states:
            continue
        state, provenance = source_runner._load_starter(
            state_source_root=state_source_root,
            system=system,
            state_id=state_id,
            audit=audit,
            base_runner=base_runner,
        )
        expected_hashes = {
            str(case["state_sha256"])
            for key, case in context["cases_by_key"].items()
            if key[:2] == state_key
        }
        if expected_hashes != {provenance["state_sha256"]}:
            raise RuntimeError(f"starter state SHA256 drifted: {state_key}")
        states[state_key] = (state, provenance)

    # The model is constructed only after every source, endpoint, landing and
    # starter hash above has closed.
    from pamssw.calculators import ASECalculator

    calculator = ASECalculator(base_runner._calculator())
    gup0 = _load_module(GUP0_RUNNER_PATH, "_true_energy_descent_gup0_runner")
    return {
        "source_runner": source_runner,
        "gup0": gup0,
        "base_runner": base_runner,
        "calculator": calculator,
        "states": states,
    }


def _new_quench_row(
    *,
    step: int,
    endpoint: Mapping[str, Any],
    source_case: Mapping[str, Any],
    starter_state,
    calculator,
    config,
    case_dir: Path,
    remaining_budget: int,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    quenched = runtime["source_runner"]._quench_checkpoint(
        source={
            "horizon": int(step),
            "checkpoint_path": str(REPO_ROOT / endpoint["checkpoint_path"]),
            "checkpoint_sha256": endpoint["checkpoint_sha256"],
        },
        starter_state=starter_state,
        starter_energy=float(source_case["starter_energy_eV"]),
        calculator=calculator,
        config=config,
        system=str(source_case["system"]),
        case_dir=case_dir,
        remaining_budget=remaining_budget,
        base_runner=runtime["base_runner"],
    )
    return {
        **quenched,
        "step": int(step),
        "new_force_evaluations": int(quenched["force_evaluations"]),
        "evidence_origin": "new_true_quench",
    }


def _landing_relation(
    *,
    crossing: Mapping[str, Any],
    terminal: Mapping[str, Any],
    starter_state,
    config,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    if int(crossing["step"]) == int(terminal["step"]):
        return "SAME_LANDING", {
            "landing_matcher_same": True,
            "landing_descriptor_same": True,
            "landing_descriptor_delta": 0.0,
        }
    return runtime["gup0"]._landing_relation(
        explicit=crossing,
        relaxed=terminal,
        starter_state=starter_state,
        config=config,
    )


def _case_record(
    *,
    manifest_case: Mapping[str, Any],
    source_case: Mapping[str, Any],
    runtime: Mapping[str, Any],
    energy_counter,
    output_dir: Path,
    total_new_fe: int,
    started: float,
    max_new_fe: int,
    max_wall_s: float,
) -> tuple[dict[str, Any], int]:
    from pamssw.config import LSSSWConfig
    from pamssw.io import read_state
    from pamssw.walker import GeometryValidator

    key = _case_key(source_case)
    system, state_id, seed, arm = key
    starter_state, provenance = runtime["states"][(system, state_id)]
    starter_energy = float(source_case["starter_energy_eV"])
    config = LSSSWConfig(**source_case["effective_config"])
    tolerance = float(config.dedup_energy_tol)
    source_checkpoints = {
        int(row["horizon"]): row for row in source_case["checkpoints"]
    }
    endpoint_by_step = {
        int(row["step"]): row for row in manifest_case["endpoints"]
    }
    case_dir = (
        output_dir
        / "cases"
        / system
        / state_id
        / f"seed-{seed:08d}"
        / arm
    )
    energy_rows = []
    validator = GeometryValidator()
    for step, endpoint in endpoint_by_step.items():
        if perf_counter() - started > max_wall_s:
            raise RuntimeError("G-E0 kernel wall budget exhausted")
        source_checkpoint = source_checkpoints.get(step)
        source_energy = (
            None
            if source_checkpoint is None
            else source_checkpoint.get("checkpoint_energy_eV")
        )
        if source_energy is not None and math.isfinite(float(source_energy)):
            row = reused_energy_row(source_checkpoint)
        else:
            if total_new_fe >= max_new_fe:
                raise RuntimeError("G-E0 force-evaluation budget exhausted")
            state = read_state(
                REPO_ROOT / endpoint["checkpoint_path"],
                fixed_mask=starter_state.fixed_mask,
            )
            if not validator.is_valid_state(state):
                row = {
                    "checkpoint_energy_eV": None,
                    "checkpoint_delta_eV": None,
                    "status": "unlearnable_invalid_geometry",
                    "new_force_evaluations": 0,
                    "purpose_counts": _zero_counts(),
                    "evidence_origin": "geometry_validation",
                }
            else:
                row = evaluate_missing_energy(
                    state,
                    starter_energy=starter_energy,
                    calculator=energy_counter,
                )
                total_new_fe += int(row["new_force_evaluations"])
            row["step"] = step
        energy_rows.append(
            {
                **row,
                "checkpoint_path": endpoint["checkpoint_path"],
                "checkpoint_sha256": endpoint["checkpoint_sha256"],
            }
        )

    crossing = protocol.first_descent_crossing(
        energy_rows,
        tolerance=tolerance,
    )
    quench_rows: dict[int, dict[str, Any]] = {}
    if crossing is not None:
        crossing_step = int(crossing["step"])
        terminal_step = int(source_case["reached_macro_steps"])
        reusable = set(source_checkpoints)
        for step in sorted({crossing_step, terminal_step} & reusable):
            quench_rows[step] = reused_quench_row(source_checkpoints[step])
        for step in required_quench_steps(
            first_crossing=crossing_step,
            terminal=terminal_step,
            reusable=reusable,
        ):
            if total_new_fe >= max_new_fe:
                raise RuntimeError("G-E0 force-evaluation budget exhausted")
            if perf_counter() - started > max_wall_s:
                raise RuntimeError("G-E0 kernel wall budget exhausted")
            quench = _new_quench_row(
                step=step,
                endpoint=endpoint_by_step[step],
                source_case=source_case,
                starter_state=starter_state,
                calculator=runtime["calculator"],
                config=config,
                case_dir=case_dir / f"step-{step:03d}-quench",
                remaining_budget=max_new_fe - total_new_fe,
                runtime=runtime,
            )
            total_new_fe += int(quench["new_force_evaluations"])
            quench_rows[step] = quench

    summary = summarize_case_outcome(
        energy_rows=energy_rows,
        quench_rows=quench_rows,
        tolerance=tolerance,
    )
    relation = "NOT_APPLICABLE"
    relation_metrics = {
        "landing_matcher_same": None,
        "landing_descriptor_same": None,
        "landing_descriptor_delta": None,
    }
    if crossing is not None:
        crossing_quench = quench_rows[int(crossing["step"])]
        terminal_quench = quench_rows[int(source_case["reached_macro_steps"])]
        relation, relation_metrics = _landing_relation(
            crossing=crossing_quench,
            terminal=terminal_quench,
            starter_state=starter_state,
            config=config,
            runtime=runtime,
        )
        summary.update(
            {
                "crossing_checkpoint_delta_eV": crossing[
                    "checkpoint_delta_eV"
                ],
                "crossing_landing_delta_eV": crossing_quench.get(
                    "landing_delta_eV"
                ),
                "terminal_landing_delta_eV": terminal_quench.get(
                    "landing_delta_eV"
                ),
            }
        )
    else:
        summary.update(
            {
                "crossing_checkpoint_delta_eV": None,
                "crossing_landing_delta_eV": None,
                "terminal_landing_delta_eV": None,
            }
        )
    all_cost_rows = energy_rows + list(quench_rows.values())
    case_counts = _aggregate_counts(all_cost_rows)
    record = {
        "system": system,
        "state_id": state_id,
        "seed": seed,
        "arm": arm,
        "state_sha256": provenance["state_sha256"],
        "starter_energy_eV": starter_energy,
        "reached_macro_steps": int(source_case["reached_macro_steps"]),
        "attempted_macro_steps": int(source_case["attempted_macro_steps"]),
        "dedup_energy_tol_eV": tolerance,
        "energy_rows": energy_rows,
        "quench_rows": [quench_rows[step] for step in sorted(quench_rows)],
        **summary,
        "crossing_terminal_landing_relation": relation,
        **relation_metrics,
        "new_force_evaluations": sum(case_counts.values()),
        "purpose_counts": case_counts,
    }
    _write_json(case_dir / "partial.json", record)
    print(
        f"[G-E0] {system} {state_id} seed={seed} arm={arm} "
        f"crossing={record['first_crossing_step']} "
        f"saved={record['saved_outer_micro_steps']} "
        f"tradeoff={record['tradeoff_class']} "
        f"new_fe={record['new_force_evaluations']}",
        flush=True,
    )
    return record, total_new_fe


def _aggregate_evidence(
    *,
    cases,
    source_context: Mapping[str, Any],
    commit: str,
    wall_time_s: float,
    max_new_fe: int,
    max_wall_s: float,
) -> dict[str, Any]:
    all_cost_rows = [
        row
        for case in cases
        for row in case["energy_rows"] + case["quench_rows"]
    ]
    purpose_counts = _aggregate_counts(all_cost_rows)
    crossing_cases = [
        case for case in cases if case["first_crossing_step"] is not None
    ]
    tradeoff_counts = Counter(
        str(case["tradeoff_class"]) for case in crossing_cases
    )

    source_labeled_escapes = 0
    source_labeled_escape_crossings = 0
    for source_case in source_context["raw"]["cases"]:
        tolerance = float(source_case["effective_config"]["dedup_energy_tol"])
        for checkpoint in source_case["checkpoints"]:
            delta = checkpoint.get("checkpoint_delta_eV")
            if checkpoint.get("label") != "ESCAPED_CERTIFIED" or delta is None:
                continue
            source_labeled_escapes += 1
            source_labeled_escape_crossings += int(
                float(delta) < -tolerance
            )

    certified = sum(
        bool(case["crossing_is_certified_lower_basin"])
        for case in crossing_cases
    )
    saved_steps = sum(
        int(case["saved_outer_micro_steps"]) for case in crossing_cases
    )
    ledger_closes = sum(purpose_counts.values()) == sum(
        int(case["new_force_evaluations"]) for case in cases
    )
    admit_g_e1 = bool(
        len(crossing_cases) >= 2
        and certified == len(crossing_cases)
        and saved_steps > 0
        and ledger_closes
    )
    return {
        "schema_version": 1,
        "execution_commit": commit,
        "source_raw_evidence_path": str(source_context["raw_path"]),
        "source_raw_evidence_sha256": source_context["source_raw_sha256"],
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "cohort": {
            "case_count": len(cases),
            "accepted_endpoint_count": sum(
                int(case["reached_macro_steps"]) for case in cases
            ),
            "attempted_endpoint_count": sum(
                int(case["attempted_macro_steps"]) for case in cases
            ),
        },
        "cases": cases,
        "aggregate": {
            "crossing_path_count": len(crossing_cases),
            "certified_lower_basin_count": certified,
            "trigger_precision_on_crossing_paths": (
                None
                if not crossing_cases
                else certified / len(crossing_cases)
            ),
            "source_labeled_escape_count": source_labeled_escapes,
            "source_labeled_escape_crossing_count": (
                source_labeled_escape_crossings
            ),
            "trigger_coverage_on_source_labeled_horizons": (
                None
                if source_labeled_escapes == 0
                else source_labeled_escape_crossings / source_labeled_escapes
            ),
            "saved_outer_micro_steps": saved_steps,
            "tradeoff_counts": dict(sorted(tradeoff_counts.items())),
            "new_force_evaluations": sum(purpose_counts.values()),
            "purpose_counts": purpose_counts,
            "wall_time_s": float(wall_time_s),
        },
        "budgets": {
            "max_new_force_evaluations": int(max_new_fe),
            "max_kernel_wall_time_s": float(max_wall_s),
        },
        "decision": (
            "ADMIT_FRESH_ONLINE_G_E1"
            if admit_g_e1
            else "DO_NOT_ADMIT_G_E1"
        ),
        "production_default_changed": False,
        "claim_ceiling": (
            "offline first-descent certificate audit on the frozen 24-path "
            "C60/PdO D0/K4 corpus; this is not an online stopping result or "
            "a production-default recommendation"
        ),
    }


def run(
    *,
    output_dir: Path,
    expected_commit: str | None,
    state_source_root: Path,
    limit_cases: int | None,
    max_new_fe: int,
    max_wall_s: float,
) -> dict[str, Any]:
    commit = _current_commit()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if limit_cases is None and not _tracked_clean():
        raise RuntimeError("full G-E0 gate requires a clean tracked worktree")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    source_context = validate_source_inputs()
    manifest_cases = list(source_context["manifest"]["cases"])
    if limit_cases is not None:
        if limit_cases <= 0:
            raise ValueError("--limit-cases must be positive")
        manifest_cases = manifest_cases[:limit_cases]
    runtime = _runtime(
        source_context,
        state_source_root=state_source_root.resolve(),
    )
    output_dir.mkdir(parents=True)

    from pamssw.accounting import EvalCounter

    energy_counter = EvalCounter(
        runtime["calculator"],
        max_force_evals=max_new_fe,
    )
    started = perf_counter()
    total_new_fe = 0
    cases = []
    for manifest_case in manifest_cases:
        key_values = manifest_case["key"]
        key = (
            str(key_values[0]),
            str(key_values[1]),
            int(key_values[2]),
            str(key_values[3]),
        )
        case, total_new_fe = _case_record(
            manifest_case=manifest_case,
            source_case=source_context["cases_by_key"][key],
            runtime=runtime,
            energy_counter=energy_counter,
            output_dir=output_dir,
            total_new_fe=total_new_fe,
            started=started,
            max_new_fe=max_new_fe,
            max_wall_s=max_wall_s,
        )
        cases.append(case)
        partial = _aggregate_evidence(
            cases=cases,
            source_context=source_context,
            commit=commit,
            wall_time_s=perf_counter() - started,
            max_new_fe=max_new_fe,
            max_wall_s=max_wall_s,
        )
        _write_json(output_dir / "partial.json", partial)
    evidence = _aggregate_evidence(
        cases=cases,
        source_context=source_context,
        commit=commit,
        wall_time_s=perf_counter() - started,
        max_new_fe=max_new_fe,
        max_wall_s=max_wall_s,
    )
    if int(evidence["aggregate"]["new_force_evaluations"]) != total_new_fe:
        raise RuntimeError("G-E0 total force-evaluation ledger does not close")
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def check_evidence(path: Path, *, require_full: bool = True) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    cases = evidence["cases"]
    if require_full and evidence["cohort"] != {
        "case_count": 24,
        "accepted_endpoint_count": 82,
        "attempted_endpoint_count": 93,
    }:
        raise RuntimeError("G-E0 evidence does not contain the full cohort")
    for case in cases:
        expected_steps = list(range(1, int(case["reached_macro_steps"]) + 1))
        if [int(row["step"]) for row in case["energy_rows"]] != expected_steps:
            raise RuntimeError("G-E0 case energy steps are not consecutive")
    all_rows = [
        row
        for case in cases
        for row in case["energy_rows"] + case["quench_rows"]
    ]
    counts = _aggregate_counts(all_rows)
    if counts != evidence["aggregate"]["purpose_counts"]:
        raise RuntimeError("G-E0 aggregate purpose counts drifted")
    if sum(counts.values()) != int(
        evidence["aggregate"]["new_force_evaluations"]
    ):
        raise RuntimeError("G-E0 aggregate force evaluations drifted")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RUN_ROOT / "output",
    )
    parser.add_argument("--expected-commit")
    parser.add_argument("--state-source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--limit-cases", type=int)
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
            require_full=args.limit_cases is None,
        )
        print(json.dumps(checked["aggregate"], indent=2, sort_keys=True))
        return
    evidence = run(
        output_dir=args.output_dir,
        expected_commit=args.expected_commit,
        state_source_root=args.state_source_root,
        limit_cases=args.limit_cases,
        max_new_fe=args.max_new_fe,
        max_wall_s=args.max_wall_s,
    )
    print(json.dumps(evidence["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
