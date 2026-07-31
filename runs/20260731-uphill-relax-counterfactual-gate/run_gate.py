#!/usr/bin/env python3
"""Replay pre-relax SSW displacements and quench them on the true PES."""

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
FIRST_PASSAGE_PATH = (
    REPO_ROOT
    / "runs"
    / "20260731-current-action-first-passage"
    / "run_gate.py"
)
SOURCE_SUMMARY_PATH = FIRST_PASSAGE_PATH.with_name("evidence.json")


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
    "_uphill_relax_counterfactual_protocol_runner",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def verify_source_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != expected_sha256:
        raise RuntimeError(f"source SHA256 drifted: {path}")


def extract_explicit_state(path: Path, template):
    """Read the pre-optimization state recorded as trajectory frame zero."""

    from pamssw.io import read_state

    return read_state(
        path,
        index=0,
        fixed_mask=template.fixed_mask,
        metadata={
            "counterfactual_source": str(path),
            "counterfactual_frame": 0,
        },
    )


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


def _resolve_source(
    source_summary_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    raw_path = Path(summary["raw_evidence_path"])
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path
    verify_source_file(raw_path, str(summary["raw_evidence_sha256"]))
    source = json.loads(raw_path.read_text(encoding="utf-8"))
    specs = protocol.pair_specs(source["cases"])
    if len(specs) != 34:
        raise RuntimeError("source does not provide the frozen 34-pair cohort")
    return summary, raw_path, source


def _case_index(source: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    result = {}
    for case in source["cases"]:
        if case["system"] != "c60":
            continue
        key = (str(case["state_id"]), int(case["seed"]), str(case["arm"]))
        if key in result:
            raise RuntimeError("source contains duplicate action cases")
        result[key] = dict(case)
    return result


def _source_cost(source: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_keys = {
        (str(spec["state_id"]), int(spec["seed"]), str(spec["arm"]))
        for spec in specs
    }
    counts: Counter[str] = Counter()
    total_generation = 0
    selected_checkpoint_fe = 0
    for case in source["cases"]:
        key = (str(case["state_id"]), int(case["seed"]), str(case["arm"]))
        if case["system"] != "c60" or key not in case_keys:
            continue
        total_generation += int(case["generation_force_evaluations"])
        counts.update(
            {
                name: int(value)
                for name, value in case["generation_purpose_counts"].items()
            }
        )
        selected = {
            int(spec["horizon"])
            for spec in specs
            if (
                str(spec["state_id"]),
                int(spec["seed"]),
                str(spec["arm"]),
            )
            == key
        }
        selected_checkpoint_fe += sum(
            int(checkpoint["force_evaluations"])
            for checkpoint in case["checkpoints"]
            if int(checkpoint["horizon"]) in selected
        )
    if sum(counts.values()) != total_generation or counts["unattributed"] != 0:
        raise RuntimeError("source generation ledger does not close")
    return {
        "generation_force_evaluations": total_generation,
        "generation_purpose_counts": dict(sorted(counts.items())),
        "selected_relaxed_checkpoint_force_evaluations": selected_checkpoint_fe,
        "cost_scope_note": (
            "generation cost is the exact full 12-action C60 source corpus; "
            "it includes later source steps when an action continued beyond h4"
        ),
    }


def _landing_relation(
    *,
    explicit: Mapping[str, Any],
    relaxed: Mapping[str, Any],
    starter_state,
    config,
) -> tuple[str, dict[str, Any]]:
    if (
        explicit["label"] != "ESCAPED_CERTIFIED"
        or relaxed["label"] != "ESCAPED_CERTIFIED"
    ):
        learnable = {
            "RETURN_STARTER",
            "ESCAPED_CERTIFIED",
        }
        relation = (
            "NOT_APPLICABLE"
            if explicit["label"] in learnable and relaxed["label"] in learnable
            else "NOT_COMPARABLE"
        )
        return relation, {
            "landing_matcher_same": None,
            "landing_descriptor_same": None,
            "landing_descriptor_delta": None,
        }

    from pamssw.archive import MinimaArchive
    from pamssw.fingerprint import descriptor_distance, structural_descriptor
    from pamssw.io import read_state

    explicit_path = Path(explicit["landing_path"])
    relaxed_path = Path(relaxed["landing_path"])
    verify_source_file(relaxed_path, str(relaxed["landing_sha256"]))
    explicit_state = read_state(
        explicit_path,
        fixed_mask=starter_state.fixed_mask,
    )
    relaxed_state = read_state(
        relaxed_path,
        fixed_mask=starter_state.fixed_mask,
    )
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    archive.add(
        relaxed_state,
        float(relaxed["landing_energy_eV"]),
        parent_id=None,
    )
    matcher_same = (
        archive.find_match(
            explicit_state,
            float(explicit["landing_energy_eV"]),
        )
        is not None
    )
    descriptor_delta = float(
        descriptor_distance(
            structural_descriptor(relaxed_state),
            structural_descriptor(explicit_state),
        )
    )
    descriptor_same = descriptor_delta < config.min_escape_descriptor_delta
    if matcher_same != descriptor_same:
        relation = "AMBIGUOUS_LANDING"
    else:
        relation = "SAME_LANDING" if matcher_same else "DIFFERENT_LANDING"
    return relation, {
        "landing_matcher_same": bool(matcher_same),
        "landing_descriptor_same": bool(descriptor_same),
        "landing_descriptor_delta": descriptor_delta,
    }


def _pair_dir(output_dir: Path, spec: Mapping[str, Any]) -> Path:
    return (
        output_dir
        / "pairs"
        / str(spec["state_id"])
        / f"seed-{int(spec['seed']):08d}"
        / str(spec["arm"])
        / f"h{int(spec['horizon']):02d}"
    )


def run(
    *,
    output_dir: Path,
    source_summary_path: Path = SOURCE_SUMMARY_PATH,
    state_source_root: Path = REPO_ROOT,
    expected_commit: str | None = None,
    limit: int | None = None,
    max_new_force_evaluations: int = protocol.MAX_NEW_FORCE_EVALUATIONS,
) -> dict[str, Any]:
    commit = _current_commit()
    if expected_commit is not None and commit != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if limit is None and not _tracked_clean():
        raise RuntimeError("full gate requires a clean tracked worktree")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    source_summary, raw_source_path, source = _resolve_source(
        source_summary_path.resolve()
    )
    specs = protocol.pair_specs(source["cases"])
    full_specs = list(specs)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        specs = specs[:limit]
    cases = _case_index(source)

    from pamssw.calculators import ASECalculator
    from pamssw.io import write_state

    first_passage = _load_module(
        FIRST_PASSAGE_PATH,
        "_uphill_relax_counterfactual_first_passage",
    )
    audit = _load_module(
        first_passage.FIXED_AUDIT_PATH,
        "_uphill_relax_counterfactual_fixed_audit",
    )
    config_gate = _load_module(
        first_passage.CONFIG_GATE_PATH,
        "_uphill_relax_counterfactual_config_gate",
    )
    action_runner = _load_module(
        first_passage.ACTION_RUNNER_PATH,
        "_uphill_relax_counterfactual_action_runner",
    )
    _strict_wrapper, base_runner = audit._load_frozen_runtime()
    calculator = ASECalculator(base_runner._calculator())

    states = {}
    for state_id in protocol.STATE_IDS:
        state, provenance = first_passage._load_starter(
            state_source_root=state_source_root.resolve(),
            system="c60",
            state_id=state_id,
            audit=audit,
            base_runner=base_runner,
        )
        states[state_id] = (state, provenance)

    rows: list[dict[str, Any]] = []
    total_new_fe = 0
    started = perf_counter()
    for spec in specs:
        if total_new_fe >= max_new_force_evaluations:
            raise RuntimeError("new force-evaluation budget exhausted")
        pair_dir = _pair_dir(output_dir, spec)
        pair_dir.mkdir(parents=True, exist_ok=False)
        raw_path = Path(spec["raw_optimizer_checkpoint_path"])
        verify_source_file(
            raw_path,
            str(spec["raw_optimizer_checkpoint_sha256"]),
        )
        starter_state, provenance = states[str(spec["state_id"])]
        explicit_state = extract_explicit_state(raw_path, starter_state)
        explicit_path = pair_dir / "explicit_pre_relax.xyz"
        write_state(explicit_path, explicit_state)
        explicit_hash = _sha256(explicit_path)

        case_key = (
            str(spec["state_id"]),
            int(spec["seed"]),
            str(spec["arm"]),
        )
        source_case = cases[case_key]
        config = action_runner._base_config(
            config_gate,
            "c60",
            int(spec["seed"]),
            pair_dir,
        )
        explicit = first_passage._quench_checkpoint(
            source={
                "horizon": int(spec["horizon"]),
                "checkpoint_path": str(explicit_path),
                "checkpoint_sha256": explicit_hash,
                "source_optimizer_trajectory_path": str(raw_path),
                "source_optimizer_trajectory_sha256": str(
                    spec["raw_optimizer_checkpoint_sha256"]
                ),
                "source_optimizer_frame": 0,
            },
            starter_state=starter_state,
            starter_energy=float(source_case["starter_energy_eV"]),
            calculator=calculator,
            config=config,
            system="c60",
            case_dir=pair_dir,
            remaining_budget=max_new_force_evaluations - total_new_fe,
            base_runner=base_runner,
        )
        total_new_fe += int(explicit["force_evaluations"])
        relaxed = dict(spec["relaxed_checkpoint"])
        relation, relation_metrics = _landing_relation(
            explicit=explicit,
            relaxed=relaxed,
            starter_state=starter_state,
            config=config,
        )
        outcome = protocol.causal_outcome(
            explicit_label=str(explicit["label"]),
            relaxed_label=str(relaxed["label"]),
            landing_relation=relation,
        )
        row = {
            "system": "c60",
            "state_id": str(spec["state_id"]),
            "seed": int(spec["seed"]),
            "arm": str(spec["arm"]),
            "horizon": int(spec["horizon"]),
            "status": "completed",
            "starter_state_sha256": provenance["state_sha256"],
            "explicit_label": str(explicit["label"]),
            "relaxed_label": str(relaxed["label"]),
            "landing_relation": relation,
            "causal_outcome": outcome,
            "explicit_checkpoint_energy_eV": explicit.get(
                "checkpoint_energy_eV"
            ),
            "relaxed_checkpoint_energy_eV": relaxed.get(
                "checkpoint_energy_eV"
            ),
            "explicit_landing_energy_eV": explicit.get("landing_energy_eV"),
            "relaxed_landing_energy_eV": relaxed.get("landing_energy_eV"),
            "explicit_landing_path": explicit.get("landing_path"),
            "explicit_landing_sha256": explicit.get("landing_sha256"),
            "relaxed_landing_path": relaxed.get("landing_path"),
            "relaxed_landing_sha256": relaxed.get("landing_sha256"),
            "explicit_pre_relax_path": str(explicit_path),
            "explicit_pre_relax_sha256": explicit_hash,
            "source_optimizer_trajectory_path": str(raw_path),
            "source_optimizer_trajectory_sha256": str(
                spec["raw_optimizer_checkpoint_sha256"]
            ),
            "source_optimizer_frame": 0,
            "new_force_evaluations": int(explicit["force_evaluations"]),
            "explicit_purpose_counts": dict(explicit["purpose_counts"]),
            "explicit_wall_time_s": float(explicit["wall_time_s"]),
            **relation_metrics,
        }
        if (
            row["explicit_landing_energy_eV"] is not None
            and row["relaxed_landing_energy_eV"] is not None
        ):
            row["explicit_minus_relaxed_landing_energy_eV"] = float(
                row["explicit_landing_energy_eV"]
                - row["relaxed_landing_energy_eV"]
            )
        else:
            row["explicit_minus_relaxed_landing_energy_eV"] = None
        rows.append(row)
        _write_json(pair_dir / "summary.json", row)
        _write_json(
            output_dir / "partial.json",
            {
                "schema_version": 1,
                "execution_commit": commit,
                "pairs": rows,
                "new_force_evaluations": total_new_fe,
            },
        )
        print(
            "[G-UP0] "
            f"{row['state_id']} seed={row['seed']} {row['arm']} "
            f"h={row['horizon']} outcome={outcome} "
            f"new_fe={row['new_force_evaluations']}",
            flush=True,
        )

    if limit is None:
        evidence = protocol.build_evidence(
            rows,
            expected_specs=full_specs,
            max_new_force_evaluations=max_new_force_evaluations,
        )
    else:
        evidence = {
            "schema_version": 1,
            "smoke_only": True,
            "pair_count": len(rows),
            "pairs": rows,
            "new_force_evaluations": total_new_fe,
            "max_new_force_evaluations": max_new_force_evaluations,
        }
    evidence.update(
        {
            "execution_commit": commit,
            "source_summary_path": str(source_summary_path.resolve()),
            "source_summary_sha256": _sha256(source_summary_path.resolve()),
            "source_raw_evidence_path": str(raw_source_path),
            "source_raw_evidence_sha256": str(
                source_summary["raw_evidence_sha256"]
            ),
            "source_corpus_cost": _source_cost(source, full_specs),
            "wall_time_s": float(perf_counter() - started),
        }
    )
    _write_json(output_dir / "evidence.json", evidence)
    return evidence


def check_evidence(
    evidence_path: Path,
    *,
    source_summary_path: Path = SOURCE_SUMMARY_PATH,
) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("smoke_only"):
        raise ValueError("smoke evidence cannot satisfy the full gate")
    source_summary, raw_source_path, source = _resolve_source(
        source_summary_path.resolve()
    )
    specs = protocol.pair_specs(source["cases"])
    rebuilt = protocol.build_evidence(
        evidence["pairs"],
        expected_specs=specs,
        max_new_force_evaluations=int(evidence["max_new_force_evaluations"]),
    )
    for key in (
        "cohort",
        "outcome_counts",
        "explicit_label_counts",
        "relaxed_label_counts",
        "new_force_evaluations",
        "decision",
        "repeated_relaxed_only_contexts",
    ):
        if evidence[key] != rebuilt[key]:
            raise RuntimeError(f"evidence field drifted: {key}")
    if evidence["source_raw_evidence_sha256"] != source_summary["raw_evidence_sha256"]:
        raise RuntimeError("source evidence hash drifted")
    verify_source_file(raw_source_path, evidence["source_raw_evidence_sha256"])
    for row in evidence["pairs"]:
        verify_source_file(
            Path(row["explicit_pre_relax_path"]),
            row["explicit_pre_relax_sha256"],
        )
        if row["explicit_landing_path"] is not None:
            verify_source_file(
                Path(row["explicit_landing_path"]),
                row["explicit_landing_sha256"],
            )
    return {
        "pair_count": len(evidence["pairs"]),
        "new_force_evaluations": evidence["new_force_evaluations"],
        "decision": evidence["decision"],
        "status": "valid",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-summary", type=Path, default=SOURCE_SUMMARY_PATH)
    parser.add_argument("--state-source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--expected-commit")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-new-force-evaluations",
        type=int,
        default=protocol.MAX_NEW_FORCE_EVALUATIONS,
    )
    parser.add_argument("--check-evidence", type=Path)
    args = parser.parse_args()
    if args.check_evidence is not None:
        checked = check_evidence(
            args.check_evidence.resolve(),
            source_summary_path=args.source_summary.resolve(),
        )
        print(json.dumps(checked, indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required unless --check-evidence is used")
    evidence = run(
        output_dir=args.output_dir.resolve(),
        source_summary_path=args.source_summary.resolve(),
        state_source_root=args.state_source_root.resolve(),
        expected_commit=args.expected_commit,
        limit=args.limit,
        max_new_force_evaluations=args.max_new_force_evaluations,
    )
    print(
        json.dumps(
            {
                "smoke_only": evidence.get("smoke_only", False),
                "pair_count": len(evidence["pairs"]),
                "new_force_evaluations": evidence[
                    "new_force_evaluations"
                ],
                "wall_time_s": evidence["wall_time_s"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
