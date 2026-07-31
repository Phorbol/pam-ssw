#!/usr/bin/env python3
"""Run the G-E0 true-PES descent early-stop audit."""

from __future__ import annotations

import importlib.util
import json
import math
from hashlib import sha256
from pathlib import Path
import sys
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
