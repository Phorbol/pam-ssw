#!/usr/bin/env python3
"""Run the bounded ASE-LBFGS -> FIRE strict-certificate closure."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from pamssw.state import State


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
CORPUS_PATH = RUN_ROOT / "strict_requench_corpus.json"
STRICT_REPLAY_PATH = (
    REPO_ROOT
    / "runs"
    / "20260728-true-quench-raw-strict-replay"
    / "run_replay.py"
)
EXPECTED_CORPUS_SHA256 = (
    "c04854175fb214b607684c69968016ee3b5534a9c85609b55b5bb970122845ec"
)
STRICT_FMAX = 0.01
MAXITER = 400
MAX_FORCE_EVALUATIONS = 1604


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _state(payload: Mapping[str, Any]) -> State:
    return State(
        numbers=np.asarray(payload["numbers"], dtype=int),
        positions=np.asarray(payload["positions"], dtype=float),
        cell=(
            None
            if payload["cell"] is None
            else np.asarray(payload["cell"], dtype=float)
        ),
        pbc=tuple(bool(value) for value in payload["pbc"]),
        fixed_mask=np.asarray(payload["fixed_mask"], dtype=bool),
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


def _tracked(path: Path) -> bool:
    relative = path.resolve().relative_to(REPO_ROOT.resolve())
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def has_strict_certificate(
    row: Mapping[str, Any],
    *,
    fmax: float,
) -> bool:
    return (
        row["termination_reason"] == "converged"
        and float(row["final"]["max_active_force_eV_per_A"]) <= float(fmax)
    )


def _task(replay, endpoint: Mapping[str, Any], index: int):
    state = _state(endpoint["state"])
    return replay.LandingTask(
        system="pdo",
        task_index=index,
        trial_index=1,
        proposal_index=index + 1,
        source_trajectory_path=endpoint["source_path"],
        source_trajectory_sha256=endpoint["source_sha256"],
        source_frame_index=0,
        initial_positions_sha256=replay.position_sha256(state.positions),
        state=state,
    )


def _fallback_task(replay, primary: Mapping[str, Any], index: int):
    state = _state(primary["final"]["state"])
    digest = replay.position_sha256(state.positions)
    return replay.LandingTask(
        system="pdo",
        task_index=index,
        trial_index=2,
        proposal_index=index + 1,
        source_trajectory_path="ase-lbfgs-terminal-state",
        source_trajectory_sha256=digest,
        source_frame_index=0,
        initial_positions_sha256=digest,
        state=state,
    )


def run(*, output_dir: Path, expected_commit: str) -> dict[str, Any]:
    if _current_commit() != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if not _tracked(CORPUS_PATH):
        raise RuntimeError("strict re-quench corpus must be tracked")
    if _sha256(CORPUS_PATH) != EXPECTED_CORPUS_SHA256:
        raise RuntimeError("strict re-quench corpus checksum drifted")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    replay = _load_module(
        STRICT_REPLAY_PATH,
        "_pdo_matcher_certificate_replay",
    )
    production = replay._production_module()
    preflight = production.preflight(
        system="pdo",
        expected_git_commit=expected_commit,
    )
    counter = replay._counter(replay._calculator())
    primary_arm = replay.Arm("ase-lbfgs", "ase-lbfgs", None)
    fallback_arm = replay.Arm("ase-fire", "ase-fire", None)

    records = []
    started = perf_counter()
    for index, endpoint in enumerate(corpus["endpoints"]):
        primary = replay.execute_strict_true_quench(
            _task(replay, endpoint, index),
            primary_arm,
            counter,
        )
        fallback = None
        if not has_strict_certificate(primary, fmax=STRICT_FMAX):
            fallback = replay.execute_strict_true_quench(
                _fallback_task(replay, primary, index),
                fallback_arm,
                counter,
            )
        records.append(
            {
                "endpoint": endpoint["endpoint"],
                "primary": primary,
                "fallback": fallback,
                "final_stage": (
                    "primary" if fallback is None else "fallback"
                ),
            }
        )
    elapsed = perf_counter() - started
    counts = counter.snapshot()
    row_total = sum(
        int(record["primary"]["force_evaluations"])
        + (
            0
            if record["fallback"] is None
            else int(record["fallback"]["force_evaluations"])
        )
        for record in records
    )
    if counts.total != row_total:
        raise RuntimeError("certificate-rescue accounting does not close")
    if counts.total > MAX_FORCE_EVALUATIONS:
        raise RuntimeError("certificate-rescue force budget exceeded")

    output_dir.mkdir(parents=True)
    (output_dir / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "execution_commit": expected_commit,
        "corpus_path": str(CORPUS_PATH.relative_to(REPO_ROOT)),
        "corpus_sha256": EXPECTED_CORPUS_SHA256,
        "model_path": preflight["model_path"],
        "model_sha256": preflight["model_sha256"],
        "calculator": preflight["calculator"],
        "cuda": preflight["cuda"],
        "protocol": {
            "primary_optimizer": "ase-lbfgs",
            "fallback_optimizer": "ase-fire",
            "fallback_trigger": "raw_active_max_force_certificate_failure",
            "fmax_eV_per_A": STRICT_FMAX,
            "maxiter_per_stage": MAXITER,
            "max_force_evaluations": MAX_FORCE_EVALUATIONS,
        },
        "endpoint_count": len(records),
        "fallback_count": sum(
            record["fallback"] is not None for record in records
        ),
        "evaluation_counts": counts.as_dict(),
        "total_force_evaluations": counts.total,
        "total_wall_time_s": elapsed,
        "records_path": str(
            (output_dir / "records.json").relative_to(REPO_ROOT)
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    result = run(
        output_dir=args.output.resolve(),
        expected_commit=args.expected_commit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
