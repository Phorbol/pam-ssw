#!/usr/bin/env python3
"""Strictly re-quench the frozen residual PdO endpoint pair on CUDA."""

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
    frozen = corpus["protocol"]
    if (
        frozen["optimizer"] != "scipy-lbfgsb"
        or frozen["strict_fmax_eV_per_A"] != 0.01
        or frozen["maxiter"] != 400
        or len(corpus["endpoints"]) != 2
    ):
        raise RuntimeError("strict re-quench corpus protocol drifted")

    replay = _load_module(
        STRICT_REPLAY_PATH,
        "_pdo_matcher_strict_replay",
    )
    production = replay._production_module()
    preflight = production.preflight(
        system="pdo",
        expected_git_commit=expected_commit,
    )
    arm = replay.Arm("scipy-lbfgsb", "scipy-lbfgsb", None)
    counter = replay._counter(replay._calculator())
    rows = []
    started = perf_counter()
    for index, endpoint in enumerate(corpus["endpoints"]):
        state = _state(endpoint["state"])
        task = replay.LandingTask(
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
        row = replay.execute_strict_true_quench(task, arm, counter)
        row["endpoint"] = endpoint["endpoint"]
        rows.append(row)
    elapsed = perf_counter() - started
    counts = counter.snapshot()
    if counts.total != sum(int(row["force_evaluations"]) for row in rows):
        raise RuntimeError("outer strict re-quench accounting does not close")

    output_dir.mkdir(parents=True)
    (output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
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
        "protocol": frozen,
        "endpoint_count": len(rows),
        "evaluation_counts": counts.as_dict(),
        "total_force_evaluations": counts.total,
        "total_wall_time_s": elapsed,
        "rows_path": str((output_dir / "rows.json").relative_to(REPO_ROOT)),
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
    summary = run(
        output_dir=args.output.resolve(),
        expected_commit=args.expected_commit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
