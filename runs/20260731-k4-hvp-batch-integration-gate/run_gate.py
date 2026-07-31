#!/usr/bin/env python3
"""Validate K=4 direction-oracle batching through the real pamssw call path."""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
SOURCE_ROOT = Path("/mnt/d/download/trae-research-code/ssw")
MODEL_PATH = Path("/root/.cache/mace/mace-omat-0-small.model")
MODEL_SHA256 = (
    "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
)
SYSTEMS = {
    "c60": {
        "path": SOURCE_ROOT
        / "runs"
        / "20260428-c60-mace-production"
        / "prerelaxed_c60.xyz",
        "sha256": (
            "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9"
        ),
        "pbc": (False, False, False),
        "seed": 4200,
    },
    "pdo": {
        "path": SOURCE_ROOT / "PdO.xyz",
        "sha256": (
            "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0"
        ),
        "pbc": (True, True, False),
        "seed": 4300,
    },
}
CALCULATOR_CONFIG = {
    "device": "cuda",
    "default_dtype": "float32",
    "enable_cueq": False,
}
CONTEXTS = ("initial", "momentum")
MODES = ("serial", "batch")
REPETITIONS = 5
EXPECTED_FORCE_EVALUATIONS = (
    len(SYSTEMS)
    * len(CONTEXTS)
    * len(MODES)
    * (REPETITIONS + 1)
    * 8
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
    RUN_ROOT / "protocol.py",
    "_k4_hvp_batch_integration_protocol",
)


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def _preflight(expected_commit: str) -> dict[str, object]:
    import mace
    import torch

    if _current_commit() != expected_commit:
        raise RuntimeError("execution commit differs from --expected-commit")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree must be clean")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if _sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("MACE model checksum drifted")
    for spec in SYSTEMS.values():
        if _sha256(spec["path"]) != spec["sha256"]:
            raise RuntimeError(f"input checksum drifted: {spec['path']}")
    return {
        "execution_commit": expected_commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mace": getattr(mace, "__version__", "unknown"),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_device": str(torch.cuda.get_device_name(0)),
        "model_path": str(MODEL_PATH),
        "model_sha256": MODEL_SHA256,
        "calculator": CALCULATOR_CONFIG,
        "inputs": {
            system: {
                "path": str(spec["path"]),
                "sha256": spec["sha256"],
            }
            for system, spec in SYSTEMS.items()
        },
    }


def _state(system: str):
    from ase.io import read
    from pamssw.state import State

    atoms = read(SYSTEMS[system]["path"])
    atoms.pbc = SYSTEMS[system]["pbc"]
    if system == "c60":
        fixed_mask = np.zeros(len(atoms), dtype=bool)
    else:
        z = np.asarray(atoms.positions, dtype=float)[:, 2]
        fixed_mask = z <= float(np.quantile(z, 0.35))
    return State(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=np.asarray(atoms.cell.array, dtype=float),
        pbc=tuple(bool(value) for value in atoms.pbc),
        fixed_mask=fixed_mask,
    )


def _calculator(mode: str):
    from mace.calculators import MACECalculator
    from pamssw.calculators import ASECalculator
    from pamssw.mace_batch import MACEBatchCalculator

    mace_calculator = MACECalculator(
        model_paths=str(MODEL_PATH),
        **CALCULATOR_CONFIG,
    )
    if mode == "serial":
        return ASECalculator(mace_calculator)
    return MACEBatchCalculator(mace_calculator)


def _choice(system: str, context: str, mode: str, backend):
    import torch

    from pamssw.accounting import EvalCounter, EvaluationPurpose
    from pamssw.walker import ProposalPotential, SoftModeOracle

    state = _state(system)
    counter = EvalCounter(backend)
    seed = int(SYSTEMS[system]["seed"])
    if context == "momentum":
        seed += 100
    oracle = SoftModeOracle(
        counter,
        np.random.default_rng(seed),
        candidates=4,
        n_bond_pairs=2,
        enable_momentum_candidate=True,
        hvp_epsilon=1.0e-3,
    )
    anchor = oracle.generator.generate_initial_direction(
        state,
        step_index=0,
        max_steps=200,
        lambda_start=0.7,
        lambda_end=0.3,
        n_bond_pairs=2,
        bond_distance_threshold=None,
    )
    previous = None
    if context == "momentum":
        previous = oracle.generator.generate_initial_direction(
            state,
            step_index=1,
            max_steps=200,
            lambda_start=0.7,
            lambda_end=0.3,
            n_bond_pairs=2,
            bond_distance_threshold=None,
        )
    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        torch.cuda.synchronize()
        started = perf_counter()
        choice = oracle.choose_direction(
            state,
            ProposalPotential(counter),
            previous_direction=previous,
            anchor_direction=anchor,
        )
        torch.cuda.synchronize()
        wall_time_s = perf_counter() - started
    counts = counter.snapshot()
    return {
        "wall_time_s": wall_time_s,
        "force_evaluations": counts.total,
        "unattributed": counts.count(EvaluationPurpose.UNATTRIBUTED),
        "selected_kind": choice.kind.value,
        "direction": np.asarray(choice.direction, dtype=float),
        "curvature": float(choice.curvature),
        "score": float(choice.score),
        "candidate_count": int(choice.candidate_count),
        "candidate_kind_counts": choice.diagnostics[
            "evaluated_candidate_kind_counts"
        ],
    }


def run(*, output_dir: Path, expected_commit: str) -> dict[str, object]:
    preflight = _preflight(expected_commit)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows = []
    total_force_evaluations = 0
    for system in SYSTEMS:
        backends = {mode: _calculator(mode) for mode in MODES}
        for context in CONTEXTS:
            for mode in MODES:
                warmup = _choice(system, context, mode, backends[mode])
                total_force_evaluations += int(warmup["force_evaluations"])
            for repetition in range(REPETITIONS):
                order = MODES if repetition % 2 == 0 else tuple(reversed(MODES))
                paired = {}
                for mode in order:
                    result = _choice(system, context, mode, backends[mode])
                    total_force_evaluations += int(result["force_evaluations"])
                    paired[mode] = result
                serial_direction = paired["serial"]["direction"]
                for mode in MODES:
                    result = paired[mode]
                    direction = result.pop("direction")
                    rows.append(
                        {
                            "system": system,
                            "context": context,
                            "repetition": repetition,
                            "mode": mode,
                            **result,
                            "direction_cosine_to_serial": float(
                                np.dot(direction, serial_direction)
                            ),
                        }
                    )
    if total_force_evaluations != EXPECTED_FORCE_EVALUATIONS:
        raise RuntimeError("integration-gate force ledger does not close")
    gate = protocol.evaluate_gate(rows)
    summary = {
        "schema_version": 1,
        **preflight,
        "protocol": {
            "systems": list(SYSTEMS),
            "contexts": list(CONTEXTS),
            "modes": list(MODES),
            "repetitions": REPETITIONS,
            "warmup_comparisons_per_stratum": 1,
            "candidate_count": 4,
            "central_difference_epsilon_A": 1.0e-3,
            "physical_force_evaluations_per_choice": 8,
        },
        "evaluation_counts": {
            "direction_oracle": total_force_evaluations,
            "unattributed": 0,
            "total": total_force_evaluations,
        },
        "gate": gate,
        "row_count": len(rows),
        "rows_path": "rows.json",
    }
    (output_dir / "rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
    print(
        json.dumps(
            run(
                output_dir=args.output.resolve(),
                expected_commit=args.expected_commit,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
