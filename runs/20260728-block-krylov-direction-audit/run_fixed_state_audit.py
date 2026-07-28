#!/usr/bin/env python3
"""Preregistered analytic and stored-state block-Krylov direction audit.

This script only constructs and diagnoses directions.  It never enters the
SSW proposal/relaxation loop and must not be used to make terminal-energy
claims.  The full mode reconstructs the absent starter-quench minimum from the
frozen production input/configuration, then audits it alongside two locked
accepted states per system.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from time import perf_counter
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pamssw.accounting import EvalCounter, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.krylov import IntentBlock
from pamssw.rigid import project_out_rigid_body_modes
from pamssw.state import State
from pamssw.walker import CandidateDirectionGenerator, ProposalPotential, SoftModeOracle, SurfaceWalker


ALLOCATIONS = {
    "variational_breadth": {"block_krylov_blocks": 6, "block_krylov_depth": 1},
    "shallow_refinement": {"block_krylov_blocks": 3, "block_krylov_depth": 2},
    "balanced_refinement": {"block_krylov_blocks": 2, "block_krylov_depth": 3},
    "deep_refinement": {"block_krylov_blocks": 1, "block_krylov_depth": 6},
}
HVP_EPSILON = 1.0e-3
MAX_HVPS = 12
OPERATOR = "total_proposal_central_fd"
MODEL_PATH = Path("/root/.cache/mace/mace-omat-0-small.model")
FROZEN_STRICT_QUENCH_WRAPPER = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-strict-quench-200-production" / "run_production.py"
)
FROZEN_BASE_PRODUCTION_RUNNER = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
PRODUCTION_EXECUTION_COMMIT = "32980ad9154ec6481b310477dd7b85597cefd49a"
MODEL_SHA256 = "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5"
STRICT_WRAPPER_SHA256 = "a6fef73c96f000a38e9233e0bd8a77899149c75dd29159ee8003ced800c1ff8f"
BASE_RUNNER_SHA256 = "d44588e06b56c1a996525c12518998d56303dc9a360e4e740662e098cd5081aa"
STRICT_OUTPUT_CONFIG_FIELDS = frozenset(
    {
        "accepted_structures_dir",
        "accepted_structures_log",
        "direction_diagnostics_path",
    }
)
CURRENT_CONFIG_SCHEMA_ADDITIONS = {
    "block_krylov_blocks": 2,
    "block_krylov_depth": 3,
}

# This registry is deliberately literal and is read before any new arm is run.
# The starter minimum was not persisted by the 200-production run and is thus
# reconstructed by the locked recipe instead of being mislabeled as trial 1.
FIXED_STATE_REGISTRY: dict[str, tuple[dict[str, Any], ...]] = {
    "c60": (
        {
            "state_id": "bootstrap_quenched",
            "phase": "bootstrap",
            "source": "reconstruct_frozen_starter_true_quench",
            "origin_run": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42",
            "origin_summary": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/summary.json",
            "origin_summary_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/summary.json",
            "origin_summary_sha256": "40d1f0d1c1cf81c374df6107fc4c31316dc313dd2b783cb503250ba931bf78cb",
            "origin_execution_commit": PRODUCTION_EXECUTION_COMMIT,
            "origin_commit": PRODUCTION_EXECUTION_COMMIT,
            "raw_input_path": "/mnt/d/download/trae-research-code/ssw/runs/20260428-c60-mace-production/prerelaxed_c60.xyz",
            "raw_input_sha256": "c63788c18cbed305963213b47eabd9fdc4d06dac118da6a1a9e16621d5e32bf9",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "strict_wrapper_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/run_production.py",
            "strict_wrapper_sha256": STRICT_WRAPPER_SHA256,
            "base_runner_path": "runs/20260728-safe-lbfgs-200-production/run_production.py",
            "base_runner_sha256": BASE_RUNNER_SHA256,
            "strict_config_diff": {
                "quench_fallback_optimizer": [None, "ase-fire"],
                "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
            },
            "intent_seed": 71001,
            "selection_rationale": "Stored starter-quench structure is absent; reconstruct the exact frozen starter true-quench from raw input and frozen configuration.",
        },
        {
            "state_id": "intermediate_accepted",
            "phase": "intermediate",
            "source": "locked_accepted_structure",
            "origin_state_id": "trial0100_entry0072",
            "origin_run": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42",
            "origin_summary": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/summary.json",
            "origin_summary_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/summary.json",
            "origin_summary_sha256": "40d1f0d1c1cf81c374df6107fc4c31316dc313dd2b783cb503250ba931bf78cb",
            "origin_execution_commit": PRODUCTION_EXECUTION_COMMIT,
            "origin_commit": PRODUCTION_EXECUTION_COMMIT,
            "structure_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/accepted_minima/trial0100_entry0072_accepted.xyz",
            "structure_sha256": "9bccc945de41f193ae7fd603d34114bb71ab3d270c5ad7bbe182f806a71f55f9",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "intent_seed": 71002,
            "selection_rationale": "Accepted trial 100 entry 72; preregistered midpoint of the 200-trial trajectory.",
        },
        {
            "state_id": "plateau_accepted",
            "phase": "plateau",
            "source": "locked_accepted_structure",
            "origin_state_id": "trial0180_entry0125",
            "origin_run": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42",
            "origin_summary": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/summary.json",
            "origin_summary_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/summary.json",
            "origin_summary_sha256": "40d1f0d1c1cf81c374df6107fc4c31316dc313dd2b783cb503250ba931bf78cb",
            "origin_execution_commit": PRODUCTION_EXECUTION_COMMIT,
            "origin_commit": PRODUCTION_EXECUTION_COMMIT,
            "structure_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-c60-seed42/accepted_minima/trial0180_entry0125_accepted.xyz",
            "structure_sha256": "7807d87f6274c22e3925c1b82a76c8e2e6880c5cfe83d346656a85fbe9886b37",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "intent_seed": 71003,
            "selection_rationale": "Accepted trial 180 entry 125; locked late-trajectory plateau-window state.",
        },
    ),
    "pdo": (
        {
            "state_id": "bootstrap_quenched",
            "phase": "bootstrap",
            "source": "reconstruct_frozen_starter_true_quench",
            "origin_run": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42",
            "origin_summary": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/summary.json",
            "origin_summary_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/summary.json",
            "origin_summary_sha256": "b34b45c0607e8b7a246ba398c30c68235be1d8e76af50f9d868bb0bce1da3c79",
            "origin_execution_commit": PRODUCTION_EXECUTION_COMMIT,
            "origin_commit": PRODUCTION_EXECUTION_COMMIT,
            "raw_input_path": "/mnt/d/download/trae-research-code/ssw/PdO.xyz",
            "raw_input_sha256": "68243ceb7c0fbb6ba7a9454d680287eb98c4e5210efbd9ebb63517ba79aaa8b0",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "strict_wrapper_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/run_production.py",
            "strict_wrapper_sha256": STRICT_WRAPPER_SHA256,
            "base_runner_path": "runs/20260728-safe-lbfgs-200-production/run_production.py",
            "base_runner_sha256": BASE_RUNNER_SHA256,
            "strict_config_diff": {
                "quench_fallback_optimizer": [None, "ase-fire"],
                "quench_fmax": [0.03, 0.01],
                "quench_optimizer": ["scipy-lbfgsb", "ase-lbfgs"],
            },
            "intent_seed": 72001,
            "selection_rationale": "Stored starter-quench structure is absent; reconstruct the exact frozen starter true-quench from raw input and frozen configuration.",
        },
        {
            "state_id": "intermediate_accepted",
            "phase": "intermediate",
            "source": "locked_accepted_structure",
            "origin_state_id": "trial0100_entry0082",
            "origin_run": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42",
            "origin_summary": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/summary.json",
            "origin_summary_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/summary.json",
            "origin_summary_sha256": "b34b45c0607e8b7a246ba398c30c68235be1d8e76af50f9d868bb0bce1da3c79",
            "origin_execution_commit": PRODUCTION_EXECUTION_COMMIT,
            "origin_commit": PRODUCTION_EXECUTION_COMMIT,
            "structure_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/accepted_minima/trial0100_entry0082_accepted.xyz",
            "structure_sha256": "d6bfadebb0d1ea9b9bac79b27fce1fdc4c02c624bbfdb9067a31591145a185a0",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "intent_seed": 72002,
            "selection_rationale": "Accepted trial 100 entry 82; preregistered midpoint of the 200-trial trajectory.",
        },
        {
            "state_id": "plateau_accepted",
            "phase": "plateau",
            "source": "locked_accepted_structure",
            "origin_state_id": "trial0180_entry0148",
            "origin_run": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42",
            "origin_summary": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/summary.json",
            "origin_summary_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/summary.json",
            "origin_summary_sha256": "b34b45c0607e8b7a246ba398c30c68235be1d8e76af50f9d868bb0bce1da3c79",
            "origin_execution_commit": PRODUCTION_EXECUTION_COMMIT,
            "origin_commit": PRODUCTION_EXECUTION_COMMIT,
            "structure_path": "runs/20260728-safe-lbfgs-strict-quench-200-production/output-pdo-seed42/accepted_minima/trial0180_entry0148_accepted.xyz",
            "structure_sha256": "58fc3053798920657ce6eb566b869bd5d7dc8324643d5bd00931a929c873d3a9",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "intent_seed": 72003,
            "selection_rationale": "Accepted trial 180 entry 148; locked late-trajectory plateau-window state.",
        },
    ),
}


class QuadraticPotential:
    """Analytic quadratic used only to exercise the real central-FD oracle."""

    def __init__(self, hessian: np.ndarray) -> None:
        hessian = np.asarray(hessian, dtype=float)
        if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
            raise ValueError("quadratic Hessian must be square")
        self.hessian = hessian

    def energy_gradient(self, flat_positions: np.ndarray, state: State) -> tuple[float, np.ndarray]:
        values = np.asarray(flat_positions, dtype=float)
        gradient = self.hessian @ values
        return float(0.5 * values @ gradient), gradient


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: State) -> str:
    digest = sha256()
    for values in (
        np.asarray(state.numbers, dtype="<i8"),
        np.asarray(state.positions, dtype="<f8"),
        np.asarray(state.fixed_mask, dtype=np.uint8),
        np.asarray(state.cell if state.cell is not None else np.zeros((3, 3)), dtype="<f8"),
        np.asarray(state.pbc, dtype=np.uint8),
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_module(path: Path, module_name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_frozen_runtime() -> tuple[ModuleType, ModuleType]:
    """Load the strict wrapper first, then the exact base runner it delegates to."""

    strict_wrapper = _load_module(
        FROZEN_STRICT_QUENCH_WRAPPER,
        "_block_krylov_frozen_strict_quench_wrapper",
    )
    loader = getattr(strict_wrapper, "_base_runner", None)
    if not callable(loader):
        raise RuntimeError("strict quench wrapper does not expose its frozen base loader")
    base_runner = loader()
    if not isinstance(base_runner, ModuleType):
        raise RuntimeError("strict quench wrapper returned an invalid base runner")
    if not FROZEN_BASE_PRODUCTION_RUNNER.is_file():
        raise FileNotFoundError(FROZEN_BASE_PRODUCTION_RUNNER)
    if _sha256(FROZEN_STRICT_QUENCH_WRAPPER) != STRICT_WRAPPER_SHA256:
        raise RuntimeError("strict quench wrapper checksum drifted")
    if _sha256(FROZEN_BASE_PRODUCTION_RUNNER) != BASE_RUNNER_SHA256:
        raise RuntimeError("frozen base runner checksum drifted")
    return strict_wrapper, base_runner


def _historical_config_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in STRICT_OUTPUT_CONFIG_FIELDS and key not in CURRENT_CONFIG_SCHEMA_ADDITIONS
    }


def _strict_bootstrap_config(
    *,
    system: str,
    case_dir: Path,
    strict_wrapper: ModuleType,
    base_runner: ModuleType,
    origin_summary: Mapping[str, Any],
) -> tuple[object, dict[str, Any], dict[str, list[Any]]]:
    """Build and verify the wrapper's exact strict true-quench configuration."""

    config_projection = getattr(strict_wrapper, "config_projection", None)
    effective_config_fn = getattr(strict_wrapper, "_effective_config", None)
    config_mapping_fn = getattr(strict_wrapper, "_config_mapping", None)
    json_mapping_fn = getattr(strict_wrapper, "_json_mapping", None)
    if not all(callable(value) for value in (config_projection, effective_config_fn, config_mapping_fn, json_mapping_fn)):
        raise RuntimeError("strict quench wrapper lacks its configuration projection API")
    source, projected_effective, config_diff = config_projection(system, case_dir, base_runner)
    source_config = base_runner.build_config(system, case_dir)
    config = effective_config_fn(source_config)
    effective = json_mapping_fn(config_mapping_fn(config))
    if source != json_mapping_fn(config_mapping_fn(source_config)):
        raise RuntimeError("strict wrapper source config differs from base runner")
    if effective != projected_effective:
        raise RuntimeError("strict wrapper effective config does not close to its projection")
    summary_effective = origin_summary.get("effective_config")
    strict_summary = origin_summary.get("strict_quench_wrapper")
    if not isinstance(summary_effective, Mapping) or not isinstance(strict_summary, Mapping):
        raise RuntimeError("strict origin summary lacks effective configuration provenance")
    summary_wrapper_effective = strict_summary.get("effective_config")
    summary_diff = strict_summary.get("config_diff")
    if not isinstance(summary_wrapper_effective, Mapping) or not isinstance(summary_diff, Mapping):
        raise RuntimeError("strict origin wrapper provenance is incomplete")
    additions = {key: effective.get(key) for key in CURRENT_CONFIG_SCHEMA_ADDITIONS}
    if additions != CURRENT_CONFIG_SCHEMA_ADDITIONS:
        raise RuntimeError("current config schema additions drifted from their frozen defaults")
    if any(key in summary_effective for key in CURRENT_CONFIG_SCHEMA_ADDITIONS):
        raise RuntimeError("origin summary unexpectedly contains current config schema additions")
    if _historical_config_projection(effective) != _historical_config_projection(summary_effective):
        raise RuntimeError("strict bootstrap effective config drifted from the origin summary")
    if _historical_config_projection(effective) != _historical_config_projection(summary_wrapper_effective):
        raise RuntimeError("strict bootstrap effective config drifted from wrapper provenance")
    if config_diff != summary_diff:
        raise RuntimeError("strict bootstrap config diff drifted from origin provenance")
    expected_protocol = {
        "quench_optimizer": "ase-lbfgs",
        "quench_fallback_optimizer": "ase-fire",
        "quench_fmax": 0.01,
    }
    if {key: effective.get(key) for key in expected_protocol} != expected_protocol:
        raise RuntimeError("strict bootstrap config does not retain the ASE strict-quench protocol")
    return config, dict(effective), dict(config_diff)


def _validate_origin(system: str, registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if system not in FIXED_STATE_REGISTRY:
        raise ValueError(f"unknown fixed-state system {system!r}")
    expected_summary = registry[0]["origin_summary_path"]
    if any(entry["origin_summary_path"] != expected_summary for entry in registry):
        raise RuntimeError(f"{system} fixed-state registry has multiple origin summaries")
    summary_path = REPO_ROOT / str(expected_summary)
    if not summary_path.is_file():
        raise FileNotFoundError(f"fixed-state origin summary is absent: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if _sha256(summary_path) != registry[0]["origin_summary_sha256"]:
        raise RuntimeError(f"{system} origin summary checksum drifted")
    if summary.get("execution_commit") != PRODUCTION_EXECUTION_COMMIT:
        raise RuntimeError(f"{system} origin execution commit drifted")
    if summary.get("model_sha256") != MODEL_SHA256:
        raise RuntimeError(f"{system} origin model checksum drifted")
    if summary.get("model_path") != str(MODEL_PATH):
        raise RuntimeError(f"{system} origin model path drifted")
    strict_summary = summary.get("strict_quench_wrapper")
    if not isinstance(strict_summary, Mapping):
        raise RuntimeError(f"{system} origin strict-quench wrapper provenance is absent")
    base_runner = strict_summary.get("base_runner")
    if not isinstance(base_runner, Mapping) or base_runner.get("sha256") != BASE_RUNNER_SHA256:
        raise RuntimeError(f"{system} origin base-runner provenance drifted")
    for entry in registry:
        if entry.get("origin_execution_commit") != PRODUCTION_EXECUTION_COMMIT:
            raise RuntimeError(f"{system} registry origin commit drifted")
        if entry.get("origin_commit") != PRODUCTION_EXECUTION_COMMIT:
            raise RuntimeError(f"{system} registry origin commit alias drifted")
        if entry.get("origin_summary_sha256") != _sha256(summary_path):
            raise RuntimeError(f"{system} registry origin summary checksum drifted")
        if entry.get("model_path") != str(MODEL_PATH):
            raise RuntimeError(f"{system} registry model path drifted")
        if entry.get("model_sha256") != MODEL_SHA256:
            raise RuntimeError(f"{system} registry model checksum drifted")
        if entry["source"] == "reconstruct_frozen_starter_true_quench":
            if entry.get("raw_input_path") != summary.get("input_path"):
                raise RuntimeError(f"{system} bootstrap raw input path drifted")
            if entry.get("raw_input_sha256") != summary.get("input_sha256"):
                raise RuntimeError(f"{system} bootstrap raw input checksum drifted")
            if entry.get("strict_wrapper_sha256") != STRICT_WRAPPER_SHA256:
                raise RuntimeError(f"{system} strict wrapper checksum provenance drifted")
            if entry.get("base_runner_sha256") != BASE_RUNNER_SHA256:
                raise RuntimeError(f"{system} base runner checksum provenance drifted")
            if entry.get("strict_config_diff") != strict_summary.get("config_diff"):
                raise RuntimeError(f"{system} strict config diff provenance drifted")
        if entry["source"] != "locked_accepted_structure":
            continue
        path = REPO_ROOT / str(entry["structure_path"])
        if not path.is_file():
            raise FileNotFoundError(f"locked fixed state is absent: {path}")
        if _sha256(path) != entry["structure_sha256"]:
            raise RuntimeError(f"locked fixed state checksum drifted: {path}")
    return summary


def _analytic_state() -> State:
    return State(
        # Five periodic atoms provide ten distinct local pairs and twelve
        # non-translational coordinates, enough for every non-rank-one 12-HVP
        # allocation to consume its full preregistered budget.
        numbers=np.array([6, 6, 6, 6, 6]),
        # Keep every atom away from a periodic boundary: central-FD probes are
        # represented by wrapped Cartesian states, so a boundary atom would
        # turn a tiny negative displacement into an artificial cell-length jump.
        positions=np.array(
            [
                [10.0, 10.0, 10.0],
                [11.1, 10.2, 10.1],
                [10.4, 11.0, 10.7],
                [11.7, 11.4, 10.5],
                [10.8, 10.6, 11.8],
            ]
        ),
        cell=np.eye(3) * 20.0,
        pbc=(True, True, True),
        fixed_mask=np.zeros(5, dtype=bool),
        metadata={"system": "analytic_periodic_five_atom"},
    )


def _projected_basis(state: State) -> np.ndarray:
    identity = np.eye(state.positions.size)
    projected = np.column_stack(
        [project_out_rigid_body_modes(state, identity[:, index]) for index in range(identity.shape[1])]
    )
    vectors, singular_values, _ = np.linalg.svd(projected, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > 1e-10))
    if rank < 2:
        raise RuntimeError("analytic state did not retain a usable projected subspace")
    return vectors[:, :rank]


def _quadratic_hessian(state: State, reduced: np.ndarray) -> np.ndarray:
    basis = _projected_basis(state)
    if reduced.shape != (basis.shape[1], basis.shape[1]):
        raise ValueError("reduced analytic Hessian has the wrong projected dimension")
    complement = np.eye(basis.shape[0]) - basis @ basis.T
    return basis @ reduced @ basis.T + 50.0 * complement


def _analytic_cases() -> list[dict[str, Any]]:
    state = _analytic_state()
    dimension = _projected_basis(state).shape[1]
    diagonal = np.diag(np.concatenate(([0.5], np.arange(1.0, float(dimension)))))
    coupled = np.diag(np.arange(4.0, 4.0 + dimension))
    coupled[0, 1] = coupled[1, 0] = -2.0
    near_degenerate_values = np.arange(2.0, 2.0 + dimension)
    near_degenerate_values[:2] = (0.5, 0.5000001)
    near_degenerate = np.diag(near_degenerate_values)
    rank_one = np.diag(np.arange(1.0, 1.0 + dimension))
    return [
        {
            "case": "diagonal_known_eigensystem",
            "state": state,
            "potential": QuadraticPotential(_quadratic_hessian(state, diagonal)),
            "intent_seed": 61001,
            "metadata": {"known_lowest_projected_eigenvalue": 0.5, "operator_shape": "diagonal"},
        },
        {
            "case": "coupled_quadratic_requires_expansion",
            "state": state,
            "potential": QuadraticPotential(_quadratic_hessian(state, coupled)),
            "intent_seed": 61002,
            "metadata": {"requires_krylov_expansion": True, "operator_shape": "coupled"},
        },
        {
            "case": "near_degenerate_lowest_eigenvalue",
            "state": state,
            "potential": QuadraticPotential(_quadratic_hessian(state, near_degenerate)),
            "intent_seed": 61003,
            "metadata": {"lowest_eigenvalue_gap": 1.0e-7, "operator_shape": "near_degenerate"},
        },
        {
            "case": "rank_one_initial_block",
            "state": state,
            "potential": QuadraticPotential(_quadratic_hessian(state, rank_one)),
            "intent_seed": 61004,
            "force_rank_one": True,
            "metadata": {"rank_one_initial_block": True, "operator_shape": "diagonal"},
        },
    ]


def _direction_row(
    *,
    case: str,
    kind: str,
    system: str,
    state_id: str,
    state: State,
    state_provenance: Mapping[str, Any],
    calculator_factory: Callable[[], object],
    calculator_provenance: Mapping[str, Any],
    intent_seed: int,
    arm: str,
    force_rank_one: bool,
    case_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    allocation = ALLOCATIONS[arm]
    counter = EvalCounter(calculator_factory(), max_force_evals=2 * MAX_HVPS)
    proposal = ProposalPotential(counter)
    rng = np.random.default_rng(intent_seed)
    oracle = SoftModeOracle(
        counter,
        rng,
        candidates=2,
        hvp_epsilon=HVP_EPSILON,
        direction_selection_mode="block_krylov",
        block_krylov_depth=allocation["block_krylov_depth"],
        direction_synthesis_mode="none",
    )
    generator = oracle.generator
    if not isinstance(generator, CandidateDirectionGenerator):
        raise RuntimeError("fixed-state audit did not construct CandidateDirectionGenerator")
    intents = generator.generate_krylov_intents(state, n_blocks=allocation["block_krylov_blocks"])
    if force_rank_one:
        intents = tuple(IntentBlock(intent.basis[:, :1], pair=None) for intent in intents)
    initial_ranks = [int(intent.basis.shape[1]) for intent in intents]
    started = perf_counter()
    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        choice = oracle.choose_direction(
            state,
            proposal,
            previous_direction=None,
            krylov_intents=intents,
        )
    wall_seconds = float(perf_counter() - started)
    counts = counter.snapshot()
    purpose_counts = counts.as_dict()
    diagnostics = dict(choice.diagnostics)
    hvp_count = diagnostics.get("krylov_hvp_count")
    if isinstance(hvp_count, bool) or not isinstance(hvp_count, int):
        raise RuntimeError("block oracle omitted an exact HVP count")
    if counter.force_evaluations != 2 * hvp_count:
        raise RuntimeError("central-FD force accounting does not close")
    if hvp_count > MAX_HVPS:
        raise RuntimeError("block oracle exceeded the HVP budget")
    if purpose_counts.get(EvaluationPurpose.DIRECTION_ORACLE.value) != counter.force_evaluations:
        raise RuntimeError("direction audit has evaluations outside direction_oracle")
    if purpose_counts.get(EvaluationPurpose.UNATTRIBUTED.value) != 0:
        raise RuntimeError("direction audit has unattributed evaluations")
    if counts.total != counter.force_evaluations:
        raise RuntimeError("EvalCounter snapshot does not close")
    required_diagnostics = {
        "krylov_blocks",
        "krylov_selected_block",
        "krylov_hvp_count",
        "krylov_dimensions",
        "krylov_initial_ranks",
        "krylov_residual_norm",
        "krylov_initial_span_overlap",
        "krylov_antisymmetry",
        "krylov_termination",
        "direction_participation_ratio",
    }
    missing = required_diagnostics - diagnostics.keys()
    if missing:
        raise RuntimeError(f"block oracle diagnostics are incomplete: {sorted(missing)}")
    return (
        {
            "case": case,
            "kind": kind,
            "system": system,
            "state_id": state_id,
            "arm": arm,
            "intent_seed": int(intent_seed),
            "intent_initial_ranks": initial_ranks,
            "force_evaluations": int(counter.force_evaluations),
            "krylov_hvp_count": int(hvp_count),
            "purpose_counts": purpose_counts,
            "diagnostics": diagnostics,
            "selection": {
                "kind": choice.kind.value,
                "curvature": float(choice.curvature),
                "true_curvature": float(choice.true_curvature),
                "direction_sha256": sha256(np.asarray(choice.direction, dtype="<f8").tobytes()).hexdigest(),
            },
            "state_provenance": dict(state_provenance),
            "state_sha256": _state_sha256(state),
            "calculator": dict(calculator_provenance),
            "case_metadata": dict(case_metadata),
            "wall_seconds": wall_seconds,
        },
        np.asarray(choice.direction, dtype=float).copy(),
    )


def _run_case_allocations(
    *,
    case: str,
    kind: str,
    system: str,
    state_id: str,
    state: State,
    state_provenance: Mapping[str, Any],
    calculator_factory: Callable[[], object],
    calculator_provenance: Mapping[str, Any],
    intent_seed: int,
    force_rank_one: bool = False,
    case_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directions: dict[str, np.ndarray] = {}
    for arm in ALLOCATIONS:
        row, direction = _direction_row(
            case=case,
            kind=kind,
            system=system,
            state_id=state_id,
            state=state,
            state_provenance=state_provenance,
            calculator_factory=calculator_factory,
            calculator_provenance=calculator_provenance,
            intent_seed=intent_seed,
            arm=arm,
            force_rank_one=force_rank_one,
            case_metadata=case_metadata or {},
        )
        rows.append(row)
        directions[arm] = direction
    reference = directions["variational_breadth"]
    for row in rows:
        overlap = abs(float(np.dot(reference, directions[row["arm"]])))
        row["subspace_angle_to_variational_breadth_degrees"] = float(
            np.degrees(np.arccos(np.clip(overlap, -1.0, 1.0)))
        )
    return rows


def _load_locked_state(entry: Mapping[str, Any], template: State) -> tuple[State, dict[str, Any]]:
    from ase.io import read

    path = REPO_ROOT / str(entry["structure_path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if _sha256(path) != entry["structure_sha256"]:
        raise RuntimeError(f"locked state checksum drifted: {path}")
    atoms = read(path)
    numbers = np.asarray(atoms.numbers, dtype=int)
    positions = np.asarray(atoms.positions, dtype=float)
    if not np.array_equal(numbers, template.numbers):
        raise RuntimeError(f"locked state composition differs from frozen {entry['state_id']}")
    if positions.shape != template.positions.shape:
        raise RuntimeError(f"locked state coordinate shape differs from frozen {entry['state_id']}")
    if np.any(template.fixed_mask) and not np.allclose(
        positions[template.fixed_mask], template.positions[template.fixed_mask], atol=1e-8, rtol=0.0
    ):
        raise RuntimeError(f"locked state moved frozen atoms: {entry['state_id']}")
    state = State(
        numbers=numbers,
        positions=positions,
        cell=None if template.cell is None else template.cell.copy(),
        pbc=template.pbc,
        fixed_mask=template.fixed_mask.copy(),
        metadata={"system": entry["origin_run"], "state_id": entry["state_id"]},
    )
    provenance = dict(entry)
    provenance["structure_runtime_path"] = str(path)
    provenance["structure_sha256_verified"] = _sha256(path)
    provenance["state_sha256"] = _state_sha256(state)
    return state, provenance


def _reconstruct_bootstrap(
    *,
    system: str,
    entry: Mapping[str, Any],
    strict_wrapper: ModuleType,
    base_runner: ModuleType,
    origin_summary: Mapping[str, Any],
) -> tuple[State, object, dict[str, Any], State]:
    template = base_runner.load_state(system)
    with tempfile.TemporaryDirectory(prefix=f"block-krylov-{system}-bootstrap-") as temporary_root:
        config, effective_config, config_diff = _strict_bootstrap_config(
            system=system,
            case_dir=Path(temporary_root) / "case",
            strict_wrapper=strict_wrapper,
            base_runner=base_runner,
            origin_summary=origin_summary,
        )
        if float(config.hvp_epsilon) != HVP_EPSILON:
            raise RuntimeError("frozen bootstrap config HVP epsilon drifted")
        # Calling the frozen constructor preserves the model path, CUDA device,
        # precision, and cueq contract of the origin production runner.
        from pamssw.calculators import ASECalculator

        walker = SurfaceWalker(
            calculator=ASECalculator(base_runner._calculator()),
            config=config,
            softening_enabled=True,
        )
        started = perf_counter()
        result = walker.relax_true_minimum(
            template,
            trajectory_name="block_krylov_bootstrap_reconstruction",
            quench_purpose=EvaluationPurpose.STARTER_TRUE_QUENCH,
        )
        wall_seconds = float(perf_counter() - started)
        counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    if purpose_counts.get(EvaluationPurpose.STARTER_TRUE_QUENCH.value, 0) <= 0:
        raise RuntimeError("bootstrap reconstruction did not use STARTER_TRUE_QUENCH")
    if purpose_counts.get(EvaluationPurpose.UNATTRIBUTED.value) != 0:
        raise RuntimeError("bootstrap reconstruction has unattributed evaluations")
    if counts.total != walker.calculator.force_evaluations:
        raise RuntimeError("bootstrap reconstruction counter does not close")
    provenance = dict(entry)
    provenance.update(
        {
            "raw_input_path": origin_summary["input_path"],
            "raw_input_sha256": origin_summary["input_sha256"],
            "strict_wrapper_runtime_path": str(FROZEN_STRICT_QUENCH_WRAPPER),
            "strict_wrapper_path": entry["strict_wrapper_path"],
            "strict_wrapper_sha256": _sha256(FROZEN_STRICT_QUENCH_WRAPPER),
            "base_runner_runtime_path": str(FROZEN_BASE_PRODUCTION_RUNNER),
            "base_runner_path": entry["base_runner_path"],
            "base_runner_sha256": _sha256(FROZEN_BASE_PRODUCTION_RUNNER),
            "effective_config": effective_config,
            "origin_effective_config": dict(origin_summary["effective_config"]),
            "effective_config_diff": config_diff,
            "current_config_schema_additions": dict(CURRENT_CONFIG_SCHEMA_ADDITIONS),
            "bootstrap_purpose_counts": purpose_counts,
            "bootstrap_force_evaluations": int(walker.calculator.force_evaluations),
            "bootstrap_wall_seconds": wall_seconds,
            "reconstructed_state_sha256": _state_sha256(result.state),
            "resulting_state_sha256": _state_sha256(result.state),
            "state_sha256": _state_sha256(result.state),
            "reconstructed_energy_eV": float(result.energy),
        }
    )
    # Deliberately return the counter-free underlying calculator.  Every arm
    # receives a fresh EvalCounter, so bootstrap evaluations cannot leak into
    # the direction-only ledger.
    return result.state, walker.calculator.calculator, provenance, template


def _run_analytic() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in _analytic_cases():
        potential = case["potential"]
        rows.extend(
            _run_case_allocations(
                case=case["case"],
                kind="analytic",
                system="analytic",
                state_id=case["case"],
                state=case["state"],
                state_provenance={"origin": "analytic_quadratic", "state_id": case["case"]},
                calculator_factory=lambda potential=potential: AnalyticCalculator(potential),
                calculator_provenance={"kind": "analytic_quadratic", "device": "cpu", "precision": "float64"},
                intent_seed=case["intent_seed"],
                force_rank_one=bool(case.get("force_rank_one", False)),
                case_metadata=case["metadata"],
            )
        )
    return rows


def _run_fixed_states() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strict_wrapper, base_runner = _load_frozen_runtime()
    for system, registry in FIXED_STATE_REGISTRY.items():
        summary = _validate_origin(system, registry)
        bootstrap_entry = registry[0]
        if bootstrap_entry["source"] != "reconstruct_frozen_starter_true_quench":
            raise RuntimeError("fixed-state registry must begin with bootstrap reconstruction")
        bootstrap, shared_calculator, bootstrap_provenance, template = _reconstruct_bootstrap(
            system=system,
            entry=bootstrap_entry,
            strict_wrapper=strict_wrapper,
            base_runner=base_runner,
            origin_summary=summary,
        )
        calculator_provenance = {
            "kind": "mace_omat_0_small",
            "model_path": str(MODEL_PATH),
            "model_sha256": MODEL_SHA256,
            "device": "cuda",
            "precision": "float32",
            "dtype": "float32",
        }
        rows.extend(
            _run_case_allocations(
                case=f"{system}:{bootstrap_entry['state_id']}",
                kind="fixed_state",
                system=system,
                state_id=bootstrap_entry["state_id"],
                state=bootstrap,
                state_provenance=bootstrap_provenance,
                calculator_factory=lambda calculator=shared_calculator: calculator,
                calculator_provenance=calculator_provenance,
                intent_seed=bootstrap_entry["intent_seed"],
                case_metadata={"phase": "bootstrap"},
            )
        )
        for entry in registry[1:]:
            state, provenance = _load_locked_state(entry, template)
            rows.extend(
                _run_case_allocations(
                    case=f"{system}:{entry['state_id']}",
                    kind="fixed_state",
                    system=system,
                    state_id=entry["state_id"],
                    state=state,
                    state_provenance=provenance,
                    calculator_factory=lambda calculator=shared_calculator: calculator,
                    calculator_provenance=calculator_provenance,
                    intent_seed=entry["intent_seed"],
                    case_metadata={"phase": entry["phase"]},
                )
            )
    return rows


def _runtime_metadata(analytic_only: bool) -> dict[str, Any]:
    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "mode": "analytic_only" if analytic_only else "full_fixed_state",
        "device": "cpu" if analytic_only else "cuda",
        "precision": "float64" if analytic_only else "float32",
    }
    if not analytic_only:
        metadata.update(
            {
                "dtype": "float32",
                "model_path": str(MODEL_PATH),
                "model_sha256": MODEL_SHA256,
            }
        )
    return metadata


def run(*, analytic_only: bool) -> dict[str, Any]:
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked worktree must be clean before the audit starts")
    started = perf_counter()
    rows = _run_analytic()
    if not analytic_only:
        rows.extend(_run_fixed_states())
    payload = {
        "schema_version": 1,
        "git_commit": _current_commit(),
        "dirty": False,
        "operator": OPERATOR,
        "hvp_epsilon": HVP_EPSILON,
        "max_hvps": MAX_HVPS,
        "allocations": ALLOCATIONS,
        "runtime": _runtime_metadata(analytic_only),
        "fixed_state_registry": FIXED_STATE_REGISTRY,
        "rows": rows,
        "wall_seconds": float(perf_counter() - started),
    }
    json.dumps(payload, allow_nan=False)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit output: {args.output}")
    payload = run(analytic_only=bool(args.analytic_only))
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
