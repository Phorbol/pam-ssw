#!/usr/bin/env python3
"""Fixed-starter 2x2 audit of LS direction and proposal-landscape effects."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from pamssw.accounting import EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.calculators import ASECalculator
from pamssw.fingerprint import descriptor_distance, structural_descriptor
from pamssw.io import read_state, write_state
from pamssw.relax import has_force_convergence_certificate
from pamssw.walker import SurfaceWalker


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
CONFIG_RUNNER = (
    REPO_ROOT / "runs" / "20260730-starter-cell-online-gate" / "run_gate.py"
)
PRODUCTION_RUNNER = (
    REPO_ROOT / "runs" / "20260728-safe-lbfgs-200-production" / "run_production.py"
)
SOURCE_ROOT = (
    REPO_ROOT
    / "runs"
    / "20260730-starter-cell-online-gate"
    / "production-20k-seed42-output"
)
SYSTEMS = ("c60", "pdo")
SCOPES = ("none", "oracle", "proposal", "both")
SEEDS = (42, 43, 44)
STATE_FILES = {
    "c60": {
        "bootstrap": SOURCE_ROOT / "c60/seed-00000042/uniform/archive_minima/entry-00000.xyz",
        "mid": SOURCE_ROOT / "c60/seed-00000042/uniform/archive_minima/entry-00020.xyz",
        "late": SOURCE_ROOT / "c60/seed-00000042/uniform/archive_minima/entry-00041.xyz",
    },
    "pdo": {
        "bootstrap": SOURCE_ROOT / "pdo/seed-00000042/uniform/archive_minima/entry-00000.xyz",
        "mid": SOURCE_ROOT / "pdo/seed-00000042/uniform/archive_minima/entry-00032.xyz",
        "late": SOURCE_ROOT / "pdo/seed-00000042/uniform/archive_minima/entry-00064.xyz",
    },
}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
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
            state.cell if state.cell is not None else np.zeros((3, 3)),
            dtype="<f8",
        ),
        np.asarray(state.pbc, dtype=np.uint8),
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def participation_ratio(direction: np.ndarray, movable_mask: np.ndarray) -> float:
    per_atom = np.linalg.norm(np.asarray(direction, dtype=float).reshape(-1, 3), axis=1)
    weights = np.square(per_atom[np.asarray(movable_mask, dtype=bool)])
    if weights.size == 0 or float(weights.sum()) <= 1.0e-30:
        return 0.0
    probabilities = weights / weights.sum()
    return float(1.0 / (weights.size * np.square(probabilities).sum()))


def factorial_effects(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, float]] = {}
    for row in rows:
        key = (str(row["system"]), str(row["state_id"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["scope"])] = float(row[metric])
    effects = []
    for (system, state_id, seed), values in sorted(grouped.items()):
        if set(values) != set(SCOPES):
            raise ValueError("each factorial block must contain exactly the four scopes")
        none = values["none"]
        oracle = values["oracle"]
        proposal = values["proposal"]
        both = values["both"]
        effects.append(
            {
                "system": system,
                "state_id": state_id,
                "seed": seed,
                "oracle_main_effect": 0.5 * ((oracle - none) + (both - proposal)),
                "proposal_main_effect": 0.5 * ((proposal - none) + (both - oracle)),
                "interaction": both - oracle - proposal + none,
            }
        )
    return effects


def _top_active_indices(direction: np.ndarray, movable_mask: np.ndarray, count: int) -> set[int]:
    magnitudes = np.linalg.norm(np.asarray(direction).reshape(-1, 3), axis=1)
    movable = np.flatnonzero(movable_mask)
    order = np.argsort(-magnitudes[movable], kind="stable")[:count]
    return {int(index) for index in movable[order]}


class ObservingWalker(SurfaceWalker):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mechanism_directions: list[dict[str, Any]] = []
        self.mechanism_softening: list[dict[str, Any]] = []

    def _record_direction_diagnostics(
        self,
        *,
        trial_index,
        proposal_index,
        step_index,
        choice,
        anchor_direction,
    ) -> None:
        direction = np.asarray(choice.direction, dtype=float)
        active_count = (
            self.config.local_softening_active_count
            or int(np.count_nonzero(self._current_movable_mask))
        )
        anchor_overlap = None
        anchor_cosine = self._direction_anchor_cosine(anchor_direction, direction)
        if anchor_direction is not None:
            anchor_active = _top_active_indices(
                anchor_direction,
                self._current_movable_mask,
                active_count,
            )
            chosen_active = _top_active_indices(
                direction,
                self._current_movable_mask,
                active_count,
            )
            anchor_overlap = len(anchor_active & chosen_active) / max(1, active_count)
        self.mechanism_directions.append(
            {
                "step": int(step_index),
                "kind": choice.kind.value,
                "direction_sha256": sha256(
                    np.asarray(direction, dtype="<f8").tobytes()
                ).hexdigest(),
                "curvature": float(choice.curvature),
                "true_curvature": (
                    None
                    if choice.true_curvature is None
                    else float(choice.true_curvature)
                ),
                "participation_ratio": participation_ratio(
                    direction,
                    self._current_movable_mask,
                ),
                "anchor_cosine": anchor_cosine,
                "active_set_overlap": anchor_overlap,
            }
        )

    def _build_softening(self, seed_state, direction=None):
        model = super()._build_softening(seed_state, direction)
        if model is not None:
            energy, gradient = model.evaluate(seed_state.flatten_positions())
            self.mechanism_softening.append(
                {
                    "energy_eV": float(energy),
                    "gradient_norm_eV_per_A": float(np.linalg.norm(gradient)),
                    "term_count": len(model.terms),
                }
            )
        return model

    def _walk_candidate_from_seed(self, seed_state, *args, **kwargs):
        self._current_movable_mask = np.asarray(seed_state.movable_mask, dtype=bool)
        return super()._walk_candidate_from_seed(seed_state, *args, **kwargs)


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


def _calculator():
    production = _load_module(PRODUCTION_RUNNER, "_ls_scope_production_runner")
    from mace.calculators import MACECalculator

    raw = MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )
    return ASECalculator(raw), production


def preflight(expected_git_commit: str) -> dict[str, Any]:
    actual = _current_commit()
    if actual != expected_git_commit:
        raise RuntimeError(f"execution commit mismatch: expected {expected_git_commit}, got {actual}")
    if not _tracked_clean():
        raise RuntimeError("tracked worktree is not clean")
    production = _load_module(PRODUCTION_RUNNER, "_ls_scope_preflight_production")
    if not production.MODEL_PATH.is_file():
        raise FileNotFoundError(production.MODEL_PATH)
    for paths in STATE_FILES.values():
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    return {
        "schema_version": 1,
        "execution_commit": actual,
        "model_path": str(production.MODEL_PATH),
        "model_sha256": _sha256(production.MODEL_PATH),
        "gpu": torch.cuda.get_device_name(0),
        "systems": list(SYSTEMS),
        "scopes": list(SCOPES),
        "seeds": list(SEEDS),
        "states": {
            system: {
                state_id: {"path": str(path), "sha256": _sha256(path)}
                for state_id, path in paths.items()
            }
            for system, paths in STATE_FILES.items()
        },
        "frozen_components": [
            "starter",
            "candidate RNG seed",
            "direction portfolio",
            "Gaussian bias updater",
            "proposal optimizer",
            "true quench",
            "structure matcher",
        ],
    }


def _run_case(
    *,
    system: str,
    state_id: str,
    state,
    seed: int,
    scope: str,
    calculator,
    config_builder,
    case_directory: Path,
) -> dict[str, Any]:
    config = replace(
        config_builder.build_production_config(
            system,
            case_directory,
            master_seed=seed,
        ),
        max_trials=1,
        max_force_evals=None,
        local_softening_scope=scope,
    )
    walker = ObservingWalker(
        calculator=calculator,
        config=config,
        softening_enabled=True,
    )
    archive = MinimaArchive(
        energy_tol=config.dedup_energy_tol,
        rmsd_tol=config.dedup_rmsd_tol,
        max_prototypes=config.max_prototypes,
    )
    started = perf_counter()
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        starter_evaluation = walker.calculator.evaluate(state)
    starter_energy = float(starter_evaluation.energy)
    seed_entry = archive.add(state, starter_energy, parent_id=None)
    proposal = walker._proposal_pool(
        state,
        archive,
        trial_index=0,
        step_target=walker.step_target_controller.target(archive),
        seed_entry_id=seed_entry.entry_id,
        allow_duplicate_rescue=False,
    )[0]
    escape_state = proposal.state
    with walker.calculator.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        escape_evaluation = walker.calculator.evaluate(escape_state)
    landing = walker.relax_true_minimum(escape_state)
    wall_time_s = float(perf_counter() - started)
    before = len(archive.entries)
    archive.add(landing.state, float(landing.energy), parent_id=seed_entry.entry_id)
    counts = walker.calculator.snapshot()
    purpose_counts = counts.as_dict()
    if counts.total != sum(purpose_counts.values()) or purpose_counts["unattributed"] != 0:
        raise RuntimeError("purpose accounting does not close")
    case_directory.mkdir(parents=True, exist_ok=True)
    write_state(case_directory / "escape.xyz", escape_state)
    write_state(case_directory / "landing.xyz", landing.state)
    row = {
        "system": system,
        "state_id": state_id,
        "state_sha256": _state_sha256(state),
        "seed": seed,
        "scope": scope,
        "starter_energy_eV": starter_energy,
        "escape_energy_eV": float(escape_evaluation.energy),
        "landing_energy_eV": float(landing.energy),
        "landing_delta_eV": float(landing.energy) - starter_energy,
        "is_new_basin": len(archive.entries) > before,
        "descriptor_delta": float(
            descriptor_distance(
                structural_descriptor(state),
                structural_descriptor(landing.state),
            )
        ),
        "fragmented": bool(walker._is_fragmented_cluster(state, landing.state)),
        "quench_certificate": bool(
            has_force_convergence_certificate(landing, config.quench_fmax)
        ),
        "force_evaluations": counts.total,
        "purpose_counts": purpose_counts,
        "wall_time_s": wall_time_s,
        "directions": walker.mechanism_directions,
        "softening_builds": walker.mechanism_softening,
        "effective_config": asdict(config),
    }
    _write_json(case_directory / "summary.json", row)
    return row


def build_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        (system, state_id, seed, scope)
        for system in SYSTEMS
        for state_id in STATE_FILES[system]
        for seed in SEEDS
        for scope in SCOPES
    }
    observed = {
        (str(row["system"]), str(row["state_id"]), int(row["seed"]), str(row["scope"]))
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("case cohort does not match the fixed 2x2 design")
    for row in rows:
        purposes = row["purpose_counts"]
        if (
            sum(int(value) for value in purposes.values()) != int(row["force_evaluations"])
            or int(purposes["unattributed"]) != 0
            or not row["directions"]
        ):
            raise ValueError("case accounting or direction record is incomplete")
        if row["scope"] == "none" and row["softening_builds"]:
            raise ValueError("none scope constructed a softening model")
        if row["scope"] != "none" and not row["softening_builds"]:
            raise ValueError("softened scope did not construct a softening model")

    paired_direction_checks = []
    by_key = {
        (row["system"], row["state_id"], row["seed"], row["scope"]): row
        for row in rows
    }
    for system in SYSTEMS:
        for state_id in STATE_FILES[system]:
            for seed in SEEDS:
                block = {
                    scope: by_key[(system, state_id, seed, scope)]
                    for scope in SCOPES
                }
                paired_direction_checks.append(
                    {
                        "system": system,
                        "state_id": state_id,
                        "seed": seed,
                        "none_equals_proposal_first_direction": (
                            block["none"]["directions"][0]["direction_sha256"]
                            == block["proposal"]["directions"][0]["direction_sha256"]
                        ),
                        "oracle_equals_both_first_direction": (
                            block["oracle"]["directions"][0]["direction_sha256"]
                            == block["both"]["directions"][0]["direction_sha256"]
                        ),
                    }
                )
    if not all(
        row["none_equals_proposal_first_direction"]
        and row["oracle_equals_both_first_direction"]
        for row in paired_direction_checks
    ):
        raise ValueError("first-step candidate pairing drifted")

    summaries = {}
    for system in SYSTEMS:
        summaries[system] = {}
        for scope in SCOPES:
            arm = [
                row
                for row in rows
                if row["system"] == system and row["scope"] == scope
            ]
            summaries[system][scope] = {
                "cases": len(arm),
                "new_basin_count": sum(bool(row["is_new_basin"]) for row in arm),
                "quench_certificate_count": sum(
                    bool(row["quench_certificate"]) for row in arm
                ),
                "median_landing_delta_eV": statistics.median(
                    float(row["landing_delta_eV"]) for row in arm
                ),
                "median_force_evaluations": statistics.median(
                    int(row["force_evaluations"]) for row in arm
                ),
                "median_first_direction_true_curvature": statistics.median(
                    float(row["directions"][0]["true_curvature"]) for row in arm
                ),
                "median_first_direction_inner_curvature": statistics.median(
                    float(row["directions"][0]["curvature"]) for row in arm
                ),
                "median_first_direction_participation_ratio": statistics.median(
                    float(row["directions"][0]["participation_ratio"]) for row in arm
                ),
            }
    return {
        "schema_version": 1,
        "cohort": {
            "systems": list(SYSTEMS),
            "states_per_system": 3,
            "seeds": list(SEEDS),
            "scopes": list(SCOPES),
            "cases": len(rows),
        },
        "scope_summaries": summaries,
        "landing_factorial_effects": factorial_effects(rows, "landing_delta_eV"),
        "force_evaluation_factorial_effects": factorial_effects(rows, "force_evaluations"),
        "paired_direction_checks": paired_direction_checks,
        "claim_ceiling": (
            "fixed-starter descriptive mechanism gate; no production default is "
            "changed without a subsequent equal-budget full-search comparison"
        ),
        "cases": list(rows),
    }


def run_gate(output: Path, expected_git_commit: str) -> dict[str, Any]:
    manifest = preflight(expected_git_commit)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _write_json(output / "manifest.json", manifest)
    calculator, _ = _calculator()
    config_builder = _load_module(CONFIG_RUNNER, "_ls_scope_config_builder")
    rows = []
    for system in SYSTEMS:
        for state_id, state_path in STATE_FILES[system].items():
            state = read_state(state_path)
            for seed in SEEDS:
                for scope in SCOPES:
                    case_directory = output / system / state_id / f"seed-{seed}" / scope
                    rows.append(
                        _run_case(
                            system=system,
                            state_id=state_id,
                            state=state,
                            seed=seed,
                            scope=scope,
                            calculator=calculator,
                            config_builder=config_builder,
                            case_directory=case_directory,
                        )
                    )
                    print(
                        f"{system} {state_id} seed={seed} scope={scope} "
                        f"FE={rows[-1]['force_evaluations']} "
                        f"dE={rows[-1]['landing_delta_eV']:.6f}"
                    )
    evidence = build_evidence(rows)
    _write_json(output / "evidence.json", evidence)
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only:
        print(json.dumps(preflight(args.expected_git_commit), indent=2, sort_keys=True))
        return 0
    evidence = run_gate(args.output, args.expected_git_commit)
    print(json.dumps(evidence["scope_summaries"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

