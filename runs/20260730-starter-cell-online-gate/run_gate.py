#!/usr/bin/env python3
"""C60/PdO online gate for full-archive uniform versus MACE-FPS cells."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from ase import Atoms

from pamssw import validated_ls_ssw_config
from pamssw.exploration import (
    PosteriorExplorationConfig,
    run_posterior_ls_ssw,
    run_posterior_ssw,
)
from pamssw.exploration.cells import FPSCellPartition, build_fps_cell_partition
from pamssw.io import write_state


RUN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUN_ROOT.parents[1]
PRIOR_HARNESS_PATH = (
    RUN_ROOT.parent / "20260728-posterior-starter-policy-gpu-ablation" / "run_ablation.py"
)
POLICIES = ("uniform", "fps_cell_uniform")
C60_PROFILE = "c60_direction_efficient_validated_20260729"
_PRIOR_MODULE_NAME = "_starter_cell_gate_prior_harness"
KERNEL_MODES = ("unsoftened_smoke", "production_ls")


def cell_count_for_trial_gate(
    planned_trials: int,
    *,
    observations_per_cell: int,
) -> int:
    """Convert an explicit posterior-resolution target into a cell capacity."""
    if isinstance(planned_trials, bool) or not isinstance(planned_trials, int):
        raise ValueError("planned_trials must be a positive integer")
    if isinstance(observations_per_cell, bool) or not isinstance(observations_per_cell, int):
        raise ValueError("observations_per_cell must be a positive integer")
    if planned_trials <= 0 or observations_per_cell <= 0:
        raise ValueError("trial resolution values must be positive")
    return max(1, planned_trials // observations_per_cell)


def pool_mace_invariants(
    atomic_features: np.ndarray,
    numbers: np.ndarray,
    *,
    species: tuple[int, ...],
) -> np.ndarray:
    features = np.asarray(atomic_features, dtype=float)
    atomic_numbers = np.asarray(numbers, dtype=int)
    return np.concatenate(
        [features[atomic_numbers == atomic_number].mean(axis=0) for atomic_number in species]
    )


class MACEFPSCellSnapshotBuilder:
    """Cache invariant MACE features and emit exact full-support cell marginals."""

    policy_name = "fps_cell_uniform"

    def __init__(
        self,
        descriptor_calculator,
        *,
        species: tuple[int, ...],
        max_cells: int,
    ) -> None:
        self.descriptor_calculator = descriptor_calculator
        self.species = tuple(species)
        self.max_cells = max_cells
        self._features: dict[int, np.ndarray] = {}
        self._records: list[dict[str, object]] = []
        self._descriptor_forward_calls = 0
        self._descriptor_wall_time_s = 0.0

    def __call__(self, archive, posterior, version: int, archive_version: int):
        del posterior
        entries = tuple(sorted(archive.entries, key=lambda entry: entry.entry_id))
        for entry in entries:
            if entry.entry_id not in self._features:
                self._features[entry.entry_id] = self._describe(entry.state)
        partition = build_fps_cell_partition(
            tuple(entry.entry_id for entry in entries),
            np.vstack([self._features[entry.entry_id] for entry in entries]),
            max_cells=self.max_cells,
        )
        self._records.append(
            self._partition_record(partition, version, archive_version)
        )
        return partition.policy_snapshot(
            version=version,
            archive_version=archive_version,
        )

    def _describe(self, state) -> np.ndarray:
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=state.cell,
            pbc=state.pbc,
        )
        started = perf_counter()
        atomic_features = self.descriptor_calculator.get_descriptors(
            atoms,
            invariants_only=True,
            num_layers=-1,
        )
        self._descriptor_wall_time_s += perf_counter() - started
        self._descriptor_forward_calls += 1
        return pool_mace_invariants(
            atomic_features,
            state.numbers,
            species=self.species,
        )

    @staticmethod
    def _partition_record(
        partition: FPSCellPartition,
        version: int,
        archive_version: int,
    ) -> dict[str, object]:
        return {
            "policy_version": version,
            "archive_version": archive_version,
            "starter_ids": list(partition.starter_ids),
            "center_ids": list(partition.center_ids),
            "members_by_center": [list(members) for members in partition.members_by_center],
            "probabilities": list(partition.probabilities),
            "cell_probabilities": [
                1.0 / len(partition.center_ids)
                for _ in partition.center_ids
            ],
            "conditional_probabilities": [
                partition.conditional_probability_for(starter_id)
                for starter_id in partition.starter_ids
            ],
        }

    def partition_records(self) -> tuple[dict[str, object], ...]:
        return tuple(self._records)

    def descriptor_cost(self) -> dict[str, float | int]:
        return {
            "descriptor_forward_calls": self._descriptor_forward_calls,
            "descriptor_wall_time_s": self._descriptor_wall_time_s,
        }


def _load_prior_harness():
    module = sys.modules.get(_PRIOR_MODULE_NAME)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(_PRIOR_MODULE_NAME, PRIOR_HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the frozen posterior-policy harness")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PRIOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _descriptor_calculator(prior):
    from mace.calculators import MACECalculator

    production = prior._load_production_runner()
    return MACECalculator(
        model_paths=str(production.MODEL_PATH),
        **production.CALCULATOR_CONFIG,
    )


def build_production_config(
    system: str,
    case_directory: Path,
    *,
    master_seed: int,
):
    """Resolve the evidence-backed system-specific LS-SSW action kernel."""
    case_directory = Path(case_directory)
    if system == "c60":
        config = validated_ls_ssw_config(
            C60_PROFILE,
            output_dir=case_directory,
            max_trials=200,
            rng_seed=master_seed,
            max_force_evals=None,
        )
    elif system == "pdo":
        prior = _load_prior_harness()
        production = prior._load_production_runner()
        config = production.build_config(system, case_directory)
    else:
        raise ValueError(f"unknown system: {system}")
    return replace(
        config,
        rng_seed=master_seed,
        accepted_structures_log=None,
        accepted_structures_dir=None,
        write_proposal_minima=False,
        proposal_minima_dir=None,
        write_relaxation_trajectories=False,
        relaxation_trajectory_dir=None,
        direction_diagnostics_enabled=False,
        direction_diagnostics_path=None,
        direction_archive_enabled=False,
        direction_archive_path=None,
        proposal_pool_size=1,
        proposal_duplicate_rescue_optimizer=None,
    )


def _run_production_campaign(
    *,
    initial_state,
    calculator_factory,
    ssw_config,
    exploration_config,
    snapshot_builder,
    prior,
) -> dict[str, Any]:
    started = perf_counter()
    result = run_posterior_ls_ssw(
        initial_state,
        calculator_factory,
        ssw_config,
        exploration_config,
        snapshot_builder=snapshot_builder,
    )
    wall_time_s = perf_counter() - started
    minima_directory = exploration_config.run_directory / "archive_minima"
    minima_directory.mkdir()
    for entry in result.archive.entries:
        write_state(
            minima_directory / f"entry-{entry.entry_id:05d}.xyz",
            entry.state,
        )
    summary = {
        "schema_version": 1,
        "policy_name": result.policy_name,
        "kernel_mode": "production_ls",
        "master_seed": exploration_config.master_seed,
        "action_force_budget": exploration_config.action_force_budget,
        "total_force_budget": result.total_force_budget,
        "bootstrap_evaluations": result.bootstrap_evaluations,
        "action_evaluations": result.action_evaluations,
        "total_evaluations": result.total_evaluations,
        "unused_force_budget": result.unused_force_budget,
        "purpose_counts": result.purpose_counts.as_dict(),
        "completed_batches": result.completed_batches,
        "completed_attempts": result.completed_attempts,
        "failed_attempts": result.failed_attempts,
        "posterior_observed_attempts": result.posterior_observed_attempts,
        "benchmark_eligible": result.benchmark_eligible,
        "benchmark_ineligibility_reasons": list(
            result.benchmark_ineligibility_reasons
        ),
        "archive_entries": len(result.archive.entries),
        "best_archive_energy_eV": min(
            float(entry.energy) for entry in result.archive.entries
        ),
        "bootstrap_energy_eV": float(result.archive.entries[0].energy),
        "duplicate_rate": result.archive.duplicate_rate(),
        "campaign_wall_time_s": wall_time_s,
        "effective_ssw_config": asdict(ssw_config),
        "event_log": "events.jsonl",
        "optimizer_diagnostics": "optimizer_diagnostics.json",
        "archive_minima": "archive_minima",
    }
    prior._write_json_exclusive(
        exploration_config.run_directory / "campaign_summary.json",
        summary,
    )
    return summary


def run_gate(
    *,
    output_root: Path,
    expected_git_commit: str,
    systems: Sequence[str],
    master_seeds: Sequence[int],
    action_force_budget: int,
    total_force_budget: int,
    trial_resolution_reference: int,
    observations_per_cell: int = 3,
    kernel_mode: str = "unsoftened_smoke",
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run the first online gate with one frozen SSW action kernel."""
    prior = _load_prior_harness()
    if kernel_mode not in KERNEL_MODES:
        raise ValueError(f"kernel_mode must be one of {KERNEL_MODES!r}")
    output_root = Path(output_root)
    prior._preflight_output_root(output_root)
    manifest = prior.preflight(
        expected_git_commit=expected_git_commit,
        systems=systems,
        master_seeds=master_seeds,
        action_force_budget=action_force_budget,
        total_force_budget=total_force_budget,
    )
    max_cells = cell_count_for_trial_gate(
        trial_resolution_reference,
        observations_per_cell=observations_per_cell,
    )
    manifest = dict(manifest)
    manifest.update(
        {
            "policies": list(POLICIES),
            "trial_resolution_reference": trial_resolution_reference,
            "observations_per_cell": observations_per_cell,
            "max_cells": max_cells,
            "cell_probability": "1 / n_nonempty_cells / members_in_selected_cell",
            "cell_first_center": "lowest_entry_id",
            "cell_representation": (
                "species-mean pooled invariant MACE features from all message-passing layers"
            ),
            "archive_nodes_deleted": False,
            "direction_uphill_optimizer_frozen": True,
            "kernel_mode": kernel_mode,
        }
    )
    if kernel_mode == "production_ls":
        manifest["projections"] = {
            system: {
                "source_config_type": "LSSSWConfig",
                "softening_enabled": True,
                "effective_ssw_config": asdict(
                    build_production_config(
                        system,
                        RUN_ROOT / ".preflight" / system,
                        master_seed=int(manifest["master_seeds"][0]),
                    )
                ),
            }
            for system in manifest["systems"]
        }
    if preflight_only:
        return manifest

    production = prior._load_production_runner()
    output_root.mkdir()
    prior._write_json_exclusive(output_root / "manifest.json", manifest)
    campaigns: list[dict[str, Any]] = []
    for system in manifest["systems"]:
        initial_state = production.load_state(system)
        species = tuple(sorted(set(int(number) for number in initial_state.numbers)))
        for master_seed in manifest["master_seeds"]:
            for policy_name in POLICIES:
                case_directory = (
                    output_root / system / f"seed-{master_seed:08d}" / policy_name
                )
                case_directory.parent.mkdir(parents=True, exist_ok=True)
                if kernel_mode == "production_ls":
                    config = build_production_config(
                        system,
                        case_directory,
                        master_seed=master_seed,
                    )
                    projection = {
                        "source_config_type": "LSSSWConfig",
                        "softening_enabled": True,
                        "effective_ssw_config": asdict(config),
                    }
                else:
                    config, projection = prior.build_ssw_config(
                        system,
                        case_directory,
                    )
                exploration = PosteriorExplorationConfig(
                    policy_name=policy_name,
                    batch_size=1,
                    max_workers=1,
                    action_force_budget=action_force_budget,
                    total_force_budget=total_force_budget,
                    master_seed=master_seed,
                    run_directory=case_directory,
                )
                owner = prior.ThreadOwnedCalculatorFactory(prior._mace_calculator_factory)
                factory = prior.InstrumentedCalculatorFactory(owner)
                builder = None
                if policy_name == "fps_cell_uniform":
                    builder = MACEFPSCellSnapshotBuilder(
                        _descriptor_calculator(prior),
                        species=species,
                        max_cells=max_cells,
                    )

                if kernel_mode == "production_ls":
                    summary = _run_production_campaign(
                        initial_state=initial_state,
                        calculator_factory=factory,
                        ssw_config=config,
                        exploration_config=exploration,
                        snapshot_builder=builder,
                        prior=prior,
                    )
                else:
                    def selected_runner(initial, calculator_factory, ssw_config, explore):
                        return run_posterior_ssw(
                            initial,
                            calculator_factory,
                            ssw_config,
                            explore,
                            snapshot_builder=builder,
                        )

                    summary = prior.run_campaign(
                        initial_state=initial_state,
                        calculator_factory=factory,
                        ssw_config=config,
                        exploration_config=exploration,
                        run_posterior=selected_runner,
                    )
                cell_sidecar = None
                if builder is not None:
                    cell_sidecar = case_directory / "cell_partitions.json"
                    prior._write_json_exclusive(
                        cell_sidecar,
                        {
                            "schema_version": 1,
                            "max_cells": max_cells,
                            "descriptor_cost": builder.descriptor_cost(),
                            "partitions": list(builder.partition_records()),
                        },
                    )
                campaigns.append(
                    {
                        "system": system,
                        "master_seed": master_seed,
                        "policy_name": policy_name,
                        "projection": projection,
                        "completed_attempts": (
                            int(summary["completed_attempts"])
                            + int(summary["failed_attempts"])
                        ),
                        "campaign_summary": str(
                            (case_directory / "campaign_summary.json").relative_to(output_root)
                        ),
                        "cell_partitions": (
                            None
                            if cell_sidecar is None
                            else str(cell_sidecar.relative_to(output_root))
                        ),
                    }
                )
    index = {
        "schema_version": 1,
        "manifest": manifest,
        "campaigns": campaigns,
    }
    prior._write_json_exclusive(output_root / "index.json", index)
    return index


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--systems", nargs="+", choices=("c60", "pdo"), required=True)
    parser.add_argument("--master-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--action-force-budget", type=int, required=True)
    parser.add_argument("--total-force-budget", type=int, required=True)
    parser.add_argument("--trial-resolution-reference", type=int, required=True)
    parser.add_argument("--observations-per-cell", type=int, default=3)
    parser.add_argument(
        "--kernel-mode",
        choices=KERNEL_MODES,
        default="unsoftened_smoke",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_gate(
        output_root=args.output,
        expected_git_commit=args.expected_git_commit,
        systems=tuple(args.systems),
        master_seeds=tuple(args.master_seeds),
        action_force_budget=args.action_force_budget,
        total_force_budget=args.total_force_budget,
        trial_resolution_reference=args.trial_resolution_reference,
        observations_per_cell=args.observations_per_cell,
        kernel_mode=args.kernel_mode,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "MACEFPSCellSnapshotBuilder",
    "build_production_config",
    "cell_count_for_trial_gate",
    "pool_mace_invariants",
    "run_gate",
]


if __name__ == "__main__":
    raise SystemExit(main())
