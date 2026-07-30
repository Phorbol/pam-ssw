#!/usr/bin/env python3
"""Offline starter-representation and archive-growth audit.

This experiment does not alter or invoke the production walker.  The functions
in this file define only the geometry needed to compare representation and
candidate-pool mechanisms on already generated minima.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.io import read

from pamssw.pbc import mic_displacement
from pamssw.fingerprint import structural_descriptor
from pamssw.state import State


@dataclass(frozen=True)
class MinimumRecord:
    campaign: str
    entry_id: int
    trial_index: int
    seed_entry_id: int
    energy_eV: float
    best_energy_eV: float
    global_best_gain_eV: float
    state: State


@dataclass(frozen=True)
class CampaignCorpus:
    path: Path
    system: str
    arm: str
    completed_trials: int
    archive_entries: int
    force_evaluations: int
    wall_time_s: float
    records: tuple[MinimumRecord, ...]


def _state_from_atoms(atoms, *, source: Path, system: str) -> State:
    pbc = tuple(bool(value) for value in atoms.pbc)
    cell = np.asarray(atoms.cell.array, dtype=float) if any(pbc) else None
    return State(
        numbers=np.asarray(atoms.numbers, dtype=int),
        positions=np.asarray(atoms.positions, dtype=float),
        cell=cell,
        pbc=pbc,
        metadata={"source": str(source), "system": system},
    )


def load_campaign_corpus(path: Path) -> CampaignCorpus:
    """Load the explicitly saved new-minimum subset of one completed campaign."""
    campaign_path = Path(path)
    summary = json.loads((campaign_path / "summary.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (campaign_path / "accepted_structures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows.sort(key=lambda row: (int(row["trial_index"]), int(row["discovered_entry_id"])))
    previous_best = float(summary["initial_energy_eV"])
    records: list[MinimumRecord] = []
    for row in rows:
        trial_index = int(row["trial_index"])
        entry_id = int(row["discovered_entry_id"])
        xyz_path = (
            campaign_path
            / "accepted_minima"
            / f"trial{trial_index:04d}_entry{entry_id:04d}_accepted.xyz"
        )
        if not xyz_path.is_file():
            raise FileNotFoundError(xyz_path)
        best_energy = float(row["best_energy"])
        gain = max(0.0, previous_best - best_energy)
        previous_best = min(previous_best, best_energy)
        records.append(
            MinimumRecord(
                campaign=campaign_path.name,
                entry_id=entry_id,
                trial_index=trial_index,
                seed_entry_id=int(row["seed_entry_id"]),
                energy_eV=float(row["energy"]),
                best_energy_eV=best_energy,
                global_best_gain_eV=gain,
                state=_state_from_atoms(
                    read(xyz_path),
                    source=xyz_path,
                    system=str(summary["system"]),
                ),
            )
        )
    return CampaignCorpus(
        path=campaign_path,
        system=str(summary["system"]),
        arm=str(summary["arm"]),
        completed_trials=int(summary["completed_trials"]),
        archive_entries=int(summary["archive_entries"]),
        force_evaluations=int(summary["force_evaluations"]),
        wall_time_s=float(summary["wall_time_s"]),
        records=tuple(records),
    )


def _canonical_species(numbers: np.ndarray, species: Sequence[int] | None) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in numbers})) if species is None else tuple(species)
    if not values or len(set(values)) != len(values) or tuple(sorted(values)) != values:
        raise ValueError("species must be a nonempty sorted unique sequence")
    if any(int(number) not in values for number in numbers):
        raise ValueError("species does not cover every atomic number")
    return values


def species_pair_rdf(
    state: State,
    *,
    species: Sequence[int] | None = None,
    n_bins: int = 16,
    r_max: float = 6.0,
) -> np.ndarray:
    """Return fixed-grid, species-pair histograms with one unit-norm block per pair."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if not np.isfinite(r_max) or r_max <= 0.0:
        raise ValueError("r_max must be positive and finite")
    species_values = _canonical_species(state.numbers, species)
    pair_values: dict[tuple[int, int], list[float]] = {
        pair: [] for pair in combinations_with_replacement(species_values, 2)
    }
    for atom_i in range(state.n_atoms):
        delta = state.positions[atom_i + 1 :] - state.positions[atom_i]
        if delta.size:
            delta = mic_displacement(delta, np.zeros_like(delta), state.cell, state.pbc)
        for offset, distance in enumerate(np.linalg.norm(delta, axis=1), start=atom_i + 1):
            pair = tuple(sorted((int(state.numbers[atom_i]), int(state.numbers[offset]))))
            pair_values[pair].append(float(distance))

    blocks: list[np.ndarray] = []
    for pair in combinations_with_replacement(species_values, 2):
        histogram, _ = np.histogram(pair_values[pair], bins=n_bins, range=(0.0, r_max))
        block = histogram.astype(float)
        norm = float(np.linalg.norm(block))
        if norm > 0.0:
            block /= norm
        blocks.append(block)
    return np.concatenate(blocks)


def pool_mace_invariants(
    atomic_features: np.ndarray,
    atomic_numbers: np.ndarray,
    *,
    species: Sequence[int] | None = None,
) -> np.ndarray:
    """Mean-pool invariant atomic features into sorted species blocks."""
    features = np.asarray(atomic_features, dtype=float)
    numbers = np.asarray(atomic_numbers, dtype=int)
    if features.ndim != 2 or numbers.ndim != 1 or features.shape[0] != numbers.size:
        raise ValueError("atomic features and atomic numbers must share the atom dimension")
    if not np.all(np.isfinite(features)):
        raise ValueError("atomic features must be finite")
    species_values = _canonical_species(numbers, species)
    return np.concatenate([features[numbers == value].mean(axis=0) for value in species_values])


def mace_representation_matrix(
    states: Sequence[State],
    calculator,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Extract and species-pool MACE invariant features, charging each forward."""
    state_values = tuple(states)
    if not state_values:
        raise ValueError("states must be nonempty")
    species = tuple(sorted({int(value) for state in state_values for value in state.numbers}))
    rows: list[np.ndarray] = []
    started = perf_counter()
    for state in state_values:
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=state.cell,
            pbc=state.pbc,
        )
        atomic_features = calculator.get_descriptors(
            atoms,
            invariants_only=True,
            num_layers=-1,
        )
        rows.append(
            pool_mace_invariants(
                atomic_features,
                state.numbers,
                species=species,
            )
        )
    elapsed = perf_counter() - started
    return np.vstack(rows), {
        "descriptor_forward_calls": len(rows),
        "wall_time_s": float(elapsed),
        "mean_wall_time_s": float(elapsed / len(rows)),
    }


@dataclass(frozen=True)
class FrozenPCA:
    mean: np.ndarray
    components: np.ndarray
    retained_variance_fraction: float

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.mean.size:
            raise ValueError("matrix feature dimension does not match fitted PCA")
        return (values - self.mean) @ self.components.T


def fit_frozen_pca(matrix: np.ndarray, *, variance_fraction: float = 0.95) -> FrozenPCA:
    """Fit one centered, non-whitened PCA that is immutable during a campaign."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("PCA matrix must contain at least two rows and one feature")
    if not np.all(np.isfinite(values)):
        raise ValueError("PCA matrix must be finite")
    if not np.isfinite(variance_fraction) or not 0.0 < variance_fraction <= 1.0:
        raise ValueError("variance_fraction must lie in (0, 1]")
    mean = values.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(values - mean, full_matrices=False)
    variances = singular_values**2
    total = float(variances.sum())
    if total <= 0.0:
        count = 1
        retained = 1.0
    else:
        cumulative = np.cumsum(variances) / total
        count = int(np.searchsorted(cumulative, variance_fraction, side="left") + 1)
        retained = float(cumulative[count - 1])
    return FrozenPCA(
        mean=mean.copy(),
        components=right_vectors[:count].copy(),
        retained_variance_fraction=retained,
    )


def farthest_point_order(
    matrix: np.ndarray,
    *,
    energies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic energy-anchored FPS order and prefix cover radii."""
    values = np.asarray(matrix, dtype=float)
    energy_values = np.asarray(energies, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("matrix must be a nonempty two-dimensional array")
    if energy_values.shape != (values.shape[0],):
        raise ValueError("energies must contain one value per row")
    start = min(range(values.shape[0]), key=lambda index: (energy_values[index], index))
    order = [start]
    minimum_distances = np.linalg.norm(values - values[start], axis=1)
    radii = [float(minimum_distances.max())]
    selected = np.zeros(values.shape[0], dtype=bool)
    selected[start] = True
    while len(order) < values.shape[0]:
        maximum = float(minimum_distances[~selected].max())
        candidates = np.flatnonzero((~selected) & np.isclose(minimum_distances, maximum))
        next_index = min(candidates, key=lambda index: (energy_values[index], int(index)))
        selected[next_index] = True
        order.append(int(next_index))
        distances = np.linalg.norm(values - values[next_index], axis=1)
        minimum_distances = np.minimum(minimum_distances, distances)
        radii.append(float(minimum_distances.max()))
    return np.asarray(order, dtype=int), np.asarray(radii, dtype=float)


def _spearman_or_none(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    if lhs.size < 2 or np.ptp(lhs) <= 0.0 or np.ptp(rhs) <= 0.0:
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(lhs, rhs).statistic)
    return value if np.isfinite(value) else None


def representation_metrics(
    matrix: np.ndarray,
    energies: np.ndarray,
    *,
    recorded_targets: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure geometry, energy continuity, and FPS coverage without fitting a selector."""
    values = np.asarray(matrix, dtype=float)
    energy_values = np.asarray(energies, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("representation matrix must contain at least two samples")
    if energy_values.shape != (values.shape[0],):
        raise ValueError("energies must contain one value per representation row")
    differences = values[:, None, :] - values[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    nearest_distances = distances.copy()
    np.fill_diagonal(nearest_distances, np.inf)
    nearest_indices = np.argmin(nearest_distances, axis=1)
    nearest_energy_gaps = np.abs(energy_values - energy_values[nearest_indices])
    upper = np.triu_indices(values.shape[0], k=1)
    pair_distances = distances[upper]
    pair_energy_gaps = np.abs(energy_values[:, None] - energy_values[None, :])[upper]
    _, fps_radii = farthest_point_order(values, energies=energy_values)
    result: dict[str, object] = {
        "n_samples": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "pair_distance_median": float(np.median(pair_distances)),
        "zero_pair_fraction": float(np.mean(pair_distances <= 1e-12)),
        "nearest_neighbor_energy_mae_eV": float(nearest_energy_gaps.mean()),
        "nearest_neighbor_energy_median_abs_error_eV": float(np.median(nearest_energy_gaps)),
        "pair_distance_energy_gap_spearman": _spearman_or_none(
            pair_distances,
            pair_energy_gaps,
        ),
        "fps_cover_radius": fps_radii.tolist(),
    }
    if recorded_targets is not None:
        targets = np.asarray(recorded_targets, dtype=float)
        if targets.shape != (values.shape[0],):
            raise ValueError("recorded_targets must contain one value per row")
        valid = np.isfinite(targets)
        target_result: dict[str, object] = {"n_observed": int(valid.sum())}
        if valid.sum() >= 2:
            valid_values = values[valid]
            valid_targets = targets[valid]
            target_distances = np.linalg.norm(
                valid_values[:, None, :] - valid_values[None, :, :],
                axis=2,
            )
            nearest = target_distances.copy()
            np.fill_diagonal(nearest, np.inf)
            neighbors = np.argmin(nearest, axis=1)
            target_result["nearest_neighbor_target_mae"] = float(
                np.abs(valid_targets - valid_targets[neighbors]).mean()
            )
            valid_upper = np.triu_indices(valid_values.shape[0], k=1)
            target_result["pair_distance_target_gap_spearman"] = _spearman_or_none(
                target_distances[valid_upper],
                np.abs(valid_targets[:, None] - valid_targets[None, :])[valid_upper],
            )
        result["recorded_target_continuity"] = target_result
    return result


def archive_growth_audit(
    rows: Sequence[Mapping[str, object]],
    *,
    completed_trials: int,
    archive_entries: int,
) -> dict[str, object]:
    """Report observable arm growth without inferring missing duplicate-trial labels."""
    if completed_trials <= 0 or archive_entries <= 0:
        raise ValueError("completed_trials and archive_entries must be positive")
    new_count = len(rows)
    if new_count > completed_trials:
        raise ValueError("new-minimum rows cannot exceed completed trials")
    productive_starters = {int(row["seed_entry_id"]) for row in rows}
    return {
        "completed_trials": int(completed_trials),
        "archive_entries": int(archive_entries),
        "new_minimum_rows": new_count,
        "new_minimum_trial_fraction": float(new_count / completed_trials),
        "new_minimum_reward_positive_fraction": float(new_count / completed_trials),
        "productive_starter_ids": len(productive_starters),
        "unobserved_trial_count": int(completed_trials - new_count),
        "node_trial_counts_reconstructable": False,
    }


def arm_pressure_audit(*, completed_trials: int, archive_entries: int) -> dict[str, object]:
    """Quantify the best-case data dilution of a growing node-as-arm bandit."""
    if completed_trials <= 0 or archive_entries <= 0:
        raise ValueError("completed_trials and archive_entries must be positive")
    observations_per_arm = float(completed_trials / archive_entries)
    return {
        "completed_trials": int(completed_trials),
        "archive_entries": int(archive_entries),
        "new_arms_per_trial": float(max(0, archive_entries - 1) / completed_trials),
        "trials_per_final_arm": observations_per_arm,
        "uniform_observations_per_arm": observations_per_arm,
        "beta11_prior_std": float(1.0 / np.sqrt(12.0)),
        "uniform_best_case_posterior_std": float(
            1.0 / (2.0 * np.sqrt(observations_per_arm + 3.0))
        ),
        "all_prior_ts_max_mean_reference": float(archive_entries / (archive_entries + 1.0)),
        "cold_start_fraction_reconstructable": False,
    }


def benchmark_legacy_selector(
    sample_sizes: Sequence[int],
    *,
    descriptor_dimension: int = 20,
) -> dict[str, object]:
    """Time the current composite selector without changing its implementation."""
    from pamssw.acquisition import AcquisitionPolicy, BanditSelector
    from pamssw.archive import ArchivePrototype, MinimaArchive, MinimaEntry

    sizes = tuple(int(value) for value in sample_sizes)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("sample sizes must be positive")
    if descriptor_dimension <= 0:
        raise ValueError("descriptor_dimension must be positive")
    rng = np.random.default_rng(20260730)
    selector = BanditSelector(AcquisitionPolicy(baseline_probability=0.0))
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    measurements: list[dict[str, float | int]] = []
    for size in sizes:
        archive = MinimaArchive(energy_tol=1e-3, rmsd_tol=0.1, max_prototypes=min(size, 1000))
        descriptors = rng.normal(size=(size, descriptor_dimension))
        archive.entries = [
            MinimaEntry(
                entry_id=index,
                state=state,
                energy=float(index) / max(1, size - 1),
                parent_id=None,
                descriptor=descriptors[index],
                node_trials=index % 4,
            )
            for index in range(size)
        ]
        archive.prototypes = [
            ArchivePrototype(
                descriptor=descriptors[index].copy(),
                representative_entry_id=index,
            )
            for index in range(min(size, archive.max_prototypes))
        ]
        started = perf_counter()
        selector.select(archive, rng)
        elapsed = perf_counter() - started
        measurements.append(
            {
                "archive_entries": size,
                "prototypes": len(archive.prototypes),
                "wall_time_s": float(elapsed),
            }
        )
    result: dict[str, object] = {
        "uses_real_bandit_selector": True,
        "measurements": measurements,
    }
    if len(measurements) >= 2 and all(row["wall_time_s"] > 0.0 for row in measurements):
        x = np.log([row["archive_entries"] for row in measurements])
        y = np.log([row["wall_time_s"] for row in measurements])
        result["empirical_loglog_slope"] = float(np.polyfit(x, y, 1)[0])
    return result


def audit_system(
    corpora: Sequence[CampaignCorpus],
    calculator,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Compare four frozen representation arms for one chemical system."""
    corpus_values = tuple(corpora)
    if not corpus_values:
        raise ValueError("corpora must be nonempty")
    systems = {corpus.system for corpus in corpus_values}
    if len(systems) != 1:
        raise ValueError("all corpora in one audit must describe the same system")
    records = tuple(record for corpus in corpus_values for record in corpus.records)
    if len(records) < 2:
        raise ValueError("at least two saved minima are required")
    states = tuple(record.state for record in records)
    species = tuple(sorted({int(value) for state in states for value in state.numbers}))
    current_rdf = np.vstack([structural_descriptor(state) for state in states])
    fixed_rdf = np.vstack(
        [
            species_pair_rdf(
                state,
                species=species,
                n_bins=16,
                r_max=6.0,
            )
            for state in states
        ]
    )
    mace_raw, mace_timing = mace_representation_matrix(states, calculator)
    pca = fit_frozen_pca(mace_raw, variance_fraction=0.95)
    mace_pca = pca.transform(mace_raw)
    matrices = {
        "current_rdf": current_rdf,
        "fixed_species_rdf": fixed_rdf,
        "mace_raw": mace_raw,
        "mace_pca95": mace_pca,
    }
    energies = np.asarray([record.energy_eV for record in records], dtype=float)
    record_indices = {
        (record.campaign, record.entry_id): index for index, record in enumerate(records)
    }
    observed_outcomes: list[list[float]] = [[] for _ in records]
    for record in records:
        source_index = record_indices.get((record.campaign, record.seed_entry_id))
        if source_index is not None:
            observed_outcomes[source_index].append(record.global_best_gain_eV)
    targets = np.asarray(
        [max(values) if values else np.nan for values in observed_outcomes],
        dtype=float,
    )
    representation_results = {
        name: representation_metrics(matrix, energies, recorded_targets=targets)
        for name, matrix in matrices.items()
    }
    total_trials = sum(corpus.completed_trials for corpus in corpus_values)
    total_saved = len(records)
    report = {
        "system": next(iter(systems)),
        "campaigns": [
            {
                "name": corpus.path.name,
                "arm": corpus.arm,
                "completed_trials": corpus.completed_trials,
                "archive_entries": corpus.archive_entries,
                "saved_new_minima": len(corpus.records),
                "force_evaluations": corpus.force_evaluations,
                "wall_time_s": corpus.wall_time_s,
                "arm_pressure": arm_pressure_audit(
                    completed_trials=corpus.completed_trials,
                    archive_entries=corpus.archive_entries,
                ),
            }
            for corpus in corpus_values
        ],
        "data_boundary": {
            "saved_new_minima": total_saved,
            "completed_trials": total_trials,
            "unobserved_trial_count": total_trials - total_saved,
            "node_trial_counts_reconstructable": False,
            "recorded_target_definition": (
                "maximum global-best gain among saved new-minimum transitions "
                "originating from the starter; duplicate and failed trials are censored"
            ),
            "recorded_target_count": int(np.isfinite(targets).sum()),
        },
        "mace_descriptor_cost": mace_timing,
        "pca": {
            "variance_fraction_target": 0.95,
            "retained_variance_fraction": pca.retained_variance_fraction,
            "input_dimension": int(mace_raw.shape[1]),
            "retained_dimension": int(mace_pca.shape[1]),
            "centered": True,
            "whitened": False,
            "frozen_within_audit": True,
        },
        "representations": representation_results,
    }
    return report, matrices


def _sha256(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_offline_audit(
    *,
    campaign_paths: Sequence[Path],
    model_path: Path,
    output_directory: Path,
    device: str,
    selector_sizes: Sequence[int],
) -> dict[str, object]:
    """Run S0+S1 on immutable campaign artifacts and write replayable evidence."""
    paths = tuple(Path(path).resolve() for path in campaign_paths)
    if not paths:
        raise ValueError("at least one campaign is required")
    model = Path(model_path).resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    corpora = tuple(load_campaign_corpus(path) for path in paths)

    from mace.calculators import MACECalculator

    calculator = MACECalculator(
        model_paths=str(model),
        device=device,
        default_dtype="float32",
        inference_precision="float32",
        enable_cueq=False,
    )
    systems: dict[str, list[CampaignCorpus]] = {}
    for corpus in corpora:
        systems.setdefault(corpus.system, []).append(corpus)
    system_reports: dict[str, object] = {}
    array_files: dict[str, dict[str, str]] = {}
    for system in sorted(systems):
        report, matrices = audit_system(systems[system], calculator)
        energies = np.asarray(
            [
                record.energy_eV
                for corpus in systems[system]
                for record in corpus.records
            ],
            dtype=float,
        )
        archive = output / f"{system}_representations.npz"
        np.savez_compressed(archive, energies=energies, **matrices)
        system_reports[system] = report
        array_files[system] = {
            "path": str(archive),
            "sha256": _sha256(archive),
        }
    selector_timing = benchmark_legacy_selector(
        selector_sizes,
        descriptor_dimension=20,
    )
    force_wall_references = {
        corpus.path.name: {
            "system": corpus.system,
            "campaign_wall_time_s_per_force_evaluation": float(
                corpus.wall_time_s / corpus.force_evaluations
            ),
        }
        for corpus in corpora
    }
    evidence = {
        "schema_version": 1,
        "execution_commit": _git_commit(),
        "scope": "offline S0 archive/arm audit plus S1 representation comparison",
        "production_walker_modified": False,
        "model": {
            "path": str(model),
            "sha256": _sha256(model),
            "device": device,
            "default_dtype": "float32",
            "inference_precision": "float32",
            "enable_cueq": False,
        },
        "campaign_sources": [
            {
                "path": str(corpus.path),
                "summary_sha256": _sha256(corpus.path / "summary.json"),
                "accepted_structures_sha256": _sha256(
                    corpus.path / "accepted_structures.jsonl"
                ),
            }
            for corpus in corpora
        ],
        "systems": system_reports,
        "representation_arrays": array_files,
        "legacy_selector_timing": selector_timing,
        "force_wall_references": force_wall_references,
        "claim_boundary": {
            "complete_new_minimum_structures": True,
            "complete_trial_level_outcomes": False,
            "node_trial_counts_reconstructable": False,
            "representation_metrics_are_offline": True,
            "online_selector_ranking_supported": False,
        },
    }
    evidence_path = output / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", action="append", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--selector-sizes",
        nargs="+",
        type=int,
        default=(100, 300, 1000),
    )
    args = parser.parse_args()
    run_offline_audit(
        campaign_paths=args.campaign,
        model_path=args.model,
        output_directory=args.output,
        device=args.device,
        selector_sizes=args.selector_sizes,
    )


if __name__ == "__main__":
    main()
