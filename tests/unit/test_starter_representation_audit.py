"""Unit contracts for the offline starter-representation audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.io import write

from pamssw.state import State


RUNNER_PATH = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "20260730-starter-representation-audit"
    / "run_audit.py"
)


def _runner():
    name = "_starter_representation_audit_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load starter-representation audit")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _mixed_state() -> State:
    return State(
        numbers=np.array([8, 46, 8, 46], dtype=int),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
    )


def test_species_pair_rdf_has_fixed_species_blocks_and_is_rigid_permutation_invariant():
    runner = _runner()
    state = _mixed_state()
    permutation = np.array([2, 0, 3, 1])
    transformed = State(
        numbers=state.numbers[permutation],
        positions=state.positions[permutation] + np.array([4.0, -3.0, 2.0]),
    )

    descriptor = runner.species_pair_rdf(state, n_bins=4, r_max=4.0)
    transformed_descriptor = runner.species_pair_rdf(
        transformed,
        species=(8, 46),
        n_bins=4,
        r_max=4.0,
    )

    assert descriptor.shape == (12,)
    np.testing.assert_allclose(descriptor, transformed_descriptor)
    assert np.count_nonzero(descriptor[:4]) == 1
    assert np.count_nonzero(descriptor[4:8]) == 3
    assert np.count_nonzero(descriptor[8:]) == 1


def test_pool_mace_invariants_is_species_resolved_and_atom_order_invariant():
    runner = _runner()
    numbers = np.array([8, 46, 8, 46], dtype=int)
    atomic_features = np.array(
        [
            [1.0, 2.0],
            [10.0, 20.0],
            [3.0, 4.0],
            [30.0, 40.0],
        ]
    )
    permutation = np.array([3, 0, 2, 1])

    pooled = runner.pool_mace_invariants(atomic_features, numbers)
    permuted = runner.pool_mace_invariants(
        atomic_features[permutation],
        numbers[permutation],
        species=(8, 46),
    )

    np.testing.assert_allclose(pooled, [2.0, 3.0, 20.0, 30.0])
    np.testing.assert_allclose(pooled, permuted)


def test_frozen_pca95_is_centered_nonwhitened_and_does_not_refit_on_transform():
    runner = _runner()
    matrix = np.array(
        [
            [-10.0, -1.0, 0.0],
            [-5.0, 1.0, 0.0],
            [5.0, -1.0, 0.0],
            [10.0, 1.0, 0.0],
        ]
    )

    fitted = runner.fit_frozen_pca(matrix, variance_fraction=0.95)
    original_mean = fitted.mean.copy()
    transformed = fitted.transform(matrix)
    shifted = fitted.transform(matrix + np.array([100.0, 0.0, 0.0]))

    assert fitted.components.shape == (1, 3)
    assert fitted.retained_variance_fraction >= 0.95
    np.testing.assert_allclose(fitted.mean, original_mean)
    np.testing.assert_allclose(np.diff(transformed[:, 0]), np.diff(shifted[:, 0]))
    assert np.std(transformed[:, 0]) > 1.0


def test_farthest_point_prefix_reduces_covering_radius_and_starts_from_lowest_energy():
    runner = _runner()
    matrix = np.array([[0.0], [1.0], [2.0], [10.0]])
    energies = np.array([0.0, -2.0, -1.0, 1.0])

    order, radii = runner.farthest_point_order(matrix, energies=energies)

    assert order.tolist() == [1, 3, 2, 0]
    assert radii.shape == (4,)
    assert np.all(np.diff(radii) <= 0.0)
    assert radii[-1] == 0.0


def test_archive_growth_audit_reports_arm_proliferation_without_inventing_node_trials():
    runner = _runner()
    rows = [
        {
            "trial_index": 1,
            "seed_entry_id": 0,
            "discovered_entry_id": 1,
            "energy": -1.0,
            "best_energy": -1.0,
        },
        {
            "trial_index": 3,
            "seed_entry_id": 1,
            "discovered_entry_id": 2,
            "energy": -0.5,
            "best_energy": -1.0,
        },
    ]

    audit = runner.archive_growth_audit(rows, completed_trials=4, archive_entries=3)

    assert audit["new_minimum_trial_fraction"] == 0.5
    assert audit["new_minimum_reward_positive_fraction"] == 0.5
    assert audit["productive_starter_ids"] == 2
    assert audit["unobserved_trial_count"] == 2
    assert audit["node_trial_counts_reconstructable"] is False


def test_arm_pressure_audit_quantifies_posterior_data_dilution_without_a_cutoff():
    runner = _runner()

    pressure = runner.arm_pressure_audit(completed_trials=200, archive_entries=195)

    assert pressure["new_arms_per_trial"] == 194 / 200
    assert pressure["trials_per_final_arm"] == 200 / 195
    assert pressure["uniform_observations_per_arm"] == 200 / 195
    assert pressure["beta11_prior_std"] > pressure["uniform_best_case_posterior_std"]
    assert pressure["all_prior_ts_max_mean_reference"] == 195 / 196
    assert pressure["cold_start_fraction_reconstructable"] is False


def test_load_campaign_corpus_pairs_rows_with_xyz_and_assigns_global_best_gain(tmp_path):
    runner = _runner()
    campaign = tmp_path / "c60-seed42"
    minima = campaign / "accepted_minima"
    minima.mkdir(parents=True)
    rows = [
        {
            "trial_index": 1,
            "seed_entry_id": 0,
            "discovered_entry_id": 1,
            "energy": -2.0,
            "best_energy": -2.0,
        },
        {
            "trial_index": 3,
            "seed_entry_id": 1,
            "discovered_entry_id": 2,
            "energy": -1.5,
            "best_energy": -2.0,
        },
    ]
    (campaign / "accepted_structures.jsonl").write_text(
        "\n".join(__import__("json").dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (campaign / "summary.json").write_text(
        __import__("json").dumps(
            {
                "system": "c60",
                "arm": "fixed",
                "completed_trials": 4,
                "archive_entries": 3,
                "initial_energy_eV": -1.0,
                "force_evaluations": 100,
                "wall_time_s": 2.0,
            }
        ),
        encoding="utf-8",
    )
    for row in rows:
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, row["discovered_entry_id"]]])
        write(
            minima
            / (
                f"trial{row['trial_index']:04d}_"
                f"entry{row['discovered_entry_id']:04d}_accepted.xyz"
            ),
            atoms,
        )

    corpus = runner.load_campaign_corpus(campaign)

    assert corpus.system == "c60"
    assert corpus.completed_trials == 4
    assert [record.entry_id for record in corpus.records] == [1, 2]
    assert [record.global_best_gain_eV for record in corpus.records] == [1.0, 0.0]
    assert corpus.records[1].state.n_atoms == 2


def test_representation_metrics_report_energy_continuity_and_full_fps_curve():
    runner = _runner()
    matrix = np.array([[0.0], [1.0], [10.0]])
    energies = np.array([0.0, 1.0, 10.0])

    metrics = runner.representation_metrics(matrix, energies)

    assert metrics["n_samples"] == 3
    assert metrics["dimension"] == 1
    assert metrics["nearest_neighbor_energy_mae_eV"] == 11.0 / 3.0
    assert metrics["pair_distance_energy_gap_spearman"] == 1.0
    assert len(metrics["fps_cover_radius"]) == 3
    assert metrics["fps_cover_radius"][-1] == 0.0


def test_mace_representation_matrix_counts_descriptor_forwards_and_species_pools():
    runner = _runner()

    class FakeCalculator:
        def __init__(self):
            self.calls = 0

        def get_descriptors(self, atoms, invariants_only=True, num_layers=-1):
            assert invariants_only is True
            assert num_layers == -1
            self.calls += 1
            return np.column_stack([atoms.numbers, np.ones(len(atoms))])

    states = (_mixed_state(), _mixed_state())
    calculator = FakeCalculator()

    matrix, timing = runner.mace_representation_matrix(states, calculator)

    assert calculator.calls == 2
    assert matrix.shape == (2, 4)
    np.testing.assert_allclose(matrix[0], [8.0, 1.0, 46.0, 1.0])
    assert timing["descriptor_forward_calls"] == 2
    assert timing["wall_time_s"] >= 0.0


def test_legacy_selector_timing_executes_the_real_full_score_path():
    runner = _runner()

    result = runner.benchmark_legacy_selector((3, 5), descriptor_dimension=4)

    assert [row["archive_entries"] for row in result["measurements"]] == [3, 5]
    assert all(row["wall_time_s"] >= 0.0 for row in result["measurements"])
    assert result["uses_real_bandit_selector"] is True


def test_audit_system_keeps_representation_arms_separate_and_marks_censored_labels(tmp_path):
    runner = _runner()
    state_one = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    state_two = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]]),
    )
    records = (
        runner.MinimumRecord("case", 1, 1, 0, -2.0, -2.0, 1.0, state_one),
        runner.MinimumRecord("case", 2, 2, 1, -1.5, -2.0, 0.0, state_two),
    )
    corpus = runner.CampaignCorpus(
        path=tmp_path,
        system="c60",
        arm="fixed",
        completed_trials=3,
        archive_entries=3,
        force_evaluations=100,
        wall_time_s=2.0,
        records=records,
    )

    class FakeCalculator:
        def get_descriptors(self, atoms, invariants_only=True, num_layers=-1):
            del invariants_only, num_layers
            return np.column_stack(
                [
                    atoms.numbers,
                    np.linalg.norm(atoms.positions, axis=1),
                ]
            )

    report, matrices = runner.audit_system((corpus,), FakeCalculator())

    assert set(matrices) == {"current_rdf", "fixed_species_rdf", "mace_raw", "mace_pca95"}
    assert set(report["representations"]) == set(matrices)
    assert report["data_boundary"]["node_trial_counts_reconstructable"] is False
    assert report["data_boundary"]["unobserved_trial_count"] == 1
    assert report["pca"]["variance_fraction_target"] == 0.95
