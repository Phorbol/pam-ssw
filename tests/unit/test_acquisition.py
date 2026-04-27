import numpy as np

from pamssw.acquisition import AcquisitionPolicy, BanditSelector, ProposalOutcome, ProposalScorer, SearchMode
from pamssw.archive import MinimaArchive
from pamssw.state import State


def _state(offset: float) -> State:
    return State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [offset, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset, 1.0, 0.0],
                [offset, 0.0, 1.0],
            ]
        ),
    )


def test_archive_density_increases_when_nearby_descriptor_points_are_added():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=1e-3)
    first = archive.add(_state(0.0), -4.0, parent_id=None)
    density_one = archive.descriptor_density(first)

    archive.add(_state(0.03), -3.9, parent_id=first.entry_id)
    density_two = archive.descriptor_density(first)

    assert density_two > density_one


def test_descriptor_degeneracy_counts_bins_with_multiple_distinct_minima():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=1e-6)
    first = archive.add(_state(0.0), -4.0, parent_id=None)
    second = archive.add(_state(0.2), -3.8, parent_id=first.entry_id)
    third = archive.add(_state(5.0), -3.5, parent_id=second.entry_id)

    first.descriptor = np.array([0.01, 0.01])
    second.descriptor = np.array([0.02, 0.02])
    third.descriptor = np.array([1.0, 1.0])

    assert archive.descriptor_degeneracy_rate(bin_width=0.1) == 0.5


def test_bandit_selector_prefers_novel_low_density_underexplored_node():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=1e-3)
    crowded = archive.add(_state(0.0), -5.0, parent_id=None)
    novel = archive.add(_state(5.0), -4.9, parent_id=None)
    archive.add(_state(0.05), -4.8, parent_id=crowded.entry_id)

    crowded.node_trials = 8
    crowded.node_successes = 0
    crowded.frontier_value = 0.0
    novel.node_trials = 0
    novel.node_successes = 0
    novel.frontier_value = 1.0

    selector = BanditSelector(
        policy=AcquisitionPolicy(
            archive_density_weight=1.0,
            novelty_weight=1.0,
            frontier_weight=1.0,
            exploration_weight=0.5,
            baseline_probability=0.0,
        )
    )

    assert selector.select(archive, np.random.default_rng(4)).entry_id == novel.entry_id


def test_policy_lowers_archive_density_weight_when_degeneracy_is_high():
    policy = AcquisitionPolicy(archive_density_weight=1.0, frontier_weight=1.0)

    effective = policy.effective(duplicate_rate=0.0, descriptor_degeneracy_rate=0.75)

    assert effective.archive_density_weight < policy.archive_density_weight
    assert effective.frontier_weight < policy.frontier_weight


def test_proposal_scorer_penalizes_duplicates_and_rewards_novel_energy_improvements():
    scorer = ProposalScorer()
    duplicate = ProposalOutcome(
        energy=-4.0,
        previous_best_energy=-4.5,
        is_new_minimum=False,
        is_duplicate=True,
        descriptor_coverage_gain=0.0,
    )
    novel = ProposalOutcome(
        energy=-5.0,
        previous_best_energy=-4.5,
        is_new_minimum=True,
        is_duplicate=False,
        descriptor_coverage_gain=0.4,
    )

    assert scorer.score(novel) > scorer.score(duplicate)


def test_global_mode_rank_key_is_lexicographic_not_weight_tuned():
    scorer = ProposalScorer.for_mode(SearchMode.GLOBAL_MINIMUM)
    small_best_gain = ProposalOutcome(
        energy=-4.51,
        previous_best_energy=-4.5,
        is_new_minimum=False,
        is_duplicate=True,
        descriptor_coverage_gain=0.0,
    )
    high_coverage_no_best_gain = ProposalOutcome(
        energy=-4.4,
        previous_best_energy=-4.5,
        is_new_minimum=True,
        is_duplicate=False,
        descriptor_coverage_gain=100.0,
    )

    assert scorer.rank_key(small_best_gain) > scorer.rank_key(high_coverage_no_best_gain)


def test_reaction_network_mode_prioritizes_validated_edges_over_best_energy():
    scorer = ProposalScorer.for_mode(SearchMode.REACTION_NETWORK)
    new_edge = ProposalOutcome(
        energy=-4.4,
        previous_best_energy=-4.5,
        is_new_minimum=False,
        is_duplicate=False,
        descriptor_coverage_gain=0.0,
        is_new_edge=True,
    )
    lower_energy_no_edge = ProposalOutcome(
        energy=-5.0,
        previous_best_energy=-4.5,
        is_new_minimum=True,
        is_duplicate=False,
        descriptor_coverage_gain=0.0,
        is_new_edge=False,
    )

    assert scorer.rank_key(new_edge) > scorer.rank_key(lower_energy_no_edge)
