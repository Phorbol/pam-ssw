import numpy as np
import pytest


def test_singletons_preserve_original_order_before_feature_filter():
    from pamssw.standalone.population import partition
    rows = [{'energy': 2., 'sims': []}, {'energy': 1., 'sims': []}]
    regions = partition(rows, 2, np.random.default_rng(1))
    assert regions == [[0], [1]]


def test_regions_only_use_first_three_projections_and_sort_energy():
    from pamssw.standalone.population import partition
    rows = [dict(energy=e, sims=[x, 0, 0, 1e8 * i])
            for i, (e, x) in enumerate([(2, 0), (1, .1), (4, 10), (3, 10.1)])]
    assert partition(rows, 2, np.random.default_rng(4)) == [[1, 0], [3, 2]]


def test_region_cap_retains_elite_and_distinct_members():
    from pamssw.standalone.population import partition
    rows = [dict(energy=float(i), sims=[0., 0., 0.]) for i in range(30)]
    region, = partition(rows, 1, np.random.default_rng(8))
    assert len(region) == len(set(region)) == 20
    assert region[0] == 0
    assert region == sorted(region)


def test_region_score_uses_population_variance_of_selected_members():
    from pamssw.standalone.population import rank_regions
    rows = [dict(energy=e) for e in [0., 2., 1., 1.]]
    result = rank_regions(rows, [[0, 1], [2, 3]])
    assert result[0].indices == (0, 1)
    assert result[0].score == pytest.approx(.1)
    assert result[1].score == pytest.approx(.8)


@pytest.mark.parametrize('indices', [[-1], [True], [0, 0], [2], [0.5]])
def test_invalid_region_members_are_not_reinterpreted(indices):
    from pamssw.standalone.population import rank_regions
    with pytest.raises(ValueError, match='indices'):
        rank_regions([dict(energy=0.), dict(energy=1.)], [indices])
