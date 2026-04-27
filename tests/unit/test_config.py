from pamssw.config import LSSSWConfig, SSWConfig
import pytest


def test_configs_keep_high_level_defaults_only():
    ssw = SSWConfig()
    ls = LSSSWConfig(local_softening_pairs=[(0, 1)])

    assert ssw.max_trials > 0
    assert ssw.target_uphill_energy > 0.0
    assert ssw.max_prototypes > 0
    assert not hasattr(ssw, "cluster_reseed_interval")
    assert not hasattr(ssw, "proposal_pool_size")
    assert ls.local_softening_strength > 0.0
    assert ls.local_softening_pairs == [(0, 1)]


def test_config_exposes_only_documented_search_modes():
    SSWConfig(search_mode="global_minimum")

    with pytest.raises(ValueError):
        SSWConfig(search_mode="lj_cluster_fast_path")


def test_config_rejects_empty_archive_prototype_budget():
    with pytest.raises(ValueError):
        SSWConfig(max_prototypes=0)
