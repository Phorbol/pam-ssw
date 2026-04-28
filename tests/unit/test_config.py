from pamssw.config import LSSSWConfig, SSWConfig
import pytest


def test_configs_keep_high_level_defaults_only():
    ssw = SSWConfig()
    ls = LSSSWConfig(local_softening_pairs=[(0, 1)])

    assert ssw.max_trials > 0
    assert ssw.target_uphill_energy > 0.0
    assert ssw.max_prototypes > 0
    assert ssw.proposal_pool_size == 1
    assert ssw.archive_density_weight == 0.5
    assert ssw.baseline_selection_probability == 0.15
    assert not hasattr(ssw, "cluster_reseed_interval")
    assert ls.local_softening_strength > 0.0
    assert ls.local_softening_pairs == [(0, 1)]


def test_ls_ssw_defaults_to_neighbor_auto_mode():
    config = LSSSWConfig()

    assert config.local_softening_mode == "neighbor_auto"
    assert config.local_softening_cutoff_scale == 1.25
    assert config.local_softening_active_count is None
    assert config.local_softening_pairs == []


def test_ls_ssw_manual_mode_keeps_legacy_pairs():
    config = LSSSWConfig(local_softening_mode="manual", local_softening_pairs=[(0, 1)])

    assert config.local_softening_mode == "manual"
    assert config.local_softening_pairs == [(0, 1)]


def test_ls_ssw_active_neighbors_mode_is_accepted():
    config = LSSSWConfig(local_softening_mode="active_neighbors")

    assert config.local_softening_mode == "active_neighbors"


def test_ls_ssw_positive_active_count_is_accepted():
    config = LSSSWConfig(local_softening_active_count=3)

    assert config.local_softening_active_count == 3


def test_ls_ssw_rejects_invalid_softening_mode():
    with pytest.raises(ValueError, match="local_softening_mode"):
        LSSSWConfig(local_softening_mode="unknown")


def test_ls_ssw_rejects_invalid_neighbor_parameters():
    with pytest.raises(ValueError, match="local_softening_cutoff_scale"):
        LSSSWConfig(local_softening_cutoff_scale=0.0)
    with pytest.raises(ValueError, match="local_softening_active_count"):
        LSSSWConfig(local_softening_active_count=0)


def test_ls_ssw_rejects_invalid_softening_strength():
    with pytest.raises(ValueError, match="local_softening_strength"):
        LSSSWConfig(local_softening_strength=0.0)


def test_ls_ssw_rejects_invalid_softening_pairs():
    with pytest.raises(ValueError, match="local_softening_pairs"):
        LSSSWConfig(local_softening_pairs=[(1, 1)])
    with pytest.raises(ValueError, match="local_softening_pairs"):
        LSSSWConfig(local_softening_pairs=[(0, 1, 2)])


def test_config_exposes_only_documented_search_modes():
    SSWConfig(search_mode="global_minimum")

    with pytest.raises(ValueError):
        SSWConfig(search_mode="lj_cluster_fast_path")


def test_config_rejects_empty_archive_prototype_budget():
    with pytest.raises(ValueError):
        SSWConfig(max_prototypes=0)


def test_config_rejects_invalid_proposal_pool_size():
    with pytest.raises(ValueError):
        SSWConfig(proposal_pool_size=0)


def test_config_exposes_hvp_and_bias_safety_controls():
    config = SSWConfig(hvp_epsilon=1e-4, bias_weight_max=3.0)

    assert config.hvp_epsilon == 1e-4
    assert config.bias_weight_max == 3.0

    with pytest.raises(ValueError):
        SSWConfig(hvp_epsilon=0.0)
    with pytest.raises(ValueError):
        SSWConfig(bias_weight_max=0.0)


def test_config_allows_disabling_proposal_coordinate_box():
    config = SSWConfig(proposal_trust_radius=None)

    assert config.proposal_trust_radius is None

    with pytest.raises(ValueError):
        SSWConfig(proposal_trust_radius=0.0)


def test_config_validates_relaxation_optimizers():
    config = SSWConfig(proposal_optimizer="ase-fire", quench_optimizer="ase-lbfgs")

    assert config.proposal_optimizer == "ase-fire"
    assert config.quench_optimizer == "ase-lbfgs"

    with pytest.raises(ValueError):
        SSWConfig(proposal_optimizer="unknown")
    with pytest.raises(ValueError):
        SSWConfig(quench_optimizer="unknown")
