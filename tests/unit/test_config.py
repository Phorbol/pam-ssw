from pamssw.config import LSSSWConfig, SSWConfig


def test_configs_keep_high_level_defaults_only():
    ssw = SSWConfig()
    ls = LSSSWConfig(local_softening_pairs=[(0, 1)])

    assert ssw.max_trials > 0
    assert ssw.target_uphill_energy > 0.0
    assert ls.local_softening_strength > 0.0
    assert ls.local_softening_pairs == [(0, 1)]
