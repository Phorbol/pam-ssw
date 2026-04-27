from benchmarks.lj_cluster_compare import run_ssw_trial


def test_lj_benchmark_reports_ssw_diagnostics():
    summary = run_ssw_trial(size=13, seed=0, budget=2)

    assert summary.force_evaluations is not None
    assert summary.force_evaluations > 0
    assert summary.n_minima is not None
    assert summary.duplicate_rate is not None
    assert summary.frontier_nodes is not None
    assert summary.dead_nodes is not None
    assert summary.mean_node_duplicate_failure_rate is not None
    assert summary.max_node_duplicate_failure_rate is not None
    assert summary.direction_choices is not None
    assert summary.direction_candidate_evaluations is not None
    assert summary.direction_mean_candidate_pool_size is not None
