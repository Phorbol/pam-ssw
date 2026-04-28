from benchmarks.lj_cluster_compare import run_algorithm_traces, run_all_trials, run_ssw_trial


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
    assert summary.direction_rigid_body_overlap_mean is not None
    assert summary.direction_post_projection_rigid_body_overlap_mean is not None
    assert summary.true_quench_count is not None
    assert summary.true_quench_unconverged is not None
    assert summary.true_quench_max_gradient is not None
    assert summary.proposal_relax_count is not None
    assert summary.proposal_relax_unconverged is not None
    assert summary.proposal_relax_max_gradient is not None


def test_lj_benchmark_accepts_ssw_walk_and_relax_depth_controls():
    shallow = run_ssw_trial(size=13, seed=0, budget=2, steps_per_walk=2, proposal_relax_steps=5)
    deeper = run_ssw_trial(size=13, seed=0, budget=2, steps_per_walk=4, proposal_relax_steps=10)

    assert shallow.direction_choices == 2
    assert deeper.direction_choices == 4
    assert shallow.proposal_relax_mean_iterations <= 5
    assert deeper.proposal_relax_mean_iterations <= 10


def test_lj_benchmark_can_record_energy_traces():
    traces = run_algorithm_traces(
        size=13,
        seeds=[0],
        budget=3,
        ssw_steps_per_walk=2,
        ssw_proposal_relax_steps=5,
    )

    assert {trace.algorithm for trace in traces} == {"ssw", "bh", "ga"}
    for trace in traces:
        assert trace.points[0]["step"] == 1
        assert trace.points[-1]["step"] <= 3
        gaps = [point["energy_gap"] for point in trace.points]
        assert gaps[-1] <= gaps[0]


def test_lj_benchmark_exports_minima_xyz_files(tmp_path):
    runs = run_all_trials(
        sizes=[13],
        seeds=[0],
        budget=2,
        ssw_steps_per_walk=2,
        ssw_proposal_relax_steps=5,
        minima_output_dir=tmp_path,
    )

    assert {run.algorithm for run in runs} == {"ssw", "bh", "ga"}
    files = sorted(tmp_path.glob("*.xyz"))
    assert len(files) == 3
    text = files[0].read_text()
    assert "energy=" in text
    assert "algorithm=" in text
