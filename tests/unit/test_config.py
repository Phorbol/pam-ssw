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
    assert config.local_softening_penalty == "buckingham_repulsive"
    assert config.local_softening_xi == 0.3
    assert config.local_softening_cutoff == 2.0
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


def test_config_allows_zero_proposal_relax_steps_only_for_no_relax_ablation():
    assert SSWConfig(proposal_relax_steps=0).proposal_relax_steps == 0

    with pytest.raises(ValueError, match="proposal_relax_steps"):
        SSWConfig(proposal_relax_steps=-1)


def test_config_validates_seed_diversity_limit():
    assert SSWConfig().same_seed_max_consecutive == 3
    assert SSWConfig(same_seed_max_consecutive=None).same_seed_max_consecutive is None
    assert SSWConfig(same_seed_max_consecutive=2).same_seed_max_consecutive == 2

    with pytest.raises(ValueError, match="same_seed_max_consecutive"):
        SSWConfig(same_seed_max_consecutive=0)


def test_config_validates_seed_selection_mode():
    assert SSWConfig().seed_selection_mode == "archive_ucb"
    assert SSWConfig(seed_selection_mode="metropolis_chain").metropolis_temperature == 0.26

    with pytest.raises(ValueError, match="seed_selection_mode"):
        SSWConfig(seed_selection_mode="unknown")
    with pytest.raises(ValueError, match="metropolis_temperature"):
        SSWConfig(metropolis_temperature=0.0)


def test_config_validates_anchor_mixing_alpha():
    assert SSWConfig().anchor_mixing_alpha is None
    assert SSWConfig(anchor_mixing_alpha=0.3).anchor_mixing_alpha == 0.3

    with pytest.raises(ValueError, match="anchor_mixing_alpha"):
        SSWConfig(anchor_mixing_alpha=-0.1)
    with pytest.raises(ValueError, match="anchor_mixing_alpha"):
        SSWConfig(anchor_mixing_alpha=1.1)


def test_config_accepts_deprecated_anchor_candidate_flag_for_compatibility():
    assert SSWConfig().enable_anchor_candidate is False
    assert SSWConfig(enable_anchor_candidate=True).enable_anchor_candidate is True


def test_config_exposes_hvp_and_bias_safety_controls():
    config = SSWConfig(hvp_epsilon=1e-4, bias_weight_min=0.2, bias_weight_max=3.0)

    assert config.hvp_epsilon == 1e-4
    assert config.bias_weight_min == 0.2
    assert config.bias_weight_max == 3.0

    with pytest.raises(ValueError):
        SSWConfig(hvp_epsilon=0.0)
    with pytest.raises(ValueError):
        SSWConfig(bias_weight_min=-1.0)
    with pytest.raises(ValueError):
        SSWConfig(bias_weight_min=2.0, bias_weight_max=1.0)
    with pytest.raises(ValueError):
        SSWConfig(bias_weight_max=0.0)


def test_config_defaults_to_per_atom_rms_step_length_controls():
    config = SSWConfig()

    assert config.step_length_mode == "per_atom_rms"
    assert config.target_step_rms == pytest.approx(0.15)
    assert config.max_step_rms == pytest.approx(0.35)
    assert config.step_rms_scope == "all_atoms"
    assert config.step_active_threshold == pytest.approx(1e-4)


def test_config_validates_step_length_controls():
    assert SSWConfig(step_length_mode="per_atom_rms").step_length_mode == "per_atom_rms"
    assert SSWConfig(step_length_mode="curvature_adaptive").step_length_mode == "curvature_adaptive"

    with pytest.raises(ValueError, match="step_length_mode"):
        SSWConfig(step_length_mode="unknown")
    with pytest.raises(ValueError, match="target_step_rms"):
        SSWConfig(target_step_rms=0.0)
    with pytest.raises(ValueError, match="max_step_rms"):
        SSWConfig(max_step_rms=0.0)
    with pytest.raises(ValueError, match="target_step_rms"):
        SSWConfig(target_step_rms=0.4, max_step_rms=0.3)
    with pytest.raises(ValueError, match="target_step_rms"):
        SSWConfig(target_step_rms=float("nan"))
    with pytest.raises(ValueError, match="max_step_rms"):
        SSWConfig(max_step_rms=float("inf"))
    with pytest.raises(ValueError, match="step_rms_scope"):
        SSWConfig(step_rms_scope="surface")
    with pytest.raises(ValueError, match="step_active_threshold"):
        SSWConfig(step_active_threshold=0.0)
    with pytest.raises(ValueError, match="step_active_threshold"):
        SSWConfig(step_active_threshold=float("nan"))


def test_config_allows_disabling_proposal_coordinate_box():
    config = SSWConfig(proposal_trust_radius=None)

    assert config.proposal_trust_radius is None

    with pytest.raises(ValueError):
        SSWConfig(proposal_trust_radius=0.0)


def test_config_validates_relaxation_optimizers():
    config = SSWConfig(proposal_optimizer="ase-fire", proposal_optimizer_alt="ase-lbfgs", quench_optimizer="ase-lbfgs")

    assert config.proposal_optimizer == "ase-fire"
    assert config.proposal_optimizer_alt == "ase-lbfgs"
    assert config.quench_optimizer == "ase-lbfgs"

    with pytest.raises(ValueError):
        SSWConfig(proposal_optimizer="unknown")
    with pytest.raises(ValueError):
        SSWConfig(proposal_optimizer_alt="unknown")
    with pytest.raises(ValueError):
        SSWConfig(quench_optimizer="unknown")


def test_safe_lbfgs_is_proposal_only_and_opt_in():
    config = SSWConfig(
        proposal_optimizer="safe-lbfgs-total",
        proposal_optimizer_alt="safe-lbfgs-total",
        proposal_duplicate_rescue_optimizer="safe-lbfgs-total",
    )

    assert config.proposal_optimizer == "safe-lbfgs-total"
    assert config.quench_optimizer == "scipy-lbfgsb"
    with pytest.raises(ValueError, match="quench_optimizer"):
        SSWConfig(quench_optimizer="safe-lbfgs-total")


def test_config_validates_direction_curvature_source():
    assert SSWConfig(direction_curvature_source="inner").direction_curvature_source == "inner"
    assert SSWConfig(direction_curvature_source="true").direction_curvature_source == "true"

    with pytest.raises(ValueError, match="direction_curvature_source"):
        SSWConfig(direction_curvature_source="biased")


def test_config_accepts_default_off_choice_aligned_softening():
    config = LSSSWConfig()
    assert config.choice_aligned_softening_enabled is False
    assert config.choice_aligned_softening_cos_threshold == pytest.approx(0.3)


def test_config_accepts_default_off_direction_type_ucb():
    config = SSWConfig()

    assert config.direction_type_ucb_enabled is False
    assert config.direction_type_success_weight == pytest.approx(0.0)
    assert config.direction_type_exploration_weight == pytest.approx(0.1)
    assert config.direction_type_ucb_window == 40


def test_config_accepts_custom_direction_type_ucb_controls():
    config = SSWConfig(
        direction_type_ucb_enabled=True,
        direction_type_success_weight=0.5,
        direction_type_exploration_weight=0.25,
        direction_type_ucb_window=7,
    )

    assert config.direction_type_ucb_enabled is True
    assert config.direction_type_success_weight == pytest.approx(0.5)
    assert config.direction_type_exploration_weight == pytest.approx(0.25)
    assert config.direction_type_ucb_window == 7


def test_config_accepts_default_off_direction_archive_controls():
    config = SSWConfig()

    assert config.direction_archive_enabled is False
    assert config.direction_archive_max_records == 10000
    assert config.direction_archive_success_only is False
    assert config.direction_archive_path is None


def test_config_accepts_default_off_physical_direction_priors():
    config = SSWConfig()

    assert config.random_direction_distribution == "unit_gaussian"
    assert config.enable_bond_form_break_split is False
    assert config.n_bond_formation_pairs == 2
    assert config.n_bond_breaking_pairs == 1
    assert config.bond_formation_max_distance == pytest.approx(4.0)
    assert config.bond_breaking_max_distance == pytest.approx(2.0)


def test_config_validates_physical_direction_priors():
    assert SSWConfig(random_direction_distribution="mass_weighted").random_direction_distribution == "mass_weighted"

    with pytest.raises(ValueError, match="random_direction_distribution"):
        SSWConfig(random_direction_distribution="unknown")
    with pytest.raises(ValueError, match="enable_bond_form_break_split"):
        SSWConfig(enable_bond_form_break_split="yes")
    for field_name in ("n_bond_formation_pairs", "n_bond_breaking_pairs"):
        with pytest.raises(ValueError, match=field_name):
            SSWConfig(**{field_name: -1})
        with pytest.raises(ValueError, match=field_name):
            SSWConfig(**{field_name: True})
    for field_name in ("bond_formation_max_distance", "bond_breaking_max_distance"):
        for value in (0.0, float("nan")):
            with pytest.raises(ValueError, match=field_name):
                SSWConfig(**{field_name: value})


def test_ls_config_validates_choice_aligned_softening_enabled_type():
    with pytest.raises(ValueError, match="choice_aligned_softening_enabled"):
        LSSSWConfig(choice_aligned_softening_enabled="false")


def test_config_accepts_custom_direction_archive_controls():
    config = SSWConfig(
        direction_archive_enabled=True,
        direction_archive_max_records=25,
        direction_archive_success_only=True,
        direction_archive_path="directions.jsonl",
    )

    assert config.direction_archive_enabled is True
    assert config.direction_archive_max_records == 25
    assert config.direction_archive_success_only is True
    assert config.direction_archive_path == "directions.jsonl"


def test_config_rejects_non_bool_direction_archive_flags():
    for field_name in ("direction_archive_enabled", "direction_archive_success_only"):
        for value in (0, 1, "false"):
            with pytest.raises(ValueError, match=field_name):
                SSWConfig(**{field_name: value})


def test_config_rejects_invalid_direction_archive_max_records():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="direction_archive_max_records"):
            SSWConfig(direction_archive_max_records=value)


def test_config_rejects_invalid_direction_archive_path():
    for value in ("", 1):
        with pytest.raises(ValueError, match="direction_archive_path"):
            SSWConfig(direction_archive_path=value)


def test_config_rejects_negative_direction_type_ucb_weights():
    with pytest.raises(ValueError, match="direction_type_success_weight"):
        SSWConfig(direction_type_success_weight=-0.1)
    with pytest.raises(ValueError, match="direction_type_exploration_weight"):
        SSWConfig(direction_type_exploration_weight=-0.1)


def test_config_rejects_nonfinite_direction_type_ucb_weights():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="direction_type_success_weight"):
            SSWConfig(direction_type_success_weight=value)
        with pytest.raises(ValueError, match="direction_type_exploration_weight"):
            SSWConfig(direction_type_exploration_weight=value)


def test_config_rejects_non_bool_direction_type_ucb_enabled():
    for value in (0, 1, "false"):
        with pytest.raises(ValueError, match="direction_type_ucb_enabled"):
            SSWConfig(direction_type_ucb_enabled=value)


def test_config_rejects_invalid_direction_type_ucb_window():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="direction_type_ucb_window"):
            SSWConfig(direction_type_ucb_window=value)


def test_config_validates_choice_aligned_softening_threshold():
    with pytest.raises(ValueError, match="choice_aligned_softening_cos_threshold"):
        LSSSWConfig(choice_aligned_softening_cos_threshold=-1.1)
    with pytest.raises(ValueError, match="choice_aligned_softening_cos_threshold"):
        LSSSWConfig(choice_aligned_softening_cos_threshold=1.1)


def test_config_validates_direction_selection_mode():
    assert SSWConfig().direction_selection_mode == "discrete"
    assert SSWConfig(direction_selection_mode="discrete").direction_selection_mode == "discrete"
    assert SSWConfig(direction_selection_mode="rayleigh_ritz").direction_selection_mode == "rayleigh_ritz"

    with pytest.raises(ValueError, match="direction_selection_mode"):
        SSWConfig(direction_selection_mode="unknown")


def test_config_accepts_regularized_ritz_synthesis_mode():
    assert SSWConfig().direction_synthesis_mode == "none"
    assert SSWConfig(direction_synthesis_mode="regularized_ritz").direction_synthesis_mode == "regularized_ritz"
    assert SSWConfig(regularized_ritz_top_k=3).regularized_ritz_top_k == 3


def test_config_rejects_unimplemented_direction_synthesis_modes():
    with pytest.raises(ValueError, match="direction_synthesis_mode"):
        SSWConfig(direction_synthesis_mode="unknown")
    with pytest.raises(ValueError, match="direction_synthesis_mode"):
        SSWConfig(direction_synthesis_mode="evolution_on_plateau")


def test_config_rejects_ambiguous_ritz_selector_and_synthesis_combo():
    with pytest.raises(ValueError, match="direction_selection_mode"):
        SSWConfig(direction_selection_mode="rayleigh_ritz", direction_synthesis_mode="regularized_ritz")


def test_config_validates_regularized_ritz_top_k():
    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="regularized_ritz_top_k"):
            SSWConfig(regularized_ritz_top_k=value)


def test_config_accepts_default_off_plateau_evolution_controls():
    config = SSWConfig()

    assert config.plateau_evolution_enabled is False
    assert config.plateau_patience_trials == 20
    assert config.plateau_evolution_children == 5
    assert config.plateau_evolution_crossover_pairs == 3
    assert config.plateau_evolution_mutation_count == 2
    assert config.plateau_evolution_history_limit == 10


def test_config_validates_plateau_evolution_controls():
    assert SSWConfig(plateau_evolution_enabled=True).plateau_evolution_enabled is True
    for field_name in (
        "plateau_patience_trials",
        "plateau_evolution_children",
        "plateau_evolution_crossover_pairs",
        "plateau_evolution_mutation_count",
        "plateau_evolution_history_limit",
    ):
        with pytest.raises(ValueError, match=field_name):
            SSWConfig(**{field_name: 0})
        with pytest.raises(ValueError, match=field_name):
            SSWConfig(**{field_name: 1.5})
    with pytest.raises(ValueError, match="plateau_evolution_enabled"):
        SSWConfig(plateau_evolution_enabled="yes")


def test_config_accepts_default_off_archive_escape_momentum_controls():
    config = SSWConfig()

    assert config.archive_escape_momentum_enabled is False
    assert config.archive_escape_momentum_limit == 2
    assert config.archive_escape_momentum_history_limit == 16
    assert config.archive_escape_momentum_same_seed_first is True


def test_config_validates_archive_escape_momentum_controls():
    assert SSWConfig(archive_escape_momentum_enabled=True).archive_escape_momentum_enabled is True
    with pytest.raises(ValueError, match="archive_escape_momentum_enabled"):
        SSWConfig(archive_escape_momentum_enabled="yes")
    with pytest.raises(ValueError, match="archive_escape_momentum_same_seed_first"):
        SSWConfig(archive_escape_momentum_same_seed_first="yes")
    for field_name in ("archive_escape_momentum_limit", "archive_escape_momentum_history_limit"):
        with pytest.raises(ValueError, match=field_name):
            SSWConfig(**{field_name: 0})
        with pytest.raises(ValueError, match=field_name):
            SSWConfig(**{field_name: 1.5})


def test_config_validates_direction_score_sigma_mode():
    assert SSWConfig().direction_score_sigma_mode == "adaptive"
    assert SSWConfig(direction_score_sigma_mode="adaptive").direction_score_sigma_mode == "adaptive"
    assert SSWConfig(direction_score_sigma_mode="trust_scaled").direction_score_sigma_mode == "trust_scaled"
    assert SSWConfig(direction_score_sigma_mode="fixed_reference").direction_score_sigma_mode == "fixed_reference"

    with pytest.raises(ValueError, match="direction_score_sigma_mode"):
        SSWConfig(direction_score_sigma_mode="unknown")
    with pytest.raises(ValueError, match="direction_score_sigma_mode"):
        SSWConfig(direction_score_sigma_mode="curvature_adaptive")


def test_config_accepts_default_off_direction_diagnostics():
    config = SSWConfig(direction_diagnostics_enabled=False)
    assert config.direction_diagnostics_enabled is False
    assert config.direction_diagnostics_path is None


def test_config_requires_path_when_direction_diagnostics_enabled():
    config = SSWConfig(direction_diagnostics_enabled=True, direction_diagnostics_path="trace.jsonl")
    assert config.direction_diagnostics_enabled is True
    assert config.direction_diagnostics_path == "trace.jsonl"

    with pytest.raises(ValueError, match="direction_diagnostics_path"):
        SSWConfig(direction_diagnostics_enabled=True)


def test_config_validates_step_length_controller_controls():
    config = SSWConfig(step_error_tolerance=2.0, step_gamma_down=0.7, step_gamma_up=1.3)
    assert config.step_error_tolerance == 2.0
    assert config.step_gamma_down == 0.7
    assert config.step_gamma_up == 1.3

    with pytest.raises(ValueError, match="step_error_tolerance"):
        SSWConfig(step_error_tolerance=0.0)
    with pytest.raises(ValueError, match="step_gamma_down"):
        SSWConfig(step_gamma_down=0.0)
    with pytest.raises(ValueError, match="step_gamma_up"):
        SSWConfig(step_gamma_up=0.0)


def test_config_rejects_unimplemented_direction_synthesis_controls():
    with pytest.raises(TypeError):
        SSWConfig(soft_mode_lanczos_enabled=True)
    with pytest.raises(TypeError):
        SSWConfig(walk_trace_enabled=True, walk_trace_dir="trace")
    with pytest.raises(TypeError):
        SSWConfig(early_exit_enabled=True)
    with pytest.raises(TypeError):
        SSWConfig(direction_true_curvature_guard_enabled=True)


def test_config_validates_quench_maxiter():
    config = SSWConfig(quench_maxiter=123)
    assert config.quench_maxiter == 123

    with pytest.raises(ValueError, match="quench_maxiter"):
        SSWConfig(quench_maxiter=0)


def test_config_validates_meaningful_escape_controls():
    config = SSWConfig(min_escape_energy_delta=0.2, min_escape_descriptor_delta=0.4, min_escape_novelty=0.3)
    assert config.min_escape_energy_delta == 0.2
    assert config.min_escape_descriptor_delta == 0.4
    assert config.min_escape_novelty == 0.3

    with pytest.raises(ValueError, match="min_escape_energy_delta"):
        SSWConfig(min_escape_energy_delta=-0.1)
    with pytest.raises(ValueError, match="min_escape_descriptor_delta"):
        SSWConfig(min_escape_descriptor_delta=-0.1)
    with pytest.raises(ValueError, match="min_escape_novelty"):
        SSWConfig(min_escape_novelty=-0.1)


def test_config_validates_novelty_probe_scales():
    config = SSWConfig(novelty_probe_scales=(0.5, 1.0, 1.5))
    assert config.novelty_probe_scales == (0.5, 1.0, 1.5)

    with pytest.raises(ValueError, match="novelty_probe_scales"):
        SSWConfig(novelty_probe_scales=())
    with pytest.raises(ValueError, match="novelty_probe_scales"):
        SSWConfig(novelty_probe_scales=(0.5, 0.0, 1.5))


def test_config_validates_trial_progress_feedback_controls():
    config = SSWConfig(
        trial_progress_patience=3,
        trial_progress_boost_factor=1.5,
        trial_progress_max_boost=2.0,
        trial_progress_duplicate_tolerance=0.8,
    )
    assert config.trial_progress_patience == 3
    assert config.trial_progress_boost_factor == 1.5
    assert config.trial_progress_max_boost == 2.0
    assert config.trial_progress_duplicate_tolerance == 0.8

    with pytest.raises(ValueError, match="trial_progress_patience"):
        SSWConfig(trial_progress_patience=-1)
    with pytest.raises(ValueError, match="trial_progress_boost_factor"):
        SSWConfig(trial_progress_boost_factor=1.0)
    with pytest.raises(ValueError, match="trial_progress_max_boost"):
        SSWConfig(trial_progress_max_boost=0.9)
    with pytest.raises(ValueError, match="trial_progress_duplicate_tolerance"):
        SSWConfig(trial_progress_duplicate_tolerance=1.5)


def test_config_validates_stagnation_bond_pair_boost_controls():
    config = SSWConfig(stagnation_bond_pair_boost=3, max_stagnation_bond_pairs=8)
    assert config.stagnation_bond_pair_boost == 3
    assert config.max_stagnation_bond_pairs == 8

    with pytest.raises(ValueError, match="stagnation_bond_pair_boost"):
        SSWConfig(stagnation_bond_pair_boost=-1)
    with pytest.raises(ValueError, match="max_stagnation_bond_pairs"):
        SSWConfig(max_stagnation_bond_pairs=0)


def test_config_validates_proposal_optimizer_alt():
    config = SSWConfig(proposal_optimizer_alt="ase-lbfgs")
    assert config.proposal_optimizer_alt == "ase-lbfgs"
    assert SSWConfig(proposal_optimizer_alt=None).proposal_optimizer_alt is None

    with pytest.raises(ValueError, match="proposal_optimizer_alt"):
        SSWConfig(proposal_optimizer_alt="unknown")


def test_config_validates_proposal_duplicate_rescue_optimizer():
    config = SSWConfig(proposal_duplicate_rescue_optimizer="ase-lbfgs")
    assert config.proposal_duplicate_rescue_optimizer == "ase-lbfgs"
    assert SSWConfig(proposal_duplicate_rescue_optimizer=None).proposal_duplicate_rescue_optimizer is None

    with pytest.raises(ValueError, match="proposal_duplicate_rescue_optimizer"):
        SSWConfig(proposal_duplicate_rescue_optimizer="unknown")


def test_config_validates_energy_sanity_guard():
    assert SSWConfig().max_energy_drop_per_atom == 5.0
    assert SSWConfig(max_energy_drop_per_atom=None).max_energy_drop_per_atom is None
    assert SSWConfig(max_energy_drop_per_atom=2.5).max_energy_drop_per_atom == 2.5

    with pytest.raises(ValueError, match="max_energy_drop_per_atom"):
        SSWConfig(max_energy_drop_per_atom=0.0)


def test_config_validates_search_output_controls():
    assert SSWConfig(accepted_structures_dir="accepted").accepted_structures_dir == "accepted"
    assert SSWConfig(write_proposal_minima=True, proposal_minima_dir="proposals").write_proposal_minima
    assert SSWConfig(
        write_relaxation_trajectories=True,
        relaxation_trajectory_dir="trajectories",
        relaxation_trajectory_stride=5,
    ).write_relaxation_trajectories

    with pytest.raises(ValueError, match="proposal_minima_dir"):
        SSWConfig(write_proposal_minima=True)
    with pytest.raises(ValueError, match="relaxation_trajectory_dir"):
        SSWConfig(write_relaxation_trajectories=True)
    with pytest.raises(ValueError, match="relaxation_trajectory_stride"):
        SSWConfig(relaxation_trajectory_stride=0)


def test_ls_ssw_validates_softening_penalty_controls():
    config = LSSSWConfig(
        local_softening_penalty="buckingham_repulsive",
        local_softening_xi=0.4,
        local_softening_cutoff=2.0,
        local_softening_adaptive_strength=True,
        local_softening_max_strength_scale=2.5,
        local_softening_deviation_scale=0.3,
    )

    assert config.local_softening_penalty == "buckingham_repulsive"
    assert config.local_softening_xi == 0.4
    assert config.local_softening_cutoff == 2.0
    assert config.local_softening_adaptive_strength
    assert config.local_softening_max_strength_scale == 2.5
    assert config.local_softening_deviation_scale == 0.3

    with pytest.raises(ValueError, match="local_softening_penalty"):
        LSSSWConfig(local_softening_penalty="unknown")
    with pytest.raises(ValueError, match="local_softening_xi"):
        LSSSWConfig(local_softening_xi=0.0)
    with pytest.raises(ValueError, match="local_softening_cutoff"):
        LSSSWConfig(local_softening_cutoff=0.0)
    with pytest.raises(ValueError, match="local_softening_max_strength_scale"):
        LSSSWConfig(local_softening_max_strength_scale=0.5)
    with pytest.raises(ValueError, match="local_softening_deviation_scale"):
        LSSSWConfig(local_softening_deviation_scale=0.0)
