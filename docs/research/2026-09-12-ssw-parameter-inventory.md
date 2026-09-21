# SSW参数配置清单：2026-09-12现场代码

本表由当前工作树dataclass直接读取。必填表示库没有默认值，研究runner必须显式设置。默认值不代表经过跨体系验证的最优值；旧PAM、独立SSW、约束SSW各有配置，尚未统一。

当前12臂真实参数完整保存在 `research/ga_ssw/evidence/constrained-gaussian-{reference,pam}-20260912/plan.json`。

本次LS预淬火fmax=0.03来自runner的外层fmax，rc_reference.py调用prepare时复用了config.fmax。它不是反编译获得的LASP阈值，也不是用户直接指定0.03；是我在用户允许外层0.01–0.05范围内选择的实验操作点。LS预淬火必须完全收敛才继续是Python当前实现规则；已有反编译明确原生还可通过optsoftmax步数上限退出，不能把两者混同。

## pamssw.config.RelaxConfig

来源：`pamssw/config.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `fmax` | 0.001 |
| `maxiter` | 200 |

## pamssw.config.SSWConfig

来源：`pamssw/config.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `max_trials` | 12 |
| `max_steps_per_walk` | 6 |
| `target_uphill_energy` | 0.6 |
| `target_negative_curvature` | 0.05 |
| `quench_fmax` | 0.001 |
| `quench_maxiter` | 400 |
| `quench_optimizer` | 'scipy-lbfgsb' |
| `quench_fallback_optimizer` | None |
| `quench_cell_mode` | 'fixed' |
| `quench_stress_tol` | 0.001 |
| `external_pressure_gpa` | 0.0 |
| `dedup_cell_tol` | 0.001 |
| `dedup_rmsd_tol` | 0.1 |
| `dedup_energy_tol` | 0.001 |
| `rng_seed` | 0 |
| `oracle_candidates` | 12 |
| `proposal_relax_steps` | 40 |
| `proposal_fmax` | 0.02 |
| `proposal_optimizer` | 'ase-fire' |
| `hvp_epsilon` | 0.001 |
| `min_step_scale` | 0.15 |
| `max_step_scale` | 1.5 |
| `bias_weight_min` | 0.0 |
| `bias_weight_max` | 10.0 |
| `proposal_trust_radius` | 1.5 |
| `walk_trust_radius` | 4.0 |
| `fragment_guard_factor` | None |
| `anchor_weight` | 0.5 |
| `anchor_mixing_alpha` | None |
| `continuity_weight` | 0.1 |
| `enable_outcome_gated_continuity` | True |
| `history_push_weight` | 0.1 |
| `enable_momentum_candidate` | True |
| `enable_anchor_candidate` | False |
| `n_bond_pairs` | 2 |
| `random_direction_distribution` | 'unit_gaussian' |
| `enable_bond_form_break_split` | False |
| `n_bond_formation_pairs` | 2 |
| `n_bond_breaking_pairs` | 1 |
| `bond_formation_max_distance` | 4.0 |
| `bond_breaking_max_distance` | 2.0 |
| `stagnation_bond_pair_boost` | 2 |
| `max_stagnation_bond_pairs` | 10 |
| `bond_distance_threshold` | None |
| `lambda_bond_start` | 0.1 |
| `lambda_bond_end` | 1.0 |
| `proposal_pool_size` | 1 |
| `same_seed_max_consecutive` | 3 |
| `use_archive_acquisition` | True |
| `seed_selection_mode` | 'archive_ucb' |
| `metropolis_temperature` | 0.26 |
| `archive_density_weight` | 0.5 |
| `novelty_weight` | 1.0 |
| `novelty_probe_scales` | (1.0,) |
| `frontier_weight` | 0.5 |
| `bandit_exploration_weight` | 0.75 |
| `baseline_selection_probability` | 0.15 |
| `bandit_energy_weight` | 1.0 |
| `search_mode` | <SearchMode.GLOBAL_MINIMUM: 'global_minimum'> |
| `max_prototypes` | 1000 |
| `max_force_evals` | None |
| `accepted_structures_log` | None |
| `accepted_structures_dir` | None |
| `write_proposal_minima` | False |
| `proposal_minima_dir` | None |
| `write_relaxation_trajectories` | False |
| `relaxation_trajectory_dir` | None |
| `relaxation_trajectory_stride` | 1 |
| `direction_curvature_source` | 'inner' |
| `direction_selection_mode` | 'discrete' |
| `block_krylov_blocks` | 2 |
| `block_krylov_depth` | 3 |
| `direction_synthesis_mode` | 'none' |
| `regularized_ritz_top_k` | 5 |
| `direction_score_sigma_mode` | 'adaptive' |
| `direction_type_ucb_enabled` | False |
| `direction_type_success_weight` | 0.0 |
| `direction_type_exploration_weight` | 0.1 |
| `direction_type_ucb_window` | 40 |
| `direction_archive_enabled` | False |
| `direction_archive_max_records` | 10000 |
| `direction_archive_success_only` | False |
| `direction_archive_path` | None |
| `direction_probe_enabled` | False |
| `direction_probe_top_k` | 5 |
| `direction_probe_ds_scale` | 0.5 |
| `direction_probe_uphill_low` | 0.05 |
| `direction_probe_uphill_high` | 1.0 |
| `direction_probe_collision_distance` | 0.5 |
| `plateau_evolution_enabled` | False |
| `plateau_patience_trials` | 20 |
| `plateau_evolution_children` | 5 |
| `plateau_evolution_crossover_pairs` | 3 |
| `plateau_evolution_mutation_count` | 2 |
| `plateau_evolution_history_limit` | 10 |
| `archive_escape_momentum_enabled` | False |
| `archive_escape_momentum_limit` | 2 |
| `archive_escape_momentum_history_limit` | 16 |
| `archive_escape_momentum_same_seed_first` | True |
| `step_length_mode` | 'per_atom_rms' |
| `target_step_rms` | 0.15 |
| `max_step_rms` | 0.35 |
| `step_rms_scope` | 'all_atoms' |
| `step_active_threshold` | 0.0001 |
| `step_error_tolerance` | 1.0 |
| `step_gamma_down` | 0.5 |
| `step_gamma_up` | 1.15 |
| `min_escape_energy_delta` | 0.1 |
| `min_escape_descriptor_delta` | 0.1 |
| `min_escape_novelty` | 1.01 |
| `trial_progress_patience` | 0 |
| `trial_progress_boost_factor` | 1.5 |
| `trial_progress_max_boost` | 2.0 |
| `trial_progress_duplicate_tolerance` | 0.75 |
| `proposal_optimizer_alt` | None |
| `proposal_duplicate_rescue_optimizer` | None |
| `max_energy_drop_per_atom` | 5.0 |
| `direction_diagnostics_enabled` | False |
| `direction_diagnostics_path` | None |

## pamssw.standalone.paper_reference.SSWConfig

来源：`pamssw/standalone/paper_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `width` | **必填** |
| `rotation_bias` | **必填** |
| `max_gaussians` | **必填** |
| `temperature_K` | **必填** |
| `fmax` | **必填** |
| `relax_steps` | **必填** |
| `fd_step` | **必填** |
| `rotation_hvp` | **必填** |
| `rotation_tol` | **必填** |
| `forward_force` | 0.1 |
| `direction_sampling` | 'paper' |
| `rotation_solver` | 'ritz' |
| `cluster_frame` | 'cartesian' |
| `quench_optimizer` | 'ase-lbfgs' |
| `lbfgs_memory` | None |
| `bias_stage_steps` | None |
| `bias_fmax` | None |
| `pre_rotation_hvp` | None |

## pamssw.standalone.constrained_reference.ConstrainedSSWConfig

来源：`pamssw/standalone/constrained_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `width` | **必填** |
| `rotation_bias` | **必填** |
| `temperature_K` | 300.0 |
| `forward_force` | 0.1 |
| `max_gaussians` | 14 |
| `gradient_tol` | 0.005 |
| `fmax` | 0.01 |
| `max_step` | 0.2 |
| `relax_steps` | 300 |
| `fd_step` | 0.0001 |
| `rotation_hvp` | 100 |
| `rotation_tol` | 0.02 |
| `lbfgs_memory` | None |
| `rotation_solver` | 'generalized-dimer' |
| `pre_rotation_hvp` | None |

## pamssw.standalone.paper_reference.LSSettings

来源：`pamssw/standalone/paper_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `bond_energies` | **必填** |
| `bond_lengths` | **必填** |
| `target_per_atom` | **必填** |
| `initial_fraction` | 0.03 |
| `xi` | 0.2 |
| `learning_rate` | 1.8 |
| `energy_filter` | None |

## pamssw.standalone.ls_native_reference.NativeLSSettings

来源：`pamssw/standalone/ls_native_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `bond_energies` | **必填** |
| `bond_lengths` | **必填** |
| `scale` | 5.0 |
| `energy_filter` | None |
| `length_filter` | None |
| `atom_filter` | None |
| `amp_c` | 2.0 |
| `length_tolerance` | 0.1 |
| `target_mev_per_atom` | 20.0 |
| `eta` | 0.005 |
| `max_change` | 0.01 |
| `frequency` | 10 |
| `presteps` | 100 |
| `cycle` | 100 |
| `ratio` | 1.100000023841858 |
| `lselfadapt` | True |
| `bond_geometry` | 'native-mic' |

## pamssw.standalone.pam_gaussian.PAMCurvatureGaussian

来源：`pamssw/standalone/pam_gaussian.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `target_uphill_energy` | 0.6 |
| `target_negative_curvature` | 0.05 |
| `min_width` | 0.15 |
| `max_width` | 1.5 |
| `min_weight` | 0.0 |
| `max_weight` | 10.0 |
| `curvature_floor` | 0.0001 |
| `mode` | 'height_width' |

## pamssw.standalone.paper_ga.PaperGAConfig

来源：`pamssw/standalone/paper_ga.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `quick_steps` | **必填** |
| `generations` | **必填** |
| `generation_steps` | **必填** |
| `fine_steps` | **必填** |
| `ga_candidates` | **必填** |
| `regions` | **必填** |
| `fine_regions` | **必填** |
| `quench_fmax` | **必填** |
| `quench_steps` | **必填** |
| `proposal_max_batches` | **必填** |
| `proposal_max_cut_attempts` | **必填** |
| `proposal_max_pair_attempts` | **必填** |
| `partition_max_draws` | **必填** |
| `projection_tolerance` | **必填** |
| `energy_window` | **必填** |
| `proposal_type` | 3 |
| `proposal_max_insertion_attempts` | 10000 |
| `cycles` | 1 |
| `offspring_steps` | 0 |

## pamssw.standalone.periodic_ga_reference.PeriodicGAConfig

来源：`pamssw/standalone/periodic_ga_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `quick_steps` | **必填** |
| `generations` | **必填** |
| `generation_steps` | **必填** |
| `fine_steps` | **必填** |
| `regions` | **必填** |
| `fine_regions` | **必填** |
| `min_ga` | **必填** |
| `max_batches` | **必填** |
| `max_cut_attempts` | **必填** |
| `max_pair_attempts` | **必填** |
| `partition_max_draws` | **必填** |
| `slots_per_parent` | 100 |
| `cuts_per_slot` | 10 |

## pamssw.standalone.surface_ga_reference.SurfaceGAConfig

来源：`pamssw/standalone/surface_ga_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `quick_steps` | **必填** |
| `generations` | **必填** |
| `generation_steps` | **必填** |
| `fine_steps` | **必填** |
| `regions` | **必填** |
| `fine_regions` | **必填** |
| `min_ga` | **必填** |
| `max_batches` | **必填** |
| `max_cut_attempts` | **必填** |
| `max_pair_attempts` | **必填** |
| `max_face_attempts` | **必填** |
| `max_insertion_attempts` | **必填** |
| `partition_max_draws` | **必填** |
| `auxiliary_evaluations` | 100 |
| `cuts_per_parent_slot` | None |

## pamssw.standalone.vc_reference.VCSSWConfig

来源：`pamssw/standalone/vc_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `strain_length` | **必填** |
| `width` | **必填** |
| `rotation_bias` | **必填** |
| `pressure` | 0.0 |
| `temperature_K` | 300.0 |
| `forward_force` | 0.1 |
| `max_gaussians` | 14 |
| `gradient_tol` | 0.005 |
| `fmax` | 0.01 |
| `stress_tol` | 0.001 |
| `max_step` | 0.2 |
| `relax_steps` | 300 |
| `fd_step` | 0.0001 |
| `rotation_hvp` | 100 |
| `rotation_tol` | 0.02 |
| `lbfgs_memory` | None |
| `bias_release` | 'strict' |

## pamssw.standalone.rc_reference.RCSSWConfig

来源：`pamssw/standalone/rc_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `torsion_length` | **必填** |
| `width` | **必填** |
| `rotation_bias` | **必填** |
| `temperature_K` | 300.0 |
| `forward_force` | 0.1 |
| `max_gaussians` | 14 |
| `gradient_tol` | 0.005 |
| `fmax` | 0.01 |
| `max_step` | 0.2 |
| `relax_steps` | 300 |
| `fd_step` | 0.0001 |
| `rotation_hvp` | 100 |
| `rotation_tol` | 0.02 |
| `lbfgs_memory` | None |

## pamssw.standalone.rc_vc_reference.RCVCSSWConfig

来源：`pamssw/standalone/rc_vc_reference.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `torsion_length` | **必填** |
| `width` | **必填** |
| `rotation_bias` | **必填** |
| `temperature_K` | 300.0 |
| `forward_force` | 0.1 |
| `max_gaussians` | 14 |
| `gradient_tol` | 0.005 |
| `fmax` | 0.01 |
| `max_step` | 0.2 |
| `relax_steps` | 300 |
| `fd_step` | 0.0001 |
| `rotation_hvp` | 100 |
| `rotation_tol` | 0.02 |
| `lbfgs_memory` | None |
| `rotation_length` | **必填** |
| `strain_length` | **必填** |
| `pressure` | 0.0 |
| `stress_tol` | 0.001 |

## pamssw.standalone.block_ssw.BlockSSWConfig

来源：`pamssw/standalone/block_ssw.py`，类型定义/验证器决定单位和适用域。

| 参数 | 库默认/必填 |
|---|---|
| `atomic` | **必填** |
| `quench_length` | **必填** |
| `cell_cycles` | 5 |
| `atomic_period` | 2 |
| `cell_step_fraction` | 0.15 |
| `cell_step_metric` | 'lattice_frobenius' |
| `cell_fd_step` | 0.005 |
| `cell_rotation_requests` | 6 |
| `cell_rotation_force_tol` | 0.1 |
| `partial_atom_steps` | 25 |
| `pressure` | 0.0 |
| `stress_tol` | 0.001 |
| `max_step` | 0.2 |
| `atomic_gaussian_policy` | None |
| `partial_atom_fmax` | None |

## Safe-total内置数值常数

来源：`pamssw/relax.py:192`，约束路径由`generalized_numerics.py`复用。

| 项目 | 值 |
|---|---|
| 历史长度 | 10，lbfgs_memory=None映射到此值；本次未用500 |
| 无历史逆Hessian标量 | 1/70 |
| Armijo c1 | 1e-4 |
| 回溯缩减因子 | 0.5 |
| 单次线搜索最多试探 | 20 |
| 最小alpha常数 | 2^-20；20次试探的最后实际alpha为2^-19 |
| 曲率判据相对系数 | sqrt(machine epsilon) |

GA配置只调度walker与proposal；局部SSW具体参数仍由walker_config显式传入，并非GA有另一个隐藏SSW默认。独立VC/RC仍是后置实验性路径，本清单并不将它们提升为已生产验证。
