# PAM-SSW 可配置参数参考

## 快速开始

### Python API

```python
from pamssw import SSWConfig, LSSSWConfig, State, run_ssw, run_ls_ssw, relax_minimum
from pamssw.calculators import ASECalculator
from ase.calculators.lj import LennardJones

state = State(
    numbers=np.array([18, 18, 18]),
    positions=np.array([[0,0,0], [1,0,0], [0,1,0]]),
)
calc = ASECalculator(LennardJones())
result = run_ssw(state, calc, SSWConfig(max_trials=50, rng_seed=42))
```

### CLI

```bash
python -m pamssw run-ssw config.yaml
```

```yaml
state:
  numbers: [18, 18, 18]
  positions: [[0,0,0], [1,0,0], [0,1,0]]
calculator:
  kind: ase
  factory: ase.calculators.lj.LennardJones
search:
  max_trials: 50
  rng_seed: 42
output: result.json
```

---

## `RelaxConfig` — 局部弛豫控制

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `fmax` | 1e-3 | 梯度收敛阈值 (eV/Å)。每个 movable 原子上的最大力分量 < fmax 即收敛 |
| `maxiter` | 200 | L-BFGS-B 最大迭代步数 |

`relax_minimum(state, calculator, RelaxConfig(fmax=..., maxiter=...))` 使用。SSW 内部使用 `quench_fmax` + 固定 400 步做最终淬火，`proposal_fmax` + `proposal_relax_steps` 做中间弛豫。

---

## `SSWConfig` — 所有参数

### 外层搜索控制

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_trials` | int | 12 | 外层 trial 数。每 trial = 选seed → n步walk → quench |
| `rng_seed` | int | 0 | 随机种子 |
| `max_force_evals` | int\|None | None | 力评估预算。None=无上限。超出时优雅终止 |
| `search_mode` | str | `"global_minimum"` | `"global_minimum"` / `"reaction_network"` / `"crystal_search"` |
| `use_archive_acquisition` | bool | True | True=BanditSelector; False=greedy最少访问 |

### 偏置行走核心物理参数

| 参数 | 默认 | 公式角色 |
|------|------|---------|
| `max_steps_per_walk` | 6 | 每 trial 的 bias 步数 H |
| `target_uphill_energy` | 0.6 | σ = √(2·ΔU/|ρ|)·s，其中 s = sigma_scale |
| `target_negative_curvature` | 0.05 | w = σ²·max(ρ+λ⋆, 0) |
| `min_step_scale` | 0.15 | σ 下限 (Å) |
| `max_step_scale` | 1.5 | σ 上限 (Å) |

### 弛豫精度

| 参数 | 默认 | 用途 |
|------|------|------|
| `quench_fmax` | 1e-3 | 最终 quench 收敛标准 (eV/Å) |
| `proposal_fmax` | 2e-2 | 中间 proposal 弛豫收敛标准 |
| `proposal_relax_steps` | 40 | proposal 弛豫最大迭代数 |

### 数值安全

| 参数 | 默认 | 说明 |
|------|------|------|
| `proposal_trust_radius` | 1.5 | 每次 proposal 弛豫中每坐标的 L-BFGS-B bound (Å) |
| `walk_trust_radius` | 4.0 | 单 walk 中原子的最大位移 (Å)。超限→裁剪→终止 walk |
| `fragment_guard_factor` | None | None=禁用。候选构型最近邻距 > factor×种子最近邻距 → 丢弃 |

### 方向选择 (SoftModeOracle)

| 参数 | 默认 | 说明 |
|------|------|------|
| `oracle_candidates` | 12 | 随机方向候选数 |
| `anchor_weight` | 0.5 | 锚定到初始混合方向的强度 μ。score -= μ·‖d-d_anchor‖² |
| `n_bond_pairs` | 2 | 每步随机生成的非近邻 bond-formation 候选数。0=禁用 |
| `bond_distance_threshold` | None | 非近邻判定距离。None=自适应: 1.5×median(最近邻距) |
| `lambda_bond_start` | 0.1 | bond 混合系数的 walk 初始值 |
| `lambda_bond_end` | 1.0 | bond 混合系数的 walk 终值 |

### 结构去重

| 参数 | 默认 | 说明 |
|------|------|------|
| `dedup_rmsd_tol` | 0.1 | 判定为重复的 RMSD 阈值 (Å) |
| `dedup_energy_tol` | 1e-3 | 能量预筛选窗口 (eV)。|ΔE| > tol → 跳过 RMSD 比对 |
| `max_prototypes` | 1000 | 描述符原型池上限 |

### Bandit 种子选择

| 参数 | 默认 | 说明 |
|------|------|------|
| `archive_density_weight` | 0.5 | 描述符密度惩罚 |
| `novelty_weight` | 1.0 | 新颖性奖励 |
| `frontier_weight` | 0.5 | Frontier 节点额外权重 |
| `bandit_exploration_weight` | 0.75 | UCB 探索强度 c |
| `baseline_selection_probability` | 0.15 | 完全随机/最低能的回退概率 |
| `bandit_energy_weight` | 1.0 | 能量惩罚 β_E |

Bandit score(i) = -β·E_norm + w_nov·novelty - w_den·log(1+density) + c·√(log(1+N_total)/(1+N_i)) + w_front·frontier - 10·is_dead

自适应调节：高 duplicate/degeneracy 率时，density_weight 和 frontier_weight 自动缩小。

### Proposal Pool

| 参数 | 默认 | 说明 |
|------|------|------|
| `proposal_pool_size` | 1 | 每 trial 的 proposal 并行数 |

### 搜索输出与轨迹

| 参数 | 默认 | 说明 |
|------|------|------|
| `accepted_structures_log` | None | JSONL 日志路径。每当 archive 接受一个新的 basin，写入 `trial_index`、`seed_entry_id`、`discovered_entry_id`、`energy`、`best_energy`。 |
| `accepted_structures_dir` | None | 被接受的新极小值结构输出目录。设置后，每个 accepted basin 立即写一个 `.xyz`。PdO slab runner 默认设为 `output_dir/accepted_minima`。 |
| `write_proposal_minima` | False | 是否输出每个 trial 的所有 proposal true-minimum，包括 accepted、duplicate、fragment_rejected。默认关闭，避免大量文件。 |
| `proposal_minima_dir` | None | `write_proposal_minima=True` 时的 `.xyz` 输出目录。 |
| `write_relaxation_trajectories` | False | 是否输出优化器完整轨迹，包括 bias PES 上的 proposal short relax 和 true PES 上的 quench/long relax。默认关闭，文件量很大。 |
| `relaxation_trajectory_dir` | None | `write_relaxation_trajectories=True` 时的轨迹 `.xyz` 输出目录。 |
| `relaxation_trajectory_stride` | 1 | 轨迹采样步长。核心默认每步记录；PdO slab runner 默认 `50`，避免 ASE FIRE 轨迹文件过大。 |

---

## `LSSSWConfig` — LS-SSW 额外参数

继承 `SSWConfig` 全部字段，增加：

| 参数 | 默认 | 说明 |
|------|------|------|
| `local_softening_mode` | `"neighbor_auto"` | 局部软化 pair 来源。`neighbor_auto` 从当前 seed 结构自动构建邻居表；`active_neighbors` 只软化方向位移最大的 active atoms 的邻居；`manual` 保留旧的手动 pair 行为。 |
| `local_softening_cutoff_scale` | `1.25` | 自动邻居 cutoff：`scale * (r_cov_i + r_cov_j)`。 |
| `local_softening_active_count` | `None` | `active_neighbors` 模式下最多选取多少个位移最大的 movable atoms；`None` 表示全部 movable atoms。 |
| `local_softening_strength` | `0.6` | 基础 softening 强度。默认全局固定；开启 adaptive strength 后作为基准强度。 |
| `local_softening_pairs` | `[]` | 仅 `manual` 模式使用的 pair 列表 `[(i,j), ...]`。 |
| `local_softening_penalty` | `"buckingham_repulsive"` | pair penalty kernel。可选 `"gaussian_well"` 或 `"buckingham_repulsive"`。 |
| `local_softening_xi` | `0.3` | Buckingham repulsive 的指数衰减长度。 |
| `local_softening_cutoff` | `2.0` | Buckingham 在 `r > r0 + cutoff` 时截断；`None` 表示不截断。 |
| `local_softening_adaptive_strength` | `False` | 是否按当前 `|r-r0|` 放大 pair strength，近似 step-dependent `A_pq^(h)`。默认关闭。 |
| `local_softening_max_strength_scale` | `3.0` | adaptive strength 最大放大倍数。 |
| `local_softening_deviation_scale` | `0.25` | adaptive strength 的距离偏移归一化尺度，单位为 `r0` 的倍数。 |

`gaussian_well` 形式为 `E = Σ strength·exp(-1/2*((r_ij - r0_ij)/tau_ij)^2)`，`tau = max(0.15, 0.25*r0)`。
当前默认 `buckingham_repulsive` 形式为 `E = strength*exp(-(r-r0)/xi)`，可选远程截断。adaptive strength 已实现为可关闭近似：随 `|r-r0|/(local_softening_deviation_scale*r0)` 放大到 `local_softening_max_strength_scale`，但默认关闭。

运行统计：

| 统计项 | 说明 |
|--------|------|
| `local_softening_builds` | 成功构建出非空 local-softening terms 的次数。 |
| `local_softening_terms_built_total` | 已构建 local-softening terms 的累计数量。 |
| `local_softening_terms_total` | 兼容 alias，当前等同于 `local_softening_terms_built_total`。 |
| `local_softening_terms_last` | 最近一次 softening build 的 term 数量；未启用或未生成 term 时为 0。 |

---

## 内部硬编码参数（当前不可配）

| 参数 | 值 | 位置 | 说明 |
|------|----|------|------|
| `continuity_weight` | 0.1 | `DirectionScorer` | 方向连续性惩罚 |
| `damage_weight` | 1.0 | `DirectionScorer` | 损伤风险惩罚 |
| `novelty_weight` (方向) | 0.5 | `DirectionScorer` | 方向新颖性奖励 |
| HVP `epsilon` | 1e-3 | `_directional_curvature` | 有限差分步长 |
| Trust `error_tolerance` | 1.0 | `TrustRegionBiasController` | model_error > 1 触发收缩 |
| Trust `gamma_down` | 0.5 | `TrustRegionBiasController` | 收缩因子 |
| Trust `gamma_up` | 1.15 | `TrustRegionBiasController` | 扩展因子 |
| Trust `min_scale` | 0.25 | `TrustRegionBiasController` | sigma/weight scale 下限 |
| Trust `max_scale` | 2.0 | `TrustRegionBiasController` | sigma/weight scale 上限 |
| Trust `damage_ratio` | 8.0 | `TrustRegionBiasController` | true_delta > 8×denom → damaged |
| true quench `maxiter` | 400 | `SurfaceWalker.relax_true_minimum` | 硬编码 |
| Descriptor `n_bins` | 16 | `rdf_histogram_fingerprint` | RDF bin 数 |
| Frontier novelty 阈值 | 0.4 | `refresh_frontier_status` | novelty ≥ 0.4 判定为 sparse |
| Frontier success 阈值 | 0.1 | `refresh_frontier_status` | success_rate ≥ 0.1 判定为 recently_successful |
| Dead trials 阈值 | 8 | `refresh_frontier_status` | node_trials ≥ 8 方可判定 dead |
| Dead dup_rate 阈值 | 0.75 | `refresh_frontier_status` | duplicate_failure_rate ≥ 0.75 方可判定 dead |
| Dead success 阈值 | 0.05 | `refresh_frontier_status` | success_rate ≤ 0.05 方可判定 dead |
| StepTarget `eta_energy_scale` | 0.2 | `StepTargetController` | archive 能量 scale |
| StepTarget `warmup` | 4 | `StepTargetController` | feedback 适应前的 trial 数 |
