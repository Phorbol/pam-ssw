# PAM-SSW 方向选择机制：理论推导与修复计划

## 一、问题诊断

当前 PAM-SSW 的 `SoftModeOracle` 方向选择机制与原始 SSW 的 biased dimer rotation 存在本质差异，导致 LJ cluster benchmark 中 proposal 高度重复、方向空间坍缩。

### 1.1 原始 SSW 的三个关键特性

**特性 1：初始方向不是纯 softest mode**

原始 SSW 的初始方向是随机全局方向 `N_i^g` 和局部 bond-formation 方向 `N_i^l` 的线性混合：

```
N_i^0 = (N_i^g + λ·N_i^l) / ‖N_i^g + λN_i^l‖
```

其中：
- `N_i^g`：服从类似 300 K Maxwell-Boltzmann 初速度分布的全局随机方向
- `N_i^l`：随机选取两个距离 > 3 Å 的非近邻原子，构造 bond-formation 方向（两原子沿键轴反向运动）
- `λ`：每步从 0.1 逐步增大到 1.0

**特性 2：Biased dimer rotation 带锚点，不是"找最软模"**

原始 biased dimer rotation 的目标函数：

```
n* = argmin_{‖n‖=1} [ ρ(n) + μ_anchor·‖n - N_i^0‖² ]
```

其中 `ρ(n) = n^T H n` 是方向曲率。这不是找最低本征矢（那只需 `argmin ρ(n)`），而是**在接近初始随机方向的前提下尽量选软模**。μ_anchor 控制锚定强度：
- `μ_anchor → 0`：纯软模（方向坍缩到少数 floppy mode）
- `μ_anchor → ∞`：纯随机方向（完全不吸收 PES 信息）
- 中间值：**带锚点的软模**——既有物理意义、每次又不同

**特性 3：局部 bond-formation 方向随机选取，不是人为指定**

对 LJ cluster 无化学键但有接触图。随机选取两个"不处于近接触"的原子（距离 > 阈值），构造 bond-formation 方向——这正好是 cluster 重排需要的原子运动模式。

### 1.2 当前实现的三个缺陷

**缺陷 1：无初始方向混合公式**

当前代码（`SoftModeOracle.choose_direction`，walker.py:328-369）生成独立候选（RANDOM、BOND、MOMENTUM），对每个独立打分后选最高分，不做混合。

**缺陷 2：无 anchor 项，continuity 权重仅 0.1**

当前 `DirectionScorer.score`（walker.py:203-217）：
```python
score = -0.5·σ²·curvature    # 偏好软模
      - 1.0 · damage_risk    # 避免损伤
      - 0.1 · discontinuity  # 连续性惩罚（仅 0.1！）
```

continuity 惩罚与**上一步**的偏离，权重仅 0.1，不锚定到第 0 步的初始混合方向。多步 walk 中方向每步漂移，逐步坍缩。

**缺陷 3：标准 SSW 中 bond_pairs 始终为空**

标准 SSW 仍不使用 local-softening pair 作为 bond candidate。LS-SSW 现在默认通过
`local_softening_mode="neighbor_auto"` 从当前 seed 结构自动生成邻居 pair；`manual`
模式才使用预配置的 `local_softening_pairs`。因此旧版“LS-SSW 只能靠手动 pair”
诊断已经过期，但标准 SSW 的 bond-candidate 空缺仍是独立问题。

---

## 二、理论推导

### 2.1 小步能量展开

在 minimum 附近 `g(q) = ∇U(q) ≈ 0`，沿单位方向 `n` 走步 `σ`：

```
U(q + σn) ≈ U(q) + (1/2)·σ²·n^T H n = U(q) + (1/2)·σ²·ρ(n)
```

低曲率方向能以较小能量代价离开 basin。但如果总是选 `ρ(n)` 最小的方向（类似最低本征矢），从同一 basin 出发每次 proposal 走几乎相同的路径。

### 2.2 锚定正则化 Rayleigh Quotient

设 `G` 为度量矩阵（当前 Euclidean 下 `G=I`）。定义目标函数：

```
L(n; n₀) = ρ(n) + μ_anchor·‖n - n₀‖_G²

约束: ‖n‖_G = 1
```

其中 `n₀` 是初始混合方向。当 `‖n‖_G = 1` 时：

```
‖n - n₀‖_G² = ‖n‖² + ‖n₀‖² - 2⟨n, n₀⟩_G = 2 - 2⟨n, n₀⟩_G
```

所以等价于：

```
n* = argmax_{‖n‖_G=1} [ -ρ(n) + 2μ_anchor·⟨n, n₀⟩_G ]
```

优化问题的物理含义：
- 第一项 `-ρ(n)`：避免高曲率方向（能量代价低）
- 第二项 `⟨n, n₀⟩_G`：投影到锚定方向，保持方向多样性

### 2.3 统一多因素方向选择

将上述推导推广到更一般形式：

```
n* = argmin_{‖n‖_G=1} [
    ρ(n)                                       // 软模性
    + μ_anchor·‖n - n₀‖_G²                     // 锚定到初始混合方向
    + μ_memory·‖n - n_prev‖_G²                 // 路径连续性
    + η_end·⟨n, e_end⟩_G²                      // DESW 端点吸引
    + κ_div·Σ_a |⟨n, n_a⟩_G|²                  // 多 walker 多样性
]
```

当前代码只实现了 `ρ(n)` + 极弱的 `μ_memory`（weight=0.1，且参考上一步而非初始方向），其他全部缺失。

### 2.4 混合方向 λ 的渐进策略

在 walk 的第 `h` 步（总共 `H` 步）：

```
λ_h = λ_start + (λ_end - λ_start) · (h / H)
```

物理含义：
- Walk 初期（λ 小，≈ 0.1）：`N_i^0 ≈ N_i^g`，以随机探索为主，寻找 basin 的任意出口
- Walk 后期（λ 大，≈ 1.0）：`N_i^0` 中 bond-formation 权重增大，引导最终落向特定邻近 basin

### 2.5 Bond-formation 方向对 LJ Cluster 的特殊意义

LJ cluster 没有化学键，但有近邻接触图。定义接触矩阵 `C`：

```
C_ij = 1   if   r_ij ≤ r_contact   (已接触)
C_ij = 0   if   r_ij >  r_contact   (未接触)
```

Cluster 重排的本质：**断旧接触、建新接触**。随机选取 `C_ij = 0` 的原子对 `(i, j)`，构造方向：

```
d_i = -r̂_ij    （原子 i 朝 j 方向移动）
d_j = +r̂_ij    （原子 j 朝 i 方向移动）
d_k = 0        其他原子
```

这个方向直接对应"让两个非接触原子形成新接触"——恰好是 cluster 重排需要的最基本集体运动模式。

对 LJ cluster 的接触阈值建议取 `r_contact = 1.3·σ_LJ ≈ 1.3 Å`（LJ σ=1）。距离大于此值的原子对视为"非接触"。

---

## 三、工程实现方案

### 3.1 改造思路

```
当前流程（有缺陷）:
  生成候选池 → 对每个独立打分(curvature + novelty + continuity) → 选最高分

改造后流程（匹配原始 SSW）:
  Step A: 生成混合初始方向 n₀ = mix(random, bond, λ)
  Step B: 生成候选池（random + bond + soft 候选）
  Step C: 对每个候选计算 curvature ρ(n)
  Step D: 按带锚定评分: score = -ρ(n) - μ·‖n - n₀‖² + ...
  Step E: 选最高分
```

### 3.2 具体改动

#### 改动 1：随机非近邻 bond pair 生成器

文件：`pamssw/walker.py`，`CandidateDirectionGenerator` 类中新增方法：

```python
def _random_non_neighbor_pairs(
    self, state: State, n_pairs: int,
    distance_threshold: float | None = None,
) -> list[tuple[int, int]]:
    """随机选取非近邻原子对，用于 bond-formation 方向"""
    movable = np.where(state.movable_mask)[0]
    if len(movable) < 2:
        return []

    positions = state.positions

    # 自适应阈值：若未指定，取 1.3 × 最近原子间距
    if distance_threshold is None:
        nn_dists = []
        for idx in movable[: min(20, len(movable))]:
            deltas = positions[movable] - positions[idx]
            dists = np.linalg.norm(deltas, axis=1)
            dists = dists[dists > 1e-6]
            if len(dists):
                nn_dists.append(dists.min())
        distance_threshold = 1.3 * float(np.median(nn_dists)) if nn_dists else 1.3

    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 50:
        i, j = self.rng.choice(movable, size=2, replace=False)
        dist = float(np.linalg.norm(positions[j] - positions[i]))
        if dist > distance_threshold:
            pairs.append((int(i), int(j)))
        attempts += 1
    return pairs
```

#### 改动 2：混合初始方向生成

文件：`pamssw/walker.py`，`CandidateDirectionGenerator` 类中新增方法：

```python
def generate_initial_direction(
    self,
    state: State,
    step_index: int,
    max_steps: int,
    lambda_start: float = 0.1,
    lambda_end: float = 1.0,
    n_bond_pairs: int = 2,
) -> np.ndarray:
    """生成原始 SSW 风格混合初始方向 n₀"""
    coords = CartesianCoordinates.from_state(state)

    # N_g: 全局随机方向 (Maxwell-Boltzmann 等价)
    active = self.rng.normal(size=coords.active_size)
    active /= np.linalg.norm(active) + 1e-12
    n_global = coords.full_tangent_from_active(active).values
    n_global = project_out_rigid_body_modes(state, n_global)

    # N_l: 随机非近邻 bond-formation 方向
    pairs = self._random_non_neighbor_pairs(state, n_pairs=n_bond_pairs)
    if pairs:
        pair = pairs[self.rng.integers(0, len(pairs))]
        n_local = self._bond_direction(state, pair[0], pair[1])
        if n_local is None:
            n_local = np.zeros_like(n_global)
    else:
        n_local = np.zeros_like(n_global)

    # 混合: λ 渐进从 lambda_start 到 lambda_end
    frac = step_index / max(max_steps, 1)
    lam = lambda_start + (lambda_end - lambda_start) * frac
    mixed = n_global + lam * n_local
    norm = np.linalg.norm(mixed)
    if norm < 1e-12:
        return self._normalized(n_global)
    return mixed / norm
```

#### 改动 3：DirectionScorer 添加锚定评分项

文件：`pamssw/walker.py`

```python
@dataclass(frozen=True)
class DirectionScorer:
    damage_weight: float = 1.0
    continuity_weight: float = 0.1
    anchor_weight: float = 0.5       # 新增
    novelty_weight: float = 0.5

    def score(
        self,
        curvature: float,
        sigma: float,
        direction: np.ndarray,
        previous_direction: np.ndarray | None,
        anchor_direction: np.ndarray | None,  # 新增
        damage_risk: float,
    ) -> float:
        energy_cost = 0.5 * sigma * sigma * curvature

        discontinuity = 0.0
        if previous_direction is not None:
            prev = previous_direction / (np.linalg.norm(previous_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            discontinuity = float(np.linalg.norm(cur - prev) ** 2)

        anchor_penalty = 0.0
        if anchor_direction is not None:
            anc = anchor_direction / (np.linalg.norm(anchor_direction) + 1e-12)
            cur = direction / (np.linalg.norm(direction) + 1e-12)
            anchor_penalty = float(np.linalg.norm(cur - anc) ** 2)

        return float(
            -energy_cost
            - self.damage_weight * damage_risk
            - self.continuity_weight * discontinuity
            - self.anchor_weight * anchor_penalty
        )

    def score_candidate(
        self,
        state, candidate, curvature, sigma,
        previous_direction, anchor_direction,  # 新增参数
        archive,
    ) -> float:
        score = self.score(
            curvature=curvature, sigma=sigma,
            direction=candidate.direction,
            previous_direction=previous_direction,
            anchor_direction=anchor_direction,
            damage_risk=candidate.damage_risk,
        )
        if archive is None:
            return score
        probe = CartesianCoordinates.from_state(state).displace(
            TangentVector(candidate.direction), sigma
        )
        novelty_gain = archive.coverage_gain(structural_descriptor(probe))
        return float(score + self.novelty_weight * novelty_gain)
```

#### 改动 4：SoftModeOracle.choose_direction 签名扩展

文件：`pamssw/walker.py`

```python
def choose_direction(
    self,
    state: State,
    proposal: ProposalPotential,
    previous_direction: np.ndarray | None,
    anchor_direction: np.ndarray | None = None,  # 新增
    archive=None,
) -> DirectionChoice:
    ...
    for candidate in candidates:
        ...
        score = self.scorer.score_candidate(
            state=state, candidate=candidate,
            curvature=curvature, sigma=sigma,
            previous_direction=previous_direction,
            anchor_direction=anchor_direction,  # 传递
            archive=archive,
        )
        ...
```

#### 改动 5：_walk_candidate_from_seed 生成锚定方向

文件：`pamssw/walker.py`

```python
def _walk_candidate_from_seed(self, seed_state, archive=None,
                               step_target=None) -> State:
    current = seed_state
    previous_direction = None
    anchor_direction = None
    biases = []
    ...

    for step_index in range(self.config.max_steps_per_walk):
        proposal = ProposalPotential(...)

        # 第一步生成锚定方向（整个 walk 保持不变）
        if anchor_direction is None:
            anchor_direction = self.oracle.generator.generate_initial_direction(
                current,
                step_index=step_index,
                max_steps=self.config.max_steps_per_walk,
                lambda_start=self.config.lambda_bond_start,
                lambda_end=self.config.lambda_bond_end,
                n_bond_pairs=self.config.n_bond_pairs,
            )

        choice = self.oracle.choose_direction(
            current, proposal, previous_direction,
            anchor_direction=anchor_direction,
            archive=archive,
        )
        ...
```

#### 改动 6：SSWConfig 新增参数

文件：`pamssw/config.py`

```python
@dataclass(frozen=True)
class SSWConfig:
    ...
    anchor_weight: float = 0.5
    n_bond_pairs: int = 2
    bond_distance_threshold: float | None = None
    lambda_bond_start: float = 0.1
    lambda_bond_end: float = 1.0
```

### 3.3 改动范围汇总

| 文件 | 改动 | 内容 |
|------|------|------|
| `pamssw/walker.py` | 新增方法 | `CandidateDirectionGenerator._random_non_neighbor_pairs()` |
| `pamssw/walker.py` | 新增方法 | `CandidateDirectionGenerator.generate_initial_direction()` |
| `pamssw/walker.py` | 修改 | `DirectionScorer` 加 `anchor_weight`、`anchor_direction` 参数 |
| `pamssw/walker.py` | 修改 | `SoftModeOracle.choose_direction()` 加 `anchor_direction` 参数 |
| `pamssw/walker.py` | 修改 | `_walk_candidate_from_seed()` 生成和传递锚定方向 |
| `pamssw/config.py` | 修改 | `SSWConfig` 添加 5 个新参数（含默认值） |

所有改动在候选生成/打分层面，不增加力评估次数。

---

## 四、正确性保证

### 4.1 可达性不变

修改不改变 baseline proposal 的可达性：随机方向候选仍在候选池中，`anchor_weight` 只是打分偏好而非硬约束。

对应 E-PAM-SSW 理论的 Axiom 3（baseline mixture）：
```
K_E = ε·K₀ + (1-ε)·K_guided
```

### 4.2 不引入额外 force evaluation

- `_random_non_neighbor_pairs()`：几何计算
- `generate_initial_direction()`：线性代数
- 锚定评分项：向量内积

HVP 计算次数不变。

### 4.3 与文档理论一致

thero ry.md §9.2 的正则化 Rayleigh quotient 和更新版形式化理论建议.md §25.3 的 value-guided soft mode 都预留了锚定项。本方案是这些理论的自然实现。

---

## 五、验证方案

### 5.1 单元测试

**方向多样性**：从同一 LJ13 极小值生成 10 个初始方向，验证两两余弦相似度均值 < 0.3。

**Bond pair 选择**：对 LJ13 验证 `_random_non_neighbor_pairs` 确实返回非接触原子对。

**锚定效果**：给定固定 `n₀`，验证 scorer 对有锚定偏离的候选赋更低分。

**双阱回归**：`DoubleWell2D` + seed=7 下 SSW 仍能发现两个极小值。

### 5.2 LJ Cluster Benchmark

运行 `benchmarks/lj_cluster_compare.py --sizes 13 --seeds 0 1 --budget 60`，预期：
- `duplicate_rate` 显著降低
- `direction_selected_bond` > 0
- `best_energy` 至少不差于当前

### 5.3 消融实验

| 版本 | anchor | bond_pairs | λ渐进 | 预期 |
|------|--------|-----------|-------|------|
| A（当前） | 无 | 0 | 无 | baseline |
| B | 0.5 | 0 | 无 | 纯 anchor |
| C | 0.5 | 2 | 0.1→1.0 | anchor+bond |
| D | 1.0 | 3 | 0.1→1.0 | 强 anchor |

预期 duplicate_rate：A > B > C ≈ D；best_energy：D ≈ C > B > A。
