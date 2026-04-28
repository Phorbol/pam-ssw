# PAM-SSW 多角度理论分析

## 一、问题本质：势能面上的非凸全局优化

### 1.1 数学形式

给定原子构型的**构型流形** `M`（去除整体平移/转动、等价原子置换后的商空间），真实势能函数：

```
U: M → R
```

维数 `dim(M) = 3N - 6`（非周期）或 `3N`（固定胞周期）。

搜索目标：找到全局最小值 `m* = argmin_{q∈M} U(q)`，并以可接受的计算成本发现尽可能多的 metastable 极小值。

### 1.2 为什么这个问题是"硬"的

**指数级的 basin 数量**：N 原子体系有 `O(e^{αN})` 个局部极小值（Stillinger, 1999）。LJ₇₅ 已知超过 10⁵ 个 distinct minima。枚举不可能。

**barrier 高度的广谱性**：从 ~0.01 eV（浅 basin 间）到 ~10 eV（断键重排）。单一 scale 的扰动策略无法覆盖。

**多重漏斗（multi-funnel）**：PES 不是单连通盆地——存在多个宏观"漏斗"，各自包含大量 minima。漏斗间 barrier 远高于漏斗内。这使 Metropolis 类方法极易困在亚稳漏斗中（如 LJ₃₈ 的 fcc 漏斗 vs icosahedral 漏斗）。

PAM-SSW 的设计正是针对这三个挑战的。

---

## 二、核心数学对象

### 2.1 度量与切空间

PAM-SSW 在构型流形上工作，隐含地使用了 Riemannian 结构：

**度量张量 G(q)**（metric.py）当前允许 Euclidean 和 Mass-weighted：
```
⟨u, v⟩_G = u^T G v
G = I          (Euclidean)
G = diag(m₁,m₁,m₁,...,m_N,m_N,m_N)   (Mass-weighted)
```

**切向量** `TangentVector`（coordinates.py）是流形切空间 `T_qM` 中的元素。切线丛上的范数：
```
‖v‖_G = √(v^T G v)
```

**指数映射的线性近似** `displace()`：
```
exp_q(σ·v) ≈ q + σ·v
```
在 Cartesian 坐标下退化为普通向量加法（固定 cell 时成立）。

**规范方向投影** `project_out_rigid_body_modes`（rigid.py）：将切向量投影到刚体运动子空间的正交补上。对于非周期体系，这个子空间由 3 个平移 + 3 个旋转（或线性分子只有 2 个旋转）张成。

### 2.2 Quench Map——PES 的商空间

PAM-SSW 不直接在连续构型空间上操作，而是在 **basin 的商空间**上工作。关键对象是 **quench map**：

```
Q_U: M → B   (B 是 basin 的集合)
Q_U(q) = argmin_{y∈basin(q)} U(y)
```

`Q_U` 将每个构型映射到它所在的 basin 底部（通过 L-BFGS-B 弛豫）。这是一个**多对一**的映射：每个 basin 是 `M` 中的一个连通开集，其中所有点 quench 到同一个 minimum。

**商空间结构**：PES 被 `Q_U` 划分为 basin 的商集。SSW 在这组 basin 上做 random walk，不直接采样连续的 `M`。

这与 **Stillinger 的 inherent structure 形式化**（Stillinger & Weber, 1982）完全一致：将连续 PES 离散化为 basin 网络。

### 2.3 Basin 转移概率

SSW 从 basin `m_i` 出发，通过 biased walk 到达 `q_H`，然后 quench 到 `m_j`。这定义了一个 transition kernel：

```
K_θ(m_i → m_j) = P[Q_U(Φ^H_{U+B+λP}(m_i; ξ)) = m_j]
```

其中 `Φ^H` 是 H 步 biased walk 算子，`ξ` 包含所有随机性（方向、anchor），`θ` 是所有可调参数。

**关键性质**：这个 kernel **不满足 detailed balance**（不要求对称性 `K(i→j) = K(j→i)`）。PAM-SSW 是**非平衡探索**，不是平衡态采样。这与 metadynamics、umbrella sampling 等增强采样方法本质不同。

---

## 三、软模 Oracle：Rayleigh Quotient 与谱理论

### 3.1 经典 Rayleigh Quotient

在 quadratic basin 近似下，Hessian `H = ∇²U` 是实对称矩阵。Rayleigh quotient：

```
ρ(v) = (v^T H v) / (v^T G v),   v ∈ T_qM
```

`ρ(v)` 的取值范围是 Hessian 的广义特征值区间 `[λ_min, λ_max]`。最小值在最低本征方向处取得，最大值在最高本征方向处取得。

在 minimum 处（`∇U=0`），二阶 Taylor 展开：
```
U(q + σn) ≈ U(q) + ½σ²·n^T H n = U(q) + ½σ²·ρ(n)
```

低曲率方向的能量代价低——这解释了为什么 basin escape 应沿软模方向。

### 3.2 PAM-SSW 的正则化 Rayleigh Quotient

PAM-SSW 不取纯最低本征方向，而是解**带锚定的正则化问题**：

```
n* = argmin_{‖n‖_G=1} [ ρ(n) + μ·‖n - n₀‖_G² ]
```

其中 `n₀` 是随机混合方向（全局随机 + bond-formation）。展开锚定项：

```
‖n - n₀‖² = 2 - 2⟨n, n₀⟩
```

等价于最大化投影到 `n₀` 的软模性：

```
n* = argmax_{‖n‖_G=1} [ -ρ(n) + 2μ·⟨n, n₀⟩ ]
```

**μ = 0**：纯最低本征方向 → 方向空间坍缩到少数 floppy mode（同一 basin 每次 proposal 走相同路径）

**μ → ∞**：纯随机方向 → 完全不吸收 PES 信息（回到朴素的 basin hopping）

**0 < μ < ∞**：**带锚点的软模**——PAM-SSW 的工作区间。μ=0.5（默认 anchor_weight）居中。

### 3.3 与 Dimer Method 的关系（Henkelman & Jónsson, 1999）

Dimer method 的旋转目标也是找最低曲率方向，使用 dimer 旋转来优化 `ρ(n)`。它与 PAM-SSW 的候选打分有本质区别：

| | Dimer | PAM-SSW 候选打分 |
|---|-------|-----------------|
| 优化方式 | 连续旋转（梯度下降在球面上） | 离散选择（从有限候选池中选最高分） |
| 初始依赖 | 从初始方向出发，收敛到最近局部极小 | 候选池覆盖全空间，不依赖初始 |
| 每步力评估 | ~6-10 次旋转迭代 × 2 次力评估 | oracle_candidates 次 × 2 次力评估 |
| 是否锚定 | 无（纯曲率最小化） | 有（anchor penalty） |
| 输出质量 | 精确最低本征方向 | 近似软模方向 |

PAM-SSW 的离散候选方法更接近原始 SSW 的 biased CBD rotation——"在保持初始方向信息的前提下尽量软"——而非 dimer 的纯最低模搜索。

### 3.4 与 Lanczos 方法的关系

若未来实现 Lanczos（开发手册 §7.4），则是：

```
Lanczos(H, v₀, k) → 前 k 个近似本征值和本征方向
```

Lanczos 在 Krylov 子空间 `K_k(H, v₀) = span{v₀, Hv₀, ..., H^{k-1}v₀}` 中构造 Rayleigh quotient 的极小化子。与 PAM-SSW 当前候选方法的关系：

- **候选打分**：从候选池（随机生成 + bond + soft）中选最优 → trade exploration with softness
- **Lanczos**：从单一初始方向出发的 Krylov 子空间中找最优 → 可能陷入局部

两者的结合（Lanczos 提供高质量候选进入候选池）是自然的下一步。

---

## 四、Gaussian Bias：非平衡势能变形

### 4.1 与 Metadynamics 的关系（Laio & Parrinello, 2002）

两者都在势能面上沉积 Gaussian 凸起来驱动探索，但机制根本不同：

| | Metadynamics | PAM-SSW |
|---|-------------|---------|
| **自由度** | 全维（或选定的 CVs）沉积 Gaussian | 每步**单方向**（1D投影）沉积 Gaussian |
| **时间尺度** | 跨整个模拟，bias 累积直至填平 FES | 每 trial 重建（bias 不跨 trial 保留） |
| **目标** | 重构自由能面 F(s) | 生成跨 basin proposal |
| **收敛** | bias 收敛到 -F(s) | 无需收敛（bias 是临时工具） |
| **接受准则** | 无需 Metropolis | Bandit 选择（非 MH） |

数学上，metadynamics 在 CV 空间沉积的 bias 是：

```
V_G(s, t) = Σ_{τ < t} W_τ × exp(-|s - s(τ)|² / (2δs²))
```

PAM-SSW 在完整 3N 空间中沉积的 bias 是：

```
B(q) = Σ_{k=1}^H w_k × exp(-(⟨q - q_k, n_k⟩_G)² / (2σ_k²))
```

关键区别：metadynamics 的 Gaussian 分布在 CV 空间的所有方向，PAM-SSW 的 Gaussian 仅在**一个** 3N 维方向上变化（directional Gaussian）。这是 **dimensionality reduction**——从 `O(e^D)` 的维数灾难降低到 `O(H)` 的控制复杂度。

### 4.2 与 OPES 的关系（Invernizzi & Parrinello, 2020）

OPES 通过在线估计 probability density 来构造 bias：

```
V(s) = -(1/β) log(p^tg(s) / P(s))
```

PAM-SSW 吸收了 OPES 的一个核心思想——**在线密度估计**——但将其降维为 outer-loop 控制信号：

```
ρ_t(z) = Σ_i a_i K(z - z_i)    (archive descriptor density)
V_arch = α log(ε + ρ_t(s))     (archive repulsion, 默认不进 inner loop)
```

这样保留了 OPES 的"密度驱动探索"思想，但不承担其"收敛 FES"的计算负担。

### 4.3 与 Trust Region 方法的关系（Conn, Gould, Toint, 2000）

PAM-SSW 的 `TrustRegionBiasController` 是经典优化理论中 trust region 方法的直接应用，但作用于**非凸、非二次**的势能面。

经典 trust region：

```
min_d m_k(d) = f_k + g_k^T d + ½ d^T B_k d
s.t. ‖d‖ ≤ Δ_k
```

PAM-SSW 的 adaptation：

```
predicted_delta = ½ σ² · ρ   (quadratic model on true PES)
true_delta = U(q_{new}) - U(q_old)   (measured on true PES)
model_error = |true_delta - predicted_delta| / (|predicted_delta| + ε)
```

若 model_error > tolerance(1.0) 或 damaged：
```
σ_scale ← σ_scale × γ_down   (0.5)
w_scale  ← w_scale  × γ_down
```
否则：
```
σ_scale ← σ_scale × γ_up     (1.15)
w_scale  ← w_scale  × γ_up
```

这等价于 adaptive trust region radius——只是不是调整显式的 Δ，而是调整 bias 的 σ 和 w，间接控制 modified PES 的变形程度。

---

## 五、Bandit 选择：多臂老虎机与势能面探索

### 5.1 MAB 形式化

将已发现的 minima `{m_1, ..., m_K}` 视为 K 个"臂"，每 trial 选一个臂（seed）后拉一次 SSW walk（proposal）。目标：最大化累计 discovery reward / 力评估成本。

PAM-SSW 的 `BanditSelector` 使用 **Upper Confidence Bound (UCB)** 风格评分：

```
S_i = -β_E·E_norm(i) + w_nov·novelty(i) - w_den·log(1+density(i))
    + c·√(log(1+N_total)/(1+N_i)) + w_front·frontier(i) - 10·is_dead(i)
```

标准 UCB1（Auer, 2002）只有 `r̄_i + c·√(log N/N_i)`。PAM-SSW 将其推广为**上下文 bandit**（contextual bandit），上下文包含：
- Energy（偏好的区域）
- Structure novelty（未探索的结构空间）
- Archive density（避免过采样）
- Frontier score（启发式前沿价值）

### 5.2 与 Bayesian Optimization 的关系

PAM-SSW 的 bandit + archive 密度机制可以视为 **Bayesian Optimization (BO) 在高维流形上的离散近似**：

- BO：用 Gaussian Process 建模 `U(q)` 的不确定性，通过 acquisition function（EI, UCB）选下一个评估点
- PAM-SSW：用 archive + kernel density 建模结构空间的覆盖度，通过 bandit 评分选下一个 seed

PAM-SSW 的 acquisition function 的隐式形式：

```
α(m_i) = -E_i + λ_N·novelty(m_i) + λ_ψ·ψ_i + exploration_bonus
```

这与 BO 的 UCB acquisition `α(x) = -μ(x) + κ·σ(x)` 结构同构——`-E_i ≈ -μ`（已知最小值替代 GP 均值），`novelty + exploration ≈ σ`（密度稀疏替代 GP 方差）。

区别：BO 的 GP 代理模型是全局平滑的；PAM-SSW 的密度模型是离散的（基于实际访问的 minima）。BO 更"主动学习"，PAM-SSW 更"经验驱动"。

### 5.3 与 Simulated Annealing / Parallel Tempering 的关系

这些方法依赖 Boltzmann 分布 `p(x) ∝ e^{-βU(x)}` 和 Metropolis 接受准则。PAM-SSW **刻意放弃** detailed balance：

- 不需要 convergence to equilibrium（不声称 canonical sampling）
- 不需要 temperature schedule（bandit 替代）
- 允许"宏观"跳跃到任意 seed（不对局部 MC 移动负责）

这与 **Wales 的 basin-hopping**（Wales & Doye, 1997）最接近——都是 basin 商空间上的探索。区别在于 proposal 生成方式：basin-hopping 用随机 Cartesian 位移，PAM-SSW 用 biased soft-mode walk。

---

## 六、结构描述符与信息几何

### 6.1 Kernel Density Estimation

Archive 的 density 估计（archive.py:89-112）使用自适应带宽的高斯核密度：

```
ρ(z) = Σ_j w_j × K_h(z - z_j)
K_h(u) = exp(-½|u|²/h²)
h = max(0.25, median(prototype_distances))
```

**自适应带宽**的关键性质：当 prototype 稀疏时，h 大（kernel 宽），单一新结构的 density 贡献被稀释 → novelty 高 → 鼓励探索；当 prototype 密集时，带宽自适应缩小 → 更好地分辨密集区域的细微结构差异。

### 6.2 与 Fisher Information Metric 的隐含联系

在密度估计的视角下，descriptor 空间中的**局部密度梯度**指向"未被充分探索"的方向：

```
∇_z log ρ(z) = -(1/ρ) Σ_j w_j (z-z_j)/h² K_h(z-z_j)
              = weighted mean shift toward unexplored regions
```

这是 **mean shift 算法**（Fukunaga & Hostetler, 1975; Cheng, 1995）的核心公式。PAM-SSW 中 `novelty_gain` 的用法隐含地在方向选择中施加了 mean-shift-like 的 bias——偏好指向稀疏区域的位移方向。

### 6.3 Descriptor Degeneracy 与压缩感知

Descriptor degeneracy（不同 minima 映射到相同描述符 bin）本质上是一个**压缩感知/信息瓶颈**问题：

```
编码: q → s(q) ∈ R^20  (3N → 20维)
```

20 维描述符无法单射地编码3N维构型空间。degeneracy rate `D_deg` 测量了这种压缩损失。当 `D_deg` 高时，`AcquisitionPolicy.effective()` 自动降低 density/frontier weight——承认描述符不足以可靠驱动探索策略，退回到更保守的随机选择。

这与 **rate-distortion theory** 中的 tradeoff 同构：低维描述符节省了密度估计的计算成本（rate 低），但引入了信息损失（distortion 高）。

---

## 七、与机器学习/强化学习的深层联系

### 7.1 Policy Gradient 视角

将 seed 选择 `i_t = argmax S_i` 视为 policy `π_θ(i_t | archive_t)`。若定义了 reward `r_t`（discovery value），则：

```
∇_θ J = E[ Σ_t r_t · ∇_θ log π_θ(i_t | archive_t) ]
```

PAM-SSW 的 adaptive policy（`AcquisitionPolicy.effective()`）是一个**启发式 policy gradient**：不显式计算梯度，而是根据 `duplicate_rate` 和 `descriptor_degeneracy_rate` 在线调节权重。这类似于 **population-based training**（Jaderberg et al., 2017）中基于在线统计量的超参数自适应。

### 7.2 与 Curiosity-Driven Exploration 的关系

Pathak et al. (2017) 的 curiosity-driven RL 使用 forward dynamics model 的预测误差作为 intrinsic reward。PAM-SSW 的 `novelty_gain` 是 curiosity 的静态版本——不依赖动力学模型，而是依赖 archive density 的高斯核估计。

### 7.3 Frontier 评分与最优实验设计

Frontier scoring 的公式（archive.py:204-207）：

```
frontier_score = 0.4/(1+N_i) + 0.3·novelty + 0.2·energy_score + 0.1·success_rate
```

这可以理解为**多目标优化问题**的标量化（使用权重向量 `[0.4, 0.3, 0.2, 0.1]`），各目标为：
- Visit 最少（尚未探索）
- 结构最稀疏（descriptor novelty）
- 能量最低（接近 GM）
- 成功率最高（productive seed）

这与 **multi-objective Bayesian optimization** 中的 scalarization approach 结构一致——不同在于 PAM-SSW 用启发式统计量替代 GP 替代模型。

---

## 八、算法的数学本质：一个统一视角

### 8.1 将 PAM-SSW 视为广义 Langevin 动力学的离散化

标准 underdamped Langevin：

```
dq = M^{-1} p dt
dp = -∇U(q) dt - γ p dt + √(2γ/β) dW
```

PAM-SSW 可视为一种**非平衡、非物理的广义 Langevin 过程**：

```
Δq ≈ -σ²/(2·ΔU_target) · (∇U + ∇B(q; H, {n_k}, {σ_k}, {w_k})) · Δt_effective
    + anchor_noise(n₀)

quench: q → Q_U(q)   (basin 投影)
```

关键差异：
- 扩散项不是白噪声 `dW`，而是结构化的 `anchor_noise(n₀)`（混合随机 + bond-formation）
- 漂移项不是纯 `-∇U`，而是 `-(∇U + ∇B)`（bias-modified gradient）
- 使用确定性 quench `Q_U` 投影到 basin 底部（相当于无限快的 dissipative relaxation）
- 无 temperature 控制（非平衡）

### 8.2 形式化的"搜索信息论"

可以将整个搜索过程建模为一个信息获取问题。定义：

- `H_t = -Σ_i p_t(i) log p_t(i)`：basin 分布的熵（搜索不确定性）
- `I_t`：已获取的关于 PES 结构的信息

搜索目标：最大化信息获取效率 `dI/d(force_evals)`。

Bandit 选择的 UCB 项 `√(log N/N_i)` 是 **entropy reduction bound** 的优化实现——选择不确定（高熵、少访问）的臂最大化预期信息获取。

Archive density repulsion 是**主动信息获取**——刻意指导搜索走向 descriptor 空间中的高熵（稀疏）区域。

这与 **infotaxis**（Vergassola et al., 2007）和 **entropy search**（Hennig & Schuler, 2012）的核心思想一致：将搜索建模为信息最大化，而非能量最小化。

---

## 九、与文献中相关方法的定位图

```
                    物理正确性 →
                (detailed balance, canonical sampling)
    高 ↑
      │  Metadynamics        Umbrella Sampling
      │  (OPES)              (WHAM)
      │
      │  Parallel Tempering   Transition Path Sampling
      │  (REMD)
      │
      │  ════════════════════════════════════════
      │  PAM-SSW 定位线: 探索效率 > 物理正确性
      │  ════════════════════════════════════════
      │
      │  Basin Hopping        Minima Hopping
      │  (Wales)              (Goedecker)
      │
      │  SSW (Liu)            PAM-SSW (本工作)
      │  ★ 方向锚定          ★ bandit + archive
      │  ★ biased dimer      ★ 自适应 trust region
      │                       ★ E-PAM-SSW 密度驱动
      │
      │  Random Search        Genetic Algorithm
      │  (纯蒙特卡洛)          (DE, USPEX)
      │
    低 ┼────────────────────────────────────────→
      低          势能面信息利用程度          高
                (gradient/Hessian usage)
```

---

## 十、小结：算法设计的五个数学支柱

| 支柱 | 数学工具 | PAM-SSW 中的体现 |
|------|---------|-----------------|
| **微分几何** | Riemannian 度量、切空间、法丛 | `Metric`, `TangentVector`, 规范模投影 |
| **谱理论** | Rayleigh quotient、Lanczos、HVP | `SoftModeOracle`, `_directional_curvature` |
| **非平衡统计力学** | Quench map、basin 商空间、Gaussian bias | `Q_U`, `SurfaceWalker`, `ProposalPotential` |
| **在线学习/决策** | Multi-armed bandit、UCB、contextual bandit | `BanditSelector`, `AcquisitionPolicy` |
| **信息几何/密度估计** | KDE、自适应带宽、mean shift | `MinimaArchive`, `descriptor_density`, `novelty` |

PAM-SSW 的独特性在于它**不追求任何一个支柱的"最优解"**（如 Lanczos 精确本征对、MH 准则下的严格采样、GP-based BO），而是**在五个支柱上各取精华的工程权衡**——使其在保持理论一致性的同时具备实际的计算效率。

---

## 十一、核心文献谱系与算法演进

### 11.1 SSW 方法族谱系（复旦大学刘智攀课题组）

| 年份 | 方法 | 论文 | 核心贡献 | PAM-SSW 继承 |
|------|------|------|---------|-------------|
| 2010 | **CBD** | Constrained Broyden Dimer method (Shang & Liu, J. Chem. Theory Comput.) | 不显式构建 Hessian 的二阶 saddle point 优化器 | SoftModeOracle 的 HVP 有限差分曲率计算 |
| 2012 | **BP-CBD** | Bias Potential CBD (Shang & Liu) | 在 CBD 上叠加 basin-filling bias，使体系能从 basin 走出 | GaussianBiasTerm 的核心机制——"抬高盆地+沿方向软化" |
| 2013 | **SSW** | Stochastic Surface Walking Method for Structure Prediction and Pathway Searching (Shang & Liu, J. Chem. Theory Comput., DOI: 10.1021/ct301010b) | 随机软模驱动的跨盆地 proposal 生成，biased dimer rotation 锚定随机方向 | `_walk_candidate_from_seed()` 的整体结构，锚定到混合初始方向 |
| 2013 | **DESW** | Double-Ended Surface Walking (Shang & Liu) | 从两个端点同时出发的双端行走，endpoint attraction 约束 | **未实现**——PAM-SSW 路线图 v0.3 |
| 2014 | **SSW-crystal** | Variable-cell SSW crystal (Shang & Liu, Phys. Chem. Chem. Phys., DOI: 10.1039/c4cp01485e) | 原子+晶胞的广义坐标，stress-consistent gradient | `variable_cell_supported: 0`——明确声明未实现 |
| 2015 | **VC-DESW** | Variable-Cell Double-Ended Surface Walking (Shang & Liu) | 变胞双端行走，固相相变路径 | **未实现** |
| 2015 | **SSW-RS** | SSW Reaction Sampling (Shang & Liu) | SSW + DESW 组合成反应采样 pipeline | **未实现** |
| ~2022 | **LASP/SSW-NN** | Global neural network potential with SSW (Shang, Liu et al., npj Comput. Mater., DOI: 10.1038/s41524-022-00959-5) | SSW 大规模生成训练数据 → NN PES → 大体系探索 | PAM-SSW 的 ASE calculator 抽象层支持任意 MLP |
| 2024 | **LS-SSW** | Local Softening SSW (Shang, Liu et al., J. Chem. Theory Comput., DOI: 10.1021/acs.jctc.4c01081) | 自动局部邻居软化强局域振动模式，论文形式包含 step-dependent pair strength | `LocalSofteningModel` 实现 automatic neighbor generation、direction-aware `active_neighbors`、Gaussian/Buckingham 可插拔 kernel，以及默认关闭的 adaptive-strength 近似；生产默认仍保持 Gaussian fixed-strength，是否切换默认需基于 auto-pair ablation |

### 11.2 跨领域关联文献

| 领域 | 关键工作 | 与 PAM-SSW 的关联 |
|------|---------|-----------------|
| **Saddle point search** | Henkelman & Jónsson (1999), "A dimer method for finding saddle points on high dimensional potential surfaces." J. Chem. Phys. 111, 7010 | Dimer rotation 是 SoftModeOracle 候选打分的理论基础——但 PAM-SSW 用离散选择替代了连续旋转 |
| **Basin hopping** | Wales & Doye (1997), "Global optimization by basin-hopping and the lowest energy structures of Lennard-Jones clusters." J. Phys. Chem. A 101, 5111 | PAM-SSW 的 basin 商空间结构直接继承自 basin-hopping——proposal kernel 是主要改进 |
| **Minima hopping** | Goedecker (2004), "Minima hopping: An efficient search method for the global minimum." J. Chem. Phys. 120, 9911 | Bell-Evans-Polanyi 原理驱动的 escape attempts——PAM-SSW 用 soft-mode oracle 替代 BEP 猜测 |
| **Metadynamics** | Laio & Parrinello (2002), "Escaping free-energy minima." PNAS 99, 12562 | Dimensional Gaussian vs full-CV-space Gaussian 的关键区别使 PAM-SSW 避免维数灾难 |
| **OPES** | Invernizzi & Parrinello (2020), "Rethinking Metadynamics." J. Phys. Chem. Lett. 11, 2731 | Archive-density anti-revisit 机制吸收了 OPES 的在线密度估计思想，但降维为 outer-loop control |
| **OPES-Explore** | Invernizzi & Parrinello (2022), "Exploration vs Convergence Speed." J. Chem. Theory Comput. 18, 1070 | PAM-SSW 明确站在"exploration > convergence"一侧——不要求 FES 收敛 |
| **Trust region** | Conn, Gould & Toint (2000), "Trust-Region Methods." SIAM | `TrustRegionBiasController` 是 trust-region 在 bias 强度控制上的直接应用 |
| **UCB1 Bandit** | Auer, Cesa-Bianchi & Fischer (2002), "Finite-time Analysis of the Multiarmed Bandit Problem." Machine Learning 47, 235 | `BanditSelector.score_entry()` 中的 `c·√(log N/N_i)` 项精确匹配 UCB1 |
| **Inherent structures** | Stillinger & Weber (1982), "Hidden structure in liquids." Phys. Rev. A 25, 978 | Quench map `Q_U` 将连续 PES 离散化为 basin 网络——PAM-SSW 的形式化基础 |
| **Stillinger decomposition** | Stillinger (1999), "A topographic view of supercooled liquids and glass formation." Science 267, 1935 | `O(e^{αN})` 个 minima 的指数增长——PAM-SSW 设计假定了这个硬度 |
| **Kernel density estimation** | Silverman (1986), "Density Estimation for Statistics and Data Analysis." Chapman & Hall | `descriptor_density()` 的自适应带宽高斯 KDE 的统计基础 |
| **Mean shift** | Fukunaga & Hostetler (1975), "The estimation of the gradient of a density function." IEEE Trans. Inform. Theory 21, 32 | Novelty gain 方向选择的 mean-shift 隐含偏向 |
| **Information bottleneck** | Tishby, Pereira & Bialek (1999), "The information bottleneck method." | Descriptor degeneracy 的信息论等价：低维编码的 rate-distortion tradeoff |
| **Curiosity-driven RL** | Pathak et al. (2017), "Curiosity-driven Exploration by Self-Supervised Prediction." ICML | Architecture novelty 的 intrinsic reward 与 RL curiosity 的平行结构 |
| **Entropy search** | Hennig & Schuler (2012), "Entropy Search for Information-Efficient Global Optimization." JMLR 13, 1809 | Archive 密度驱动的探索与 entropy search 的信息最大化同构 |
| **Bayesian Optimization** | Shahriari et al. (2016), "Taking the Human Out of the Loop: A Review of Bayesian Optimization." Proc. IEEE 104, 148 | PAM-SSW 的 bandit + archive 可视为高维流形上的离散 BO 近似——GP 被 KDE 替代 |
| **Kinetic Monte Carlo** | Voter (2007), "Introduction to the Kinetic Monte Carlo Method." In Radiation Effects in Solids, Springer | PAM-SSW 的未来方向：将 SSW 的 transition kernel 校正后用于 kMC 动力学模拟 |
| **String method** | E, Ren & Vanden-Eijnden (2007), "Simplified and improved string method." J. Chem. Phys. 126, 164103 | PAM-SSW 的 biased walk trajectory 提供 candidate path，可后接 string/NEB 精修 |
