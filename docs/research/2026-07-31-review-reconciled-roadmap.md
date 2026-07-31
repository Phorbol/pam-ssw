# PAM-SSW 审阅后研究路线：先辨识 escape kernel，再扩动作支持

> 本路线以根目录 `2026-07-31-review.md` 为新的审阅输入，并以
> `feature/direction-continuation-ablation@680ad13` 的代码与实验记录为证据基线。
> 它替代“立即训练 D0/K4 context classifier”的旧优先级，但不改变已经验证的生产配置。
> 审阅源文件 SHA256：
> `8821fc2d37c34cf84efc15a8bda0d9b1eb15f60c2ad79ed96d7ab4d36ee1a829`。

## 执行状态更新（2026-07-31）

本路线的 R1 与唯一触发的 R2-N 分支已经完成：

- R1-B 的两次完整 K4 回放均出现 C60/PdO 质量符号翻转，因此关闭
  “删除 all-candidate HVP”与 live selected-only-HVP 主张
  （`runs/20260731-hvp-value-of-information-replay/`）。
- R1-A 完成 24/24 generation paths、59 个可达 checkpoint quenches，
  总计 11,454/15,000 FE、`unattributed=0`。R2-N 闭环后标签为
  31 `ESCAPED_CERTIFIED`、20 `RETURN_STARTER`、8 `INVALID_GEOMETRY`；
  early-escape→H8-return context 与 D0/K4 action-support-gap context 均为 0
  （`runs/20260731-current-action-first-passage/`、
  `runs/20260731-pdo-matcher-numerical-gate/`）。
- 五个 PdO matcher ambiguity 中，四个是现有全局 descriptor collision；
  一个 residual pair 经严格双端点重淬火后仍保持 0.278076 eV 能差。闭环额外消耗
  77 FE（SciPy strict 53 FE；ASE-LBFGS certificate closure 24 FE），没有修改
  matcher、optimizer 或阈值默认值。
- 现有 K4 support 的 exact order-statistic 表明 unbiased B2 在 PdO 具有正
  benefit-cost elasticity，但 C60 两次均低于 1，因此关闭 full-action live B2
  （`runs/20260731-action-breadth-order-statistic-gate/`）。
- 同 starter 的八个 K4 central-FD 请求可安全做 MACE graph batching：
  batch 2/4/8 在 C60 分别加速 1.96/3.17/3.64 倍，在 PdO 加速
  1.70/2.75/2.96 倍；384 FE 账本闭合且 HVP 误差通过冻结容差
  （`runs/20260731-k4-hvp-batch-force-gate/`）。
- 最小集成门已经完成：实际 `SoftModeOracle` 在 C60/PdO 的 initial 与 momentum
  上下文中完成 20 组 serial/batch 配对，全部选择同一原子位移方向；C60 中位
  加速 2.590/2.443 倍，PdO 为 1.869/2.016 倍。含预热共 384 FE，
  `unattributed=0`（`runs/20260731-k4-hvp-batch-integration-gate/`，
  `feature/direction-continuation-ablation@99aa55a`）。

这个工程后续已经关闭：显式 `MACEBatchCalculator` 只批量执行同一状态的八个
中心差分构型，普通 `ASECalculator` 和生产默认值不变。它只能降低 wall time，
不能表述为 FE 或搜索质量提升。R2-S、R2-H、新物理 action、full-action B2 与
posterior 均未满足准入条件；不得自动展开。

- CuO 第三体系暴露出独立于 selector 的 proposal-relaxation 瓶颈：当前
  `local_softening_scope=both` 将 38 项、参考点一阶力非零的指数排斥 LS 势持续
  加入坐标传播。固定首个 proposal 上，float32 Safe-LBFGS 262 FE 后线搜索失败，
  float64 跑满 300 步/781 FE 仍有 0.508 eV/A 最大力；只移除 proposal LS 后
  59 FE 收敛到 0.0385 eV/A。硬 cutoff、双精度和 FIRE 均未解决底层 modified-PES
  收敛问题（`runs/20260731-cuo-safe-lbfgs-line-search-root-cause/`）。
- 随后的共享 bootstrap、Metropolis starter、每臂 20,000 FE CuO 门只将 scope
  从 `both` 改为 `oracle`。oracle-only 将 proposal 失败率从 97.37% 降到
  56.02%，proposal FE 从 17,153 降到 15,173，完整 macro trials 从 18 增至
  32，最低能低 0.168 eV，且两臂 `unattributed=0`。结合已有 C60/PdO scope
  固定起点中位结果，删除 proposal LS 晋级为显式生产候选；既有命名 C60 profile
  因绑定旧 200-step provenance 不静默修改，全局 `LSSSWConfig` 默认也不改变
  （`runs/20260731-cuo-oracle-scope-production-gate/`）。
- 随后的 G-LS0A 四算符门控已经替代直接长任务 `oracle` 对 `none`。同一批 C60
  frozen candidates 上，当前 exponential operator 在 33/36 个方向中加硬，中位
  `+0.7620`；预应变后的真实 PES 在 36/36 个方向中软化，中位 `-1.7651`，但
  A/B/C/D 在 9/9 blocks 中都给出相同 selected candidate 和完整候选排序。因而
  不开放完整 action 门或 paper scalar-strength feedback；在出现新的、物理上不同且
  action-level 可重复的 family 前，仍不开发 pair 规则、posterior、TS/UCB 或
  quadratic propagator（`runs/20260731-ls-four-operator-gate/`）。
- 随后的 G-UP0 用现有 C60 proposal-relax 轨迹的 frame 0 构造了严格配对的
  `explicit displacement → true quench` 反事实。34 对中有 5 对只在 biased-PES
  relax 后逃出 starter，且在 plateau/D0/h1 与 plateau/K4/h2 两个完整 context 中
  都达到 2/3 seeds；另有 3 对在两臂都逃逸时落入不同 minima。新增 2,886 FE 全部
  属于 true-PES check/quench/validation，direction HVP、biased relax 与 unattributed
  均为 0。因此关闭“删除 proposal relax”的路线，但不晋级当前 80-step 长度或自适应
  控制；下一唯一可准入问题是沿已有 optimizer frames 确定最早发生 basin-label 改变的
  relaxation first passage（`runs/20260731-uphill-relax-counterfactual-gate/`）。
- G-UP1 随后逐一 true-quench 了四条重复因果轨迹的全部 238 个 Safe-LBFGS
  accepted frames。四条轨迹“稳定进入最终 basin”的步数为 45/80、15/49、
  23/57、12/48；首次 escape 与稳定进入最终 basin 可相差 32 步，证明 biased
  relaxation 会先跨越并重定向多个吸引域，而不是单调放大位移。冻结的 45-step
  cutoff 在四条 holdout 上 4/4 保持最终 basin，但只有两条原轨迹长于 45；因此按
  预注册的 4/4 positive-headroom 规则不晋级、不改生产默认。20,562 个新增 FE 中
  20,098 属于 true quench，direction/bias replay/unattributed 均为 0。探索性地，
  将 45 解释为最大步数 cap 会把八条记录的 481 accepted steps 降到 357，但这不
  等于 FE 节省；下一步若继续，只允许在未见 action 上做 `natural convergence`
  对 `max_steps=45` 的完整 action 总成本配对，不能引入自适应 controller
  （`runs/20260731-uphill-relax-first-passage-gate/`）。

## 一、重新组织后的总判断

当前首要科学问题不是 starter selector、TS 或 UCB 的形式，而是：

1. 现有 action 是否曾进入另一个 basin 的吸引域；
2. 若进入过，后续 cumulative bias / proposal relax 是否又把它带回；
3. 若从未进入，付费 HVP 是在扩大 escape support，还是只在归一化和重排一组无效候选；
4. terminal label 是否被 quench failure 或 matcher ambiguity 污染。

因此后续依赖关系应改为：

```text
已有证据对账
    ↓
当前 D0/K4 escape first-passage + HVP value-of-information
    ↓
按失败类型只打开一个机制分支
    ├─ 中途逃逸后返回 → 离散 horizon/stop gate
    ├─ 从未逃逸       → 最简单 support-expansion control
    ├─ invalid/quench  → 数值路径修复
    └─ ambiguous label → matcher gate
    ↓
稳定、物理上不同的 action portfolio
    ↓
单 GPU ForceService / batch execution
    ↓
完整 action propensity 与一个 posterior allocation gate
```

这一路线的核心不是构造一个“更高级的 SSW”，而是用最少的新自由度回答：

\[
\text{support}\rightarrow\text{propagation}\rightarrow
\text{label}\rightarrow\text{allocation}.
\]

上游未辨识时，不优化下游。

生产边界保持不变：长任务仍参考
`pam/experiment/pdo-k4-k8-fixed-budget@91217d3`。本路线只在
`feature/direction-continuation-ablation` 研究分支上建立诊断和 survivor gates；
任何新 action、matcher 或 posterior 都不会因一次固定状态实验直接进入生产默认。

## 二、对新审阅建议的技术裁决

| 审阅建议 | 裁决 | 原因 |
|---|---|---|
| checkpoint-quench first-passage 最高优先 | **接受但缩窄** | 已有 C60 block-Krylov 12 trajectories / 68 checkpoints，未观察到 categorical overshoot；只补当前 D0/K4 和 PdO |
| selected-only HVP | **先零 FE 回放** | D0 单 HVP对 K4 已跨体系翻转；先计算 HVP ranking 的实际 value of information，回放不过则不新跑 |
| G0 全面重构 action/matcher/log | **拆成最小诊断层与后置平台层** | 大改 `walker.py` 会先增加工程耦合；当前 gate 只需 run-local action trace、明确 termination 和 ambiguous label |
| 同时比较 7 个 action arms | **拒绝** | 会把方向、传播器、horizon、MD 参数和局域化重新耦合，无法归因 |
| failed-route orthogonal | **条件接受** | 仅当同一 starter 上出现可重复的 `return starter`，才测试一个正交补 arm |
| localized ART / bowl breakout | **延后** | 物理动机成立，但 active region、负曲率确认、Lanczos 更新和 MMF 同时进入，不能作为第一新 arm |
| kick+quench / basin hopping control | **优先于 ART** | 是判断 cumulative Gaussian、HVP 和 serial relax 是否值得的最简单正交 control |
| short MD / minima hopping control | **kick control 后再决定** | timestep、步数和 kinetic level 会增加新的自由度 |
| Luby horizon schedule | **不进入当前阶段** | H8/H14 已显示混合收益，且 PAM action 不满足 universal restart 的原假设 |
| Gaussian 改为多个 escape levels | **暂缓** | `H/range/Fmax/active radius` 加有限状态切换仍是一组新启发式；只在 first-passage 指向力度/范围不足后开放 |
| 约束变分 / quadratic direction | **暂缓** | 数学更干净，但 exact anchor、energy-bounded combination 等邻近方案尚无 terminal 增益 |
| ForceService | **保留为后置独立工程 gate** | 先确定未来主要流量是 all-candidate HVP 还是异步 relax；batch 只能降 wall time，不能降 FE |
| posterior / Bayesian SSW | **后移** | 现有 D0/D1/K4 context-free transfer gate 已失败；只有新 portfolio 出现稳定可重复 family 差异才重开 |
| CEM、WE、AMS、MAP-Elites、生成模型 | **退出当前路线** | 都依赖稳定 action、score、strata 或高吞吐，目前条件不成立 |
| PR、lint、type、artifact 治理 | **单独维护线** | 重要但不应阻塞下一轮科学辨识，也不能冒充算法进展 |

## 三、已经回答的问题，不重复实验

### 1. checkpoint overshoot 不是完全未知

`runs/20260728-direction-conditioned-checkpoint-shooting/` 已完成：

- C60；
- intermediate / plateau 两个 starter；
- seeds 42–44；
- balanced/deep block-Krylov；
- 12 条 trajectory、68 个严格 checkpoint landing。

结果为：

- `productive_earlier_and_final = 5`；
- `overshoot = 0`；
- `no_productive_checkpoint = 7`。

因此不能再把“现有所有失败主要来自越过了好 checkpoint”当作开放假设。尚未回答的是：

- 当前生产相关的 D0 与 K4，而不是旧 block-Krylov；
- PdO；
- “逃出 starter”而不要求 landing 同时更低的 first-passage 标签。

### 2. horizon 不是单调质量旋钮

`runs/20260731-uphill-horizon-gate/` 的五个完整 H8/H14 pairs 为：

- H14 更低 2；
- 更高 1；
- 同 basin 2；
- 多付 1,087 FE。

所以不扫描 H，不引入连续 stop controller，也不因一次长路径成功就移除 walk radius。

### 3. D0 已经是 selected-only-HVP 的重要对照

最新 action-family gate 中：

- D0 每次 direction selection 只付一个 central HVP；
- K4 每次付四个 central HVP；
- C60 平均 terminal landing：K4 优于 D0；
- PdO：D0 略优于 K4；
- context winners 发生体系与 starter 翻转。

这已经否定“单 HVP exact anchor 普遍替代 K4”。新问题只剩：

> 不使用 HVP ranking、但保留 K4 的零 FE support/diversity，是否比当前 static scorer 更好？

### 4. 更软、更深和短期更猛均已关闭为通用规则

- static scorer 在 12 个 shared K4 pools 中 10 次错过更好 terminal candidate；
- min true curvature 仅在 PdO 稳定一些，伤害 C60；
- H2 larger-rise prediction 为 C60 `0/6`、PdO `2/6`；
- deeper/block/anchor Krylov 改善 Rayleigh proxy，但未稳定改善 landing。

下一阶段不再开发新的 curvature scalar score。

## 四、阶段 R1：补齐当前 action 的 escape-kernel 诊断

### R1-A：D0/K4 checkpoint first-passage

**假设**

当前平台可能来自三类不同机制：

- D0/K4 在 H8 内从未进入不同 basin；
- 早期已经逃出，后续传播返回 starter；
- 路径有效，但 terminal quench / geometry label 失败。

**冻结矩阵**

- systems：C60、固定底层 PdO；
- starters：现有 locked `intermediate_accepted`、`plateau_accepted`；
- seeds：42、43、44；
- arms：`D0_exact_anchor`、`K4_discrete`；
- checkpoints：`h = 1, 2, 4, 8`；
- trajectory count：`2 × 2 × 3 × 2 = 24`；
- 最大 checkpoint quenches：96；
- Gaussian、LS、proposal optimizer、quench 和 geometry guard 完全沿用
  `20260731-action-family-transfer-gate`；
- 不加入 D1、posterior、starter selector、ART、MD 或新 bias level。

**实现边界**

复用 `runs/20260728-direction-conditioned-checkpoint-shooting/run_audit.py`
的 trajectory extraction 和独立 checkpoint quench。新建：

```text
runs/20260731-current-action-first-passage/
  PLAN.md
  protocol.py
  run_gate.py
  analyze.py
  conclusion.md
  .gitignore
tests/unit/test_current_action_first_passage.py
```

不在 `pamssw/walker.py` 中添加 callback 或新生产参数。

**标签**

每个 checkpoint 保留连续能量、结构 descriptor、证书和 FE，并给出：

```text
RETURN_STARTER
ESCAPED_CERTIFIED
AMBIGUOUS_MATCH
INVALID_GEOMETRY
FRAGMENTED
QUENCH_UNCONVERGED
BUDGET_EXHAUSTED
```

“逃逸”只要求 certified landing 与 starter 明确不同，不要求更低能，也不要求
reaction-network edge。对 existing matcher 与 invariant fingerprint 不一致的情况标记
`AMBIGUOUS_MATCH`，不强制写成 Bernoulli 成功或失败。

**预算**

- 总上限：15,000 FE；
- unattributed FE 必须为 0；
- 24 条 generation path 与所有实际可达 checkpoints 必须完整入账；
- 目标 GPU kernel wall time：不超过 10 分钟；wall time 不是晋级指标。

**决策规则**

- 同一 `(system, starter, arm)` 至少 2/3 seeds 出现早期
  `ESCAPED_CERTIFIED`、H8 `RETURN_STARTER`，才开放离散 horizon stopping gate；
- 同一 `(system, starter)` 上 D0 与 K4 均为 3/3 全程
  `RETURN_STARTER`，才把该 starter 定义为可重复 action-support gap；
- 任一轨迹 `QUENCH_UNCONVERGED` 或 `AMBIGUOUS_MATCH` 时，该 pair 不用于学习；
  若同一体系出现两个以上此类标签，先进入数值/matcher gate；
- 其余混合结果只说明 context dependence，不产生 adaptive rule。

本阶段是机制分类，不修改生产默认。

### R1-B：all-candidate HVP 的零 FE value-of-information 回放

**问题**

当前 adaptive score sigma 在未 clipping 的正曲率候选上满足：

\[
\tfrac12\sigma_{\rm score}^2\kappa=E_*,
\]

因此付费曲率在 static ranking 中被归一化。需要判断的是 HVP ranking 的 terminal
价值，而不是再次证明公式。

**数据**

只使用：

- `20260731-direction-candidate-counterfactual-gate` 的 12 个 shared K4 pools；
- 两次重复、96 个 terminal landings；
- `20260731-action-family-transfer-gate` 的 D0/K4 paired outcomes。

**对照**

1. current static K4 winner；
2. full-pool uniform expectation：执行前对四个候选等概率，不读取 HVP；
3. family-stratified expectation：仅按候选 family 做固定轮转，不用曲率、能量或 learned score；
4. D0 exact anchor：已有单 HVP live evidence。

完整候选数据允许计算 2/3 的无选择偏差期望；不得从同一 12 pools 拟合权重。

**预算与 gate**

- 新增 FE：0；
- 若某个零 FE 规则在 C60、PdO 的 paired median terminal regret 均不劣于 static
  K4，且按现有 ledger 投影后总 FE 更低，才允许一个 live selected-only-HVP gate；
- 任何体系符号翻转则关闭“删除 all-candidate HVP”的主张；
- 无论结果如何，都不训练 classifier。

R1-A 与 R1-B 可并行准备，但 R1-A 的 GPU execution 和 R1-B 的分析分别提交，
不能混成一个结论。

## 五、阶段 R2：由 R1 的失败类型选择唯一分支

### 分支 R2-S：确认 action-support gap

第一对照不是 ART，而是 **matched-RMS kick+quench**：

- 使用与 D0 相同的 random+bond intent；
- 位移 RMS 直接取当前 production `target_step_rms`，不扫描 amplitude；
- 不做 HVP、不加 bias、不做 biased-PES relax；
- 立即使用同一 true-quench protocol；
- 与 D0/K4 在 paired starter、seed、总 FE 下比较。

它回答 cumulative Gaussian、HVP 和 serial proposal relax 是否真正提高 escape/FE。

**晋级**

- 若 kick 在两个体系的 escape/FE 与 best-energy AUC 均不劣于当前 SSW，则它只作为
  简单 action family 进入下一轮 fixed-portfolio validation，仍不进入生产默认；
- 若 kick 明显更差，但 R1 显示重复 return starter，再只测试
  `failed-route orthogonal`；
- short MD/minima hopping 只在 kick 失败且仍需不同动力学 support 时开放；
- localized ART 只在上述简单 control 均失败后开放。

每个新 family 单独使用不超过 20,000 FE 的 paired gate。禁止一次同时比较 kick、
orthogonal、MD 和 ART。

### 分支 R2-H：确认早期逃逸后返回

只比较离散 `H2/H4/H8`，每个 action 只做一次 terminal quench：

- 不用 checkpoint quench 作为在线 oracle；
- 不引入连续 patience、Luby sequence 或 learned stop；
- 若无单一 horizon 跨体系稳定获胜，保留 fixed horizon portfolio，由多 walker
  提供覆盖，不学习 context controller。

### 分支 R2-N：数值或标签失败

- quench failure：先固定 endpoint 比较 backend/certificate，不调整物理 bias；
- matcher ambiguity：只实现解决该 ambiguity 所需的最小 species/permutation-aware
  matcher；
- full matcher、archive multigraph 和 delayed credit 不在本分支顺带实现。

## 六、ART、failed-route orthogonal 与 Gaussian levels 的准入条件

### failed-route orthogonal

只有当同一 starter、同一 arm 的至少两个独立 seed 给出：

```text
all checkpoints RETURN_STARTER
```

才允许用这些失败位移构造一个正交补 arm。baseline intent 必须保留；不调 crossover、
mutation、population 或 overlap weight。

### localized ART / bowl breakout

文献支持 ART 的“minimum → activated configuration → new minimum”两阶段物理图景，
也支持在正定 Hessian 区域限制位移局域性、检测到负曲率后释放原子。但这只证明该
action 有物理依据，不证明适合 PAM-SSW。

ART gate 必须：

- 作为独立 arm，不替换 SSW；
- 固定现有 active-region 定义，不扫描原子数；
- 只使用一个预注册的 curvature confirmation rule；
- 同时对比 matched-RMS kick 和当前 SSW；
- 不能在同一实验里再改 proposal optimizer、quench 或 Gaussian。

### Gaussian escape levels

仅当 R1/R2 明确显示“现有方向可行但推动范围不足”时，才把 Gaussian 参数改写为
“作用范围 + 最大 bias force”的 2 个离散 levels。没有该证据时，当前 cumulative
finite-tail Gaussian 保持冻结；不加入 OPES、连续 sigma controller 或多规则有限状态机。

## 七、阶段 R3：确定 traffic 后再做 GPU ForceService

当前 ThreadPool 解决的是不可变 snapshot 与确定性 commit，不是 GPU batch inference。
ForceService 是否优先取决于 R1-B：

- 若 all-candidate HVP 保留：先 batch 同 starter 的 `2K` central-FD geometries；
- 若 selected-only HVP 通过：优先动态组批不同 action 的 proposal/quench force requests。

架构保持：

```text
Synchronous Action Scheduler
  → CPU optimizer/action workers
  → one GPU ForceService owner
  → dynamically bucketed force responses
```

**单 GPU gate**

- fixed 12 C60 + 12 PdO actions；
- serial、现有 ThreadPool、ForceService batch 2/4；
- FE 与 purpose ledger 完全一致；
- certificate 一致；
- matcher tolerance 下 landing 相同；
- float32 能量/力满足预注册容差；
- 中位 wall speedup至少 1.3×；
- GPU contention、queue wait 或 graph construction 抵消收益时停止。

不先做锁步 batch optimizer，不先做多 GPU，也不把 wall speedup写成 FE 减少。

## 八、阶段 R4：稳定 action 后再补完整统计对象

R1/R2 只使用 run-local immutable record。只有至少两个物理上不同的 action families
通过 terminal gate 后，才将下列对象提升到 core：

```text
ExplorationAction
  starter snapshot/context
  direction family + seed
  propagator family + discrete horizon/level
  proposal/quench backend
  per-purpose budget
  starter/family/action propensity
  config/model hashes

TerminationOutcome
  explicit terminal cause
  continuous energy/descriptor/cost
  certificate
  ambiguity state
```

届时再最小拆出：

```text
DirectionOracle
UphillPropagator
ActionRunner
LandingMatcher
```

不是预先重写 5,045 行 `walker.py`。重构的验收标准是同一 frozen action 的
ledger、checkpoint、landing 与 termination 语义不变，而不是文件变小。

full archive replay、hash chain、multigraph、delayed stepping-stone credit 和完整
symmetry matcher 继续后置；它们只有在 posterior/delayed credit 真正需要时才进入。

## 九、阶段 R5：只比较一个 posterior 与 balanced allocation

旧的“立即做 D0/K4 零 FE context classifier”被本路线取代。posterior 重新开放必须
同时满足：

1. 新 portfolio 中至少两个 family 在相同 starter 上有重复 terminal labels；
2. complete action propensity 已在执行前冻结；
3. termination cause、certificate、FE 和 ambiguity 完整；
4. 每个 family 的 posterior 相比 prior 有实际收缩；
5. 两体系都存在方向一致的 family-level signal。

第一模型只做：

- family × system fixed effects；
- Dirichlet-multinomial termination categories；
- energy 与 log(FE) 分开保存；
- batch 内 posterior 冻结；
- 每个 arm 硬配额保证 full support。

第一次在线比较仅有：

1. balanced/stratified uniform allocation；
2. 一个预注册 posterior allocator。

不同时比较 TS、UCB、classifier、reward weights 和 embedding。若出现体系 sign flip、
prior sensitivity、无 posterior contraction、calibration 失败或 validity 回退，立即退回：

```text
stratified multiwalker + random full support
```

## 十、明确退出当前路线的内容

以下内容不应出现在下一轮实现 PR：

- D1 depth sweep；
- 新 curvature scalar scorer；
- node-as-arm TS/UCB；
- PCA/UMAP 上直接决策；
- CEM；
- Weighted Ensemble、AMS、MAP-Elites；
- OPES-like bias；
- unbounded quadratic / CCQN propagator；
- generative displacement model；
- 多 GPU；
- 全量 matcher / multigraph / delayed credit；
- Ruff/Pyright/Hypothesis 工具链扩建；
- 为清理 PR 历史而重写科学代码。

这些不是永久否定，而是当前缺少上游前提。

## 十一、预算、里程碑与停止线

| 阶段 | 新 FE 上限 | 产物 | 停止线 |
|---|---:|---|---|
| R1-A first-passage | 15,000 | 24 trajectories、≤96 checkpoint labels | 标签/账本不闭合立即停止 |
| R1-B HVP VOI replay | 0 | static vs uniform/stratified/D0 counterfactual | 任一体系符号翻转则不删 all-candidate HVP |
| R2 首个机制分支 | 20,000 | 只比较一个新 family 与 controls | 两体系不一致则不晋级、不加复杂度 |
| R3 ForceService | ≤2,000 replay FE | serial/thread/batch wall profile | <1.3× 或语义不等价即停止 |
| R5 posterior | C60/PdO 各20,000 | balanced vs one posterior | contraction/calibration/sign 任一失败即关闭 |

在 posterior 之前的无条件新增预算上限为：

\[
15{,}000 + 20{,}000 = 35{,}000\ \text{FE},
\]

R1-B 不消耗新 FE，R3 只重复固定请求做工程 profiling。orthogonal、MD 和 ART 是互斥的
条件后续，不允许全部自动展开。

## 十二、下一次实际执行只做什么

下一轮只完成两件互不污染的工作：

1. 建立并运行 R1-B 的零 FE HVP value-of-information 回放；
2. 为 R1-A 写最小 run-local protocol/tests，并在同一 commit 冻结 24-case matrix、
   checkpoint 语义、15,000 FE cap 和三态 matcher 规则。

随后才运行 R1-A GPU gate。没有 R1 结论前：

- 不实现 kick、orthogonal、MD 或 ART；
- 不实现 ForceService；
- 不修改 posterior；
- 不重构 production walker。

## 参考的一手物理依据

- Activation–Relaxation Technique:
  <https://arxiv.org/abs/cond-mat/9710023>
- Bowl breakout / localized minimum-mode following:
  <https://arxiv.org/abs/1406.4606>
- Basin hopping:
  <https://www-wales.ch.cam.ac.uk/pdf/JPCA.101.5111.1997.pdf>
- Minima hopping:
  <https://arxiv.org/abs/cond-mat/0402136>

这些文献只用于说明候选 action 的物理来源；是否进入 PAM-SSW 仍由上述本地 paired
equal-budget gates 决定。

## 十三、2026-07-31 执行闭环与下一门控

上述“下一次实际执行”已经完成，后续证据改变了下一轮优先级：

1. 24 条 C60/PdO first-passage 轨迹显示，当前位移方向与 Gaussian-bias 推动经常在
   第八个 uphill micro-step 之前就越过原盆地边界；全部 PdO 轨迹都在 H8 前终止。
2. 与 terminal action 对齐后，24 个相同 starter/action 中有 21 个最终落入新盆地。
   因此平台现象不是“整体推不出原盆地”，也没有发现可重复的“已经逃逸、随后被
   true quench 拉回原盆地”机制。
3. 延长 uphill horizon、增加 proposal relaxation、删除历史 Gaussian、删除
   proposal-side local softening 都没有给出跨体系单调收益；这些机制保持冻结。
4. all-candidate curvature probe 的价值在 C60 与 PdO 间符号翻转，故不删除全部
   Hessian--vector-product 方向评估。
5. 已完成同 starter 的有限差分 force stencil batch gate：方向选择和 FE 完全不变，
   C60/PdO wall speedup 为约 1.9--2.6 倍。它是显式 opt-in 的工程加速，不是搜索
   质量改进。

随后对六条等概率 starter、每条 20,000 FE 的 C60/PdO 轨迹做了零新增 FE 的
残差审计：

- 337 个 action 中，297 个产生新 archive minimum，但只有 36 个刷新历史最低能；
- C60 从当前最低能 starter 出发时 9/13 次刷新最低能，其他 starter 为 8/111；
- PdO 对应为 9/16 与 10/197；
- 固定成功发生时刻和当时 archive 大小后，成功 starter 仍在两体系中显著偏向低能
  排名；详细证据见
  `runs/20260731-starter-energy-residual-audit/`。

物理图景因此从“方向或 uphill 强度不足”转为：

> 当前 action 常能跨盆地，但昂贵 action 被分配给高能 archive 成员时，多数只增加
> 盆地多样性，不能继续降低已经找到的势能面底部。

下一次 live gate 只比较 starter 机制，其他物理模块完全冻结：

1. `uniform_archive`：在全部已知极小值中等概率选择，提供 full-support 参考；
2. `archive_ucb`：冻结当前固定权重、UCB-like 的实现；
3. `metropolis_chain`：经典 SSW 的连续起点链；向低能 landing 必然移动，向高能
   landing 按 `exp[-(E_new-E_current)/T]` 接受。

该比较使用同一个单体 `SurfaceWalker`，而不是混用独立 one-action worker 与连续
walker；每个 system/seed/mode 使用相同 20,000 FE 上限。第一阶段先跑 C60/PdO 的
seed 42；只有出现可解释且账本闭合的差异，才扩展 seeds 43--44。

首次 seed-42 smoke 暴露了一个必须先消除的混杂：旧实现让 starter 抽样与物理
direction/action 共用同一随机流；三种 selector 消耗的随机数个数不同，因此相同
master seed 并不产生配对的首个 action。该 smoke 只保留作诊断，不进入算法比较。
正式 gate 将 starter/Metropolis acceptance 放到独立确定性随机流，direction、
bond、momentum 和其他 action 随机数保持原 master-seed 流。这样 selector 抽样本身
不会平移后续物理 proposal 的随机序列。

这仍不是 posterior 准入。Metropolis 是串行物理基线，不是并行 production 方案。
只有它在两体系中优于等概率选择和现有 archive-UCB-like，才研究如何把“低能 funnel
连续性”变成具有非零全局支持、可批量并行的最小概率分配；在此之前不加入 TS、
MACE embedding、hard top-k/FPS 或新的 acquisition 权重。

## 十四、Safe-LBFGS 审阅并入路线

`Safe-LBFGS-review.md` 对当前优化器证据的分层是合理的，但必须保持以下 claim
boundary。

### 已由现有实验直接支持

对 Gaussian-biased proposal relaxation，`safe-lbfgs-total` 的收益不是来自新的
BFGS 更新公式，而是标准 limited-memory BFGS 与该 modified PES 的组合：

1. 最近可靠 secant 给出的自适应 inverse-Hessian 整体尺度有显著、但不充分的正
   贡献；
2. 保留至少一条 secant 后，优化器开始学习“少数 bias collective directions”和
   “其余硬键/软模自由度”之间的各向异性；这是目前最大的局部效率来源；
3. history 10 相对 history 1 的额外收益主要来自 PdO，在 C60 上不稳定；
4. 冻结 one-bias task 中，safe 相对 FIRE 的证书率和 FE 更好。

这只能证明 safe 是一个有效的 **proposal backend arm**。端到端等预算 SSW 中 C60
三 seed 回退、PdO 三 seed 改善，因此它不是跨体系 universal replacement，也不能把
“更精确地最小化 modified PES”本身当作搜索目标。

### 尚未完成因果归因

当前实现还同时包含：

- total-objective Armijo backtracking；
- 只接受正曲率 secant；
- 每原子单步最大位移；
- minimum-image branch 改变时清空 history；
- raw active-force 终止证书。

这些机制在代码和数学上各自对应明确 failure mode，但尚未通过单因素 gate 区分各自
贡献。当前证据也没有严格证明 safe 普遍优于 ASE-LBFGS 或 SciPy L-BFGS-B。
特别是 `proposal_trust_radius` 目前只有 SciPy backend 实际转化为 Cartesian bounds，
safe 与 ASE 会静默忽略；因此默认 safe--SciPy 对比的可行域和终止证书并不相同。

### 为什么 total-gradient secant 可能有效

新 Gaussian 以当前结构为中心，随后沿同一方向显式移动约一个 width。proposal
relaxation 因而从 Gaussian 拐点附近开始：

\[
q\approx\sigma,\qquad b''(q)\approx0,\qquad |b'(q)|>0.
\]

点 Hessian 在这里恰好不能表达随后从负曲率到正曲率、再进入有限衰减尾部的变化。
total-gradient secant 测量的是一次有限原子位移两端的梯度差，因此会自动积累该路径
上的平均曲率。这个物理图景解释了为什么“只从 secant 中减去解析 bias”的
`bias-separated-lbfgs` 会失败：它删除了 bias 的有限步曲率，却没有把 exact nonlinear
bias 放回局部模型。

### 后续准入顺序

Safe-LBFGS 支线不抢占当前 starter gate，并按以下顺序受限推进：

1. **零 FE telemetry 审计**：从已有 C60/PdO 任务统计每个 accepted step 的
   line-search evaluation、拒绝比例、accepted/rejected secant、MIC reset。若
   `alpha=1` 已几乎总是接受，则不做 Gaussian-aware line search。
2. **比较语义闭合**：在任何 safe--SciPy/ASE 新结论前，显式记录各 backend 是否执行
   coordinate trust region、采用 raw force 还是 projected constrained residual；不再
   把不同可行域的结果合并为一个“收敛率”。
3. **最小因果 gate**：只在同一批冻结 one-bias tasks 中比较 baseline safe、
   history 1/10，以及由 telemetry 指出的一个尚未归因 safeguard。禁止一次展开
   Armijo、curvature threshold、MIC reset、scale smoothing 的全排列。
4. **Exact-bias 变体的条件准入**：只有现有 telemetry 表明 line-search 或 secant
   rejection 是显著 FE 瓶颈，才实现“unknown PES 的 L-BFGS residual model +
   低维 bias subspace 中的完整 nonlinear Gaussian”。第一门控只做 fixed MIC branch
   和 1/2/4 个 cumulative biases，并保留当前 safe fallback。
5. **端到端裁决**：任何固定 proposal task 的 FE 改善都必须回到完整
   `uphill → proposal relax → true quench → terminal basin`，在 C60/PdO 等预算下
   比较最低能量、new basin/1000 FE、duplicate return、validity 和 endpoint diversity。

因此，Exact-Bias Composite L-BFGS 是一个有底层结构依据的候选，不是已获准实现的
下一组件。当前可立即执行的只有步骤 1 的零 FE 审计；其余步骤由该审计和正在运行的
starter gate 决定。

上述步骤 1 已完成。六条现有 20,000-FE C60/PdO 轨迹中：

- C60 proposal line-search 拒绝率为 2.91%，相对 accepted step 的额外 line
  evaluation 为 2.99%，rejected secant 为 0.56%；
- PdO 对应为 4.16%、4.34% 和 0.057%；
- 两体系均没有 MIC branch reset。

因此 backtracking、secant rejection 和 MIC reset 都不是当前主导 FE 瓶颈。
Gaussian-aware line search 与 Exact-Bias Composite L-BFGS 暂不准入；Safe-LBFGS
目前最可信的正机制仍是用保留的有限步 secant 学习 modified PES 的各向异性。

## 十五、跨体系反号、有限预算与 CuO 判别实验

C60 与 PdO 反号不能继续被写成组件的永久否决。今后的负结论必须分成三类：

1. **机制否决**：冻结 task 上直接证明所声称的中间机制没有发生，或账本/物理语义
   错误；
2. **有限预算下不晋级**：机制存在，但在已测试体系、seed 和 FE horizon 下没有
   转化为端到端收益；
3. **统计未决**：体系间反号、seed 方差或终点稀疏，现有数据不足以确定方向。

只有第一类允许停止该机制；第二类保留为非默认 research arm；第三类必须增加 paired
seeds、体系或 horizon 后再裁决。不得用固定 trial 数比较，因为不同 arm 的 proposal
relax 和 true quench 成本可以相差数倍。

仓库根目录的 `Cu110_Cu10O8.zip` 提供第三个真实体系：

- 54 原子 Cu(110)-Cu10O8 slab，46 Cu + 8 O；
- 周期条件为 `(True, True, False)`；
- 固定 mask 沿用历史 `z <= quantile(z, 0.35)` 规则；由于 CuO 晶层内 z 完全
  简并，该阈值实际固定底部两层共 24/54 个 Cu 原子，而不是恰好 35%；
- 使用包内 `CuO-OMAT_finetune.model`，而不是 C60/PdO 共用的通用 OMAT 模型。

CuO starter gate 刻意继承冻结的 PdO 通用 slab action kernel，只替换结构和
calculator model，不做针对结果的超参数调整。它的判别语义是：

- PdO 与 CuO 同号、C60 异号：增加“表面与团簇的物理差异”解释的可信度；
- PdO 与 CuO 异号：不能宣称 slab 共性，优先检查模型域、结构自由度和 seed 方差；
- 三者同号：才允许把 selector 机制提升为跨体系候选。

由于 CuO 使用专门微调模型，它不能单独排除 model bias。执行顺序保持分阶段：

1. 收完当前 C60/PdO seed-42、20,000-FE 三 selector gate；
2. 在 CuO seed 42 做相同三臂 gate，并验证 exact purpose ledger 与固定层；
3. 只有结果可解释，才扩展 C60/PdO/CuO seeds 43--44；
4. 对仍可能依赖长 horizon 的组件，报告 best-energy-vs-FE 曲线，并将“20,000 FE
   未晋级”与“生产级长任务无效”严格区分。

首次 CuO 执行发现 independent bootstrap 是新的配对混杂：相同 raw ARC 分别 true
quench 到 -198.676666 与 -201.044769 eV，bootstrap 成本为 100 与 225 FE。2.368 eV
的起点差异已经大于待比较的 selector 效应，因此该 CuO 输出整体不进入 selector
结论。修正后的 runner 每个 system/seed 只执行一次 bootstrap，复用完全相同的 minimum
坐标和能量，并把相同 bootstrap FE 计入每个 arm 的 20,000-FE 总预算。这个规则也将
用于后续 C60/PdO paired-seed 扩展；不能再以“通常会收敛到同一 minima”为前提。

## 十六、CuO 修正门控后的优先级改变

修正后的 CuO seed-42 三臂共享 -198.677673 eV bootstrap minimum 和相同 108 FE
bootstrap 成本。20,000-FE 终点为：

- uniform archive: -201.872681 eV；
- current archive-UCB-like: -201.686523 eV；
- Metropolis chain: -201.874359 eV。

CuO 的 uniform 与 Metropolis 差 0.00168 eV，应判为并列；单运行下 UCB-like 落后
0.186 eV，但 CuO float32 GPU 长 relaxation 从第一个 action 就会因微小数值扰动发生
basin-level 分叉，故不能判为 selector 机制否决。结合 C60/PdO seed 42，只能支持：

1. full-archive uniform 对 global-minimum objective 通常太分散；
2. low-energy funnel continuation 是正信号；
3. fixed-weight node-UCB-like 尚未证明优于经典 Metropolis；
4. 不准入 TS、MACE-feature top-k/FPS 或 node-level posterior。

CuO 同时改变了 Safe-LBFGS 支线的优先级。三臂有 83.3%--85.3% 的全部 FE 花在
biased proposal relaxation；line-search rejected trials 占 line-search evaluations
的 57.6%--61.7%，而 C60/PdO 只有约 1.8%--5.6%。77/78、88/93 和 83/89 个 CuO
proposal relaxations 分别以 `line_search_failed` 终止。此前“backtracking 不是
瓶颈”的结论只能保留为 C60/PdO 局部证据，不能跨体系推广。

因此下一实际门控从“扩大 selector seeds”改为更底层的 CuO fixed-task line-search
consistency probe。冻结 proposal state、全部 cumulative Gaussian、L-BFGS history
与 search direction，沿同一方向计算 modified-PES 的 alpha scan，并比较 finite
difference slope 与 `g dot p`：

- 小 alpha 下不一致：先处理 calculator precision/energy-force consistency；
- 存在下降区间但 Armijo 漏掉：只测试 acceptance/scaling；
- 能量与力一致但强非二次：才允许 nonmonotone 或 analytic-bias-aware step model。

在该门控前不实现 Exact-Bias Composite L-BFGS。完成后再用 shared bootstrap 扩展
C60/PdO/CuO selector repeats；下一 selector 候选应先是具有全局非零支持的固定
continuation/restart policy，而不是每个 archive node 一个 posterior arm。

## 十七、真实 PES 首次下降是充分证书，不是完整 escape 判据

针对“每个外层 micro step 后若真实 PES 能量已低于 macro starter 就停止”的建议，
G-E0 没有改 walker，也没有重跑方向或 biased relaxation，而是冻结 current-action
first-passage 的 24 条 C60/PdO D0/K4 路径，并补齐全部 82 个已接受 endpoint 的真实
能量。停止阈值直接复用 `dedup_energy_tol=0.001 eV`，没有新增连续超参数。

结果有四条路径触发，4/4 crossing 真实淬火后都是认证的更低盆地，合计可少走 15 个
外层 micro steps。物理原因很直接：现有 walker 已在每一步计算 `true_energy_after`；
当去 bias 前的真实能量已低于 starter 时，后续真实 PES 淬火提供了一个强的低能盆地
证书。在线判断本身因此为零额外 FE。

但该证书覆盖很窄。原 first-passage 中 26 个认证 escape horizon 只有 6 个同时满足
真实能量下降，覆盖率 23.1%。四个触发里，两个避免继续累计 bias 后走向更差终点，一个
与自然终点等价，另一个从 -7.013 eV 提前兑现却会错过 -9.030 eV 的更深终点。因此：

1. first descent 可以作为“兑现已发现低能 basin”的充分停止事件；
2. 它不能替代高于 starter 能量处的 barrier crossing/escape 判据；
3. 它也不是动作内部的最优停止定理，不能加耐心窗口、趋势分数等启发式后直接晋级；
4. 下一步只准入一个 fresh paired G-E1：相同 starter、方向、Gaussian 和随机流下比较
   自然传播与 first-descent stop，并按完整 action 总 FE 和后续搜索收益裁决。

G-E0 离线补证新增 117 FE，其中 33 次真实能量检查、82 次 landing true quench、2 次
post-relax validation；方向、biased proposal relax 和 unattributed 均为零。这个成本
用于验证规则，不是在线规则的固有成本。当前不修改生产默认。

## 十八、首次真实能量下降不晋级为在线停止策略

G-E1 在 fresh seeds 45--47 上执行了 24 条 C60/PdO D0/K4 action。由于两次名义相同的
float32 GPU MACE 执行会在停止事件前发生 endpoint 级数值分叉，正式比较没有放宽容差，
而是只执行一条自然传播路径，用 observer 记录同一路径的精确 crossing checkpoint 和
累计 purpose ledger，再分别淬火 crossing 与自然终点。24/24 shadow prefix 闭合，
7,157 次新 FE 全部归因，`unattributed = 0`。

只有两条 C60 plateau D0 action 触发，所有 PdO 和 C60 intermediate 均未触发。两个提前
landing 都是低于 starter 的认证 basin，继续支持“first descent 是充分证书”；但完整
action 成本合计反而多 58 FE：

- seed 45 在第 2/4 步停止，虽避免了能量走过头并少付 171 FE 的传播成本，但提前构型
  true quench 花 485 FE，自然终点只需 71 FE，最终多付 243 FE；
- seed 47 在第 1/4 步停止净省 185 FE，却错失了再低 4.206 eV 的自然终点。

这给出了干净的物理边界：真实能量已经下降，不代表 crossing state 已接近一个容易收敛
的局部极小值，也不代表后续 bias 不会打开更深的下降通道。用 micro-step 数估计成本会
遗漏 true-quench conditioning；用首次低能盆地估计动作价值则会遗漏 continuation value。
因此 G-E1 返回 `DO_NOT_ADMIT_G_E2`，不增加 patience/window/概率停止器，不修改生产
默认。内部 hook 保持 inert，只作为研究观测缝隙。

该负结果关闭“低能首次下降 early stop”支线。下一优先级不再优化这个低覆盖事件，而是
回到决定 basin 可达性的主线：先分离并验证方向产生/评估与给定方向后的 uphill
propagation；只有形成稳定 action 语义和足够 transition 数据后，才准入
action-conditioned posterior selector。G-E1 不能外推到 CuO、生产级长任务或统计显著性。
