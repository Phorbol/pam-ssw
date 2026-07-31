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
