# PAM-SSW 无偏并行 PES 探索：阶段总结与咨询问题

> 面向不了解本仓库的计算化学、全局优化和统计学习专家。研究证据截至 `feature/direction-continuation-ablation@1e4edb7`；其中最新 action-family GPU 矩阵实际执行于 `c82603c`，汇总结论提交于 `1e4edb7`。目标是有限预算下寻找低能极小值并覆盖 PES；不研究 reaction network，也不声称正则系综采样。

本工作的核心命题不是“为 SSW 换一个更高级的 soft-mode solver、优化器或 bandit”，而是把一次 PES 探索写成可核算、可重放的条件 action：

\[
(\text{starter},\text{direction},\text{uphill propagator},
\text{proposal relax},\text{true quench})\longrightarrow\text{landing}.
\]

在此基础上，批量并行用于扩大具有全支持的物理 action 覆盖；贝叶斯后验只有在低维 action family 能重复出现、其条件收益能跨 context 预测时才允许进入。否则，多 walker 随机分臂比一个形式更漂亮但不可学习的 selector 更诚实。

## 1. 问题与最小算法

局部优化只能落入当前吸引盆。SSW 的物理动作是

\[
x_i\xrightarrow[\text{direction }u]{\text{biased uphill}}x_{\rm esc}
\xrightarrow{\text{true-PES quench}}x_j .
\]

random/bond 等方向提供事件意图，曲率信息避免过硬；逐次加入局域 Gaussian bias，在修改 PES 上弛豫，移除 bias 后在真实 PES 淬火。局部曲率、方向夹角和短期上升都只是 proxy，完整 uphill+quench 才给 action 可靠标签。

当前流水线是：

1. 原始 `State` 先在真实 PES 淬火，作为 archive minimum 0（bootstrap）。
2. selector 从 minima archive 选 starter。
3. random、bond、momentum 产生 K 个方向；central-FD HVP 评估。
4. 沿选定方向显式位移，累计 Gaussian bias，serial proposal relax。
5. true quench 得到 landing；energy+RMSD 去重并更新 archive/credit。
6. 每个 action/campaign 有 force-evaluation（FE）上限；ThreadPool 从同一不可变 snapshot 派发并按 slot 顺序提交。

学习对象至少应是

\[
a=(\text{starter context},\text{direction family},\text{uphill policy},
\text{proposal backend},\text{fidelity}),
\]

而非孤立 node 或 direction label。

## 2. 开发原则

- 先解释物理机制；数学更完整、更漂亮不等于搜索更好。
- 一次只改一个因素，冻结 starter、seed、uphill、optimizer、quench 和预算。
- HVP、失败 line search、未收敛 quench、validation 均计 FE；wall time 单列。
- 用活动原子最大力证书，而非 optimizer success 判断收敛。
- 正负翻转时不加阈值、混合权重或连续调度器事后抢救。
- 先证明可预测再上 TS/UCB/classifier；保留 random/physics 与 starter 的非零支持。

## 3. 原始 review 后完成与未完成的事

| 原问题 | 已完成 | 仍有边界 |
|---|---|---|
| HVP 绕过预算 | direction oracle 统一经 `EvalCounter`，各物理阶段分 purpose 核算 | descriptor 时间单列 |
| bootstrap 不清 | runner 先 true-quench 原始 State，证书通过才入 archive，并计总预算 | 历史一次仅标签误写，未漏成本 |
| 并行 credit 不确定 | 不可变 policy/archive snapshot、确定性 seed、ThreadPool、slot-order 原子提交、event log | 当前同步 batch；异步未做 |
| winner-only 偏差 | terminal action/失败均记录；共享 K4 实验执行每个候选 | production 不会 quench 所有未选方向 |
| proposal/quench 耦合 | backend、证书、fallback 与成本已显式化 | `walker.py` 只做最小重构 |
| PBC geometry | collision 已用 MIC | archive matcher 仍不处理完整置换/对称 |
| archive 非 multigraph | 未做；当前主线排除 reaction-network | parent-only 不适合一般 transition credit |
| LS 等价性 | 已区分 current 与 paper-ordered | 当前实现不能称为文献 LS 的等价实现 |
| UCB-like 有效性 | 只保留为 baseline | 未证优于 uniform/Metropolis |

## 4. 核心证据

### 优化器与真实淬火

16 个冻结 one-bias 任务中，safe total-gradient L-BFGS 为16/16证书、FIRE为14/16，且 safe 全部少 FE；但等预算 SSW 里，safe 相对 FIRE 在 C60 三个 seed 全回退，在 PdO 三个 seed全改善。故 safe-LBFGS 是有效的独立 proposal backend，不是普适默认。其核心是标准两循环、secant 自适应尺度、正曲率筛选、每原子0.2 Å步限、Armijo 回溯和 PBC image branch 变更时清历史；bias-separated 版本未晋级。

严格 raw-landing corpus 上 ASE-LBFGS 单臂28/32；`ASE-LBFGS→仅证书失败才 FIRE` 达32/32，额外成本约9.6%。200-step 旧长任务中，C60 true quench 仅35/201证书，PdO为201/201，说明 true quench 是 C60 特有的数值瓶颈。

### 方向

C60 固定任务支持 K4 省 FE；但200-step K4 比历史 K12 少12,650 FE、minima 更多，却最终高1.532 eV且 AUC 更差。PdO K4/K8 两 seed 胜负1–1，故保留K8。momentum 在 C60 K4 长任务占被选方向79%；开关结果翻转，因此“未证明但保留”。

更深 block-Krylov 确实更软、residual 更小、更离域，却不稳定改善 landing。anchor-Lanczos 更软但终态更差；energy-bounded 组合把 anchor overlap 提高约7倍仍未提高重复成功数。一次 C60 小门中，projection transport 降 direction FE 89.8%、action FE 19.2%，但未跨体系验证。

最关键的 shared K4 实验共96个严格 landing：现有静态 scorer 在12个 pool 中10次错过更好候选；且正曲率候选满足
\(\tfrac12\sigma_{\rm score}^2\kappa=0.8\) eV，adaptive sigma 正好抵消付费 HVP 的曲率区分。但“选最小 true curvature”只改善 PdO、伤害 C60；H2“大上升优先”在 C60 0/6、PdO 2/6，并多花约105/150 FE。评分是瓶颈，但最软或短期更猛都不是通用解。

### Uphill 与 LS

固定 Gaussian、关闭 sigma/曲率 feedback 都未稳定胜出。中心曲率匹配的无界 quadratic 在 C60/PdO 造成约260/295 eV 额外真实能量、非定域 runaway 和证书崩溃，说明 Gaussian 的有限衰减尾部是有效机制。proposal relax 从80增至300使6/6修改-PES任务收敛，却全落回原 basin并多花417 FE。只留 newest Gaussian 在3/6改变 basin（两个更低、一个高1.917 eV）；累计历史不是冗余。H14 比同轨迹 H8 为2胜/1负/2同 basin，多花1,087 FE，故保留H8。

当前 active-neighbor moving-reference LS 在18/18配对中不改变首方向，active atoms 又常与最终方向错位。paper-ordered C60 实现使9/9首方向改变并一致软化，但 participation ratio 未提高，landing 对 no-LS 的 paired 中位仅 -0.000153 eV。因此它仅 mechanism-positive，未 production-positive。

### Starter、表征与后验

四个200-trial archive 已增长到123–195 nodes；即使均匀分配，Beta(1,1) 后验标准差仍约0.232–0.249，接近 prior 0.289，node-as-arm 无法集中。legacy selector 在1,000 prototypes 时一次约4.24 s，近二次成本来自重复能量/density 扫描。

pooled invariant MACE feature 的局部能量连续性优于当前 RDF；PCA95 压至5维却只保留约47%–57%的精确最近邻，故仅是压缩 control。MACE-FPS cell-first 通过全支持与核算门，但20k FE、两体系、seeds42–44 无稳定增益，C60 gain AUC 三 seed 全负。

48 candidates 的零 FE feature gate 未通过留体系验证。最新 D0 exact anchor、D1 单次 Krylov expansion、K4 discrete 的36-case gate中，C60三臂各赢2/6，PdO D0/K4各赢3/6；训练 PdO 选 D0，迁移 C60 比均匀分臂差1.919 eV。现在没有依据加入 context-free posterior、TS/UCB系数或 classifier。

## 5. 归因与真实瓶颈

**确认的工程或机制正向：**精确核算/bootstrap；不可变并行批次；safe-LBFGS 作为独立 proposal backend；C60 K4省 FE、PdO K8保质量；momentum 暂留；有限尾部与累计 Gaussian；C60 ASE-LBFGS→FIRE 严格 quench。pooled MACE 仅确认具有更好的局部诊断连续性，paper-ordered LS 仅确认能真实改变并软化模式；后二者都尚未证明端到端搜索增益。

**否定或未支持：**safe-LBFGS 普适替代、bias-separated L-BFGS、最软 Ritz、deeper/anchor Lanczos、energy-bounded 默认化、true-curvature 通用 ranker、short racing、无界 quadratic、固定 Gaussian、H14、relax300、current LS、hard FPS/top-k、node TS/UCB、小数据 classifier、context-free action posterior。

**尚未归因：**momentum 为何有效；D0/K4 切换能否由零 FE context 预测；adaptive sigma 何时过度收缩；paper-ordered LS 强度能否转化为 landing gain；matcher 对 unique minima 的偏差；单 GPU ThreadPool scaling。

证据强弱排序：

1. 强：静态 direction score 与 terminal quality 不一致。
2. 强：proposal relax 是最大 FE 项，但局部提速不等于全局提效。
3. 中强：direction–propagator 强耦合，现有 proxy 不充分。
4. 中：C60 true quench 困难，fallback 已基本控制。
5. 中：node-as-arm 和 legacy selector 不可扩展。
6. 弱：尚不能说 starter selector 是平台首因；动作本身缺失时 selector 无能为力。

因此当前优先级不是先把 `archive_ucb` 换成 Metropolis、TS 或更强表征。starter selector 的统计对象和计算规模确有问题，但现有证据首先表明：给定同一 starter，我们还不能从廉价信息稳定判断哪种方向会产生更好的 terminal landing。若 action quality 本身不可预测，更复杂的 starter policy 只是在更聪明地选择一个仍然缺乏可靠动作的起点。

## 6. 后验、并行与“无偏”

“无偏”有三种不同含义：

1. **搜索全支持**：所有 starter 和基础物理 action 始终有正概率——这是当前目标。
2. **统计无选择偏差**：执行前冻结 propensity，失败也入账；off-policy 评价再用 IPS/DR。
3. **物理系综无偏**：detailed balance 与指定平衡分布——当前算法不满足，也不是目标。

贝叶斯层应作用于重复的低维 action families，并 partial pooling；context 只用执行前缓存信息；reward 保留 landing energy、new minimum、validity、FE cost 向量。现有同步 ThreadPool 已保证同 snapshot 派发、独立 worker/cap、完成顺序无关和 batch 后更新。batch relaxer 只能作为 wall-time backend，不能称为减少 FE；异步 stale posterior 应等同步算法增益被证明后再做。

## 7. 生产基线与研究边界

稳定生产参考是 `pam/experiment/pdo-k4-k8-fixed-budget@91217d3`。C60 使用 validated K4/B8、safe-LBFGS proposal、ASE-LBFGS→FIRE、`fmax=0.01`；PdO 保留K8和 SciPy L-BFGS-B true quench、`fmax=0.03`。legacy `archive_ucb` 只是 UCB-like baseline，未证优于 uniform/Metropolis。adaptive cumulative Gaussian 因替代者未过门而保留，不代表其反馈律已最优。D0/D1、block-Krylov、paper-ordered LS、cell/posterior policy 均不得进入默认。

## 8. 下一步及停止条件

### G1：D0/K4 零 FE context gate

先做可识别性审计，而不是模型竞赛。离线仅使用 starter energy rank、缓存 pooled-MACE、anchor localization、random/bond composition，预注册一个带固定正则的线性 score，不做特征搜索、超参数调优或新增 FE。两个 leave-one-system-out 方向都优于 equal-arm regret 才通过；任一失败即停止 classifier/TS/UCB，改用多 walker 随机分臂。若样本量不足以给出稳定区间，结论同样是“不具备晋级后验模型的证据”，而不是继续扩充模型。

### G2：算法等价 batch gate

C60/PdO 各12个冻结 actions，比 serial 与 ThreadPool batch=2/4。ledger、证书和 landing 语义必须一致，且中位加速至少1.3×；低于10%、GPU contention 或结果不稳即保留单 worker，不扩建 batch relaxer。

### G3：仅在 G1 通过后做后验

两体系 paired seeds、每臂20k FE，只比全支持 uniform family allocation 与一个固定 posterior，不同时比较 TS/UCB。两体系 FE-normalized best-energy AUC 中位不负且 validity/certificate 不回退才晋级；符号翻转或 posterior 不集中即退回随机并行。

LS 只再允许一次 paper scalar-feedback 门；curvature gain 若仍不转化为 paired landing gain则关闭。受约束 quadratic/CCQN 必须先给出无需体系调参的约束和局域化机制，不能靠给失败无界模型逐项加 cutoff、quartic、mixing weight。

## 9. 希望外部专家回答

1. “严格 quench 后更低或新 basin”是否低估高能 stepping-stone？如何保留向量 reward 又能消融？
2. committor、minimum-mode 或 rare-event 理论中，是否有比曲率/短期上升更接近 terminal basin 的廉价方向判据？
3. random+bond intent 分散在 Krylov eigenbasis 中；是否存在无任意 overlap 权重的约束变分形式？
4. Gaussian 有限尾部重要但 sigma feedback 正负翻转；能否从模型误差或受限位移几何推导离散更新？
5. 每体系仅十几个 context 时，hierarchical Bayesian action model 的“不可学”判据应是什么？
6. 多 walker 是否更适合 weighted-ensemble/stratified allocation，而不是 bandit？如何避免 MACE cell 成为人为 reaction coordinate？
7. pooled MACE 可能抹平 active region，direction-conditioned/local pooling 应如何零 FE 验证？
8. C60 float32 CUDA 有路径敏感性；关键 gate 是否应 float64 或多模型复核？
9. cluster 与固定 slab 的最小可靠 symmetry/permutation matcher 应是什么？

## 10. 证据边界与索引

多数结果限于 MACE-OMAT-0-small、float32 CUDA、RTX3060、C60 与固定底层 PdO，常只有2–3 seeds；固定 starter 又来自 seed-42 轨迹。它们足以否定不稳定晋级假设，不足以证明总体优越性。尚无 DFT 复核、跨模型/化学空间泛化、总体统计或热力学采样验证。

核心本地证据：

- optimizer/quench：`runs/20260727-030502-safe-lbfgs-multiseed-threadsafe-gpu/conclusion.md`、`runs/20260728-true-quench-raw-strict-replay/conclusion.md`、`runs/20260728-true-quench-sequential-rescue/conclusion.md`、`runs/20260728-safe-lbfgs-200-production/conclusion.md`
- production/direction：`runs/20260729-c60-k4-profile-200/final_report.md`、`runs/20260729-pdo-k4-k8-fixed-budget/final_report.md`、`runs/20260731-direction-candidate-counterfactual-gate/README.md`
- uphill/LS：`runs/20260730-uphill-bias-shape-u3/final_report.md`、`runs/20260731-uphill-mechanism-closure/u4_conclusion.md`、`runs/20260730-ls-softening-scope-gate/conclusion.md`、`runs/20260731-paper-ordered-ls-gate/conclusion.md`
- selector/posterior：`runs/20260730-starter-representation-audit/conclusion.md`、`runs/20260730-starter-cell-online-gate/production_20k_seeds42_44_conclusion.md`、`runs/20260731-action-family-transfer-gate/conclusion.md`

背景入口：原始 SSW（DOI `10.1021/ct301010b`）、dimer（DOI `10.1063/1.480097`）、LS-SSW（PubMed `39636281`）。本文数值结论均以本地证据为准，不由文献形式外推。
