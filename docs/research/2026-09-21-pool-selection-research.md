# 结构池选点：研究问题与最小验证路线

2026-09-21。状态：源码审查与诊断准备；未实现新默认策略，未证明搜索收益。

## 用户目标与当前判断

吸收 PAM 的 archive + 历史反馈选点思想，使 SSW 产生的有效真实极小值进入池，再从池选择下一起点；不要求复制原有加权项。用户进一步指定优先考察 MACE features 而非 SOAP。最终仍以相同总成本下低能合格结构发现、C60 成笼和参考能量目标为验收，覆盖是辅助指标，不静默把全局优化换成最大化结构数量。

原先两条 C60 的旋转改进使完成外步从 117 增至 182，NativeMC 接受从 39 增至 41。这提示外层选择值得比较，但不证明拒绝是坏事或 MC 是唯一瓶颈。选择不同起点也会改变后续提议分布，旧轨迹不能离线给出另一策略的真实搜索成功率。

## 已核查的源码事实

- `pamssw/acquisition.py` 的 BanditSelector 混合能量、novelty、density、访问次数、frontier 和 dead penalty，是启发式 UCB-like 选择，不是已训练 RL 模型。不要整套继承其未验证权重。
- `exploration/policies.py` 的新控制器支持 uniform、posterior_proportional、minimal_ucb。posterior 更新的是相对 dispatch archive 的新发现事件，不是低能收益，也不直接按计算成本计分。
- `fingerprint.py` 的描述符为 16-bin 距离直方图加四个统计量。默认上界随每个结构最大距离改变；无元素对/角度通道；长度统计与无量纲直方图直接拼接。PBC 使用 MIC，不等于展开所有周期局部邻域。
- `archive.py` 的身份判断独立于上述描述符：非周期 Kabsch 有序原子 RMSD，周期 MIC RMSD，并检查允许边界/固定掩码。未处理等价原子置换。仅升级评分特征不会修复此问题。
- 新 exploration worker 目前接旧 SurfaceWalker；独立 standalone kernel 的最小单落点接入仍在审查，尤其初始淬火重复开销、RNG 和失败成本。此处不预先承诺公共 API 重构。

## 分开三件事

1. 入池：真实目标淬火合格才可成为 starter；重复落点记录访问/成本但不重复奖励。失败与预算截断不得作为新结构。
2. 表示：用于近邻、新颖性与状态信息；结构特征相近不等于相同盆地，更不等于低势垒连接。
3. 决策：从合格池选择起点。纯能量 top-k 可能排除通向其他 funnel 的中间态，这是待验证风险，不能把 k 当通用物理常数。

## MACE feature 路线

优先用现有势模型的旋转不变原子特征，固定 checkpoint、head 与层选取，不混用 MH-1 与 OMAT 的特征空间。官方接口为 `MACECalculator.get_descriptors`；真实安装版本、额外 forward 成本及能否复用能量/力输出须以本地代码和实测确认，不能预称零成本。

全结构表示首先考察按元素汇总的 invariant 原子特征；平均可能掩盖少数缺陷或不同环境分布，若保存轨迹暴露该问题，再比较环境集合匹配，避免一开始堆叠统计量和距离参数。完整 equivariant 张量不能直接当旋转不变欧氏向量使用。特征归一化和层选择必须固定后再评估，不能用最终测试收益反复挑选。

MACE 内部特征服务于能量模型，不自动就是搜索价值或动力学坐标。即使区分结构更强，仍须检验是否改善 starter 决策。通用 ASE kernel 不依赖 MACE；无此特征的 Calculator 保持原接口和明确的几何比较路径。

## 验证顺序与停止条件

- 已有保存轨迹零 PES 诊断：RDF 分箱变化范围；刚体变换/原子置换；真实不同拓扑/能量落点的距离。接口检查不作搜索收益结论。
- 有界 MACE 特征检查：固定少数已保存结构与模型，测不变性、维度/层/head、提取成本。计入模型 forward 次数及实际时间；只在合格落点提取，先不侵入每个内层优化步。
- 先固定表示与单步 kernel，比 MC 链、全池均匀重启、现有历史收益选点，区分池可访问性和学习收益。先不同时改 rotation/height/LS/GA。
- 再在同一选点规则下比较原描述符与 MACE 特征。预算截断、初始化、失败、描述符开销全部计入，保留旧基线与跨体系验证。
- 若复杂特征不改善重复识别/决策，或成本抵消搜索收益，不保留为默认；若历史收益预测不足，不追加任意权重挽救结果。评分的确切收益定义与协议须在性能运行前固定。

## 原始资料与证据边界

- MACE 官方 descriptor 文档：https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html （已核对接口、默认 invariant、按层提取与原子特征形状；本地适配另查源码。）
- Bartók, Kondor, Csányi, 2013, *On representing chemical environments*, DOI 10.1103/PhysRevB.87.184115, https://arxiv.org/abs/1209.3140 。本轮摘要核对，支持 SOAP 作为局部不变表示的背景；不引入 SOAP 依赖。
- De, Bartók, Csányi, Ceriotti, 2016, *Comparing molecules and solids across structural and alchemical space*, DOI 10.1039/C6CP00415F, https://arxiv.org/abs/1601.04077 。本轮摘要核对，支持区别原子局部表示与整个结构匹配；未移植 REMatch 参数。
- Pozdnyakov et al., 2020, *Incompleteness of Atomic Structure Representations*, DOI 10.1103/PhysRevLett.125.166001, https://arxiv.org/abs/2001.11696 。本轮摘要核对，低阶相关表示存在非唯一性，不能以 descriptor equality 证明 basin identity；不把此结论扩大成 MACE 一定失败。


## 方向状态审查后的最小设计（用户已批准）

目标不变：保留 SSW 本体的有效方向反馈，让所有合格落点都有机会成为后续起点；以总成本下的低能/目标结构发现判断收益。此处改变的是搜索控制接口，不是新增 Gaussian、奖励权重或模型依赖。

当前阻塞：`RecoveredDirectionController` 不只是单次旋转器，它在连续外步之间保存 pair/group/marker，且在 MC 前从实际落点更新。逐次 `run_ssw(steps=1)` 会重新初始化这些状态；即使各策略都重置，也只能评价另一个失去该记忆的 kernel。另有重复初始淬火。现有 checkpoint 明确不支持完整 recovered_direction，不能伪造 checkpoint 绕过。

推荐的最小契约：保持一个连续搜索运行及其内部状态，在外步完成处增加受限的起点选择入口，选择已经认证的历史落点（用观察索引引用，不由策略注入任意未认证坐标）。保留既有 `run_ssw` 参数与返回行为；未开启池策略时，旧 MC 路径与随机调用顺序不变。开启池策略后，观察归档、MC诊断与实际下一starter分别记录，不将“入池”冒充“沿MC链接受”。同一路径继续时保存现有方向状态；选择另一个历史起点时明确记录restart并重建方向选择状态，不把跨池几何跳跃伪装成真实relax位移。策略随机流与kernel随机流分开。第一版限普通固定胞SSW；LS的强度记忆和GA跨walker状态不默默套用本契约。

这不要求第一版创建通用Session框架或重写持久化格式；完整方向状态仍可由当前连续driver持有。具体入口命名与实现拆分在用户确认状态语义后确定。另一方案是不动核心，先用stateless recovered_rotation做池策略对照，但它不能回答完整方向反馈与池选点的组合问题。

验证：未开启新入口时同输入/RNG的旧路径回归；MC拒绝但有效landing仍可见；连续选择不重新初始化方向器；非局部restart明确重置；失败/未认证结构不能成为starter；选择不改变已归档结构；初始化/失败/特征开销均计账。通过这些契约检查后，使用不同保存起点、同后端和固定总预算比较MC/全池均匀/已有PAM评分，再分别比较表示，不同时改kernel。

评分理论边界（本项目推导）：固定starter i的落点分布为P_i，archive由A扩大到A'时，“新发现”概率 p_i(A)=P_i(landing不在A) 满足 p_i(A')<=p_i(A)。因此现有Beta-Bernoulli累计发现率不是平稳成功概率，也不是低能收益/成本率；它可以作经验对照，不能直接借用平稳bandit保证。暂不新增衰减参数或多项reward来补偿。MACE特征提取只验证可用性与成本，不能据不变性检查宣称更好的选点。

[源码契约审查](../../../ls-constraint-qualification/research/ga_ssw/evidence/pool-selection-runner-review-20260921/README.md)。用户已批准“持久搜索状态 + 显式池重启”。当前实施：受限starter_selector返回已认证观察索引或None；使用独立selector_rng；不启用时旧MC/RNG路径不变。固定胞、无LS、无checkpoint范围先完成，不默默扩展状态格式。


## MACE特征接口实测（已完成，仅实现检查）

GPU1436296在1V100上完成，44秒。使用已记录模型/输入：C60 MH-1 head=omol、Cu55 OMAT-small实际解析head=omat_pbe；float64，16次descriptor API、4次E/F，hook实测共20次模型forward，无SSW搜索。MH-1每原子全层invariant为1024维（第一层512）；OMAT-small为256维（第一层128）。两个输入的元素平均与按已知置换对齐的原子特征均有限，并保持刚体/置换不变到此次数值误差范围；MH-1最大原子特征刚体差约1.53e-7，其余更小，不将此作为待优化科学问题。

热调用示例：MH-1 descriptor约0.049–0.050秒，OMAT约0.0157–0.0158秒；这只有少量接口计时，不是严格模型速度排名。最关键是**每次提取都会新增一次模型forward，即使刚对相同坐标做过E/F**；没有触发ASE calculate并不代表免费缓存。原始get_descriptors还走模型默认forward配置，不预称只做能量前向。按合格落点提取的额外成本可明确计账，尚未实现复用E/F内部node_feats缓存，也未实现特征梯度方向。

Cu55该探针直接读取原输入，未重新淬火；其E/F仅作接口计时参考，不能沿用另一个搜索包淬火后的能量/力资格。未用本次结果判定池身份、挑特征层或宣称搜索收益。

决定：MACE可作为用户要求的可选池信息来源，不新增SOAP依赖，不让通用ASE kernel依赖MACE。先解决完整方向状态与池调度契约，再分别比较选点策略与表示；不把维度更高等同于更有效。原始探针、源码manifest和结果：[实测包](../../../ga-ls-composition/research/ga_ssw/evidence/mace-invariant-feature-probe-20260921/README.md)、[result.json](../../../ga-ls-composition/research/ga_ssw/evidence/mace-invariant-feature-probe-20260921/result.json)。官方接口依据：[MACE descriptors](https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html)，实际行为以上述本地版本源码和实测为准。


## 已批准设计的实施与验收顺序

1. 核心实现限定为一个外步边界选择入口及独立的数据契约模块：复制合格观察，索引引用，记录实际starter与restart。沿当前MC结果继续保留controller；非局部选择新建controller，从所选结构初始化。不重复真实淬火、不伪造位移。默认不调用任何选择器。
2. 先验证新增合同测试在旧代码失败，再验证透明None hook、MC拒绝可见、状态不被snapshot修改、错误索引和不支持组合在正确边界拒绝、连续/重启生命周期。复用原固定胞/CBD/MC/checkpoint回归，保护未启用路径。
3. 真实资格：两个C60 first保存态和Cu55，default/transparent/uniform三个臂，各最多4外步、3000搜索请求+2独立复核、150秒搜索时间。总请求最多27018，单V100最多30分钟。default与transparent须全程相同；uniform允许几何与代价分叉。此次uniform从原始合格观察选取，重复观察的权重不校正，只是入口/状态资格，不据此比较盆地覆盖或宣称池策略科学优势。
4. 通过后再接现有PAM归档/评分，明确去重与评分收益定义，固定kernel、表示和预算做完整策略对照。MACE feature不与状态接口在同一实验同时改变。

## 实施审查补充：评分状态与连续方向记忆

已有方向记忆保留后，更精确的提议分布是 K(落点 | 起点结构, controller状态)，而非只由一个archive entry决定。同一盆地的连续行走和显式重启也未必是同一分布。上面的archive扩张使新发现概率下降的推导，条件是固定这一提议分布；它不证明实际自适应行走中的逐步成功概率必然单调。原PAM按entry累计统计仍是经验近似，当前仅作为不改权重的对照，不宣称平稳bandit保证。

实施边界：所有新的合格观察在callback中立即进入caller-owned archive，包括MC拒绝的落点；不是等待整个run结束才插入。entry到首个代表观察的映射稳定，重复落点记访问但不增新entry。选中与当前同entry时返回None避免无意义方向重启；不同entry返回认证观察索引。trial只归属已经执行的外步，不预先计入最后一个尚未执行的选择。终止失败及初始化成本由result逐记录对账。研究适配器首先限定无约束保存输入，通用SSW的FixAtoms/Hookean既有能力不变。

## 当前可运行方式（开发接口，不是新默认）

固定胞无约束输入可用连续`run_ssw`保留方向状态，再通过研究适配器选择已有合格落点：

```python
from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter

policy = PoolStarterAdapter(mode="pam", energy_tol=1e-3, rmsd_tol=0.1)
result = run_ssw(atoms, surface, steps=40, config=config,
                 recovered_direction=direction, mc=mc,
                 rng=np.random.default_rng(seed),
                 starter_selector=policy,
                 selector_rng=np.random.default_rng(seed + 1000003))
audit = policy.finalize(result)
```

`mode="uniform"`使用完全相同的近似归档，按entry均匀选择；`mode="pam"`复用原BanditSelector的全部既有混合概率与分数。不给starter_selector仍是默认MC路径。两个容差沿用原PAM配置，RMSD有序匹配不解决同元素置换，energy_tol只诊断近几何下能量差异，不把接近能量当盆地身份。不是严格RL或新优化公式。

所有模型调用仍由传入surface承担；适配器不直接依赖MACE，不调用势能。研究适配器当前明确拒绝约束输入，避免State丢失约束语义；这不改变独立kernel既有FixAtoms/Hookean支持。LS与checkpoint不通过这一入口接入，需以后单独处理其状态契约。

GPU1436617采用上述两策略与默认MC、三个保存起点的固定开发协议。每臂6000搜索请求/最多40外步/600秒，最多54000搜索+18fresh；独立同势能复核与拓扑结果分别报告。预算/失败只改变可用证据长度，不能把小梯度当成C60成笼。后续判断是否值得加MACE信息，须先看同一归档下池访问和PAM评分是否有实际收益。

解释边界：本轮复用原PAM评分公式及其select混合规则，反馈单位为一个独立SSW外步；旧SurfaceWalker的多proposal/trial机制没有移植。因此这是固定kernel下的评分对照，不是整套旧PAM算法逐数值复现。该单位差异与recovered controller的隐状态均限制对原Bandit统计的理论解释。


## 本阶段结论与后续边界

GPU1436617与CPU1436624已完成。两份C60输入中当前池策略均未改善最低能量，Cu55归档退化为单条目、不能区分策略；成本与接口审计通过。[结果、资格和不推广决定](2026-09-21-pool-policy-pilot-results.md)。目前不扩大评分复杂度，也不直接将此策略推广到LS/GA。

以后若启用LS池跳转，必须先明确自适应状态归属。现有LS封存的邻居/参考距离必须属于所选几何；不能把跳转前的frozen potential直接施加在新起点。NativeLS另有元素对强度表、历次更新的键数、周期计数和save/restore笔记；paper LS有总强度及响应历史。可选择显式重启时重新初始化、按分支保存/恢复，或共享强度但定义几何重新建立规则；它们有不同的学习历史和内存/接口成本，源码和此次固定胞批准不能自动决定哪一种。当前保持明确拒绝LS+selector，不新增参数或公共状态框架。真正推进该扩展时按AGENTS第8节提交具体方案讨论。
