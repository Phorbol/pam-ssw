# AutoSella：面向 PAM-SSW 局部优化的论文卡片

> Source coverage: Full paper HTML, v1
> Extraction confidence: High for text and tables; figure interpretation uses captions and associated text
> Locator mode: structure-grounded
> Primary analytical lens: methods
> Secondary analytical lens: None
> Context verification: Targeted external check — pinned public optimizer source and local PAM source/ledgers
> Card completeness: Complete relative to supplied source

## 01 基本信息

[Paper] Artem Tsypin et al., *Optimizing the Optimizer: Language Models Discover Faster Molecular Relaxation*, 2026, arXiv:2610.06577v1，预印本。领域：分子几何优化、自动算法开发。[全文](https://arxiv.org/html/2610.06577v1)。机构信息本卡不展开，DOI 未核实。[Paper: Metadata]

[External] [代码](https://github.com/vdeshchenya/autosella)固定到 `b964218eed6efdac55cb023f5a18170a6aed29e1`；本次阅读 2026-10-07。与项目的关系是局部淬火候选，不是 SSW 全局搜索方法。详细源行在 [源码核查](source-review.md)。

## 02 一句话总结

[Analysis] 它给我们的主要启示是：在学习梯度历史之前，先用合适的自由度和曲率尺度表示分子，而不是只增大 L-BFGS 历史。

## 03 研究问题

[Paper] 目标是在共同收敛要求下减少分子优化的力调用；通过固定评价器约束代码演化。[Paper: Section 3.1 Task and metrics]

[Analysis] 对 PAM 的具体问题是：相同保存态与完整目标下，内坐标优化能否降低合格淬火的实际成本，且不降低后续有效盆地发现效率？

## 04 背景与发展路径

[External] 当前 Safe-total 是 Cartesian L-BFGS 加 Armijo 回溯；AutoSella-F 使用 Sella-derived 内坐标、模型 Hessian 和受限拟牛顿步。这里由实际源码核实；不据论文 related work 宣称已完成全领域综述。

## 05 痛点

|痛点|本项目的具体表现|解释与证据边界|
|---|---|---|
|不同自由度刚度相差大|共价键与柔性集体重排需要不同步长尺度|[Analysis] 曲率/坐标预条件有理由；本轮没有测量本项目条件数|
|短淬火重复初始化|每次 Safe-total 调用从空历史开始|[External] `pamssw/relax.py:_relax_with_safe_lbfgs`；模型曲率可能帮助初始几步，尚未验证|
|优化器收益与初始化混合|F 入口先调整部分起始几何|[External] `autosella_f.py:minimize_func`；需要分离研究|

## 06 核心思想

[External] 分子内部自由度及片段刚体自由度 → 几何相关曲率模型 → 梯度历史校正 → 根据模型预测调整步长。冻结的算法不在每次优化时调用语言模型；源码审查支持这一事实。

[Analysis] 值得检验的是曲率信息质量，而不是机制数量。许多化学分类规则是经验先验，不能直接升级为我们通用默认。

## 07 方法总览

[External] 公开输入是原子序数、nm 坐标、返回 kJ/mol 能量与 kJ/(mol nm) 力的 callback，以及调用预算和外部收敛回调。输出是最后已评估坐标与计数。F 入口没有晶胞、PBC 或固定原子输入；内部变胞分支明确抛出 `NotImplementedError`。详见 [源码核查](source-review.md)。

[Analysis] 因此先考察非周期真实势面淬火，偏置表面和 VC 不能直接承接其论文结果。

## 08 核心模块

|模块|与 Safe-total 的差别|去除影响如何解释|
|---|---|---|
|内部/片段坐标|非线性坐标和按模式定标，替代裸 Cartesian 尺度|[Analysis] 需普通 Sella 对照才能区分基础坐标收益与 AutoSella 新增收益|
|模型 Hessian|化学分类及非键接触曲率，几何变化时更新|[Analysis] 可改善初始模型，也可能在成键/断键附近失准|
|多割线与曲率传输|同时约束多个历史方向，校正历史平均与当前模型差别|[External] F `_collect_secant_pairs`、`_track_analytic_model`；不能等同于 history500|
|步长控制|F 默认 QN/trust radius；Safe-total Armijo|[External] F minimum 默认不是 RFO；更复杂代数成本需实测|
|几何预处理/对称曲率|预处理改变输入；对称扩充需目标不变性|[Analysis] 对固定 Gaussian/约束需另外论证，不原样迁移|

## 09 必要公式

[Paper] 两种调用指标为总调用比与逐体系调用比的均值；能量恢复指标比较初末能量降幅。[Paper: Equation 1] [Paper: Equation 2]

[Analysis] 对本项目，最直接的比较是同一完整目标的共同力证书及成本，而不是平均能量恢复指标。

设内部坐标为 \(q=f(x)\)、Jacobian 为 \(J\)。当目标可在这些坐标上表示时，链式法则为
\[
g_x=J^Tg_q,\qquad
H_x=J^TH_qJ+\sum_a(g_q)_a\nabla_x^2f_a.
\]
因此远离极小值时，单纯投影 Hessian 可能遗漏坐标曲率；更换坐标不是只换一个预条件矩阵。[Analysis，数学推导]

## 10 实验设计与证据链

[Paper] 优化器演化在 GFN2-xTB，冻结程序在未见分子及其他势上评估；主要成本结果以共同收敛的配对体系计算。[Paper: Sections 3–4]

本次完整图表清单（按 caption/表格及对应正文核对，未声称 PDF 页码或图形像素审查）：

|图表|论证用途|
|---|---|
|Figure 1|成熟优化器成本/能量比较|
|Figure 2|代码演化过程|
|Figure 3|模块移除|
|Figure 4|另一条演化路线的负结果|
|Table 1|未见体系，xTB|
|Table 2|其他势，含 DFT|
|Table 3|单体系能量差分布|
|Table 4|结构与振动资格|
|Table 5|成熟优化器完整比较|
|Table 6|成熟优化器终态比较|
|Table 7|G 版本负结果|

[Analysis] 这些证据支持优先研究 F；未包含我们的 Safe-total、SSW Gaussian/LS 目标或 VC 逃逸。本项目未运行 AutoSella，不存在自己的加速结果。

## 11 正确解释结论

[Analysis] 基础 Sella 内坐标收益、AutoSella 曲率改进、初态改动、终态盆地变化应分别归因。整包结果不能当作单组件收益；共同收敛子集的平均成本必须与全分母失败统计并列。

[Paper] 多片段结果可进入不同盆地，单模块移除效果也有相互依赖。[Paper: Appendix E Structural agreement with the reference] [Paper: Appendix G.4 Dependencies, results, and unsuccessful evaluations]

## 12 作者明确指出的局限

[Paper] 作者指出频率阈值只是实用判据；G 版本泛化不佳；模块贡献不能独立相加。[Paper: Appendix E] [Paper: Appendix H] [Paper: Appendix G.4]

## 13 批判性分析

|[Analysis] 观察|风险/竞争解释|如何区分|
|---|---|---|
|F 输入预处理|调用减少可能部分来自另一个初态|原始 F 整包与保留输入的类调用分开标注|
|分子坐标消去刚体自由度|外场、定点约束或方向偏置可能不具备相关不变性|检查完整目标梯度在被去掉子空间的分量|
|对称扩充|真实 PES 的对称性不代表偏置目标的对称性|核对 Gaussian/LS/约束是否随操作不变|
|高质量曲率需要更多代数|MH-1 小体系可能由 CPU/JAX 开销抵消少调用收益|分开计时 Calculator 和优化器；报告总耗时|
|真实势面最终淬火仅占部分成本|即使这一步更快，也未改变大量偏置淬火和方向评价|以本次 [成本读出](cost-report.md)确定下一步，而非直接跑长轨迹|

## 14 学到的知识

Agent-derived knowledge candidates：[Analysis] 建模坐标/曲率应优先于无目的增长历史；目标不变性是复用对称信息的前提；评价应分离预处理、算术开销、昂贵势调用与盆地终态。

## 15 与已有知识的连接

[External] Safe-total 是总梯度拟牛顿路径，未用 Gaussian 解析 Hessian 构造初始矩阵。AutoSella 模型曲率传输与这种情况不同；本轮不重新启用用户已否定的 bias-separated 方案。

[Analysis] LS 改的是用于逃逸的表面；优化器预条件改的是同一目标上的走法。两者可能组合，但改变局部求解轨迹也可能改变候选盆地，需要端到端验证。

## 16 研究候选

Agent-derived research candidates：**保存态的普通 Sella / AutoSella-F / Safe-total 配对对照**。

- 来源：已保存真实分子轨迹，局部淬火请求比例高。
- 核心假设：更好的内部坐标与模型曲率能减少同一力证书所需实际成本。
- 与原文差别：测试 SSW 产生的偏离平衡状态、MH-1 后端及最终有效盆地贡献。
- 最小方法：先真实势面非周期淬火；同一个输入、预算和共同力阈值；普通 Sella 隔离基础坐标收益。
- 如何验证：计入失败及计时、终态能量/结构/约束；通过后再检查 Gaussian/LS 的完整目标与几何条件。
- 可能失败：曲率先验不适用于断键/成键；额外代数抵消收益；进入不同/不合格盆地；偏置不变性不成立。
- 创新状态：unverified。本轮只是来源与可行性审查；尚未实现或运行该对照。
