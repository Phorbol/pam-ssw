# 批量计算、多副本和局域并行：作用层与适用前提

2026-09-27；源码核查与推导，不是性能实验。主问题是降低找到合格新盆地的成本，
同时分别报告oracle工作量与实际墙钟。当前无新增PES调用/作业/依赖安装。

## 1. Batch inference可以先验证；batch relax不是直接替换Safe-total

本机mace_env为MACE0.3.16；`mace/calculators/mace.py:527–583`把一个Atoms转为
一个真实graph（可加padding graph），`593–655`每次处理一个ASE调用。变量名batch
不等于同时推理多个真实构型；遍历models则是模型委员会，不是多构型batch。
源码版本与本机环境资格分开：本机MACE仓库HEAD
`59ad3a473ca02101a1cf02db242197e8d616dd11`，有无关setup.cfg本地改动，未修改它。
`mace/calculators/mace_torchsim.py:491–687`支持system_idx/ptr、多体系head、E/F/stress；
不能据此宣称当前ASE入口已经批量运行。

[固定TorchSim源码](https://github.com/TorchSim/torch-sim/tree/5540d893a849db9772081229a86d5f68537ad037)
为MIT许可证；`torch_sim/models/mace.py:1–47`重导出MACE自有适配器，
`autobatching.py:768–785,1037+`移出完成体系、补入新体系。其
`optimizers/lbfgs.py:263–313`明确采用有步长上限的固定步L-BFGS，不是Safe-total线搜索。
历史/力收敛/约束/stress契约必须分别验证；不能用“都叫L-BFGS”宣称等价。

最低风险入口：已存在的独立终态复核或离线保存态E/F，不改变SSW决策。
随后才考虑多独立walker当前待求值的构型，或既有GA候选的独立淬火。
一次CBD旋转后的下个方向、一次线搜索拒绝后的下个步长、后一个Gaussian依赖前一结果，
不能把这些未知未来构型预先当作独立batch；猜测分支的调用全部必须计费。
全流程批量化需每个walker保留自己的RNG、history、LS、Gaussians与预算，
请求服务负责收集/分发E/F。现有`ASESurface.evaluate`同步且calculator有可变缓存，
不能用线程共用一个ASE calculator声称实现；该服务属于待讨论的重要架构。

建议先独立研究脚本使用固定已有C60/MH1与slab/OMAT保存态（不同模型/head分批）：
比较B=1/2/4/8的E/F（周期还含stress）、构图+传输+forward+回传时间及显存。
固定模型/dtype/边界/约束；先正确性再计时，完整开销无稳定收益则不扩接口。
这些batch数是吞吐探针，不是算法参数；不许凭forward加速推广全SSW加速。
下一阶段同一优化器逐请求驱动、相同独立初态/种子，验证每条链的行为和总成本，
再讨论替换优化器。尚未执行此探针。

## 2. 多副本交换：有直接SSW先例，但必须识别交换对象

[Zhang、Shang、Liu 2013](https://doi.org/10.1021/ct400238j)官方摘要明确把富勒烯
发现与parallel replica exchange结合；本次只取得摘要，未取得完整温度/交换配方，
不把它自动等同于温度parallel tempering。2014VC原文讨论段（本地
`vc2014-author.txt:543–558`）还提出多seed、更稳定replica附近追加搜索和保持结构多样性。
这提示2013/2014整体效果不能仅拿单条SSW内核来承担；具体归因仍需全文/代码。

三个不同机制：独立多起点只增加并行覆盖；岛模型/池迁移交换候选；温度副本交换
按不同温度目标交换配置。不能把三者或模型委员会都称为同一种ensemble。
对共享真实目标E的两个严格canonical副本，交换率为

$$a=\min[1,\exp((\beta_i-\beta_j)(E_i-E_j))].$$

这是联合Boltzmann权重比的直接推导。交换规则正确不意味着现有SSW链具备canonical
不变分布：SSW有方向记忆、定向偏置和多对一淬火，其proposal比没有自动抵消。
目前可做优化启发式多温度搜索，但不能宣称平衡分布、自由能或真实动力学。
参考[Chodera/Shirts2011](https://doi.org/10.1063/1.3660669)的目标分布与状态更新契约。

最小因果对照应是K条同预算链不开交换 vs 同K条/同温度/同初态链开交换；
两者均按所有副本总E/F与wall计费。不能用K条交换链对一条低预算链证明收益。
优先在LJ38测跨漏斗与回访/碎裂，然后真实C60；尚未冻结温度梯度、交换间隔或K。
状态处理（只换结构且明确重启，或迁移整条方向/LS状态）必须在设计阶段讨论。
不把minima能量间隔直接等同实际逃逸势垒。

## 3. 动态MC温度与ITS：两条不同的路线

改变外层T只改变已有候选的接受率；不直接改变当次内层偏置路径。
如果主要费用耗在生成碎裂或重复的候选，提温不会凭空生成好proposal；若有大量有效
跨盆地候选被拒，才有选点层面的依据。项目C4H6与C60已有相反的接受尺度证据，
所以禁止把一个温度规则当统一修复。ASE MH的MD动能温度与Ediff接受窗也不是SSW T。

对固定正权重w_k与正温度，ITS混合势可写为

$$U(x)=-\beta_0^{-1}\log\sum_k w_k e^{-\beta_k V(x)}.$$
$$\nabla U=a(V)\nabla V,\quad a(V)=\langle\beta\rangle_V/\beta_0>0,$$
$$\nabla^2U=aH-\mathrm{Var}_V(\beta)\nabla V\nabla V^T/\beta_0.$$

前式来自ITS混合分布；后两式是本项目推导，条件为固定权重/单一标量V，
依据[You、Li、Lu、Ge2018 Sec.IV–VI](https://arxiv.org/abs/1806.00725)。
因此在V的驻点，Hessian只被正标量缩放，模式本征向量与条件数不变；连续裸势梯度
流的曲线也仅改时间参数。这不证明ITS无效（有噪声MD/有限步优化/添加Gaussian会不同），
但说明ITS不能直接替代LS的选择性谱改变，且需要额外权重/温度控制。
温度依赖的统计定理不能直接搬到历史依赖SSW。当前排在多副本机制筛查之后。

## 4. kMC的局域性：先并行生成候选，不直接合并空间更新

[EON KDB](https://github.com/TheochemUI/eOn/blob/9f8ef20e64ad5074b732a4d9331460fac49d5796/docs/source/user_guide/kdb.md)
值得学的是局域事件种子/环境匹配与重新收敛，不是未经检查复用势垒。
[Arampatzis等的并行kMC](https://arxiv.org/abs/1105.4673)使用Markov生成元空间分解
及受控误差；这套证明前提不自动适用于SSW。

若局域更新A/B影响的能量项完全不相交，才可能有
$$\Delta E_{A\cup B}=\Delta E_A+\Delta E_B.$$
全pair LJ无严格有限截断，联合cell变量是全局耦合；有限截断message-passing MLIP
也要按多层有效感受野/共享能量项判断，不能只用一个neighbor cutoff。
局部位移之后的全体系淬火会传播应变，破坏独立性。长程静电模型更不能套用。

可行的更小实验是同一slab母态不同局部区域独立产生候选、批量评估、逐个验证；
它没有并行接受同一构型上的多个更新，也不赋予SSW物理时间。
只有先测到局域事件与足够多低耦合区域，再考虑冲突图着色/缓冲区/最终全势验证。
对目前38/60原子团簇和54原子CuO slab，空间拆分通常缺少独立区域；230原子PdO
需先量化而不是预设。该判断是规模/耦合推论，尚无新实验支持性能。
