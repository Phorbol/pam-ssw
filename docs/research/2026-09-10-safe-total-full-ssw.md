# Safe-total 接入独立 SSW：完整 Cu13 轨迹结果

同一 Cu13/ASE EMT 初态、seeds 3/17、Ritz/dimer 各15步，方向使用每个 Gaussian 中心的
刚体补空间投影，偏置淬火保持自由 Cartesian。仅换全部淬火数值后端，SSW walker 仍为独立 Python。
SSWConfig 新增显式 quench_optimizer='safe-lbfgs-total'；原默认 ase-lbfgs 保留。
Safe 复用 PAM Relaxer 数值内核，不调用 PAM walker、LASP 或 Java。

## 测得结果

| 指标 | ASE LBFGS | Safe-total |
|---|---:|---:|
| 外层步骤 |60|60|
| 失败步骤 |31|1|
| 搜索 E/F 请求 |57671|32517|
| 返回真面结构（含4个初始）|33|63|
| 严格结构指纹组 |12|14|
| 接受的跨严格结构组移动 |0|0|
| 严格验证额外 E/F |3879|5263|

Safe 返回63个结构的新计算器真力复核全部通过（额外63请求）；严格淬火 fmax=1e-5 后全部通过。
14个代表的去刚体内部 Hessian 在差分步长1e-4和5e-5 Å下均正定。
结构身份采用全体排序原子对距离的最大差1e-4 Å；这是非单射的结构指纹分类，不能声称穷尽basin。

共同已完成结构调用预算为7483（四条两后端轨迹总成本的最小值，仅由成本选择，非预设预算）：
ASE各条覆盖3组；Safe按17-dimer/17-ritz/3-dimer/3-ritz覆盖6/6/8/6组。
仍报告所有完整运行成本，跨过预算才完成的landing不计入该前缀。

## 结论与下一步

这支持把 Safe-total 作为独立 SSW 的工作后端：在该 Cu13 任务中，完整轨迹失败显著减少，
搜索请求减少约44%，严格可辨认结构覆盖增加。它并未带来接受的跨basin迁移，也未发现更低能量：
初态能量约9.36136 eV，新结构均更高能，在300K的当前选择规则下被拒绝。
因此局部优化稳定性与全局routing效果必须分开。继续LS/GA闭环是有依据的下一阶段，
不应继续为这个低能Cu13起点堆叠局部优化器参数。

这个体系曾用于选择Safe后端，结果不是独立跨体系泛化证据。也不是与native优化器的对照；
native机器码kernel结果单独记录。未变更稳定PAM生产分支或默认后端。

## 复现实物

- research/ga_ssw/compare_cu13_safe_total.py：完整脚本，目录含运行前plan、代码快照、全部记录和真力检查。
- research/ga_ssw/evidence/cu13-safe-total/：4条完整轨迹，summary.json complete=true。
- 同目录 strict-validation/：严格淬火与Hessian证据。
- research/ga_ssw/summarize_cu13_optimizer_search.py 与 comparison.json：成本与结构分类汇总。
- tests/standalone + tests/reproduction：152 passed；ASE/NumPy shape弃用警告，非科学验证结论。

适配器保留原 ASE atoms 的cell/PBC/metadata，数值坐标不自动wrap，避免改动固定Gaussian的坐标分支。
不支持 Safe+Eckart 固定截面组合，明确拒绝；direction_only可用。LS预淬火、Gaussian+LS目标和最终
真淬火均透传后端选择；该接口覆盖已经测试，LS科学效果仍待真实分子验证。
