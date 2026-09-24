# C60 局部缺陷输入与 MH-1 资格检查

日期：2026-09-25。用途：开发前的输入/模型资格检查，非搜索性能评估。

问题：在已有 C4H6 证据之外，能否构造来源明确、能够隔离局部成键重排的 C60 对照？随机云聚集同时包含成笼、解离和局部重排，不能独自定位 LS 对硬模的作用。本检查不替代随机 C60 的成笼及参考能量验收。

## 来源与边界

- Guan, Shang, Liu, *Local-Softening Stochastic Surface Walking for Fast Exploration of Corrugated Potential Energy Surfaces*, JCTC 2024, DOI [10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081)，§2.2 为五元环相邻 C60 到 buckyball 的机制依据。原文在外部 literature/215.txt；不把论文势面上的能垒移植到 MH-1。
- Liu, Jin, Liu, *Mapping structure-property relationships in fullerene systems: a computational study from C20 to C60*, npj Computational Materials 10, 227 (2024), DOI [10.1038/s41524-024-01410-7](https://www.nature.com/articles/s41524-024-01410-7)。Methods 明确几何优化为 B3LYP-D3/6-31G*，力阈值 0.02 eV/Å；6-311G* 是后续单点基组。论文未做全库振动验证，故不能称已验证所有结构为极小值。
- 后者正文指定 C60 #1 为 Ih；#2 是作者编号，未验证其与 Atlas #1809 的映射。只读取 SI zip 的这两个成员，先独立检查拓扑。原始下载及来源记录在 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/c60-defect-input-20260925/`，不将整个数据库复制入 Git。

检索范围截至本日，限裸 C60 异构体原始论文及公开坐标。排除氯化/氢化衍生物和由 C60 分子组成的团簇。图上的一次边交换只证明组合连接关系，不证明动力学路径或能垒。

## 固定顺序与停止条件

1. 零 PES 图审查：60C、三配位连通平面笼、12五环/20六环，1.64/1.7/1.8 Å 使用已有判据；#1 与 ASE Ih 同构、#2 不同构；枚举一次 Stone–Wales 型边交换并保存见证。不满足时先报告输入问题，不提交搜索。
2. 仅当上述通过，对两份原始几何分别做非周期 MH-1/omol float64 真势淬火。复用现有模型 `/home/gengjianrui/.cache/mace/mace-mh-1.model`；不加约束、LS 或 Gaussian。采用成熟 ASE LBFGS（memory=100, maxstep=0.2 Å, alpha=70，明确写出其常规默认），fmax=0.03 eV/Å 来自项目既有外层资格标准，最多200步/结构。步数是有界诊断上限，不是充分收敛保证。
3. 每个终点重置 Calculator 缓存并独立请求 E/F，保存原始/终态结构、优化轨迹、日志和调用数。上限402次淬火计算 + 2次复核，单 V100 最多10分钟，零搜索。异常保留，不自动重试或放宽条件。
4. 比较收敛、连接图保持、两结构非同构及能量差。若缺陷直接消失或结构越域，不用它测试越障；若保留，可成为局部 SSW/LS 对照输入。力合格不能替代振动稳定性，后续必要时先核查局部稳定性，不能直接宣称真正势阱或直接启动大预算搜索。

竞争解释：随机 C60 的困难可能主要是全局组装，也可能是局部刚性重排。这里先建立能区分后者的输入条件；单纯输入合格并不能区分两种算法解释。下一项搜索实验须另有配对协议及停止条件。
