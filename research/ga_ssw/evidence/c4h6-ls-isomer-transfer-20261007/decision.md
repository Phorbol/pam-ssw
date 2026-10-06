# C4H6 论文异构体迁移：候选曲率核查结果

原始固定协议：[protocol](protocol.md)；[连通图与费用读出](readout-1662786/report-v2.md)；[精修/曲率固定方案](candidate-qualification.md)。

GPU1662853，源码090570f；4/4代表完成，594 E/F = 594 MACE实际计算。搜索本身未重跑。
Raw: `qualified-1662853/{summary.json,representative-*/{requests.jsonl,refine.traj,refined.extxyz,curvature.npz}}`。

| 图类与角色 | 精修后fmax eV/Å | 相对本组butadiene eV | 最低内禀曲率 h=.01/.005 eV/Å² | 两差分矩阵谱范数差 | 原图保持 |
|---|---:|---:|---:|---:|---|
| 3: SSW-only connected graph | 0.003882 | 1.209855 | 0.094436/0.094965 | 0.021645 | True |
| 2: LS-only connected graph for this start | 0.004811 | 0.403781 | 0.643098/0.643935 | 0.023900 | True |
| 1: shared butadiene graph control | 0.002631 | 0.000000 | 0.231625/0.231750 | 0.030804 | True |
| 4: starting bicyclobutane graph control | 0.004800 | 1.045925 | 1.341742/1.342549 | 0.035319 | True |

四例保持各自原连接图，两个差分尺度的24个内禀曲率均正，最低值大于尺度差；
支持MH-1模型上的稳定候选，不是连续极限严格证明，也不把混合C/H的刚度特征值当频率。
SSW独有图3含一个三碳环、环碳连接CH3及一条约1.286Å C–C短键，其图与论文Fig5的F
（1-methylcycloprop-1-ene）骨架相容；这是结构解释，未作独立电子结构鉴定。

## 对搜索的判断与决定

原四臂12步，搜索29251+fresh52；本次资格另594，总29897 E/F。
排除碎裂后新连通类（各自原起点以外）为cyclobutene SSW/LS=2/1，bicyclobutane=1/2。
对照中的SSW独有图与LS独有cyclobutene均通过曲率核查；区别并非全部来自宽松力阈值，
但两起点优势方向相反，单seed无法支持LS通用效率提升。保留LS为显式选项，不改变默认，
不因已有target响应接近.7就称理论已完整验证。两条LS碎裂C2H3+C2H3继续为物理目标失败。

本面板收口：不扩该12步面板种子/响应/温度/罚势参数，不追加TS/NEB或PBE对照来追逐正例。
下一项回到SSW的明确全局目标与成本验收，先以已知LJ55 GM目标检验当前方向组合；
C4H6原400步PBE反应网络与当前MH-1开发迁移严格区分。

[LS2024原论文](https://doi.org/10.1021/acs.jctc.4c01081)，本地215.txt Fig5/§3.1。
