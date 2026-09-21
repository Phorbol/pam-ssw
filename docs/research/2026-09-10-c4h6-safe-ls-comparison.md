# C4H6 / GFN2 的独立 SSW–LS 对照

**阶段结论：真实分子上LS反馈确实把预淬火响应推近预设目标，但本小预算对照没有
显示异构体覆盖或成本收益。** 两臂都到达trans和扭曲s-cis构象；LS调用更多。
另有三个小力合格落点实际已经解离，说明力证书不能替代分子完整性检查。

## 预登记与科学边界

输入为本机ASE G2数据库的trans-butadiene，10原子C4H6；不是作者坐标。目标是在同一
物理模型、同一初态附近比较真实驻点结构与异构体覆盖，不把输入已接近trans最低点
误设成寻找其全局最低点的任务。

原文：Guan等，*Local-Softening Stochastic Surface Walking for Fast Exploration of
Corrugated Potential Energy Surfaces*，DOI 10.1021/acs.jctc.4c01081。
实际读取本地SI `literature/ct4c01081_si_001.txt` §7.1–7.2：T=150 K、NG=25、
ds=0.1 Å、fmax=0.01 eV/Å、目标700 meV/atom。独立实现使用这些可对照设置，
数值预算relax_steps=400与原SI MaxOptstep=3000不同。后端是tblite0.7.0的GFN2-xTB、
accuracy=0.001，单CPU线程；不是论文PBE。

使用已经实现的Safe-total局部优化器、direction_only刚体方向投影及dimer方向求解。
fd_step=1e-4 Å、rotation_hvp=100、rotation_tol=0.02、rotation_bias=100、
forward_force=0.1沿用已声明的独立内核数值设置，不称其为原版参数等价或最优值。

**这是明确的混合参数独立实验，不是论文或发行版的数值复现。** LS生命周期、反馈
lambda=1.8、xi=0.2、初始比例0.03来自paper模式；完整C/H标准键能表则采用已保存的
native原指令raw查询：CC=3.4468400478363037、CH=4.29817008972168、
HH=4.526579856872559。cutoff使用同查询原长度+0.1 Å，分别约1.64/1.19/0.84 Å。
没有把发行版归一化表B当成pair振幅A，也没有使用其调度和幅度限制。详见
[LS有效振幅与响应审查](native-ls-effective-amplitude-response.md)。

seed固定3、17，各有SSW/LS两臂。先四条各1外层步测试成本，再从原输入和同seed
重新运行等长完整轨迹，最多10步；重复pilot成本单列并计入全部开销。
总墙钟上限600秒，按纯成本规则
`min(10, floor((剩余秒数-60)/四条pilot总秒数))`选完整轨迹步数，60秒预留验证。
所有参数在执行calculator前写入plan.json，没有观察结果后调参。

每次真实E/F请求均写JSONL，包含失败请求；每次局部淬火及LS预淬火响应另存。
每个完整返回的真实驻点记录采用新初始化GFN2 calculator独立重算能量与全原子力。
对连接图按元素标签做图同构比较，检查CC/CH/HH距离、连通分量和碳链二面角。
旋转平移对齐RMSD仅作为辅助几何量，不当作置换不变的basin认证。未计算Hessian，
因此小力记录不自动代表不同稳定极小值。

成本曲线只比较完整轨迹共同请求预算以内已完成的真实landing；跨预算才完成的
landing不进入前缀。完整轨迹实际开销、pilot及独立验证都另外报告，不用相同outer
step数量冒充相同成本。

## 复现产物

- Runner：`research/ga_ssw/compare_c4h6_safe_ls.py`。
- 分析：`research/ga_ssw/analyze_c4h6_safe_ls.py`。
- 目录：`research/ga_ssw/evidence/c4h6-safe-ssw-ls/`。
- `plan.json`、`input.extxyz`、`source_sha256.json`与`runner.py`保存预登记、原结构及
  代码来源；各轨迹保存全部evaluations、quenches、LS准备和最终result。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/tmp/pam-ssw-tblite-20260909:. timeout --signal=TERM 620s \
python -m research.ga_ssw.compare_c4h6_safe_ls
```

## 已完成结果

四个1-step pilot合计7774次搜索请求、19.66秒，成本规则允许随后四条各10步。
全部运行和独立验证于203.32秒结束，在600秒预算内；未改变配置、未重试失败、未调参。
完整轨迹40/40步有true landing，没有quench数值失败；独立新calculator验证完整轨迹
44/44个记录（含4个初始）满足fmax=0.01，含pilot则52/52。没有Hessian证明。

| seed | 方法 | 完整搜索请求 | 首次完整s-cis请求 | 11,435共同请求前的小力记录 / 连通分子记录 |
|---|---|---:|---:|---:|
| 3 | SSW | 11,435 | 1,630 | 11 / 11 |
| 3 | LS | 21,578 | 2,024 | 6 / 6 |
| 17 | SSW | 13,928 | 7,905 | 9 / 7 |
| 17 | LS | 22,038 | 10,256 | 6 / 5 |

表中记录包含initial且可能重复，**不是不同basin数量**。共同预算取四条完整轨迹
总请求最小值11,435，按预登记规则仅纳入此前已完成的landing。完整两臂搜索成本
分别SSW25,363、LS43,616；加pilot后总76,753搜索请求，另52独立验证请求，总76,805。
逐轨迹JSONL行数已与请求计数核对一致。这里计数为ASESurface E/F接口请求，包含
初始化/方向/偏置/真实淬火等阶段，不等于后端SCF迭代数。

LS两个种子的响应从相同的0.0031966 eV/atom开始，逐步到0.6957829、0.6948590，
与目标0.7接近。两者轨迹约为0.003→0.246→0.452→0.563→0.623→0.656→0.675
→0.686→0.692→0.695。此结果证明在本混合参数/GFN2体系内反馈具有预期的响应方向，
不等于这种软化强度有助于跨越我们关心的反应通道。

连通分子落点保持原始元素标注连接图，碳链二面角主要约180°与±18–21°；后者能量
约高0.0922–0.0924 eV，对应扭曲s-cis候选构象。这里按内坐标描述观察，没有把
±号、氢置换或微小能差分成多个独立极小值。没有发现新的连通骨架异构体。

## 小力合格的解离落点

seed17的三个落点连接图已断开，最短片间距离明显大于成键阈值：

| 方法 / 0-based step | 片段组成 | 相对初始能量/eV | 最大力/eV Å⁻¹ | 最短片间距离/Å |
|---|---|---:|---:|---:|
| SSW / 1 | C4H4 + H2 | 2.40478 | 0.00855 | 2.76085 |
| SSW / 2 | C2H3 + C2H3 | 6.11398 | 0.00934 | 3.58363 |
| LS / 2 | C2H2 + C2H4 | 2.03096 | 0.00879 | 6.07790 |

这些是实际GFN2轨迹的碎片候选，不作为新增连通C4H6异构体或稳定basin的收益。
未做分离极限、电子态或Hessian检查，不能进一步把它们认证为正确反应产物或动力学
通道。全部候选距离离本次分类阈值最近仍有0.060 Å，断开结论并非几乎压在cutoff上。
所有原始坐标/力/距离在fresh-checks.json中，便于后续独立审核。

## 对下一步的约束

保留现有Safe-total内核和显式paper模式接口；不把本试验转成通用LS默认设置。
没有理由因反馈已达标就增加目标或另加启发式。下一步若推进方法效能比较，应先
明确覆盖目标排除解离平台的规则、核验构象稳定性，并选择确实需要成键重排的可判别
任务及参考异构体；保持后端/成本可比。当前两个种子、10步仅提供真实流程与失效机制
诊断，不能作统计显著或跨体系优越性的结论。

最终汇总在 `summary.json`，逐结构独立检查在各子目录 `fresh-checks.json`，成本与
几何分析在 `analysis.json` 和 `geometry-summary.json`。纯读分析可运行：

```bash
OPENBLAS_NUM_THREADS=1 python -m research.ga_ssw.analyze_c4h6_safe_ls
```
