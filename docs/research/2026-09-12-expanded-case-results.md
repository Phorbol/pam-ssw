# 更复杂案例的固定胞 SSW/LS 开发测试

本轮不修改生产算法和默认参数。依据上传examples/论文选题，加上明确的ASE结构，
执行3体系×2策略×2种子共12组配置。10条搜索触及预算，2条LS初始化不适用；
不存在“12条均完成20步”的结论。

## 输入与独立性

- 水15聚体：上传TYPE3 addition/add.arc，45原子。IfPer0决定非周期，50Å存储
  cell在新输入中置零。原示例H2O_pf.pot替换为GFN2-xTB accuracy .001。
- bicyclobutane：ASE G2 C4H6，10原子；采用GFN2-xTB。是论文相关强共价反应
  体系类别，非声称上传同一坐标/同一势。
- Cu55：ASE Octahedron(Cu,5,cutoff=2)生成，EMT。大于此前Cu13接口例，
  但不是指定论文结构的精确重现。

每组最多20 attempts/6000搜索请求/21fresh；每臂60秒。第一批全局240秒被
4条水轨迹用完，其余8臂记录not_run_global_wall。第二批仅补这8个零PES臂，
同样输入/搜索设置、全局180秒；没有重跑水，也没有调参重试失败臂。
原runner在evaluate前检查wall，并非强制中断单次后端调用，fresh也未受独立
硬中断；不能将计划秒数冒称实测绝对上界。该局限保留在冻结runner中。

首目录只做准备、v2导入tblite失败，均无PES；有效运行在v3及supplement。
依赖采用PYTHONNOUSERSITE=1、mace_env、tblite本地隔离包，CPU单线程。
具体config、结构、source、逐请求ledger与结果均保存。原runner未调用
checkpoint_path，本轮结果文件与ledger可审计，但不能据此声称中途checkpoint可用。

## 成本与完整性

根智能体独立审计：43,126搜索请求 +74 fresh尝试 =43,200请求。
74个存储观察中72个满足冷启动能量误差<=1e-7 eV、fmax<=.03 eV/Å以及
组成/cell/PBC保持；2个GFN2 fresh在250次SCF上限未收敛，未重试。
二者分别为C4H6/SSW seed11、C4H6/LS seed11的index5，均为解离结构，
不影响下表的最佳低能落点，但不得计为独立复核合格。

| 体系 | seed | SSW最佳ΔE / eV | LS最佳ΔE / eV | 共同前缀或限制 |
|---|---:|---:|---:|---|
| 水15 | 11 | -0.073656 | 0 | 1819搜索请求 |
| 水15 | 29 | 0 | -1.656550 | 1709搜索请求 |
| bicyclobutane | 11 | -0.306133 | -0.306290 | 6000搜索请求；小差值不作排名 |
| bicyclobutane | 29 | -0.422930 | -0.164227 | 6000搜索请求 |
| Cu55 | 11 | -2.176705 | 不适用 | SSW6000；LS仅5次初始请求 |
| Cu55 | 29 | -2.178115 | 不适用 | SSW6000；LS仅5次初始请求 |

ΔE相对各臂相同初始真实势淬火能量。水使用共同累计搜索请求前缀，全部最佳
变化在该前缀内发生；Cu不得把仅初始化的LS当6000请求有效对照。
本轮非独立最终评估集；两seed不足以给置信排名或GM命中率。

## 几何/化学结果

水：6个存储观察均独立合格，按既定共价图检查均保持15个H2O。LS seed29
找到显著更低GFN2结构，seed11则SSW改善；只支持“值得扩大检验”，不支持
普适LS优势、严格氢键拓扑身份或更正确物理势。

C4H6：共31个存储观察含4初态，27非初态中10个图示解离；这些不计为目标
分子的有效构象。SSW seed29找到butadiene参考连接图，SSW及LS seed11找到
methylenecyclopropane连接图。图采用恢复键长+.1 Å、元素标记同构，不含键级
和完整立体信息；因此只称连接关系命中，不称严格盆地/过渡态验证。
低能但未匹配五个参考图的结构保留为未知，不能直接判不合理或新异构体。

Cu55：两条SSW均发现更低EMT能量结构，35个存储观察全部独立合格。
几何诊断保留直径、最近邻最小/最大距离；尚未完成严格结构置换分类或Hessian
验证。LS两个初始结构也合格，但原版raw CuCu fallback截断1.875+.1 Å小于
淬火后最近邻约2.466 Å，导致零LS邻居。该结果揭示元素表适用域，不能据此
说LS搜索失败或自动把长度调到2.9 Å。原版真实输入过滤/用户表覆盖尚需核对。

## 对下一阶段的直接影响

1. 保留这三类困难作为开发集，停止增加小型Hookean重复实验。水与C4H6均有
   可辨别搜索结果；下一轮比较必须同时看真实淬火失败、解离、有效目标发现成本。
2. LS不应默认认为原始lookup对所有元素都物理适用。先核对真实LASP输入表/
   邻居过滤来源，再决定金属LS的有依据参数；不做单case事后拟合。
3. 先固定这些PES与proposal，对现有Safe-total和成熟优化器作必要对照，定位
   真正的失败/成本来源；本轮没有证明需要新增Gaussian策略或回退组件。
4. 下一批周期材料优先上传AlOH的26原子帧，选定适用后端后先固定胞；172/514
   原子复杂例后置。MACE-OMAT的可运行性与势的适用域应分别核验。

## 可复现产物

根目录：`research/ga_ssw/evidence/`
- `water-bicyclobutane-cu55-ssw-native-ls-20260912-v3/`
- `water-bicyclobutane-cu55-ssw-native-ls-20260912-supplement/`

每目录均有`root-cost-audit.json`和`root-structure-audit.json`，由
`audit_expanded_case_costs.py`与`audit_expanded_case_structures.py`生成，0新PES。
根成本审计核对连续请求号、结果/分阶段合计、预算拒绝不收费、fresh能量力与
几何证书；它不将保存次数当独立basin数。

未来runner已补checkpoint传递、独立能量/几何资格与SIGALRM/调用前时间检查，
未重跑本轮PES；历史冻结runner保持不变。这里只验证编译和preflight，
不将新runner控制功能追认到旧运行，也不保证Python信号立即中断阻塞的native调用。
