# C60 后端与防裂解约束：最小可行方案

## 问题与新证据

用户允许考虑其他势及防裂解约束。延续已有两结构交叉单点方案，未改核心代码、未启动新搜索。
MACE-MH-1/omol 固定几何结果（1386349，COMPLETED 0:0）：

| 结构 | 能量/eV | 最大力/(eV/Å) |
|---|---:|---:|
| OMAT 异常落点 | -61436.85495902111 | 1375.3419087495458 |
| 原 OMAT 参考笼 | -62214.86705137192 | 0.9168574493616477 |

同一模型内 ΔE(异常−参考)=+778.0120923508075 eV；OMAT 对应为约 -369.59 eV。
两模型的绝对能量零点/理论级别不同，禁止直接相减；这里只比较各模型内部的两结构差。
强烈支持 OMAT 在这个搜索区域的适用域失效解释，不等于已经获得 DFT 真值。
参考笼未在 MH-1 上优化，0.92 eV/Å 不满足当前数值要求；换势后须重新优化参考结构并定义能量靶标。

执行记录：第一次1386342因本站禁止显式cpus-per-task被取消，但进程仍留下两点结果；
1386343因拒绝覆盖该结果在模型加载前退出；修正调度参数后1386349使用新输出完成。
总计实际产生4次E/F结果，原失败记录与输出保留，不把取消状态当完成。
[输入、代码、结果与调度记录](../../research/ga_ssw/evidence/c60-height-anomaly-20260918/)。

## 势模型选择

1. **立即可检验的候选：MACE-MH-1/omol。** 本地已缓存、ASE可直接调用，这次避开了已发现的异常坑；
   尚未验证随机碳云路径或 C60 全局搜索，不把两点检查提升成生产合格。
2. **C60 专项对照优先候选：GAP-20。** 有碳团簇 AIRSS 结构搜索文献，模型/训练集公开，
   可经 quippy Potential 作为 ASE Calculator，并可接已有 LASP external 能量/力服务。
   本环境尚未安装 quippy；未安装、未下载模型、未运行。该接口可行性来自已有 Calculator 边界和官方接口，
   不是已完成的 LASP/GAP 联调。GAP-20 也有适用域限制，不能承诺无异常坑。
3. 传统碳键级势可作低成本补充基线，但不能凭速度取代候选间能量/结构物理资格验证；
   目前不增加多个势的扫描矩阵。后续更强核验可采用少量 DFT。

## 约束方案与适用范围

ASE Hookean 的配对形式在超过 rt 后增加拉回力；给初始邻居逐对加它，会偏向保留初始连接。
对从随机碳云生成笼、需要断键换键的任务，这种偏好不合适。
更小且已在接口支持范围内的方案是每个原子对同一个固定空间中心施加平底软球壁：

Uwall = (k/2) sum_i max(0, |r_i-c|-R)^2。

球内能量和力均为零；球外将原子拉回。现有 Hookean(a1=i,a2=point,k=k,rt=R) 可表达，
无需新增约束框架，但目前要走 run_constrained_ssw：普通 run_ssw 明确拒绝 atoms.constraints，
GA 尚未提供持久约束支持，不能直接往现有普通 C60 runner 加 constraint 就认为已接通。
现有 constrained 证书分别记录物理和 Hookean 能量/力。固定中心属于外部容器，影响整体平移；若改为动态质心必须完整求导，
不能在每次求值时偷换点坐标却漏掉质心依赖。
R 定义允许搜索的空间尺度，k 为 eV/Å²；不能从本次异常结构反向调出一组通用参数。
可用允许越界长度 δ 与边界能量尺度 Ewall 的关系 k=2Ewall/δ²设计候选，仍需报告尺度来源。

软球壁限制蒸发和远距离分离，不保证化学连接，不能纠正局部9配位/短距离异常吸引，
压缩过强还可能放大密集结构问题。短程排斥是另一项势能修改，不在本次擅自叠加。
任何约束实验均须区分原始模型能量与约束能量，最终移除壁重新淬火并检查完整笼、力及碎片。
Python与LASP比较必须使用相同辅助势/力和参考协议，不能仅给Python加约束后称纯算法对照。

## 下一项建议

先筛查并资格化新后端（参考笼、异常落点、两随机碳云及少量真实中间结构），
不立即再跑60000调用；优先MH-1/omol的现成检查，GAP-20作为碳专项对照。
在后端不存在当前明显异常的前提下，再独立比较无容器/软容器，不能把换势和加约束混成一个变量。
这是待讨论的研究设计；当前未加入任何新约束或替换主搜索默认后端。

## 原始来源

- Karasulu et al., Accelerating the prediction of large carbon clusters via structure search: Evaluation of machine-learning and classical potentials, Carbon (2022), DOI https://doi.org/10.1016/j.carbon.2022.01.031 。已核验摘要；出版商全文403，未声称完成全文审查。
- Rowe et al., An Accurate and Transferable Machine Learning Potential for Carbon, JCP 153,034702 (2020), DOI https://doi.org/10.1063/5.0005084 ; https://arxiv.org/abs/2006.13655 。
- 模型与训练数据 https://github.com/patrickwrowe/Carbon_GAP/ 。
- QUIP/ASE 接口 https://libatoms.github.io/GAP/quippy-potential-tutorial.html 。
- MACE模型发布说明 https://github.com/ACEsuit/mace-foundations/releases 。
- ASE Hookean 原始实现 https://docs.ase-lib.org/_modules/ase/constraints/hookean.html 。
