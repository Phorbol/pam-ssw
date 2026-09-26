# Ag30Au30：新合金案例的输入与势面资格

2026-09-27，探索/开发实验。此前C60局部路径与VC优化器诊断已收口，
不据此新增方向参数或重复旧GA/SSW面板。

## 问题和边界

全局优化的连续坐标自由度与合金占位自由度不同。Java启发的既有`exchange_atoms`
在保持坐标/总组成时做10N次随机索引对的物种交换；这改变物理占位，不是同元素重编号。
先问：公开合金几何在既有OMAT上能否形成可用于后续对照的数值合格、紧凑结构？
占位置换是否产生可分辨的该模型响应与局部落点？不在本轮判断GA、SSW或该算子效率。

竞争解释：原始Gupta结构在OMAT上仍能形成紧凑合金最低点附近结构；或者更换势面后
发生大形变/碎裂、优化不收敛，使它不适合直接进入后续比较。第二种结果不能归咎SSW。
该初态来自源模型低能结构，绝非独立困难随机起点或已知OMAT全局最低点。

## 可追溯输入

Rongbin Du, Sai Tang, Xia Wu, Yiqing Xu, Run Chen, Tao Liu (2019),
*Theoretical study of the structures of bimetallic Ag–Au and Cu–Au clusters up to
108 atoms*, Royal Society Open Science 6:190342, DOI10.1098/rsos.190342。
[补充数据v2](https://doi.org/10.6084/m9.figshare.9164993.v2)，CC BY 4.0；
`rsos190342_si_001.txt`第一组60原子Ag30Au30，源Gupta energy=-188.373819 eV，
源Rsuc=12/100。仅记录这些为源标签，不与OMAT能量或我们的成功率混合。
原文数据和下载metadata保留，prepare_input.py验证块边界/组成/坐标并记录行号。

## 固定有界资格协议

- 非周期60原子，全部可动、无额外约束。输入A为源坐标，B为同坐标经过既有
  `exchange_atoms`、NumPy default_rng(270927)处理的组成不变置换；不按结果挑seed。
- `/home/gengjianrui/.cache/mace/mace-omat-0-small.model`，omat_pbe，float64，
  CUDA，关闭cueq/oeq；PYTHONNOUSERSITE=1。不下载或引入新模型。
- 每臂用成熟ASE LBFGSLineSearch、history500、fmax=0.05 eV/Å、最多300步。
  不使用SSW/LS/Gaussian、GA排程、池打分；这一资格不构成新算法消融。
- 全任务最多1000次实际Calculator.calculate（初态、局部优化和最终reset复核均计入），
  程序10分钟/Slurm15分钟、单V100，不自动续跑或提高上限。先静态和输入核查再提交。
- 保存初末结构/E/F、逐臂轨迹、optimizer log、成本与退出原因。报告fmax独立复核；
  另外报告最短距离、包围盒以及3.0/3.5/4.0Å距离图分量敏感性，仅为结构诊断，
  不是成键模型或热力学稳定证明。不要把小力自动写成物理资格通过。

## 下一决定

如果程序、力资格或结构诊断失败，先识别层次并保留失败，不启动搜索面板。
若两臂数值合格且保持紧凑，才提出同势、同初始化、同总成本的局部SSW与非局部占位
操作比较；预算由本次每落点实测费用确定。不得由两次淬火的能量差推出效率优势。
任何重要公共搜索/持久化接口变化先按AGENTS与用户讨论；本轮仅研究输入/runner。

## 第二体系的预定资格（Ag结果后，Cu运行前）

Ag源与置换资格已在1504820完成48次E/F，两终态均力合格且三种距离图阈值均连通。
这只支持该案例可进入后续设计。为不围绕单一AgAu结构展开，选择同一原始SI的
Cu30Au30（4451–4512行；Gupta −199.346823 eV，Rsuc19/100）作为第二合金体系。
保持相同种类交换seed、阈值、优化器、模型、总1000调用/10分钟程序/15分钟单卡上限，
单独运行和记账。总资格上限因此为两任务2000实际调用，不重跑Ag、不以Cu结果改Ag配置。
