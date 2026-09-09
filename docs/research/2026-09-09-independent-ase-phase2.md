# 阶段二：独立 Python/ASE SSW、LS-SSW 与 TYPE3 GA-SSW

现在已有可执行的独立搜索循环，而不只是原程序包装器。公共入口在
`pamssw.standalone`：`ASESurface`、`SSWConfig`、`LSSettings`、
`PaperGAConfig`、`run_ssw`、`run_ls_ssw`、`run_ga_ssw`。
搜索不调用上传的 LASP/Java，也不调用旧 PAM walker。物理后端通过标准 ASE
Calculator 提供能量和力；它自身使用编译库或 DFT 可执行程序不等于调用原 SSW。

## 实现与来源边界

- SSW：2013 正文的随机混合方向、软化、累积 Gaussian、modified-PES 局部优化、
  退出、真面淬火及普通 Metropolis。每次旋转保持同一个最初随机方向作为参考。
  Gaussian 高度采用其引用的 BP-CBD 2012 前向力投影 0.1 eV/Å 规则。
- LS：冻结成键列表/r0、指数对势、独立 soft-only 预淬火、真实每原子能量响应、
  去偏置淬火，以及 eq15 跨步强度更新。键能与成键长度表显式输入。
- GA：非周期、固定内部单元 TYPE3，Python 切割/对接/三种变异、父代竞争、
  短程搜索、区域评分及精细搜索。所有阶段预算显式输入。失败、落点、父代、
  请求成本分别保留；只有 true/converged 且达到统一力阈值的结构进入档案。

这不是上传发行版的逐状态等价实现。明确的数值或行为差异包括：单边有限差分
Ritz 代替 native Broyden 旋转，ASE LBFGS 代替原局部优化器；论文前向力规则
代替后来 ELF 的87°规则；普通 MC 不含 ELF NSAME 加热；GA 无隐式3/4/5倍率、
carry与整数9规则。现有 NNA 描述符保留发行版已知的原子序依赖，不能把投影
档案大小当不同 basin 数量。不能将这个组合称为所有原程序分支均已完整复现。

来源细节与原文页码见 `ssw2013-paper-contract.md`、
`ga-ssw-audit/ls-rigid-comparison.md`、
`research/ga_ssw/GA_OPERATORS.md`、`research/ga_ssw/PAPER_GA_CONTROLLER.md`。

## 真实体系执行证据

| 体系与后端 | 本次完成内容 | 力复算与边界 |
|---|---|---|
| Cu13 / ASE EMT | 通用真面淬火与软方向连接；另有两步普通SSW实际执行 | surface探针最大力7.64e-4 eV/Å；不据近等能落点称新盆地 |
| 水15 / tblite GFN2-xTB | 初始淬火→quick→4个GA交叉后代→后代淬火→short→fine | 第04次完整流程139.72秒，4619搜索请求+10新Calculator复算；10/10档案条目通过0.01 eV/Å；2次biased quench失败保留 |
| C60 / tblite GFN2-xTB | 真实初始淬火→LS预淬火→2个Gaussian→真面淬火→MC/LS更新 | 52.55秒，总146请求（含7次显式初始淬火及2复算）；两个true落点最大力0.00517/0.00258 eV/Å |

水第04次10条结构按最近O分派，全部每个O分配到2个H，最近OH范围
0.9564–1.0062 Å。这是基本组成/几何检查，不是键级、Hessian或不同盆地证书。
水流程总状态为 `completed_with_failures`，失败是quick与fine中各一次偏置面
淬火未在预算内收敛。每阶段请求数之和与总计4619一致。

C60使用论文中的C-C键能3.61 eV、初始乘子0.03、响应目标0.02 eV/atom。
成键cutoff取本次真实起点三个近邻壳层与下一壳层之间的中点，配置记录实际值，
不伪称原版元素表。实测真实预淬火响应0.002124 eV/atom。当前短轨迹没有证明
离开C60原盆地，更没有证明LS效率提升。GFN2-xTB不是原论文NN/DFT势面。

产物在 `research/ga_ssw/evidence/independent-*`。脚本分别为
`run_independent_surface_probe.py`、`run_independent_water_ga.py`、
`run_independent_c60_ls.py`。每次水/C60运行保存脚本快照、配置、输入、结果、
所有成功E/F请求的坐标/能量/最大力日志；失败尝试计数也在结果中。
tblite==0.7.0仅安装到临时目录 `/tmp/pam-ssw-tblite-20260909`，未改变全局环境。

## 保留的失败及其诊断

1. 水第01次：3次接口请求失败。tblite源码设置energy==free_energy，但
   implemented_properties漏列free_energy；依据源码显式请求energy后重跑，
   没有向通用ASESurface增加静默fallback。
2. 第02次：603请求，3个初始BFGS均在200步用尽预算后未达0.01 eV/Å。
   第03/04次保持力阈值，明确把初始预算增加到800步。
3. 第03次：1547搜索请求+3复算，得到3条力合格档案；所有5个方向求解失败，
   GA首次100次配对也未命中。没有删掉这些失败以包装第04次。
4. 固定同一结构/初始方向的数值诊断表明，默认电子accuracy=1时曲率残差
   约0.225 eV/Å²。accuracy=0.001、dimer separation=1e-4 Å后，80-HVP预算内
   实际7次力请求便得到0.00881残差。提高的是数值精度，不是搜索奖励或策略。
   这符合有限差分误差包含力噪声/位移及单边截断误差的基本分析。
5. GA精确随机回放：300×300池有1218合法配对，100次全不命中的概率约25.6%；
   第一次合法对在第128次。G=4每批只产1个cross，因此2批也不可能完成4个候选。
   第04次提高配对/批次数预算，未改分子身份或配对规则。

这些是实现/数值贯通实验，不能将调整后的单次结果作为独立调参测试或效率提升证据。
请求计数是ASE接口请求，不是内部SCF迭代或成本完全相同的原生force evaluations。

## 尚未完成的范围

当前共同driver限非周期、无约束；GA限固定内部单元TYPE3。
TYPE0/其他原始体系分支、TYPE3可变内部单元、完整周期与约束、RC链、联合变胞、
可恢复的完整跨调用搜索状态仍需开发。原ELF逐状态对照及与PAM同势面、同初态、
同预算的多种真实体系性能比较尚未完成。现有测试与短流程不能升级为生产默认。

## 安装与检查

本阶段相关测试 `pytest tests/standalone tests/reproduction`：93项通过；警告来自
ASE 3.29 与 NumPy 2.5 的数组shape弃用，不是失败。Wheel构建曾因顶层runs/research
被自动识别为包而失败，已将setuptools发现范围明确限定为pamssw及子包。
最终wheel成功构建，并在 `/tmp` 下隔离安装后完成一次Cu13/EMT普通SSW，
确认导入来自安装包、运行不需要research目录或上传二进制。
