# Completed panel

GPU1499633 plus startup-only recovery1499684 completed. Twelve numerical tasks
qualified; one initial import-stage timeout remains recorded. Total1092 EFS.
Safe common-first-passage costs were26/45/102/62, ASE55/80/179/117,
SciPy26/50/103/64. No defaults changed or end-to-end advantage claimed.
[Full result and limits](../../research/ga_ssw/evidence/vc-mature-frozen-panel-20260926/report.md).

# 当前执行：0.05阈值下的变胞冻结任务

主对照为AlOH26和TiO2-phase87(48原子)，每个材料两项独立固定起点任务
（联合偏置松弛、无偏置变胞淬火），对照Safe-total/ASE/SciPy，共12项。
全部history500；偏置联合梯度阈值0.05、真实原子fmax0.05 eV/Å，残余应力
0.001 eV/Å³；原档0.005只是历史参数，不是本轮要求。

主审纠正了AlOH旧schema重构：必须使用归档原输入的晶胞作persistent chart，
首Gaussian中心是初始淬火结果在该chart中的坐标，不能用淬火后晶胞作新chart。
CPU1499611四项测试通过，两个材料零PES输入抽取通过。测试包括实际EMT的
六阶段接口/账本检查，测试通过不等于六阶段都收敛或三solver有效性已验证。

保留失败：CPU1499534的原单原子简单立方Cu预检在5分钟超时；45秒限时诊断
1499592定位到SciPy无界试探将体积压到4.7317e-10 Å³，卡在EMT周期邻居表构造。
同诊断中Safe两阶段均maxiter(各3请求)，ASE两阶段均evaluation_failed(各6请求)，
不是三算法收敛率对照。原夹具/源码/日志保存在preflight1-source及diagnose日志，
未声称该病态问题已修复。常规接口预检另用物理fcc Cu，仍检查原计数/缓存契约。

因已观测到单次Calculator调用可卡住，正式面板用12个独立进程，每任务120秒
硬限时(另5秒终止宽限)，一张V100顺序运行，总作业30分钟。独立进程只隔离失效，
不加体积/应力保护或修改算法；每项最多600搜索+1源物理能量核对+1独立终态核对，
共最多7224 E/F/stress请求。中断的在途请求单列，未完成项留在分母。源能量核对
不符时停止该任务；缺少历史biased梯度黄金值仍限制“历史精确等价”的声明。

执行入口为 research/ga_ssw/evidence/vc-mature-frozen-panel-20260926/run.sbatch。
以下保留最初固定胞审查；其暂停结论不适用于当前变胞面板。

# 2026-09-26 局部优化器机制面板：冻结偏置证据复核

## 决策问题与结论

问题是：Safe-total、ASE `LBFGSLineSearch`、SciPy L-BFGS-B 或 LASP 启发的原生 LBFGS，是否有一个当前尚未重复验证、足以改变优化器优先级的机制问题？本轮只复核既有脚本、报告和产物，不运行 PES、不提交作业。

**本节结论只覆盖固定胞和先前读取的冻结 LS 数据，不代表 VC/cell-relax 已有足够证据，也不据此暂停变胞优化器比较。** 固定胞证据只支持 Safe-total history 500 在若干局部冻结偏置淬火中的残差改善；不支持完整搜索普遍收益或后端替换。变胞目标及物理应力证书另见 [VC 冻结优化器面板](../../research/ga_ssw/evidence/vc-mature-frozen-panel-20260926/plan.json) 与下文之外的独立协议。该面板专门测试共同变胞偏置目标、accepted-iterate 原始 force/stress 与末端无偏置 cell-relax certificate。

## 证据与适用边界

- Fe7C3 冻结 LS 案例在两个种子、`ls_all`/`ls_filter` 两种已保存偏置下分别比较 Safe-total history 10 和 500。总计 8 个局部重淬火；history 500 四例末态梯度范数均更小，其中一例达到既定收敛标准（243 次请求），history 10 无一收敛且每例都到 300 步。每对的调用数差异并不总有利于 history 500（例如部分双方都截在 300 步），所以它是局部机制证据，不是普遍成本保证。四个失败起点的重算偏置梯度与归档源值一致到约 1e-15，支持冻结目标复现。
- CuO64 的两个失败爬升点也在相同冻结目标下比较了 history 10/500。history 500 两例分别用 166/195 次优化请求收敛；history 10 均到 300 步上限、分别用 312/318 次请求，未收敛。该结果加强“较长曲率记忆可缓解这两个点的局部停滞”这一解释，但两例同材料、同局部任务，不证明通用最优记忆长度、全局搜索收益或整体运行时间更快。
- 固定胞 SSW 三优化器基线覆盖 Cu13、Cu31、环丁烯和一个非 Ih C60 输入，每例仅一个外步/一个 seed。前三例的结果均在 Gaussian limit 结束、并非充分搜索；C60 三个优化器都因 GFN2 SCF 失败或运行 cap 结束。所有记录的 E/F 账本闭合，已保存落点的 fresh 力检查通过，但这一设计只验证接口和有限的一步行为。它没有相同冻结失败淬火点的 ASE/SciPy 重放，不能比较优化器解决特定失败机制的能力。
- 五个相同保存起点的 Safe-total/native LBFGS 回放中，Safe-total history 10、500 和原生 history 400 分别为 4/5、5/5、4/5 达到独立 force 标准；请求总数分别为 1883、1731、1964。逐起点有快有慢，不能排序。原生数值核报告记录到 1964 次 LBFGS 入口、3913 次 MCSRCH 入口，仅 6 次 MCSTEP 插值调用；这组数据没有显示其复杂插值分支是主要成本或收益来源。
- 已归档的 Fe7C3 原生核轨迹曾出现负 secant 后下降方向失效并以线搜索失败返回。此为具体失败证据，适用范围限于该隔离轨迹；已有分析已经说明不应照搬 `GTOL=900`，并建议保留 Safe-total 的正曲率筛选。重复做一组泛化原生 LBFGS benchmark 不会解除该局部机制边界。

来源：`evidence/fe7c3-ls-frozen-quench-history/frozen-quench-diagnosis/audit-summary.json`、`evidence/cuo64-frozen-quench-history/diagnosis/summary.json`、`evidence/fixed-optimizer-baselines-20260912/summary.json`、`docs/research/2026-09-17-recovered-ls-integration-results.md`、`docs/research/2026-09-17-safe-total-versus-native-lbfgs.md`。所有证据均来自旧工作树 `ga-ssw-behavior-parity`；本记录不把它们当作本 worktree 新运行。

## VC/cell-relax：已存在一类完整比较，仍需跨材料共同证书读出

Fe7C3-80 已有三种成熟数值实现对同目标的联合 VC 比较：相同 OMAT-small 快照、chart、压力、Gaussian 偏置、fmax/stress 限值、种子与请求预算。原始记录中三种实现均未通过严格 biased common-gradient gate；SciPy 的相对能量停止造成 406 次总 EFS 的提前退出，不能当作低成本成功。随后对保存的十个付费 attempt endpoint 作无偏置 cell-relax 和物理证书复核，共追加 835 EFS，十个结构均通过。这说明“严格偏置停止失败”不等于端点无法获得物理 cell-relax stationary certificate，但该跨阶段链式复核并没有隔离三种 solver 在同一个无偏置 cell-relax 起点上的成本差异。

因此此轮不是重跑 Fe7C3。新准备的两材料面板使用 OMAT-small 下已保存的 AlOH26 joint-VC 与 TiO2 phase-87 联合轨迹：biased 任务各自从其归档最后一个 Gaussian 的同一位移起点，固定同一全 Gaussian 目标；unbiased cell-relax 则由三种方法各自从完全相同的归档 `q_saved` 起跑，作为独立固定起点任务。它不从各自 biased endpoint 链式继续，因此能比较无偏置 cell-relax 的同起点行为。

旧 AlOH JSON 保存了每个 climb 的 `q`、direction、weight，但没有展开存储每项 Gaussian center；其 persistent chart 从原始 `input.extxyz` 建立，脚本按源代码状态转移用初始 quench 后 `result.initial` 在原 chart 中的坐标及前一 accepted `q` 重建中心，并严格截到选定 climb。无PES回归检查重建的 `result.initial` 几何和 selected `q` 对应 cell 与档案一致。TiO2 记录已保存完整 `record.chart_reference` 与 `frozen_gaussians`，脚本同样截断到选定 climb。源文件未保存 `q_saved` 处的 frozen biased objective/gradient，故不会声称 native replay parity；共同目标仍由同一重建式提供给三种算法，梯度公式有CPU有限差分测试。两组 `q_saved` 的原始物理 `E+pV` 会各用一个显式 E/F/stress 请求与归档 `selected.objective` 核对，原先整批计划另计2 EFS；当前独立任务各核对一次，共12 EFS。每个 accepted iterate 只从该点已付费 oracle 缓存读取实际请求号、体积、原始物理 fmax、压力残差全应力 max 及共同偏置 q 梯度范数；缓存未命中即失败，不补算。biased trace 同时报告 0.1 和 0.05 eV/Å 两个预设首达点，0.05 为本面板新阈值；源运行使用的 0.005 不会被混称为本轮标准。Safe biased 阶段按共同范数停止；SciPy/ASE 保留经VC范数换算的原生q-gradient/force停止，离线报告其首个共同资格点以及完整原生终止成本。无偏置任务各自报告 `fmax=.05 eV/Å` 与 `stress_tol=.001 eV/Å^3` 的共同物理 certificate 首达点；Safe 以该 certificate 停止，SciPy/ASE 的原生终止与物理首达分开列示；最终另做一次独立 E/F/stress 复核。

具体预算、执行脚本与仅供 root 审核的 CPU/GPU 命令见 [`plan.json`](../../research/ga_ssw/evidence/vc-mature-frozen-panel-20260926/plan.json) 和 [`run_vc_frozen_optimizer_panel.py`](../../research/ga_ssw/run_vc_frozen_optimizer_panel.py)。六个方法×case 组合各跑两个固定任务，共 12 个 optimizer stages；Safe、ASE、SciPy 全部配置相同 history=500，不扫描参数。最大 7200 search E/F/stress requests 加 12 次独立 endpoint E/F/stress。它是冻结目标的数值证书/成本诊断，不是全局搜索排名或统计效应估计；本记录阶段只生成协议和 runner，未执行 PES 或提交作业。

## 机制判断与仍存的不确定性

**观测：** 对已检查的 Fe7C3/CuO64 冻结偏置失败点，较长历史多次显著改善收敛残差/请求数；原生插值只在少数入口中实际参与。**解释：** LS/周期自由度混合的曲率尺度可能使短历史较快遗忘慢方向信息，是与结果相容的解释，不是已证明的因果机制。另一种解释是有限的冻结起点和 300-step 截止对 history 500 有利，换一类目标后优势消失。区分它们需要同一冻结目标上的历史截断/曲率诊断或独立目标复现；但 history 10/500 已在多个材料/目标复现过，而且本轮未发现当前决策必须知道的残差。

ASE LBFGSLineSearch 和 SciPy L-BFGS-B 是不同成熟后端，现有数据没有把它们放在同一个冻结的失败偏置点上，因此不能回答“哪个解决已知失败更省总 E/F”。不过这属于尚未测量，不自动构成值得测量：现有证据尚未定位到 Safe-total 回溯步长或曲率筛选导致的、而另一个后端可以避免的重复失败；一个泛化胜负面板也不会改变当前保留 Safe-total、把 history 500 留作候选的决定。

## 停止条件与重开触发

本轮优化器后端比较暂缓，不追加实验、不改默认、不补 LASP 功能。以下任一新证据出现时再重开，且只做能隔离该机制的冻结点回放：

1. 新失败轨迹表明 Safe-total 的具体 Armijo 回溯/步长缩小反复消耗请求，且同一目标下梯度仍有限；
2. 负或病态 secant 被 Safe-total 丢弃后，重复出现相同方向停滞，而有证据显示成熟后端在同点能够稳健前进；
3. 下一项已批准的论文级真实体系案例中，局部淬火明确成为总成本或落点资格的主导瓶颈。

触发后协议应在同一冻结偏置势、同一已保存失败几何、同一坐标/梯度/约束和同一 E/F 请求上限下比较 Safe-total、ASE LBFGSLineSearch 与 SciPy L-BFGS-B；只改变优化器，其余历史长度/终止力阈值保持固定。至少采用 Fe7C3 周期 LS 失败点和 CuO64 周期 LS 失败点两个物理案例，每例纳入两个已保存种子/点；保留所有失败和达到上限的项。报告达到共同 force 标准的请求数、实际 E/F 数、终止残差、线搜索失败/回溯数和总墙钟，不以迭代数单独代替成本。单臂最多 329 个真实 E/F 请求，四个冻结点×三后端的搜索上限为 3948 次请求，另计每端点一次独立复核；达到上限即停止，SCF/计算器失败不重试。若 Safe-total 未出现已定位的机制故障且其他后端没有一致减少总请求并改善资格，则停止该支线，不移植 LASP 优化器。

## 不可宣称

现有证据不支持：history 500 是普遍最优；LASP 比 Python 优化器更快；ASE/SciPy 更适合完整 SSW；MCSTEP 插值带来效率优势；任一局部 force 收敛等价于 Hessian 稳定、化学合理或全局搜索成功；这些案例证明适用于任意周期/非周期体系。C60 失败主要由优化器导致的说法也不成立，旧基线明确记录了 GFN2 SCF 未收敛。
