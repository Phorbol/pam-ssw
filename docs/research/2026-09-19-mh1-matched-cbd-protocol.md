# MH-1同状态旋转机制诊断：预定协议

## 问题与决策

承接C60同势面对照：当前Python的内层旋转成本高，但已有粗阈值/短偏置阶段消融没有一致收益。
要检查的是：恢复的CBD预旋转—有偏/无偏旋转组织是否在相同起点上减少成本，同时保留后续逃逸能力。
不是证明完整LASP复现、独立成功率或通用性能优势。本实验不改变核心默认参数。

输入为已完成Python MH-1/omol两条轨迹各自的初始淬火结构和最低能量结构，共四个开发状态。
每状态一个固定global随机方向（种子继承17093/17094），两臂共用并在该状态ClusterFrame内投影。
这不是在原有随机轨迹上精确续跑，也不构成未参与开发的独立验证。

## 固定条件与唯一被比较的机制包

- 基线：现有Euclidean Broyden，固定旋转秩一偏置1 eV/Å²，残差tol .02 eV/Å²，39端点+1中心请求上限。
- 对照：恢复CBD阶段控制，同样Euclidean度量、同样40次请求上限；含预旋转、动态秩一偏置和阶段切换。
- 两臂共用旋转坐标映射与力投影，避免现有不同调用路径的投影差异混入。该对照不是逐指令native几何复现。
- 两臂fd=.001 Å。实际LASP allkeys为dr=.005 Å、pre/main ftol=1/.1 eV/Å。
  由已恢复公式F_report=10*dr*||r||，native对应残差阈值20/2 eV/Å²；
  本对照换算后的报告力阈值为.2/.02 eV/Å。不要将.2/.02误写成HVP残差。
  PreRot/Rot上限为5/15，保持原版严格大于判据；不是有效力收敛证书。
- 一个Gaussian，宽度.6 Å；保留已有前向力规则
  w=(.1-F_parallel)*.6*exp(.5)，不人为固定同一高度。
  方向改变导致F_parallel和高度改变是该机制包的下游响应，必须记录，不能归因于单一停止阈值。
- 偏置及真实淬火均用Safe-total/history500，分别fmax .1/.03 eV/Å，每段1000优化步上限。
  direction_only：偏置/真实淬火不施加frame约束。偏置失败后不额外引入bare恢复。
- 每臂全部搜索/复核合计最多1500次E/F，8臂合计最多12000，1张V100、20分钟硬上限。
  不在运行中改参、不自动增加预算或重跑；所有失败、截断计入成本。

## 读出与下一步规则

保存公共方向、旋转终止原因/阶段轨迹、实际E/F、Gaussian参数、两次淬火的资格、落点及相对初态能量。
只有旋转更便宜、完整尝试总成本也降低且落点质量未恶化，才构成推进完整多Gaussian对照的理由；
仍需跨状态及后续真实体系端到端验证，不据四状态直接升级默认。
如果两臂均返回原盆地，单Gaussian试验不能判断全局逃逸，转向已存在的多Gaussian调用链对照，
不靠微调残差或挑选新随机方向反复制造改进。能量近似相等不自动证明盆地相同。
如果偏置淬火失败占主导，先检查已记录目标/力与终止原因，区分内层优化问题和方向问题。

## 来源与当前执行状态

参数出处：`research/ga_ssw/evidence/c60-mh1-native-20260919/seed17093/allkeys.log`第86、102–105行；
公式恢复见[CBD阶段与停止量](2026-09-17-cbd-stage-and-tolerance.md)。
现有驱动语义：`pamssw/standalone/paper_reference.py`的rotation_surface、高度构造与landing条件。
主agent与独立审查都检查了投影、单位、预算起点和失败语义，未改写验收标准。

2026-09-19 CPU分析任务提交被`AssocGrpBilling`拒绝，无job ID；GPU诊断尚未提交。
工作目录选择私人账户，未擅自切换组账户。代码/协议准备继续，计算资格待解除。
[提交错误](../../research/ga_ssw/evidence/c60-mh1-stage-preparation-20260919/submission-failure.txt)。

## 准备完成，尚未执行

权威待运行包：
[冻结输入、源码、配置及job.sh](../../research/ga_ssw/evidence/c60-mh1-matched-cbd-20260919-r2/)。
初次准备包只用于静态审查，从未提交，因JSON基本类型编码问题被r2替代，原包保留说明。
r2在首次执行前完成了输入校验及环境设置顺序审查，冻结runner摘要与实际文件一致。

主agent核验：四份输入能量与前序产物一致；111份Python源码及runner/输入清单匹配；
8臂总预算12000；`py_compile`、`bash -n job.sh`、`git diff --check`通过。
无PES的JSON类型往返与总预算拒绝边界检查通过。
科学验收尚未开始；不将准备、静态检查或fake evaluator检查称为真实体系有效性证据。

独立审查建议对displaced位置增加frame投影，主agent核对后未采纳：
现有`paper_reference`仅eckart模式设置frame，direction_only仅设置rotation_frame，
因此后续Gaussian位移/淬火保持自由Cartesian。保留已有行为，避免凭二手审查引入新的约束。

待账户解除限制后，按批准账户提交上述job.sh。不得在登录节点运行模型计算，
不得为了绕过计费限制自动改用另一账户。CPU phase-cost脚本已准备，未运行，不报告其新结果。

## 用户授权后执行

用户明确允许本次实验使用组账户sjtu-caoxiaoming；以显式--account参数提交1404774。
scontrol核实Account=sjtu-caoxiaoming、Partition=4V100、gres/gpu=1、TimeLimit=00:20:00。
原冻结配置和12000 E/F总上限不变；不是扩大预算或重跑旧轨迹。

实验已完成，见[结果与边界](2026-09-19-mh1-matched-cbd-results.md)。原冻结协议保留，未事后调整。
