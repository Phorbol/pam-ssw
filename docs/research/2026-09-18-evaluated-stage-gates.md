# 固定胞阶段控制：已评估状态上的可移植对照

本阶段承接内层预算与 PAM Gaussian 消融的负结果，检验原生多条件阶段停止是否值得吸收。
原接口已经存在，本轮不新增控制器或公共架构，不把实验变体称为原版逐步复现。

## 新增事实与证据边界

已有静态反编译、原指令谓词检查以及旧端到端对照均已查阅，避免重复证明同一件事。
原生 `climbstep` 从 1 开始、每次 optimizer dispatch 加一；首次/后续阈值分别 5/15，严格大于才停止。
原生正常 allstop 恢复 pre-dispatch work/energy；返回的下一试探坐标不一定已评估或被优化器接受。
这些事实已有证据，但当前 Python 监视器在每次 E/F 后检查，不能冒充同一个 dispatch 边界。

本轮尝试直接动态观察：交互环境和计算节点的 GDB `/bin/true` 均报 `ptrace: Operation not permitted`。
计算节点 job1378573 为 FAILED/1:0、1 秒，未加载模型、无 oracle 调用。未更改系统权限、保护代码或二进制。
动态观察暂缓；这不阻止运行原二进制，也不阻止已有指令级隔离模拟。

`research/ga_ssw/evidence/stage-gate-offline-20260918/result.json` 为零 oracle 的离线重放：
四个 strict 起点的第一外步，各偏置阶段的首坐标、终点偏置能量和力均从原账本及 Gaussian 历史重建吻合。
C60/17093 首阶段实际 52 次偏置 E/F，第 14 次已满足最大力分量 <.15；另一 C60 的首阶段为 48 与 10。
这些点可能是 Safe-total 试探点；不能据此计算假想后续收益，也不将 E/F 次数映射为原版接受步数。
脚本 `research/ga_ssw/replay_first_move_stage_gates.py` 保存可复算公式和输入路径。

## 固定的真实体系对照

复用 factorial 的四个 double-precision saved-minimum 输入、strict 配置、种子与冻结核心源码，以及已有基线结果。
唯一新臂使用既有 `StatefulNativeStageAdapter`：在已评估的 Cartesian 点上检查恢复出的标量条件，
返回有限已评估快照，再由原公共流程继续 Gaussian 或执行真实淬火。
使用原生记录的 force component .15、首次/后续消费阈值 5/15、三个 E/F limit=100；
multi_pes=False、起始计数 1。initial reference 映射当前 minimum、GM reference 映射 best，均显式记录。
这些引用及 E/F 消费计数是 Python 对照契约，不是原生整个状态变量生命周期的恢复。

每条最多 6000 搜索请求、100 外步、600 s，独立复核最多 101 帧；四条合计最多 24000+404 请求。
SSW 各数值参数与 strict 基线保持一致；不用 LS、PAM 高度、恢复方向或重连。
预期信号是阶段内请求与真实淬火次数改变；保留它的必要证据是固定成本下有效低能结构发现改善，
不能用阶段减少、返回快照数或接受数替代。失败/截断均保留，最终 fmax=.03 不放宽。
若收益不一致，不沿同一批种子扫描阈值，也不由此反推原版流程无用。

## 提交前发现并修复的力数组所有权错误

实际 Cu13/EMT 接口检查出现值得解释的不一致：部分正常偏置收敛没有触发更宽松的 force component gate。
检查能量/力传递发现 `_MonitoredSurface._last` 与返回给 SurfaceCalculator 的 `base_forces` 是同一数组。
后者原地加入 Gaussian 后，监视器又加一次，从而用 `F_base+2F_gaussian` 检查门限和生成快照。
能量没有同样重复，因此快照的能量/力不一致。这个 bug 在研究监视器，不在核心 Safe-total 或标准 SSW。

独立实际 ProjectedGaussian 复现：物理力 x=-.2，Gaussian 力 +.0980198673，
正确合力 -.1019801327，却记录为 -.00396026534 eV/Å；阈值 .05 会被错误满足。
修复仅将缓存改为 `base_forces.copy()`，隔离优化器的原地更新。
两项新增测试先失败后通过，原阶段测试加回归为 10 passed；root 另用实际 Gaussian 验证力与门控修复。
旧阶段控制结果报告已经明确更正；未使用该 adapter 的 24 条 factorial/PAM 结果不受此错影响。

`evaluated-stage-gates-20260918/` 保留未提交的旧快照并标记 superseded。
新系列 `evaluated-stage-gates-v2-20260918/` 保留相同科学协议和核心 111 个文件，仅研究监视器修复。
新的 Cu13/EMT 一外步检查用了 603 搜索请求，完成真实淬火且账本守恒；只是接口证据。
job1378698/1378699 已提交各一个 V100，最多各 12000 搜索请求。无默认策略或公共架构变动。

## 修复版真实体系结果与决定

job1378698/1378699 均已 COMPLETED/0:0，分别 2:42/2:25；四条均达到预定 6000 请求终点，
共 24000 搜索加 27 独立复核，27/27 数值资格通过。没有 oracle failure；每条保留一次预算拒绝。
`analysis.json` 含账本对齐与阶段成本；water15 一次 nonpositive_height 的 41 请求保留为 record residual，未隐去。

| 起点 | strict 最佳 E (eV) | 阶段控制最佳 E (eV) | 相对 strict |
|---|---:|---:|---|
| C60/17093 | -486.001931 | -478.548330 | 高 7.453600 eV |
| C60/17094 | -480.058111 | -488.497816 | 低 8.439706 eV |
| Cu55 | -173.978129 | -173.978205 | 仅低 .000076 eV |
| water15 | -219.084709 | -218.741129 | 高 .343581 eV |

通过 `audit_evaluated_gate_forces.py`，从实际 MACE 力账本加各时点完整 Gaussian 历史独立重建了
420 个已停止快照的 modified energy、原子力范数、力分量和 force gate，全部与记录吻合。
无 post-stage/request 信息的终止事件保留为未复核项，不能冒充额外门控证据。
这是真实体系上的 force ownership 修复核验，不是原版运行时状态机核验。
两个 C60 仍未达到三种构图距离的完整笼条件或参考能量；water15 最佳点保持 15 个完整水分子、
没有 <2 Å O–O 接触，3.5 Å 的氧图连通。Cu55 最佳点与初态同标签对齐 RMSD .00501 Å，无新盆地证据。

结论分开记录：研究监视器错误已修复；正确门控会显著改变 C60 轨迹；当前没有跨起点一致收益，
不能升级默认或归因原版优劣。两个 C60 的效果方向相反，不能挑其中一个代表算法。
现有科学判断应优先依据修复后的证据，而不是继续引用旧 stage-control 的门控统计。
原生精确 pre-dispatch 状态恢复仍受动态调试限制；现有隔离指令证据不自动补齐完整进程生命周期。

验证命令：`pytest -q research/ga_ssw/test_native_stage_quench.py research/ga_ssw/test_native_stage_quench_adapter.py`
为 10 passed；独立实际 ProjectedGaussian 数组与门限检查通过；四条原始账本和冻结文件逐项核验通过。
除上述研究监视器和回归外，新增内容为实验/分析脚本及研究记录；核心默认和公共 API 没有变化。
