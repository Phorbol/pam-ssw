# 完整方向与原版启发高度：一次逃逸的组合检查

## 决定

完整方向控制与既有ConservativeNativeHeightPolicy可以共同完成真实体系的一次SSW逃逸，当前没有接口或数值衔接阻塞。收益依赖起点：保留为可选组合，默认策略不变；不把它推广成“原版高度总是更好”，不继续在这五个开发态调参。此次只验证组合行为与一次逃逸的成本/落点，不提供长程PES覆盖、C60成笼成功率或通用效率排名。

## 为什么做

此前只分别验证了完整方向和旧旋转下的高度变化。零PES同状态检查显示当前固定前向力与原版87°条件并不等价，且原版启发高度还含首高度5.6 eV及旧历史重写。[同状态推导与791点读出](2026-09-21-pool-policy-pilot-results.md)。因此保持方向控制不变，只改变既有高度策略做一次受控组合检查，不新增规则。

## 固定协议及资格

四个C60/MH-1 omol保存态（两来源各first/best）及Cu55/OMAT-small，seed、完整方向、per_atom域处理、Safe-total/history500、宽度、力阈值及MC均固定。两臂为当前forward-force高度与原版启发ConservativeNativeHeightPolicy(.5,.2,level1,max10,growth1/2)。每臂一个外步，最多3000请求/180秒及2fresh；没有重试或中途调整。

CPU1437060预检通过；GPU1437065在4分21秒完成10/10外步，无cap/exception。共5315搜索请求、4921实际calculate，加20次独立fresh calculate。CPU1437066读出errors=[]，10/10成本闭合；五对初始坐标与独立初始能量一致，20/20fresh通过。fresh检查的是初态和实际合格落点，包括MC拒绝的落点，而非仅最佳结构。

| 输入 | 默认/原版启发请求数 | 默认落点相对初态/eV | 原版启发落点相对初态/eV |
|---|---:|---:|---:|
| C60 17093 first | 359 / 421 | -2.167263 | -2.415446 |
| C60 17093 best | 436 / 468 | +4.345467 | +5.089808 |
| C60 17094 first | 827 / 840 | -0.001162 | -0.002438 |
| C60 17094 best | 877 / 477 | -0.000132 | -1.760345 |
| Cu55 | 269 / 341 | +0.000201 | +1.262059 |

后三位毫电子伏变化不作为有意义的性能优势。17094 best的候选在此一次试验中既低能又省调用；其他四对原版启发高度耗费更多请求。上坡候选不自动代表搜索失败：17093 best两臂均MC拒绝，Cu55原版启发落点也被拒绝，需区分候选落点与实际继续的结构。两臂的原始MC参数相同，不对拒绝结果另加奖励或补跑。

C60八个落点在三个既定键图阈值下均连通，均未满足目标富勒烯拓扑或参考能量窗口。几何筛查不证明Hessian正定、动力学通道或第一性原理准确性。CPU1437094完成独立几何补充（同目录geometry-1437094.json，20个含重复初态的保存minimum）：Cu55两落点在既有3.0/3.2/3.4Å阈值下均单一连通分量；最短距离分别2.374/2.358Å，直径9.576/10.002Å。未见该筛查定义下的裂解，不将其升级为一般物理稳定性证明。

## 证据路径和工程范围

运行包：`/home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/direction-height-composition-20260921/`，含plan、runner、输入来源、冻结源码/执行manifest、原始账本、结果和fresh。实际写入目录不同于初始委派路径，所有提交/分析均已按实际路径审查。

主读出：`/home/gengjianrui/bin/pam-ssw-worktrees/vc-qualification-audit/research/ga_ssw/evidence/direction-height-composition-readout-20260921/analysis-1437066.json`。原始结果不覆盖。root提交前修复实验runner误把climb字典当属性访问的问题，并以实际历史result做CPU schema预检；此缺陷未进入GPU执行。本阶段无核心算法改动，没有把实验包当作发布版本。

## 下一项需讨论的设计

连续run的状态已完成，但完整方向尚不支持跨进程checkpoint恢复。已有SSW checkpoint缺少方向pair/group/group_marker。建议先补完整方向恢复、保持池checkpoint禁用，不同时扩大任意策略序列化契约。[具体最小方案、备选和验收](2026-09-21-direction-checkpoint-proposal.md)。持久化格式变动按AGENTS第8节等待用户讨论；尚未实施。
