# 当前缺口：实现、替代方案与效果证据分开

源码基线：14b03c1，2026-09-26。目标仍是独立 Python/ASE 的有效 SSW 家族，
不是复制 LASP 全部数值轨迹。本次核对现有源码和归档，不把旧报告再次当作新实验。

| 项目 | 当前核实结果 | 下一决定 |
|---|---|---|
| 完整 Q 描述符梯度方向 | 独立 walker 仍只支持 Q-off；S1 是研究原语，不是完整 S1–S6 选择器 | 保留缺口，不优先扩建：既有四起点强制 Q 对照收益混合，按原停止条件执行 |
| globalcompress | 多段方向公式及加权消费者已恢复；没有接入当前完整方向控制器 | 非坐标压缩，也不保证防碎裂；不因原版有此选项就推广 |
| CBD 历史/旋转 | 已有恢复的阶段、角度限制、重试和历史实现；新阶段 reset 已核实 | 独立算法替代与完整 native recurrence 不同，不再笼统写“CBD 未实现” |
| LS 周期表状态 | `NativeLSCycleState.advance` 已包含 save_zero/restore，runtime 确实调用它 | 旧 normal-update helper 的说明不是整个 LS 缺失证据；不重写已实现组件 |
| ASE 约束与完整方向/池 | `run_constrained_ssw` 已按批准方案接通非周期活动坐标完整方向及schema2恢复；Cu13/EMT普通与native LS、C4H6/MH-1点约束资格完成。`run_ssw`已有非周期原子对Hookean及约束身份检查 | 受限完整方向缺口关闭；受限池、周期选轴仍未接入，不推广搜索收益 |
| VC/RC | block/joint、RC/RC-VC 均有可运行独立实现；这些入口缺少固定胞同等恢复/池功能，native cell lifecycle/RC 映射仍有差别 | 明确保留功能缺口，暂不扩到主线外的新状态框架 |
| GA | 原子/分子/周期/表面入口和 active-walk 恢复均已实现；论文 DCCD 与 Java 公式身份有差别 | 暂不扩性能试验或复制 Java 调度常数 |
| 搜索效果 | 随机 C60 成笼与参考能量联合验收未达成；LS 和方向机制收益不一致 | 不把接口测试、短缺陷修复或 LJ 结果称为全局验收 |

## 已排除的资料阻塞

用户确认 VC-SSW SI 只有几个结构坐标，故不再把它列为算法细节资料需求。
SSW/CBD/BP-CBD/LS/RC/GA 正文已归档；VC 正文在相邻
`ga-ssw-behavior-parity/literature/benchmark-sources/vc2014/`。
2018 PTSD 原始论文 C8SC03427C 已确认 Europe PMC fullTextXML 可访问；
它不能自动补足二进制 Q 选择器/周期映射契约。

S1 指令验证实际已完成：CPU1447836 十项算术检查，CPU1447868 四项 pytest。
以 [原始产物和后续记录](../../research/ga_ssw/evidence/s1-cutoff-contract-20260922/README.md)
为准；旧 candidate 的“未执行”状态已失效，不重复申请计算。

## 原验证决策（历史，后续已完成批准的入口增强）

Hookean 是附加势，不等于删掉活动自由度。原子对 Hookean 仅依赖距离，与
完整团簇方向所用的平移/转动不变性兼容；固定点/平面 Hookean 一般不兼容该假设。
先只核查已有原子对 HookeanSurface + 完整方向 + checkpoint 的组合，不引入
新的防碎裂参数、不改变默认目标函数、不把效果测试变成接口测试。

若既有组合可运行，剩余问题应具体写成公共入口/约束身份持久化缺口，而不是
“算法不支持约束”。普通 SSW checkpoint 本来就不保存 Calculator 身份；
不能把传入不同 surface 的恢复行为错误归因于此次 Hookean 组合。

独立源码审查支持先复用这一外部 surface 契约，不为手动组合扩展 checkpoint。
自动读取 `Atoms.constraints` 并拒绝约束配置变化属于另一项公共入口增强，
不是组合可用或断点恢复的前置条件；本轮不改持久化格式。

依据：`paper_reference.py:441`、`constrained_reference.py:335`、
`ase_constraints.py`、`native_direction_control.py`、`native_ls.py:232`；
[Q 停止决定](2026-09-21-forced-q-results.md)、
[压缩方向闭合](native-globalcompress-geometry.md)、
[CBD reset](2026-09-17-cbd-history-reset.md)。

后续：用户批准自动入口与持久化扩展，已完成82项回归和Cu13/EMT四路径验证；
见[验收记录](../../research/ga_ssw/evidence/hookean-entry-20260926/report.md)。
上述“不改持久化格式”仅对应批准前的手动组合核查。

2026-09-26受限方向更新见[验收记录](../../research/ga_ssw/evidence/constrained-direction-20260926/report.md)，后续事实优先于上表原基线。
