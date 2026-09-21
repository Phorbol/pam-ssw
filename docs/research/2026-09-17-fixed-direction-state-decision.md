# 固定胞 SSW：需要区分论文方向规则与上传 LASP 的状态规则

状态：用户已批准共享 ASE 驱动下保留两套明确方向规则的设计边界；
要求先完成理论分析。尚未更改生产方向规则。
目标仍是独立 Python/ASE 的 SSW 系列。这里不要求逐位相同，也不声称原版更高效。

## 新增证据

`probe_native_direction_update.py` 执行上传 ELF 的 `update_mode0`，只替代
同形状内存分配，在原始 `gen_randommode` 入口停止；原始归一化函数照常执行。
输入覆盖 N=2/5、两种相反的旧方向，均使用非零几何差。
4/4 通过，实际结果见
`../../research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/direction-update.json`。

核验分支为 modelevel=0、不压缩、不做约束投影。原版将 n0 覆盖为

```
normalize(current_positions - selected_trajectory_record.positions)
```

再把它和十项系数传给 `gen_randommode`。输入旧方向的正负未改变该输出。
这证明 generator 的输入是新位移方向，**没有证明 generator 最终输出就等于该方向**。
当前测试选一个非零索引轨迹记录，尚未核验全搜索中记录选择、周期映射与零位移分支。

系数也由原指令产生：本分支 c4..c6 复制 control+0x178..0x188，
c9=1.2*sum(c0..c8)；输入 (.2,.3,.4) 得到 c9=1.08。这是受控诊断输入，
不是推荐参数或实际默认。调用表 +0x1a8 的目标是 0x5d5c50 `gen_randommode`，
不再标作“未知记录回调”。主要地址：0x5d5b16–0x5d5b39 为坐标差写入，
0x5d5c02 归一化，0x5d5c15 模式生成回调。

前一轮已原指令核验的 `status != Allopt` 重入条件仍适用。
[历史重置审查](2026-09-17-cbd-history-reset.md) 的静态链进一步说明新 CBD stage
逻辑清空 active Broyden history；保留静态数组不等于跨阶段沿用 secant 历史。
主 agent 复核了 BRIONS4 input_step、ITER 与 BRZERO4 首迭代分支；纠正了审查稿
将字符串比较操作码写成 ecx 的笔误，实际是 r8d=3，ecx=6 是字符串长度。

论文 2013 的从初始随机 N0 重定向，与该上传版本的轨迹相关输入是不同的
算法契约。不能把它们无提示地混成一个“SSW 默认实现”，也不能据此推断谁更好。
官方教程将 Run_type 5 列为固定胞 SSW，既有上传案例 allkeys.log 也记录 5；
[官方教程](https://laspmol.lasphub.com/tutorial/LASP-2.pdf) 仅用于运行类型背景，
上述状态结论全部来自本地 ELF，不依赖网页摘要解释。

## 最小设计提案

建议保留一个 ASE 搜索驱动和共享能量/力、Gaussian、淬火、选择与预算接口，
给方向控制明确的两套完整语义：

- 论文参考规则：维持现在的 outer anchor 和现有验证结果，作为可复现基线。
- 原版恢复规则：由独立、实例持有的逃逸状态负责轨迹参考、方向生成和 CBD
  阶段切换；Broyden 状态归属单个旋转阶段。继续恢复 generator 后才宣称该规则可用。

不把所有差异拆成可任意组合的开关，不复制整份 SSW/LS/GA 驱动。LS 提供的
软化势通过现有求值接口进入两者，ASE Calculator 的接口保持不变。

备选是直接覆盖现有规则。这减少一项模式选择，但会使已归档结果和当前入口
改变算法含义；不建议。另一备选是另写一套完整 walker，会重复预算、失败处理、
约束和检查点逻辑，也不建议。

兼容要求：旧配置仍选论文规则；原版规则先显式选择、实验标记。若核心配置或
检查点需要新字段，必须迁移旧快照默认语义并保留来源，不静默重解释旧结果。
此核心状态归属/接口边界已获用户同意；具体恢复仍须满足下述证据和验收要求。

验收顺序：补全必要原版方向生成及零位移分支 → 原指令小片段比对 → 两规则的
状态转移/预算/恢复测试 → 固定输入和预算的多体系端到端验证。真实材料比较
沿用 MACE-OMAT-0-small；不会用解析势检查代替科学效果验证。

## 本轮检查

`python -m pytest tests/standalone/test_paper_reference.py
 tests/standalone/test_staged_direction.py tests/standalone/test_public_broyden.py
 tests/standalone/test_native_rotation_control.py -q`：38 passed。
原指令 direction-update 4/4 passed；没有新 GPU 任务、没有更改方向默认值。
