# 原生原子对 Hookean 入口与恢复验收

类别：实现检查/真实体系接口资格；不是独立搜索效果评估。
用户批准方案：[设计](../../../../docs/research/2026-09-26-hookean-entry-proposal.md)。

## 实现与边界

`run_ssw` 自动读取非周期原子对 Hookean，复用现有 HookeanSurface，内部清除
Atoms 约束避免重复计力；调用方输入不变。同配置预包装 surface 不重复叠加。
约束目标为 V+U，返回普通 Atoms 不附约束，保存的能量为合计目标。

仅带此原生约束时采用 checkpoint schema6，记录 canonical Hookean 配置和
base_schema_version（原能力版本1–5）。恢复前核对配置和原必需状态，在调用PES或
推进RNG前拒绝变更。无约束旧格式保持，旧无元数据检查点不能恢复成新增约束的任务。
手工包装且输入无约束的旧用法仍由调用方维护 surface 身份。
FixAtoms、点/平面Hookean及周期Hookean不属于本入口；已有受限入口不受影响。

## 检查与结果

最终 CPU1494025：82 passed in 2.64s。命令为 `sbatch --wait --parsable tests.sbatch`
加以下测试路径（脚本工作目录为本工作树，路径相对仓库）：

- tests/standalone/test_ssw_hookean_entry.py
- tests/standalone/test_ssw_checkpoint.py
- tests/standalone/test_recovered_direction_checkpoint.py
- tests/standalone/test_pool_direction_checkpoint.py
- tests/standalone/test_pool_checkpoint.py
- tests/standalone/test_ls_pool_restart.py
- tests/standalone/test_ase_constraints.py
- tests/standalone/test_recovered_rotation_integration.py

覆盖配置变更的提前拒绝、旧格式、错误metadata、完整方向/池必需状态、LS初始化失败
检查点、现有约束和恢复契约；日志 [tests-1494025.out](tests-1494025.out)。

最终 CPU1494085：`sbatch --wait --parsable qualify.sbatch`，COMPLETED 0:0，3秒。
Cu13/EMT，显式非零原子对附加势，两外步；普通与实际替代池重启各连续/暂停恢复。
[协议](runs-v3/protocol.json)、[汇总](runs-v3/summary.json)、[runner](qualify.py)。

| 路径 | 搜索请求 | 独立复核 | 连续/恢复状态与成本 |
|---|---:|---:|---|
| 普通连续 | 388 | 1 | 严格一致 |
| 普通暂停恢复 | 388 | 1 | 严格一致 |
| 池重启连续 | 244 | 1 | 严格一致 |
| 池重启暂停恢复 | 244 | 1 | 严格一致 |

合计1264+4；全部独立ASE EMT+Hookean能量一致，最大合计力
0.01771019394994255 eV/Å，小于预定0.03。能量9.363538338435264 eV。
实际比较包括记录、极小值、当前/最佳结构、RNG、方向、池和约束状态。
非零附加势的淬火合格不代表裸EMT淬火合格，也不代表全局搜索有效。

## 保留的失败、修正与实际源码

- CPU1493564：6失败/3通过，原入口不支持，作为实现前失败证据。
- CPU1493574：首版75通过。
- CPU1493577：ASE extxyz无法序列化Hookean的move_mask，PES前失败。
  原runner及部分输入保留在本地runs/；只改实验序列化，坐标和约束配置分别保存。
- CPU1493580：首版四路径通过，1264+4，见runs-v2。首版source delta/helper及runner
  在v1-source/，相对4c13652fea66752dd1dc94f04d2b3de57813bb3d可恢复。
- 随后审查发现schema6会漏掉缺失方向/池字段的必需性校验；CPU1493683两项失败
  复现。保存原能力版本后重用旧校验，产生上述最终82通过及runs-v3资格。

两次真实体系资格总计2528搜索+8次独立复核，无GPU。数值测试成本不含在该计数中。
最终源码为本报告同次提交的paper_reference.py、ssw_restraints.py及qualify.py；
原始大结果/检查点留在本工作树runs-v2/runs-v3，不放入Git。
`git diff --check`、脚本语法检查通过。独立代码审查未发现剩余具体缺陷。

## 决定

保留已批准入口；不改默认约束、模型、偏置/旋转参数或搜索预算。
没有从本次接口资格推出C60成笼收益、任意约束支持或生产算法优势。
