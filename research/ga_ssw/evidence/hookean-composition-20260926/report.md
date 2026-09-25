# 原子对 Hookean 与完整方向：现有接口组合通过

2026-09-26，功能资格，不是搜索效率或防碎裂有效性证据。
核心源码14b03c1；实际runner和提交脚本随报告保存，未改变生产代码。

CPU1493555：COMPLETED，0:0，墙钟5秒，1 task、调度分配2 CPU；
上限5分钟/6000 E/F。实际780次：连续388、暂停/恢复合计388、初态预检1、
终态裸EMT/包装surface/ASE直接约束各1。无预算截断，无GPU。
提交前runner AST、`bash -n check.sbatch`、`git diff --check`、Slurm test-only通过。

两条轨迹各2外步，完整恢复方向启用；记录、所有保存极小值/current/best、方向状态、
主RNG、调用成本严格相同。初态pair(0,1)的Hookean校正非零：
k=1 eV/Å²，rt=0.9倍原始距离，U=0.03391715 eV。这是接口压力检查，非推荐束缚参数。

|独立终态量|结果|
|---|---:|
|裸EMT能量/eV|9.361825991951306|
|Hookean能量/eV|0.0017123464839580828|
|合计能量/eV|9.363538338435264|
|合计最大原子力/eV/Å|0.01771019394994255|
|裸EMT最大原子力/eV/Å|0.0705189211770662|

包装器与独立ASE Atoms+EMT+Hookean能量/力在1e-12绝对容差内一致；
stored best与fresh合计能量一致；合计力满足0.03，裸势力不满足。
这是受约束目标的合格端点，不是裸势极小值；C60裸势验收仍须另外移除约束淬火。

改变k后的零外步恢复被允许，新增PES为0。这体现既有调用者重建相同oracle的
普通checkpoint契约；不是算法回归，也不是改变目标后的有效科学续跑。
必须保存同一约束配置，不能根据恢复后的距离重新计算rt。

决定：补充既有surface组合用法，暂不重构walker/checkpoint。普通run_ssw自动解析
Atoms.constraints仍未实现；FixAtoms/point/plane走既有constrained入口。
此次未验证LS或池选点与Hookean联合运行，也不外推其他Calculator或周期体系。

证据：[协议](README.md)、[runner](runner.py)、[汇总](runs/summary.json)、
[连续性](runs/continuity.json)、[端点](runs/endpoint-check.json)、
[目标身份边界](runs/identity-check.json)、[作业输出](cpu-1493555.out)。
