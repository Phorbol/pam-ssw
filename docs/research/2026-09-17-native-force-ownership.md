# 旋转调用者的力归属

目标：在共享 ASE 驱动内接入恢复的方向规则时，不能将被旋转子程序改写的
临时力当成同一结构上的真实力。这里仅记录原程序的数据生命周期，不声称搜索增益。

`soften_mode0` 入口 `0x5c38b2–0x5c3ada` 复制
`fa (object+0x1d0) -> work1 (object+0x9c8)`。
随后计算曲率并调度 unbiased/biased rotation。
`tf0 (object+0x1a60)` 是另一份中心参考力，不是这次保存操作的源或目标。

`probe_native_force_snapshot.py` 实际执行该入口和复制循环，N=2/5/15、两种力符号
共6例。分别初始化 fa、work1 和 tf0 为不同值；断言 work1 等于输入 fa，
fa 与 tf0 保持原值。6/6通过。仅同形状 Fortran 分配例程用等价 no-op 替代；
在曲率计算前停止，没有 PES、旋转或主程序调用。
产物：`research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/force-snapshot.json`。

对应恢复在 `unbiasedrot` 的 `LCONVERGE` 分支：
`0x5c423f–0x5c4488` 为 `work1 -> fa`。该恢复的动态验证由单独的
`probe_native_rotation_caller.py` 负责；入口保存探针本身不证明恢复已执行。

实现要求：中心参考力、最近一次实际求值的端点力与旋转工作向量分别持有；
正曲率预旋转结束后进入有偏旋转可复用相同端点的真实力，不能重复添加
上一阶段的旋转偏置。阶段复用不得产生虚构的 calculator 调用或 Broyden 观测。
