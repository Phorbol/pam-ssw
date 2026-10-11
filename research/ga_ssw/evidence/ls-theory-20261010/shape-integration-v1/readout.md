# 可选 shape LS 接入验证（2026-10-11）

用途：实现/状态契约检查，不是搜索性能证据。

用户已批准最小设计。新增不可变形状偏置与 `LSSettings.bias_geometry`，默认 pair。
shape 仅允许自由非周期无约束团簇；VC/受限入口显式拒绝。
检查点 schema 7 区分形状模式，保留既有基础能力版本和旧 pair 恢复。

CPU 作业 1739721：`cpu.sbatch` 中的六组 pytest，共 87 passed in 9.05s；
Slurm COMPLETED/0:0，总用时 70s（含环境启动）。原始 stdout/stderr 保存在本目录。
覆盖非参考几何的能量/力差分、旋转/伸缩不变性、重建尺度、旧格式和跨模式恢复、
池重启、Cu3/EMT 两步连续与 1+1 恢复，以及旧 LS/Hookean 契约。

限制：Cu3 是接口轨迹检查；真实 C4/C60 同预算完整外步结果尚无。
响应控制器对 shape 的跨步有效性尚未验证，不升级默认。
