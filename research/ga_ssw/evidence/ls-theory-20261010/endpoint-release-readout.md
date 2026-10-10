# T2c五个偏置端点撤偏置核查

GPU1728694完成，220 E/F请求，215实际calculate；五端完整力均过3e-4门。
主agent复核完整几何、能量与相同编号图后，确认最高档仍通向原物理盆地：

|偏置端点|撤偏置身份|同编号proper RMSD (Å)|最大力(eV/Å)|
|---|---|---:|---:|
|a-16-easy_ts-minus|torsion_gauche|2.9e-05|8.38e-05|
|a-16-easy_ts-plus|torsion_trans|1.18e-05|7.21e-05|
|a-16-ring_ts-minus|torsion_gauche|2.8e-05|7.1e-05|
|a-16-ring_ts-plus|cyclobutene|2.5e-06|4.15e-05|
|a-8-easy_ts-plus|torsion_trans|3.48e-06|3.02e-05|

各匹配能差绝对值<1e-9eV，仅是同模型数值身份核查。a8 easy-plus图异常撤去
偏置后恢复trans；不能把偏置图越过cutoff解释成真实解离。a16两条TS一侧回gauche，
另一侧分别回trans和cyclobutene。本次端点核查不构成严格IRC或所有中间a的
分支无跳转证明，但足以解除预定T3一次逃逸试验的输入身份门。

失败也保留：1728686在hash辅助接口收到str而非Path时停下，0 E/F请求；
改为Path后，先对三个实际参考文件的几何/能量/hash接口做无PES检查，再用
新目录endpoint-release-v2运行。旧源与结果未覆盖，模型/科学参数不变。

原始与冻结输入/源码：endpoint-release-v2/run-1728694及source/finite/torsion/ring。
父失败：endpoint-release-v1/run-1728686。
下一项[一次完整逃逸开发对照](escape-protocol.md)，不再增加有限加载幅度。
