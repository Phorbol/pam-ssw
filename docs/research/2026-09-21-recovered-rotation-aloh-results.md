# AlOH旋转策略对照：逃逸成本下降，低能结果基本不变

## 结论与决定

在同一组AlOH输入/种子和6000请求上限下，已有原版启发CBD旋转使完整逃逸从23次增至43次，旋转请求占比从72.7%降至30.8%，实际搜索calculate从22276降至20690。但两种策略发现的最低能量基本相同；新增逃逸不能自动称为新增盆地。保留为可选策略，不升级默认，不围绕AlOH继续调参。

下一步只核对已有C60长预算基线能否进行严格配对的跨体系旋转策略比较。目的仍为独立ASE算法的有效搜索，不追求LASP逐数值一致。是否扩算取决于输入/源码/配置能否匹配，以及可控计算预算。

## 固定协议

沿用[AlOH SSW/BH对照](2026-09-21-aloh-ssw-bh-results.md)中SSW的原example零基frame2/3、seed11/29、OMAT-small/omat_pbe、float64、固定胞、25Gaussian、Safe-total500、fmax0.03、bias_fmax0.1与MC150K。只替换旋转策略：分阶段dimer/Ritz换为已有RecoveredRotationSettings（pre_rotmax5、rotmax15、pre_ftol0.2、ftol0.02、Euclidean、max_force_calls40）。不启用完整原版方向控制器或原版MC。

原pre_rotation_hvp=5关闭；rotation_bias=1仅为现有配置校验需要的非工作占位值，恢复旋转分支实际采用自身计算的rotation_weight。没有修改核心源码或公共API。

两策略不是同精度求解器：fd_step=0.001Å时，恢复旋转ftol0.02 eV/Å对应HVP残差2 eV/Å²；原Ritz要求0.02 eV/Å²。两者还分别使用Broyden阶段历史与Krylov子空间，阶段切换不同，不能把结果归因于阈值单因素。完整差异及竞争解释记录在[冻结实验包](../../research/ga_ssw/evidence/aloh-recovered-rotation-20260921/README.md)。

## 实际执行与成本

CPU1434485完成周期Cu32/EMT接口预检：22搜索请求/15实际calculate，另1次独立检查；真实落点、恢复旋转诊断、checkpoint、cell/PBC/元素及数值资格全部通过。GPU1434495完成四臂，耗时7:13。CPU1434687完成零PES账本/结构分析。

新增AlOH搜索24000请求/20690实际calculate，另8次fresh检查，合计24008请求/20698实际calculate。四臂均到请求上限，末次预算截断保留成本；8/8初始/最佳独立数值检查通过。不是整条SSW轨迹正常步数终止，更不等于所有盆地/候选获得物理资格。

|输入/seed|Ritz实际calculate|恢复旋转实际calculate|Ritz最低ΔE/eV|恢复旋转最低ΔE/eV|完整逃逸Ritz/恢复旋转|
|---|---:|---:|---:|---:|---:|
|frame2/11|5602|5157|0.000|约0.000|5/11|
|frame2/29|5585|5174|0.000|0.000|5/10|
|frame3/11|5543|5175|-0.238|-0.238|7/11|
|frame3/29|5546|5184|-0.238|-0.238|6/11|

亚meV差别不作为科学改善。两种方法请求相同但实际calculate不同，因此报告两种成本，不称完全相同实际计算数的比较。没有用minima数量替代结构去重或低能成功率。

|策略|旋转请求|偏置淬火请求|真实淬火含初始化|未分配/中断等|
|---|---:|---:|---:|---:|
|分阶段Ritz|17442|4247|1061|1250|
|恢复旋转|7382|12187|2140|2291|

恢复旋转把更多预算用于偏置爬坡和实际落点，仍未改变这四臂的低能结果。未分配成本不丢弃；阶段请求不能解释为逐阶段实际calculate。

最佳态H最近O距离约0.969–1.031Å；沿用2.3Å邻居标准的Al–O配位4–6（含周期镜像）。筛查未见明显异常，但不证明模型域、化学稳定性、正定Hessian或DFT有效性。只覆盖两个开发初态、每个两seed，不能推导普遍效率或统计成功率。

[分析产物与脚本](../../../vc-qualification-audit/research/ga_ssw/evidence/aloh-rotation-comparison-20260921/)：analysis-final.json、recovered-geometry.json及预执行source-manifest.json。原始失败/截断记录保留，未改变默认策略，未宣称完整LASP复现或稳定发布。
