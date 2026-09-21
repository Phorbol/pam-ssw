# 恢复方向机制：更便宜的单次逃逸，不等于更有效的搜索

目标是吸收有用机制，不要求LASP数值轨迹一致。本次两臂共用恢复CBD、Gaussian
与Safe-total500，仅比较global anchor和既有Q-off方向生成/更新整组机制；不是
单独位移记忆消融。配置在[协议包](../../../ga-ls-composition/research/ga_ssw/evidence/direction-utility-20260921/plan.json)。

GPU1433114完成5分21秒；CPU1433142完成成本/结构读出，CPU1433189完成对齐分析。
五起点×两臂全部完成一个完整外步，6711搜索请求+20独立复核=6731；
实际Calculator.calculate为6305搜索+20复核=6325。20/20力/能量/组分/cell资格通过，
最大力.0297973 eV/Å；均连通。额外接口检查4次EMT单点单列，不是科学样本。

| 输入 | 完整方向实际calculate | global+CBD实际calculate | 完整方向落点ΔE/eV | global+CBD落点ΔE/eV |
|---|---:|---:|---:|---:|
| C60/17093-first |340|466|−2.1673|−3.7633|
| C60/17093-best |399|671|+4.3455|+22.6868|
| C60/17094-first |790|1512|−.00116|−.99260|
| C60/17094-best |840|852|−.00013|+4.0265|
| Cu55/OMAT |193|242|+.00020|+1.0223|

ΔE相对各臂已淬火起点；高能落点保留，不把被MC拒绝等同数值失败。
完整方向在5/5个输入省实际计算，合计2562对3743；但不能把31.6%局部节省
写成全局效率提升。八个C60落点均未满足既定目标笼拓扑；不同开发起点不是独立成功率样本。

竞争解释的后续检查使用初末同标签刚体对齐和三个既有距离图阈值。
完整方向的两个17094输入键图均不变，但RMSD为1.7001/.2759 Å，不能简单断言
两者都回到完全相同结构或盆地：存在柔性构型运动的解释。Cu55 RMSD仅.00811 Å，
三个距离图均不变，当前没有新盆地证据。其余两个C60完整方向发生键图变化。
global+CBD五例均有距离图变化，但这也不保证更有用或物理上不同的盆地。
Kabsch初稿行/列约定错误在执行前由root修正，并用已知旋转和平移检查通过；
未产生旧公式的科学结果。几何partial文件保留，正式分析另存final文件。

**决定：** 保留完整方向可选，不提升默认、不沿这批输入调参数、不立即扩大该对照。
已有证据支持局部成本下降，不支持更好的有效低能探索。下一项针对实际native
对照开启的Q-mode做有界证据核查：确认覆盖和数学机制后再决定是否值得实现。
这不是将Q列为性能差距的已证实原因，也不重做已经闭合的c1/c4/c6/c9数值细节。

读出入口：
[成本与数值资格](../../../vc-qualification-audit/research/ga_ssw/evidence/direction-utility-readout-20260921/analysis-final.json)、
[终态图筛查](../../../ls-constraint-qualification/research/ga_ssw/evidence/direction-utility-geometry-20260921/geometry-final.json)、
[初末对齐与图变化](../../../ls-constraint-qualification/research/ga_ssw/evidence/direction-utility-geometry-20260921/geometry-initial-landing-final.json)。
原账本、冻结源码、参数及失败位置均在协议包；本轮无核心算法/API/默认值修改。
