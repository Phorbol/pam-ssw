# 已批准：约束固定胞 SSW 接入恢复 CBD

## 要解决的问题

用户要求面向分子、slab和bulk，支持ASE FixAtoms/Hookean。已有约束入口正确维护这些约束，但只提供generalized-dimer、Ritz、paper dimer和paper Broyden；当前恢复CBD阶段控制不能用于该入口。这是明确功能差异，不证明现有约束算法错误或恢复CBD一定更好。已完成自由团簇的方向checkpoint和当前等成本对照不覆盖该能力。

## 最小方案与边界

仅在ConstrainedSSWConfig追加可选typed recovered_rotation设置以及显式rotation_exit_policy；默认None/force保持原行为。复用_active_rotation_callback，将活动位移q解释为原子形状向量，经现有ReducedCartesianChart回到完整结构计算，再返回活动梯度。恢复CBD函数和旋转参数直接复用，不复制Broyden实现。

对FixAtoms，x=x_ref+Pq且P固定，梯度Pᵀ∇E、HVP对应PᵀHP；没有新的几何近似。Hookean已进入一致的能量/力surface，CBD使用同一目标。固定基底不投影全局刚体模式，direction_fixed_indices保持仅限制旋转而不新增物理约束。

恢复CBD返回真实收敛标志与stop_reason。默认仍要求收敛；显式force_or_budget只对现有固定胞路径已认可的预算/旋转上限出口放行，保留未收敛标志，数值错误/无效向量不放行。不得把预算出口伪写成converged。PAM自适应Gaussian如不能直接复用恢复CBD的实际anchor/曲率语义，第一版明确拒绝此组合，不做静默参数替代。

不在此步增加nativeMC、pool、完整pair/group方向、VC或RC新算法。shared reduced-loop只增加现有回调结果的显式出口判定；RC等既有调用保持force默认。公共callback无需变成通用插件框架。

现有约束checkpoint已保存config；新增配置需显式兼容旧配置缺字段、恢复设置一致性，并验证无需另一套策略状态。若实际需要改变checkpoint payload结构，应使用兼容旧schema1的新版本，不能默默读旧文件再启用新旋转器。

## 为什么需要讨论

改动虽有界，但触及约束公开配置、共享reduced-loop的停止契约和checkpoint兼容，按本地AGENTS第8节作为接口设计讨论。备选是继续保留现有约束旋转器，暂不补齐；不推荐复制paper_reference整个循环，因为会扩大两条实现之间的差异。

## 实施顺序与验收

1. 先补接口拒绝/兼容和活动坐标恢复CBD的目标测试，保留实际失败。
2. callback适配及显式停止条件；数值有限差分只用于链式法则，不当成搜索收益。
3. 旧默认/旧checkpoint回归、FixAtoms精确保留、Hookean能量力一致、direction-only子空间、预算出口仍标未收敛、新模式分段恢复。
4. 原有example或已冻结slab/吸附初态，使用OMAT-small、原有力阈值和预算，做两种旋转的有界端到端比较，独立复核活动力与约束，并报告完整请求和失败。Free Hookean团簇用第二个几何场景资格检查；不据单例推广默认。

用户已批准该最小方案；现开始实现与验证，保持公开范围不扩展。数学推导基于现有线性活动坐标；ASE的官方dimer代码也区分旋转与外层搜索，但本方案不把ASE实现当成LASP执行等价证明：https://docs.ase-lib.org/_modules/ase/mep/dimer.html 。
