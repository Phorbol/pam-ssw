# 固定 Eckart 截面用于孤立团簇 SSW：数学与源码审查

2026-09-10；本agent只读设计与随后实现审查，未修改算法代码、未运行新搜索。

## 判断与适用问题

对于能量严格不依赖整体平移和转动、无外场、无约束、非周期的非线性孤立团簇，
每次 outer escape 固定一个参考构型和线性截面，并把 **整个修改后目标函数** 拉回此截面，
是一个有明确链式法则、值得作最小实验的修复。它比只投影初始方向更完整。
它在正则截面内消除自由的刚体轨道，不保证所有位移在当前构型的瞬时刚体空间上投影为零，
也不是全局无奇异的结构坐标。必须作为独立的 `cluster` 几何选项；不是 LASP 等价复现，
不是 RC-SSW，也不能从此推出搜索效率或正确热力学采样。

具体失效依据见 `2026-09-10-cu13-escape-diagnosis.md`：未约束的 Cartesian Gaussian
可通过团簇整体运动降低偏置，Ritz 和 dimer 两种求解器都出现此问题。修复应作用于目标函数
的定义域，而不是再调一个方向打分或增大 Gaussian。

## 文献核对和证据边界

- Carl Eckart, *Some Studies Concerning Rotating Axes and Polyatomic Molecules*,
  Physical Review **47**, 552 (1935), DOI [10.1103/PhysRev.47.552](https://journals.aps.org/pr/abstract/10.1103/PhysRev.47.552)。
  本轮查到原始期刊摘要，明确假设 rotation-displacement invariance，并提醒大振幅下的限制；未取得其全文。
- Viktor Szalay, *Eckart ro-vibrational Hamiltonians via the gateway Hamilton operator: theory and practice*,
  JCP **146**, 124107 (2017), DOI [10.1063/1.4978686](https://doi.org/10.1063/1.4978686)，
  [作者预印本全文](https://arxiv.org/html/1701.01823v1)。本轮实际阅读 II.1–II.2：
  式(5)、(7)给质量加权平移和旋转条件；式(11)把它写成齐次线性方程，式(13)–(19)
  给固定参考的零空间坐标。式(8)仍有转振耦合，不能说在任意形变下完全分离。
- Flemming Jørgensen, *Orientation of the Eckart frame in a polyatomic molecule by symmetric orthonormalization*,
  IJQC (1978), DOI [10.1002/qua.560140106](https://onlinelibrary.wiley.com/doi/10.1002/qua.560140106)。
  本轮仅摘要：Eckart 条件与最小二乘定向相关，并有多个旋转解；不依摘要推断具体求解公式。
- R. G. Littlejohn, K. A. Mitchell, V. Aquilanti, S. Cavalli,
  *Body frames and frame singularities for three-atom systems*, PRA **58**, 3705 (1998),
  DOI [10.1103/PhysRevA.58.3705](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.58.3705)。
  本轮只读原期刊摘要，其论域为三原子 frame singularities；不把它直接当成任意 N 的定理证明。

以下 SSW 拉回、正则性矩阵及验证设计是针对本项目的推导，不是文献已经实现的 SSW 算法。

## 固定欧氏截面及精确目标函数

把中心化参考原子坐标记为 x0_i，组合向量为 x0∈R^(3N)。取 B0 的列为三个平移方向
与三个转动方向 e_alpha×x0_i 的独立正交基。Q 为其正交补：

    B0^T Q = 0,  Q^T Q = I,  x(q) = x0 + Q q.

它精确满足

    Σ_i x_i = 0,     Σ_i x0_i × (x_i - x0_i) = 0.

这里用算术中心和普通 Cartesian 内积，应称 **等权 Eckart 型截面**。对于同元素 Cu13
与质量加权 Eckart 只差常数；异元素时它不是动力学意义的质量加权 Eckart，但仍可作为
静态势能搜索的坐标规范。不要悄悄把质量权重同时引入 Gaussian 宽度、方向归一化和 fmax。
若以后选择质量矩阵 M，须一致用 x=x0+M^(-1/2)Qq，g_q=Q^T M^(-1/2)g_x，并重新说明单位。

一次 escape 内令

    U(x) = E(x) + P_LS(x) + Σ_j B_j(x),
    u(q) = U(x0 + Qq).

因为 x(q) 是仿射映射，链式法则严格给出

    g_q = Q^T g_x,       F_q = Q^T F_x,
    H_q = Q^T H_x Q.

不需额外的坐标二阶导数。这是受限问题的真实梯度，不是“报告原能量但随意改力”。
要求Gaussian存在期间的所有试探点、line search、方向有限差分和Gaussian累积优化通过同一个映射。
若优化器仍可沿完整 Cartesian 自由度移动，仅投影返回力不能证明执行的是同一受限问题。

方向需先在 q 空间产生/投影并归一化；若原随机锚点投影后为零，应显式失败或重新抽样并记录，
不能回退到未经投影的刚体向量。方向偏置在 q 中仍可写为 -a*n0_q*n0_q^T。

若 Gaussian 的中心 x_j=x0+Qq_j，单位方向 n_j=Qv_j，则

    (x-x_j)·n_j = (q-q_j)·v_j.

因此 Gaussian 能量、解析力及高度控制可以在 q 中一致定义，sigma仍为欧氏长度 Å。
背景力的前向投影也相同：F_x·n_j = F_q·v_j。87°若以后启用，必须用可运动空间的合力角度；
否则约束反力会污染角度控制。该改动属于新几何版本，不应声称原版高度策略逐步等价。

Q必须在加入第一个Gaussian之前建立并冻结。可以在LS预淬火之后建立：pair-distance LS势
自身旋转平移不变，没有Gaussian造成的刚体偏置逃逸。最后撤去所有Gaussian后也可以解除
frame、在真实PES上自由淬火，并按完整真力认证；这种分阶段定义数学上成立，而且省去
最终淬火的chart有效域限制。早先建议把LS预淬火和最终真淬火都放入同一frame，是一种
可选一致实现，不是必要条件。必要条件是不能在仍有Gaussian时更新Q却沿用旧中心/方向
历史；这会改变定义域和势函数。参考原子索引不得在当步重排。

## 为什么能阻止自由刚体逃逸；为什么不是全局保证

截面条件记为 c(x)=Σ_i x0_i×x_i=0。让当前构型作无穷小整体旋转
δx_i=ω×x_i，则

    δc = A(x) ω,
    A(x) = Σ_i [(x0_i·x_i) I - x_i x0_i^T].

这个公式由向量三重积直接得到。满足截面条件时，Σ_i x_i x0_i^T对称，因此A对称。
在参考点 A(x0)=Σ_i[|x0_i|² I-x0_i x0_i^T]，是等权惯性矩阵。
非共线参考的A正定，所以截面与整体旋转轨道局部横截，附近不存在留在截面内的连续自由
刚体旋转。平移已由中心条件完全消除。这是局部坐标规范成立的依据。

在大变形下，A可能失秩：某个当前整体旋转切向此时属于Q空间，刚体泄漏可能重新出现。
参考的rank=6并不能保证当前A可逆。还可能存在满足相同线性条件的远处有限旋转副本，
例如沿参考主轴转180°；仅检查B0^T(x-x0)=0不能排除它们。

建议最小实现记录A的三个特征值和相对于A(x0)的无量纲版本

    A0^(-1/2) A(x) A0^(-1/2).

保持与参考相连的正定分支给局部规范意义。A在截面内对x为线性，正定锥为凸集，故两个
正定端点之间的直线试探不会穿过A奇异面。数值近奇异界应来自机器/几何误差与力精度，
不另发明用来优化成功率的物理阈值；跨越或无法分辨正则性时报告失败，不悄悄加刚体惩罚。
这只是坐标有效域检查，不是完整保证所有远距离结构可在单一光滑chart中搜索。

固定Q的速度Q*qdot并不处处垂直于当前刚体基B(x)。因此会有与变形耦合的瞬时转动分量；
不能要求后验按当前B(x)投影的刚体占比严格零。要检验的是原先几乎不改变内部几何的
低成本逃逸是否消失，而不是把所有非零瞬时转动都称为bug。

线性参考只有5个独立刚体自由度，单原子/重合构型等更退化；接近共线时也会病态。
最小试验可明确限定非共线非线性cluster，不为所有rank情形加入临时补丁。
平面但非共线结构的三维转动惯性矩阵仍可正定，不应机械地把“平面”当作退化。

## 淬火证书与动力学边界

在真实旋转平移不变PES上，B(x)^T g_x=0。若q淬火达到g_q=0，则g_x属于span(B0)。
当B(x)^T B0可逆（等价于该截面的横截性）时，可推出g_x=0。
因此正则截面内的精确驻点不是人为产生的受限假驻点。

但有限容差下，病态的B(x)^T B0会放大误差；calculator也可能有小的非零总力/总力矩。
所以最终必须用独立calculator检查 **完整Cartesian真力** max_i|F_i|，不能仅凭max|F_q|
或受限优化器converged标签发证书。修改势面的受限驻点可以有约束反力，这是正常的，
不能强制它也满足完整修改后Cartesian力为零。

本方案没有包含约化动力学的非平凡metric或统计Jacobian；不能据普通Metropolis就宣布
得到了正确平衡采样。结构身份仍需旋转、平移和适用的同元素置换比较，chart坐标不是全局descriptor。

## PAM当前源码与本方案差异

本工作树 `pamssw/rigid.py` 用当前位置的平移/旋转列构造QR基，再对方向投影。
`walker.py:_candidate` 中若投影后norm极小会回退原向量；严格Krylov锚点路径则拒绝零投影。
`_choose_block_krylov_direction` 对HVP的total/true分量再次投影。
这些有助于初始方向不被零模支配，但没有建立当步固定的整个目标函数截面。

`walker.py:_relax_proposal_task` 把完整 `ProposalPotential.evaluate` 交给 `Relaxer`；
`relax.py` 操作可动原子的Cartesian坐标，存在trust bounds但没有固定Q。
`bias.py` 对非周期结构仍使用Cartesian差 `(x-center)·direction`。
因此当前PAM也不能仅因方向投影就宣称彻底消除了偏置淬火的有限刚体逃逸。

额外源码边界：`rigid.py` 对有cell或PBC的state走periodic translation分支；对N<3
不移除任何刚体模；对部分fixed原子只用movable子集构基。受约束结构这些方向未必是
真实对称性，因此不应复制成新独立cluster实现的通用规则。

## 最小实现约束与可证伪实验

1. 显式选择isolated-cluster/Eckart几何，默认不自动推测任意ASE calculator具有此对称性。
   先以旋转/平移重评E/F检查后端数值精度，不增加位置势或额外惩罚参数。
2. 一个固定Q覆盖本步Gaussian存在的全部阶段；LS预淬火可以在建frame前，最终撤偏置true quench可以解除frame；保留原Cartesian对照。
   dense Q便于小团簇验证，但存储O(N²)、映射O(N²)；以后可用thin B的O(N)投影实现
   同一固定子空间，不能因此改成位置依赖P(x)。
3. 数值检查只验证exact pullback、Q正交性、dimer HVP、Gaussian解析梯度、A正则性；
   它们不证明搜索有效。普通Cartesian每原子独立裁剪step可能离开Q子空间，需检查实际trial映射。
4. 重用原失败Cu13初态、同seed/势面/预算及独立真力证书，比较Cartesian与固定截面；
   同时保留Ritz/dimer以分离几何与方向求解器贡献，报告所有失败与总E/F调用。
5. 主要检验：Kabsch后的内部位移、真淬火后的有效不同basin，而非原始Cartesian距离或acceptance。
   若仍回原basin，即使刚体泄漏被抑制也不能说修复了全局搜索，只能排除一个失效机制。
6. 若A频繁病态、q收敛而真力不合格，或相比对照只增加成本且没有内部探索，应暂停推广，
   重新考虑大变形坐标方案；不要靠叠加奖励、阈值或高度策略挽救这一设计。

当前决定：支持一个范围明确的独立固定截面实验；保留原Cartesian实现作对照，
不修改稳定PAM、不宣称复现LASP的新证据、不扩展到PBC/外场/RC，等待真实体系结果。


## 随后实际实现的只读复核

同日主agent新增 `cluster_frame.py`、`quench(frame=...)` 和 opt-in `cluster_frame='eckart'`。
本段是源码审查，不替代主agent实测。

- `ClusterFrame` 用薄SVD旋转基和归一化平移基，P=I-BB^T，无dense Q；
  `positions`投影完整trial，`ClusterFrameFilter.get_forces`投影完整calculator的合力。
  对Filter支持的Cartesian optimizer坐标而言是同一仿射目标的精确pullback。
- 实际reference保留原几何中心，不先搬到原点；相对坐标用于旋转基，因此固定的是初始中心，
  与中心化推导等价。实际A先计算x0^T x再对称化，截面上与上面A的转置等价。
- LS预淬火后建frame、最终true quench解除frame：认可此边界，原因见上文。最后证书是完整真力，
  `modified_cluster_section`明确标记受限修改势面证书，不能误读成完整Cartesian偏置力收敛。
- 需补边界：投影anchor后检查norm，不要直接除零；chart失效需要明确step记录而不是让整个run
  未记录异常退出；rotation HVP trial和height probe也应进入同一frame映射/正定分支检查。
  当前rotation_surface只投影force，理论上初始方向和HVP闭合于Q，但数值泄漏和有限探针越过
  chart边界仍需显式处理，不能只等后续quench setter才发现。
- 建议增加A正定/奇异/180度分支的测试与运行诊断，及FD能量-力一致性跨所有probe的证据。
  这不新增物理搜索机制，只把所宣称的坐标定义域贯彻到全部实际调用。

### 边界修复后的最终复核

同日再次只读核查 `cluster_frame.py` 和 `paper_reference.py`：

- anchor 投影后现在检查有限性和可分辨norm，不再直接除零。
- rotation callback 先把FD trial映射到frame并检查正定chart，再评估E/F和投影力。
- height probe先映射/检查chart，之后才请求calculator；偏置quench仍走相同Filter。
- 新的 `ClusterFrameDomainError` 在Gaussian循环内变成 `cluster_frame_failed`，保留
  本步请求计数与错误，不生成landing。测试新增180°旋转副本拒绝及注入chart异常记录。

没有发现阻断本轮受限Cu13实验解释的目标/力/坐标不一致。审查没有代替独立真力、
结构身份及Hessian验证，也没有重跑主agent报告的141项测试。

两个非阻断记录边界：零anchor/退化参考目前仍是输入级ValueError并终止run，而非step失败；
chart异常的记录几何保留最后完成的work，不是被拒绝的trial。应按实际含义报告，后续可统一
状态和失败几何记录；不能声称已经完整序列化每个被拒绝试探点。
