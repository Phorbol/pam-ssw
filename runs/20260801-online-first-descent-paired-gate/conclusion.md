# G-E1：在线真实 PES 首次下降配对门控

## 结论

G-E1 返回 **`DO_NOT_ADMIT_G_E2`**，生产默认不变，也不再沿这个停止规则增加耐心窗口、
趋势拟合或概率模型。

24 条 fresh C60/PdO D0/K4 action 的 exact shadow prefix 和 purpose ledger 全部闭合，
`unattributed = 0`。规则仅在 2 条路径触发，两个提前构型经真实 PES 淬火后都进入低于
starter 的认证 basin，因此 G-E0 的窄物理结论仍成立：首次真实能量下降是一个精确但
稀疏的低能盆地证书。

但两个触发 action 的完整成本合计不是节省，而是**多 58 次 force evaluation**。它没有
通过预注册的成本门控，不能进入等总预算长搜索 G-E2。

## 触发路径与完整 action 成本

| 路径 | crossing / terminal | early landing ΔE | reference landing ΔE | early / reference FE | 净节省 | 物理结果 |
|---|---:|---:|---:|---:|---:|---|
| C60 plateau seed 45 D0 | 2 / 4 | -3.959 eV | -3.650 eV | 633 / 390 | -243 | 避免走过头，但提前构型很难淬火 |
| C60 plateau seed 47 D0 | 1 / 4 | -0.870 eV | -5.076 eV | 121 / 306 | +185 | 节省成本，但错失深 4.206 eV 的终点 |

seed 45 的传播前缀确实少用了 4 次 direction、165 次 biased relaxation 和 2 次真实
能量检查；然而 crossing landing 的 true quench 花了 485 FE，而自然终点只需 71 FE。
少走传播节省的 171 FE 被额外 414 FE 的淬火成本完全反转，完整 action 最终多花
243 FE。

seed 47 的提前构型没有这个病态淬火问题：它少用了 6 次 direction、175 次 biased
relaxation 和 3 次真实能量检查，true quench 只比 reference 多 1 FE，故净省 185 FE。
代价是自然传播在第 4 步进入了更深 4.206 eV 的另一盆地。

所有 12 条 PdO 路径均未触发；C60 的 6 条 intermediate 路径也均未触发。两个触发都
来自 C60 plateau 的 D0 exact-anchor action，说明现有证据既不广泛，也不能支持跨体系
停止策略。

## 底层物理解释

`E_true(x_k) < E_true(x_0)` 只说明当前累计 bias 已经把结构送到一个能够向更低 basin
下降的位置。它没有说明这个未充分弛豫的 crossing state 靠近该 basin 的 harmonic
吸引域，也没有说明后续累计 Gaussian 不会打开更深的下降通道。

所以“少走几个 uphill micro steps”不等于“少做 force evaluation”：停止点离局部极小值
越远、正交方向残余力越大，去 bias 后的 L-BFGS 淬火就可能越昂贵。反过来，继续传播虽
增加 biased-relax 成本，却可能把结构送入更容易淬火或更深的吸引域。G-E1 的两条触发
恰好分别观察到了这两个机制。

这也解释了为什么不应再给该规则增加启发式补丁。要预测 crossing 的后续淬火条件数和
继续传播的潜在深度，需要解决的已经不是一个无参数停止证书，而是新的动作价值预测问题；
在两个触发样本上引入窗口、阈值或 Bayesian stopping 只会制造不可消融的局部复杂性。

## 实验核算

本轮新执行 7,157 次 force evaluation，GPU kernel wall time 156.02 s：

- 4,387 次 `biased_proposal_relax`；
- 400 次 `direction_oracle`；
- 143 次 `escape_true_pes_check`；
- 2,201 次 `landing_true_quench`；
- 26 次 `post_relax_validation`；
- 其余 purpose，包括 `unattributed`，均为 0。

初始双执行 smoke 没有进入科学证据：两次名义相同的 float32 GPU MACE action 在停止
决策发生前就已有约 3.05e-5 eV 的 starter 漂移和约 2.14e-4 eV 的首个 endpoint 漂移。
正式实验因此只执行一条 reference 路径，observer 记录同一条路径的精确 crossing
checkpoint 和累计核算，再分别淬火 crossing 与自然终点。没有用放宽数值容差掩盖 GPU
非确定性。

## 后续边界

这条支线到此关闭：

1. 保留 inert 的内部 post-step hook 和 research observer，便于重放机制证据；
2. 不暴露用户配置，不改变 walker 默认行为；
3. 不执行 G-E2，不继续优化 first-descent edge case；
4. 主线回到能改变 basin 可达性的核心问题：方向质量、给定方向后的 uphill propagation，
   以及只在这两者稳定后再研究 action-conditioned posterior selector。

## Claim ceiling

结论仅适用于 fresh seeds 45--47 的 24 条 C60/PdO、D0/K4 exact-shadow-prefix action。
它证明 first descent 在两次触发中仍是认证低能盆地的充分证书，但没有净 FE 收益，并
暴露了 true-quench conditioning 与继续传播深度的权衡；它不证明生产级长搜索无效、
CuO 泛化、统计显著性或 canonical sampling 性质。
