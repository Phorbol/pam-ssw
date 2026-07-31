# G-E0：真实 PES 首次下降提前停止门控

## 结论

G-E0 返回 **`ADMIT_FRESH_ONLINE_G_E1`**，但不修改生产默认值。

冻结的 24 条 C60/PdO D0/K4 路径共有 82 个已接受外层 micro-step
endpoint。规则

\[
E_{\mathrm{true}}(x_k)<E_{\mathrm{true}}(x_0)-10^{-3}\ \mathrm{eV}
\]

在 4 条路径上首次触发；这 4 个 crossing 经真实 PES 淬火后全部进入能量低于
starter、几何有效且有力收敛证书的 basin。因此，在这个冻结语料上它是 4/4 精确的
**充分低能盆地证书**。它不是必要条件：已有 first-passage 数据的 26 个认证 escape
horizon 中仅 6 个满足该能量下降条件，覆盖率只有 23.1%。

## 四个触发路径

| 体系与动作 | crossing / terminal | crossing landing ΔE | terminal landing ΔE | 省外层步 | 结果 |
|---|---:|---:|---:|---:|---|
| C60 plateau seed 42 K4 | 5 / 7 | -2.465 eV | -2.405 eV | 2 | 避免轻微走过头；matcher/descriptor 对盆地关系有歧义 |
| C60 plateau seed 43 D0 | 1 / 8 | -7.079 eV | +0.176 eV | 7 | 避免继续加 bias 后丢失低能盆地 |
| C60 plateau seed 44 K4 | 2 / 8 | -7.013 eV | -9.030 eV | 6 | 会错失更深的后续盆地 |
| PdO plateau seed 43 D0 | 1 / 1 | -0.840 eV | -0.840 eV | 0 | 原路径已自然终止，完全等价 |

前三条 C60 路径合计可少走 15 个外层 micro steps。这个数不是 force-evaluation
节省量：提前 true quench 的收敛成本可能不同，且少走后续 bias 也会改变以后进入的
basin。特别是 seed-44 K4 清楚证明“第一次得到更低 basin”不等于“当前动作内的最优
停止点”。

## 物理语义

每个外层 micro step 都在累计 Gaussian-biased PES 上先显式位移、再局部松弛，随后
现有 walker 已经在真实 PES 上计算 endpoint 能量。若 endpoint 的真实能量已经低于
macro starter，那么它不可能仍属于一个严格更高能的稳定 minimum；对它做去 bias
淬火，能量还应继续不增。这就是该规则 4/4 精确的底层原因。

但 SSW 的主要作用是穿越能垒。一个已经跨过盆地边界的结构完全可能仍位于高于
starter 的真实能量处，随后淬火到另一个 basin。因此该规则天然低覆盖，不能替代
几何/盆地 escape 判断，也不应被包装成新的自适应 trust controller。它只回答一个
窄问题：**既然一个更低盆地已经有充分证据，是否值得立即兑现，而不继续在累计 bias
下传播？**

## 核算

离线补证共新增 117 次 force evaluation、GPU kernel wall time 4.79 s：

- 31 个原语料缺失 endpoint 的真实能量，加上 2 个新 quench 自带的 checkpoint
  能量检查，共 33 次 `escape_true_pes_check`；
- 两个未被原语料覆盖的 crossing/terminal 共 82 次 `landing_true_quench`；
- 2 次 `post_relax_validation`；
- 方向 HVP、biased proposal relaxation 和 unattributed 均为 0。

其余 crossing/terminal 直接复用已哈希的 first-passage quench。所有 24 case、82 个
accepted endpoint、93 个 attempted endpoint 以及 purpose ledger 均机械闭合。

## 下一步边界

G-E1 只允许做一项新实验：对新的、成对的完整 action，在同一 starter、方向、Gaussian
参数和随机流下比较：

1. reference：保持现有传播与自然终止；
2. first-descent：首次满足上述真实能量下降条件时立即结束外层传播并 true quench。

比较必须使用相同总 FE 预算，并分别报告方向、biased relaxation、true-energy check、
true quench、最低能量、landing-basin support 和后续 macro action 收益。由于在线 walker
已经计算每一步 `true_energy_after`，停止判断本身不需要新增 force evaluation。

G-E1 的晋级条件不是“每次都省步”，而是单位总 FE 的低能 basin 发现效率提高，同时
没有不可接受地压缩更深 landing 的支持。G-E0 不支持把规则直接加入生产默认，也不支持
引入更多阈值、耐心窗口或概率模型。

## Claim ceiling

本结论仅适用于冻结的 C60/PdO、D0/K4、24 路径离线反事实语料。它证明首次真实能量
下降在 4 个触发样本上是精确但低覆盖的低能盆地证书，并揭示了“避免走过头”与“错失
更深终点”的真实权衡；不证明在线 FE 节省、长程搜索收益、CuO 泛化或生产最优性。
