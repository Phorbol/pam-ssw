# S-CR1：最低能连续出发 / 全档案重启配对门控

## 结论

S-CR1 返回 **`DO_NOT_ADMIT_S_CR2`**。固定二通道策略只在 PdO 的最低能下降随
force-evaluation 预算积分上同时胜过现有 UCB-like selector 与经典 Metropolis；C60 和
CuO 均未成立，因此没有达到预注册的“两体系通过”条件。生产默认不变，不执行 seeds
46--47，不在看到结果后调整两条通道的比例，也不准入 Thompson sampling 或其他后验。

本轮与前置 S-CR0 构成一个完整闭环：S-CR0 证明现有 node-level UCB-like 确实会在增长
archive 上变得高度分散；S-CR1 则证明，把状态空间压缩成“最低能连续出发”和“全局均匀
重启”两个更干净的物理动作族，并不会自动带来跨体系收益。**统计对象更合理，不等于
搜索动力学更有效。**

## 预注册等预算结果

`gain AUC` 是在完整 20,000-FE 横轴上对“相对 shared-bootstrap minimum 的当前最佳
能量下降”积分后再除以预算，单位仍为 eV。它同时奖励降得深和降得早，避免用末尾一次
幸运命中替代整个搜索过程。

| 体系 | starter 策略 | 最终下降 | gain AUC | 达到最终最低能时 FE | trials | minima | duplicate | 墙钟 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C60 | uniform archive | 15.427 eV | 10.429 | 18,340 | 41 | 40 | 4.8% | 341 s |
| C60 | archive UCB-like | **30.558 eV** | **28.290** | 18,164 | 52 | 43 | 18.9% | 354 s |
| C60 | Metropolis | 29.191 eV | 26.082 | 18,951 | 52 | 39 | 26.4% | 336 s |
| C60 | paired best/uniform | 27.682 eV | 22.684 | 19,527 | 47 | 29 | 39.6% | 345 s |
| PdO | uniform archive | 4.126 eV | 3.634 | 16,035 | 75 | 62 | 18.4% | 440 s |
| PdO | archive UCB-like | 3.698 eV | 2.926 | 14,861 | 76 | 62 | 19.5% | 442 s |
| PdO | Metropolis | 4.714 eV | 3.755 | 17,580 | 75 | 60 | 21.1% | 441 s |
| PdO | paired best/uniform | **5.057 eV** | **4.043** | 19,379 | 75 | 70 | **7.9%** | 446 s |
| CuO | uniform archive | 3.310 eV | **3.141** | **1,537** | 27 | 28 | 0% | 579 s |
| CuO | archive UCB-like | 3.126 eV | 2.915 | 9,275 | 28 | 29 | 0% | 579 s |
| CuO | Metropolis | 3.332 eV | 3.029 | 10,301 | 28 | 29 | 0% | 577 s |
| CuO | paired best/uniform | **3.613 eV** | 2.907 | 19,306 | 28 | 29 | 0% | 574 s |

paired 相对两个 exploitation comparator 的 gain-AUC 差值为：

| 体系 | paired - UCB-like | paired - Metropolis | 是否通过体系门控 |
|---|---:|---:|---|
| C60 | -5.606 eV | -3.398 eV | 否 |
| PdO | +1.117 eV | +0.288 eV | 是 |
| CuO | -0.008 eV | -0.122 eV | 否 |

## 物理语义

### C60：反复从当前最低能盆地出发会强化漏斗俘获

C60 paired 的 duplicate rate 达 39.6%，archive 只有 29 个 minima；同预算下 UCB-like
得到 43 个 minima，且最低能更低 2.876 eV。固定一半动作从当前最低能结构继续出发，并
没有形成稳定的“向下接力”，反而反复返回已经访问的盆地。这里的瓶颈不是 archive
支持过宽，而是给定低能 starter 后，当前方向与 Gaussian-bias propagation 不能稳定找到
新的下降出口。Metropolis 保留少量低能 continuation，但不会机械地把一半预算锁在唯一
最低点，因而也优于 paired。

### PdO：连续下降和远程重启在这一条轨迹上互补

PdO 是唯一正例。paired 同时取得最高 gain AUC、最深末态、最多 minima 和最低 duplicate
rate。这里从低能表面重排继续走仍能产生新盆地，而均匀重启又避免完全困在单一局域重排
序列；两种通道确实表现出预期的互补物理图像。

但这个最低点直到 19,379 FE 才出现。正收益来自整条曲线而非仅末态，却仍只有一个 seed，
不能把 PdO 个例提升为通用 selector，更不能据此拟合体系特化的 continuation 概率。

### CuO：starter 不是本轮主要限制

四种策略都几乎每个 action 产生一个新 minimum，duplicate rate 全为 0，最终 archive
也都是 28--29 个。说明在这个 20k-FE 窗口内，改变 starter 分布几乎没有改变“能否离开
已有 basin”；昂贵的 biased proposal relaxation 和每个 action 的实际可达性才支配成本。

uniform 在第 2 个 action、1,537 FE 就达到 -201.990 eV，随后 18,463 FE 只增加覆盖而未
改善最低能。paired 最终找到全场最低的 -202.292 eV，却发生在 19,306 FE，因此其 gain
AUC 反而略低于 UCB-like、低于 Metropolis 和 uniform。这是为什么选择器不能只按末态
排名：晚期偶然命中不能证明此前预算分配更好。

## 与 growing-arm / Bayesian 假设的关系

S-CR0 的九条 seed-42 轨迹显示，UCB-like 在只有 19--74 个 minima 时，Shannon effective
starter support 已占最终 archive 的 62.2%--85.4%；三体系平均 72.4%。新节点以零 trials
进入 archive，UCB exploration bonus 会持续把物理 action 分给一次性新 arm。这个
growing-arm 问题是真实存在的。

S-CR1 同时给出了更重要的否定边界：当前数据没有一个跨体系稳定的 starter family 排序。
若现在对两个 family 做 TS，posterior 主要学习的会是体系、漏斗和 action kernel 条件差异，
而不是一个可转移的 stationary success probability；若对每个 node 做 TS，数据稀释只会
更严重。因此：

1. node-level UCB-like 不应再增加权重、表征或每节点 posterior；
2. `paired_best_uniform` 保留为 opt-in 实验对照，不作为生产默认；
3. 不删除 UCB-like：它在 C60 本轮是明确最佳，扩散性不是等价于低效率；
4. 不执行 S-CR2 和 family-posterior admission，因为前置因果门控已经失败；
5. starter selector 暂时不再作为主线瓶颈。

## 成本闭合

12 个正式任务共执行 240,000 次 force evaluation，GPU 墙钟 5,436.5 s（约 1.51 h）：

- 162,392 次 biased proposal relaxation（67.7%）；
- 45,183 次 landing true quench（18.8%）；
- 28,224 次 direction oracle（11.8%）；
- 2,805 次真实 PES escape 检查（1.2%）；
- 780 次 shared bootstrap quench 与 616 次 post-relax validation；
- `unattributed = 0`。

每个 arm 都严格计入 20,000 FE；同一体系的四个 arm 复用完全相同的 bootstrap 坐标、
能量和成本。C60 bootstrap 为 49 FE，PdO 为 38 FE，CuO 为 111 FE。方向、局部软化、
Gaussian bias、proposal optimizer 和 true quench 全部冻结，唯一实验变量是 starter mode。
低预算 smoke 和最初错误的 partial-cohort 决策均未进入科学证据。

## 下一主线

这轮关闭的不是“所有 starter 选择都无效”，而是“当前应优先复杂化 selector/posterior”
这条路线。跨体系共同证据仍指向更底层的 action 可达性：

- C60 的最低点连续出发产生高重复，说明需要改善从给定 basin 找到新出口的能力；
- CuO 的所有 selector 都能产生新 minima，但绝大部分预算仍花在 proposal relaxation，且
  最低能命中时刻差异巨大；
- PdO 的 selector 正收益不能跨体系复现。

因此下一轮回到冻结 starter 的 action 机制门控：优先检验给定方向后的 Gaussian-bias
传播是否把结构送到“可淬火且通向新盆地”的区域，而不是继续调整 starter 权重。只有方向
和 uphill propagation 形成可重复、低维且有条件可预测的 action family 后，才重新开放
action-conditioned Bayesian/posterior allocation。批量并行仍可用于同时执行具有全支持的
独立 action，但不需要先引入后验。

## Claim ceiling

结论来自 fresh seed 45、C60/PdO/CuO、12 条 shared-bootstrap、每条 20,000-FE 的门控。
它因果否决固定 1:1 lowest-energy continuation / uniform restart 进入重复阶段，并说明
growing-arm 扩散不是当前跨体系首要瓶颈；它不证明任何 selector 的生产最优性、长任务
渐近行为、统计显著性、canonical sampling 无偏性，或 Bayesian 方法永久无效。
