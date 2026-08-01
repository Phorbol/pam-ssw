# Archive-scaled versus fixed macro uphill target: three-seed closure

## 问题与物理假设

当前 walker 并不直接使用配置中的 `target_uphill_energy=0.8 eV`。每次 macro action
开始前，`StepTargetController` 会用已发现 archive 的能量跨度构造新的 target，并在
`0.04--4.0 eV` 内截断。该值随后共同影响方向评分、初始位移、Gaussian width 和
micro-step trust。

这个 archive range 不是当前 starter 附近的局域势垒，也不是该方向上的真实 PES
能量上升。因此本门控检验一个最小假设：取消这项全历史缩放、始终使用已经存在的
`0.8 eV` reference，是否能在不加入新参数的情况下提高搜索收益。

两臂只在 macro target 来源上不同。shared bootstrap、Metropolis starter、方向、局域
软化、累计 Gaussian、micro-step trust、Safe-LBFGS proposal relaxation、true-PES
quench、随机流和每臂 20,000 force-evaluation 总预算全部冻结。

## 零 FE 审计 U-T0

对既有 12 条 S-CR1 轨迹精确重建 target 历史后，archive scaling 在 C60、PdO、CuO
分别有 97.9%、98.7%、96.4% 的 action 偏离 0.8 eV。它不是 inactive bookkeeping，
而是一个强作用、此前未单独归因的上坡传播机制。

## 三体系、三种子结果

主判据是相对 shared bootstrap 的 best-energy gain 对 force-evaluation budget 的积分
（gain AUC）。表中 `ΔAUC = fixed - scaled`，正值表示 fixed 更好；`fixed endpoint
advantage` 为 scaled 最低能量减 fixed 最低能量，正值表示 fixed 终点更深。

| system | seed | ΔAUC / eV | AUC winner | fixed endpoint advantage / eV |
|---|---:|---:|---|---:|
| C60 | 46 | -3.555 | scaled | -3.673 |
| C60 | 47 | -10.859 | scaled | -11.653 |
| C60 | 48 | +9.256 | fixed | +10.880 |
| PdO | 46 | +0.236 | fixed | -0.747 |
| PdO | 47 | -1.733 | scaled | -3.512 |
| PdO | 48 | +0.769 | fixed | +1.034 |
| CuO | 46 | +0.0157 | fixed | +0.159 |
| CuO | 47 | +0.262 | fixed | +0.269 |
| CuO | 48 | -0.0740 | scaled | +0.0225 |

预注册晋级条件是 fixed 至少赢 6/9 个 paired block，且九个 ΔAUC 的中位数为正。
实际为 5/9；中位数虽为 `+0.0157 eV`，仍返回：

> `RETAIN_ARCHIVE_SCALED_DEFAULT`

这表示 fixed 0.8 eV **没有通过替换门槛**，不是 archive-range scaling 已被证明具有
正确的势垒物理意义。生产默认未改变，也不继续调固定 target 或 archive scale 系数。

## 跨体系物理图像

- C60 的 scaled target 在 seeds 46/47 的均值为 1.87/2.10 eV，明显高于 0.8 eV，
  scaled 两次获胜；seed 48 的 scaled target 均值只有 0.705 eV，fixed 获胜。对该团簇，
  持续足够强的上坡是重要信号，但 archive range 仍有强 seed dependence。
- PdO 的 scaled target 均值仅 0.36--0.51 eV，fixed 赢 2/3；CuO 更低，仅
  0.18--0.24 eV，fixed 也赢 2/3，但 CuO 的差值很小且 seed 48 反号。这支持用户提出的
  “某些 slab 上 adaptive uphill 可能不够猛”，但不支持统一固定 0.8 eV。
- 同一体系内仍可反号，说明 target 改变的不只是单步高度，而是通过 basin-level 分叉
  改变整条后续路径。它应作为随机搜索策略比较，不能作逐帧 deterministic 配对解释。
- PdO seed 46 的 fixed AUC 更好、最终最低能量反而更差；CuO seed 48 的 fixed 终点略深、
  AUC 反而更差。终点、早期预算收益和 continuation value 不是同一个量。

## 成本闭合

18 个 case 共使用 359,998 次 force evaluation，只有 CuO seed-46 scaled 因剩余 2 次
预算不足以提交不可拆分的批量 HVP 而未精确达到 20,000；该尾差小于一次 16-geometry
方向批次，`budget_exhausted=true`，用途账本完全闭合。`unattributed=0`。

| mode | FE | wall time | proposal relax | true quench | direction oracle |
|---|---:|---:|---:|---:|---:|
| archive-scaled | 179,998 | 85.2 min | 68.0% | 18.1% | 12.0% |
| fixed 0.8 eV | 180,000 | 77.4 min | 68.2% | 18.3% | 11.6% |

两臂完成的宏步（447/453）和 archive entries（387/383）接近；target 机制没有通过简单
增加 action 数或 archive 数取胜。wall time 差异主要受少数 CuO 长淬火支配，不能据此
宣称 fixed 更快。

## 关闭什么，保留什么

1. 关闭“把 archive-scaled target 直接替换为固定 0.8 eV”的支线。
2. 不加入 OPES-like bias、CCQN/quadratic step、额外 trust 参数或 target 分类器；当前
   数据尚未识别它们要解决的可观测 failure mode。
3. archive scaling 暂时保留为默认，但降格为经验性的 exploration-intensity proxy，
   不是局域 barrier estimator。
4. 下一门控先记录每个 action 已经计算过的真实 PES micro-step energy：实际达到的上坡
   高度、何时达到、传播后构型的 true-quench conditioning 和 terminal basin。该记录不
   增加 PES 调用。只有证明“达到何种可观测逃逸状态”能稳定预测新低能 basin 和成本，
   才比较固定传播、局域反馈或限制性二次步；不再直接调 target 数值。

完整机械证据见 `evidence.json`；原始 seed-46 与 seeds-47/48 evidence 的 SHA-256、每条
accepted trace hash、执行 commit、用途成本和 per-case target 分布均包含其中。
