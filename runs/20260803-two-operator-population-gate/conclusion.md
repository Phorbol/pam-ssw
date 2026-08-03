# Two-operator population gate：最终结论

## 最终决定

本轮先后完成了原始 Stage B 和一次 measurement-validity repair。修复后的预注册判决为：

```text
CLOSE_TWO_OPERATOR_PORTFOLIO
direct_viability_contexts: 0
ssw_exclusive_support_contexts: 6
numerical_acceptability: true
```

因此关闭

```text
direct displacement + true quench
```

作为当前通用 SSW 低成本互补算子的分支。它不进入 batch scheduler，不进入 UCB/TS，
也不参与后验 allocation。保留 H8 serial SSW 作为 reference action。

这不是因为 direct 算子更便宜但收益略低，而是因为在当前共享方向和位移尺度下，它没有
产生一个经过几何确认的跨盆地结果。

## 为什么原始 Stage B 的形式判决失效

原始实验完成了 C60、PdO、CuO 上 18 个 paired input 和 36 个 terminal action，形式判决为
`ADMIT_STAGE_C_DESIGN`。其中 direct 被记为 5 个 non-starter landing。

随后对保存坐标进行零新增力评估审计，发现这 5 个结果的 starter--landing RMSD 只有
0.0089--0.0209 Å，远低于 0.4 Å 几何阈值。它们只是比冻结 starter 多下降了
1.19--3.11 meV，恰好超过 archive 的 0.001 eV energy tolerance。现有 matcher 要求能量和
RMSD 同时满足阈值才能合并，因此仅由毫电子伏能量差便创建了新 basin ID。

更具体地说，原始 runner 使用 `STARTER_TRUE_QUENCH` 记账名做了一次能量/力验证，但没有
真正淬火 starter；direct arm 随后执行严格 true quench，于是同一势阱内残余弛豫被错误
解释成 escape。

这属于 observable 失效，不是 direct 的正向机制证据。

## 修复实验

修复只改变两个测量语义：

1. 每个 pair 在分叉前用与 terminal landing 相同的 optimizer 和 `fmax` 真正淬火一次
   starter；该成本在 pair 中只计算一次。
2. `same_starter_basin` 由等价淬火端点之间的几何 RMSD 决定。能量差继续记录，但不能单独
   创建跨盆地标签。

没有改变：

- C60、PdO、CuO 三个体系；
- bootstrap、H8-best 两种 starter context；
- seeds 55、56、57；
- direction generator、direction ranking 和局域软化；
- displacement scale；
- H8 Gaussian bias、proposal optimizer 和 terminal true quench；
- force-evaluation 上限和判决阈值。

18/18 个 starter SHA-256 和 displacement scale 与原实验相同；17/18 个初始方向哈希完全
相同。CuO/H8-best/seed56 的方向哈希发生变化，作为 float32 GPU 路径敏感性保留，未补跑或
替换。这个单点变化不影响 direct 在六个 context 中全部失去 viability 的结论。

## 修复后的物理结果

| system | direct 几何跨盆地 | SSW 几何跨盆地 | direct median FE | SSW median FE | direct median ΔE (eV) | SSW median ΔE (eV) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C60 | 0/6 | 5/6 | 35.5 | 317.0 | +0.000031 | +0.303055 |
| PdO | 0/6 | 6/6 | 39.0 | 336.5 | -0.000336 | +0.406342 |
| CuO | 0/6 | 5/6 | 39.0 | 776.0 | -0.000580 | -0.460327 |

汇总后：

- direct：0/18 个几何跨盆地；
- SSW：16/18 个几何跨盆地；
- direct-only support：0；
- SSW-only support：16；
- neither：2；
- 36/36 个 action 均得到收敛、几何有效的 terminal certificate；
- 无 fragmentation、无 budget censoring、无 unattributed force evaluation。

direct 的终点能量分布在修复前后几乎不变。这进一步说明原来的 5 个“成功”不是随机复验
丢失，而是同一批局部弛豫被新的正确 observable 重新标记。

## 底层物理图像

direct arm 执行

\[
x_{\rm direct}=x_0+\sigma u
\]

然后立即回到真实 PES 做下降。若这个点仍处于原 attraction basin，梯度流会抹去位移并
返回原极小值。当前 18 个方向和尺度全部属于这种情况。

SSW arm 从同一个初始位移出发，但累计 Gaussian bias，并在修改后的势能面上允许其余自由度
持续弛豫。bias 抑制了沿原路径返回的通道；正交自由度可以协同适应局部结构变化，直至越过
basin boundary。移除 bias 后再 true quench，16/18 次仍落入几何不同的 basin。

因此 proposal relaxation 不是可以由一次大位移替代的数值装饰。它承担的是“保持逃逸方向的
同时，让高维结构完成横向适应”的物理作用。

这个结果只证明 biased propagation 的必要性，不证明当前 adaptive Gaussian schedule、
trust feedback、H8 或 SAFE-LBFGS 已经最优。

## 成本闭环

corrected campaign 使用 9,166 次 force evaluation；原始 campaign 为 9,147，修复净增加
19 次。共享 starter true-quench 因 starter 已有 force certificate，实际只需 30 次
`starter_true_quench` 和 18 次相应 post-relax validation。

| purpose | force evaluations | 总成本占比 |
| --- | ---: | ---: |
| biased proposal relaxation | 6,210 | 67.8% |
| direction oracle | 1,272 | 13.9% |
| terminal true-PES quench | 1,487 | 16.2% |
| escape true-PES check | 113 | 1.2% |
| post-relax validation | 54 | 0.6% |
| shared starter true-quench | 30 | 0.3% |
| unattributed | 0 | 0.0% |

direct fully-loaded cost 为 682 FE；SSW fully-loaded cost 为 8,676 FE。direct 的便宜并没有被
昂贵 terminal quench 抵消，它确实便宜；但它的物理 support 为零，所以“每千 FE 成功率”
不能通过把同盆地 refinement 计作成功来人为抬高。

单卡 RTX 3060 的 summed pair telemetry 为 223.95 s。wall time 只作为执行遥测，科学预算仍
使用逐结构 force evaluation。

## 对整体研究计划的约束

1. 暂停 two-operator posterior、UCB/TS allocation 和 direct/SSW batch BO；不存在可学习的
   non-dominated direct arm。
2. 不因 direct 失败而引入 MD、GA、CCQN 或更多 operator。它们是新的独立假设，不能作为
   当前实验的 rescue component。
3. 保留 serial H8 SSW reference。现有证据再次确认 proposal propagation 是跨盆地的因果
   block，同时占据约三分之二计算成本。
4. geometry-primary 标签足以否定这批 direct escape，但并不解决一般 periodic、permutation、
   symmetry-aware minima matching；不能将本研究辅助标签直接宣称为生产 matcher 完成版。
5. 不改变 production default。

```text
production_default_changed: false
```

## 唯一下一步

返回 SSW-only 的单轴 propagation-cost 问题，但不再扫描 horizon：此前 H4/H8 复验已经保留
H8 并关闭 universal short-horizon 分支。下一轮应先复用当前 18 个共享 starter/direction
输入，构造一个只改变 biased propagation 更新律的 paired gate；terminal quench、direction、
H8 上限和 basin observable 全部冻结。候选必须先从现有 adaptive schedule 中删去或替换一个
明确的机制，而不是添加多个 Gaussian/OPES/CCQN 组件。只有当该单一改变在每千 FE 的几何
跨盆地支持上重复改善，才进入 SSW-only GPU active-set batching；否则保留当前 serial SSW，
转回方向/action posterior 研究。
