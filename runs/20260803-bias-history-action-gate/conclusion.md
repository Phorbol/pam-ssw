# 累计 Gaussian 历史完整 action 门控：最终结论

## 裁决

本轮返回：

```text
RETAIN_CUMULATIVE_CLOSE_NEWEST_ONLY
production_default_changed: false
```

`newest-only` 不替换当前累计 Gaussian SSW，也不作为新的 posterior/UCB/TS arm。
它在 CuO 上明显降低成本，但该收益没有跨体系复现；在 C60 上还丢失了一个真实的
跨盆地 landing。继续增加 history decay、history length 或体系条件开关，会把一个清楚的
离散消融重新变成不可归因的连续启发式调参。

这个裁决不表示每一个旧 Gaussian 都不可删除。它表示：在当前 H8 serial SSW 中，旧 bias
既不是纯冗余，也没有形成一个可以被全局删除的稳定成本负担。

## 被隔离的物理机制

两臂分别使用

\[
B_{\rm cumulative}^{(k)}(x)=\sum_{j=1}^{k}b_j(x),
\qquad
B_{\rm newest}^{(k)}(x)=b_k(x).
\]

每个 pair 先共享：

- 同一个经过严格 true-PES quench 的 starter；
- 同一个 step-0 direction、Gaussian、proposal relaxation endpoint；
- 同一个 step-0 后完整 continuation state 和 RNG state。

从第二个 micro-step 起，两臂只改变旧 Gaussian 是否仍参与：

1. direction oracle 所看到的 modified PES；
2. history-bias gradient；
3. proposal relaxation objective。

完整 bias 列表仍保存在 continuation 和 trace 中。运行遥测证明，累计臂最多同时使用
8 个 Gaussian；18/18 个 newest-only action 从第二步起实际只使用最新的 1 个。

H8、方向生成与排序、local softening、adaptive sigma/weight、SAFE-LBFGS、true quench、
几何有效性和 basin 标签全部冻结。18/18 个 starter hash、初始方向 hash 和 execution
sigma 与上一轮 corrected two-operator gate 完全一致。

## 完整结果

| system | cumulative escape | newest-only escape | cumulative fully-loaded FE | newest-only fully-loaded FE | cumulative median ΔE | newest-only median ΔE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C60 | 5/6 | 4/6 | 2,227 | 1,928 | +0.3030 eV | +0.3029 eV |
| PdO | 6/6 | 6/6 | 2,104 | 2,288 | +1.3038 eV | +0.0874 eV |
| CuO | 5/6 | 5/6 | 4,384 | 2,612 | -0.3598 eV | -0.6884 eV |

配对 basin outcome：

- 两臂都逃逸：15；
- 只有 cumulative 逃逸：1；
- 只有 newest-only 逃逸：0；
- 两臂都返回 starter：2。

36/36 terminal actions 都有严格力收敛证书；无 fragmentation、预算截断或 unattributed
force evaluation。landing energy 的配对排序为 cumulative 更低 8 次、newest-only 更低
9 次、近似相同 1 次，说明删历史后不是单调变好或变差，而是进入不同 basin。

唯一丢失的 support 是 `C60 / h8_best / seed55`：cumulative landing 与 starter 的几何
RMSD 为跨盆地结果，而 newest-only 返回 starter。该差异只出现 1/3 seeds，未达到
“cumulative-only repeated context”门槛，但已经足以否定无损全局替换。

## 成本闭环

实际 campaign 共使用 13,841 FE，summed pair wall telemetry 为 324.77 s：

| purpose | FE | 总成本占比 |
| --- | ---: | ---: |
| biased proposal relaxation | 9,538 | 68.9% |
| direction oracle | 2,360 | 17.1% |
| terminal true-PES quench | 1,658 | 12.0% |
| escape true-PES checks | 201 | 1.5% |
| validation/bootstrap | 84 | 0.6% |
| unattributed | 0 | 0.0% |

fully-loaded action 成本会把每个 pair 的共享 step-0 prefix 分别计入两臂，因此不能与实际
campaign 总成本直接相加：

- cumulative：8,715 FE，207.76 s；
- newest-only：6,828 FE，152.62 s。

newest-only 总计少 1,887 FE（21.7%）和 55.14 s，但该平均由 CuO 主导。exclusive
purpose 对比显示：

- biased relaxation：4,868 → 3,196 FE；
- terminal quench：1,069 → 589 FE；
- direction oracle：936 → 1,184 FE。

它并非简单地“每一步更便宜”：删历史会改变路径长度、方向重选次数和 terminal quench
conditioning。

## 跨体系物理图像

### C60：旧 bias 保持不可逆性

newest-only 降低约 13% 成本，但少一个 escape。有限尾部 Gaussian 离开各自中心后会衰减，
然而旧中心的合力仍能抑制沿已走路径返回。删掉旧项后，某些路径重新进入 starter 的
吸引域。这正是 cumulative metadynamics-like memory 的本来作用。

### PdO：删历史反而使路径更长

两臂都是 6/6 escape，但 newest-only 多花约 9% FE。累计 bias 更容易把位移推到现有
walk-radius 边界，使 action 较早终止；删历史后多个 action 继续走到 H8，direction HVP
和 proposal relaxation 次数增加。因此“modified PES 项更少”不等于完整 action 更便宜。

### CuO：累计 bias 主要表现为优化刚性

两臂都是 5/6 escape，而 newest-only 少约 40% FE。这里多个累计 Gaussian 形成更复杂、
更强各向异性的 modified PES，SAFE-LBFGS 与后续 true quench 都更昂贵。删历史保留了
本 cohort 的 escape support，并显著改善成本和中位 landing energy。

这三种行为说明同一个旧 bias 同时承担两个相反作用：

\[
\text{阻止返回旧盆地}
\quad\text{与}\quad
\text{增加 modified-PES 优化刚性}.
\]

哪一项占主导取决于体系和局部路径。用体系名、原子数或一次 optimizer telemetry 触发
history switch，会是新的启发式，当前证据不支持这样做。

## 对研究路线的约束

1. 停止继续调 Gaussian history length、decay、sigma/weight feedback、fixed target 和 H。
2. 保留 cumulative H8 作为唯一 reference uphiller；newest-only 只保留为已关闭的诊断 control。
3. 不引入 OPES、CCQN、MD 或 GA 来“修复”本次负结果，它们是新的 action 假设。
4. 不建立 history-arm posterior。newest-only 没有任何独占 escape support，也没有跨体系
   一致的 escape/FE 优势。
5. 当前 uphiller 局部调参路线到此停止。下一项科学问题回到方向/action 的统计可学习性，
   但不能再用单次 terminal landing 当稳定标签。

## 唯一下一步

冻结当前 cumulative-H8 kernel，构造一个小型 repeated-rollout learnability gate：对少量
共享 starter context，按预注册 random、bond、momentum 三个低维 direction family 重复
采样完整 action，分别估计 family 内方差和 family 间差异。先用 leave-system predictive
log score 判断“执行前的 family/context 是否比 pooled baseline 更能预测 escape、低能
landing、成本和 invalidity”。

只有该统计辨识门控通过，才实现 family-level Bayesian posterior 和 batch allocation；
若不通过，就保留全支持的固定混合，贝叶斯只用于不改变动作支持的监测而不用于倾斜预算。
已验证的 K4 HVP 图批处理可以降低 direction-oracle wall time，但不改变 FE，也不应被写成
算法质量改进。
