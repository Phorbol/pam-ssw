# U-T0：archive-scaled macro uphill target 零 FE 审计

## 结论

U-T0 返回 **`ADMIT_U_T1`**。12 条 S-CR1 轨迹的目标序列、最终 archive 大小、原始日志
哈希和下一次 action 的 `adaptive_step_target` 全部精确闭合；新增 force evaluation 为 0。

这个结果只证明 archive-scale target 是一个强活跃、此前未归因的算法块，不证明固定
0.8 eV 更好。它准入一个且仅一个 fixed-reference 因果门控。

## 实际使用的目标爬升能量

配置中的 `target_uphill_energy=0.8 eV` 只在 archive 尚无第二个 minimum 时使用。之后
controller 计算：

\[
E_{\rm target}=\operatorname{clip}
\left[0.2S(\{E_i-E_{\min}\}),0.04,4.0\right]\;{\rm eV},
\]

其中 `S` 是 archive 能量差的 MAD/正值中位数最大者。

跨四种 starter 策略合并后：

| 体系 | completed actions | target mean | median | min--max | 偏离 0.8 eV |
|---|---:|---:|---:|---:|---:|
| C60 | 192 | 1.125 eV | 0.887 eV | 0.466--3.479 eV | 97.9% |
| PdO | 301 | 0.423 eV | 0.408 eV | 0.121--0.800 eV | 98.7% |
| CuO | 111 | 0.226 eV | 0.191 eV | 0.088--0.800 eV | 96.4% |

PdO 与 CuO 除各自首个 action 外全部低于 0.8 eV。C60 的行为更复杂：不同 selector
改变 archive 的能量分布后，既会低于也会高于 reference；四臂均值为
0.965--1.195 eV，单步最高达到 3.479 eV。

## 底层物理语义

这个 controller 不是根据当前 starter 的局部曲率、局域 barrier 或当前方向的可达性选择
目标能量。它把**全部已发现 minima 相对当前最低点的能量离散度**当作下一 action 的目标
爬升尺度。因此：

- selector 改变访问和接受哪些 minima，会反向改变 uphiller；
- 搜索偶然发现一个高能或深低能构型，会改变后续所有 action 的尺度；
- 相同局部环境和方向可以仅因 archive 历史不同而获得不同 Gaussian width；
- 不同体系的 target 差异来自已发现 archive 能量跨度，而不是已验证的 barrier scale。

这不是自动错误。archive 能量尺度可能是廉价的 landscape 粗尺度。但它也不是原始固定
Gaussian SSW 的直接物理参数，并含 `0.2`、MAD/median 选择和 clipping 三层未经单独验证
的规则。

本 cohort 中 `adaptive_step_multiplier=1`、`adaptive_progress_boost=1` 在 12/12 case
保持不变。也就是说，这个巨大体系差异不是 escape/damage feedback 在线学出来的；活跃
机制几乎完全是 archive-scale formula 本身。

## 为什么现在值得做 fixed-reference gate

U0--U2 已经处理的是 walk 内部的 sigma/weight feedback，U3 处理 Gaussian tail，U4
处理 history、relax capacity 和 LS，H8/H14 处理传播长度。它们都没有删除 macro action
开始前的 archive-scale target。U-T1 因而不是重复调参，而是第一次隔离这个上游历史依赖。

fixed arm 直接使用已经存在的 0.8 eV reference，不拟合新值；micro-step trust、累计
Gaussian、方向、Safe-LBFGS 和 true quench 全部保留。若 fixed 失败，就关闭这个支线，
不调整 0.2、上下限或按体系选择固定值。

## 证据边界

- source raw evidence SHA-256：
  `8be48216862fd3fbdd14c0587f42f9dc1429805664a98e9d38c5275761342ec2`；
- compact U-T0 evidence SHA-256：
  `717a6a3822fc4ca394f2b235c253b0d882bbeb34ee254f12e6740d464cd82c8b`；
- 12 个 accepted log 的独立 SHA-256 均记录在 `evidence.json`；
- 新增 FE：0。

结论只适用于 seed-45 S-CR1 的 604 个 completed macro actions。它不证明 fixed target
有效、0.8 eV 最优、archive energy scale 与真实 barrier 无关，或任何 canonical sampling
无偏性。
