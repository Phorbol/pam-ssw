# H4/H8 三种子复验结论

## 决定

保留 production H8，停止 universal H4 分支：`RETAIN_H8_STOP_SHORT_HORIZON_BRANCH`。
不扫描 H5/H6，也不加入新的 adaptive early-stop 规则。seed49 的 H4 survivor 信号在
seed50/51 上发生了 C60、CuO 符号翻转；没有足够证据把固定四步提升为通用默认。

## 冻结边界与成本

18 个 arm 为 `C60/PdO/CuO × seed49/50/51 × H4/H8`。每个 H4/H8 pair 共享同一
bootstrap minimum；除 `max_steps_per_walk` 外，starter、方向池、静态方向评分、局域软化、
Gaussian bias、proposal optimizer、true quench 和 matcher 全部冻结。总 campaign budget
为 360,000 次力评估，实际使用 359,960 次，剩余 40 次，`unattributed=0`；单卡总 wall
time 为 7,781.37 s（129.69 min）。没有 H4 landing certificate regression。

## 预算归一化能量收益

表中为 `gain-AUC(H4) - gain-AUC(H8)`；正值支持 H4，负值支持 H8。

| 体系 | seed49 | seed50 | seed51 | 中位数 | H4 胜数 |
|---|---:|---:|---:|---:|---:|
| C60 | +3.4907 | -1.1543 | -0.5474 | -0.5474 | 1/3 |
| PdO | +0.0874 | +0.7426 | +0.6725 | +0.6725 | 3/3 |
| CuO | +0.2399 | -0.5083 | -0.2809 | -0.2809 | 1/3 |

## 物理解释

### C60：更多 terminal quench 不等于更高低能命中率

H4 在三个种子都完成更多 action，分别比 H8 多 29、29、16 次，同时也分别多 35、24、
13 次 duplicate。seed49 中更多 quench 偶然较早撞到深 basin，产生很大的正 AUC；seed50、
51 中同一吞吐机制没有重复，H8 反而获得更好的 AUC。C60 的 H4 均值仍为正完全由 seed49
大效应支配，而中位数为负。不能把“更多 action”当作方向或 propagation 质量。

### PdO：AUC 正号可重复，但 horizon 几乎不是 active constraint

H4 相比 H8 的总 micro-step 差仅为 -2、-6、-2，action 数差为 0、-4、+1。多数 walk 在
第四步之前已被几何或位移条件终止。因此三次正 AUC 更像相近传播长度下随机路径的早期收益，
而不是删掉第五至第八步带来的稳定吞吐优势。最终能量下降只有 seed50 支持 H4；seed49、51
都由 H8 得到更低末点。PdO 结果不足以推翻 H8 默认。

### CuO：复验支持连续累积 bias 的价值

H4 比 H8 多完成 9、14、15 个 action，并少走 20、12、14 个 micro-step。seed49 中更多
quench 较早撞到更低 basin；但 seed50、51 中 H8 用更少 action 更早进入更深 basin，AUC 和
最终最低能均获胜。这说明第五至第八步有时确实在累积 bias 并跨越短 trajectory 难以越过的
势垒，而不是普遍冗余成本。

## 对下一阶段的约束

1. horizon 不再作为全局常数继续调参；H8 保持生产默认。
2. H4/H8 最多作为未来 action-conditioned posterior 的两个离散 fidelity arm，并保留全支持。
3. 当前证据不支持立即训练 horizon selector：C60/CuO 的标签对 seed 路径敏感，PdO 的
   nominal horizon 又很少真正生效。
4. 下一实验转向共享 candidate pool 的方向来源与排序归因。必须同时记录完整 landing outcome，
   分开 escape、new basin、global improvement 和计算成本，不能再用单一静态 proxy 代替。

证据上限是三个 paired seeds 的机制停止门，不是统计显著性证明。
