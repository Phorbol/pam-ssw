# H4 与 H8 等总预算传播门控

## 裁决

H4 通过了第一轮 survivor gate，但不改变 production H8：C60、PdO、CuO 三体系的
20,000-FE gain-AUC 都高于 H8，三体系中位差为 `+0.239900 eV`，且相同总预算下没有
增加严格 landing certificate failure。它只获得多 seed repeat 的资格。

六臂共执行 120,000 次 force evaluation，GPU wall time 为 2,646.7 秒；每臂都精确
用满 20,000 FE，`unattributed=0`。每个体系的 H4/H8 共享完全相同的 bootstrap minimum、
能量和 bootstrap 成本。除输出路径外，effective config 只允许
`max_steps_per_walk: 4 → 8` 不同。

## 为什么比较 fixed H4/H8，而不是 adaptive early stop

U-O1 已证明当前 eV target 在 `per_atom_rms` 执行下不是物理 barrier height，因而“达到
target 就停止”没有充分物理语义。H4/H8 则是两个离散 fidelity：每个 action 要么走到
固定 horizon，要么由已有 geometry/displacement 条件终止，随后只做一次 true-PES
quench。它检验的是

\[
\text{单 action 的连续传播能力}
\quad\leftrightarrow\quad
\text{相同 FE 内可完成的独立 landing 次数}.
\]

## 三体系结果

| system | arm | gain AUC (eV) | final drop (eV) | actions | new minima | global improvements | proposal FE | landing FE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C60 | H4 | 24.7164 | 33.0321 | 79 | 33 | 12 | 10,650 | 6,352 |
| C60 | H8 | 21.2257 | 31.4456 | 50 | 39 | 15 | 13,276 | 3,606 |
| PdO | H4 | 5.6004 | 6.4176 | 77 | 70 | 6 | 12,222 | 5,244 |
| PdO | H8 | 5.5130 | 6.7176 | 77 | 68 | 10 | 12,138 | 5,179 |
| CuO | H4 | 0.7931 | 1.0976 | 42 | 42 | 10 | 14,406 | 2,895 |
| CuO | H8 | 0.5532 | 0.6400 | 33 | 33 | 5 | 14,774 | 2,187 |

### C60：H4 更早命中深 basin，但 continuation capacity 确实被削弱

H4 少走 18 个总 micro-step，却完成 29 个额外 action；proposal relax 少 2,626 FE，
landing quench 多 2,746 FE。它更早进入约 `-507.70 eV` 的 basin，因此 gain-AUC 高
3.491 eV。

但 H4 只有 33 个新 minimum、12 次 global improvement，并出现 46 次 duplicate；H8
有 39 个新 minimum、15 次 improvement，duplicate 仅 11 次。H4 的正结果不是“后四步
都是冗余”，而是这个 seed 中更频繁的 terminal quench 更早撞到一个深 basin。后四步
仍提高了单 action 的 basin 多样性。

### PdO：horizon 几乎没有被实际控制

H4/H8 分别只有 119/121 个总 micro-step，完成 action 都是 77。绝大多数 walk 由已有
geometry 或 displacement clip 在 H4 前结束。H4 的 AUC 只高 0.087 eV，而 H8 最终低
0.300 eV。这里不能把小正 AUC 当作 H4 机制支持；它主要说明 PdO production path 对
这个 horizon 开关不敏感，float32 路径分叉足以改变后续 basin 链。

### CuO：最干净的 H4 正信号

H4 少 20 个 micro-step，完成 9 个额外 action；42 个 action 全部得到新 minimum，global
improvement 从 5 增至 10，gain-AUC 高 0.240 eV，最终能量低 0.458 eV。proposal FE
仅减少 368，因为节省的预算被重新用于更多 action；landing quench 增加 708 FE。
这说明 H4 的价值不是让同一个 action 更快，而是把昂贵的串行 modified-PES relaxation
预算重新分配给更多独立 quench。

## certificate 指标修正

首次派生 decision 错把 landing success rate 直接比较：PdO H4 为 `74/76`，H8 为
`75/77`，从而产生 `0.973684 < 0.974026` 的伪回退。action history 显示两臂都恰有两个
真实未收敛 landing；H4 另有一个在 landing 前耗尽预算的 right-censored action。

等总 FE 问题应比较每固定预算的 certificate failure 数，而不是不同 attempted-landing
分母下的成功率。修正不需要新 FE，原始 execution evidence 和修正 evidence 的 SHA256
都已保留。回归测试要求：相同失败计数不能因 attempted-action 分母不同而产生回退。

## 对 direction 与 selector 的含义

本实验冻结了 random/bond/momentum 候选、central-FD HVP、静态 direction score 和
Metropolis starter chain，因此没有验证任何 UCB 或 posterior。它进一步说明 direction
quality 必须与 propagator fidelity 联合定义：同一初始方向在 H4/H8 下会进入不同的
escape configuration 和 landing basin，不能给方向贴一个脱离传播机制的全局“好/坏”标签。

当前方向研究已经支持“静态 scorer 经常错过更好候选”，但不支持“最软”“最低真实曲率”
或“短期上升最大”作为通用替代。starter `archive_ucb` 同样只是 fixed-weight UCB-like
baseline，且 growing-node arms 会稀释重复证据。H4 repeat 之前不修改它们，避免把 horizon
收益与 direction/starter allocation 混合。

## 下一步与停止线

下一步只允许 H4/H8 多 seed repeat，仍冻结方向、starter、bias、proposal optimizer 和
true quench。需要确认：

1. C60 的 AUC 正号是否能跨路径敏感性重复，同时监测 duplicate return；
2. CuO 的 action-throughput 收益是否重复；
3. PdO 是否继续表现为 horizon-insensitive。

若重复中出现体系/seed 符号翻转，不做 H5/H6 扫描，也不增加 adaptive stopping 规则；保留
H8 production default，并把 horizon 作为具有全支持的离散 action arm，等待以后只有在
context 可预测性通过留体系验证后才交给 posterior allocator。
