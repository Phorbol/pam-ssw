# S-CR0：node-level selector 支持度审计

## 结论

用户提出的 growing-arm 疑虑在现有三体系 seed-42 长任务中已经可见，而且比“最终 archive
达到几千个节点后才退化”更早出现：`archive_ucb` 在只有 19--74 个 minima 时，就使用了
最终 archive 的 62.2%--85.4% 作为 Shannon effective starter support；三体系平均为
72.4%。这不是严格均匀分布，但它比实际的 `uniform_archive` 更接近逐节点覆盖。

原因来自算法本身，而非 MACE/RDF 表征成本。每个新 minimum 都以 `node_trials=0` 进入
archive，固定 UCB-like exploration bonus 随即把它提升为新的未访问 arm。archive 继续
增长时，selector 的相当一部分动作会用于给新节点各付一次物理 action，而不是反复验证
已经显示生产力的低能 continuation。

## 九条已有 20k-FE 轨迹

| 体系 | selector | actions | archive | unique starters | effective support / archive | repeat fraction | energy drop |
|---|---|---:|---:|---:|---:|---:|---:|
| C60 | uniform | 42 | 40 | 22 | 49.3% | 47.6% | 15.424 eV |
| C60 | UCB-like | 49 | 40 | 38 | 85.4% | 22.4% | 21.990 eV |
| C60 | Metropolis | 40 | 38 | 16 | 26.5% | 60.0% | 21.671 eV |
| PdO | uniform | 72 | 63 | 33 | 44.0% | 54.2% | 4.124 eV |
| PdO | UCB-like | 79 | 74 | 56 | 62.2% | 29.1% | 5.362 eV |
| PdO | Metropolis | 74 | 68 | 24 | 26.0% | 67.6% | 6.769 eV |
| CuO | uniform | 17 | 18 | 10 | 48.7% | 41.2% | 3.195 eV |
| CuO | UCB-like | 18 | 19 | 14 | 69.6% | 22.2% | 3.009 eV |
| CuO | Metropolis | 20 | 21 | 5 | 16.4% | 75.0% | 3.197 eV |

三体系平均而言，UCB-like 访问了最终 archive 的 81.5% 个不同 starter，只有 24.6% 的
action 重复使用已有 starter；Metropolis 分别为 33.7% 和 67.5%。这说明二者不是同一
selector 的小参数差异，而是两种不同物理搜索机制：

- node-UCB-like 主要扩大 archive-wide starter coverage；
- Metropolis 主要沿少量已接受 minima 做 funnel continuation。

能量结果也不能支持“覆盖越广越好”。UCB-like 在 C60 与 Metropolis 近似并列，在 PdO
和 CuO 均不如 Metropolis；uniform 在 C60/PdO 明显更弱。这些是单 seed 的描述性结果，
不能单独否决 UCB-like，却足以否决“先给每个节点建立更复杂 posterior 就会自然改善”
的前提。

## 后续裁决

S-CR0 不新增 force evaluation，也不晋级任何 selector。它只准入一个最简单的物理对照：
每个冻结 archive snapshot 固定发出一个最低能 continuation 和一个 uniform restart。
这样全局 restart family 的概率质量固定为二分之一，不会随 node 数量增加而变成更多独立
bandit arms；低能 exploitation 也不会被大量一次性节点稀释。

只有这个两通道对照先在真实 20k-FE 搜索中成立，才有理由研究 continuation/restart
family 之间的 posterior budget allocation。当前数据不支持 node-level TS、更多 UCB
权重、top-k 删除、PCA/FPS 或结构表征升级。

## Claim ceiling

结论只来自已有 seed-42、C60/PdO/CuO、九条 20k-FE 轨迹的 starter-ID 分布。它验证了
小到中等 archive 中现有 UCB-like 的高支持度行为，不证明几千节点的渐近复杂度、因果
搜索收益、生产最优性或 canonical sampling 性质。
