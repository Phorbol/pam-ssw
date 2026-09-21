# Native mode mixing and the update direction

日期：2026-09-17。范围是冻结 ELF 的有限静态闭合，加上已执行的
`update_mode0` 指令 probe。没有运行 LASP main、保护路径或 PES；本轮没有
调用 native RNG。probe 中的随机/旧方向输入是显式构造的控制输入，不能解释为
原始 RNG 样本。

## 结论

`update_mode0` 先从 selected record 构造位移

\[
s = x_{current}-x_{selected},\qquad n_0=s/\|s\|_2,
\]

并在 `0x5d5bf4–0x5d5c06` 调用 native `n_normal`。因此后续
`gen_randommode`（入口 `0x5d5c50`）收到的是归一化位移，而不是原先保留的
随机方向。`direction-update.json` 的 4/4 cases 对 `n=2,5` 和旧方向正负两种
输入均验证了这一点，最大方向误差为 `1.11e-16`；旧方向符号不影响该结果。

这一步只决定 generator 的方向输入。它不等于最终 bias anchor，因为 generator
还会按运行分支执行随机分量、约束、归一化和后续混合。

## c9 的保留与零方向分支

`get_random_mode0` 的十槽工作区按 `c[0..9]` 排列。它在
`0x5c04ec–0x5c0508` 整体复制到 control；本次审计中它写入 `c4,c5,c6` 的地址
分别为 `0x5c0493–0x5c049b`、`0x5c0385–0x5c0398` 和
`0x5c04c4–0x5c04cc`。更新后的 `c9` 是输入数组的第十个 double（偏移
`+0x48`），并没有被这些 `c4..c6` 单独写入覆盖。

在 `gen_randommode`：

* `0x5d5df8` 读取 `c9 = [rsi+0x48]`；与阈值比较后，`0x5d5e11` 的 `jbe`
  在 `c9 <= threshold` 时跳到 `0x5d5ff6`，而 `c9 > threshold` 才进入
  `0x5d5e17` 起始的随机/混合路径。该阈值是 ELF rodata 的比较常量；本记录不
  把它命名为物理参数。随后 `0x5d5eb6` 读取保存的 c9，`0x5d5ef5` 读取
  `object+0x1788` 的 n0；在 `0x5d5fba–0x5d5fc4` 对 n0 每个分量执行
  `n0_i <- c9*n0_i`。专属 probe 将 `c4=10,c5=c6=0` 注入 update，得到
  `c9=12`，在 `0x5d5ff9` 重汇合处读取的 n0 与 `12*n0_before` 最大误差为 0。
* 当 `update_mode0` 的归一化位移为零时，native `n_normal` 将整个向量置零。
  随后的 generator 仍被调用；已闭合的尾部在 `0x5d865a` 再次完成 `n_normal`，
  `0x5d865f` 检查零标志，零分支在 `0x5d870c–0x5d8732` 经表项
  `+0x188` 设置 `Allopt`，并写 `control+0x120=1`（`0x5d8732`）。该退出条件作用于最终混合结果；输入位移为零时仍可能由局部分量得到
  非零输出，不能直接断言终止，也不能称为失败后的自动随机重试。

## c4..c6 如何成为新的量

在 `update_mode0` 的有限 probe 中，设

\[
(c_4,c_5,c_6)=(0.2,0.3,0.4).
\]

静态的系数混合段 `0x5d5997–0x5d59d0` 将相关槽及当前 stage 的项相加，随后
乘 ELF 中的 `1.2` 常量并形成第十槽；在该受控输入下严格得到

\[
c_9 = 1.2(c_4+c_5+c_6)=1.08.
\]

这一关系在 4/4 cases 中通过，误差为 0。它是该 update 输入布局下的已验证代数
关系，不能推广成 `get_random_mode0` 每一条条件分支都必然同时产生非零
`c4,c5,c6`；这些槽的产生条件仍由 `0x5c0385`、`0x5c0493` 和 `0x5c04c4`
各自的随机/能量分支决定。

generator 中对应的重复处理块读取这些连续 double：`c3` 在
`0x5d6ed8` 和 `0x5d7376`，`c4` 在 `0x5d7634`，后续块还在
`0x5da284` 读取 `c5`，并由相邻分支继续处理 `c6`（`+0x30`）。它们对已有方向/辅助
场执行“系数乘法后相加”的同形操作（例如首个完整块
`0x5d68ba–0x5d68e1`）。所以它们提供的是附加方向量的权重，而不是新的
归一化因子；归一化仍由 `n_normal` 在 `0x5d6724–0x5d6733` 等块完成。

这闭合了 c9-positive 前缀中的主项缩放，但不能据此给出最终方向的圆锥保证。
后续 c3/c4/c5/c6 分支、`n_normal` 和约束投影仍可能改变方向；还没有在同一
次完整 generator 执行中闭合 `v=c9*n0+Σc_j u_j` 的最终表达式。相应地，
`c9 <= threshold` 跳到 `0x5d5ff6` 只说明跳过 c9-positive 入口块，不能简化成
“纯保留 n0”。

本轮最可靠的方向链是：

```text
selected record → s = x_current - x_selected
               → n0 = s / ||s||2
               → c4,c5,c6 coefficient mix → c9 = 1.2 Σ(c4..c6)
               → gen_randommode c9 branch and component mixes
               → n_normal / zero-state handling
```

这里的 `c9` 是保留下来的第十 double，并作为 generator 的分支门控；它不是
“把旧随机方向保留到最终输出”的证据。最终输出仍依赖 generator 所达分支及其
后续混合，需另做实际 bias-anchor probe 才能关闭。

证据产物：`research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/`
下的 `direction-update.json`、`record-and-zero.json` 和
`mode-mixing.json`；复核脚本为 `research/ga_ssw/probe_native_mode_mixing.py`。
