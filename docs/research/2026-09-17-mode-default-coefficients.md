# Fixed-cell default mode coefficients: bounded native audit

日期：2026-09-17。范围限定为外部归档 ELF 的 `get_random_mode0` 系数分支、DWARF
参数字段和已有 `update_mode0` probe；没有执行 LASP main、PES、生产代码或作业。

## 输入与地址

核查 ELF：
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`

SHA256：`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。
`get_random_mode0` 的保存反汇编是
`research/ga_ssw/evidence/native-cluster-control-generator/get_random_mode0.asm`，
入口为 `0x5c0100`。十个临时 double 从 `rbp-0x80` 开始，对应
`c[0..9]`；最终整体复制到 `control+0x158..0x19f`，地址为
`0x5c04ec..0x5c0508`。

DWARF member map（外部 archive 的
`analysis/kernel-dwarf-member-offsets.txt`）给出：

| 字段 | `para` 偏移 | ELF 中的读点 |
|---|---:|---|
| `ratiolocal` | `+0x2db54` | `0x5c01d7..0x5c0203` |
| `ratio_atomcell` | `+0x2db58` | 该函数未作为普通 localmode 系数读点闭合 |
| `compress_mode` | `+0x2db70` | Run_type 5 分支 `0x5c0a77..0x5c0abb` |
| `modelevel` | `+0x2db78` | get_random_mode0 这段未读到 |
| `modelevel_cell` | `+0x2db7c` | get_random_mode0 这段未读到 |
| `lmode_q` | `+0x2db80` | `0x5c01aa` 分支到 pair-list 路径 |
| `localmode_pro` | `+0x2dba8` | `0x5c0475..0x5c0489` |

一个可复现实例是
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/runs/water-singlepoint-probe/allkeys.log`：
`Run_type` 的上下文由该归档运行使用，且第 75--82 行记录
`globalcompress=0.5`、`Lmode_Q=T`、`Ratio_local=50`、`Ratio_atomcell=2`、
`Modelevel=0`、`Modelevel_cell=0`。`allkeys.log` 是生效参数转储，不能单独证明
每个值都显式写在原始输入中。

## 普通 Run_type 5 的 c4..c6 可达性

Run_type 5 在 `0x5c0167..0x5c016a` 跳到 `0x5c0a53`；当 `object+0x1660 != 0`
时回到普通路径，并在 `0x5c071b..0x5c0725` 先写 `c[1]=1.0`。普通路径随后
计算

```text
localmode = 0.1 + 0.1 * int(para+0x2db54) * u
```

地址为 `0x5c01d2..0x5c0203`，其中 `u` 来自 `rd_numb_`，且 `0.1` 是 ELF
常量 `0x4a45e08`。在默认 `Ratio_local=50`、通常 `u∈[0,1)` 的条件下，
`localmode∈[0.1,5.1)`，因此为正。

若此前没有选择c5，后续 local-mode 分支（从 `0x5c03a2` 进入）有两个互斥结果：

* `localmode_pro > u_localmode` 或 group 整数和为零时，`0x5c0493–0x5c049b` 写 `c[4]=localmode`；
* `localmode_pro <= u_localmode` 且 group 整数和非零时，`0x5c04c4–0x5c04cc` 写 `c[6]=localmode`。

上述比较方向由本轮18项原指令检查核实，含相等边界；纠正早期稿中“测试成功/失败”
含糊且反向的解释。独立Python `select_local_coefficients` 对18项输出均匹配。
该函数限定Run5/modelevel0/Q-off/no-compression，不包含其它分支。
两条路径都随后到 `0x5c04d1` 的整体复制；更早的c5分支
`0x5c0350..0x5c0398` 写入后在0x5c039d直接跳到0x5c04d1，跳过c4/c6选择。
因此不能声称每个普通调用都必有非零c4或c6：也可能只有c5这一局部槽非零。故在默认正的 `localmode` 下，普通
Run_type 5 的 `c[4..6]` 全零不可达；关闭 `Lmode_Q`（`para+0x2db80` 的
bit 0）只跳过 pair-list 分支，仍会落到上述 `c[4]` 或 `c[6]` 之一。若走
pair-list 且成功选中记录，`0x5c0a28..0x5c0a47` 将 `localmode=5.0`，也不能
产生全零三槽。只有脱离默认参数域、使公式恰好得到零，或走出本核查范围的异常
早退，才可能改变这个结论；本 audit 不把这种未验证情形称为普通默认路径。

## c5 的激活条件与复现边界

`c5`（十槽的0-based索引）在 `get_random_mode0` 中只有一个明确写点：
`0x5c0385..0x5c0398`，写入 `localmode`，同时把 `control+0xc4` 置为 `-1`。
它位于 `0x5c0350` 对 `para+0x2db80`（DWARF 名为 `lmode_q`）的 bit-0 测试
之后：bit-0 为零时直接跳到 `0x5c03a2`，该写点不可达；bit-0 为一时，在
`0x5c036a..0x5c0383` 先将 `para+0x2db98` 与另一个参数内存值做 `max`，
再和新的 `rd_numb_` 随机数比较。比较不满足时仍跳过 c5，满足时才写 c5。

因此，`Lmode_Q=F` 明确关闭 c5 的这条激活路径；`Lmode_Q=T` 只使它有条件
可达，并不是必然启用。归档 `water-singlepoint-probe/allkeys.log:76` 的
`Lmode_Q=T` 只能证明该运行的生效设置，不能证明某一步确实命中随机阈值。当前
DWARF 表只可靠命名了 `lmode_q`、`mode_q_pro`（`+0x2db98`）和
`mode_q_cut`（`+0x2dba0`）；比较中另一个 `+0x2dbb8` 的字段在本核查中没有
完成成员命名，不能把它猜成某个用户选项。

这个激活判断本身只调用 `rd_numb_`、读取参数和对象计数/数组，并未调用 PES。
所以复现“get_random_mode0 是否写 c5”需要相同参数、随机流和相关对象状态；不
需要原始 LASP potential。若要复现 c5 对 generator 的最终几何输出，还需要
generator 所需的完整描述符、邻域/约束状态和其自身随机输入；仅凭坐标文件不足，
但这仍不是已证明的 PES 依赖。`gen_randommode` 对 c5 的下游语义（包括
`hinit_ptsd_str`/`hget_ptsd_mode` 的完整数据契约）在本任务中保持未解析。

## update_mode0 后是否自动补随机方向

已有 `research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/direction-update.json`
的 probe 在 `modelevel=0`、无压缩、无约束投影下验证了：`update_mode0` 入口
`0x5d55f0` 在 `0x5d5bf4..0x5d5c06` 调用 native `n_normal`，把
`current-selected_record` 归一化后作为新方向输入；它复制输入 `c4..c6`，并在
`0x5d5997..0x5d59d0` 形成 `c9=1.2*(c4+c5+c6)`。四个受控 case 均通过，
但 probe 在 `gen_randommode` 入口前停止。

因此，对“`c4..c6=0` 后 gen 是否自动另抽随机方向”的可支持结论是：
`update_mode0` 本身不把全零系数改成非零，也不在这里证明自动补抽。它会继续把
新归一化位移和 `c9=0` 交给 `gen_randommode`（入口 `0x5d5c50`）；generator
内部的完整随机/混合分支由另一份核查负责，不能从本 audit 越界声称“保持旧位移”
或“自动新随机”。若是零位移而非仅零系数，`n_normal` 确实把输入方向置零，
但不能据此直接说后续设置 Allopt：generator 仍可能由其它正系数分支生成非零
随机/辅助分量；只有最终混合向量本身仍为零时，`0x5d865a..0x5d8663` 的零检验
才会到达 `0x5d870c..0x5d8732` 的 Allopt/`control+0x120=1` 路径。

## c4..c6 与归一化的边界

在保存的 `gen_randommode` 反汇编中，统一采用十槽的0-based索引，三个槽的
读点和归一化边界为：

| 槽 | 系数读点 | 相关零门控/分支 | 该分量前的归一化边界 |
|---|---|---|---|
| `c4` | `0x5d7634` (`[rbx+0x20]`) | `0x5d7639..0x5d7641` | `n_normal` at `0x5d7a8d` (group path) or `0x5d7dd7` (pair path) |
| `c5` | `0x5da284` (`[rbx+0x28]`) | its following coefficient branch | `n_normal` in the preceding c5 workspace, before the weighted accumulation |
| `c6` | `[rbx+0x30]` at the c6 block (the earlier entry is `0x5d80aa`, with the later load at `0x5d82a1`) | corresponding positive-coefficient branch | `n_normal` at `0x5d828e` before the c6 weighted block |

这些块的数组循环以 `mulsd` 后接 `addsd` 累加。可见的 `n_normal` 是每个
workspace/分支的阶段性归一化，而不是“每个系数乘一个独立单位向量后再相加”
的充分证明；最终还在 `0x5d865a` 对混合结果统一归一化。因此，本报告只确认
系数的分支门控、workspace 归一化位置和带权数组累加，不能声称每个槽贡献恰好
一个单位范数向量，也不能从这些指令单独证明系数权重在所有异常输入下非负。

结论边界：本记录确认普通默认 Run_type 5 不会由“关闭相关选项”得到
`c4=c5=c6=0`，并确认 update 的系数传递与零位移处理；不替代
`gen_randommode` 主体的独立反汇编，也不推出最终 bias 方向或搜索效果。
