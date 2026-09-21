# Native LS fixed-atom audit

日期：2026-09-11。范围是冻结 ELF 的 LS 键计数、分母和成对振幅数据流；不运行 LASP 主程序、不调用 PES、不修改生产代码。ELF SHA256 为 `bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。原始反汇编、摘录和事实 JSON 保存在 `research/ga_ssw/evidence/native-ls-fixed-atom-audit/`。

## 已闭合的合同

`pot_bond_mod_mp_bond_counter_`（入口 `0x6c4c80`）先调用 `bond_info_init_`（`0x6c4cda`）。在候选无序对循环中，它从 `para+0x2df30` 取得 `fixatom` allocatable 数据指针，并用 `para+0x2df70` 作下界偏移：外层端点检查位于 `0x6c4ec5–0x6c4ed2`，内层端点检查位于 `0x6c4f12–0x6c4f21`。两处都是 `comisd 0.0, fixatom[k]` 后以 `ja` 跳过，因此按该指令的可观察语义，只有 `fixatom[k] < 0` 会排除端点；`fixatom[k] == 0` 或正值仍进入距离判断。通过检查后，`mirror_min_dist_` 返回距离，严格条件在 `0x6c5050–0x6c5055` 比较，计数在 `0x6c5057–0x6c505e` 增加。`selfadapt_nbondcounter_`（`0x6c6530`）有同样的外层/内层固定值门（外层 `0x6c675d–0x6c6771`，内层 `0x6c67d9–0x6c67e6`）和严格距离计数（`0x6c6903–0x6c690f`）。所以 native 的已证分母是：先保留两个端点 `fixatom >= 0` 的候选，再计数 `distance < bond_length(type_i,type_j) + len_toller` 的无序对；它不是“只数两端都可移动”的合同。

`bond_info_init_` 在 `0x6c7ef2` 和 `0x6c9d1e` 调用该计数器；返回值在 `0x6c9d35` 保存到 `bond_info_init_$BONDNUM_SAVE`（`0x7916158`），并在 `0x6c9df3–0x6c9e85` 进入
`B(type)=D(type)*filter(type)*scale*float64(float32(N)*float32(.02))/(Nb*D_CC)`。
因此，只要某个输入把固定原子编码为零，固定原子仍改变分母 `Nb` 的计数，而不是从分母剔除；只有负值编码会走已观察到的排除门。当前证据尚未动态重放输入解析器把用户 `fixatom` block 映射成哪一种 per-atom 数值，因此“固定列表在该版本必然为零/负值”仍需标为输入映射问题。

## 振幅和 pair eligibility

`pot_bond_mod_mp_pot_bond_add_`（`0x6c5c50`）在 `0x6c5f27–0x6c5f3d` 读取全局 `nbonds`，在 `0x6c5fd1–0x6c5fdb` 读取已经建立的 pair index；在 `0x6c60d9` 取得 `para+0x16e08` 的 `atom_filter` 数组。`0x6c6130–0x6c6141` 的乘法数据流是

`A_ij = amp_c * bond_ener_list[type_i,type_j] * atom_filter[i] * atom_filter[j]`。

`fixatom`（`para+0x2df30`）和 `atom_filter`（`para+0x16e08`）是不同的 descriptor；在本次覆盖的 LS 初始化、计数和加势片段中没有从 `fixatom` 写入/改写 `atom_filter` 的指令。因此，已能确认的是：fixatom 的负值门影响 `Nb` 候选资格；pair 振幅由独立的 `atom_filter` 控制。不能把固定原子自动解释为 `A_ij=0`，也不能把 `atom_filter=0` 反推为从 `Nb` 删除。

pair 的最终 eligibility 仍有一个精确边界：`pot_bond_add` 消费的是 `allbonds` 列表而不是重新按 fixatom 过滤。该列表在本次 LS 入口片段之外的建立/更新位置，尚未追到能同时显示其写入端和 fixatom 的完整指令串。因此对于“负值 fixatom 是否让 pair 从 `allbonds` 永久消失、还是只使计数器跳过”的更高层合同，当前结论是**精确缺口**，不能声称完整恢复。已闭合的是计数器门和振幅乘法，不是所有上游列表生命周期。

`class_struc_mp_set_atomfix_`（`0x59d970`）读取同一 `para+0x2df30`，并在 `0x59da79–0x59db67` 以每项因子缩放力样数组；这证明该数据确实是 per-atom fix factor，但不替代 LS pair-list 证据。`readsswpara_` 对 `para+0x16e08` 的初始化/清零在 `0x6863b5–0x6865a9` 与 `0x6866d4–0x68671f`，是独立的 LS atom-filter 输入路径。

## 对 PAM 现状的影响

`pamssw/standalone/native_ls.py:_bonds` 当前遇到 ASE constraints 直接抛出 `NotImplementedError`。这是一项保守的未恢复标记；它不能被解读为 native 只统计 mobile-mobile pairs。现有 all-mobile Python 实现与已恢复 native 计数器的共同部分是严格距离条件和 `Nb` 进入 B 分母。要支持固定子集，至少还需先闭合两件事：用户 block 到 `fixatom` 数值（特别是零/负值语义）的解析映射，以及 `allbonds` 写入端是否复用同一门。没有这两项证据，不应实现新的 LS 策略或改变默认行为。

## 证据索引与限制

- `bond-counter-excerpt.asm`：`bond_counter` 的 `fixatom` 两端门、距离比较和计数。
- `selfadapt-nbondcounter-excerpt.asm`：同一计数器的独立实现路径。
- `pot-bond-add-excerpt.asm`：`nbonds`、pair index、`atom_filter` 和振幅乘法。
- `set-atomfix-excerpt.asm`：`fixatom` per-atom 因子用于力数组缩放。
- `readsswpara-atom-filter-excerpt.asm`：独立 `LS_atomFilter` 数组初始化/输入处理。
- `audit-facts.json`：地址、公式和已证/未证字段的机器可读摘要。

以上是静态 ELF 与既有隔离分析证据；没有声称动态主程序行为、完整输入解析覆盖、固定原子 pair-list 生命周期，或固定约束下的 PES/LS 科学效果。

## Allbonds 写入端已闭合

进一步追到同一 `bond_counter` 的 pair-list 建立。`pot_bond_var_def_mp_allbonds_` descriptor 位于 `0x5520520`，分配发生在 `0x6c54c2–0x6c54c9`。写入循环先在 `0x6c563c–0x6c5653` 检查外层端点的 `fixatom`，再在 `0x6c568f–0x6c56a2` 检查内层端点；两端均只有 `fixatom < 0` 才跳过。通过后调用 `mirror_min_dist_`，在 `0x6c57cc–0x6c57da` 应用同一严格距离门；通过时于 `0x6c581a–0x6c582b` 写入 pair record：距离/参考距离和两个 **1-based** 端点索引（record offsets `+0x20`、`+0x24`）。因此 pair eligibility 已闭合：`allbonds` 不是独立的“全 pair”列表，而是同一 `fixatom >= 0` 两端门加严格几何阈值的结果。

这也修正了此前的缺口表述：在当前 ELF 的 LS `bond_counter` 中，负 `fixatom` 同时从 `Nb` 和 `allbonds` 排除；零 `fixatom` 仍通过两端门。尚未闭合的只剩用户输入 block 到 per-atom 数值的映射，不能把 ASE `FixAtoms` 的零自由度直接等同于 native 的负值编码。

独立的 N=3 算术核对可由上述指令谓词直接得到：取三对都满足距离门时，`fixatom=[1,1,1]`、`[0,1,1]`、`[-1,1,1]` 分别保留 3、3、1（涉及负端点的两对被排除）；`[-1,0,1]` 保留 1（仅 pair 1–2，按 1-based 输出为 2–3）。这是从 `0x6c563c–0x6c56a2` 的实际分支条件逐项展开的无 PES 算术核对，不是对主程序的动态运行。

## `fixatom` block 的 parser 数值映射

进一步检查 `get_fix_` 的输入解析。ELF `.rodata` 中 `fixatom` 字符串地址为 `0x4a3e1a0`；`get_fix_` 在 `0x4e4314–0x4e4330` 将该字符串传给 block reader。随后每个用户范围被展开为 per-entry 数组。在 `0x4e4611–0x4e461f` 和 `0x4e463a–0x4e4648`，展开数组对应位置明确写入整数 `0`；写入循环由范围长度和起止索引控制（同一片段 `0x4e43e9–0x4e4680`）。因此当前输入

```
%block fixatom
1 297
%endblock fixatom
```

的语义证据是：对应 1-based atom entries 被写成 `fixatom=0`，而不是 `-1`。与前述 LS counter/allbonds 门结合，固定列表中的这些零值仍通过 `fixatom >= 0`，所以它们仍进入 `Nb` 和 `allbonds`；它们是否在后续 force projection 中被冻结由 `class_struc_mp_set_atomfix_` 的独立缩放逻辑处理，不能改变 LS eligibility 结论。

这个 parser 片段没有显示负值写入；负值仍是计数器门的可观察特殊输入，而非普通 `fixatom` block 的固定编码。独立摘录为 `get-fix-fixatom-parser-excerpt.asm`。因此 ASE `FixAtoms` 若要复现该普通 block 语义，应保留 pair eligibility/分母中的零值端点，同时在力/方向空间应用冻结投影；直接把它映射成“从 LS 邻居中删除”是不符合该 ELF parser+counter 证据的。
