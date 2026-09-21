# Native LS domain audit: MIC pairs, periodic images, and fixed atoms

日期：2026-09-12。周期镜像扩展已实现，默认仍为 `native-mic`。
这是独立 Python/ASE 实现；没有运行 LASP 主程序或 GPU。原版 MIC 行为
与显式周期扩展分开保留，不能把扩展称为原指令完全复现。

## 当前 Python 行为

`NativeLSSettings.bond_geometry` 支持两种模式：

- `native-mic`（默认）：`i<j + find_mic`，每原子对一个距离，无自镜像。
- `periodic-images`（显式选择）：枚举 `(i,j,S)`，与 `(j,i,-S)` 规范化去重，
  包括非零自镜像；对非周期体系退化为原来的普通原子对行为。

两种模式都使用原来的严格距离门、native B 算术、响应单位和 cycle state。
镜像模式的初始化、后续强度更新及冻结势使用同一组镜像记录和同一个 Nb；
**不采用 MIC 计数与镜像求和混合的方案**。后者会无依据地放大软势。
冻结后保留 image shifts 和参考距离，固定胞的连续坐标运动不重新选镜像。
该接口不提供应力，不能作为 VC-LS 完整实现。ASE constraints 仍明确拒绝。

## 为什么需要显式扩展

以理想 fcc Cu、a=3.6 Å、D=1 eV、length=2.9 Å 和已有 scale=5 为
几何诊断输入，根智能体独立检查得到：

| 模式 | 原子数 | Nb | 软势 E/N (eV) | 最大原子软势力 (eV/Å) |
|---|---:|---:|---:|---:|
| native-mic | 4 | 6 | 0.05802415916 | 0.1861125773 |
| native-mic | 8 | 20 | 0.05802415916 | 0.04558808490 |
| native-mic | 32 | 192 | 0.05802415916 | 约 3.2e-17 |
| periodic-images | 4 | 24 | 0.05802415916 | 0 |
| periodic-images | 8 | 48 | 0.05802415916 | 约 1.7e-17 |
| periodic-images | 32 | 192 | 0.05802415916 | 约 1.7e-17 |

这里旧模式的 E/N 已因 N/Nb 归一化而相同，不能声称其能量广延性必然失败。
问题是短胞中最小镜像选择未保留完整的对称邻居，导致完美晶格上的软势力
不相消，且随超胞表示改变。完整镜像求和修复这一几何问题；这本身还不是
全局搜索效率优势的证据。保留 float32(N)*.02 的 native 算术也意味着
一般复制测试应允许对应的浮点误差，不能要求任意 N 的逐位相等。

新增检查覆盖原胞自镜像、非零位移下有限差分、完美晶格力相消、重复结构力
等变性、冻结镜像连续性、非周期等价和非法输入。完整 standalone 回归为
511 passed / 1 skipped；skip 需要显式 native MC ELF 环境变量。
真实 Cu/Al 完整晶体与空位晶体的有界 EMT 对照另行记录；D=1 eV 是诊断
参数，不能冒充原版金属查表或经过材料标定的默认参数。

## ELF 证据与已实现 cycle state

已有静态反汇编和事实记录来自 ELF
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`
（完整 SHA256 与 `docs/research/native-ls-fixed-atom-audit.md`、
`native-ls-initialization-contract.md` 一致）。关键观察是：

- `bond_counter` 和 `selfadapt_nbondcounter` 调用
  `mirror_min_dist_`，在严格 `distance < bond_length + len_toller` 后对
  原子无序对计数；现有证据没有显示多镜像 records 或 self-image 计数。
- `allbonds` 写入端同样对端点应用 `fixatom < 0` 跳过门，再写入一对
  **1-based** 原子索引和距离；它不是 image-resolved list。
- 普通 `fixatom` block 的 parser 写入 per-atom `0`。因此普通固定列表的
  零值仍通过 native LS 的 `fixatom >= 0` 计数/allbonds 门；只有负值才
  排除候选。`fixatom` 与独立 `atom_filter` 不是同一数组，不能把固定
  原子自动解释成零振幅。
- `NativeLSCycleState.advance` 已实现 adaptive/nonadaptive 的
  save-zero/off/restore、Fortran NINT、响应整数 note、restore-before-normal
  update 和 transactionality。旧的 “periodic save/restore unrecovered”
  docstring 只准确描述 `update_native_table(branch=...)` 的单独分支参数，
  不应覆盖该 state class 已有的纯状态机实现。
- 默认 `cycle=100, ratio=1.100000023841858` 给出 `nsoftstep=110`，不满足
  `0<nsoftstep<cycle`，所以默认 native-derived 设置不会进入 save-zero/
  restore 周期分支。周期状态机仍已由静态 fixture 覆盖；它不是完整 native
  caller 的步号、接受种子关联或失败资格证明。

固定原子证据足以支持“native 计数器使用 per-atom fix 值门，零值不从
`Nb`/allbonds 删除，负值才排除”这一窄结论；不足以证明 ASE `FixAtoms`
如何映射成 native 的数值数组，也不足以证明后续力投影、LS pair eligibility
和用户输入的全部生命周期。因此当前 `_bonds` 保守拒绝 constraints 是清楚的
未恢复边界，不是“native 只数 mobile-mobile pair”的结论。

## 剩余核心边界

固定原子现已通过 `run_constrained_ssw(..., ls=NativeLSSettings(...))`
接通：内部副本清除约束以构造全部键，既有活动坐标负责固定端点和力投影，
没有删除固定端点或自动将其振幅归零。Cu/Al表面证据见
`2026-09-12-constrained-native-ls-results.md`。底层 `native_ls` 仍拒绝直接
传入ASE constraints；该适配只覆盖现有FixAtoms范围，不是任意ASE约束或
完整native输入映射。

本扩展没有改变 native cycle state、响应目标、GA、VC/RC 或默认优化器。
多体系端到端结果用于检验执行行为和失败边界；短轨迹不能证明泛化效率，
仍需与普通 SSW 在相同 oracle 预算下进行独立比较。
