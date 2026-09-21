# Ti/O native LS 初始化调用链审计

2026-09-17。本记录只核对一个调用链：`bond_info_init_` 的无
`pot_bond_input.txt` 分支，从元素 raw lookup 到长度/能量矩阵构建，边界为
`0x6c7530`。没有运行 LASP 主程序、PES 或新搜索，也没有修改生产代码。

## 原指令证据

ELF 为 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`，
SHA256 为
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。
既有 leaf lookup 探针在隔离 Unicorn 环境中仅执行 `bondeneval_`
(`0x6cb660`) 和 `bondlenval_` (`0x6cc0b0`)，本次将元素参数限定为
Z=22 (Ti) 与 Z=8 (O)，得到：

| pair | raw `bondeneval_` | raw `bondlenval_` |
|---|---:|---:|
| Ti–O / O–Ti | 3.6298000812530518 | 1.9429999589920044 Å |
| Ti–Ti | 3.6298000812530518 | 1.899999976158142 Å |
| O–O | 1.515779972076416 | 1.4800000190734863 Å |

Ti–Ti 与 Ti–O raw 能量相同是该 lookup 的观测结果；不能据此改写成化学解释。

随后直接调用已有 `probe_native_ls_initialization.Oracle.run`，输入一个 TiO2
几何（Z=[22,8,8]，Ti–O=1.943 Å，50 Å 非周期盒），仍停在同一边界。原指令
结果为 `N=3`、`N_b=2`、`bond_ener_scale=5.0`，两个元素过滤矩阵均为 1，
`len_toller=0.1`。初始化矩阵（元素顺序 Ti,O）为：

```text
B = [[0.15796207322120637, 0.15796207322120637],
     [0.15796207322120637, 0.06596389376180652]]
L = [[1.899999976158142, 1.9429999589920044],
     [1.9429999589920044, 1.4800000190734863]]
```

这与调用者公式逐项一致：

\[
L_{ab}=l_{ab}h_{ab},\qquad
B_{ab}=D_{ab}f_{ab}s\,
\frac{\operatorname{float64}(\operatorname{float32}(N)\operatorname{float32}(0.02))}
{N_bD_{CC}},
\]

其中 `s=5.0`、`D_CC=3.4468400478363037`。例如
`B_TiO=3.6298000812530518*5*0.06/(2*D_CC)`，结果为
`0.15796207322120637`。因此 raw lookup 先经过长度过滤、成键计数和全体系
原子数/成键数归一化，不能直接当作 LS pair 振幅。最终成对振幅还要经过
`A=amp_c*B*atom_filter_i*atom_filter_j`；本边界没有执行首次 `pot_bond_add`。

静态调用链位置与既有材料一致：`ls-bond-info-init.asm` 在 `0x6c6c81`、
`0x6c6cbd` 读取自定义文件分支，在 `0x6c6cc2` 后建立能量表、在
`0x6c6d86` 后建立长度表；已有审计将原几何计数条件定位为严格
`r < L + 0.1`。本次 TiO2 受限输入的原计数和独立几何计数均为 2。

## Python 对照与确定遗漏

当前 [native_ls.py](../../pamssw/standalone/native_ls.py) 已实现这条“显式传入
表 → 归一化 → 成键筛选 → 冻结振幅”的算术：`initialize_native_ls` 在
151–172 行使用 `D_CC`、`N`、`N_b` 和 `scale`；`_bonds` 在 83–102 行使用
MIC 与严格长度条件；`effective_amplitude` 在 105–108 行实现 `amp_c` 与两个
atom filter 的乘法。用 Ti/O raw 表显式传入时，上述受限 TiO2 数值可复现。

可确认的遗漏如下（本轮目标是无 custom 初始化；custom-file 不构成本轮前置）：

1. Python 公开的内置 raw/参考表只提供 H/C 与 H/C/O（16–29 行），没有 Ti
   (Z=22)；因此当前 TiO2 调用必须由调用方自行传入 Ti–O、Ti–Ti、O–O 表，
   不能声称已提供 Ti 材料默认初始化。
2. 当前接口明确是“无 custom file 的显式表”初始化（154 行）；没有恢复
   `pot_bond_input.txt` 的解析、覆盖优先级或其 atom-type 表语义。
3. 当前实现只支持 all-mobile 计数；约束/冻结原子路径显式抛出
   `NotImplementedError`（89、116 行），所以不能把它当作原版任意 fixatom
   初始化的等价实现。
4. 当前实现将 `atom_filter=0` 解释为保留成键计数但令最终振幅为零（115–116
   行）。本次原指令边界只观察到初始化过滤矩阵为 1，没有证据证明所有输入
   覆盖和零过滤分支已与原版完整一致。
5. 周期 `periodic-images` 是显式 Python 扩展；本次非周期 TiO2 不能证明
   原版任意晶胞镜像计数或自定义文件后的完整 parity。

这些是行为/接口缺口，不是 Ti/O 参数拟合建议。raw lookup 与初始化矩阵已经足以
把 Ti 表作为明确的 opt-in 复现参考表；它来自固定 ELF 的经验参数，不是物理最佳值，
因此不能把它静默升级为通用生产默认，也不能用 O/O 或 Cu fallback 替代 Ti lookup。

## 下一步与停止条件

若要关闭初始化验收缺口，下一项最小工作是对同一 ELF 恢复一个真实
`pot_bond_input.txt` 覆盖样例，并沿 Ti/O 调用链确认覆盖后的元素表、过滤和
`N_b`；随后才有理由核验首次 `pot_bond_add` 的 E/F。当前已取得的 Ti/O 证据
足以说明归一化公式和 Python 算术路径，不能升级为完整 native LS 或材料搜索
验证。到此停止本有界审计，不启动计算。
