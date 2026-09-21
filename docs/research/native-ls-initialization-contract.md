# LS 默认成键表初始化：原指令矩阵核验

2026-09-10。有界逆向结论；没有修改生产参数，没有执行势能计算。

本轮恢复并原指令核验了 `bond_info_init_` 无 `pot_bond_input` 文件分支的
元素对矩阵构建。**原版不是把查得的键能直接作为 LS 振幅：它还按全体系原子数与
成键数归一化。纯碳体系中，raw C–C 值与分母完全抵消。** 这改变了此前仅比较
发行版3.44684与论文3.61的解释，但尚不能据此声称两套最终 LS 势相同。

## 证据及执行边界

- 探针：`research/ga_ssw/probe_native_ls_initialization.py`。
- 完整输出（含输入元素、坐标、矩阵、独立计数及误差）：
  `research/ga_ssw/evidence/native-ls-initialization/result.json`。
- 原查表输出：`research/ga_ssw/evidence/native-ls-pair-table/result.json`。
- 外部反汇编目录：`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/`，
  文件 `ls-bond-info-init.asm`、`ls-bond-counter.asm`、`ls-pot-bond-add.asm`、
  `kernel-dwarf-member-offsets.txt`。
- ELF SHA256：`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。

从入口0x6c6a30执行至首次到达0x6c7530，停止在默认矩阵构建完成、后续调度之前。
实际执行了原 `selfadapt_nbondcounter_`、`reclat_loc_`、`mirror_min_dist_` 和元素查询
指令；没有用 Python 成键计数替换原函数。仅文件查询返回不存在、内存分配/释放、
memcpy/memset及输出例程使用运行时替身。每例上限2000万指令/20秒，未知运行时调用
直接报错。没有执行程序主入口、输入解析、自定义文件分支或 PES。

显式设置 LS 开关、50 Å立方盒、所有原子可动的 `fixatom` 数组。测试使用已有C60输入
和ASE G2 trans-butadiene；它们是真实原子几何，但本次仅证明初始化实现，不能当作
真实体系端到端搜索验证。元素过滤保持ELF静态值；两组尺度分别保留静态5.0及显式
覆盖2.5，后者只是检查线性关系的探针，不是推荐参数。

## 精确默认矩阵公式

以元素编号a、b，原子数N，原始查询值D_ab、l_ab，元素能量过滤f_ab、长度过滤h_ab，
尺度s表示，原指令构造

\[
L_{ab}=l_{ab}h_{ab},\qquad
B_{ab}=D_{ab}f_{ab}s\,
\frac{\operatorname{float64}(\operatorname{float32}(N)\operatorname{float32}(0.02))}
{N_b D_{CC}}.
\]

这里B是 `bond_ener_list`，不是最终成对势振幅。N_b由原几何成键计数函数得到，
为可动原子之间满足严格距离条件 `r_ij < L_(Zi,Zj) + len_toller` 的无序对数量。
默认H/C过滤均为1；本次独立计数采用相同阈值的欧氏距离，输入没有跨盒接触。
没有核验N_b=0、冻结子集或自定义过滤的行为，不将该表达式直接延伸为这些场景的
完整安全契约。

关键地址及成员（参数基址0x53ed7a0）：

| 对象 | 位置/值 | 证据角色 |
|---|---|---|
| bond_ener_scale | para+0x148，ELF静态5.0 | 0x6c9e5b乘入 |
| bond_filter | para+0x188，H/C项均1 | 0x6c9e3b乘入 |
| bond_lengthen | para+0x16e50，H/C项均1 | 0x6ca023乘入 |
| float32常数0.02 | 0x4a4c1a0 | 0x6c9e6d–0x6c9e79单精度N乘积 |
| D_CC | 0x4a4c188，3.4468400478363037 | 0x6c9e85作除数 |
| len_toller | 0x5520568，0.1 | 0x6c9efb起默认分支重设 |
| B矩阵描述符 | 0x55206a0 | 0x6c9e8d写元素 |
| L矩阵描述符 | 0x55205e0 | 0x6ca040写元素 |

两个元素过滤数组均按108×108 Fortran布局，元素(a,b)字节偏移为
`8*((a-1)+108*(b-1))`。静态默认不是正常输入解析后参数的保证；本轮未追踪解析器
是否覆盖s。能量/长度角色及eV/Å解释与已恢复势函数一致，但未找到表的原始书目或
正式单位声明；保留raw查询值的出处边界。

## 原指令与独立公式结果

| 输入 | s | N_b | B_CC | B_CH | B_HH |
|---|---:|---:|---:|---:|---:|
| C60，N=60 | 5.0 | 90 | 0.06666666269302368 | — | — |
| C60，N=60 | 2.5 | 90 | 0.03333333134651184 | — | — |
| trans-C4H6，N=10 | 5.0 | 9 | 0.1111111044883728 | 0.1385542756031437 | 0.1459172112635894 |
| trans-C4H6，N=10 | 2.5 | 9 | 0.0555555522441864 | 0.06927713780157185 | 0.0729586056317947 |

四例独立距离计数与原计数相同；公式和原矩阵最大绝对差均0。H/C元素顺序为[C,H]，
矩阵对称。L_CC=1.5399999618530273，L_CH=1.090000033378601，
L_HH=0.7400000095367432，tolerance均0.1。原指令查询D_CH=4.29817008972168、
D_HH=4.526579856872559。主agent另行重跑四例，报告相同零误差与计数。

纯碳且f_CC=1时，B_CC约为 `s*0.02*N/N_b`。由此可推断这里控制的是一个按成键数
分摊的全局尺度；不能推断每步偏置总能量恰为 `s*0.02*N`，因为成对距离形状函数、
激活列表及其他振幅因子仍参与。C60中raw3.44684抵消，所以只把它替换成论文3.61
不能说明有效软化强度变大或变小。

## 最终振幅仍有边界

已经恢复的 `pot_bond_add` 还使用

\[
A_{ij}=\mathrm{amp\_c}\,B_{Z_iZ_j}\,
\mathrm{atom\_filter}_i\mathrm{atom\_filter}_j.
\]

本探针边界上amp_c=2.0（静态地址0x5520450）。`para+0x16e08` 是atom_filter的
可分配数组描述符；空指针绝不能解释为标量过滤值0。本前缀未完成正常调用链中该
数组的初始化，也没有执行后续amp_c调度。因此“若amp_c=2且过滤全1，C60振幅约
0.1333333”仅是条件算术，不是原版首次LS步的实测振幅。论文方案 `0.03*3.61=0.1083`
也不能直接与表B_CC=0.0666667对比来宣称复现偏差。

结论：H/C无custom文件默认表的过滤、计数、尺度归一化已有原指令闭环证据。
下一步若实现发行版初始化兼容，先恢复输入解析后的s、atom_filter和amp_c调用状态，
再对完整首次成对势做E/F核验。当前保留论文参数配置，不把raw表或本表B接入生产
默认值，也不宣称完整LS自适应流程已恢复。

复现命令（Unicorn是研究用隔离依赖，不是独立SSW生产依赖）：

```bash
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -m research.ga_ssw.probe_native_ls_initialization \
  --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp \
  --c60 research/ga_ssw/evidence/independent-c60-gfn2-ls/input.extxyz
```
