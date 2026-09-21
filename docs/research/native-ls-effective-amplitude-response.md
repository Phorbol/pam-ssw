# LS 有效振幅与响应更新：本轮收束

2026-09-10。结论：**当前论文模式的LS生命周期不必改成发行版更新策略。**
已恢复的发行版沿用“冻结成键→软面预淬火→真实能量响应→后续强度更新”的物理顺序，
但默认初始化、更新单位、频率和幅度控制与当前按论文式15实现的模式不同。
应保留paper模式并明确它不是release逐指令复现；将来需要发行版兼容时另设明确模式。
本轮没有修改生产代码或启动新的PES算例。

## 已确证的有效振幅

承接[默认初始化矩阵](native-ls-initialization-contract.md)，成对指数势为

\[
P_{ij}(r)=A_{ij}\exp[-(r-r_{ij}^0)/(0.2r_{ij}^0)],\qquad
A_{ij}=\mathrm{amp\_c}\,B_{Z_iZ_j}\,m_i m_j.
\]

`pot_bond_add` 0x6c6130、0x6c6136、0x6c6141实际连续乘入B及两个atom_filter；
0x6c61a4后直接将指数×振幅加到能量，没有在此额外除2。
amp_c静态值2.0。扫描0x400000–0x900000主程序指令只找到该值在pot_bond_add的
读取引用，未找到写入引用；这不是全程序不存在其他写入的证明。

readsswpara的0x686531–0x68659d明确将atom_filter初始化为1.0（常数0x4a49c60），
随后输入区间分支0x6866d4–0x686710可将指定原子区间清零。因此默认无该过滤输入时
`m_i=m_j=1`有初始化依据，不再把空静态描述符误解成未知标量。

有界原指令算术探针核验3组，包括正常C60、一个过滤为0及人为非整数过滤，均精确吻合：
C60原表B_CC=0.06666666269302368，经amp_c=2与过滤全1，A_CC=
**0.13333332538604736**。这是这些明确状态下的有效pair系数；探针没有执行正常
主入口/输入解析/完整首次LS步，不能称其为端到端LASP首次偏置的测量。

## 响应单位已闭合

`ssw_move`在软面优化计数0时保存真实能量（0x5bd701–0x5bd717）。在对应退出分支
0x5bf6b4–0x5bf6f4写入

\[
R=1000\,(E_{\rm after}-E_{\rm before})/N.
\]

常数0x4a45e00确认为1000。若物理能量为eV，保存的R和target的单位是meV/atom。
原指令测试E_before=-100、E_after=-99.8、N=10得到R约20，和独立表达式精确一致。
退出条件中也有优化步数上限；不能仅凭该响应值推断预淬火已达到力收敛。

输入读取0x686148–0x6861b0直接把 `SSW.LS_biasAtom` 读入para+0x180，缺省参数
指针0x4a498d0的值确认为20.0。因而这里默认目标对应0.020 eV/atom，与论文C60目标
在量纲转换后相符；用户输入同样按该保存响应单位比较，不能传0.02后声称等价。
本轮没有执行输入解析器。

## 发行版更新并非当前paper式15

令η=steplenselfadapt，c=steplenselfadaptmax，N_b_old为旧成键数，N_b_new为当前
成键数，T为目标。对非负B、正成键数的正常更新路径，静态指令给出

\[
Q=\max\left(1,\frac{\max_{ab}[B_{ab}\eta N|R-T|/N_{b,new}]}{c}\right),
\]
\[
B_{ab}^{new}=B_{ab}\frac{N_{b,old}}{N_{b,new}}
-B_{ab}\frac{\eta N(R-T)}{N_{b,new}Q}.
\]

0x6c7f0c–0x6c81e6计算最大变化再除c，0x6c8227与1取max；
0x6c8571–0x6c85a0执行上述逐元素更新。Q不是从paper推导的物理常数，而是发行版
现存数值幅度控制。原指令算术探针对R=0/20/40/10000和N_b_new=90/80的4组情形
核验更新表达式；**Q由静态公式预先计算并传入，未原指令执行整个max归约和调度**。
这些是算术验证输入，不是LS推荐参数或真实搜索结果。

ELF静态参数：η=0.005、c=0.01、freqselfadapt=10、npreselfadapt=100、
softmodecycle=100、softratio≈1.1、bond_ener_scale=5。除上面LS_biasAtom的显式
读取默认外，其余这里只称静态值，未核验所有正常输入覆盖路径。

调度地址0x6c7635起：step=0不更新；0x6c7ebc起在 `step % freqselfadapt == 0`
或 `step <= npreselfadapt` 时进入该更新。还有软化周期中的表保存/恢复分支；其中
0x6c7b98把保存的double响应转换为整数，恢复时再转double，属于发行版行为差异。
本轮未完整执行周期分支，不为其补猜测的高层语义。

当前 `LSResponseState.update` 使用论文模式：
`total_A_next = total_A - N*1.8*(response_eV_per_atom-target)`，再按新邻居的
标准键能比例分配。它与上面的发行版相乘更新、Q控制和调度不能通过简单改单位或
一个学习率就宣布等价。当前paper模式参数、失败处理、每步响应更新应保持明确来源；
没有证据要求为了这份发行版而马上替换它。

## 产物与边界

- `research/ga_ssw/probe_native_ls_response.py`：3个振幅、4个矩阵更新、1个响应保存，
  共8个原指令算术案例均通过；没有外部函数替身或PES调用，所有入口寄存器状态显式。
- `research/ga_ssw/evidence/native-ls-response/result.json`：完整输入、输出、静态常数。
- 同目录保存相关3份原始反汇编及readsswpara的LS前缀，便于审查地址与寄存器链。
- 原ELF SHA256沿用初始化契约，探针硬检查对应版本。探针语法检查通过。

剩余非阻塞项：完整首次bond_counter→pot_bond_add的E/F原指令串联、输入覆盖、
周期表恢复与失败预淬火后的更新资格。未得到这些证据之前，不宣称完整native LS
生命周期闭环。已有paper模式可以继续真实体系LS验证，但必须注明paper模式、显式
键表及参数来源，并与相同Safe-total SSW基线配对；不能混用原始表B作为pair A。
