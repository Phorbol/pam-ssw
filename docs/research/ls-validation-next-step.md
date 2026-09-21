# LS-SSW 下一步真实体系验证：输入、参数与可比预算

2026-09-10。只读审查与实验准备；没有启动新 calculator、长计算或HPC任务，没有修改生产算法。

## 当前能支持的结论

`softening.py` 实现论文无序pair指数势及解析力，冻结当步pair和参考距离；
`ls_cycle.py` 在真实驻点检查后做E+LS预淬火，以真实E差/N测响应；
`LSResponseState.update`按论文反馈更新下一步总强度，再按键能权重分配给新pair。
`paper_reference.run_ls_ssw`接到SSW爬坡、撤掉所有偏置的真淬火、MC及下一步更新。

已完成真实体系证据位于：

- `research/ga_ssw/evidence/independent-c60-gfn2-ls/`：一外层步、两Gaussian，137搜索请求；
  另7初始淬火请求、2独立复核，总146；52.55秒。软响应0.0021240133 eV/atom。
  initial和landing都通过0.01 eV/Å真力检查；能量仅降低约3.59e-5 eV。
- `research/ga_ssw/evidence/dimer-ritz-c60-ls/`：同seed分别137/153搜索请求、51.35/54.72秒；
  两种solver都得到相近末能与真力。它比较方向数值求解器，不是SSW vs LS消融。

尚未验证：LS多步适应是否达到目标响应、冻结pair更新后的稳健性、GA+LS完整真实体系循环、
LS对有效新basin/化学重排的增益、相同总成本下效率、跨体系泛化。现有C60没有不同basin
认证和完整Hessian认证；不把两个真力合格记录称为两个极小值。

## C60初态：不能直接用于“找GM加速”

现有 `input.extxyz` 来自已归档脚本的 `ase.build.molecule('C60')`；不是随机碳团簇，
是预先构造的buckyball。`initial-quench.json` 和 `result.json.initial`保存其GFN2-xTB淬火结果。
这适合检查从低能笼状结构逃逸、探索邻近结构和LS响应，不适合衡量“从未知结构找到buckyball的速度”。
不能进一步称它已经是GFN2 PES的全局最低点：尚无GFN2异构体全局能量比较。

论文215.txt §3.2/Table1采用C-H-O G-NN势，从随机结构寻找fullerene；与我们的GFN2模型不同。
论文对C60的平均步数增益本身仅约8.8%，比C70更小。因此不应预设在当前buckyball短程探测中
一定看到LS巨大加速，也不能把没找到更低能结构解释为算法失效。

## 参数来源逐项核对

原始材料：Guan, Shang, Liu, *Local-Softening Stochastic Surface Walking for Fast Exploration
of Corrugated Potential Energy Surfaces*, DOI [10.1021/acs.jctc.4c01081](https://doi.org/10.1021/acs.jctc.4c01081)。
本轮实际读取本地完整 `literature/215.txt` 和 `ct4c01081_si_001.txt`；
[官方SI存档](https://acs.figshare.com/articles/journal_contribution/27969226)的本地API快照仅列一个PDF。

| 参数 | 可直接使用的数值/定义 | 来源与限制 |
|---|---|---|
| C-C标准键能 | 3.61 eV | 正文§2.3，引用CRC Handbook；不是由GFN2拟合 |
| 初始比例 | 0.03 | 正文§2.3 |
| xi | 0.2，无量纲 | 正文eq11 |
| 反馈lambda | 1.8 | 正文eq14 |
| C60响应目标 | 0.02 eV/atom | 正文§3.2；SI SAbiasAtom=20，输入单位换算不可误作20 eV |
| 当前C-C cutoff | 1.886864590515379 Å | 现存初态第3近邻最大1.4416422和第4近邻最小2.332087的中点；只是该初态壳层规则 |
| 论文C60温度/NG/位移 | 150 K / 12 / 0.6 Å | SI§7.3–7.4；不是旧单步probe的300 K/2/0.2 Å |
| 论文C60真力阈值 | 0.05 eV/Å | SI；可使用现有较严格0.01，但必须明示差异 |
| 现有GFN2精度 | tblite0.7.0 accuracy=0.001 | 前期数值诊断选择，非论文NN设置 |
| fd_step/rotation_tol | 1e-4 Å / 0.02 eV/Å² | 现有数值设置，不是通用物理参数 |
| rotation_bias | 100 eV/Å² | 当前已声明研究参数；不把其称作论文最优值 |

C60 cutoff可以作为固定、公开的试验输入沿用，但它尚不是原LASP元素表。
若后续结构严重改变配位，需检查pair集合和化学合理性；不要按观察到的成功率反复调cutoff。
文献SI还有globalcompress、vapor_cri、Ratio_Local等native选项，独立walker未复现它们；
将温度/NG/位移改成论文数值也不等于整体论文配置等价。

## 可直接使用的最小实验选择

选择现有 `independent-c60-gfn2-ls/result.json` 中的initial结构，后端继续使用已验证可用的
`/tmp/pam-ssw-tblite-20260909` 下tblite0.7.0/GFN2-xTB。无需新输入模型或未知键表。

科学问题限定为：在同一低能C60笼状结构附近，LS是否比SSW增加 **真实淬火后的内部结构探索**，
及反馈是否把可测的真实能量响应拉向目标。暂不做GM速度或原论文效率复现声明。

预登记两臂：SSW和LS-SSW；均使用标准dimer和显式Eckart cluster截面，排除已发现的整体刚体
泄漏。固定两个seed：20260909、20260910；每臂每seed独立calculator与工作目录。
LS独有参数按上表；其他参数完全相同：T150 K、max_gaussians12、width0.6 Å、
fmax0.01 eV/Å、relax_steps400、fd_step1e-4 Å、rotation_hvp100、rotation_tol0.02、
rotation_bias100、forward_force0.1、paper方向采样。
这些是带明确数值替代和Eckart几何的独立算法实验，不是native对照。

建议每条先设置最多10个outer steps作为操作上限，单CPU、不并行占GPU；壁钟上限在启动前
由主agent/用户按已有单步成本确定，不在此授权或启动。预期四条至少是分钟级而非单元测试。
如果预算只允许原来两Gaussian一外层步，应继续标记工作流诊断，不能替代本实验。

## 预算匹配：不能把相同步数当成相同成本

LS每步多一次soft-only quench且会改变后续优化难度。必须记录所有surface requests，
包括重复/失败请求、初始淬火、软面预淬火、方向HVP、偏置淬火和最终真淬火；
独立验证成本另外逐项报告，也纳入总实际成本。

现有run返回每步请求数，足以对 **完整返回** 的轨迹做共同调用预算的前缀比较：

1. 记录每个可用landing完成时的累积调用数（先计initial请求）。
2. 预登记目标比较预算B=1000次搜索接口请求；比较每条曲线在B前已经完成且通过认证的landing。
   跨过B才完成的landing不能计入，即使它很好；尚未完成步消耗仍记为成本。
3. 如果某条在10步操作上限前没有达到B，明确报告未达到预算；可附加按
   `B_common=min(各完整轨迹总请求数)`匹配的诊断前缀。该规则只使用成本，不看结果择预算，
   但不能冒称完成了原定B=1000比较。报告完整运行总成本，不隐藏用于前缀分析的超额计算。
4. 如果需要真正硬停止于B，当前 `run_ssw` 没有完整checkpoint/部分结果observer，单纯在
   ASESurface抛预算异常可能丢失已完成records。应先补研究runner的中断与事件持久化，
   再做硬预算实验；不要用异常后残留的最后坐标当成合法landing。

这给现有runner一个无需改变生产算法的可执行前缀比较方案；严格硬预算runner仍是明确准备项。
同seed只保证相同开始随机流，不保证后续路径消费相同；不应把它当成逐步相同扰动对照。

## 验证产物与停止条件

保留每步LS pair数量、sum(A)/N、响应、下一步强度、真实/修改能量、Gaussian数量、失败状态、
frame分支失败及全部请求。保存初态和每个合法真landing，不只保存best。

从头用独立GFN2 calculator重算能量/力；严格淬火统一终点精度后再做旋转平移及同元素置换
处理的结构比较，配合C-C连接图、配位、笼是否破裂/解离及必要的Hessian证据。
不能只用Kabsch固定索引或一个能量阈值宣布不同basin。几何证据不明确时列为待判定。

预期可证伪：LS增加真实内部变化而非只改变笼尺度，且至少不会用额外调用伪造效率。
若两臂都回同basin，即使响应控制正常，也只证明该小预算下未得到新basin；
若LS频繁负强度/预淬火失败/解离，应保留失败并审查目标与GFN2模型适配，不调参抹掉失败。
两seed只够诊断，不够普适或统计显著效率声明。

## C4H6：现有实际结构与真正缺口

上传GA examples在已检索目录没有C4H6坐标；LS SI只有lasp.in与图示，没有机器可读C4H6坐标附件。
但本机ASE g2数据库 **实际已有**：`butadiene`、`bicyclobutane`、`cyclobutene`、
`methylenecyclopropane`、`2-butyne`，均为10原子的C4H6结构。
来源文件 `/home/gengjianrui/.local/lib/python3.12/site-packages/ase/collections/g2.json`。
本轮只读几何检查：butadiene碳索引0–3，链二面角180°；C-C为1.342467、1.456254、1.342467 Å，
对应可用的trans起点。没有生成新结构、没有运行能量。它不是作者的确切输入。

论文§3.1的目标是从trans-butadiene探索异构体，所以初态是最低异构体并不妨碍这个**覆盖**目标。
设置为PBE、T150 K、NG25、ds0.1 Å、Upsilon0.7 eV/atom、400 minima；
GFN2可作较便宜的独立模型对照，不能复制PBE反应能垒/成功率结论。

本轮开始时完整C-H/H-H键表尚未恢复；随后对发行包、Java和ELF做了有界核查，
发现了可直接执行的native来源。因此 **不再把这项缺口当成需要用户提供新论文的问题**。
现有SI已经足够明确数学形式，下一步是核查已定位的原程序初始化缩放。

## 本轮新增：H/C原指令查表证据

发行包只有可执行文件/JAR/README等，未找到pot_bond_input.txt自定义表。
Java `nna.BasicInfo.getPredictedBondLength`用`ElementPara`原子半径和服务描述符/GA过滤，
不是native LS键能来源，不应混用。

ELF `bond_info_init_` 的默认分支明确调用：

- 0x6c9dd2 → `bondeneval_`，函数入口0x6cb660；
- 0x6c9fd3 → `bondlenval_`，函数入口0x6cc0b0。

已用只映射这两个函数code/rodata的Unicorn执行8次查询，每次限10000指令/1秒，
没有外部调用替身、没有运行LASP初始化/main或任何PES计算。完整probe及输出：

- `research/ga_ssw/probe_native_ls_pair_table.py`
- `research/ga_ssw/evidence/native-ls-pair-table/result.json`

```bash
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python -m research.ga_ssw.probe_native_ls_pair_table --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp
```

| pair | bondeneval_ 原始返回 | bondlenval_ 原始返回 |
|---|---:|---:|
| H-H | 4.526579856872559 | 0.7400000095367432 |
| H-C | 4.29817008972168 | 1.090000033378601 |
| C-H | 4.29817008972168 | 1.090000033378601 |
| C-C | 3.4468400478363037 | 1.5399999618530273 |

**发行版 C-C 查询返回3.44684，与论文明确举例的3.61不同。** 这不是浮点误差，
不能把当前论文版3.61参数悄悄改掉后继续称完全相同。表值的能量/长度角色由函数名、
调用链和数量级支持，工作单位对应eV/Å的解释与其他已恢复势函数一致；但本轮没有
DWARF单位声明或标准键能原表出处，严格产物保留“raw_return”标记。

静态 `bond_info_mp_len_toller_`地址0x5520568为0.1；默认分支0x6c9efb–0x6c9f0c
还显式写入double0.1。`ls-bond-counter.asm`接受距离<bond_len_list+len_toller。
如果length-filter全1且没有自定义文件，C-H/H-H/C-C对应阈值会是约1.19/0.84/1.64 Å；
这是条件推论，**尚未作为生产表使用**。

进一步静态追踪表明返回值不是最终LS振幅：

- energy在0x6c9e3b乘元素对过滤表，0x6c9e5b乘para+0x148；随后还乘一个涉及atom数、
  成键计数和.rodata常数的比例，并在0x6c9e85再除常数，才写bond_ener_list。
- length在0x6ca023乘para相关元素对长度过滤表，才存bond_len_list。
- 自定义pot_bond_input路径另有读表和tolerance覆盖。

随后原指令初始化探针已恢复默认H/C过滤、尺度与几何成键计数归一化，详见
[初始化契约](native-ls-initialization-contract.md)及
`research/ga_ssw/evidence/native-ls-initialization/result.json`。4例矩阵/独立公式
最大误差均0，C60计数90，C4H6计数9。关键修正：纯碳raw C-C3.44684在归一化
分母抵消；此前3.44684与3.61的差异不能独自说明最终软化强度不同。

初始化表B仍需乘amp_c及两个atom_filter才是最终pair振幅。后续调度和正常输入
解析后的实际状态未闭环，不能把B直接作为A加入生产。下一步核对这些调用状态，
分别保留paper方案和发行版兼容方案；不估计新表、不向用户索要已上传论文、不自动
启动C4H6。

## 本轮主线新证据对优先级的影响

主agent报告方向-only与full-frame对照得到相同的12结构组；31/60次偏置淬火失败的
瞬时刚体力占比中位约0.00545，内部力全部仍大于0.01 eV/Å。这些数字由主agent产物负责，
本审查没有独立重算。它们提醒：当前优先要检查偏置淬火稳定性，而不是把LS添加到一个
大量数值失败的内核上然后把失败归因于物理软化。

因此上面的C60 matched-cost配置作为准备方案保留，**不在本轮启动**。先完成有界
quench诊断和H/C表初始化核对；若数值基础通过，C4H6的多异构体覆盖任务比已有buckyball
的GM搜索任务更贴近LS核心主张。真实GFN2结果仍不能当作论文PBE复现。
