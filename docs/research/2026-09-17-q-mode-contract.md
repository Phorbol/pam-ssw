# Q-mode data contract (bounded native audit)

## 2026-09-21 增量证据与当前决定

下文保留9月17日静态核查的原始范围；“运行时缺文件/模式库是否初始化”已不再是当前阻塞。
两条已完成native C60轨迹实际打印S1–S5模式，分别涉及21/188、16/189个NewStart区间；
不是按模式行数计成功率。可复算脚本在[运行时统计](../../../ga-ls-composition/research/ga_ssw/evidence/native-q-runtime-census-20260921/census_counts.py)。

新的S1局部核查闭合了打印vector到setter的路径。示例`S1 : 6.48507 6 -1`
依次是cutoff、名为`neigb_atom`的字段和径向幂指数n；不把中间6解释为邻居数。
打印来自init_randomize_sympara，不是syminfo；根agent复查并纠正了辅助报告的字符串地址。
[字段与指令证据](../../../vc-qualification-audit/research/ga_ssw/evidence/q-s1-contract-20260921/README.md)。
原文[Huang et al., Chem. Sci. 2018, DOI 10.1039/C8SC03427C](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289100/)
式5–6给出径向基R^n(r)=r^n f_c(r)、S1_i=sum_j R^n(r_ij)。
它解释表示形式，未证明这种梯度用于SSW搜索的收益。

对非周期、完整求导的距离/角度不变描述符，梯度满足sum_a g_a=0及sum_a R_a×g_a=0。
固定中心原子索引和距离cutoff本身不破坏这些恒等式；只取中心导数或遗漏邻居导数才会破坏。

对单中心S1，令phi(r)=r^n f_c(r)、e_ij=(R_j-R_i)/r_ij，则
`g_j = [n*r^(n-1)*f_c(r) + r^n*f_c_prime(r)]*e_ij`，
`g_i = -sum_j g_j`。这直接给出完整原子梯度和刚体恒等式；不是只推中心原子。
不同n与cutoff改变邻居径向权重，归一化只消去整体标量，不消去相对权重。
因此它可能提供与随机初值不同的局部形变，但没有给出PES势垒大小或有用盆地的保证。

FixAtoms需要允许自由度投影；Hookean应计入能量/力，不应被当作冻结切空间。
原版6个cell量共享3N原子向量范数，不是已定义物理度量下的联合归一化。

进一步的真实日志分层（CPU1433616，零PES）确认：37个Q首Gaussian窗口无旋转标记，
后续398个Q窗口全有CBD_PreRot，其中397个有CBD_biasedRot；全部按实际moveds数字序号归属。
这支持Q提供几何方向、后续仍进入CBD的流程解释，不支持“Q普遍替代CBD”。
标记不是实际模型调用计数；[分层证据](../../../vc-qualification-audit/research/ga_ssw/evidence/q-cbd-runtime-20260921/result-stratified.json)。

**原定验证计划（2026-09-21已完成，见[结果与决定](2026-09-21-forced-q-results.md)）：** 用既有桥接在相同四个C60保存态做强制Q/关闭Q的有界分支诊断，
其他配置相同，每臂一个外步及最多2400搜索请求，共19200+16独立复核。
强制概率1只用于确保实际触发；不能由此声称默认p=.1混合策略的效率、也不要求相同随机流。
先验证实际Q日志、落点资格与成本，再决定是否移植。S1仅占运行模式的一小部分，
不以S1实现冒充完整Q；暂不扩建公共描述符框架或改变默认策略。


日期：2026-09-17。范围是归档 ELF `lasp` 中 `gen_randommode` 的 Q-mode
分支（`0x5d8fe8..0x5da5dd`）及其直接/静态可达 helper；没有运行 LASP、PES
或 GPU/HPC 作业，也没有修改生产代码。

## 已恢复的调用链

`gen_randommode` 在 `0x5d8fe8` 进入 Q-mode 准备段。它先按 MPI rank 分发/构造
整数和工作数组：`hget_random_int_array_` (`0x5d9038`)、
`hcalfact_by_n_atm_` (`0x5d9055`) 和条件 `mpi_bcast_` (`0x5d908a`)。随后调用

```text
hinit_ptsd_str_(natm, coord_like, type_like, ..., mode_state, ...)
    @ 0x70619c  (caller 0x5d9112)
hget_ptsd_mode_(natm_ptr, structure_state, scalar_out, mode_out, stress_out)
    @ 0x70670f  (caller 0x5d96b9)
```

这里的 `hinit_ptsd_str_` 是结构初始化 helper，不是模式库初始化的
`hinit_ptsd_mode_`。调用者在 `0x5d911f..0x5d9286` 清零/分配 PTSD 工作数组；在 `0x5d96b9` 之后，
`hget_ptsd_mode_` 的 `mode_out` 被继续作为局部方向 workspace 使用。其后续
数组运算和归一化属于 Q-mode 生成器本身，不是 PES 调用。

## 数据依赖与数学语义

`hinit_ptsd_str_` 将每个原子坐标的 3 个 double 转成 float，复制到全局
`hinit_ptsd_str_$LATIN` (`0x7943160`)，并把尺寸/类型信息传给 C++
`init_ptsd_str` (`0x724250`)。后者首次调用时分配 `huslib::PTSDstr`，调用
`Structure::read_info` (`0x7242b2`) 和 `cdnt2fdnt` (`0x7242cd`)；这里没有
PES、力、能量或模型文件读取。

`hget_ptsd_mode_` (`0x70670f`) 将 `mode_out` 和 6 个应力/晶胞量复制到内部
float 工作区，调用 C++ `get_ptsd_mode` (`0x716ea0`)，再把结果复制回 double
数组。`get_ptsd_mode` 的可见行为是：

1. 按 `PTSDstr` 中的原子类型/坐标描述，在 `typeptsds` 的 `BrickPTSD` 集合中
   查找匹配类型。`israndPTSD != 0` 时，入口转到另一条路径：以当前结构记录中
   的类型整数为键，在 `current_chose_sym` (`0x7944b08`) 红黑树中查找/插入
   节点，再由 `maptypeptsds` (`0x7944bf8`) 找到该类型的 `BrickPTSD` 向量。
2. 通过 `BrickPTSD` vtable 的 offset 0 调用 `cal_rc_deri`
   (`0x74d176`；vtable slot 0 地址 `0x5522d90`)，根据记录的描述符/类型与
   当前 `Structure` 计算方向相关量；offset `+0x28` 是 `set_centerid`
   (`0x75851c`)。这不是已证实的“从库中直接复制一个存储的 3N 位移向量”。
3. 对长度 `3*N` 的 float 向量计算 `sqrt(sum_i v_i^2)`，逐分量除以该范数；
   对后续 6 个量按同一范数缩放。对应平方和/开方/除法在
   `0x717474..0x717558`、`0x7175a7..0x717675` 和
   `0x717695..0x7176dd`。

因此，Q-mode 的核心输入是结构几何、原子类型、N 及已初始化的模式库/对称性
状态；该函数不读取当前 PES，也不根据能量或力选择模式。可以确认方向计算
经过 `BrickPTSD::cal_rc_deri` 和归一化；记录究竟保存哪些描述符、这些描述符
如何生成完整方向，以及随机状态如何参与选择，当前 ELF 证据尚未完全恢复，
不能把它简化成“存储向量查表”或“坐标现场随机抽样”。随机分支的具体可见动作
是：若类型键没有 `current_chose_sym` 节点，就分配 `0x28` 字节节点并插入树
(`0x717126..0x717178`)；随后按节点中的索引从该类型的 `BrickPTSD*` vector
取对象，在调用 vtable slot `+0x28` (`set_centerid`) 时传入递减后的索引
(`0x717308..0x717324`)，再调用 slot 0 的 `cal_rc_deri`
(`0x717437..0x71746b`)。因此这里的“随机”至少包含类型键/模式索引选择，
最终方向仍由 `cal_rc_deri` 根据当前 `Structure` 计算；该分支本身没有看到从
文件直接读取 3N 位移向量。

## 可移植数学边界

`BrickPTSD::cal_rc_deri` 的确定性核心可以恢复到“选定一个 PTSD 对称函数的
坐标梯度”。函数在 `0x74d822` 调用
`hchemlib::AtomDescriptor::cal_symf_deri`，随后在 `0x74d840..0x74d8ad`
按内部索引表把该调用产生的三个笛卡尔导数分量写入输出。因此对选中的描述符
`G_k(R_1,...,R_N; type, cell)`，其输出可表达为

```text
v_(a,alpha) = d G_k / d R_(a,alpha),   a=1..N, alpha=x,y,z
q = v / sqrt(sum_(a,alpha) v_(a,alpha)^2)
```

Q-mode 外层还将 6 个晶胞/应变量按同一范数缩放；这对应
`get_ptsd_mode` 的 `0x717474..0x7176dd`。`cal_rc_deri` 本身没有调用
`ReactCoord`、PES、能量或力函数；其唯一描述符求导调用是上述
`AtomDescriptor::cal_symf_deri`。因此当前证据支持“PTSD 对称函数梯度”，不支持
球谐展开、反应坐标导数或从 NN 势反传。

`cal_rc_deri` 在进入描述符求导前还执行周期胞处理：`0x74d3a4..0x74d443`
对三个胞坐标使用 `fmod`，按每轴整数网格数计算格点和余量；`0x74d547` 调用
`hcpplib::cal_volume`，随后 `0x74d58d..0x74d6a2` 用胞矢叉乘/体积构造周期
几何量。可见的数值常量包括 `1.0f`（`0x4a4e058`，传入
`cal_symf_deri` 的 cutoff/尺度参数）以及工作网格中的 `500` 和 `1000`
（`0x74d793`、`0x74d816`）；后两者是内部网格/表尺寸，不能当作 Q-mode
概率或物理超参数。

初始化随机分支拼接的字符串序列在 `0x724e89..0x72504f` 为
`TypePTSD_1`、元素名、`6.0`、随机索引字符串和 `subinfo`；`BrickPTSD` 构造器
从向量中建立 `AtomDescriptor`（`0x74c038..0x74c115`），再按描述符类别调用
对应的 `cal_symf_deri_*` 实现。ELF 同时保留 `S1..S6` 字符串和
`SymfK*`/`SymfS*` 求导符号，但没有足够证据把 `TypePTSD_1` 的 `6.0`、索引和
`subinfo` 一一映射到某一个 S/K 家族；实现时必须保留这些输入字段，不能只写
一个固定径向公式。

## 外部文件边界

字符串常量 `PTSDmode` 位于 ELF `.rodata` `0x4a4df68`，后跟用于模式库解析的
`TypePTSD_1` (`0x4a4df74`)、`6.0`/`0.0` 等格式片段。`hinit_ptsd_mode_`
自身对传入 Fortran 字符串执行 `adjustl`/`trim`，再用编译器字符串片段
`0x4a4d154` 参与 `for_concat`；但在该 ELF 的静态反汇编中没有找到对
`hinit_ptsd_mode_` 的直接 call-site；`0x5d9112` 的真实 call 目标是
`hinit_ptsd_str_`，不是 `hinit_ptsd_mode_`，因而不能把 `PTSDmode` 常量直接等同于
运行时最终路径。完整库初始化在
`hinit_ptsd_mode_` (`0x706021`) → `init_ptsd_mode`
(`0x724340`)。`init_ptsd_mode` 明确调用 `hcpplib::readfile`
(`0x72439a`)，随后执行 `MPI_Bcast` (`0x7243c9`)、`srand` (`0x7243e3`)，
按文件行建立 `ntpatom`、`pair_elements`、`maptypeptsds`，并构造
`BrickPTSD` (`0x725210`)。文件行通过 `linesplit` 和数值解析读取。可确认
`readfile` 返回的行容器随后直接进入解析循环；在 `init_ptsd_mode` 可见范围内
没有针对空容器的内置 `TypePTSD_1`、`6.0`、`0.0` 数据填充或随机方向 fallback。
空输入会留下空的类型/模式集合；打开失败的异常或错误输出由 `readfile` 实现
决定，当前未进一步反汇编。这是“无内置库 fallback”的静态证据。

此外，值比较命中 `PTSDmode` (`0x724a1f..0x724a35`) 后，代码为每个
`alltypes` 元素拼接 `TypePTSD_1`、元素名、`6.0`、`0.0` 以及 `subinfo`，
再以这些字符串向量调用 `BrickPTSD` 构造函数 (`0x725210`)。这证明存在
“现场构造描述符配置”的分支，不能把 `BrickPTSD` 解释成预训练 3N 位移库。

这是文本模式库输入，但当前静态证据没有恢复其调用者传入的最终路径，也没有
恢复记录字段到方向计算的完整格式；不能称作固定预训练模型文件。

`israndPTSD` 的 ELF 初始字节位于 `.data` `0x5522cd8`，静态字节为 `0x01`；
`init_ptsd_mode` 解析 `SSW.randomPTSD` 时又会在 `0x7249e5`/`0x7249ee` 按
`True`/`true`/`.true.` 明确写入 1/0。因此编译期初值是非零，但若实际初始化
解析到该配置键，运行值会被覆盖；归档 `allkeys.log` 没有该键，不能仅凭
`Lmode_Q=T`、`mode_Q_pro=0.1000` 判定最终运行值。

结论：Q-mode 不是纯“坐标 + 随机数”的无状态几何算法；它依赖已建立的类型到
`BrickPTSD` 对象的状态，但 `israndPTSD` 非零路径可由 `PTSDmode` 分支现场拼接
描述符参数并构造对象，不应再表述为必需的预训练 3N 位移文件。当前归档运行目录
中没有名为 `PTSDmode` 或 `TypePTSD*` 的文件；也尚未恢复 `hinit_ptsd_mode_`
调用者传入的路径和实际默认初始化时序。若初始化输入确实为空，静态代码显示不会
填充这些字符串/对象；更外层是否回到 c4/c6 或 Allopt，仍需受控调用/运行确认，
不能称为已知 native fallback。

## 最小下一步策略

当前不冻结公共适配器设计。下一项必要证据是恢复 `hinit_ptsd_mode_` 的真实
调用者/路径参数和实际 `SSW.randomPTSD` 生效值，并取得一份合法 `PTSDmode`
文本样本；随后只做 helper 级 fixture/probe，区分“随机类型/索引选择后由描述符
导出的解析方向”和“存储位移查表”，同时记录 `mode_Q_pro` 的门控。若初始化
输入确实缺失，应把 Q-mode 标为输入状态缺口；不要静默回退为未标注的随机方向。

证据入口：`research/ga_ssw/evidence/native-cluster-control-generator/gen_randommode.asm`、
外部 ELF `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`，
以及该 ELF 的 `analysis/lasp-symbols.txt`。
