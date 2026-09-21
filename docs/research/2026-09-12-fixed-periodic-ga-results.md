# 固定胞 TYPE1 GA 接通与真实 EMT 检查

## 实现与几何依据

`run_periodic_ga(..., fixed_cell=True, walker_config=SSWConfig(...))`现在复用
既有三阶段控制、TYPE1交叉/变异与periodic routing，默认walker为独立
`run_ssw`。此前接口无条件读取VC pressure/stress_tol与E/F/stress结果，
因此不能直接用于固定胞。新模式用真实`QuenchResult`的energy、max_force、
converged及surface='true'资格；objective是energy别名，完整forces/stress
未请求时记None，不伪造零stress，也不额外调用PES隐藏复核费用。

所有parents必须具有同一原始cell/PBC和组分，全周期且至少两个原子。
固定输入cell方向不变。TYPE1交叉内部canonical cellpar表示的child仅按
fractional coordinates映回共同cell；因为二者Gram矩阵相同，这是刚体
坐标还原，不是strain。其他cell变化在child walker前拒绝。测试另验证
非标准cell朝向的还原保持所有MIC距离。真实两臂为立方cell，该还原分支
的真实体系端到端证据尚未单独覆盖。

默认VC模式保留。TYPE2的分子proposal会按extent+.5A创建新cell，故其
fixed_cell=True在PES前明确拒绝，不能将此处TYPE1接通称为完整TYPE2固定胞
实现。TYPE4已有固定support路径，本轮未重复实现或更改。

## 冻结实验

协议为`2026-09-12-fixed-periodic-ga-candidate-protocol.md`。输入取已有
fresh合格Cu31三结构（一个低能、两个高能MC拒绝落点）和Al31三近重复
落点，不施加额外扰动。使用EMT、seed7、每臂4000请求/60秒。

quick和offspring阶段设为0个SSW attempt，仍各有真实势初始化淬火；fine
执行1个SSW attempt。真实dimer/14Gaussian/.1内层/.01外层/Safe-total10/
200步参数沿用上一阶段。该设置检查GA生命周期接线，不能称完整长程
GA-SSW性能实验。结构matcher采用已安装pymatgen的.2/.3/5三个默认容差，
现有helper的scale=False/primitive_cell=True/attempt_supercell=True保持。
结构近似身份与不同basin的证明分开。

| 体系 | 搜索请求 | 独立复核 | 观察结构 | matcher代表 | 实际阶段 |
|---|---:|---:|---:|---:|---|
| Cu31 | 347 | 8 | 8 | 4 | 3 quick、3 offspring、1 fine |
| Al31 | 174 | 5 | 5 | 1 | 3 quick、无offspring、1 fine |

Cu31实际产生一个crossover和两个disturb_random_sparse子代，真实淬火分别
付费82/11/13请求；fine付费238请求。Al31匹配为一个代表，不满足TYPE1
至少两区且多于两parent的条件，记录no_proposal后进入fine；没有使用
虚假精确坐标matcher或扰动制造多样性，也没有改参数重跑。

两臂共521搜索+13fresh=534请求，无预算/SCF失败。13次fresh每个独立EMT
实例，全部fmax≤.01eV/A，最大能量差8.9e-15eV；每个付费请求cell/PBC和
组分都与输入一致。所有未归档观察仍保存。Cu31最佳能量较输入最低约降
1.31e-5eV，Al31无进一步降低，均不支持科学效率或新全局极小值结论。

## 验证与边界

根智能体完整standalone回归527 passed / 1 skipped，最后几何资格补充后
新定向7项再次通过。唯一skip仍为需要显式原二进制数值片段环境变量的项。
没有运行LASP主程序、GPU或HPC任务。

产物位于`research/ga_ssw/evidence/fixed-periodic-ga-e2e-20260912/`，包含
来源、源码快照、运行实际import、完整参数、全部matcher调用、lineage/
阶段/raw结果、每次搜索E/F和fresh观察。根智能体独立核对源码快照摘要、
逐请求几何、请求总数与各阶段和、资格、观察数、fresh完整力，保存
`root-audit.json`。本轮只执行两臂各一次。

当前功能完成于声明的固定胞TYPE1接口域；不同晶胞parent混合、TYPE2
固定胞分子算子、任意约束和长程GA优越性不在本轮完成范围。已有原子SSW
的平移投影还要求oracle具有全局平移对称性，周期边界本身不保证该假设。

## 后续：所有实际walker阶段均执行SSW

为闭合上述quick/offspring零attempt的证据边界，按运行前补充协议，仅将
quick_steps/generation_steps设为1，fine仍为1；相同输入、算子、matcher、
数值参数，seed7/19、每臂4000请求/60秒。此处不是长程性能评价，也未按
观测结果调整算法。四臂各执行一次：

| 体系/seed | 搜索 | fresh | quick/offspring/fine walker数 | 观察/代表 |
|---|---:|---:|---|---|
| Cu31/7 | 1494 | 12 | 3/2/1 | 12/6 |
| Cu31/19 | 2006 | 14 | 3/3/1 | 14/7 |
| Al31/7 | 698 | 8 | 3/0/1 | 8/1 |
| Al31/19 | 714 | 8 | 3/0/1 | 8/1 |

共4912搜索+42fresh=4954请求。每个实际调用的walker均有一个完整SSW
record，各阶段成本与其initial+record和一致。Cu31/7仅两个稀疏变异通过
proposal过滤，Cu31/19有一个交叉和两个稀疏变异；不能宣称每个seed都执行
了交叉。Al两seed均保持一个近似结构代表，如实no_proposal后进入fine。

全部42观察fresh力、能量差和cell/PBC/组分合格，无付费失败或预算拒绝。
低能改善仍仅为1e-5eV数量级或没有，不作GA效率/GM结论。根智能体复核
每次付费几何、全部fresh力、stage records/成本与源码快照；证据为
`fixed-periodic-ga-full-seed7-20260912/root-audit.json`与
`fixed-periodic-ga-full-seed19-20260912/root-audit.json`，均位于
`research/ga_ssw/evidence/`。这补足非零SSW三阶段执行证据，仍不是论文级
长期搜索性能或未知全局极小值验证。
