# 主线重评：算法流程已能运行，当前缺的是可归因的搜索证据

本次只读核对当前代码、上传论文全文/SI、原指令审查和真实产物，更新研究队列；
不改算法默认，不启动新计算。ACS两篇出版页面本次返回403，使用已有完整PDF，
不是因网页失败而依赖摘要。目标仍为独立Python/ASE SSW→LS→GA，再VC/RC。

## 对照原理、原版、实现

| 环节 | 论文/原版依据 | 当前实现与尚存差异 | 对主线的影响 |
|---|---|---|---|
| 随机方向与biased dimer | SSW 2013式5–6及步骤1–2：沿初始随机anchor施加旋转偏置，逐Gaussian更新方向；不是必须无条件找到真实PES最软模 | paper_reference有dimer/Ritz/Broyden独立求解及显式两阶段选项；原版CBD_PreRot、anchor写回和curvature系数已部分恢复，但每Gaussian重入频率、持续历史非完整parity | 高：影响proposal分布；优先限定重入/anchor问题，不能无限反编译整个CBD |
| Gaussian continuation | SSW式7–8及步骤3–6：积累Gaussian、偏置面局部优化、达到H/能量条件后去偏置真实淬火 | 主流程已有；87°高度和PAM曲率高度均有选项，不代表任何一个已证明更优。普通入口已有bias_fmax、bias_stage_steps和释放adapter | 不能再把“内层未收敛不能真实淬火”笼统称公共缺失；配置默认与原版状态机等价分开 |
| LS势与软面预处理 | LS 2024 §2.3–2.4：建立邻居/penalty，先重新优化，再SSW，并以真实能量响应调强度 | 指数pair势、预淬火、真实能量响应、更新、去LS后的真实淬火已有；LSPrequenchSettings已分离软面与最终证书 | Cu预淬火不再是主阻塞；尚缺显式预算退出后response/继续语义（当前严格收敛） |
| 周期LS距离 | helper镜像候选替换条件 candidate² < best²−.001，有限27镜像且有直接距离快捷返回 | 当前native-mic标签实际使用ASE MIC；periodic-images为独立求和扩展 | 是明确复现差异。先用独立命名/规则记录；不得把原版容差当光滑化或默认修复 |
| LS跨步更新 | 论文式15/目标Y；原版normal_update、save_zero/restore及参数已恢复 | NativeLSCycleState已有算术/状态实现；调用者按完成attempt计数，失败climb/MC拒绝和response关联仍是独立约定 | 高于继续调线搜索：需核对跨步物理含义、长轨迹强度响应。默认ratio1.1使110>=cycle100，不启用周期零化；前100步可每步更新 |
| 局部优化器 | 原版LBFGS→MCSRCH→MCSTEP，reverse communication；派发次数不是accepted iterations | Safe-total与ASE/SciPy基线已有，原指令工具不是独立Python生产后端；Safe-total失败回退尚未补 | 比较服务于SSW，不另立history500优化项目；固定目标、坐标、终止与成本后才可排名 |
| 真实落点/MC | 论文步骤6–7：真实势淬火并按真实能量选择 | 已有、失败保留、checkpoint、结构matcher接口已有；不同入口约束范围不等同 | 力合格不代表不同basin/稳定化学结构；这是当前科学结论的关键缺口 |
| GA外层 | population proposal + short/fine SSW + archive/selection | 固定胞多个类型已有流程及预算/续跑；GA任意约束索引语义、全部TYPE覆盖不完整 | 先验证内核，然后同预算检验GA增量，不重写Java替代或加新算子 |

重要的理论边界：LS添加P_LS改变势能、极值点与通道，并不严格等价于只对原势
使用预条件矩阵；SSW人工climbing不是物理轨迹，普通Metropolis也不自动保证
平衡采样。当前主要目标是有效低能盆地发现，不以TS或系综性质评价这一阶段。

## 卡点重新分类

**已缓解，不再占P0：** Cu/native预淬火；软面/真实面精度耦合；是否能够Gaussian
预算停止/去偏置；checkpoint有无；1e-4是否必然太小。最后一项三个几何/两个
后端未见噪声主导，仅数值诊断，仍不是通用步长结论。

**仍有真实本体差异，但可界定范围推进：** CBD阶段重入/anchor/history；LS预算
出口response；LS更新计数/失败边界；周期镜像容差。反编译只回答这些影响状态
转移的问题，不继续为了数量恢复无关函数或自动移植压缩/重连等经验设计。

**接口覆盖缺口：** 普通run_ssw拒绝partial PBC和约束，而constrained入口已有
slab/FixAtoms/Hookean与paper/native-derived LS。因此应交付明确入口/支持矩阵，
不能说完全不支持slab，也不能称任意ASE约束或GA约束已统一。原版固定原子
邻居计数是否等同本项目约束投影仍未证明。

**当前首要科学卡点：** 有接口和可合格落点，但尚无稳健的跨体系、跨seed、同
总成本优势证据。已有ASE BH等预算对照未显示SSW明显best-energy优势；不能
说“从未比较”，也不能因为初态接近稳定结构、分子SCF中断就宣称算法劣于BH。
现有多数1–2步LS结果不足以评价自适应强度的长时间响应和有效盆地覆盖。

## 新的唯一执行队列

1. **冻结一个可解释的固定胞参考流程和资格指标。** SSW与加入LS的对应版本
   共用方向、Gaussian、优化器、外层证书；明确选择paper还是native-derived
   LS规则，不把两者默默混合。已有接口足够时不再加组件。保存完整配置，
   只补会影响候选/状态的关键缺口；原版budget退出若实现须作为明确独立策略，
   不把line_search_failed或未评估trial当收敛/有效响应。
2. **推进多步真实搜索，不再把主预算花在局部单点排障。** 首组用既有C4H6
   GFN2异构体和非平凡金属团簇/固定胞缺陷EMT，覆盖不同PES困难；Hookean水/
   Cu表面只保留回归角色。现成Cu13理想基态本身不适合承担“找低能”主要证据。
   C60保留为困难复核，但先证明选定初态/后端可稳定求值；已有GFN2 SCF失败
   使它不能作为必须先通过的门槛。MACE-OMAT可用于后续适用域合格的周期材料，
   不能默认用于任意分子/碳团簇。具体输入、种子、目标、预算在执行前冻结。
   轨迹长度要能观察LS强度响应/正常更新区间；不为触发默认关闭的周期分支调ratio。
3. **按上述轨迹的实际成本瓶颈比较优化器。** 同一LS/偏置目标、相同初态与
   资格标准，比较Safe-total、ASE LBFGSLineSearch、SciPy L-BFGS-B；需要时
   用隔离LASP原指令作行为参照。分别记录原生停止、共同证书、失败trial与总
   oracle。先比较既有版本，不同时改history、回退、FD和Gaussian。
   只有平滑目标上重复出现相同失效机制才实现对应恢复。
4. **内核结果确定后检验GA增量。** 同初始池、同backend、同总请求预算比较
   GA-SSW与SSW多起点，报告实际有效不同低能结构；不要以archive大小或
   crossover成功率代替搜索质量。必要ASE约束支持跟随明确案例实现。
5. **之后VC/RC。** 现有变胞/刚体代码保留，禁止回到Fe7C3调参挤占固定胞主线。

每组预先规定：目标异构体/低能结构发现成本（有已知目标时），合格且结构不同
的盆地覆盖，best energy对累计请求，失败/解离/MC拒绝成本与seed间离散程度。
没有已知GM时不伪造成功率；同图不同构象、同能不同结构须由明确matcher区分。
预算耗尽样本保留；优化器原生能量停止不等同强制失败，但不放宽最终证书。

## 本次证据入口

- `literature/74.pdf`，SSW，10.1021/ct301010b，式5–9、算法步骤1–7。
- `literature/215.pdf`、`ct4c01081_si_001.pdf`，LS-SSW，10.1021/acs.jctc.4c01081，§2.3–2.4、式15、输入示例。
- `native-rotation-bias-field-trace.md`、`native-ls-cycle-state.md`。
- `2026-09-12-native-ls-prequench-stop-audit.md`、`2026-09-12-native-ls-image-semantics-deep-audit.md`。
- `2026-09-12-fixed-lbfgs-rc-lifecycle.md`、`2026-09-12-stage-control-e2e-results.md`。
- `2026-09-12-fixed-cell-progress.md`、`2026-09-12-ase-bh-baseline-results.md`、`ls-prequench-decoupled-multicase.md`。

文献目录绝对根：`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/`。
旧报告中的缺失判断只作当时快照，须以当前函数和本队列为准。
