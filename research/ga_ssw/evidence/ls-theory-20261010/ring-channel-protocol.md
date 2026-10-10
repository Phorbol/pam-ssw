# T2b：同一gauche起点的闭环通道与容易回返通道

问题：T2a中LS降低了gauche→trans扭转势垒的一阶项，但SSW本来就容易找到该通道。
真正影响后续设计的是：同一起点、同一个冻结W，是否更有利于成断键通道？
本次只核查一个gauche-butadiene→cyclobutene通道，不扩大搜索/调参，也不要求它
是唯一或最低MEP。失败只说明该候选诊断未完成，不据此否定通道或模型。

## 固定输入和原子身份

- gauche：channel-v1/run-1728173/plus-0.050.extxyz，MH1力/双模板正H已核查。
- cyclobutene：mechanism-v1/mh1-1728065/case-0/minimum.extxyz，同MH1/omol。
- 构HC长度表+0.1 Å图，cyclobutene删除两个CH2碳之间的闭环C-C边后与gauche同构。
  枚举保元素/C-H邻接映射，以proper Kabsch对齐最小RMSD选择并冻结；不在NEB中重排H。
  只读核查8个图映射，最优之一B→A=(1,2,3,0,6,4,8,7,9,5)，RMSD0.89568 Å；
  正式运行保存实际选中的映射与所有选择范围，不强行要求近简并映射的舍入次序。

## 执行与门槛

借鉴已有c60-path-diagnostic-20260927/run_neb.py的ASE诊断方式：7images，
improvedtangent，k=0.1 eV/Å²，linear+IDPP初始化，FIRE标准默认；
普通NEB50步、CI-NEB150步，路径fmax0.05 eV/Å。普通阶段可满步后进入CI，
CI不收敛则停止，不自动换映射/加images/加预算。记录每一步路径、物理末端E/F和阶段成本。
NEB只提供候选，最高image不是已验证TS。

CI收敛后将最高内image交给已执行的qualify_ls_torsion_channel.run：
同一MH1上root驻点、完整力门、双模板单负模、两侧各两步长淬火及端点正H。
只有两侧落点实际分别对应预定gauche/cyclobutene，才比较目标通道；若落到别的异构体，
保留为未达到本次验收，不把发现别的通道包装为目标通道成功。
局部下坡淬火证据不同于严格IRC、动力学或DFT精度；本轮不混称。

## 预先规定的比较

所有目标比较都从**同一个原始gauche端点**冻结HC LS（3%、xi=.2、长度+.1），
不把两次独立精修后重新冻结的不同W当作完全相同的加载。
以索引保持的两个TS计算b_easy=W(s_torsion)-W(m_gauche)、
b_ring=W(s_ring)-W(m_gauche)；同时报告原势垒、起点形变柔顺性chi和-b/sqrt(chi)。
共同起点的势垒差导数为b_ring-b_easy=W(s_ring)-W(s_torsion)。
负值表示弱加载下困难闭环相对容易回返的势垒差缩小；正值表示差扩大；
这还不等于SSW的方向命中率/转移概率改变，更不等于有限强度搜索收益。
结果接近残差或模型数值分辨率则保留未分辨，不以微小符号作选择。

## 预算与停止

NEB最多1500物理E/F请求、600秒；复用资格阶段独立最多1500请求、600秒，
相加绝不超过3000请求。分段保守上限而非共享动态配额：前段节省不自动扩大后段。
单张V100、Slurm总20分钟（含两次模型载入），任一阶段超限/失败即记录并停止。
实际calculate、IDPP非物理成本、载入和失败分别归档；不占用长轨迹生产额度。
CLI与实现仅研究脚本，不改核心API/默认算法。根据结果决定是否值得有限a分支核查，
不会因为成功获得TS就自动扩为反应网络工程。
