# 周期 TiO2 的 NativeLS 完整路径验收

目标是补齐周期材料上的运行链证据，不进行短预算方法排名。输入为文献 SI（DOI 10.1039/c7sc01459g）的 rutile/anatase 结构在既有固定胞 OMAT-small 实验中得到的合格初态；使用明确恢复的 Ti/O 参数表，不将其当作拟合最佳参数。

几何预检 1415311 对原始结构和已资格初态均得到 rutile 20、anatase 24 条非零软化邻居；native-mic 与 periodic-images 计数一致。该计数不证明任意晶胞上的两种模式等价。

GPU 作业 1415364 完成，93秒，MACE-OMAT-0-small、float64、固定胞；每个输入 seed11、3外步。实际 1633+1780=3413 搜索 E/F，独立复核 4+4=8，总计3421。初始化、失败或终止开销均保留，不以外步数替代成本。

- 两例均有3个返回落点；6次软面预淬火均达到既定 .1 eV/Å 力阈值，6次真实响应更新发生。
- 8个保存帧的独立能量相符，原子力最大范数 <=.03 eV/Å，cell/PBC保持。
- CPU分析1415592核对初始+逐步成本与总请求一致，6次LS响应的 eV/atom→meV/atom 换算一致。
- 周期几何诊断中，rutile保存帧最短Ti–O为1.796–1.933 Å，anatase为1.788–1.962 Å；所有帧LS邻居数非零。周期位移已使用ASE `find_mic`，没有以分数坐标直接四舍五入替代非正交晶胞的MIC。

证据只支持这两个输入上独立Python的NativeLS准备→SSW→真实淬火→选择→强度更新链条可执行，并通过上述落点检查。不把有序原子MIC位移、邻居数量或小力当作相身份、不同盆地、正定Hessian或全局搜索优势。未验证100步以后LS周期分支。

[协议和源码入口](../../research/ga_ssw/evidence/native-ls-tio2-lifecycle-20260920/plan.json)、[实际配置](../../research/ga_ssw/evidence/native-ls-tio2-lifecycle-20260920/execution.json)、[原始结果](../../research/ga_ssw/evidence/native-ls-tio2-lifecycle-20260920/summary.json)、[独立分析](../../research/ga_ssw/evidence/native-ls-tio2-lifecycle-20260920/analysis.json)。

8臂长比较草案未执行，因旧冻结源码接口不兼容及fresh计账安排不足被标NOT READY；不将该草案列作完成工作，也不通过长算绕过接口审查。目前更优先消除LASP非周期桥接的映像不一致，避免带着错误对照研究算法排名。

## 混合元素、部分周期和 ASE 约束的组合验收

独立分支 `research/ls-constraint-qualification` 的1416337验证Cu/Ag(111)非正交8原子slab、二维PBC、底层FixAtoms和一条Hookean约束。EMT用于此接口诊断；显式LS Cu/Cu、Cu/Ag、Ag/Ag表及弹簧参数属于诊断设置，不作为化学推荐值。两外步产生两次合格落点，均被MC拒绝；初态和两落点共3帧独立检查总目标能量、投影后力、固定坐标和cell一致。857搜索+3复核=860 E/F。

发现原事件只写`ls_update=updated`，不便逐步检查强度变化。因此仅把已有内部`last_update`深拷贝到可选`native_ls_update`字段，保持优化、选择和checkpoint格式不变。1416464的32项针对性测试通过；同协议真实复跑的857条原始请求（坐标/E/F/cell/PBC）与原记录逐项完全相同，剔除新增字段后搜索结果相同。两次响应分别为0.2496902224和0.2605243328 meV/atom，均与预淬火真实能量响应乘1000一致；邻居计数37。新增观测字段未改变本次轨迹。异常更新分支已加同字段，但本次真实运行未触发，不声称该路径已端到端验证。

[原始运行](../../../ls-constraint-qualification/research/ga_ssw/evidence/native-ls-constrained-cu-ag-skew-20260920/prepared/result.json)、[复跑及比较](../../../ls-constraint-qualification/research/ga_ssw/evidence/native-ls-constrained-cu-ag-skew-replay-20260920/prepared/RESULT.md)。主agent已独立核对ledger和结果后集成两处源码/测试文件；属于未提交研发改动，未发布。
