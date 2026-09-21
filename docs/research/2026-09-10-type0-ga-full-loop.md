# TYPE0 独立 GA-SSW 完整闭环

已实现非周期、无约束原子/合金 TYPE0 的 crossover、扰动、元素交换、低配位原子重插，
并接入既有初始化→quick SSW→GA子代真淬火→short SSW→区域排序→fine SSW控制器。
通过 PaperGAConfig(proposal_type=0)、groups=None 使用，SSWConfig 的 Safe-total 设置
同时用于初始化与GA子代淬火。所有物理E/F由传入ASE calculator提供，正式运行不调用JAR/LASP。

## 明确保留和修正

- 沿用已对照的cut-and-splice、整数配额和分区mutation分配，不引入新奖励或择优公式。
- 原子组分保持；允许合金元素排序改变；交叉和mutation均记录逐原子的父代来源。
- 原JAR重插存在碰撞条件反向及超时塞原点行为。正式独立实现明确使用无碰撞接纳，
  超出显式尝试预算报告失败；这个差别不能称原JAR逐轨迹等价。
- 原版固定3.2 Å配位阈值、0.3 Å重插距离下限和扰动幅度均是兼容参数，非普适物理常数。
- 本次Cu13没有附加BLLimit过滤。数值可淬火不自动证明所有体系上的输入物理合理性。

## Cu13 / ASE EMT 端到端结果

来源：此前独立SSW严格验证文件17-dimer.json按出现顺序的前三个不同结构组，
并非随机未知体系，也不是原论文初态。预登记seeds3/17，1 quick/1 generation-short/1 fine，
1代GA，G=8，保留完整批次（每条13子代），相同Safe-total/dimer方向配置。

| 指标 | seed3 | seed17 |
|---|---:|---:|
| 搜索E/F |2480|2605|
| GA子代真淬火通过 |13/13|13/13|
| 返回结构独立真力复核 |25/25|26/26|
| 严格结构指纹组 |9|9|
| GA子代新增组（相对本条初始+quick）|3|3|
| 后续SSW失败 |1|0|

两条合并14严格指纹组；所有51个返回结构严格淬火通过，各组代表的去刚体内部Hessian
在1e-4/5e-5 Å中心差分下均正定。每条新增3组来自crossover的1组和重插的2组；
本次扰动没有产生超出初始+quick的新组。以上是该短实验的观测，不证明哪个算子普遍最优。

搜索总5085请求；新计算器复核51请求；严格淬火/内部Hessian另4131请求。
原descriptor归档分别10/12行，严格几何只得到每条9组，提醒不能用archive大小冒充basin覆盖。
排序原子对距离本身仍非单射，不等于完整结构同构识别。

## 结论和限制

TYPE0完整流程已经在真实元素势面上运行，GA子代实际带来新的、数值稳定的结构，
而不是只通过mock调度测试。没有相同预算的多起点SSW对照，不能宣称GA更快或复现论文提升倍数。
合金交换已做原JAR操作对照，合金端到端有效性尚未验证。纯元素完整mutation要求N>10，
合金要求N>5，周期/变胞与刚体链不包含在本次交付中。

证据：research/ga_ssw/evidence/atomic-cu13-ga/{plan,summary,strict-summary}.json及完整轨迹。
脚本：research/ga_ssw/run_atomic_cu13_ga.py、validate_atomic_cu13_ga.py。
