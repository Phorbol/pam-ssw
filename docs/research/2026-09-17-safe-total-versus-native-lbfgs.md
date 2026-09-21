# Safe-total 与实际 LASP 固定胞局部优化器

## 先区分三个层级

1. SSW的方向生成/CBD旋转决定如何构造下一段偏置；它不是局部最小化器。
2. Safe-total、ASE、SciPy或LASP LBFGS执行给定势能面上的局部淬火。
3. 外层搜索根据落点判断接受/拒绝并更新LS等状态。

因此`recovered_direction`配合`safe-lbfgs-total`意味着采用恢复方向，但局部淬火仍用Safe-total。
`native_stage_quench.py`只是阶段监视适配器，也没有实现LASP LBFGS。
独立Python搜索不依赖二进制；目前原版LBFGS机器指令仅在研究对照中执行，未移植为Python生产后端。

## 哪个原生函数，做什么

[实际调用链证据](native-local-optimizer-linesearch.md)：
`bfgs_class_mp_bfgsdriver_ → bfgs_basics_mp_lbfgs_ → bfgs_basics_mp_mcsrch_l_ → bfgs_basics_mp_mcstep_l_`。
LBFGS入口0x6e87b0；MCSRCH入口0x6ea9d0；MCSTEP入口0x6eb690。
不是根据ELF里碰巧有LBFGS符号猜测，而是沿固定胞路径确认call指令。

LBFGS用少量历史近似逆Hessian并生成下降方向；MCSRCH管理沿方向的一维搜索；
MCSTEP根据能量和方向导数做带保护的插值、更新区间。这是同一优化器的分工。

## 当前实现差异

| 项目 | Safe-total | 已恢复原生数值核心配置 |
|---|---|---|
| 方向 | 标准L-BFGS双循环 | L-BFGS |
| 历史 | 默认10；显式500仍是同算法 | 当前原生对照400，不代表所有输入 |
| secant梯度 | 当前总目标梯度差 | 提供给内核的梯度差 |
| 步长 | 每原子方向上限.2 Å；alpha从1开始减半 | 有标量STPMIN/STPMAX和区间插值；当前上限.5不能等同.5 Å |
| 接受步 | Armijo充分下降，c1=1e-4 | 充分下降加方向导数条件，MCSRCH/MCSTEP |
| 曲率历史 | sTy>sqrt(eps)||s||||y||才保留 | 不能假设存在同样的显式筛选 |
| 线搜索试探 | 最多20次回溯 | MAXFEV20，试探位置与失败条件不同 |
| 力合格 | 原始最大原子力及最终核验 | 本次由外部相同fmax核验；完整BFGSDRIVER判据另有边界 |

Safe-total既不从总梯度里扣除Gaussian，也不单独利用Gaussian解析Hessian；那是另一条
bias-separated设计，不是本次研究对象。对于本次无偏置淬火，总目标就是物理势能。

## GTOL=900 为什么不能照搬

令d0=grad(E)(x)·p<0，d(alpha)=grad(E)(x+alpha p)·p。
原生已恢复条件形似强Wolfe：充分下降并且|d(alpha)|<=-GTOL*d0。
但标准强Wolfe要求0<c1<c2<1；[SciPy标准接口](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.line_search.html)
区分Armijo参数c1和曲率参数c2，并以c2=.9为默认值。

若GTOL=900，取d(alpha)=2d0也满足该导数上界，却有
sTy=alpha*(d(alpha)-d0)=alpha*d0<0。因此仅由这个条件不能保证BFGS正曲率。
900是该ELF初始化和隔离执行配置的证据，不是对每次完整LASP运行参数的断言。

这并非只有理论担忧：[既有Fe7C3审查](fe7c3-native-lbfgs-results.md)归档了一个接受负secant后
产生非下降方向并以MCSRCH INFO=0退出的隔离内核轨迹；完整BFGSDRIVER是否恢复不能由此推断。
所以应保留Safe-total曲率筛选，不能因为原版名字带线搜索就删除现有保护。

## 值得学习什么、尚不能归因什么

长历史值得对照：在刚性/柔性尺度混合时，10对历史可能遗忘对集体慢模有用的曲率；
这是合理解释而非本次已证明机制。更长历史也增加计算/内存，并不保证每个起点更快。

带保护的插值线搜索是有明确数学目标的组件，值得与单纯减半回溯比较；不能预先宣称它
提高搜索效率，也不能将同时改变历史长度、步长规范、初始缩放后的收益归因于MCSTEP。
标准强Wolfe还可能增加每步E/F成本，必须按总请求比较。

不照抄GTOL900、将标量STPMAX当原子位移，或把完整驱动的force_factor直接塞进一致梯度接口。
本次原生对照明确提供scale=1的真实梯度，未执行BFGSDRIVER的.05力缩放/重启/外层判据。

## 当前相同五起点补测

原四优化器结果：[五起点回放](2026-09-17-recovered-ls-integration-results.md)。
追加原生核心使用同一五起点、同一MACE-OMAT-0-small float64、每臂600请求/600接受步，fmax=.03。
job1371850已完成；计划及固定研究代码在
`research/ga_ssw/evidence/native-true-quench-replay-20260917/`。
最多新增3000搜索+5复核，总搜索不超过68643，仍低于原72000预算。
这是隔离原生数值核心的比较，不是完整LASP、Python生产后端或运行速度比较（指令模拟开销不可比）。
结果：job1371850退出0，6分49秒，1 V100。五起点执行结束，4/5达到独立复核fmax<=.03，
1964次搜索+5次复核；请求序号、初态、接受坐标及资格核验无错误。
[原始对照审查](../../research/ga_ssw/evidence/native-true-quench-replay-20260917/analysis.json)。

| 起点 | Safe10请求 | Safe500请求 | 原生history400请求 |
|---|---:|---:|---:|
| 水seed11步0 | 535 | 455 | 423 |
| 水seed11步1 | 600，截断 | 423 | 600，截断 |
| 水seed29步0 | 443 | 528 | 529 |
| 水seed29步1 | 292 | 312 | 399 |
| Cu55步0 | 13 | 13 | 13 |
| 合格/总数 | 4/5 | 5/5 | 4/5 |
| 总搜索请求 | 1883 | 1731 | 1964 |

所有非截断项独立满足同一fmax；截断的原生终态fmax=.162983，不能当作合格结构。
原生有些起点较快，有些较慢；Safe500也并非逐起点最优。这个小型开发矩阵不证明全局排名。
该比较沿用原已恢复参数，未根据结果调整；完整BFGSDRIVER没有运行，且原生history400不同于Safe500。

新增机制证据：1964次LBFGS入口、3913次MCSRCH入口，仅6次MCSTEP插值调用。
各起点MCSTEP次数为1/0/3/2/0；多数步在初始试探上直接接受，不能把本次表现主要归因于插值。
本组已接受secant未发现非正sTy，因而本次原生截断不能归因于负曲率历史；
既有Fe7C3的负曲率失效证据仍限于那个已审查例子。

本次原生core的模拟耗时不是其硬件速度，不能与Python优化器墙钟直接排名。
同起点微小浮点差及路径分叉限制仍见前一阶段报告；首次力差已逐臂保存。
总搜索累计65643+1964=67607，低于原72000上限。

## 本轮决定

不整体替换Safe-total，不把GTOL900作为推荐值。保留正曲率筛选和独立力资格。
长历史保留为优先候选，但未升级默认；对方向与LS的下一完整搜索应固定预算、使用独立输入/种子，
将历史长度与局部方向因素分开。原生线搜索独立移植暂不优先：本组其插值分支使用很少，
现有结果尚未证明新增后端的收益，不能为追逐原版名称偏离完整SSW主线。
这是可撤销的优先级决定；若未来同历史/同方向的失败定位到回溯步长选择，再做单因素线搜索替换。

本轮只新增研究回放/分析脚本并更新主线文档，未改生产优化器；研究脚本语法检查通过，
原生指令调用和所有数值核验由上述真实体系执行验证。工作区未提交，未声称已发布。
