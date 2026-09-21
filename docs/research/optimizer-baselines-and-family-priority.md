> 历史快照：其中“缺失/尚未接入”的判断可能已被后续实现取代；当前队列与覆盖状态以 [2026-09-12主线重评](2026-09-12-mainline-reassessment.md) 和当前代码为准。

# 优化器角色与 SSW 家族本体优先级

2026-09-11，按用户最新指令。目标是尽可能完整的独立 ASE/Python SSW 家族，
成熟优化器对照服务于本体实现，不形成独立的无限调参项目。

本轮Luna交付、主agent修订验证：`research/ga_ssw/lbfgs_baselines.py` 已有
SciPy/ASE研究适配器；共同证书与原生停止分开，精确坐标缓存、失败请求与
预算拒绝分别记账。已通过研究bridge接入完整VC生命周期，公共生产选项尚未
接入；主审在user-site与隔离环境均验证16项相关测试。完整Cu4实验与停止
范数修正见 `vc-mature-baseline-norm-contract.md`，不以小体系结果排名。
`check_lbfgs_baselines_cu_cell.py` 的Cu4/EMT零压/.005 eV/A³两组真实联合
淬火共46优化EFS+6fresh，三算法均满足共同gtol=.005；fresh梯度差均为0。
每压力Safe-total6、SciPy6、ASE11次优化调用。输入较简单，且原生停止和
步长限制不同，此结果仅支持适配/力应力映射，不作优化器优劣排名。
路径 `evidence/lbfgs-baselines-cu-cell/`。后续主审改进了缓存命中与预算拒绝
的日志分类，未修改求解路径；上述真实检查执行于日志分类修订之前。

## 已有优化器的准确状态

| 方法 | 当前范围 | 执行角色 |
| --- | --- | --- |
| SciPy L-BFGS-B | 旧PAM已有；联合q研究适配完成，公共入口未接入 | 成熟基线 |
| ASE LBFGSLineSearch | 联合q研究适配完成，公共入口未接入；q维数需为3的倍数 | 成熟基线 |
| ASE普通LBFGS | 旧PAM、独立固定胞SSW已有 | 已有对照，不能冒称line-search版 |
| ASE FIRE/FIRE2 | 旧PAM LocalRelaxer已有 | 库存，不扩展本轮矩阵 |
| Safe-total | 旧PAM及独立固定胞/广义坐标/VC/RC已有 | 主开发版本，收益待比较 |
| 原版LBFGS/MCSRCH/MCSTEP | ELF隔离原指令工具已有；不是独立Python生产后端 | 反编译行为对照 |
| bias-separated | 旧代码保留 | 停止推进，不参与新开发 |

`history500`仅改变可保留的secant pair数。Fe7C3历史对照两臂maxiter均300，
history500在四冻结目标中1/4成功，成功点233步；另三点300步未收敛。
对应原子/cell梯度分别为all7(.10106,.05178)、all101(.08255,.02991)、
filter7(.07265,.02204)、filter101(.0009006,.0004195)。不是只有stress未过。
CuO两个冻结目标history500收敛，但完整受预算限制搜索仍没有落点。

## 基线契约

同一目标、相同初态、相同广义坐标与精确梯度、相同物理约束和最终证书。
保留各成熟优化器原生线搜索；不通过截断返回坐标或伪造导数使其看起来相同。
SciPy无同义的block max_step限制时显式披露；history/maxiter/request cap分开。
SciPy能量相对变化ftol及分量projected-gtol和ASE范数停止均不自动等同共同证书。
记录native termination与certificate两列，成功以共同目标标准为准。
失败/被拒trial/最后accepted状态/fresh检查分别记账，不把最后trial当最终落点。

原始文档：
- SciPy: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html
- ASE: https://docs.ase-lib.org/_modules/ase/optimize/lbfgs.html

## 本体交付顺序

先闭合SSW/VC的方向、偏置、续接、淬火、落点/选择所依赖的真实缺口；
成熟线搜索adapter是必要验证依赖。随后补LS、RC/RCVC、GA本体缺失路径。
仅声称实现覆盖明确的定义域；闭环刚体等额外功能无论文/原版需求证据时不
自动升为复现必需。checkpoint/CLI整合属于后续工程交付，当前不能挤占算法本体。

每项保留四种状态：已独立实现、源行为未闭合、代码缺失、真实体系未验证。
原版的已知不一致不照抄成默认。不能通过全部探测函数存在宣称完整复现，
也不能因某源码入口拒绝一种输入而忽视已经存在的独立专用入口。
