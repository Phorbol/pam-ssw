# 当前核心交付状态

目标是独立Python/ASE的SSW系列实现，优先固定胞SSW、LS、GA，然后VC/RC。
以下按当前公共代码及实际产物区分功能、原版行为证据和科学效果；不能互相替代。
运行不依赖LASP可执行文件或LASP pot。ASE calculator仍需提供对应流程需要的
一致能量/力（VC另需应力），接口开放不保证模型物理适用域。
分子/slab/bulk具体入口见`2026-09-12-system-support-matrix.md`：标准固定基底
slab已支持，但全活动partial-PBC自由slab尚缺；约束入口也未覆盖普通入口
的全部PAM Gaussian/优化器选项，不能概括为所有体系功能完全一致。

| 核心 | 当前公共入口及已接通流程 | 明确边界 |
|---|---|---|
| SSW | `run_ssw`：方向、Gaussian、偏置淬火、真实淬火、MC；Ritz/dimer/Cartesian-Broyden和PAM Gaussian选项；实验阶段停止接口 | 独立数值实现，不宣称逐指令全LASP轨迹等价；阶段停止未证明通用收益，默认关闭 |
| paper/native LS | `run_ls_ssw`/`run_native_ls_ssw`及约束入口：冻结pair、软面预淬火、SSW、真实落点、响应更新；native cycle和periodic-images显式扩展 | native-MIC与独立image模式分开；周期/约束端到端通过不等于强键全局搜索效率普适成立 |
| GA TYPE0/3 | `run_ga_ssw`：已有quick、交叉/变异、子代短SSW、归档/分区/fine/回流和边界续跑 | 非周期原子团簇或显式单体拓扑域；GA比普通SSW更优仍需同预算多体系证据 |
| 固定胞TYPE1 GA | `run_periodic_ga(fixed_cell=True)`：已有周期算子与三阶段控制接入固定胞SSW；独立结构matcher、真实E/maxforce资格 | 同一固定cell、全周期；不虚构stress。混合不同cell的parent不支持；已有两臂实际子代/fine与独立复核 |
| TYPE2/4 | `run_molecular_periodic_ga`、`run_surface_ga`已有公共流程；TYPE4固定support/活动力检查 | TYPE2算子创建新cell，固定胞请求明确拒绝，仍需独立分子proposal设计；TYPE4不是任意约束/重构基底的通用算法 |
| VC/RC | `run_vc_ssw`、`run_rc_ssw`等独立公共入口及已有研究实现保留 | 当前后置，不能称全部细节与广泛生产验证完成；不可将固定胞逃逸后变胞淬火混称联合VC |

## 本轮新增的实际证据

- native-LS周期镜像8臂：2880搜索+20fresh；短cell的MIC预淬火失败保留，
  image模式通过，不能据此对所有体系排名。
- 约束native-LS的Cu/Al表面连续与续跑：2356搜索+12fresh，固定基底/晶胞
  不变、活动力合格；Al的MC拒绝保留。
- SSW阶段停止12臂：9147搜索+36fresh，36存储minimum合格，最佳能量无
  实质优势，决定不改默认。见`2026-09-12-stage-control-e2e-results.md`。
- 固定胞TYPE1两臂：521搜索+13fresh，Cu31实际交叉/变异/fine，Al31近重复
  合并后如实无proposal。见`2026-09-12-fixed-periodic-ga-results.md`。
  进一步非零SSW全阶段四臂4912搜索+42fresh均结束，所有实际walker均有
  一个真实attempt；Cu31实际子代/fine，Al保留近重复导致的no_proposal。
- 最近完整standalone回归527 passed / 1 skipped；这检查实现，不代替科学效果。

## 尚不能宣称完成的工作

需要完成/进一步验证的是明确算法域和科学证据，而不是继续堆更多命名组件：
TYPE2固定胞分子proposal；VC/RC联合自由度的完整适用域与困难真实体系验证；
当前固定胞方案在更长、独立多种子、复杂体系的同成本低能发现/不同盆地覆盖
评估。原版尚未闭合的能量参照生命周期、特定方向/优化器控制细节按各反编译
证据文档保留，不因为原版存在就自动替换当前基线。

固定胞生产默认、实验性选项和原版数值复刻是不同层级。没有证据时，保持
已验证默认；不从单例成功推导普适优势，也不为单例失败继续添加补偿参数。
