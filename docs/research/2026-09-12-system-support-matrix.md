# 分子、slab、bulk：当前支持域

2026-09-12按公共入口代码复核。这里的支持是已存在可调用流程，不是所有
算法选项相同，也不是已证明所有后端和体系上的科学效果。

| 体系 | SSW及LS入口 | GA入口 | 必须保持的条件 |
|---|---|---|---|
| 非周期分子/团簇 | `run_ssw`、`run_ls_ssw`、`run_native_ls_ssw` | `run_ga_ssw` TYPE0原子或TYPE3显式单体分组 | 普通入口all-mobile、PBC=False；有FixAtoms/Hookean时另走约束入口。TYPE3不是任意分子拓扑生成器 |
| 固定基底slab/吸附 | `run_constrained_ssw(..., ls=None/LSSettings/NativeLSSettings)` | `run_surface_ga` TYPE4 | full/partial/no PBC均可用于约束SSW，至少一个活动原子；支持FixAtoms/显式fixed_indices及Hookean（pair/point/plane）。TYPE4还要求自身二维surface topology、显式support/adsorbates、routing与matcher |
| 固定胞bulk | `run_ssw`及两种LS入口 | `run_periodic_ga(..., fixed_cell=True)` TYPE1 | 全周期、all-mobile、满秩cell；`translation_only`及global/isotropic方向；GA各parent同一cell和组分 |

## 不应省略的限制

1. **全活动、部分PBC的自由slab已可使用约束入口的全Cartesian chart。**
   不再要求人为固定一个原子；普通run_ssw仍拒绝partial PBC，未做隐式分派。
   Hookean-only分子已有GFN2短流程，FixAtoms+Hookean slab已有EMT短流程；
   自由slab全活动搜索效率尚无专门证据。GA不继承任意Hookean支持。
2. 固定基底只要求活动原子力合格。固定原子可有非零反力；它不能被解释为
   完全无约束的极小值。固定胞bulk/slab同样不要求被禁止松弛的cell零应力。
3. 约束入口与普通入口尚非全部选项对等：前者使用活动坐标Safe-total，
   已显式支持`gaussian_policy=PAMCurvatureGaussian()`，有分子/slab短流程
   及默认ledger不变证据；仍没有`bias_quench_adapter`或通用
   `quench_optimizer`参数，不能因此声称全部选项对等。
4. TYPE2分子周期proposal会创建新cell，固定胞请求明确拒绝；已有TYPE4
   固定support GA不能被当作任意表面重构、任意约束的GA。
5. ASE calculator须满足所选流程的能量/力契约；固定胞流程不强制应力
   接口，真正VC另需应力。平移投影需oracle具有全局平移对称性，不能仅从
   PBC推出。LS还需有来源的pair参数，MLIP适用域需独立判断。

## 对接下来验证的影响

分子、固定基底slab、固定胞bulk均纳入跨体系验证；三个入口用各自正确的
物理资格和成本统计，不把API差异藏在一个笼统的“支持ASE”标签下。
先按共同已实现的SSW/LS机制做比较，PAM/优化器消融仅在其实际支持域进行。
约束入口选项对齐仍是明确缺口。Hookean支持及真实六臂结果见
[约束实现记录](2026-09-12-hookean-constraint-results.md)，不能宣称所有ASE constraint均受支持。

代码依据：`paper_reference.py`的run_ssw入口PBC/constraints检查；
`constrained_reference.py`的ReducedCartesianChart、ConstrainedSSWConfig和
run_constrained_ssw；`surface_ga_reference.py`的run_surface_ga；
`periodic_ga_reference.py`的fixed_cell模式。

PAM约束接口与对照见[结果记录](2026-09-12-constrained-pam-gaussian-results.md)。
