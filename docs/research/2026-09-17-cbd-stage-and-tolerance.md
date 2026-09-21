# CBD阶段组合与旋转停止精度：不能直接比较同名tol

## 目标与当前结果

固定晶胞SSW恢复继续沿共享ASE驱动推进。本轮新增独立Python的
`pamssw/standalone/recovered_cbd.py`：组合已恢复的Broyden更新、预旋转、
有偏/无偏阶段切换、端点力复用和外部硬预算。未接入默认SSW入口；
完整native模式生成与约束投影未完成。不是调用LASP二进制的接口。

## 闭合的证据

- 原始 `soften_mode0` 力快照复制6例通过：保存当前端点力到work1，中心力tf0独立。
- 原始post-CBD调用者8例通过：严格曲率边界、work1恢复、n0保存为anchor、
  weight和rotstep写入；强制不同buffer并断言复制内容，非只检查到达终点。
- 原始 `rotate_dimer` 到BRIONS入口12例通过，N=1/2/5，两种dr、两种FACT1：
  Broyden输入坐标为`r0+dr*n`，输入响应为
  `FACT1*(f_endpoint-f_center+dr*curvature*n)`，最大响应误差1.39e-17。
  只hook等价memcpy；显式输入端点力，无PES，无完整旋转求解声明。
- 新求解器4个数值/生命周期测试通过；检查同端点阶段复用、不篡改初态、
  负曲率预旋转override受总预算约束、原版旋转上限不算力收敛，以及返回方向已求值。
  独立审查核对了PreRot负曲率转换不额外重置history，另两种阶段转换需要重置。

原指令产物位于 `research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/`：
`force-snapshot.json`、`rotation-caller.json`、`rotation-response.json`。

## 真实结构的有界接口检查

Slurm **1361832**，1张V100，5分钟硬上限；实际COMPLETED，48秒作业时间。
模型为已有MACE-OMAT-0-small，float64，冻结模型/源码/输入与配置见
`research/ga_ssw/evidence/recovered-cbd-material-interface-20260917/plan.json`。
两种周期结构均固定cell、无约束；seed11，显式输入去净平移随机方向。
每例最多14次E/F请求；这不是完整Gaussian逃逸、盆地搜索或生产性能验证。

| 案例 | 实际E/F请求 | 停止原因 | HVP残差(eV/Å²) | 旋转报告力(eV/Å) |
|---|---:|---|---:|---:|
| AlOH26 | 8 | 旋转力阈值 | 1.1633065 | 0.0116331 |
| brookite48 | 6 | 旋转力阈值 | 1.8506112 | 0.0185061 |

两例计数均与逐请求记录一致，返回方向均对应实际已求值端点；
PreRot都在第3次旋转触发严格`rotnum>2`，随后同端点进入biasedRot，没有多记PES。
用的是显式Euclidean history，而不是退化的native block-sum算术。
参数是有界诊断设置，不宣称它们是LASP默认或推荐生产配置。

## 对研究主线有影响的结论

令 `r=H_b n-(n^T H_b n)n`，已执行的原始响应为`-FACT1*dr*r`。
结合原始终止代码 `F_report=10*||f2_workspace||/FACT1`，在上述
固定胞、无约束、同一旋转面条件下：

    F_report = 10*dr*||r||
    equivalent_HVP_tolerance = ftol_native/(10*dr)

所以dr不仅控制有限差分误差；若原版ftol保持不变，dr还改变等效方向残差精度。
本诊断的dr=0.001Å、ftol=0.02eV/Å对应HVP残差阈值2eV/Å²。
直接HVP阈值0.02eV/Å²严格100倍。两例均通过原版形式的旋转力判据，
但都不能通过直接HVP残差0.02的判据。此审计复用同一批力，没有追加PES。

这不是新的全局优化增益结论，也不表明原版停止策略错误：短方向软化可能
有意接受较粗解。它意味着**同名同数值tol不构成公平的旋转精度对照**。
此外两例有偏曲率为负而真实旋转面曲率为正；不能把秩一偏置制造的负曲率
当成真实PES已找到负曲率方向或已经逃出盆地。

## 更新后的对照与验收

1. 保留原版语义的独立停止量与单位，不能静默改写成HVP残差。
2. 数值求解器比较要换算到同一HVP残差或使用同预算性能曲线，明确force-stop
   与step-stop；不同停止精度的结果不能用于宣称Broyden比Ritz更快。
3. 方向规则效果仍须在同后端、同初态、同预算的完整SSW/LS-SSW搜索中检验；
   本次2例接口检查不能代替该验收。约束/局部模式未完成前不升级默认策略。
4. 完成自动轴/组和Q模式生成的必要契约后，接入已批准的共享驱动；
   不将可选模式未恢复包装成需要用户补文件，现有证据显示还有内置构造路径可查。

运行环境出现pynvml弃用和torch.load设置提示；未改动依赖。无数值异常。

最终针对性验证（研究工作区，mace_env Python，PYTHONNOUSERSITE=1、PYTHONPATH=.，
OMP/OpenBLAS/MKL均单线程）：

```sh
python -m pytest tests/standalone/test_recovered_cbd.py tests/standalone/test_native_rotation_control.py tests/standalone/test_native_local_group.py tests/standalone/test_public_broyden.py -q
git diff --check
```

结果：23 passed；diff空白检查通过。新模块没有导入默认驱动，未修改已有默认策略。
自动轴/组选择子任务因ABI和解释矛盾未获root验收；详见该项诊断草稿，
不计入本轮已恢复功能。相应不合格解释不会进入生产实现。
