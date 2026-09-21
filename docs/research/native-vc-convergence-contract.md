# 原版 VC 停止判据与 Safe-total 精度边界

2026-09-11。原版 ELF SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。

`ssw_crystal_basic_mp_allopt_judge_converg_` 的 0x5f6950–0x5f6bbb
计算如下 convergence flag（非完整停止/释放状态机）：

```
max(abs(sfa)) < ftol
OR (control[0x18] < ftol AND control[0x28] < strtol)
```

sfa 是对象+0x4d8 的 rank-two 数组；比较取最大绝对分量，不是每原子向量
模长。其完整坐标缩放仍需结合 producer，不能把它等同本实现的六应变联合梯度。
27个包含等于阈值的组合用原指令隔离执行全部吻合，严格小于而非小于等于。
后续 maxoptstep、BFGS 状态及其他分支不在这个布尔量验证范围内。

`update_forcepara_` 的 0x5e451b–0x5e459c 确认：

```
control[0x28] = abs(trace(stored_stress)/3 + external_pressure) * 160.2176565
```

三个对角元素为对象+0x128、+0x148、+0x168；字面除数3，转换常数
`eva3togpa`。隔离执行静水、无迹对角、纯剪切、外压抵消四例均吻合。
后两类非静水应力可让这个诊断为零；它不是完整残余应力范数。这里控制
字节的准确 producer 已闭合，不声称完整原版优化会最终接受任意这种状态。
普通 ssw_move 0x5e79e2–0x5e7a1f 也有 ftol/strtol 与替代力分支，
但没有据此实现未收敛偏置点释放。

TYPE1-AlOH 的原版例子确实设置 ftol=.05/strtol=.05；二者数值相同不代表
同一单位，也不能用于推导独立实现 `gradient_tol=.05`。本实现采用完整
偏置梯度的 max(逐原子L2, 六cell分量L2)，并单独用真实势最大原子力与
完整允许应力证书检查最终落点。保持这一契约与 Safe-total。

原指令证据脚本 `research/ga_ssw/probe_native_vc_convergence.py`，
输出 `evidence/native-stress-producer-review/vc-convergence-prefix.json`；
全部是零PES的数值/接口检查，不构成真实体系算法收益。

## 有界单变量对照

job1255870 使用原 Fe7C3-80 LS 对照 job1252915 的冻结 Python 源、runner、
输入和模型，唯一配置差为 joint.gradient_tol 从.001变为接口已有默认.005。
不是新增优化策略或复刻原版压力判断；也不是将 LS 升级为推荐默认。
这是内层求解精度敏感性诊断，.005不是新证明的最优值。

保持 seeds7/101、all/filter两臂、各2步、2000EFS/480秒含fresh预留；总上限
8000EFS。最终真实 fmax=.001 eV/A、完整应力阈值.0001 eV/A³不变。
固定胞LS准备也不变。比较有效不同候选、严格证书、失败阶段和全部调用，
初始结构不得计入搜索发现。无改善则终止此轮精度试探，不增加容差扫描。
产物目录 `research/ga_ssw/evidence/fe7c3-safe-total-inner-tolerance/`。

## 已完成结果

job1255870 COMPLETED，1分58秒，单V100。离线审核器
`research/ga_ssw/audit_fe7c3_inner_tolerance.py` 核对唯一配置差、LS设置、模型、
全部请求账本、阶段成本和fresh证书。结果如下：

| 条件 | EFS总数 | 有效新候选/请求步数 | 迭代上限 | 预算耗尽 | 已进入但未付费 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 偏置gtol .001 | 5502 | 0/8 | 6 | 2 | 0 |
| 偏置gtol .005 | 5497 | 0/8 | 5 | 2 | 1 |

新组实际付费7次尝试，旧组8次；不能把未付费的一次说成完成搜索失败。
四个fresh检查均通过，但都是初始结构，不是搜索候选。新组5493search+4fresh。
all/seed7为659、all/seed101为842、filter/seed7与filter/seed101各1998次。
过滤组部分轨迹跨越更多Gaussian阶段，仍不能在同一预算内给出候选。
5次调用差不构成算法效率收益。结论只限于此组同源输入/模型和预算，不能
证明任何阈值都无效。按预定停止条件封存此轮精度试探，不追加参数扫描，
不改Safe-total或默认值。Fe7C3当前登记总成本39216EFS。
