# Fixed-cell climb-convg 参数来源审计

日期：2026-09-12。本文只审计已有 LASP ELF 的静态反汇编和 DWARF 偏移；没有启动主程序或 PES。分析对象为
`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`，SHA-256 为
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`。

## 已闭合的 `climb_convg` 公式

DWARF 将 `control+0x58` 标为 `maxe_height`，将 `control+0x60` 标为
`maxe_height_gm`。在 `kernel-ssw_fixlat_mp_climb_convg_.asm` 的
`0x5cd27f–0x5cd2d1`：

* `object+0x1ac0` 读为 `energy0`，`object+0x1ac8` 读为 `tene0`；
* 局部增量是 `delta = tene0 - energy0`；
* `control+0x58` 被更新为 `max(old_control+0x58, delta)`，并同时保存为本次
  判据使用的最大能量 excursion。

随后 `0x5cd4eb–0x5cd548` 执行严格比较：

```
e_maxlimit    < max_energy_excursion
f_maxlimit    < max_force
e_maxlimit_gm < control+0x60       # maxe_height_gm
```

三个位按 OR 组合；当 `ng == 1` 时，调用路径会屏蔽这组 limit 位。这里的
`control+0x60` 不是 `control+0x58` 的别名，不能由上述 `max` 更新式推断。

`maxe_height_gm` 的写入者在 `run_ssw_.asm` 中是：

* `0x52fa1b–0x52fa39`（普通 SSW）：从栈上的候选能量读 `xmm0`，从
  `run_ssw_$SSW_A+0xb90` 读 `xmm1`，计算
  `control+0x60 = xmm0 - xmm1`；
* `0x534f76–0x534f94`（cell/变胞路径）：同样计算
  `control+0x60 = candidate_energy - run_ssw_$CSSW_A+0xb90`。

参考量的机器级来源因此是对应 run-state 的 `SSW_A/CSSW_A+0xb90` 字段。
在 `kernel-ssw_fixlat_mp_ssw_move_.asm:0x5bd46e–0x5bd475`，该字段被写成
对象 `+0x230` 的当前能量；`make_decision` 也在
`0x5d3623–0x5d363e` 将对象 `+0x230` 写回该字段。后一次写入不是每个
candidate 无条件发生：`0x5d1d0a–0x5d1d1e` 先比较旧的对象 `+0xb90`
与对象 `+0x230`，只有旧快照严格大于当前能量时才跳到更新块
`0x5d34dc`；该块中的诊断分支最终汇合到 `0x5d362b`，随后执行写入。
因此它的可证据语义是沿该对象维护的“当前较低能量快照”（至少由严格降能
条件更新），不能安全地称为每个 candidate、初始 seed 或全历史最低值；是否
与完整 best/archive 生命周期等价仍取决于外层 caller。`+0xb90` 的高层
Fortran 成员名在现有 DWARF 摘录中没有闭合，故不作更强命名。

## `para` 字段：写入者与默认值

DWARF 基址为 `para=0x53ed7a0`，字段如下：

| 字段 | 偏移 | 静态写入证据 | 可确认的默认值 |
|---|---:|---|---|
| `ngaus_relax` | `+0x2dd20` | `readsswpara-convergence.asm:0x688aaa–0x688b61` 调 `readinput_mp_get_int_`；另有同一字段的条件分支 `0x688b18` | 未发现编译期/`ssw_options_` 写入 |
| `ngaus_relax_ini` | `+0x2dd24` | `readinput_mp_get_int_`，`0x688ba3–0x688be7`；条件分支 `0x690216–0x690282` | 未发现编译期/`ssw_options_` 写入 |
| `ngaus_relax_half` | `+0x2dd28` | `readinput_mp_get_int_`，紧随上述输入项 | 未在本审计中推断 |
| `e_maxlimit` | `+0x2dd38` | `readinput_mp_get_real_`，`0x688cdb–0x688ccf` | `ssw_options_` 的 preset 0/1/2/3 在 `0x6c446d/0x6c4535/0x6c45fe/0x6c46bd` 写入 IEEE-754 `100.0`；显式输入随后可覆盖 |
| `e_maxlimit_gm` | `+0x2dd40` | `readinput_mp_get_real_`，`0x688d4b–0x688daf` | 未发现 `ssw_options_` preset 写入；默认值未闭合 |
| `f_maxlimit` | `+0x2dd48` | `readinput_mp_get_real_`，`0x688dbb–0x688e1f` | `ssw_options_` 的 preset 0/1/2/3 在 `0x6c4474/0x6c453c/0x6c4605/0x6c46c4` 写入 `100.0`；显式输入随后可覆盖 |

`ssw_options_` 的这些写入是 preset 赋值，不等于所有输入文件的最终默认值。
`ngaus_relax*` 和 `e_maxlimit_gm` 没有在所审计 preset 中找到赋值；不能把
静态零初始化、输入提示中的字符串或某个研究 runner 的配置当作默认值。
`readinput_mp_get_int_`/`readinput_mp_get_real_` 调用中可见的 by-reference
参数是目标字段及其输入描述；在这些调用点没有可识别的默认数值立即数或独立
默认变量，因此默认值仍保持 unknown，不能从调用约定反推。

固定胞 `climb_convg` 的预算分支也已闭合：`ng == 1` 时在
`0x5cd9da` 使用 `para+0x2dd24`（`ngaus_relax_ini`），否则在
`0x5cd7bc` 使用 `para+0x2dd20`（`ngaus_relax`），比较为严格的
`climbstep > budget`。

## 对公共 stage predicate 的结论

可以安全接通的部分是严格预算选择、`maxe_height` 的
`max(previous, tene0-energy0)` 更新，以及 `e_maxlimit/f_maxlimit` 的字段读取。
不能完整接通 `maxe_height_gm`：虽然其写入公式和参考字段地址已经找到，但
参考快照字段的高层生命周期仍未完全闭合；同时 `e_maxlimit_gm`、两个
`ngaus_relax` 的最终输入默认值也不能从本静态切片确定。因此，把公共参数
直接映射成一个假定的“初始能量差”或给未闭合字段补默认值，会改变基线语义。
在完成 caller/输入配置的额外证据前，不建议把该 predicate 宣称为完整恢复。
