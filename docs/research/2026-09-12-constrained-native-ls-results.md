# Native-LS 接入固定基底 SSW：实现与真实表面检查

## 已完成接口

`run_constrained_ssw(..., ls=NativeLSSettings(...))` 已可调用。
`ConstrainedNativeLSRuntime` 组合已有 NativeLSRuntime：初始真实淬火后冻结，
复用原强度与cycle更新，预淬火/偏置搜索走已有活动Cartesian坐标，真实落点
移除全部软势。固定端点仍参与键、Nb和能量，优化与力证书只看活动坐标。
既有checkpoint保存native table/cycle/steps/frozen，不保存calculator。
只支持既有固定原子约束范围，不宣称任意ASE约束或LASP caller逐指令等价。

## 验证协议与结果

产物 `research/ga_ssw/evidence/constrained-native-ls-resume-20260912/`。
复用已保存的Cu/Al(111) 13原子表面/同种吸附原子输入，固定8原子、活动5原子，
PBC=(true,true,false)。源代码和fixture在子进程导入前冻结，实际import路径
在runtime-import.json。seed3、dimer、Safe-total，原config不变；显式
periodic-images，诊断D=1eV、Cu/Al长度2.9/3.0Å及native既有默认。
每材料连续2步对比1+1续跑，各4000请求/60秒共享上限；无调参或PES补跑。
第一次顶层execute因目录已存在而在PES前退出，后从冻结runner执行唯一一次
实际验证；保留这个准备阶段记录，不能算作算法失败。

| 材料 | 连续搜索请求 | 分段搜索请求 | 独立复核请求 | 最大活动力(eV/Å) |
|---|---:|---:|---:|---:|
| Cu | 475 | 80+395=475 | 6 | 0.008642以下 |
| Al | 703 | 360+343=703 | 6 | 0.009426以下 |

根智能体逐条独立比较两条轨迹的付费几何/E/F序列，Cu和Al均相同；分段本地
计数不同，不参与序列内容比较。2356次搜索+12次新EMT复核，共2368次请求。
全部保存结构的复核能量误差≤1e-8 eV、活动力≤.01eV/Å、固定坐标与晶胞
精确不变。固定原子的非零原始力不作为失败条件。

两种材料最终native steps=2，Nb=63，其中fixed-fixed36、fixed-mobile12、
mobile-mobile15；连续与续跑table和last_update一致。Cu两落点接受，Al两
落点均MC拒绝，拒绝落点仍保留且复核合格。不能仅凭Al当前结构相同判断
重放成功，本次结论来自完整付费序列。证据不证明所有后端都能逐位重放。

## 决定与限制

保留接口。完整standalone回归513 passed / 1 skipped；skip为需显式ELF的
native MC probe。真实表面结果支持活动坐标、LS状态和落点资格的接线正确，
不是低能盆地发现率或全局搜索效率的验证。诊断表不是势特定标定参数。
继续固定胞SSW原版stage/release控制，不扩展更多续跑层，也不转回VC/RC。
