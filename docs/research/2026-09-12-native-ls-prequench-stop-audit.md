# LS 预淬火：输入默认值与实际退出条件

2026-09-12。证据来自上传的 LASP ELF（摘要见 result.json），并与本仓库
`2026-09-12-ssw-ls-primary-parameter-audit.md` 的论文/SI 输入核对。
仅执行隔离的退出条件指令，不运行 LASP 主程序、保护代码、优化器或 PES。

## 可复现证据

脚本：`research/ga_ssw/probe_native_ls_prequench_stop.py`。
产物：`research/ga_ssw/evidence/native-ls-prequench-stop-20260912/`，包括 JSON
与三个原始 objdump 片段。运行环境为 mace_env、PYTHONNOUSERSITE=1、
PYTHONPATH=.:/tmp/pam-ssw-unicorn-probe，脚本 --elf 指向上传的 lasp。

`readsswpara_` 的调用参数显示：

| 输入 | 缺省常量 | 输出字段 | 调用地址 |
|---|---:|---|---|
| SSW.LSoptsoftmax | 50 | 参数基址+0x2dad0 | 0x68694a |
| SSW.ftol | 0.1 | 参数基址+0x2db28 | 0x687d4e |

这些是解析器的缺省实参，不是任意实际运行的最终配置；用户输入和后续覆盖仍可能改变它们。
`SSW.MaxOptstep` 与独立的 `SSW.LSoptsoftmax` 不能混称。

从 0x5bf699 执行到两条出口，9 个有限标量边界组合全部匹配：

```
exit = force_measure < ftol or counter >= LSoptsoftmax
```

力测度的上游归约见既有 native-ls-cycle-state 审查：最大 Cartesian 分量绝对值，
不是每原子向量范数。严格小于，而非小于等于。这里验证的是计数器比较，
未由片段证明计数器与每次力调用或 accepted optimizer step 的一一对应。
到达正常退出位置后存储每原子真实能量变化（乘1000）；原始指令保存在 stop.objdump。

## 对当前 Python 的影响

当前 constrained/reduced 入口复用外层 fmax 与 relax_steps，且只有预淬火
optimizer.converged 才继续。最近 Cu111 开发实验为 0.03、300；这不是上述
原版参数分层/停止逻辑的逐项复现。论文 SI 对不同体系又显式使用 ftol 0.01、
0.02、0.05，不能把解析器缺省 0.1 声称为所有体系的推荐精度。

下一项最小实现应分离软势预处理与最终真实势资格检查的参数，并明确区分
force convergence、budget exhaustion 和 line-search failure。当前证据只支持
原版的正常力/预算出口，不支持把数值失败一概改为成功或继续。
必须保持旧配置可复现，随后做跨体系受控比较，不能靠放宽 Cu111 的阈值宣布修复。

本次未修改算法默认值、未加入线搜索回退，也不构成端到端科学验证。

## 已实施的最小解耦

新增公开 `LSPrequenchSettings(fmax, steps)`，作为 `LSSettings` 和
`NativeLSSettings` 的可选 `prequench`。固定胞普通/native/constrained入口已接通；
未设置保持旧继承行为，显式设置不放宽真实起点和最终力证书。预淬火仍要求收敛，
不把原版 budget 出口提前嫁接为成功。旧checkpoint缺字段视作None，变化配置
在新oracle调用前拒绝；VC/joint入口暂不消费新字段。

根智能体独立执行完整 `tests/standalone`：581 passed、1 skipped（31.38秒），
日志 `/tmp/pam-root-ls-prequench-suite-20260912.log`。新增17项包括真实Cu2 EMT
预处理、普通/native续跑和受约束入口；它们证明接口/状态契约，不证明搜索收益。
后续受控跨体系对照只改变软面fmax=.1，保持旧steps=300与外层.03，以避免
同时改变精度与预算后无法解释差异。

VC接口后续新增显式拒绝非None prequench，防止未实现配置被静默忽略；
新增拒绝测试与LS/VC相关回归34 passed（2.88秒），日志
`/tmp/pam-root-ls-prequench-vc-boundary-20260912.log`。
四臂实际对照已完成，见`ls-prequench-decoupled-multicase.md`。
