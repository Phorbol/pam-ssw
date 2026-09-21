# 固定胞方向生命周期：论文契约与原版重入条件纠错

## 本轮问题与决定

这是实现检查/故障诊断，不是效果实验。目标是确定每次 Gaussian 前方向使用什么
势能面、哪个参考方向，以及现有原版证据是否足以要求改变 Python 状态机。

保留当前论文参考路径：一个 outer escape 采样一次随机 anchor，每次 Gaussian
在当前几何重新求方向；方向势能面不含累计 Gaussian，爬坡淬火包含累计 Gaussian。
LS 开启时方向势能面包含冻结的 LS 项。当前没有证据支持把前次求得的方向直接
替换成下一次的论文 anchor，或跨 Gaussian 无条件保留 Broyden 历史。

本轮没有改变生产算法、公共 API 或默认参数；新增了方向契约回归检查，并纠正
旧反编译报告中一个写反的条件。原版完整 CBD 生命周期仍未验收通过。

## 依据及证据边界

原始 SSW：Shang, Zhang, Liu (2013), *Stochastic Surface Walking Method for
Structure Prediction and Pathway Searching*, DOI 10.1021/ct301010b，Methods
Eq. 5–7 与 Overall Algorithm steps 1–6。已上传资料的本地全文位于外部研究目录
`literature/74.txt`：Eq. 5 为 `V_R1 = V_real + V_N`，Eq. 6 是方向约束；Eq. 7
才是累计 Gaussian 爬坡势。文中明确每个方向从初始随机 N0 更新。
这支持现有论文路径，不能据此断言所有后续 LASP 版本具有完全相同的内部状态。

当前 `paper_reference.py` 的方向 closure 加入 `soft_terms`，不加入 Gaussian
`terms`；每 Gaussian 调用独立求解器。普通 dimer、Ritz、Broyden 和实验性两阶段
方向求解器的数值停止规则并不因此成为逐指令 LASP 复现。

## 新发现：旧重入条件写反

针对上传 ELF（SHA256
`bba43b391619ae03cf7e9c455d8fe3a7c7bb7d434a56c7ef297bee38aa9b6704`）：

- `climb` 在 `0x5ccbc0` 将 `r8d=3` 传入 `for_cpstr`。
- `for_cpstr` 不相等分支的跳转表 `0x4ba5b90` 第 3 项指向
  `0x49a8475`（返回 1）；相等返回表 `0x4ba5bc0` 第 3 项为 0。
  所以 operator 3 表示“不等于”，不能只凭函数名和后续 `je` 猜条件。
- `0x5ccbcd` 在返回 0 时跳到 `0x5ccbf8`；返回非零才进入
  `0x5ccbcf`，随后调用 `set_status(CBD)`。
- 正确的局部条件是 `run_type == 5` 分支中，mode update 后
  **status != Allopt → CBD 重入**。Allopt 恰好走另一分支。
- `0x5cb365` 和 `0x5cb928` 传入的六字节常量 `0x4a45ed4` 已直接读取为
  `Allopt`；这两处不能当作 CBD 重入证据。

已修正三份依赖报告并保留纠错说明：
[字段追踪](native-rotation-bias-field-trace.md)、
[调用链](native-cbd-reentry-source-audit-20260912.md)、
[频率边界](native-cbd-reentry-frequency-audit.md)。旧报告相互引用不构成独立证据。

原始汇编来自外部研究目录 `analysis/kernel-ssw_fixlat_mp_climb_.asm`；
二进制比较表使用 `objdump -s --start-address=0x4ba5b90 --stop-address=0x4ba5bc8` 核查。
新探针 `research/ga_ssw/probe_native_cbd_reentry_guard.py` 仅执行 post-update 条件片段，
在分支目标处停止，使用原始字符串运行库；不执行 mode update、CBD 回调、主程序或 PES。
它不能证明 run_type=5 的运行时覆盖率，也不能证明每次 Gaussian 都进入此片段。

## 回归检查

`tests/standalone/test_paper_reference.py` 新增两 Gaussian 流程测试，核对：
同一 escape 的 anchor 不被上次 refined direction 替换；方向求值的能量和力均
无旧 Gaussian；淬火项数为 1、2 且保留第一项。另断言旧 Gaussian 在第二次方向
求值点具有非零能量/力，避免测试因偏置为零而空过。

使用现有 mace_env、PYTHONNOUSERSITE=1、PYTHONPATH=. 与单线程 BLAS，执行
`python -m pytest tests/standalone/test_paper_reference.py -q`：**15 passed**。
其中 mock/解析势仅验证流程契约，不支持材料搜索性能或物理有效性的结论。

## 下一项有界验收

从已经纠正的“阶段结束且未进入 Allopt”分支继续核对 Gaussian 计数与 CBD
reset/copy，区分“同一旋转迭代内历史”与“跨 Gaussian 历史”。只有发现改变
proposal 的确定遗漏才改实现；不凭静态字段名引入持久状态架构。之后转入
既定 LS Ti/O 初始化调用链，不重复开展碳异常或优化器调参。

## 原指令核验结果

运行 `PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python
research/ga_ssw/probe_native_cbd_reentry_guard.py --output
research/ga_ssw/evidence/native-cbd-reentry-guard-20260917/result.json`
（解释器为 `/home/gengjianrui/.conda/envs/mace_env/bin/python`，其余环境同上）：
**4/4 passed**。Allopt 到达 `0x5ccbf8`；climb、CBD、Allopt_softPES 到达
`0x5ccbcf`。每例恰好执行一次原始 `for_cpstr`。后二者是区分字符串条件的
诊断输入，不声称它们都会在真实运行中到达此调用点。

开始时原临时 Unicorn 目录已失效，前两次调用均在导入阶段报
`ModuleNotFoundError: unicorn`，没有执行目标片段。重建临时
`/tmp/pam-ssw-unicorn-probe`（unicorn 2.1.4，无生产依赖变更）后核验通过。
原始结果和探针保留；零新 GPU/材料搜索任务。
