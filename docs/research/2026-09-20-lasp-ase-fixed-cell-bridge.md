# LASP 本体外接 ASE：固定胞研究接口

目标是让原版 LASP 和独立 Python SSW 使用同一 Calculator，以研究算法差异。此接口不替代 Python SSW，也不是 LASP 内部函数的 Python 绑定。执行原二进制，保留其选轴、偏置、优化和 MC；仅用 `potential external` 替换 E/F 后端。

## 当前最小接口

[lasp_external_ase.py](../../research/ga_ssw/lasp_external_ase.py) 的 `run_lasp(command, *, cwd, atoms, calculator, pbc, max_requests, env)` 接收调用者构造好的 ASE Calculator。Calculator 常驻，回调复制 Atoms 模板并更新坐标/晶胞，保留 charges、magmoms、info 等模板数据。要求返回 eV 能量和 eV/Å 的 `(N,3)` 力；ASE Calculator 自身负责底层软件单位转换。

调用链：原版 LASP → external.coord → lasp.external.sh → 标准库 socket 客户端 → ASE Calculator → external.ene → 原版 LASP。客户端复用 [lasp_external_mace_client.py](../../research/ga_ssw/lasp_external_mace_client.py)，其名称是历史命名，传输本身不依赖 MACE。

调用者须准备与模板一致的 LASP input.arc/lasp.in/lasp.external.sh，并传入现有 `bounded_process.py` 包装后的 argv。完整可运行示例见[居中校准 runner](../../research/ga_ssw/evidence/lasp-external-ase-centered-20260920/runner.py)。Calculator 可换成 EMT、MACE 或具备所需属性的其他 ASE Calculator；这说明注入机制可用，不表示所有后端均经过实际验证。

## 真实验证及修正

- 1415217：Cu13 非周期与非对角周期晶胞，6 E/F。收到的坐标/晶胞对应的 ASE E/F 与 LASP 导出一致。
- 1415226：交错 Cu/Ag 元素顺序，3 E/F；顺序与 E/F 传递正常。
- 但前两项非周期 fixture 含负坐标，LASP 将各原子折回盒内，改变了原团簇几何。原始结果保留，只支撑回调协议，不支撑原输入几何保持。
- 针对该真实问题新增启动前校验：非周期方向的分数坐标须位于 `[0,1)`，否则明确拒绝并要求调用者居中后重写输入；PBC 参数须等于模板。回归 1415251 原实现 2 failed，修正后 1415265 为 2 passed。
- 1415271：三项校准使用盒内居中的非周期模板；共 9 E/F，均正常退出，原模板 E/F、逐请求直接 ASE E/F、LASP 导出一致，非周期初始坐标保持。非对角晶胞文件行为与 ASE 行矩阵一致，不需要凭 Fortran 内存布局猜测转置。
- 1415237：Calculator 主动异常和 NaN 两项负例各调用一次，均无成功响应、旧 external.ene 被删除、LASP 返回 29、无残留后代。合计为 18 次 EMT E/F 加 2 次合成失败请求；不是搜索性能实验。

[有效三案例产物](../../research/ga_ssw/evidence/lasp-external-ase-centered-20260920/)、[失败路径产物](../../research/ga_ssw/evidence/lasp-external-ase-failure-20260920/)。能量/力判据为 1e-6 eV/eV Å⁻¹，用于协议差错检查，不用于研究机器学习势以下的精度效应。

## 明确边界

- 当前是研究 helper，未改变公共 PAM-SSW API、未提交或发布。
- 固定胞 E/F；constraints 明确拒绝，没有 stress/VC 协议或应力符号保证。
- 使用与 LASP ARC 标准取向一致的完整存储晶胞；任意旋转晶胞不能据本次下三角晶胞校准自动推广。
- 非周期初始结构需在存储盒内，并留足真空；检查初态不保证长搜索永不越界。尚无通用非周期轨迹展开实现。
- Calculator 永久阻塞时，当前常驻 Python 服务没有独立进程隔离；现有监督器管理 LASP/MPI 后代，Slurm 墙钟负责整个研究作业。不能承诺任意 DFT Calculator 都有可靠的请求级超时。
- `max_requests` 当前限制成功响应数；错误记录另存。LASP 在已测 Calculator 失败后退出，不重试。若需要持久重试服务，应另行设计总尝试预算，而非在这里隐藏重试。
- 此桥接不保证 LASP 与独立 Python 的附加压缩项、MC、初态/停止规则相同；公平算法对照仍须显式匹配和记录。

主线保持独立 Python SSW/LS/GA 的实现与真实体系验收。协议校准结束后，不继续扩展桥接框架或启动接口性能调参。

## 随后发现的长轨迹问题：非周期坐标映像

离线审计 1415591 扫描了 MH-1 原版两条轨迹：seed17093 的 60000 个请求中有 1165 个局部/混合 50 Å 映像跳变候选，seed17094 的 7551 请求没有此类候选；另有 1 行缺少可用坐标，不计为成功请求。

单点复核 1415616 对首个事件（4520→4521）只做 4 次 MH-1/omol E/F：

| 配置 | 能量 (eV) |
|---|---:|
| 跳变前原坐标，非周期 | -62176.564962865 |
| 跳变后原坐标，非周期 | -62168.044035760 |
| 跳变后撤销该原子的 50 Å 映像偏移，非周期 | -62176.571117673 |
| 跳变后原坐标，50 Å 周期胞 | -62176.571117673 |

原始最大分量位移 49.903 Å，撤销映像后仅 0.133 Å。后两种表示在本事件得到相同能量/力，而原 nonperiodic 回调多出约 8.527 eV。由此确认至少这一个事件中，LASP 的折回坐标与非周期 Calculator 的坐标解释不一致。1165 是候选跳变数，不是 1165 个已逐个单点证明的错误。

此前 seed17093 原版-vs-Python 效率和机制解释需要暂缓：原版轨迹包含这一接口几何不一致，不能作为忠实 LASP 算法的公平排名。已有保存终态的独立非周期 E/F 仍是那些终态的有效数据；独立 Python 内部的旋转/高度/LS 单因素对照不受该 LASP 桥接问题影响。

居中只解决初始表示，**不能保证长程非周期 LASP 搜索正确**。当前正在核查是否存在原生取消 wrap 的输入路径；不静默改为 PBC=True，也不将事件级撤销映像直接升级为通用无歧义解包算法。

证据：[全轨迹候选审计](../../research/ga_ssw/evidence/c60-native-image-audit-20260920/audit.json)、[4次单点复核](../../research/ga_ssw/evidence/c60-native-image-replay-20260920/result.json)。


### 原生坐标开关的限定探针

CPU 1415848（6秒，12次EMT E/F）分别测试默认、`SSW.LmaintainXYZ T`、`SSW.Lcentralize F`及两者组合；四组均正常退出，但负坐标输入的首个 external 回调仍有30 Å映像偏移，与原始非周期输入的能量相差2.562322 eV。`allkeys.log`保留实际解析值，故不是只写配置但未核查。CPU 1415868（2秒，3次EMT E/F）仅将ARC头改为`PBC=OFF`，结果相同。这些是初始坐标诊断，不能判定两个开关在其他阶段无作用，也不能证明所有原生选项均已排除。

探针故意冻结使用包含坐标启动校验之前的研究helper，以观察原二进制行为；当前helper的拒绝保护未移除。没有修改二进制或将计算后端改成周期模型。

[四组探针](../../research/ga_ssw/evidence/lasp-native-image-flags-20260920/plan.json)、[ARC头探针](../../research/ga_ssw/evidence/lasp-native-image-pbcoff-20260920/plan.json)。下一步只定位实际external写出调用链；不扩大到无界反编译或追加长搜索。


### 通用接口的数学边界与待定方案

将每原子坐标折回胞内的映射记为 `w(x)`。存在不同的非周期几何 `x` 与 `x+nH` 满足 `w(x)=w(x+nH)`，但一般 `E(x) != E(x+nH)`。因此，若传输层仅提供折回坐标，任何无额外信息的适配器都不能对任意非周期 Calculator 唯一恢复原能量与力。这是信息缺失，不是提高坐标打印精度能够解决的问题。

对真正周期且对整数晶格平移不变的势，折回坐标本身足以计算 E/F，不必为了普通周期 PES 强行恢复 unwrapped 坐标。对非周期势，逐请求选最近映像必须另有可靠参考及每原子试探位移上界；SSW/MC 的恢复和大跳需要单独保证，不能从一个连续线搜索片段推广。多个原子同时折回本身不是歧义，丢失参考或超出可唯一恢复的位移域才是问题。

目前不改变公共架构。若限定原生接口仍无法输出原坐标，后续有两个明确选择：

1. 将原版对照限定在已验证的周期 E/F 域；C60 可另外提出双方都使用同一大真空周期 MH-1 的新协议，验证其与孤立团簇的 E/F 一致范围，并在超出范围时保留失败而不继续比较。它不能静默替换已有非周期协议，也不适用于任意带长程项、外场或约束的 Calculator。
2. 为任意非周期 Calculator 继续寻找提供未折回坐标/映像数的原生路径。若必须修改二进制或拦截内部状态，则是新的适配架构，需要先讨论收益和维护成本；不将此成本无条件绑在独立 Python SSW 交付上。

独立 Python SSW 直接持有完整 ASE Atoms，不依赖这个有损通道，仍是通用 Calculator 接入的研发主线。


最后的限定静态核查已定位 `otherpot_` 的真实 external writer：`0x41677c` 引用文件名，输出 `module_str_mp_strt_%lat` 和 `%xcart`。root 用DWARF复核了字段偏移；没有向外输出image数。此前RIP-only检索漏掉movabs，已纠正，不能把旧“未找到xref”当作不存在路径的证据。此轮只定位了消费者，尚未定位更早的映像变换生产者，停止在这里扩大反编译。
[指令与字段证据](../../research/ga_ssw/evidence/c60-native-image-audit-20260920/external-writer-static-note.md)。
