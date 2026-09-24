# C60 大预算验收：资源方案与路线决定

**状态：`AUTHORIZED_PREFLIGHT_PASSED_NOT_SUBMITTED`。** 2026-09-24 用户明确批准四轨迹合计最多1000万搜索请求＋12次复核、176 V100 GPU小时、最多两卡并发，要求先完成续跑预检再提交。选择下文路线A；生产任务尚未提交。C4H6类别代表资格已完成并关闭该实验系列；新输入和协议已冻结；状态/预算续跑预检已通过（最终CPU1477564、GPU状态诊断1477513、I/O1477518）。GPU后续轨迹不承诺逐位一致，最早差异已定位为相同坐标下约5e-15 eV/Å的力微差。详见实验入口；批准及接口资格不代表科学验收。

## 决策问题

长期目标仍有两个独立验收条件：MH-1/omol势上的完整C60笼，以及同一势、同一固定胞条件下相对参考结构不高于+0.01 eV的能量。过去对17093–17096均已读出；17095/17096虽曾叫holdout，但现在也已使用，不能再作为独立留出输入。下一轮必须换初态。当前读到的C60专属协议和证据中没有17101/17102命中，因此将其列为**候选新seed**；这只是有限路径检查结果，不是整个仓库/历史作业的穷举证明。提交前仍须对归档索引、当前研究分支和输入清单作一次定向冲突检查。

## 建议的最小可解释比较

候选设计是2个新随机初态 × 2种算法臂（普通SSW、NativeLS），每个初态在两臂间完全共用，唯一有意改变因素为LS开关。它测量长搜索预算下LS相对同起点SSW的差异，不估计普适成功率，也不声称完整参数或轨迹匹配。

| 设置 | 建议值 | 来源与边界 |
|---|---:|---|
| 新输入seed | 17101、17102 | 仅为提名；使用既有 `generate.py` 的均匀立方体拒绝采样入半径5 Å球、逐原子最小间距1 Å、平移至(25,25,25) Å、非周期C60规则。记录试采次数、原始坐标和清单；该输入分布是开发协议约定，不是文献最优分布。用户已授权生成并冻结结构；两算法臂共用，不按初始淬火结果替换。 |
| 模型 | 固定MH-1，omol，float64，CUDA | 复用C60 MH-1运行协议；提交前记录模型实际文件和SHA256。模型适用域及结论只限此势面。 |
| SSW臂 | 复用C60冻结主体：width 0.6 Å、max Gaussians 12、150 K、fmax 0.03 eV/Å、relax最多1000、FD步长0.001 Å、forward force 0.1、global方向、direction_only、safe-lbfgs-total、memory 500、NativeMC maxtrap 99999 / energy tol 0.1 eV。**实际旋转控制采用恢复CBD**：pre_rotmax 5、rotmax 15、pre_ftol 0.2、ftol 0.02、metric Euclidean、max_force_calls 40。 | 耗时基线来自恢复CBD运行。它们的config中仍记录Broyden-Euclidean、rotation bias/HVP/tol等普通旋转字段，但恢复CBD控制器覆盖这些字段，字段值不代表实际执行算法。SSW和NativeLS两臂均用同一恢复CBD设置，唯一有意算法差异是LS开关。 |
| NativeLS臂 | SSW臂全部设置 + 既有NativeLS：C–C键能3.44684 eV、键长1.54 Å、scale 5、amp_c 2、长度容差0.1、target 20 meV/atom、eta 0.005、max_change 0.01、frequency 10、presteps/cycle 100、ratio 1.1、lselfadapt、prequench fmax 0.1 eV/Å / 50步 / force_or_step_limit | 明确继承C60旧协议的20 meV/atom target及 `mh1-native-ls-equal-budget-20260920/plan.json`。不借用C4H6的700 meV/atom。此Python实现与参数来源有已知近似边界，不宣称原生LASP完整轨迹等价。 |
| 搜索预算 | 每轨迹最多2,500,000累计E/F请求，**包括该轨迹的初始quench**；四轨迹合计硬上限10,000,000 search requests。每臂另最多3次fresh E/F请求（初态、最佳态、至多一个不同笼候选），四臂最多12次fresh；总上限明确为10,000,000 search + 12 fresh。 | 请求计数器是硬预算，失败请求也计入其所属search或fresh上限。不得因命中候选自动追加搜索。 |
| GPU墙钟预算 | 每轨迹最多44小时；总上限176 V100 GPU小时 | 最多2张V100并发；四任务两波并行时理论墙钟至多88小时，不把GPU小时与历时混写。单臂到44小时仍未达请求上限按walltime截尾归档。 |

### 运行时间量级：来自实际MH-1记录的有限外推

现有C60恢复CBD、无LS记录各做了60,000搜索请求：17093为3,234.47 s（0.05391 s/请求，57,022次实际Calculator `calculate`），17094为3,212.53 s（0.05354 s/请求，57,241次实际`calculate`）。恢复CBD+NativeLS记录报告51:53和52:41 GPU运行时间、每臂60,000搜索请求；即约0.0519和0.0527 s/请求。此处不使用普通Broyden控制记录作耗时外推。分母是预算请求，不等同实际`calculate`调用数；调用数与失败、拒绝都应另报。

按上述四条恢复CBD短运行的观察速率线性估算，单轨迹250万请求约需36.1–37.4小时，四条约需144–150 GPU小时。44小时/臂留出约17–22%的墙钟余量，176 GPU小时是硬上限而非预测需求。该估算不校正长程温度/结构造成的吞吐变化、GPU争用、fresh开销、节点故障或checkpoint回退；不能声称足够或保证跑满预算。长跑每段须同时报告wall seconds、总请求、成功Calculator调用、拒绝/失败数，禁止从模型单次延迟代替端到端成本。

## 续跑与分段边界

更新：此前已获批准的SSW外步checkpoint回调现已实现并合入3a8313a，见[接口与验证](../../research/ga_ssw/evidence/ssw-boundary-pause-20260924/README.md)。长程runner推荐使用该回调在完成外步后合作式暂停；下文单步循环是既有API备选，不再是必须路径。GA active-walk/v2仍未实现，且不属于本次两臂C60协议。

既有固定胞API原则上可用于安全边界分段：`run_ssw`、`run_ls_ssw` / `run_native_ls_ssw`提供`checkpoint_path`，每个完成的外步原子替换保存checkpoint；续跑时用`load_ssw_checkpoint`显式加载，传入新的ASE surface、相同设置和相同bit-generator类别，checkpoint恢复当前结构、archive、LS响应、策略状态、RNG状态与累计请求。`steps`表示额外外步；以`steps=1`反复调用，可将每次调用限制为一个外步并在返回后安全保存/检查预算，不依赖新的公共`on_step`回调。**这一调用方式在长程协议上的状态连续性和请求账本仍须通过小规模预检验证**，不能仅由API文档推断数值等价。

风险边界：checkpoint不保存calculator；硬终止若发生在一个外步内部，该外步已支付的请求只能由外部账本保留，恢复会从最后完成外步重新开始，可能重复成本。预检需将checkpoint累计请求、fresh/search分类与追加上限统一核对，账本要求唯一请求ID并保留重跑/失败。每段以`steps=1`边界调用并核对12小时段长；若单外步本身可能超过段长，须调整段落策略或停止长跑，不新增公共API。最小预检至少验证：中断前后累计预算不回退、RNG/LS状态恢复、restart后续外步结构/事件与连续运行接口一致、fresh不计入search cap且总额封顶正确。

## 验收和停止

四条轨迹分别报告执行状态、数值资格、结构资格和科学结论。结构条件单独检查连通60C、每个碳三配位、三连通平面图，面数为12个五元环和20个六元环，并对1.64/1.70/1.80 Å键截断敏感性复核。能量条件单独用同模型同胞fresh E/F复核，阈值为既有C60参考能量+0.01 eV。一次成功候选不代替另一个条件；小力、三配位或搜索程序结束都不等于C60验收。

外步总数不继承旧诊断脚本的1000步上限；由同一总请求/累计墙钟预算管理长轨迹，在安全外步边界暂停。若完整笼与参考能量两项均通过独立复核，该轨迹可提前结束并记录首次同时达标的累计成本；其余轨迹继续按各自预算，未命中按删失结果报告。输入淬火或物理资格失败保留为对应初态的失败，不补抽有利初态。

每条轨迹到达请求cap或44 GPU小时先停；不因结果差、没成笼或接近成功而扩容/重跑。输入失败、checkpoint不兼容、模型哈希变化或预算账本不能闭合时停止该臂，不用其它臂的剩余额度顶替。即使四条全部完成，两个新初态也仅给有限的可重复性证据，不能形成稳健成功率估计。

## 预算是否值得：路线建议

论文结果都只作各自协议内的尺度参照，不作跨论文精度比较。主agent已直接核对`/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/215.txt` Table1（约410行）：LS2024论文C60的SSW与LS各20/20，平均E/F评估量分别为240万和220万；它们可作为每轨迹250万请求提案的数量级参照，但原论文评估数与本项目请求数口径不完全相同。GA-SSW 2026论文报告其SSW与GA-SSW结果各5/5；由该文步数和SI评估数粗略相乘的约3.5M/5.5M来自不同算法及步数/评估口径，只表明搜索成本可达百万量级，不能当作本提案同协议目标或与LS 2024作同精度横向比较。两项工作的势、初态和协议均不同于MH-1；其成功次数不能保证MH-1找到笼，也不能将250万请求解释为成功概率阈值。

当前已有四条60,000请求C60记录均未达到笼与能量双验收，且两个复用初态的NativeLS结果方向相反（一个最佳能量下降、另一个上升）。这些证据说明早期比较不足以支持组件收益或长预算成功保证；它们没有证明增加预算必然无效，也没有显示明确的算法修复信号。完整方案要占最多176 GPU小时，约等于两张卡连续运行88小时；相比之下，当前C4H6代表结构曲率资格仍处收敛阶段，且其它功能开发可直接形成可复用软件能力。

**有两种可选路线：** A）C4H6资格读出后执行四轨迹C60比较，预算上限176 GPU小时，取得MH-1下长预算双验收和LS方向的有限证据；B）继续功能开发，C60双验收保持未完成，把这笔GPU预算保留给后续有更强决策信号的实验。历史提案推荐先完成C4H6资格读出再选择；该读出已完成，用户现已批准路线A，须先完成续跑预检。即使选择A，两个新初态仍不足以给出稳健成功率估计。

## 主要证据入口

- [`2026-09-23-c60-budget-scale.md`](2026-09-23-c60-budget-scale.md)：论文用量级、势面差异和不外推的边界。
- `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/literature/215.txt` Table1（原PDF同目录215.pdf，DOI10.1021/acs.jctc.4c01081）：主agent本轮直接核对C60平均评估数；提案只将其用作预算量级参照，未把它与GA-SSW 2026跨口径数字精确比较。
- [`2026-09-22-recovered-rotation-ls-results.md`](2026-09-22-recovered-rotation-ls-results.md)：旧初态完整NativeLS对照结果、0/2笼、能量变化方向相反。
- [`c60-mh1-python-20260919/plan.json`](../../research/ga_ssw/evidence/c60-mh1-python-20260919/plan.json)、[`c60-recovered-rotation-long-20260921`实际汇总](../../research/ga_ssw/evidence/c60-recovered-rotation-long-20260921/c60_17093-seed17093/summary.json)：冻结SSW参数及实际60k请求耗时/调用账本。
- [`c60-recovered-rotation-ls-prospective-20260922/plan.json`](../../research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922/plan.json)、[GPU日志17093](../../research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922/gpu-17093-1441611.out)、[独立读出](../../research/ga_ssw/evidence/c60-recovered-rotation-ls-readout-20260922/analysis-1441616.json)：NativeLS参数、实际请求和控制差异。
- [`c60-random-inputs-development-20260917/generate.py`](../../research/ga_ssw/evidence/c60-random-inputs-development-20260917/generate.py)：候选新seed沿用的随机初态生成规则。
- [`standalone/README.md`固定胞checkpoint契约](../../research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922/source/pamssw/standalone/README.md)、[`paper_reference.py`](../../research/ga_ssw/evidence/c60-recovered-rotation-ls-prospective-20260922/source/pamssw/standalone/paper_reference.py)：既有API实现与checkpoint约束。
- 论文：DOI [10.1021/acs.jctc.6c01078](https://doi.org/10.1021/acs.jctc.6c01078)。SI量级见`2026-09-23-c60-budget-scale.md`所记录的作者版SI本地逐页核验；该文明确说明论文势、MH-1及计数口径不可直接等同。
