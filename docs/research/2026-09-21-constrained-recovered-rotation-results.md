# 约束固定胞 SSW：恢复 CBD 接口与验证

## 能力与范围

用户批准的最小方案已落实到 `ConstrainedSSWConfig`、既有活动坐标回调和共享 reduced loop；默认 `None/force` 保留。固定原子通过线性活动坐标排除，Hookean 进入旋转及淬火的一致目标。新增显式 `force_or_budget` 只释放认可的预算出口，保留未收敛标志；无效向量仍失败。恢复 CBD 使用既有实现，没有复制另一套旋转算法。

选择性方向坐标仍只限制旋转；方向与 bias_reference 均映射回完整活动坐标。PAM Gaussian + recovered 和显式 presweep + recovered 在计算前拒绝。本次不包含 nativeMC、池状态或周期 pair/group 选轴；不把已有 LS 组合接口视为这次真实体系验证已覆盖。

## 工程证据

- CPU Slurm 1439809：`python -m pytest -q tests/standalone/test_constrained_*.py tests/standalone/test_rc_reference.py tests/standalone/test_rc_forest.py tests/standalone/test_recovered_rotation_integration.py`，88 passed；覆盖新接口、既有约束/LS/PAM及共享 RC 路径。
- 同一作业加载改动前实际 Cu/Al schema-1 checkpoint：2/2 零 PES 恢复；旧配置无新增字段时默认仍 None/force，位置、RNG、累计成本不变。
- CPU Slurm 1439833：加强 Hookean 每个实际 CBD endpoint 与 ASE 直接约束修正的一致性断言后，10 个新接口测试全部通过；真实输入、模型及两配置预检 errors=[]。
- EMT 连续两步与分段 1+1 恢复的坐标、累计请求及 RNG 一致；配置变化在 PES 前拒绝。它是实现验证，不是 MLIP 上逐位重现承诺。
- 子 agent 早期测试在本地 CPU 运行，未遵守本任务计算节点约束；以上正式检查在 CPU Slurm 重做。没有保留下来的改动前失败记录，故不声称完成严格先失败后通过的开发流程。

源码：`pamssw/standalone/constrained_reference.py`、`rc_reference.py`；测试：`tests/standalone/test_constrained_recovered_rotation.py`；用法：`pamssw/standalone/README.md`。

## 真实体系资格检查（完成）

GPU 1439842：Cu111+adatom（FixAtoms + plane Hookean）与 Cu55（人工点 Hookean），各比较旧 Broyden 与恢复 CBD，两外步，OMAT-small/omat_pbe/float64。总上限 12000 搜索请求 + 8 fresh，1 V100/30 min，每臂最多 300 s，不重试。两臂停止政策也不同，因此不是单独旋转算术效率消融。

Cu55 初始约束力规定为 0.1 eV/Å，由 k=1 eV/Å² 推得 rt=r_initial−0.1 Å，仅用于非零 Hookean 接口测试，不是防碎裂通用参数。初淬火后若约束失活，将明确报告。

原始协议/运行入口：`../../research/ga_ssw/evidence/constrained-recovered-rotation-20260921/`（实际源码与测试快照、CPU日志、旧checkpoint结果）；GPU 包在相邻 `vc-qualification-audit/research/ga_ssw/evidence/constrained-recovered-rotation-real-20260921/`。GPU1439842已完成，4/4臂均完成两次外步、8/8初态与末次落点fresh通过；364搜索请求、299实际calculator调用、8次fresh，作业27秒。8个fresh活动总力均≤0.03 eV/Å，固定坐标/胞/PBC/约束metadata在端点保持。Cu111的Hookean在fresh时有作用，Cu55初淬火后失活。恢复组7次Gaussian旋转均真正force_tolerance通过，无budget release；放行逻辑仅有实现测试覆盖。CPU1439925零PES独立审计完成，4/4成本闭合、errors=[]；全部364条搜索轨迹的固定坐标、胞/PBC/组分不变。Cu111全部8个Gaussian阶段约束非零，Cu55全部7个阶段约束为零。落点与初态能差绝对值均≤0.00125 eV；未做置换/对齐后的basin分类，不能凭这些微小能差或未对齐位移声称发现新basin。因此只交付接口/短程端到端资格，不声明逃逸收益、效率优势或生产资格。

审计首次作业1439901因分析器未识别旧控制器的 `residual_converged` 标志而退出2；补正标志映射后1439925重分析通过。分析器使用固定文件名，首次派生结果被覆盖；现仅保留明确标注为重建的失败记录，不能视为原始1439901产物。成功结果已另存带job编号的文件及provenance。未修改原始实验、核心算法或验收阈值。

### 各臂实际成本

| 体系 | 旋转 | 搜索请求 | 实际 calculator 调用 | fresh |
|---|---|---:|---:|---:|
| Cu111+adatom | 旧 Broyden | 94 | 77 | 2 |
| Cu111+adatom | 恢复 CBD | 75 | 58 | 2 |
| Cu55 | 旧 Broyden | 121 | 104 | 2 |
| Cu55 | 恢复 CBD | 74 | 60 | 2 |

初态淬火已计入搜索请求；ASE缓存使命中请求不一定触发calculate，两列不能混用。本次预算远未用尽是短程接口检查自然结束，不是12k长程搜索成功。

## 决定

保留显式实验接口，不升级默认。当前验收目标是恢复旋转与受支持约束及checkpoint契约正确组合，搜索收益与长程C60验收仍是独立问题。实际工作区含继承的未提交改动；这是研究实现，不是已合并或已发布版本。
