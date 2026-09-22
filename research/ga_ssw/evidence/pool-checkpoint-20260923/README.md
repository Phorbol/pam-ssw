# 池搜索外步断点：实现与恢复资格

用途：实现检查／真实 Calculator 接口资格；不是搜索效率比较。
批准依据：`docs/research/2026-09-21-pool-checkpoint-proposal.md`，用户2026-09-23确认。

## 问题与判据

只保存原子结构不能恢复池搜索。必须在同一个完成外步边界保存当前观察索引、最近落点、池archive/统计/映射、LS与方向状态、MC及两套随机流。

判据：在相同配置下，连续与分段运行具有相同选择、状态及累计成本；真实EMT确定性环境要求坐标一致。错误策略配置在新PES调用前拒绝。暂停不finalize。终止错误只保存诊断、不可恢复。

## 实现

- schema5用于显式池checkpoint，旧非池schema1–4保留。
- 核心要求 `checkpoint_contract()`、`export_state()`、`restore_state(payload)`；不依赖research适配器。
- PAM适配器明确序列化archive/原型/统计、观察映射、代表索引、outcomes/decisions、source/executed。执行步数按实际外步计，允许两次回调之间有失败外步。
- 原子写入复用已有保存函数。恢复不重新初始化LS；只有实际池跳转才遵循既有LS重启语义。
- 不保存Calculator、不支持Gaussian或内层优化中断恢复、不改变评分和参数默认。

## CPU检查

命令见 `tests.sbatch`，环境 `/home/gengjianrui/.conda/envs/mace_env/bin/python`，PYTHONNOUSERSITE=1，CPU-MISC，单任务，组账户，10分钟上限。

| 作业 | 结果 | 解释 |
|---|---|---|
| 1453116 | 3失败、47通过 | 修改前新恢复测试均被旧池checkpoint禁用规则拦下 |
| 1453172 | 1失败、60通过 | 原有普通回调拒绝测试只因错误消息变为缺失显式契约而不匹配；拒绝行为保留 |
| 1453186 | 1失败、63通过 | 新组合夹具错误地要求零力平面势完整方向成功；连续和恢复都在第二步触发既有acos保护 |
| 1453191 | 定位检查 | `direction-diagnostic.py`确认首步完成、第二步 `native acos outside numerical domain`；不改科学算法 |
| 1453194 | 64通过，2.50秒 | 明确检查上述失败被一致重放；真实EMT成功恢复对照通过 |
| 1453209 | 64通过，2.51秒 | 独立审查修复后，终止checkpoint在调用任何selector hook之前拒绝 |
| 1453251 | 配置预检通过、零PES | 实际解析两体系输入/config/LS/MC及预算；原runner字段错误在GPU提交前修正 |

新真实对照：Cu2/EMT论文LS，uniform及PAM各连续4步对2+2；Cu13/EMT原生LS+原生MC+PAM连续2步对1+1。新构造surface/adapter及不同初始RNG恢复；核对实际选择、两套随机流、archive完整状态、LS更新、MC及累计成本。Cu2还核对finalize结果一致。原有参数仅作接口夹具，未升为物理默认。

完整方向组合使用退化解析夹具，只证明状态与错误重放，不能称真实PES成功。另一组合检查验证MC错误终止checkpoint零PES拒绝恢复。

子agent曾在登录节点运行适配器单测，违反本任务的执行约束；该自报结果不计资格。上述验收由root独立提交CPU节点执行。没有以子agent自报代替验证。

## 后续资格

C60/MH-1 omol与anatase/OMAT-small的模型重载连续/分段检查已提交GPU1453252。协议 `protocol.json`，运行命令 `gpu.sbatch`，实际源码提交46c231c；1V100、30分钟、两体系合计11988 E/F上限。不承诺GPU逐位一致，也不把数值差异当算法收益。结果尚待读取；`sbatch --wait`和依赖CPU读出用于完成通知，不高频轮询。
