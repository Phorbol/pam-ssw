# 2026-09-22 研究分支远端同步

用户授权及时 push。本次将累积的独立 Python/ASE 研究源码、测试、研究脚本和说明同步到 `research/ga-ssw-behavior-parity`，不合并 main，不作为稳定发布。基于 92adf28；其中 VC/RC 等实验实现的存在不代表已完成科学资格验证，以 MAINLINE 为准。

## 提交边界

包含 pamssw、tests、研发说明、research/ga_ssw 顶层研究脚本及必要的小型测试数据。模型、LASP 二进制、论文附件、源码运行快照、大型轨迹和原始实验目录不在本次提交中；uv.lock 未纳入。历史说明中的本机路径、相邻 worktree 链接和未提交的原始证据只作为存储定位，并不保证 GitHub 上可直接访问。远端同步不是完整实验数据备份。

## 本次验证

从 Git index 导出副本后，在 CPU Slurm 使用 mace_env Python，PYTHONNOUSERSITE=1，单线程；无新增模型搜索。

- 1440337：`python -m pytest -q tests/standalone --disable-warnings --tb=short`：746 passed, 1 skipped。
- 1440348：`python -m pytest -q tests/test_*.py tests/unit/test_relax.py tests/unit/test_relax_fallback.py tests/unit/test_safe_lbfgs_history_depth.py tests/unit/test_safe_lbfgs_scale_decomposition.py --disable-warnings --tb=short`：167 passed, 23 failed。
- 其中一项失败是遗漏 native-vapor-reference/comparison.json，已补入版本控制；1440352运行 `python -m pytest -q tests/test_cluster_reconnection.py --disable-warnings --tb=short`，6 passed。
- 其余22项保持可见：2项 atomic_climb 与当前完整driver事件字段不一致；20项旧history/scale研究测试依赖缺失的 `/tmp/SSW-worktrees/fixed-proposal-replay/.../summary.json`。本轮不修改验收断言或算法来消除失败，不宣称完整测试全绿。两项事件对照在字段断言处停止，不能据此推断后续几何/成本断言通过。

新分支是可追溯研究快照。后续修复应另作提交；当前稳定基线与 main 均保持。
