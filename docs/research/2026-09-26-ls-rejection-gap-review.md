# P2 LS 拒绝路径几何指针审查

## 决定

当前可证实的是“原版 native 调用者的 accept/reject 几何指针未恢复”，不是已发现 Python 功能缺陷。保留 post-MC、MC-selected-current 更新；不改核心代码。P2 的这一项应标为 native 等价性未证，而非实现缺失。周期覆盖及元素表/镜像身份仍是各自独立的范围问题，本审查不据此关闭它们。

## 证据与语义边界

论文 LS-SSW Sec. 2.4 将 MC 接受/拒绝置于落点弛豫之后，并说明被接受的落点取代当前极小值；公式 Eq. 15 本身没有定义调用者的坐标指针时序。当前 Python 调用者在 [paper_reference.py](../../pamssw/standalone/paper_reference.py) 的 MC 判定后仅于 `accepted` 时赋值 `current = landing.atoms.copy()`（约 1289–1292 行），然后把 `current` 传给响应更新（约 1300–1305 行）。[ls_native_reference.py](../../pamssw/standalone/ls_native_reference.py) 记录该约定为 `completed_outer_attempts_selected_current`。这满足文中接受/拒绝状态规则。

静态 native 证据尚不能判断其内部等价关系：`ssw_fixlat_mp_make_decision_` (`0x5cfe20`) 在 MC 调用 `0x5d195c` 后复制标量决策结果（`0x5d1961–0x5d1996`），附近仅看到写入 `object+0x2298`，没有坐标数组写入；对象运行时 vtable/间接 continuation 未解析。下一外步 `ssw_fixlat_mp_ssw_move_` (`0x5bc760`) 在 `0x5bcf43` 调 bond counter 前读取既有 `+0x8/+0xe0/+0x170` 字段，但此处只有能量快照，没有几何选择/恢复。故 native 在接受支路是否切到 landing、拒绝支路是否恢复 current，或是否由未追踪的间接调用完成，仍未证实。MC 返回标量并不等于几何状态已更新。

已有真实 C60 序列提供了可区分的 Python 拒绝路径：MH-1 `c60_17093-first` 的保存运行 `/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920/c60_17093-first-native_ls/result.json` 在外步索引 3 拒绝；该 landing 在独立 1.64 Å 图阈值有 84 对，而上一已接受落点有 82 对。索引 3 的更新记录仍为 `bond_count=82`，下一步更新的 `old_bond_count=82`。这与所选 current 一致，且若误用拒绝的试落点会给出 84；它支持当前 Python 行为，但不是 LASP native 调用链的动态证据。运行输入、协议及边界见相邻 worktree 的 `research/ga_ssw/evidence/mh1-native-ls-equal-budget-20260920/plan.json`；此 12 外步开发运行到请求上限，不能提升为独立效率/普适性结论。

## 可证伪测试与结论

若后续需要闭合 native 等价性，应在一条实际原版轨迹中捕获一次拒绝，且该次候选与此前接受结构的邻居计数不同；记录MC返回至下一次bond counter之间的实际几何/邻居表指针。上面的Python 82/84对是可区分夹具，不能假定原版相同输入会在同一个索引3拒绝或复现相同计数。只有逐次确认原版自己的起点、候选、决策及后继输入，才能判定其调用约定。当前静态切片未恢复continuation，不为追求逐位等价扩大反汇编或修改符合当前契约的Python实现。

**结论：**没有已证实的功能遗漏，代码无需修改。待闭合的是 native accept/reject continuation 的具体几何指针等价性；需要新证据时，按上述预先定义的 82 对/84 对拒绝分支判据做一次有针对性的 caller 恢复或运行态观测。
