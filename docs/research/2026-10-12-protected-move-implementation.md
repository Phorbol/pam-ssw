# Protected move implementation plan

Goal: implement the approved optional fixed-cell scalar-guard move, without changing default SSW.
Architecture: immutable named policy and pure geometry helper; one integration before Gaussian height preparation; schema9 retains underlying checkpoint identity. Python base-PES release is an explicit design, not native E/F parity.
Stack: Python/NumPy/ASE, existing pytest and Slurm verification.
Spec: [approved design](2026-10-12-protected-move-proposal.md); user approved 2026-10-12 in conversation.

## Global constraints

Default None preserves behavior and RNG. No LS/RC/VC/pool/constraints/PAM-width combination in initial implementation; reject unsupported combinations before PES. No guessed species table. Parameters 2 A maximum atom move, 0.25 maximum closest-distance contraction, 0.95 shrink factor, 0.1 A minimum retry length and native counter boundary 50 are explicit native heuristics, not universal defaults. No login-node numerical compute. Preserve raw failed runs.

## Review focus

- restore first/previous saved center, never final rejected trial; release must enter actual base-PES quench
- accepted actual displacement/width/direction shared by energy, force and persistent Gaussian history
- valid periodic images including self images, no falsely zero self distance
- schema9 validates policy and wrapped schema; old default and Q/shape/Hookean paths remain intact
- unsupported combinations and invalid geometry fail before paid PES; finite geometric retry without RNG

## Task 1: core and targeted tests

Files: new pamssw/standalone/protected_move.py, tests/standalone/test_protected_move.py; change paper_reference.py plus narrowly required checkpoint validation/export/docs.
Interface: `ScalarMovePolicy` frozen settings, `protected_move(atoms, direction, width, policy)` returns accepted flag, trial atoms, actual direction/width and diagnostics. Public run_ssw `move_policy=None`; typed immutable policy only. An exhausted move returns diagnostics without adding a Gaussian; driver owns saved-center restoration and true quench.
- [x] Inspect native counter initialization/increment and ng-to-Python center mapping, record chosen exact semantics.
- [x] Write regression tests for direct pass, shrink pass, exhaustion first/later center restoration, actual persistent width, periodic distance, unsupported combinations, None equivalence, Q+policy checkpoint and split/resume, mismatched restore rejection.
- [x] Run red test on CPU node, implement minimal helper/integration/schema9, run targeted suite on CPU.
- [x] Independent scoped review and focused fixes before real computation.

## Task 2: paired paper-case experiment

Files: research/ga_ssw/probe_protected_move_paper_cases.py and evidence/protected-move-20261012/{protocol.md,*.sbatch}; do not edit Task1 files.
Reuse input/backend/hash/provenance/ledger helpers from probe_q_paper_cases.py. C60 isomer2 MH1/omol, TiO2 phase87 OMAT-small, S2 mode; exactly one outer step each protected on/off. Same seed20261012/config/model/determinism. Maximum four arms x3000 search requests, six Gaussians; four independent endpoint checks, one V100 <=30min. Save every attempt, errors, guards, costs, structure metrics and frozen code. No duplicated second deterministic outer step or threshold tuning.
- [x] Freeze protocol and complete runner wiring CPU smoke (synthetic does not count as science).
- [x] Submit after core gate; inspect independent endpoints and all failure costs.
- [x] Record whether guards improve endpoint qualification/cost; keep default off regardless of this small development panel.

## Task 3: integrate and report

- [x] Verify artifacts and targeted results, update MAINLINE, approved proposal status and readout.
- [ ] Commit only scoped files/evidence and push authorized research branch; verify remote SHA.
- [ ] Continue next evidence-led task unless architecture/user decision is genuinely required.

## 执行记录

- 基线1926306；CPU1754084运行基线+新测试，按预期缺少protected_move模块。
- v1核心CPU1754151：87项通过，2.91秒。未因此直接提交GPU。
- 独立审查发现保护成功后若背景E/F或高度准备失败，guard日志会丢失；要求记录并
  测试失败路径。Q恢复测试追加实际Gaussian与RNG断言；README恢复中心措辞纠正。
- runner CPU1754155：四臂都在人工各向同性二次势的CBD acos退化处失败，保留原结果。
  下一版只在--smoke显式使用identity CBD stub验证接线；真实MACE不替换CBD。
  无性能结论，无GPU预算消耗。v2完成后重新冻结并验证。

- v2 CPU1754239：88项通过（2.49秒），独立审查无剩余实质问题。
- v2 smoke1754240：四臂接线通过；仅smoke有identity CBD。
- GPU1754252：四臂完整，1354搜索+4fresh请求，四终态均力合格。
  首次缩步前配对一致至浮点量级；C60连通改善但仍异常配位，所有落点高于初态。
  [读出](../../research/ga_ssw/evidence/protected-move-20261012/readout.md)。
- 继续固定参数跨结构检查，不调阈值，不将短程结果升级为搜索效率结论。
