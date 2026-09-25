> 后续用户决定：先做固定输入编号验证和原版启动核查，再判断是否需要此公共选项。此文保留为暂缓备选，不表示已批准或正在实现；目前不改变方向/RNG/checkpoint契约。

# Explicit startup order for independent recovered-direction searches

Status: proposal, not implemented. Evidence:[local C60 outcome](../../research/ga_ssw/evidence/c60-local-defect-20260925/direction-probe/report.md).

## Problem and goal

The complete recovered direction bundle can reach the sourced defect's Ih basin (2/4 short trajectories; global+same CBD0/4), but actual input-to-quench displacement is zero in all four arms. `select_native_local_group` resolves tied displacements by index. Reordering identical atoms changes its deterministic first axis in physical space. This is an existing recovered selection rule combined with our Python startup contract, not a failed gradient or evidence that observed hits are false. Goal: independent ASE search should not privilege the input file's first physical atom when it has no movement information. Preserve legacy behavior for comparison and old checkpoint compatibility.

## Minimal candidate

Add an explicit startup-order choice to existing `RecoveredDirectionSettings`, with legacy/native order as the compatibility default and an opt-in randomized order for independent searches. On initialization only, draw one permutation using the existing search RNG, apply it consistently to both input and quenched coordinates for the existing selection/refresh operation, then map pair and group indices back. Do not permute the user's stored Atoms object, modify subsequent displacement history, add a displacement tolerance, change CBD/LS/bias settings or invent an energy reward. One shared random ordering preserves the native tie relationship between its two searches; independent random tie breaks would be a different mechanism and are not proposed.

Record this choice with the existing direction settings/checkpoint identity. Old checkpoints imply legacy order; reject resume under a different explicit choice. Use existing RNG persistence rather than a second generator. Exact serialization compatibility must be checked before implementation; no claim that changing a class field alone suffices.

This changes startup sampling and random-number consumption. It is not a fix to the low-level recovered LASP helper, which stays unchanged. It is not a proven speedup, complete permutation-equivariance theorem for the whole algorithm, or a request to change the default now. Startup numerical roundoff is not the subject: the observed tie is exact because the already-qualified input is not moved.

## Alternatives and tradeoff

1. Keep core startup untouched and randomize input ordering only in a frozen validation ensemble. No public/state change; averaging can reveal ordering sensitivity, but ordinary callers still receive order-dependent startup.
2. Adopt the opt-in startup policy above, retaining legacy default. Public settings/state identity gain one semantic choice; routine callers can explicitly request unbiased label ordering. Recommended if this behavior is part of the independent ASE implementation rather than just benchmark preparation.
3. Replace legacy startup by default. Not recommended now: changes old trajectories and obscures comparison with recovered behavior before independent validation.

## Verification and boundary

Use existing fixed-direction checkpoint tests for old-default behavior and continuation/RNG equality. Verify index round-trip of pair/group and unchanged user coordinates; test the zero-displacement case with coupled permutations, not identical-seed path equality after arbitrary relabeling. Then use a frozen, bounded relabeling panel on the existing source defect, and an already available second molecular input, holding kernel/model/LS settings fixed. Report failed/censored outcomes and all costs. Do not fit new constants to C60 or call label robustness a general efficiency win.

This proposal requires discussion under AGENTS.md§8 because it changes a previously explicit startup-state contract, RNG consumption and checkpoint identity. Current experiments and archive work are complete and do not depend on approving it. No new GPU job waits for approval; no global default or core code has been changed.

## 2026-09-25 symmetry argument and scope before any implementation

The saved zero-displacement C60 startup is not a finite-difference precision question: the true quench leaves coordinates exactly unchanged. The recovered helper first resolves an `argmin(movement)` tie by row index, then constructs its distant axis/group from that physical origin. Thus the physical origin, not merely an index in the output, depends on input order. This is confirmed by the existing zero-PES selector audit. The full relabeling panel is now prepared separately with a fixed permutation; it does not isolate this tie from changes in physical random-vector assignments.

For any selector S, a permutation-symmetrized startup can be written S_sym(x;P,U)=P^-1 S(Px;U), with P uniform over row permutations and U the selector random stream. For any fixed relabeling Q, substitute R=PQ: S_sym(Qx) has the same distribution as Q S_sym(x), assuming independent ideal random draws and consistently permuted atom metadata. This establishes distributional row-order consistency of the startup operation only. It does not imply equal same-seed trajectories, rotation equivariance of every recovered helper, full-walker symmetry, higher success probability, or a need to copy every native numerical choice.

The minimal implementation, if approved after the requested qualification, remains one explicit settings choice with legacy behavior as default. Apply one shared permutation to startup reference and current geometry, execute the existing selection and refresh, map **both diagnostic and active** pair/group fields back, then discard the permutation. Existing checkpoint state owns the mapped pair/group and RNG; no second RNG or trajectory-wide reordering is needed. Old serialized settings must resolve to legacy behavior; configuration mismatch must be rejected before advancing RNG. Pool restarts already invoke the initialization contract and must follow the same explicit setting. Constrained/periodic controller generalization is outside this proposal.

Alternative: retain the public API and only randomize row order in prospectively fixed benchmark ensembles. This costs no new state contract and is sufficient for the present qualification panel, but leaves the ordinary search's startup origin dependent on file order. Neither option is justified as an efficiency improvement by this C60 panel. The requested decision concerns the independent ASE interface's symmetry contract, not whether to copy native startup exactly or retune C60.
