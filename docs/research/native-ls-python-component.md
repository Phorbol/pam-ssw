# Independent Python component for recovered native LS arithmetic

2026-09-10. `pamssw/standalone/native_ls.py` now implements the recovered
initialization, effective amplitudes and normal response update without invoking
LASP. It is a reusable component, **not a full native LS walk/controller**.
Existing paper-mode `FrozenBondSoftening` and `LSResponseState` are unchanged.

The implementation is grounded in [initialization](native-ls-initialization-contract.md)
and [amplitude/response](native-ls-effective-amplitude-response.md) original
instruction evidence. Public entry points are:

- `initialize_native_ls(atoms, bond_energies=..., bond_lengths=..., ...)`: strict
  geometric bond count, recovered float32 N*.02 normalization and B table.
  Returns table, length table, bond count and a `FrozenBondSoftening` potential.
- `freeze_native_ls(atoms, table, lengths, atom_filter=..., amp_c=2., ...)`:
  freeze current pairs/distances with A_ij=amp_c*B_ab*m_i*m_j and xi=.2.
- `response_mev_per_atom(E_before,E_after,N)`: 1000*(E_after-E_before)/N.
  These energies must belong to the true PES before/after soft prequench.
- `normal_update_due(step, frequency=10, presteps=100)`: step!=0 and
  (step%frequency==0 or step<=presteps). This is only the normal-branch
  predicate, not a recovered periodic-cycle scheduler.
- `update_native_table(..., response_mev_per_atom=..., target_mev_per_atom=20.,
  eta=.005, max_change=.01, branch='normal')`: recovered max-reduction Q,
  old/new bond normalization and response correction. Inputs are not mutated.
  It returns the new table and Q; it neither advances a step nor runs a PES.

HC lookup constants retain their promoted release float32 values. Other tables
must be explicitly supplied with provenance. Scale=5, amp_c=2, tolerance=.1 A,
eta=.005, cap=.01, frequency=10 and presteps=100 are recovered release settings,
not suggested universal physical parameters. B/A are interpreted in eV and
response/target in meV/atom; eta is a numerical conversion/update coefficient,
not the paper's learning rate 1.8. The response cap controls the response term,
not the separate old/new count normalization.

The caller still owns prequench, convergence qualification, update eligibility,
next accepted seed selection and potential lifetime. In particular the periodic
save/restore branch raises `NotImplementedError`; failed-prequench eligibility
is not invented. Frozen-atom constraints and zero-bond states are rejected,
as the recovered initialization oracle did not close those branches. Zero atom
filters suppress amplitude but do not remove bonds from normalization. Negative
updated B is rejected without clipping. ASE MIC supplies geometry; arbitrary
periodic-cell native minimum-image parity is not asserted.

Validation: `tests/standalone/test_native_ls.py`, **3 passed**. Eight frozen
original-instruction arithmetic cases (three amplitudes, four updates, one
response) match. Four initialization fixtures on supplied C60 and trans-C4H6,
including scales 5 and 2.5, reproduce counts and matrices. Frozen-potential
energy/force finite differences and zero-filter handling are also checked on
these geometries. No native binary, MACE or physical PES job was run. These
checks establish arithmetic/geometry implementation, not scientific LS search
performance or a complete native cycle reproduction.

## Followup: explicit cycle state implemented

`NativeLSCycleState` now implements the recovered adaptive and nonadaptive
save/zero/restore transitions as an independent pure Python state machine. Its
`advance(step, measured_response_mev_per_atom=..., new_bond_count=...)` accepts
caller-identified observations and returns ordered actions, phase, table, count,
response and Q. Normal arithmetic remains the same reusable component above;
its standalone `branch` argument still does not perform periodic transitions.
Use the state class to orchestrate those known transitions.

The class implements Fortran NINT for nsoftstep, the different adaptive versus
nonadaptive phase origins, adaptive count and int32-truncated response notes,
restore-before-update order, and transactionality on invalid/unsupported state.
An attempted restore without saved state raises rather than inventing a note.
It does not infer step/accepted-seed semantics or backend-failure eligibility.
No physical calculator or walk wrapper is introduced by this addition.

`tests/standalone/fixtures/native_ls_cycle_static.json` freezes independently
computed transition cases with exact instruction addresses. These are labeled
**static-formula fixtures, not original-instruction execution outputs**. They
cover adaptive save/zero/off/restore+update and nonadaptive save/restore. Existing
four original normal-update oracle cases also pass through the new controller;
all original eight arithmetic cases remain checked. Suite now **6 passed**,
including failed-state rollback and negative response truncation. The next layer
is walk integration with an explicitly chosen caller lifecycle, distinct from
unproven native caller parity.

## Followup: complete native-derived LS walk integration

`ls_native_reference.py` supplies `NativeLSSettings` and `run_native_ls_ssw`.
The existing SSW driver now dispatches the explicit settings type to the native
initialization/controller adapter, sharing its full prequench, direction search,
Gaussian climbing, bare final quench and MC lifecycle. Paper `LSSettings` remains
the unchanged default algorithm; its periodic image-pair selection is preserved.
Both modes now return `ls_initialization_failed` with the already paid initial
minimum/ledger if construction fails after the initial physical quench.

The independent caller convention is explicit: update after each completed
outer attempt with a converged soft prequench, including a rejected candidate or
failed climb. The controller's one-based step counts those attempts. Recount and
freeze at the **selected current** structure after the MC decision. Biases never
enter bare final quench/MC. Each native step records `ls_update` with measured
meV/atom response, restored/used response, count, phase, table, Q and ordered
cycle actions. Native upstream caller parity is not asserted.

Prequench nonconvergence stops without an update; this deliberately preserves
the existing independent physical-certification policy rather than adopting the
native optsoftmax response eligibility without a full numerical failure model.
Native-derived frozen geometry currently uses the recovered simple MIC pair
selection, not the new paper-mode image-resolved periodic potential. That
limitation is visible on Cu4 half-cell contacts: the native-mode prequench fails
and is retained as a failure, not rescued by swapping neighbor semantics.

Driver validation: real Cu13/EMT, existing Ritz direction solver, two attempts,
**65 physical E/F requests**, two completed Gaussian-limit proposals, three
bare physical minima including initial. Independent E/F certificate checks
verify no LS energy contamination; initialization and step costs reconcile.
Explicit Cu table (3 eV, 2.8 A) and scale .1 are a wiring fixture, not a recovered
Cu lookup or scientific parameter recommendation. Cu4 prequench failure and
paper/native missing-table initialization failures also have contract tests.
`test_native_ls_driver.py`, native arithmetic/controller tests and existing
atomic extraction tests pass together: **14 passed**. No MACE jobs or native
runtime were used. This closes an executable native-derived LS lifecycle with
explicit independent caller policy, not exact release parity or search efficacy.
