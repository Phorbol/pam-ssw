# Block atomic-climb reference energy: bounded contract review

2026-09-11. Static review of the reference passed from the sequential
cell→atomic block to `atomic_climb`. No PES calculation or production change
was made.

## Current Python behavior

At `pamssw/standalone/block_ssw.py:167-170`, the atomic stage receives

```text
reference_energy = current.objective - pressure * volume(work)
```

where `current.objective` is the previously accepted full-quench objective
(`E_current + p V_current`), while `work` is the structure after the cell
cycles and fixed-cell partial atomic relaxation. `atomic_climb` then compares
each fresh bare fixed-cell energy against this scalar at
`pamssw/standalone/atomic_climb.py:131-135`.

At zero pressure this is the previous outer basin's energy. At nonzero
pressure the candidate comparison is exactly equivalent to
`candidate_energy + p*V_work < current.objective`: it compares candidate
enthalpy against the old accepted objective. It does not require the old and
`work` structures to have the same volume. The code comment at
`block_ssw.py:102-103` is therefore algebraically valid as an outer-enthalpy
threshold translated to the fixed-cell candidate energy; the unresolved point
is whether this old accepted objective is the intended anchor after the cell
block.

## What the 2014 paper defines

Section 2.1 introduces the SSW trajectory from a current local minimum
`R_i^m`, with the modified surface built from the real PES and deposited
Gaussians (`vc2014-author.txt:145-163`). Section 2.2/2.3 then specifies the
cell block ordering: identify a cell mode, displace the lattice, relax atoms
at fixed lattice, optionally apply the atomic SSW module at that fixed
lattice, and finally relax all degrees of freedom
(`vc2014-author.txt:322-341`).

Those sections do not define an energy threshold for the optional atomic SSW
after a cell displacement. They do not say to reuse the pre-cell minimum's
energy, to evaluate the post-partial-relaxation `work` energy, or to subtract
the new `pV` from an old objective. Consequently the current Python reference
is an explicit independent policy, not a paper-derived contract. The paper
ordering alone cannot justify changing it toward either threshold.

## Native VC energy-state evidence

For the VC `ssw_crystal_basic_mp_ssw_move_` (`0x5e5de0`) slice, DWARF names
`energy0=object+0x1b20`, `tene0=object+0x1b28`, and `cell0=object+0x1b30`.
The inspected writes include `object+0x1b28 <- object+0x230` at
`0x5e708e–0x5e709f`; the same slice has no direct write to `object+0x1b20`.
Earlier, `object+0x230` is copied to `object+0xb90` (`localbest`) at
`0x5e6520–0x5e6527`. These facts show VC energy bookkeeping and a `tene0`
write, but do not close which write feeds the VC
`ssw_crystal_basic_mp_climb_convg_` (`0x5f4360`) reference or when it is
consumed after the cell block. There is therefore no direct VC evidence for
either the Python old accepted objective or a post-cell `work` energy anchor.

The VC slice does show objective bookkeeping around `object+0x230`: at
`0x5e5f35–0x5e5f59` it adds the pressure-volume term controlled by
`para+0x2db38`, and at `0x5e5f70–0x5e5f90` it subtracts a volume-scaled term
before a subsequent callback. Those operations are not an energy-reference
comparison and do not identify the `energy0` write. The available VC chain
therefore stops at an unresolved energy-state consumer.

The native evidence does not specify the 2014 paper's omitted hybrid
threshold. It leaves a precise independent-policy boundary: after cell
movement/partial relaxation, the old outer objective and current `work` energy
can differ even at `p=0`.

## Bounded decision

Do not retune the threshold from the new experiment outcomes. The precise
remaining question is whether the independent block intends to preserve the
outer accepted-basin threshold (current behavior) or to define an atomic
climb from the actual post-cell `work` energy. A minimal future audit can use
a counted synthetic surface with different old/current energies and volumes,
then verify which scalar reaches `atomic_climb`; it need not run a full search.
Until that policy is explicitly chosen, report the current behavior as
`outer accepted objective translated with work volume`, not as a
paper/native-derived reference.

Sources: `block_ssw.py:102-103,167-170`, `atomic_climb.py:131-135`,
`literature/benchmark-sources/vc2014/vc2014-author.txt:145-163,322-341`,
`docs/research/native-climb-convergence-early-exit.md`,
`docs/research/native-gaussian-caller.md`, and the VC
`native-cell-direction-evidence/ssw_move.asm` slice.


## Root follow-up: energy0 participates in MC, but the climb link is separate

A bounded scan of the complete named VC `allopt`, `make_decision`, `new_cssw`,
`set_status`, and `climb` routines found these direct energy0 references.
Raw disassemblies are in `native-cell-reference-evidence/`.
At0x5f96ba–0x5f96d9, `make_decision` passes `object+0x1b20` as the first
argument and `object+0x230` as the second to `metropolismc`.
The latter computes second-minus-first at0x57e84f–0x57e861 and returns an
acceptance flag through its fourth argument; its inspected body does not
write back either energy argument. Thus energy0 has a concrete MC reference
role at this call, not merely an inferred name.

A direct write `energy0 <- object.energy` occurs at0x5fbf06–0x5fbf0d after
an indirect callback+0x28, conditional on the preceding integer-remainder
branch at0x5fbedf–0x5fbee3. Its entire mode/initialization meaning is not
assigned here. A specialized path also adds an indexed scalar to energy0 at
0x5fc6fe–0x5fc712. Consequently the reference is not universally immutable.
These are write/call facts; they do not establish that the VC Gaussian
termination gate reads energy0, or that a2014 cell→atomic block shares this
modern native lifetime. The consumer must be checked independently.
