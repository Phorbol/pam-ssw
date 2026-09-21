# Native fixed-cell cluster rigid-motion treatment

> **Execution correction (same-day follow-up):** [The complete warm-cache instruction oracle](native-setconstraints-instruction-oracle.md) resolves the earlier orthogonalization uncertainty. General non-principal-axis N=4/7/13 cases agree with the Euclidean SVD orthogonal projector to 5e-16. Inline Gram-Schmidt is present at original source lines 1064 and 1071. The static uncertainty below is retained as history, not a current assertion that the native modes are independently normalized without orthogonalization. A rank-deficient two-atom case does expose a separate normalization/degeneracy problem. Cold-cache initialization and full native trajectories remain unverified.

Date: 2026-09-10. Bounded static audit of the uploaded ELF. No production code, binary, protection, or Git commit changed. This report answers whether native fixed-cell mode generation and rotation contain rigid-motion treatment; it is not a full port or numerical parity certificate for that treatment.

## Main conclusion

**Yes: native fixed-cell random-mode generation and dimer rotation explicitly call `ssw_commsub_mp_setconstraints_`, which constructs translation and rotation modes and subtracts their components.** Consequently the independent walker should not be compared to native as though native searched unrestricted Cartesian directions with no rigid-motion treatment. The observed independent Cu13 rigid-motion failure makes this a concrete missing implementation feature to investigate, not a reason to add an arbitrary penalty.

Important qualification: the static arithmetic does **not** justify equating this routine with an exact QR/SVD orthogonal projector. It constructs the three rotations about laboratory Cartesian axes, normalizes each, and later subtracts the three mode components. This audit has not established cross-mode orthogonalization. For a non-principal-axis configuration those rotations need not be mutually orthogonal. The formal quotient-space projector and exact native arithmetic must remain distinguishable.

## Artifact and reproducibility

ELF: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
Existing assembly: adjacent `analysis/kernel-rotate_dimer-lines.asm`, fixed-cell `biasedrot`, `climb`, `moveds` assemblies, and `lasp-symbols.txt`.
Additional read-only command pattern:

```sh
objdump -dl -Mintel --disassemble=ssw_commsub_mp_setconstraints_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=ssw_fixlat_mp_gen_randommode_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=ssw_fixlat_mp_get_random_mode0_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=ssw_commsub_mp_random_pert_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=ssw_commsub_mp_velo_loc_ "$LASP_ELF"
```

Disposable assembly files are `/tmp/pam-native-setconstraints.asm`, `/tmp/pam-native-gen-randommode.asm`, `/tmp/pam-native-get-mode0.asm`, `/tmp/pam-native-random-pert.asm`, `/tmp/pam-native-velo.asm`. The existing `/tmp/pam-native-bfgsdriver.asm` was also inspected. No complete-program execution or new test count is claimed.

## Call sites and gates

### Random direction

In `ssw_fixlat_mp_gen_randommode_` (`0x5d5c50`):

- At source `Class_ssw.F90:1686`, the code sums squares of the integer atom mask (`object+0x8a8` descriptor), then compares with `3*N`: `0x5d5d08–0x5d5dca`, including scalar and SIMD reductions. The Boolean is stored at `[rbp-0x48]`. For ordinary 0/1 masks this means all coordinates free.
- After random-velocity generation (`vmb2`, `0x5d66fe`), `0x5d6703` tests that flag.
- `0x5d671f` calls `setconstraints(N, coordinates, direction)`, with coordinates from `object+0x170` and direction data from `object+0x1848`.
- `0x5d6733` then normalizes the resulting direction through `n_normal`.
- Another flagged branch at `0x5d61f1` jumps to `0x5da7d0`; that alternate random-mode path was not fully traced here. Different configured random-mode types must not be assumed to use identical assembly paths.

This establishes an actual mode-generation consumer rather than mere existence of an unused symbol.

### Dimer rotation

`newssw_basics_mp_rotate_dimer_` applies the integer coordinate mask at `0x6e77a0–0x6e77e8`, then tests `LFFIX` at `0x6e77f8`. If true, it calls:

```text
0x6e780e: setconstraints(N, reference_coordinates, updated_direction)
```

The updated direction is then renormalized (source `newssw_basics.F90:791`, scalar path near `0x6e7a4f`). The direction is therefore cleaned during rotation updates, not only once when first sampled.

`LFFIX` is initialized on the first rotation iteration. The scalar mask reduction at `0x6e8238–0x6e824f` takes a minimum; a zero minimum sets LFFIX false at `0x6e8270`, otherwise true at `0x6e825d`. For valid 0/1 masks, the fully free case enables this cleanup and any fixed component disables this particular whole-cluster cleanup. The preceding allocation and empty-size paths exist and are not a validated general constraint policy.

The fixed-cell `biasedrot` calls `add_rotation_bias` and then CBD rotation, so this update is on the direction-search path used by the original biased rotation. It should not be confused with physical molecule-fragment rigidity or RC-SSW generalized coordinates.

## Recovered mode geometry and arithmetic

`setconstraints_` has signature compatible with `(N, R, vector)` and local static arrays `NX`, `NY`, `NZ`, `ROTMODEX`, `ROTMODEY`, `ROTMODEZ`, `COM`, `AXISATOM`, and `V`. Its DWARF source mapping is `simplefun.F90:963–1087`.

### Translation

- The three 3-by-N arrays NX/NY/NZ are initialized with Cartesian unit entries (`0x579d29` onward), then normalized (e.g. NX square-root norm at `0x57a04c`).
- The vector is dotted with each translation vector, then the three components are subtracted at `0x57a800–0x57a8ba`.

For ordinary N>0 this corresponds to normalized translations

\[
t_\alpha(i)=e_\alpha/\sqrt{N},\qquad
v\leftarrow v-\sum_\alpha t_\alpha(t_\alpha^Tv),
\]

or subtracting the arithmetic mean of each vector component.

### Rotation

- COM is reset at `0x57a8df`, then constructed by summing each coordinate divided by N (`0x57a910–0x57a943`). **Despite the name, this is the arithmetic coordinate centroid, not a mass-weighted center.** No masses are input to this routine.
- AXISATOM is formed from COM plus Cartesian unit offsets at `0x57ae96–0x57aed2`. ELF constants read at `0x4a43830` and `0x4a43730` are 1.0.
- At `0x57af82–0x57b034`, the explicit cross-product arithmetic uses `a=R_i-COM`, `b=R_i-AXISATOM_alpha` and forms `a cross b`. Since `AXISATOM_alpha=COM+e_alpha`, this equals `e_alpha cross (R_i-COM)`.
- The resulting vectors are written into ROTMODEX/Y/Z at `0x57b03f` onward.
- Each rotation norm is evaluated and compared with a small threshold; the X norm is square-rooted at `0x57b2c8`. Small-norm/zeroing branches exist. A constant read at `0x4a43840` is `1e-150`; this value must not be adopted as a recommended numerical rank cutoff.
- The final dot products and component subtraction are visible at `0x57e41d` onward and `0x57e5aa–0x57e63f` (source 1079–1080).

The observed structure is a normalized rotational-mode subtraction. No SVD, inertia-tensor eigensolve, or external linear-algebra call occurs within the routine; all non-runtime arithmetic is inline. Cross-axis Gram-Schmidt has **not** been established by this bounded reading. Exact zero-norm handling, normalization and whether hidden inline orthogonalization exists should be checked with a full-routine numerical oracle before claiming an exact algebraic projector.

For the clean mathematical implementation, the relevant rigid tangent span is nonetheless clearly identified:

\[
\mathcal R(R)=\operatorname{span}\{t_x,t_y,t_z,\ e_x\times q,\ e_y\times q,\ e_z\times q\},\quad q_i=R_i-\bar R.
\]

A rank-aware orthonormal basis Q for this span gives `P=I-QQ^T`. This is a project derivation/design option, **not a claim that the native code computes Q this way**. Linear, one-atom, and degenerate geometries have fewer than six independent rigid directions.

## What was not established

- No Kabsch coordinate alignment was identified on the inspected direct direction/climb path. `center_cart`, `findcenter`, `optim_mindist_mp_rotgeom_`, `remove_tranvec_`, and `symmol_fornewssw_mp_rotateframe_` exist in the ELF, but symbols alone do not prove they affect this walker.
- The ordinary BFGS driver inspected for Gaussian consumption has no direct call to `setconstraints`. Its optional branches, indirect initialization, and full upstream force dispatcher were not exhausted. No claim is made that *all optimizer forces* are cleaned by this same routine.
- This routine projects an input vector using the passed reference geometry; it is not an alignment operation that moves all coordinates. Coordinate gauge handling throughout biased local optimization still needs a consistent design in Python.
- Initial direction cleanup does not alone establish every Gaussian's subsequent force/geometry behavior. Whole rigid translations/rotations along the biased PES need separate observation.
- Native mass-weighted initial velocities and unweighted rigid cleanup are distinct operations; neither should be silently replaced by a mass-weighted metric without specifying the intended coordinate objective.
- This report provides implementation evidence, not proof that native never follows a rigid mode or that removing rigid components improves measured global-search efficiency.

## Development consequence

The primary next comparison should preserve a translation/rotation-invariant physical PES, use the same initial structure and random anchor, and measure internal displacement versus rigid displacement. Compare a rank-aware rigid tangent projection against the current independent implementation, while keeping rotation and Gaussian policies fixed. Force/Hessian-vector projections and modified-potential optimization must be geometrically consistent; simply projecting final output coordinates could conceal the cause rather than fix it.

Native-specific projection arithmetic can be validated separately without blocking the formal energy/force-consistent implementation. Keep physical endpoint checks and cost accounting; a cleaned direction is not by itself evidence of a new basin or improved search performance.
