# Fixed-cell Gaussian consumers: optimizer entry and restoration

Date: 2026-09-10. Static follow-up to [the complete addgaussian instruction oracle](native-addgaussian-instruction-oracle.md). Scope is the uploaded fixed-cell ELF, not all LASP releases or crystal/rigid-body kernels. The user-approved production implementation remains energy/force consistent; no search code, binary, expiry protection, or committed history was changed here.

## Finding

The immediate fixed-cell caller does **not** subtract the extra old-Gaussian force before passing energy and forces to its local optimizer. The returned `structure.fa` and `structure.energy` are passed directly to `bfgs_class_mp_bfgsdriver_`. The subsequent climbing convergence routine also reads the current force array. This strengthens the previous function-local observation: the inconsistent pair is an optimizer input on this statically resolved call path, not merely an unused scratch result.

This does not prove which full native runs reach a multi-Gaussian state, which optional optimizer transformations are enabled, whether every incoming force is the bare physical force, or the magnitude of any trajectory/search effect. No full native multi-Gaussian trajectory was instrumented in this task.

## Reproducible evidence

Artifact: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`.
Existing fixed-cell assembly: adjacent `analysis/kernel-ssw_fixlat_mp_climb_.asm` and `analysis/kernel-ssw_fixlat_mp_climb_convg_.asm`.
The following complete functions were additionally disassembled without running the binary:

```sh
objdump -dl -Mintel --disassemble=class_struc_mp_noncrystal_opt_ "$LASP_ELF"
objdump -dl -Mintel --disassemble=bfgs_class_mp_bfgsdriver_ "$LASP_ELF"
```

`LASP_ELF` denotes the above file. Temporary inspection outputs were `/tmp/pam-native-noncrystal-opt.asm` and `/tmp/pam-native-bfgsdriver.asm`; these are disposable, and the commands reproduce them. Type-bound tables were read directly from ELF PT_LOAD mappings, as little-endian 64-bit addresses, and matched to the existing symbol listing. No new emulation test count is claimed.

## Exact dispatch and arguments

| Site | Observed action |
|---|---|
| `climb:0x5caf55–0x5caf71` | Save incoming energy (`object+0x230`) into `tene0` (`+0x1ac8`). |
| `climb:0x5caf9c` | Call `addgaussian` through slot `+0x1e0`. |
| `climb:0x5cafa2–0x5cb02b` | Prepare descriptors and optimizer/control arguments; no intervening energy/force arithmetic or force-restoration call. |
| `climb:0x5cafb7` | Set `r8=0`, selecting the BFGS branch of `noncrystal_opt`. |
| `climb:0x5cb02b` | Call slot `+0x110`, resolved to `class_struc_mp_noncrystal_opt_`, entry `0x5a88c0`. |
| `noncrystal_opt:0x5a88e3–0x5a88e6` | Test `r8`; zero jumps directly to `0x5a91b9`. The force-norm/direct-displacement branch is skipped for this caller. |
| `noncrystal_opt:0x5a91bd` | Set `r8=&structure.energy` (`+0x230`). |
| `noncrystal_opt:0x5a91c4` | Set `rdx=structure.coordinates` data pointer (`+0x170`). |
| `noncrystal_opt:0x5a91ce` | Set `rcx=structure.fa` data pointer (`[r8-0x60]`, hence `+0x1d0`). |
| `noncrystal_opt:0x5a91d2` | Call optimizer table slot `+0x10`. |

The optimizer descriptor is formed from `object+0x1230` in `climb`, with table address `0x53cd060`. Reading `0x53cd060+0x10` gives `0x5b2970`, symbol `bfgs_class_mp_bfgsdriver_`. This resolves the second indirect call rather than assuming it is a generic optimizer from the function name.

Thus the immediate interface is equivalent to passing coordinates, returned modified forces, and returned modified energy to BFGS. Neither wrapper retrieves the saved `work1` force or `tene0` energy before this call.

## Inside the BFGS driver: what was established

On entry, the force pointer in `rcx` is saved to `r13` (`0x5b298a`), and the energy pointer in `r8` is saved at `[rbp-0x168]` (`0x5b298d`). On the fixed-coordinate branch, force entries are negated and multiplied by an optimizer scale before being stored in the optimizer gradient buffer; a scalar example is `0x5b2bd7–0x5b2bf4` (original `Class_bfgs.f90:122`). This conversion operates on the incoming total force. It is not a reconstruction of the old-Gaussian gradient.

At `0x5b41aa`, the saved energy pointer is reloaded into `rcx`; at `0x5b41b8`, the optimizer gradient-buffer pointer is loaded into `r8`; `0x5b41cd` calls `bfgs_basics_mp_lbfgs_` (`0x6e87b0`). This is the downstream E/gradient interface, not just the outer wrapper.

The driver contains additional optional scaling/projection, displacement and convergence branches. They were not all algebraically reconstructed here. In particular, this document does **not** claim that its internal gradient is always exactly minus the unscaled incoming force, or that the force array cannot be changed by optimizer processing. No old-Gaussian-specific compensating subtraction was established. A generic scale alone could not remove only one copy of the old terms while retaining the physical and latest-Gaussian force for arbitrary inputs.

## Convergence consumption and later restoration

After `noncrystal_opt` returns, `climb:0x5cb03d` calls `climb_convg`. There is no restoration call between these calls. `climb_convg:0x5cd187` loads `structure.fa`; `0x5cd1ca–0x5cd1f9` reads entries, takes their absolute values and accumulates a maximum. Hence this convergence statistic is computed from the current optimizer-side force array, not explicitly from the saved bare-force snapshot.

There **are** real snapshot/restoration operations later in `climb`, but their location matters:

- Before adding bias, the force array is copied to `work1` (`+0x9c8`); the copy block begins near `0x5ca9dc`, with a memcpy at `0x5caae8` and scalar/SIMD alternatives. Coordinates are separately saved to `work2`.
- On a later branch, `0x5cb8dd–0x5cb8e4` copies `tene0` back to structure energy.
- A later force restoration copies `work1` into `structure.fa`: at `0x5cbaca–0x5cbad7`, memcpy destination derives from `fa` and source from `work1`.
- `0x5cbf53–0x5cbf61` is another saved-energy restoration; `0x5cc209–0x5cc217` stores `tene0` in the next trajectory record, followed by the saved-force copy into that record.

These are full incoming/base-state restorations for later control/trajectory use. They are not a subtraction of a single old-Gaussian force contribution before the optimizer consumes the biased pair. Restoring a physical force after taking an optimization decision would not retroactively make that decision use a conservative biased potential.

The saved incoming state may itself include LS or other upstream enabled terms. Establishing that it is the bare physical PES in every mode requires the outer force-evaluation dispatcher; that has not been exhausted here.

## Implication and remaining test

The whole-function oracle established, for frozen Gaussian parameters,

\[
E_{out}=E_{in}+\sum_jB_j,\qquad
F_{out}=F_{in}+2\sum_{j<ng}F_j+F_{ng}.
\]

This audit establishes that the inspected caller passes those returned fields into BFGS before its later snapshot restoration. It therefore rules out the specific proposed explanation that **the immediate caller first removes the duplicate old term**.

For a full release-behavior claim, the remaining decisive check is a complete native fixed-cell run reaching `ng>=2`, with input/output at `addgaussian` and the optimizer E/gradient interface recorded at the same frozen geometry and history. Optional LS/constraints and upstream E/F provenance must be recorded, and the no-constraint case should be examined first. Full call-graph coverage, all modes, original source intent, and scientific consequences remain outside this bounded audit.

Production decision: retain analytically consistent Python Gaussian energy and force. Keep release arithmetic in an explicitly named reference/oracle path only. No new heuristic, parameter, or performance claim follows from this static finding.
