# Native VC evidence and the independent log-strain implementation

> 2026-09-10 更新：用户提供的 http://lasphub.com/publication/87.pdf 已成功取得并核验为12页正式排版全文（2,477,395 bytes），存于 `literature/benchmark-sources/vc2014/`。以下此前获取失败的记录保留作历史；正文缺口已解决，SI仍未取得。全文揭示2014为CBD-cell/atomic-SSW分块耦合，见 `vc2014-native-crosscheck.md`；当前joint log-strain方法须保留独立扩展标识。

2026-09-10. Bounded static review only: existing driver audit plus two crystal symbols. No native execution, new oracle, broad decompilation or parameter transfer was performed. The independent log-strain chart is mathematically specified, but **native coordinate/metric/force parity has not been established**.

## Evidence scope and provenance

Binary: `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp`. Existing symbol table: sibling `analysis/lasp-symbols.txt`. Existing driver evidence: [SSW kernel comparison, section 5](ga-ssw-audit/ssw-kernel-comparison.md), backed by `analysis/run_ssw_.asm`.

Only these crystal routines were newly inspected with `objdump -dl -Mintel --disassemble=<symbol> <binary>`:

- `ssw_crystal_basic_mp_update_forcepara_`, entry `0x5e3f80`; DWARF maps the reviewed blocks to `Class_ssw_crystal.F90:383–396` and later lines. Local review output `/tmp/pam-vc-updateforce-review.asm` is reproducible with the command above.
- `ssw_crystal_basic_mp_moveds_`, entry `0x5eccf0`; reviewed selection/update blocks map to source lines 1028–1030 and 1124–1125. Local output `/tmp/pam-vc-moveds-review.asm`.

These line mappings are compiler debug metadata, not possession of the original Fortran source. Unresolved object offsets and indirect dispatch targets remain unresolved here.

## What is actually known

The existing driver audit identifies `run_ssw_:0x530822` calling fixed-cell `ssw_fixlat::ssw_move` and `0x535ddf` calling a separate `ssw_crystal_basic::ssw_move`. This is positive evidence of a distinct crystal walker path, not merely cell relaxation added to the fixed-cell endpoint.

The uploaded `GA-SSW_examples_run/global_exploration/input-templates/TYPE1-AlOH/lasp.in` specifies Run_Type 15, `SSW.NG=10`, `SSW.ds_atom=0.6`, `SSW.NG_cell=7`, `SSW.ds_cell=1.6`, `SSW.ftol=0.05`, `SSW.strtol=0.05`, T=400 and a supplied AlOH NN potential. It also enables periodic MD calls. These are **one example's input values**, not recovered universal defaults. NG_cell is a separate input; neither its exact scheduling meaning nor ds_cell's coordinate normalization follows just from the label. Do not copy `strtol=0.05` into ASE eV/Å³ units without closing its conversion and convergence consumer.

The force-update routine contains a diagnostic conversion involving the named global `constants_ssw_mp_eva3togpa_`: `0x5e4549` obtains its address, `0x5e458c` multiplies by it, and `0x5e4592` divides by another scalar before storing at `control+0x28`. The preceding block adds three object scalar entries and a parameter-dependent term, then takes an absolute value. This confirms an explicit pressure/stress-unit conversion in this routine. It does **not** establish that the optimizer consumes stress in GPa, that these three entries have already been mapped to diagonal stress, or that the metric-conjugate cell force is simply `-stress`. The producer and consumer of those fields must still be closed before giving that formula.

At `0x5e4b5c–0x5e4c03` the routine explicitly forms nine pairwise scalar differences, sums their squares and takes a square root (source line 396). The symbol table also contains local `LENGTH` and `CELLL` scalars, updated at `0x5e5058` and `0x5e5060`. Thus a nine-component difference norm and separate tracked lengths exist. Without the complete object-field mapping, this is **not enough** to call it a nine-dimensional cell optimization metric, a nine-independent-DOF cell, or the native equivalent of our L. A 3×3 tensor diagnostic can coexist with six independent strain degrees of freedom.

The displacement routine selects a scalar from parameter offsets `+0x2db18` or `+0x2db08` at `0x5ece5f–0x5ece71`, depending on a branch, and has a flagged multiplication/clamp at `0x5ece79–0x5ece9e`. Later arithmetic at `0x5edd81–0x5eddaa` is an additive array update of the form `old + scalar * direction`; indirect method calls follow at `0x5edea3`. This is evidence of branch-dependent displacement controls and an additive internal update. The fields and the subsequent coordinate conversion have not been identified here. Consequently it does not prove an additive lattice update in physical Cartesian cell vectors, or rule out an internal strain representation. In particular neither selected offset is labeled ds_cell on this evidence alone.

## Comparison to our explicit contract

Our [log-strain chart](vc-logstrain-chart.md) defines, in ASE row conventions,

```
S = sum_a s_a B_a, F = exp(S), H = H0 F, R = X F,
q = (vec(X), L*s), Phi = E(R,H) + p det(H).
```

There are six orthonormal symmetric strain coordinates, no cell rotation coordinate, positive determinant by construction for a valid initial cell, and an explicitly supplied positive L in Å. The physical ASE tensile-positive stress gives `G_F = F^(-T) V (sigma+p I)`, followed by the adjoint Frechet derivative of exp and division by L. Atomic forces, strain derivatives, bias and finite differences use the same coordinate map. Our full force/stress certificates are physical observables and do not depend on L.

| Issue | Independent implementation/design | Native evidence boundary |
|---|---|---|
| Finite cell update | Symmetric matrix exponential from one fixed chart | Additive internal array update observed; physical conversion unresolved |
| Atom/cell metric | Explicit L in `q`; benchmark L sensitivity is our prospective convention | Native scale and size dependence not recovered; LENGTH/CELLL names are insufficient |
| Strain dimensionality | Six symmetric orthonormal components | Nine-component diagnostic exists; independent native cell DOFs unresolved |
| Stress convention | ASE tensile-positive eV/Å³, exact E+pV derivative | GPa conversion observed in a diagnostic; full optimizer sign/units/volume/lattice transformation unresolved |
| Separate cell moves | Joint atom/strain proposal as explicitly configured | Example has NG_cell and ds_cell, with unresolved scheduling/normalization |
| Convergence | Fresh physical force and full allowed stress check | Example strtol and diagnostic fields are not a fully recovered stopping predicate |

These are established design choices versus native unknowns, **not all proven implementation differences**. We can already say that our explicit L sensitivity and logarithmic-chart contract are independently chosen and have no native parity evidence. We cannot yet say that native uses linear strain, nine free lattice components, no metric, or an inconsistent cell force.

## Narrow questions for the original VC paper/SI and later parity work

The remaining primary source is 2014 VC-SSW, DOI [10.1039/c4cp01485e](https://doi.org/10.1039/c4cp01485e). Retrieval attempts and the still-missing full paper/129 KB SI are recorded in [benchmark literature](ssw-benchmark-literature.md). Its contents must be read before attributing any of the following answers to the authors:

1. What generalized atomic/cell coordinates and metric are defined; what finite update and lattice-rotation treatment are used; how do cell and atomic step scales depend on N/volume?
2. What cell force is conjugate to those coordinates, including pressure, volume, lattice inverse/transposes, virial convention and stress units? Does the paper define the same map used in this release?
3. What exactly do NG_cell and ds_cell control: mode selection, alternating segments, Gaussian budget, or displacement? How are cell/atom Gaussian widths and convergence criteria coupled?

If parity is later required, the smallest useful next binary task is resolving the object fields/indirect conversion and stress-force consumer for these two routines, then one original-instruction coordinate/gradient oracle. It should not precede the current independent coordinate-consistency checks merely to imitate unclosed arithmetic.

**Mainline decision:** keep the independently derived log-strain/L implementation explicit and experimental; test its exact gradients and real-system behavior. Do not identify ds_cell=1.6 with L=1.6 Å, copy the example's pressure tolerance, or insert unknown native displacement adaptations. Native parity gaps remain separate from mathematical correctness and from joint-VC scientific effectiveness.
