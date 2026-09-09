> 原始报告位于 `/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/analysis/native-convergence-review.md`；下文的原资料相对路径按该外部研究目录解析。

# Uploaded LASP local-convergence audit

Audit date: 2026-09-09. Scope: original `GA-SSW_program/lasp` ELF; existing water run inputs/logs; static DWARF/objdump/GDB memory inspection. No new optimization, search, or singlepoint jobs were run by this audit.

## Definitive binary findings

1. **The requested force tolerance is parsed explicitly and quick-setting 1 does not override it.** `readsswpara_` reads `SSW.quick_setting` at 0x686b58, stores it at `ssw_parameters_mp_para_ + 0x108`, and calls `ssw_options_` at 0x686be3. The quick-setting-1 branch is 0x6c4571–0x6c4639 and does not write `para+0x2db28`. The explicit `SSW.ftol` real parser follows later at 0x687cd8–0x687d4e, destination `para+0x2db28`. GDB static string inspection confirms 0x4a30898 is `SSW.ftol`; its parser default at 0x4a49900 is 0.1. Existing `runs/water-singlepoint-probe/allkeys.log:69` records the actual requested value 0.0010, with quick-setting 1 at line 35.
2. **The fixed-cell force convergence test is a strict maximum-component test.** `ssw_fixlat_mp_allopt_judge_converg_` (0x5cf6f0) reduces absolute force components with `maxsd`, stores the result to `control+0x18`, then compares it strictly against `para+0x2db28` at 0x5cf976–0x5cf991. DWARF names these fields `maxf` and `ftol`. The force-converged flag is stored in `control+0xc0` (`opt`). This is not an RMS criterion or maximum per-atom vector norm.
3. **Termination is broader than convergence.** At 0x5cfaf1–0x5cfb0a, the return/stop flag ORs force convergence with `control.alloptstep > para.maxoptstep`. DWARF gives `alloptstep=control+0x74` and `maxoptstep=para+0x2ddd0`. Thus a step-limited local optimizer can return normally without `opt=true`. The parser at 0x689204–0x689273 reads `SSW.MaxOptstep`; existing singlepoint allkeys line 112 records 300.
4. **The top-level fixed-cell path also explicitly allows the cap.** `run_ssw_` reduces absolute coordinate components at 0x523225–0x5232f0 and exits the local loop at 0x5232f5 when either `maxabsforce < ftol` (0x5231f1–0x5231ff) or `NSTEP_LOCAL > MaxOptstep` (0x523205–0x52321a). Both branches join the same continuation. This independently confirms that the lower-level stop/convergence distinction matters to this executable's global driver, rather than only to an unused library function.
5. **Additional termination conditions exist.** In the fixed-cell judge, `control.bfgs_must_stop` (`+0x1b0`) forces termination at 0x5cfd9e–0x5cfda7. Enabled vapor checking can stop after step 50 at 0x5cfb64–0x5cfbc8. `multi_pes` (`control+0x1bc`) permits an intermediate transition at 5*ftol, but the final ordinary force test remains ftol; the multiplier 5 is stored at 0x4a45e10. `lts_extra` (`para+0x2dad8`) adds an extra conditional test; this audit does not establish it is enabled for water.

## Existing run evidence

All six existing `runs/water-original-complete-04/output/iterate/0/opt/{0..5}/lasp.out` local optimizations end their `Minimum found` row with `302`, followed by `SSW all done !`. For example, opt/0 line 43 reads:

```
Minimum found       0        0           -217.562180           -219.581467   C1     250.00   F    0.0000       0.217      0.00   0   302
```

The corresponding preserved `lasp.in` requests `ssw.sswsteps 1`, `SSW.quick_Setting 1`, `SSW.ftol 1E-3`; the wrapper set temperature 250 and output/printevery false. The repeated count 302 is strong evidence consistent with the 300-step cap and the strict `>` stopping branch, although this audit has not mapped every printed column or every counter increment. The `T/F` column must not be described as a convergence flag without proving its print argument.

## Interpretation and remaining uncertainty

Normal `Minimum found` / `SSW all done` output is **not a force-convergence certificate** in this binary. The high fresh forces reported by the independent archive audit cannot be dismissed as quick-setting silently relaxing ftol: static parser order, writes, and allkeys contradict that explanation. Step-capped local minimization is an evidence-supported candidate explanation; identifying the exact stop reason for every archive structure would require structure-to-local-run provenance and/or an instrumented local-optimization trace, which was outside this bounded audit. The report does not claim fresh forces are equal to native in-loop force values, and does not resolve precision, force/energy consistency, auxiliary core repulsion, or coordinate-rounding effects.

## Reproducible static artifacts

- `analysis/readsswpara-convergence.asm`: full `readsswpara_` disassembly.
- `analysis/ssw-options-convergence.asm`: full `ssw_options_` disassembly.
- `analysis/judge-convergence.asm`: full `ssw_fixlat_mp_allopt_judge_converg_` disassembly.
- `analysis/allopt-convergence.asm`: full `ssw_fixlat_mp_allopt_` disassembly.
- `analysis/run_ssw_.asm`: pre-existing driver disassembly.
- `analysis/selected-dwarf.txt`: pre-existing named field metadata, including `ftol` at lines 495–500 and `maxoptstep` at lines 1017–1022.

Commands: `objdump -d --disassemble=SYMBOL GA-SSW_program/lasp`; `gdb -batch -ex "x/s 0x4a30898" -ex "x/gf 0x4a49900" GA-SSW_program/lasp`. GDB was used without running the ELF.
