# Native instruction recovery progress

Scope: independent Python SSW family from uploaded executable, primary papers and
PAM comparison. This increment closes selected instruction-level evidence gaps;
it is not a full walker or real-system efficacy result.

## Weight update: beyond static reconstruction

`probe_native_weight_emulated.py` loads the inspected ELF PT_LOAD segments into
Unicorn 2.1.4, prepares Fortran by-reference operands, and executes the original
`set_thisgaussw` instructions at 0x6e1730. Only external acos is replaced with
host math.acos. No ptrace, LASP process, calculator or HPC job is used.
Implementation API checked against https://www.unicorn-engine.org/docs/tutorial.html.

54/54 cases agree with Python within absolute 1e-10 for weight, energy, force,
angle and update count. Cases include scalar/vector loop lengths (1,2,3,5,15,45
atoms), 86.999/87/87.001 degree boundaries, entry above maxw, post-update maxw
precedence and growing scales. Synthetic operands verify numerical behavior,
not physical usefulness. Original ELF identity, full inputs and outputs are in
`research/ga_ssw/evidence/native-weight-emulated/result.json`; a reference-free
regression replays these outputs. This replaces the earlier 'static only' status
for this function, not the unknown full caller state.

Reproduce (Unicorn is optional and not a production package dependency):

```bash
PYTHONPATH=/tmp/pam-ssw-unicorn-probe:. python research/ga_ssw/probe_native_weight_emulated.py --elf /home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp --output /tmp/native-weight-emulated.json
```

## MC and rotation

`native_mc.py` implements the recovered NSAME recurrence and fixed delta/20
exponent; see `native-mc-contract.md`. 224 comparisons execute relocated original
MC and integer-power instructions with explicit RNG and host exp. This is distinct
from the weight emulator. Python is binary-independent; the optional oracle is not.

`native-rotation-followup.md` recovers FACT1 retries, 40-degree cap, first-iteration
Broyden reinitialization and returning the evaluated direction. The unresolved
BRZERO4 update and force/constraint semantics still prevent claiming a complete
native rotation implementation.

## Paper, executable and PAM must stay distinguishable

Primary BP-CBD (2012), DOI 10.1021/ct300250h, and SSW (2013), DOI
10.1021/ct301010b, are available locally as literature/65.txt and 74.txt in the
external research archive. The author publication listing was checked again:
https://chemistry.fudan.edu.cn/chemen/wwwwwwwheng/list.htm.
The SSW paper describes ordinary Metropolis in step 7. The ELF's fixed factor20
has not been explained by this literature; it is not an inferred atom count or
first-principles recommendation. Web search did not establish its rationale.

Current PAM `_update_metropolis_chain` rejects non-new archive candidates and
uses exp(-delta/metropolis_temperature), where temperature is an energy scale.
Paper-reference Python uses temperature in K and ASE kB, while recovered ELF uses
NSAME and the factor20. These are three different acceptance policies. A same-
nominal-temperature comparison would confound kernel and selection differences.

Likewise PAM curvature-based Gaussian heights, paper forward-force heights and
ELF angle-based heights must be separately named. The recovered constants remain
compatibility behavior; none has earned a universal default through this work.

## Next implementation dependency

Complete BRZERO4 and the native addgaussian/climb caller-state contract before
connecting a purported native walker. Then recover enclosing make_decision
filters/overrides and preserve NSAME in restart state. Reuse ASE E/F and explicit
failure accounting, not PAM heuristics as substitutes for unresolved native code.
After one-escape matching on the same real PES, compare independent native-rule,
paper-rule and PAM strategies at matched E/F cost and properly matched acceptance
scales. No new real-system E2E was performed in this increment.
