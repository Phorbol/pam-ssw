# Reference qualification (not discovery)

Five published target structures now have executable qualification under the intended potentials. CPU1487995 completed in2s; CPU1488032 produced both Morse results. Source, scripts, raw point files and JSON results are retained here.

| Case | Independent energy / epsilon | Max atom force eV/Angstrom | Qualification |
|---|---:|---:|---|
| LJ38 | -173.928426512 | 0.000338 | source coordinates |
| LJ55 | -279.248470308 | 0.002444 | source coordinates |
| LJ75 | -397.492330681 | 0.000724 | source coordinates |
| Morse29 rho14 | -102.774589204 | 0.00000356 | rho6 shape reoptimized at rho14,9 optimizer steps |
| Morse80 rho14 | -340.811370660 | 0.0000650 | rho6 shape reoptimized at rho14,28 optimizer steps |

All energies agree within1e-5epsilon with the applicable Cambridge target. The LJ55 value printed in SSW2013 Table1 (-297.24847) was visually checked on PDF p1843 and disagrees with both the public table and independent full-pair evaluation. It is treated as a source error, not a target our search must reach.

For Morse, ASE MorsePotential was used with inner/outer cutoff100/101 r0, epsilon1,r0=2.7Angstrom (rho0=14). Actual diameters stayed below12.38Angstrom throughout both optimizations, far below270Angstrom; therefore no pair entered the switching region. This verifies the full formula on these references, not a blanket claim about ASE default cutoff equivalence. Exact E/F cost for these reference reoptimizations was not instrumented; optimizer steps are not force calls. No basin search or target-discovery success is claimed.

Published SSW random initialization is not sufficiently specified to reproduce its success-rate distribution exactly. BH1997 offers a sphere radius5.5sigma and T*=0.8, but also uses a container and auxiliary moves; an ASE or independent plain-BH run is not automatically that full published algorithm. We will report operational choices separately and compare on actual E/F cost, preserving failures.

## Runner and geometry checks

CPU1488209 exit0 in5s: known LJ55 reference stops at initial quench (1 search+2fresh); known LJ38 SLM exercises two completed attempt records and a1000-request boundary (1000search+1fresh). Both costs equal checkpoint accounting. The latter returns evaluation_failed at the deliberately imposed cap; this is expected containment, not an algorithm success. Neither is random-search evidence. Geometry preflight1488122: shuffled/rotated LJ38 and55 match; known LJ38 SLM does not. Earlier1488078 checked the two positive cases then failed because the old SLM artifact lives in the parent worktree, not this checkout; the absolute source path fixes that input lookup, and the failed log remains.
