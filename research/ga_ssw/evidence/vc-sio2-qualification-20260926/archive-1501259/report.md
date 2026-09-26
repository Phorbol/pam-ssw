# COD alpha-quartz single-quench qualification — job 1501259

The one-V100 job completed with exit code 0 in 38 seconds. It ran the existing
Safe-total all-degree-of-freedom `cell_quench` once on the 9-atom COD 1011097
input, using MACE-OMAT-0-small (`omat_pbe`, float64), zero pressure,
`fmax=0.05 eV/Å`, stress residual `0.001 eV/Å³`, strain length 5 Å, maximum
step 0.2 Å, 300 iterations and history 500. No SSW proposals, MC selection,
second optimizer, or phase comparison were run.

The CIF expands to Si₃O₆ in P3₁21 with PBC in all directions. Before
optimization, the OMAT-small evaluation gave E = −70.91762893 eV,
V = 112.96385387 Å³, fmax = 0.35135236 eV/Å and maximum absolute stress
component = 0.03247510 eV/Å³; the input fails both frozen physical thresholds.
After the cell quench, E = −71.00637291 eV, V = 120.52493426 Å³,
fmax = 0.01071637 eV/Å and stress maximum = 0.00085509 eV/Å³. The independent
fresh evaluation reproduces and passes both criteria. Composition remains
Si₃O₆; the cell changes from (a,b,c) = (4.913, 4.913, 5.404) Å to
(5.040255, 5.040255, 5.478242) Å, with volume increasing by 7.56108 Å³
(6.69%). The minimum periodic distances change as follows: Si–O
1.613974→1.626531 Å, Si–Si 3.060702→3.119466 Å, and O–O
2.625052→2.640601 Å. These are geometric readouts, not phase assignments.

The counted initial evaluation plus quench used 16 E/F/stress requests; two
independent fresh checks brought the total to 18. No cap or wall censor
occurred. Raw CIF, input and relaxed structures, full optimizer trace,
effective model/configuration, search and fresh ledgers, and certificates are
in this run directory. Model SHA-256 is
`0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`; worker
SHA-256 and source provenance are recorded in `results/effective-plan.json`.

This qualifies this specific small COD geometry for the stated OMAT-small
numerical certificate after a joint cell/atomic quench. It does not establish
the identity or stability of the relaxed phase, SiO₂ phase ordering, the 2014
paper's BKS results, optimizer ranking, or SSW search quality. The initial
structure's failure is retained; the qualified endpoint should not be
described as an independently verified global minimum.
