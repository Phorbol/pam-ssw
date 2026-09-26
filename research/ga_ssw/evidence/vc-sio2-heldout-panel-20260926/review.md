# SiO₂ held-out joint-VC optimizer panel: evidence review

## Outcome

All six planned arms completed with exit code 0 and no budget censor. Together they contain 18 attempted outer records: 17 returned landings and one ASE seed-83 `biased_quench_failed` record with no landing. All 23 independent fresh checks (six initial geometries and 17 landings) passed the configured local MACE OMAT force/stress certificate (`fmax ≤ 0.05 eV/Å`, maximum stress residual `≤ 0.001 eV/Å³`). There were no accepted proposals: all 17 returned landings were MC-rejected. Thus the force/stress certificate does not indicate an accepted or low-energy search result.

| Optimizer | Search requests | Fresh requests | Attempted outer records | Returned landings | No-landing failures | MC accepted |
|---|---:|---:|---:|---:|---:|---:|
| Safe-total | 7,174 | 8 | 6 | 6 | 0 | 0 |
| SciPy L-BFGS-B | 8,256 | 8 | 6 | 6 | 0 | 0 |
| ASE LBFGSLineSearch | 10,759 | 7 | 6 | 5 | 1 | 0 |

All returned landings were algorithm-qualified and fresh-valid. The no-landing ASE record is counted among six attempted outer records, not as a landing. Every landing had positive objective change relative to its arm's fresh-checked initial structure, from +8.84 to +16.75 eV; that is consistent with, but does not independently explain, the 17 MC rejections. Outer records ended as `gaussian_limit`, except the one ASE biased-quench failure.

Safe-total used fewer charged search requests in this short panel. The arm-normalized counts are about 1,196, 1,376, and 1,793 requests per attempted record for Safe-total, SciPy, and ASE respectively. This is an operational cost observation under each adapter's native stopping rules; those stopping rules and step-cap behavior differ, so it is not an equal-stopping-cost comparison or stable solver ranking. No lower-energy structure or accepted-chain move was observed. The different returned geometries remain candidate coverage evidence; MC rejection alone does not make them useless for PES exploration. The panel does not establish better global-minimum discovery or identify new physical phases.

## Endpoint geometry checks

The initial fresh-checked cells span 963.76–964.20 Å³. The 17 landing cells span 937.67–1,072.69 Å³. Across the initial structures, shortest periodic Si–O, O–O, and Si–Si distances are 1.6260–1.6265, 2.6404–2.6412, and 3.1191–3.1195 Å. Across landings they are 1.5421–1.6140, 2.2158–2.3738, and 2.2657–2.7661 Å, showing substantial local compression and cell-volume variation despite the force/stress certificates.

A descriptive coordination-cutoff sensitivity check gives the same warning without assigning bonds or phases. In each initial structure, every Si has four O neighbors and every O has two Si neighbors at a 1.7 Å cutoff. Across landing structures at that cutoff, Si counts range from 0–4 and O counts from 0–2; at 1.8 Å, Si counts range 3–5 and O counts 0–3. These are diagnostic distance bands, not chemical bond orders. No phase, quartz identity, global-minimum, or thermodynamic claim is made.

## Scope and provenance

The panel used the frozen 72-atom Si₂₄O₄₈ 2×2×2 repeat of the qualified COD 1011097-derived starting geometry, OMAT-small (`omat_pbe`), the same joint log-strain chart/configuration, seeds 71 and 83, three planned outer steps per arm, and caps of 6,000 search plus four fresh requests per arm. The report compares the operational adapters; it does not reproduce the paper's BKS model or establish phase ordering. The same 5 Å strain length was held fixed as an operational transfer and is not size-invariant.

GPU arms: 1502803 (index 0) and 1502823 (indices 1–5). Existing artifact analysis: CPU 1502824, 9 seconds, zero calculator calls. Added geometry audit: CPU 1503192, 4 seconds, zero calculator calls; it read the saved `fresh-checks.json` atoms and wrote 23 endpoint geometry rows. Cost reconciliation in the existing readout is exact: 26,189 charged search and 23 charged fresh requests, with zero ledger, stage, or summary discrepancies. Raw GPU artifacts remain in their original arm directories; compact panel analysis is archived in `archive-readout-1502824/`; geometry rows and their hashes are at `geometry-audit-1503192/`.

Decision: retain Safe-total as a candidate for joint-cell development and close this bounded panel without extending its budget or changing defaults. The evidence supports short, system-specific operational cost and numerical endpoint qualification; it does not establish general solver superiority. The next measurement improvement records first common-certificate passage separately from native termination, without changing either optimizer behavior or old results.
