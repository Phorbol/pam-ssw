# Periodic rotation-priority comparison: offline readout

This is an audit of the four archived seed-41 arms, not new independent validation. The two inputs are previously used; this run is a prospective strategy discriminator. Geometric identity is an initial-versus-best pairwise match only and does not establish basin or phase identity.

| Case | Method | Run / audit | Requests: ledger / result | Init E (eV) | Best ΔE (eV) | Records / landings / minima | Rotation requests | Bias-quench requests | True-quench requests | Initial fresh | Best fresh | Initial~best tight / broad |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| aloh3 | ritz | evaluation_failed / consistent | 12000 / 12000 | -177.3293050620191 | -0.23812879174374757 | 14 / 13 / 14 | 8451 | 2478 | 433 | True | True | False / False |
| aloh3 | recovered | evaluation_failed / consistent | 12000 / 12000 | -177.32930506201916 | -0.23808183508560887 | 23 / 22 / 23 | 3593 | 6405 | 899 | True | True | False / False |
| brookite48 | ritz | evaluation_failed / consistent | 12000 / 12000 | -427.59036474751946 | -0.0009667382371958411 | 17 / 16 / 17 | 8943 | 1855 | 338 | True | True | True / True |
| brookite48 | recovered | evaluation_failed / consistent | 12000 / 12000 | -427.59036474751946 | -0.0009307792797130787 | 30 / 29 / 30 | 3914 | 5551 | 1077 | True | True | True / True |

Rotation requests use method-specific archived diagnostics: Ritz PreRot/main force-call counts; recovered-CBD per-stage trace increments. Bias-quench and true landing-quench requests are reported separately. Missing fields remain unallocated; outer record/request-ledger totals are authoritative. Search failures are counted separately from successful requests, and denials are not counted as requests.

A missing geometry file, a false initial~best match, or an incomplete fresh check is not evidence of a new basin. Fresh numerical qualification checks the archived recalculation energy error, force threshold, fixed cell, PBC, composition, convergence, and finiteness; it does not establish physical stability.

Expected arms: 4; top-level summary rows: 4; analyzed arms: 4.
Analysis state: complete; cost anomaly arms: 2.
Pymatgen 2026.5.4; analysis elapsed 10.8 s; calculator/PES calls: zero.

## Pairwise common-cost prefixes

| Case | Common requests | Policy | Prefix requests | Records | Failed records | Converged landings | Best qualified E (eV) | Initial included |
|---|---:|---|---:|---:|---:|---:|---:|---|
| aloh3 | 12000 | ritz | 12000 | 14 | 1 | 13 | -177.56743385376285 | True |
| aloh3 | 12000 | recovered | 12000 | 23 | 1 | 22 | -177.56738689710477 | True |
| brookite48 | 12000 | ritz | 12000 | 17 | 1 | 16 | -427.59133148575665 | True |
| brookite48 | 12000 | recovered | 12000 | 30 | 1 | 29 | -427.5912955267992 | True |
A record that would cross the common-budget boundary is excluded whole; its required cost is retained in `excluded_straddling_record`. Failed records fully inside the prefix remain in request and failure counts but contribute no landing energy. This is a cost-prefix comparison, not independent validation.
