# C4H4 + H2 saved-minimum audit

This is a zero-PES re-read of the existing six-arm, 400-step MH-1 campaign, not
new search evidence. It counts common-prefix minima (first 200,000 charged
requests) whose saved component formulas are exactly `C4H4` and `H2`, then
requires convergence, matching fresh-check identity, and the saved fresh
numerical-qualification flag. The script is `readout.py`; its result and
aggregate-source SHA-256 are in `readout.json`. It reuses the archived graph
builder and H/C distance cutoff from `analyze_c4h6_ls_reaction_coverage.py`,
and checks the saved per-arm `result.json` coordinates for these selected
minima. Reproduce with:

```bash
python research/ga_ssw/evidence/c4h6-fragment-source-audit-20261012/readout.py \
  --output research/ga_ssw/evidence/c4h6-fragment-source-audit-20261012/readout.json
```

| Arm | Seed | Formula observations / force-qualified | Distinct classes | First request per class |
|---|---:|---:|---:|---|
| SSW | 61 | 0 / 0 | 0 | — |
| Paper LS | 61 | 4 / 4 | 2 | 48,214; 101,153 |
| Native LS | 61 | 2 / 2 | 2 | 37,613; 129,910 |
| SSW | 67 | 1 / 1 | 1 | 191,291 |
| Paper LS | 67 | 4 / 4 | 4 | 77,198; 121,627; 142,970; 170,155 |
| Native LS | 67 | 2 / 2 | 2 | 88,469; 107,799 |

Among the 13 force-qualified observations, exact element-labeled graph
isomorphism to the paper-described G or H connectivity was found for four:
Paper LS seed67 has one H candidate (minimum 191, request 121,627) and one G
candidate (minimum 266, request 170,155); Native LS seed61 has one G candidate
(minimum 55, request 37,613) and seed67 one G candidate (minimum 136, request
88,469). Paper LS seed61 and both SSW arms have no G/H connectivity match in
this prefix. These are connectivity-compatible saved structures under the old
geometric cutoff, not confirmed chemical identities.

The tested G template is a four-carbon chain with the paper connectivity
assignment H counts `[2, 1, 0, 1]` along the chain plus a disconnected H-H
edge. The tested H template is a three-carbon ring with one exocyclic carbon,
one H on each of the two non-substituted ring vertices, two H on the exocyclic
carbon, and the same disconnected H-H edge. Both templates are compared as
element-labeled undirected graphs; multiple-bond order is intentionally absent.

Class IDs are assigned in one graph-isomorphism map shared across all six arms,
not independently per arm. Repeated IDs across arms therefore denote the same
saved graph class under the existing classifier; within-arm distinct-class
counts above should not be summed as globally unique structures. The old
campaign report already separates connected and fragmented classes, so these
counts do not revise its connected-isomer coverage result.

The result supports only that the saved MLIP minima have graph components
labelled `C4H4` plus `H2`, pass the existing fresh force/numerical check, and
that four meet the element-labeled G/H connectivity templates under the old
geometric cutoff. This cannot establish bond orders, spin/radical state,
chemical stability, reaction barriers, or PBE identity/energies. Formula,
connectivity, and force qualification must not be reported as reproducing the
paper products. No PES calls, trajectory writes, or changes to the original
connected-class conclusions were made.

Sources: the input `analysis.json` is in
`/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/`;
the per-arm `result.json` files in the same directory supply saved coordinates;
the original campaign summary is
`research/ga_ssw/evidence/c4h6-mh1-coverage-20260924/report.md`. The paper
discussion of G/H is in `ga-ssw-20260909/literature/215.txt`, Fig. 5 and §3.1;
it describes G/H as dissociated products involving H2. This graph-composition
audit does not establish identity with them.
