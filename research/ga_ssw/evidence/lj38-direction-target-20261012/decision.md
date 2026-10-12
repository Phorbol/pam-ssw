# LJ38 direction-transfer decision (1755511)

**Decision.** Retain full-direction `per_atom` as a promising candidate for
further hard-case evaluation, but do not promote it to a default or claim a
general success-rate improvement. On these two reused starts full direction found the
global-minimum structure only on seed 2, which passed the frozen
1 meV joint target criterion. Rotation-only found the global-minimum
structure only on seed 1 without passing that energy window; on seed 2 it remained
fragmented throughout its saved minima. These outcomes separate structural
discovery from target-energy success and show that intact-cluster search and
fragmented search are materially different cases.

## Paired outcomes

The experiment compared the unchanged Oct. 7 rotation arm with full direction
using `per_atom`, the same archived raw initial structure per seed, and the
same `FullPairLJ(epsilon=1 eV, sigma=2.7 Å)` potential. The joint target remains
the Cambridge LJ38 structure at -173.928427 eV, energy window +0.001 eV,
converged force no greater than 0.05 eV/Å, connected 1.3 sigma geometry,
proper-rotation/permutation match, and independent fresh validation.

| Seed | Arm | First structural GM match (paid search requests) | Best / target energy (eV) | Frozen joint result |
|---:|---|---:|---:|---|
| 25092501 | Rotation | 466,001 | -173.927214550; 1.212 meV above reference | No hit; outside 1 meV window |
| 25092501 | Full `per_atom` | None through 800,000 | -173.252211355; matches qualified second-minimum geometry | No hit |
| 25092502 | Rotation | None; all 2,001 saved initial/landing minima fragmented | -117.425656278; best is fragmented | No hit |
| 25092502 | Full `per_atom` | 629,409 | -173.927640175; 0.787 meV above reference | Joint hit; fresh validation passed |

The seed 1 rotation structure at request 466,001 is force-qualified and
connected, and matches the GM with proper-rotation RMS 0.00683 Å. Its energy
is 2.040 meV above the tabulated target, so this is a structural-only hit,
not a joint success. Its lowest-energy saved structure also matches the GM (RMS 0.00414 Å) but
remains 1.212 meV above target after fresh evaluation. Conversely, seed 1
full-direction reaches the independently qualified LJ38 second-lowest
minimum by request 2,714 (RMS 0.00158 Å) and its lowest-energy saved structure matches that reference; that is not a GM
match. Seed 2 full-direction matches the GM at its final request, with
force 0.02438 eV/Å and RMS 0.00433 Å; the fresh candidate check passed.

Seed 2 rotation has zero single-component minima among the initial structure
and all 2,000 saved landing minima: every saved record is fragmented under the
same 1.3 sigma graph. Its best structure has component sizes 19+11+4+4. Thus
its failure cannot be summarized as merely missing a lower energy within an
intact-cluster trajectory; that trajectory did not produce an intact saved
minimum. The evidence does not identify why fragmentation persisted or prove
that the direction treatment alone caused the paired difference.

## Resource and execution accounting

The search ledger is the primary request count; `actual_calculator_calls` is
counted separately because a charged request need not call the calculator.
The seven final fresh E/F requests are separate from each search cap.

| Seed / arm | Search requests charged | Search calculator calls | Fresh requests / calls | Denied requests | Charged failures | Outer callbacks | Wall (s) | Boundary / status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 25092501 rotation | 800,000 | 762,625 | 2 / 2 | 1 | 0 | 900 | 1,433.47 | 800k request cap; `evaluation_failed` after next request was denied |
| 25092501 full `per_atom` | 800,000 | 750,941 | 2 / 2 | 1 | 0 | 1,267 | 1,494.39 | 800k request cap; `evaluation_failed` after next request was denied |
| 25092502 rotation | 678,674 | 610,766 | 2 / 2 | 0 | 0 | 2,000 | 1,216.58 | 2,000 outer-step cap; `completed` |
| 25092502 full `per_atom` | 629,409 | 590,900 | 1 / 1 | 0 | 0 | 1,026 | 1,126.73 | Target callback stopped after `first_hit` |

Across the four searches: 2,908,083 charged search requests and 2,715,232
actual search calculator calls. Add seven charged fresh requests/calls for
2,908,090 requests and 2,715,239 actual E/F calls overall. The two extra
request-cap denials were uncharged and occurred before calculator evaluation;
there were no charged failed requests or calculator errors. The two
`evaluation_failed` status fields are therefore budget-boundary outcomes, not
PES failures. Per-run logs and ledgers agree with the summaries. Seed 1 was
request-censored before the 2,000-step cap; seed 2 rotation reached its step
cap; seed 2 full direction stopped at its target hit. No run was retried or
extended.

## Interpretation and boundary

This paired development result supports a narrow conclusion: under this
frozen configuration, full direction reached the joint LJ38 target on seed 2
and the second-minimum endpoint on seed 1, while rotation reached the GM
geometry but missed the energy window on seed 1 and remained fragmented on
seed 2. Structural matches are deliberately reported separately from the
predeclared joint criterion. The second-minimum match is endpoint identity
only; it does not establish a transition path or funnel residence.

There are only two reused starting structures. The result is not an estimate
of general success probability, does not resolve whether the energy-window
near miss reflects convergence precision or the final minimum actually
reached, and does not justify changing defaults. Keep the paired failures,
the structural-only rotation discovery, the SLM endpoint, and the fragmented
rotation trajectory together as counterevidence and follow-up inputs.

## Evidence

- Frozen settings and acceptance criteria: [`protocol.md`](protocol.md).
- Paired summary: [`paired-readout-1755511.json`](paired-readout-1755511.json).
- Structural readout for seed 1: [`output-1755903/readout.json`](structural-readout-v1/output-1755903/readout.json).
- Structural readout for seed 2: [`output-1756096/readout.json`](structural-readout-v1/output-1756096/readout.json).
- Raw run artifacts: `runs/run-1755511-{0,1,2,3}/` (`result.json`, `progress.jsonl`,
  `search-ef-ledger.jsonl`, `fresh-ef-ledger.jsonl`, and
  `outer-structures.extxyz`).
- Structural method: existing `analyze_panel.compare_geometry`, 1,000 mapping
  cap and proper-rotation RMS tolerance 0.081 Å; connected components use the
  frozen 1.3 sigma graph. All structural comparisons reported here were
  conclusive; no target mapping-cap cases occurred.
