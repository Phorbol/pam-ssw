# Fe7C3-80 LS Fe–Fe filter comparison

This report audits job `1252915` in
`research/ga_ssw/evidence/fe7c3-80-ls-filter-comparison/`.  It compares the
paper/SI-motivated independent LS response with all-pair response on the same
Fe56C24 periodic input and seeds 7 and 101.  The archived no-LS joint arms from
job `1252472` remain a separate baseline.  No new PES calls were made for this
report.

## Accounting and outcomes

The requested denominator is eight outer attempts: two seeds × two LS modes ×
two steps.  All eight entered and were charged at least one search evaluation.
The filter arms were censored at the 1997-search-request cap; the all-pair arms
ended after their two inner maxiter failures.

| arm | seed | search requests | fresh checks | total requests | outer outcomes | valid proposals |
|---|---:|---:|---:|---:|---|---:|
| LS all-pair | 101 | 838 | 1 | 839 | maxiter, maxiter | 0 |
| LS all-pair | 7 | 666 | 1 | 667 | maxiter, maxiter | 0 |
| LS Fe–Fe filter | 101 | 1997 | 1 | 1998 | maxiter, budget exhausted | 0 |
| LS Fe–Fe filter | 7 | 1997 | 1 | 1998 | maxiter, budget exhausted | 0 |
| **total** | — | **5498** | **4** | **5502** | **6 maxiter, 2 budget** | **0** |

The four fresh checks are the common initial structure checks; no candidate
landing was produced that required a separate fresh endpoint check.  The raw
per-arm `evaluations.jsonl` ledgers reconcile exactly with the reported search
requests.  The job completed normally; the two censored arms are scientific
budget outcomes, not scheduler failures.  The 46 LS pre-quench calls are
included in these physical request totals: 20 for each all-pair arm and 18 for
each filtered arm.

For each outer attempt one primary outcome was assigned from terminal strings.
This avoids recursive double counting of nested `evaluation_failed` records:
there were six maxiter outcomes and two budget-exhausted outcomes, with no
backend failure.  Every attempt's LS response record is present.  The LS
response diagnostics were:

| mode | `bond_count` | pre-quench calls/attempt | energy response values (eV/atom) |
|---|---:|---:|---|
| all-pair | 456 | 10 | 0.02543693499, 0.02426703113 |
| Fe–Fe filter | 456 | 9 | 0.03566314059, 0.03112870315 |

## Filter contract

The saved filtered softening objects contain 456 pair entries: 144 C–Fe and
312 Fe–Fe.  Exactly those 312 Fe–Fe strengths are zero under
`energy_filter=[[[26,26],0.0]]`; the 144 C–Fe strengths remain nonzero.  The
all-pair objects contain the same 456 pair entries with no zero strengths.
Thus the requested filter persisted through both filtered attempts and did not
silently remove the pair list or change its bond-count denominator.

This is an independent paper-response implementation using the recovered pair
lookup and explicit filter.  It is not a claim of native LASP LS normalization
or chemical parameter validation.  The response difference and the different
number of pre-quench calls demonstrate that filtering changes the LS response
path, while the present four-arm run produced no valid search landing from
which to assess search quality.

The archived no-LS joint arms provide context only: both used 1997 search
requests (one per seed), produced no valid proposal, and had their own nested
budget/evaluation outcomes.  Their costs and provenance remain in
[fe7c3-80-vc-comparison.md](fe7c3-80-vc-comparison.md); they are not merged into
the 5498-request LS-filter denominator.

## Limits

This is one Fe56C24-derived input, two seeds, one approximate MACE finite-cell
backend and a bounded two-step comparison.  The zero Fe–Fe response is an
interface property of the registered filter object, not evidence that Fe–Fe
interactions should be removed in a physical Fe7C3 model.  The run establishes
no efficiency, basin, phase, magnetic, or chemical-stability conclusion.
The reproducible aggregate is
`research/ga_ssw/evidence/fe7c3-80-ls-filter-comparison/comparison/summary.json`,
with its summarizer alongside it.
