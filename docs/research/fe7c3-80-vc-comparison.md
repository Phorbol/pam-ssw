# Fe7C3-80 PQC/joint VC comparison

This report summarizes the completed four-arm, two-seed comparison in
`research/ga_ssw/evidence/fe7c3-80-pqc-joint-comparison/`.  It is an
MACE-OMAT finite-cell experiment on the qualified Fe14C6-derived `(2,2,1)`
80-atom input, not native VC parity or a chemical phase benchmark.

## Cost and denominator

There were exactly 8 requested outer attempts: two seeds (`7`, `101`) for each
of `pqc` (posterior cell quench) and `joint` (joint strain escape).  All 8
attempts entered the controller; 7 received at least one paid search request,
and one entered attempt had zero paid requests after the shared search budget
was exhausted. Thus the true not-started count is zero; the latter is reported
separately as an uncharged entered attempt.  The initial qualification record is excluded from this outer
attempt denominator.

| quantity | total |
|---|---:|
| requested attempts | 8 |
| entered attempts | 8 |
| paid attempts | 7 |
| not-started attempts | 0 |
| uncharged entered attempts | 1 |
| search E/F/stress requests | 7988 |
| fresh endpoint checks | 6 |
| final requests including fresh | 7994 |
| charged requests in raw ledgers | 7988 |
| observed valid proposal landings | 2 |

Each arm reconciles `sum(record.requests) == search_requests` and
`requests == search_requests + fresh.checked`; each raw ledger independently
contains exactly the recorded charged search requests.  The scheduler record
for job `1252472` is `COMPLETED`, elapsed `00:03:05`, exit `0:0`.

## Per-arm outcome

| arm/seed | outer attempt statuses | paid search EFS | fresh | valid proposals | best ΔE from initial |
|---|---|---:|---:|---:|---:|
| joint/101 | evaluation_failed, evaluation_failed | 1997 | 1 | 0 | 0 eV |
| joint/7 | biased_quench_failed, biased_quench_failed | 1997 | 1 | 0 | 0 eV |
| pqc/101 | valid_landing, atomic_evaluation_failed | 1997 | 2 | 1 | 0 eV |
| pqc/7 | valid_landing, atomic_evaluation_failed | 1997 | 2 | 1 | 0 eV |

The joint/7 run reached one inner `maxiter` biased quench and one later budget
failure.  Joint/101 reached 11 completed Gaussian climbs before a budget
failure; its second requested attempt had no paid request.  Each PQC arm's
second attempt ended in `BudgetExhausted`; the first attempt produced a valid
numerical landing that was MC-rejected.  Using one primary outcome per outer attempt from the recorded terminal error
strings, there was one maxiter stage, four reached `BudgetExhausted`/request-limit
stages, and no backend failure. The remaining entered attempt was uncharged
because the shared budget was already exhausted; it is not counted as a fifth
budget stage.

## Landing geometry

The common qualified initial landing is Fe56C24, volume
`686.1044372817767 Å^3`, with minimum MIC distances (C–C, Fe–C, Fe–Fe)
`2.98755926`, `1.90000440`, and `2.42021356 Å`.  It is the initial record and
is not counted as a search landing.

The two observed PQC candidates retained the same Fe56C24 composition:

| arm/seed | candidate ΔE from initial | volume (Å³) | minimum MIC C–C / Fe–C / Fe–Fe (Å) | MC |
|---|---:|---:|---|---|
| pqc/101 | `+13.7426174509 eV` | `736.4452240102` | `1.38892737 / 1.82076556 / 2.26773793` | rejected |
| pqc/7 | `+7.9533534413 eV` | `719.7272100803` | `2.21957567 / 1.73969889 / 2.31346863` | rejected |

The candidate volume and MIC distances are diagnostics of the saved endpoint;
they do not establish a stable Fe7C3 phase.  No new proposal was accepted and
the best record remained the qualified initial structure in all four arms.

## Scope limits

The experiment preserved the registered two-step, 2000-request-per-arm cap,
MACE model, pressure, tolerances, and seeds.  It tests one supplied
Fe14C6-derived structure under one approximate backend.  The result does not
support a general PQC-versus-joint efficiency claim, native LASP equivalence,
DFT/magnetic stability, or a chemical conclusion from the two MC-rejected
high-energy candidates.  Full raw per-arm records, ledgers, fresh checks and
the reproducible summarizer are in the evidence directory; the generated
aggregate is `comparison/summary.json`.


The strict-quench and full-Hessian audit of the two retained PQC candidates is
reported in [fe7c3-80-landing-qualification.md](fe7c3-80-landing-qualification.md).
