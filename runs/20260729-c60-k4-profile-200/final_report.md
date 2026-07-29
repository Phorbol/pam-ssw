# C60 public K4 profile: 200-step production confirmation

## Outcome

The user-facing profile
`c60_direction_efficient_validated_20260729` completed all 200 macro steps on
the frozen execution commit
`111c9d2967cdb6ffd2f3bb4ca167c48bd60ff30c`. The run used the reviewed C60
input, MACE model, seed 42, K4 direction portfolio, at most eight bias
micro-steps, safe-LBFGS proposal relaxation, and certified
ASE-LBFGS-to-FIRE true quenching. No posterior direction selector, direction
UCB, probe, archive direction reuse, or plateau evolution was enabled.

The profile is operationally validated as a 200-step C60 search configuration:

- 200/200 macro steps completed;
- 201/201 true quenches had convergence certificates;
- 4 ASE-LBFGS fallbacks were attempted and all 4 FIRE rescues converged;
- the purpose ledger closes exactly to 78,840 force evaluations;
- `unattributed=0`;
- 164 minima were retained;
- the best energy fell from -474.669800 to -506.546478 eV, a 31.876678 eV
  decrease;
- total measured wall time was 2,213.8 s (36.90 min).

This confirms usability and accounting closure. It does not establish that K4
is universally better than K12, that seed 42 is representative, or that the
archive minima are distinct chemical isomers under a symmetry-aware matcher.

## Descriptive historical comparison

The historical K12 strict-quench run used the same input hash, model hash,
runtime versions, calculator settings, seed, B8 policy, proposal optimizer, and
true-quench protocol. Its execution commit differs, however. After removing
output paths and two inactive block-Krylov schema defaults added later, the
scientific config delta is `oracle_candidates: 12 -> 4`. Consequently the table
is a descriptive historical comparison, not a same-commit paired causal
ablation.

| Metric | K4 public profile | Historical K12 | K4 - K12 |
| --- | ---: | ---: | ---: |
| Best energy (eV) | -506.546478 | -508.078308 | +1.531830 |
| Energy decrease (eV) | 31.876678 | 33.408630 | -1.531952 |
| Best-drop AUC (eV trial) | 5251.136887 | 6188.496643 | -937.359756 |
| Total force evaluations | 78,840 | 91,490 | -12,650 |
| Direction force evaluations | 10,016 | 29,058 | -19,042 |
| Non-direction force evaluations | 68,824 | 62,432 | +6,392 |
| Wall time (s) | 2213.8 | 1566.6 | +647.2 |
| Archive minima | 164 | 140 | +24 |
| Duplicate rate | 18.41% | 30.35% | -11.94 percentage points |
| Energy decrease / 1000 FE (eV) | 0.4043 | 0.3652 | +0.0392 |
| Minima / 1000 FE | 2.0802 | 1.5302 | +0.5500 |

K4 therefore produced more archive minima and used 13.83% fewer total force
evaluations, but it reached a 1.5318 eV higher final minimum and had a 15.15%
lower best-drop AUC. Its endpoint energy decrease per force evaluation was
higher, but its wall time was longer and its energy descent was substantially
slower in macro-step space:

| Progress threshold | K4 trial | K12 trial |
| --- | ---: | ---: |
| 10 eV decrease | 8 | 4 |
| 20 eV decrease | 50 | 6 |
| 30 eV decrease | 75 | 25 |
| Last best-energy improvement | 130 | 194 |

The clean conclusion is a trade-off, not a promotion claim: reducing K removed
19,042 direction evaluations, but the divergent trajectory incurred 6,392
additional non-direction evaluations and did not preserve K12's landing-energy
quality.

## What is now the bottleneck?

For the K4 run, the measured force-evaluation shares were:

| Purpose | Force evaluations | Share |
| --- | ---: | ---: |
| Biased proposal relaxation | 52,007 | 65.97% |
| Landing true quench | 15,164 | 19.23% |
| Direction oracle | 10,016 | 12.70% |
| Escape true-PES check | 1,404 | 1.78% |
| Post-relax validation | 201 | 0.25% |
| Initial raw-State quench, raw label `starter_true_quench` | 48 | 0.06% |
| Unattributed | 0 | 0% |

This separates two different bottlenecks:

1. **Computational bottleneck:** proposal relaxation. K4 made direction
   evaluation a secondary cost. The 1,204 proposal relaxations averaged 38.88
   iterations, had median 30 and p90 80, and 193 reached the 80-step cap.
2. **Search-quality bottleneck:** direction-to-landing quality. The run kept
   discovering minima after trial 130 but never improved the best energy again.
   Lower duplicate rate therefore did not imply better descending transitions.

True quenching was fully certified, but the K4 trajectory was harder to quench
than the historical K12 trajectory: mean/median/p90 iterations were
66.70/52/102 instead of 55.45/43/83. This and the additional proposal work
explain why fewer total force evaluations did not translate into lower wall
time. Because the runs occurred on different commits and at different times,
the wall-time delta must remain descriptive.

## Direction behavior

K4 performed 1,252 bias micro-steps, or 6.26 per macro step, below the B8 cap.
The selected directions were:

| Direction source | Selected | Fraction |
| --- | ---: | ---: |
| Momentum | 989 | 79.0% |
| Bond | 153 | 12.2% |
| Random | 110 | 8.8% |

Momentum remains the dominant effective source and must not be removed merely
because its physical interpretation is less clean. At the same time, its
long-run dominance did not prevent a 70-step terminal best-energy plateau.
That makes landing-quality prediction or physically constrained refinement a
more relevant next direction question than adding another exploration bonus.

## Accounting issue found

The initial raw `State` is truly quenched once before the first starter is
selected and all 48 force evaluations are included in the total budget.
However, `SurfaceWalker.run` passes
`EvaluationPurpose.STARTER_TRUE_QUENCH`, so the raw ledger reports:

```text
bootstrap_true_quench = 0
starter_true_quench   = 48
```

This is a naming/credit-assignment defect, not missing cost. It was corrected
after the frozen execution by making the initial-quench purpose explicit:
top-level `SurfaceWalker.run` defaults to `BOOTSTRAP_TRUE_QUENCH`, while
`SSWAttemptWorker` passes `STARTER_TRUE_QUENCH` for a dispatched starter. Both
boundaries have end-to-end ledger regression tests. The raw production summary
remains unchanged as execution evidence.

## Decision

- Keep the K4 profile as an evidence-backed, force-budget-efficient C60 option;
  do not change the package-wide default.
- Do not claim K4 energy-search superiority over K12.
- Do not promote direction TS/UCB, a classifier, or a generative direction
  model from this one run. The present evidence does not show out-of-sample
  predictive gain.
- Stop reducing K for now. Direction oracle is only 12.70% of K4 cost, so
  further K reduction has limited total-cost headroom and risks more quality
  loss.
- Keep proposal-relax optimization as the main cost-reduction track, but make
  the next search-algorithm experiment target landing quality under a frozen
  K4/B8/cost protocol.
- Require any posterior direction model to beat the fixed physics portfolio on
  held-out trials/seeds in both landing-energy AUC and force-normalized reward
  before online activation.

The next algorithmic step is therefore not another selector block. It is a
small, same-commit experiment that isolates whether a constrained soft-mode
refinement of the existing random/bond/momentum intent improves landing quality
without increasing total force budget. Only after that physical baseline is
established is posterior reweighting scientifically justified.
