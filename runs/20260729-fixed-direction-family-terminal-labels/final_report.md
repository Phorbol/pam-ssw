# Fixed C60 direction-family terminal-label experiment

## Question and claim boundary

This experiment asks whether raw direction-family identity contains enough
repeat-stable, held-out predictive signal to justify a direction-level
posterior selector.

The comparison is `random_only` versus `bond_only`. It is not a comparison of
the complete mixed production portfolio, and it is not evidence that bond
information is universally harmful. In particular, it does not reconstruct
the original SSW biased-CBD refinement of a random-plus-bond intent.

No production default, starter selector, UCB-like rule, Thompson sampler,
classifier, regressor, proposal optimizer, or true-quench optimizer was
changed.

## Frozen cohort

- System: C60.
- Model SHA256:
  `0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5`.
- Exact accepted starters:
  `intermediate_accepted` and `plateau_accepted`.
- Seeds: 42, 43, and 44.
- Two exact repeats with reversed arm order.
- Four candidates per direction selection.
- Eight Gaussian-bias steps, with normal early exit.
- 80-step `safe-lbfgs-total` proposal relaxation.
- ASE-LBFGS true quench with ASE-FIRE fallback at `fmax=0.01 eV/A`.
- Momentum and the outcome-gated bond-pair boost disabled in both arms.

The only arm-specific control is `n_bond_pairs`: zero for `random_only` and
four for `bond_only`. Every selection therefore evaluates exactly four
candidates using four central HVPs, or eight force evaluations. Realized total
cost remains an outcome because the arms can induce different walk lengths,
proposal-relaxation work, and quench difficulty.

## Protocol corrections before the final run

The first committed attempt (`0cbc88a`) stopped after a nominal
`random_only` trajectory became 2x bond plus 2x random. Root-cause tracing
showed that an unproductive proposal relaxation activated the existing
`stagnation_bond_pair_boost=2`. Commit `999574b` froze that feedback to zero in
both arms.

The second attempt stopped when one true quench lacked a strict certificate.
Independent reconstruction showed that the trajectory is numerically
path-sensitive near the `0.01 eV/A` boundary; five standalone repeats all
converged, but not to the same escape basin. Commit `88dd4de` therefore retained
uncertified terminals as costed negative outcomes instead of deleting them or
retrying until success. The meaningful-label certificate requirement was not
relaxed.

The two incomplete output trees were preserved separately and were not reused
in the final evidence.

## Execution integrity

- Final execution commit:
  `88dd4def973f2f7a7c656c87227dd64e41407f9c`.
- Completed cases: 24/24.
- Certified terminal outcomes: 24/24.
- True-quench fallbacks: 2/24.
- Fragmented outcomes: 0/24.
- Unattributed force evaluations: 0.
- Total force evaluations: 10,553.
- Measured generation time: 136.814 s.
- Measured true-quench time: 47.662 s.
- Total measured sequential section time: 184.476 s.

The final run happened not to contain an uncertified terminal. The retention
rule remains necessary because it prevents future action-dependent
nonconvergence from being silently censored.

## Direction-family purity

| Arm | Selections | Evaluated candidates | Selected directions | Direction FE |
|---|---:|---:|---:|---:|
| random only | 56 | 224 random | 56 random | 448 |
| bond only | 70 | 280 bond | 70 bond | 560 |

There are no cross-family candidates or selections. The differing selection
counts arise from the existing early-exit dynamics; each individual selection
still costs exactly eight force evaluations.

## Terminal outcomes and realized cost

| Arm | Raw meaningful | Repeat-stable meaningful | Median landing delta (eV) | Mean landing delta (eV) | Total FE | Mean FE/case |
|---|---:|---:|---:|---:|---:|---:|
| random only | 6/12 | 3/6 | -0.622299 | +0.206057 | 4,132 | 344.3 |
| bond only | 2/12 | 1/6 | +7.201324 | +7.979317 | 6,421 | 535.1 |

All three repeat-stable random successes come from the plateau starter. The
single repeat-stable bond success is plateau seed 43. Neither family produces
a meaningful intermediate-starter outcome.

The bond-only arm consumes 55.4% more total force evaluations while producing
fewer lower basins. This is not caused by a larger per-selection oracle: it
comes from more completed walk steps, more biased relaxation, and more
expensive terminal quenches.

Most repeat pairs agree to within 0.0002 eV. The intermediate-seed44 bond pair
is the exception: its two nonproductive landings differ by 6.212 eV. The
binary outcome agrees, but the nonlinear GPU trajectory is not bitwise
determined by the nominal seed.

## Cost decomposition

| Purpose | Force evaluations | Share |
|---|---:|---:|
| biased proposal relaxation | 6,576 | 62.314% |
| direction oracle | 1,008 | 9.552% |
| landing true quench | 2,753 | 26.087% |
| escape true-PES checks | 192 | 1.819% |
| post-relax validation | 24 | 0.227% |
| bootstrap, starter quench, unattributed | 0 | 0.000% |
| total | 10,553 | 100.000% |

After reducing the portfolio to K=4, direction HVPs are no longer the primary
cost bottleneck. Proposal relaxation plus true quenching account for 88.4% of
the force evaluations. Further reducing K cannot fix a direction that drives
the propagator into an expensive, unproductive basin.

## Held-out posterior gate

The pre-registered audit uses a Beta(1,1) Bernoulli posterior predictive model
and leaves one seed out at a time.

| Held-out seed | Starter-only Brier | Starter-plus-family Brier | Improved |
|---:|---:|---:|:---:|
| 42 | 0.152778 | 0.109375 | yes |
| 43 | 0.138889 | 0.187500 | no |
| 44 | 0.152778 | 0.109375 | yes |
| aggregate | 0.148148 | 0.135417 | not sufficient |

The family-conditioned model has a lower aggregate Brier score, but it worsens
the held-out seed-43 fold. More importantly, the stable-positive counts are
only 3 and 1, below the pre-registered minimum of five per family.

Decision: `enter_posterior_stage = false`.

## Scientific decision

1. Do not add a direction-family TS, UCB-like bonus, classifier, regressor, or
   MACE-feature ranker at this stage. The action set lacks enough stable
   positive support, and held-out improvement is not fold-stable.
2. Do not promote `bond_only`; on this fixed C60 cohort it is both less
   productive and more expensive than `random_only`.
3. Do not remove bond or momentum sources from the mixed validated production
   profile based on this experiment. Raw family isolation is not the same
   algorithm as mixed intent plus soft-mode refinement.
4. Do not prioritize further HVP-count reduction. At K=4 the oracle is only
   9.552% of total FE.
5. The remaining bottleneck is the direction-propagator interaction and the
   missing productive event family from the intermediate starter. A later
   direction experiment must first create a better physical action; a
   posterior selector cannot learn an action that never succeeds.

The full machine-readable evidence remains in the local execution artifact
`runs/20260729-fixed-direction-family-terminal-labels-output/evidence.json`.
Its SHA256 is
`d549f87ba89f28a99fa5678659089708722673c8517713b94552b929561a976a`.
The compact gate is recorded in `posterior_gate.json`.
