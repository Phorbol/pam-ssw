# Shared-candidate counterfactual direction gate

## Question

At the validated low-cost `K=4` direction scale, is the current bottleneck the
candidate generator or the static candidate scorer?

For each locked `(system, state, seed)` group, generate the exact same four
step-zero native candidates once, evaluate each with the existing central-FD
HVP and unchanged static score, then force each candidate through the otherwise
unchanged serial-Gaussian walk and true quench.

This removes winner-only label bias. It does not add UCB, Thompson sampling,
new descriptors, a learned model, or a new direction family.

## Frozen matrix

- Systems: C60 and PdO.
- States: locked accepted minima at production trials 100 and 180.
- The runner takes an explicit `--state-source-root`; state and origin-summary
  bytes are still checked against the preregistered SHA256 registry.
- Seeds: 42, 43, 44.
- Candidates per shared pool: four.
- Total terminal cases: 48.
- C60 action kernel: current validated production profile.
- PdO action kernel: current production profile with only
  `oracle_candidates=4`, because this experiment asks whether selection can
  rescue the already-tested low-cost K4 scale.
- The first candidate pool is charged once per group (8 force evaluations);
  every forced arm has an independent, purpose-accounted walk and quench.
- The RNG is advanced to the post-pool state before every arm continues, so
  later stochastic proposals do not depend on which candidate is forced.

## Readout

No weighted acquisition score is introduced. For every shared pool report:

- which candidate the existing static score selected;
- the best strictly certified, geometry-valid terminal candidate;
- exact static-winner energy regret;
- within-pool Spearman ordering between static score and terminal quality.

The posterior stage opens only if both systems independently show a selection
bottleneck: a majority of comparable pools contain a better non-winner and
the median score/terminal ordering is non-positive. If the static winner is
best in every comparable pool with positive ordering, candidate generation is
the bottleneck. All other outcomes are explicitly ambiguous and do not justify
adding a posterior selector.

## Repeated result

Two complete executions used 27,860 force evaluations and produced 96/96
strictly certified, geometry-valid landings. Ten of twelve shared pools missed
the static winner in both repeats, but the PdO score-ordering sign was not
repeat-stable. The static scorer is therefore a demonstrated bottleneck, while
the cross-system posterior-promotion gate remains closed.

All 48 positive-curvature candidates had
`0.5 * score_sigma**2 * curvature = 0.8 eV` to numerical precision. Thus the
adaptive score scale exactly cancels the curvature term used for ranking while
still paying for its HVP. Repeat-averaged counterfactual replay promoted only a
prospective test of the already-paid true-curvature ranker. A leave-one-group
beta direction-family rule selected `random` in all 12 folds, so it was a fixed
family prior rather than context-sensitive posterior learning.
