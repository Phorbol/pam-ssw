# Choice-aligned local-softening survivor gate

## Hypothesis

The current active-neighbor penalty is built from the initial anchor, while
the executed direction is selected later from the full portfolio. Rebuilding
the same penalty on the selected direction may remove this active-region
mismatch without changing the penalty form or adding a fitted parameter.

## Single change

Compare the existing `both` scope against:

```python
choice_aligned_softening_enabled = True
```

All other configuration fields, fixed starters, seeds, direction candidates,
Gaussian-bias updater, optimizers, true quench, and evaluator accounting remain
unchanged.

The existing cosine threshold `0.3` is not tuned in this experiment.

## Cohort

- systems: C60 and PdO;
- fixed starters: bootstrap, middle, and late;
- paired seeds: 42, 43, and 44;
- 18 complete escape-plus-true-quench pairs.

## Metric and decision

Primary metric: aligned-minus-current landing delta in eV; lower is better.

Alignment survives only if:

1. its median paired landing effect is non-positive in both systems;
2. it improves at least 5/9 paired blocks in both systems;
3. its median force-evaluation increase is no more than 10% in either system.

Failure stops the current moving-reference exponential LS line. No strength,
width, threshold, active-count, or optimizer sweep follows.

