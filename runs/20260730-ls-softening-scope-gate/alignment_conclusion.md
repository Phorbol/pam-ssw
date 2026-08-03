# Choice-aligned local-softening survivor result

## Decision

Choice-aligned rebuilding does not pass the preregistered system-general
survivor gate. The current moving-reference exponential LS line stops here.

The result contains a real PdO-specific signal, but it is not a clean
cross-system improvement and it costs materially more force evaluations on
PdO. The threshold is not relaxed after observing the result, and no
strength, active-count, cosine-threshold, or optimizer sweep follows.

## Single tested change

Relative to the existing `both` scope, the candidate changes only:

```python
choice_aligned_softening_enabled = True
```

When the selected direction has cosine below the existing `0.3` threshold
with the initial anchor, the same active-neighbor exponential penalty is
rebuilt from the selected direction before curvature-controlled biasing and
proposal relaxation.

There is no new fitted parameter. The 18 pairs use the same C60/PdO fixed
starters and seeds 42--44 as the preceding 2x2 scope gate.

## Paired result

Aligned-minus-current landing effect is lower-is-better.

| System | Pairs improved | Mean landing effect | Median landing effect | Median FE ratio | Aligned rebuilds |
|---|---:|---:|---:|---:|---:|
| C60 | 5/9 | -0.251444 eV | -0.000031 eV | 1.0301 | 64 |
| PdO | 7/9 | -0.530891 eV | -0.756775 eV | 1.1768 | 20 |

The preregistered survivor conditions were:

1. non-positive median landing effect in both systems;
2. at least 5/9 improved pairs in both systems;
3. median FE ratio no greater than 1.10 in either system.

The first two conditions pass. The third fails on PdO.

## Mechanistic interpretation

The alignment change occurs after direction selection, so it cannot rescue the
observed absence of a direction-generation effect. It changes the local
penalty used for the curvature-controlled bias and biased proposal landscape.

C60 is effectively neutral at the median. Its non-zero negative mean comes
from large, sign-changing case effects rather than uniform improvement.

PdO has a meaningful landing signal, especially for the middle and late
starters, but:

- two of three bootstrap pairs become worse;
- median force evaluations rise by 17.7%;
- the effect is therefore neither system-general nor uniformly
  starter-independent.

This supports a narrow statement:

> Aligning the penalty with the executed displacement can improve the
> proposal landscape for some progressed PdO states.

It does not support:

- a global LS-SSW default;
- a claim that the direction oracle was improved;
- a system-independent search-efficiency gain;
- further tuning of the existing moving-reference exponential penalty.

## Final LS stage boundary

The accumulated evidence now distinguishes three claims:

1. **Numerical activity:** confirmed. The penalty is evaluated correctly and
   materially changes trajectories.
2. **Current direction-softening role:** rejected. First selected directions
   were unchanged in all 18 scope blocks and were not more delocalized.
3. **Current proposal-landscape role:** unaligned production behavior is not
   robustly beneficial; selected-direction alignment yields a PdO-specific
   benefit at increased cost but no general survivor.

No equal-budget 20k-FE or 200-step run is launched for this candidate. Such a
run would be justified only for a PdO-specific method-development objective,
which is outside the present goal of unbiased, system-general PES
exploration.

