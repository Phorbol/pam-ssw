# Minimal87° height: isolate the criterion without preset amplitudes

2026-09-10. `minimal_angle_height.py` provides an independent pure `MinimalAngleHeightPolicy`, with **no walker integration or PES experiment**. It implements the analytic force-angle criterion after the preceding8-run comparison showed0 growth updates among132 native-profile stages. It does not retune the native initial-height levels.

For one unit direction n, at the explicit preparation point define the background force (physical/LS plus the unchanged old history once) as `F=f_parallel*n+Fperp`. A new Gaussian contributes `c*W*n`, where `c=exp(-z²/(2sigma²))*z/sigma²>0`. The smallest height on the nondegenerate positive-height branch is

```
W = (cot87*norm(Fperp)-f_parallel)/c.
```

The resulting angle is87° up to arithmetic error. Any strictly smaller positive height fails that condition. There is no initial weight, growth rate or max-height parameter. The87° constant follows the recovered native criterion; the implementation uses the same archived pi denominator as the existing helper. This is mathematical criterion isolation, not a reconstruction of the native discrete growth controller.

`prepare(history,center=...,direction=...,width=...,point=...,background_force=...)` returns the unchanged old terms plus one immutable new `FrozenHeightGaussian` when a finite positive minimum exists. The background force contract is the same single-count convention as `ConservativeNativeHeightPolicy`; old weights are never rewritten. Subsequent optimization must freeze all returned terms and differentiate their exact summed energy. It must not recompute W on every force call. Forces and directions must share the declared Cartesian or scaled generalized metric.

## Necessary domain and numerical boundaries

- Nonpositive z or an underflowed/nonfinite c cannot produce the assumed forward Gaussian force. Reject; do not reflect the direction or invent a force floor.
- If `cot87*norm(Fperp)-f_parallel<=0` and the background has a defined angle, return `already_forward_satisfied`, W=0 and **the unchanged history**. No zero-height term is appended and no next walker action is silently selected.
- If Fperp=0 and f_parallel<=0, the cancellation height produces zero resultant force, at which the angle is undefined. Every strictly larger W gives0°, but there is no attained smallest feasible positive height. Reject this branch explicitly; adding an arbitrary positive epsilon would change the mathematical problem.
- A positive numerator whose division underflows to zero is not 'already satisfied'; it is an unrepresentable positive height and is rejected. Nonfinite weight/resultant is also rejected.
- At large nearly cancelling longitudinal forces, floating-point arithmetic may not resolve the small target resultant. A nonpositive computed forward resultant raises. Returned angle and signed criterion residual expose ordinary roundoff at the equality boundary. The helper does not silently round W upward or claim an exact strict inequality in floating arithmetic. Very small perpendicular norms may themselves underflow, leading to explicit refusal rather than a guessed finite perturbation.
- Unit-direction validation and coordinate dimensions remain explicit. The height is not a proof of basin escape, local stability, or an energy barrier.

## Why no automatic zero-bias continuation is supplied

The recovered native routine starts from positive initialized W and only increases it; its inspected branch does not define what to do with a zero or negative analytic required height. The current independent forward-force walker treats nonpositive computed height as `nonpositive_height` failure. Neither source establishes that satisfying an angle at one displaced point completes a basin-to-basin proposal or authorizes a true quench/another displacement. Such a quench could simply return to the old basin, and repeated displacement would add a separate continuation rule.

Therefore the first possible walker integration should restrict to W>0 and map `already_forward_satisfied` to the existing explicit nonpositive-height failure unless a separately justified lifecycle is designed. The helper returns that state so it cannot be mistaken for a completed candidate. No fallback or new heuristic is included here.

Three mathematical tests pass: analytic87° equality and failure of a smaller W; already-satisfied and unattained-zero-force branches; unchanged history and finite-difference agreement of the frozen total energy/force. Combined with the native-height helper tests: **7 passed**. These are arithmetic tests only. The actual8-run comparison and its full denominator remain in `conservative-native-height-comparison.md`; this new criterion has not been tested on those systems and no scientific benefit is claimed.
