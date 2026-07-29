# Uphill propagation U0/U1 protocol

This experiment isolates one question: on the same frozen SSW walk prefix and
direction, does the current adaptive Gaussian, an analytic curvature-matched
Gaussian without feedback, or a system-calibrated fixed Gaussian produce the
most useful biased-PES proposal per force evaluation?

The three arms change only the newest Gaussian and its explicit displacement.
Earlier Gaussian terms, the direction, optimizer, force certificate, iteration
limit, calculator, and structural constraints are held fixed. Fixed controls
are calibrated from separate first-step tasks and never from evaluation
outcomes.

The curvature-matched arm uses the same `bias_weight_min` and
`bias_weight_max` bounds as the production configuration. This keeps numerical
weight bounds fixed while removing only history-dependent `sigma_scale` and
`weight_scale` feedback. GPU schema version 1 omitted this shared bound and is
diagnostic only; schema version 2 is the production-bounded comparison.

The result is conditional one-step mechanism evidence. It is not evidence that
one arm is a better complete SSW search policy, an unbiased thermodynamic
sampler, or a generally superior optimizer.
