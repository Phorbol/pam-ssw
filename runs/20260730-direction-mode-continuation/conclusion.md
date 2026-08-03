# Direction-mode continuation C60 conclusion

Decision: `transported_direction_survives`.

This is a paired six-condition C60 mechanism result, not a production-default
promotion or a cross-system claim. Each starter-seed group paid for one shared
12-HVP initial Ritz mode, and all three arms executed that exact vector at bias
step zero. The 18 terminal trajectories produced 18 strict force certificates,
zero invalid geometries, zero fragmented landings, zero continuation
degeneracies, and zero unattributed evaluations.

| arm | meaningful | median mode cosine | direction FE | case FE |
|---|---:|---:|---:|---:|
| fixed_intent_ritz | 1 | 0.981530 | 744 | 2,289 |
| transported_direction | 2 | 0.999999 | 76 | 1,850 |
| continuation_lanczos | 2 | 0.846500 | 1,008 | 2,372 |

The six shared initial modes cost another 144 FE, so the complete C1 cost was
6,655 FE. Transport reduced direction-oracle cost by 668 FE (89.8%) and total
per-arm case cost by 439 FE (19.2%) relative to the fixed-intent control. It
also reproduced the control's meaningful plateau/seed43 event and found a
meaningful lower basin at plateau/seed44 where the control did not.

Continuation Lanczos is rejected. It paid the same post-step direction budget
as the control but reduced rather than increased branch continuity
(0.846 versus 0.982). Its two meaningful outcomes therefore do not validate
the proposed mode-tracking mechanism.

The result supports a narrow physical interpretation: once a useful soft
direction has been found, repeatedly re-solving a 12-HVP soft-mode problem at
every biased geometry is often redundant in these C60 trajectories. Projection
transport is sufficient to preserve that local escape branch and frees budget
for additional macro actions. It does not show that transport can replace
fresh direction generation between macro actions or in another material.

The original independently recomputed C1 cohort is retained locally as
`output-unpaired-cdaba43`. It was excluded from the conclusion because CUDA
float32 HVP/Ritz repeats did not yield bitwise-identical step-zero directions.
That failure led to the shared-initial-mode design above rather than a relaxed
analysis threshold.

No weighted score, posterior, UCB, Thompson sampler, continuity weight, angular
threshold, adaptive Krylov depth, or fallback mixture was introduced.
