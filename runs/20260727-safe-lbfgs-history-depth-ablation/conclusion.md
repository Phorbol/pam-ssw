# Safe L-BFGS history-depth ablation — evidence-limited conclusion

## Certificate outcome

Both arms satisfied 16/16 force certificates. The certificate is derived strictly from `fmax > 0` and `0 <= final active max force <= fmax`; optimizer status is not substituted for that force test.

## Fixed-task cost outcome

History 10 used 1494 evaluator calls; history 1 used 1833.

C60: 718 versus 773 calls (history 10 versus history 1). Per-task history10-minus-history1 differences were 3 lower, 1 tie, and 4 higher, with median paired delta +0.5. The C60 effect is not consistent across tasks.

PdO: 776 versus 1060 calls. History 10 was lower on 7 of 8 paired tasks, with median paired delta -42.

## Next validation choice

Advance history 10 to the 200-step validation because it preserves all certificates, shows a strong PdO cost reduction, and is the current default. This is a validation choice, not a new default claim.

## Claim ceiling

Endpoints can differ. The recorded paired endpoint-energy and raw Cartesian position differences are diagnostics only: they support no endpoint-equivalence claim and no general SSW claim.
