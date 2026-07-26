# GPU G1b frozen ablation

G1b repeats the complete eight-arm matrix on commit
`c598096170d40625b6d127668935a166cc1507c1`.

The scientific design and all proposal settings are identical to G1. The only
algorithm change is the certificate-aligned SciPy L-BFGS-B stopping condition
used by true quenches. The driver reuses the original frozen G1 implementation
and adds structured, exact accounting for bootstrap failure.

All eight arms must be rerun because the stopping change affects bootstrap,
starter, and landing true quenches. Mixing old successful arms with new arms
would not be a clean ablation.

No automatic campaign retry is allowed after physical calls begin.
