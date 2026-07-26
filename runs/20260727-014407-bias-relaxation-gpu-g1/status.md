# Status

Prepared from frozen commit `47212f7e280671b76d87af95250d469c163353d0`.

Current state: G1 complete; G2 not authorized by the frozen gate.

- GPU preflight passed on NVIDIA GeForce RTX 3060 with the frozen input and
  model hashes.
- Four of eight arms reached and completed proposal exploration.
- Four arms stopped before action dispatch because the repeated bootstrap
  L-BFGS-B endpoint did not satisfy the per-atom force certificate.
- C60 total-vs-separated was the only complete custom pair. Bias separation
  increased biased-proposal calls by 258 and proposal backend calls by 341.
- PdO produced no total-vs-separated pair.
- Failed bootstrap calls were charged internally but not persisted when the
  bootstrap function raised. This is an accounting observability defect.

Decision: do not advance to the multi-seed G2. First align the optimizer stopping
criterion with the per-atom force certificate and persist failed-bootstrap
accounting.
