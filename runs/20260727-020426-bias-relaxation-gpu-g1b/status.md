# Status

Complete.

- GPU preflight passed with the frozen model and input hashes.
- All eight campaigns passed bootstrap certification.
- All terminal actions have exact cost and optimizer diagnostics.
- Every purpose ledger closes with zero unattributed calls.
- C60 bias-separated minus safe-total:
  - charged biased-proposal evaluations: +144;
  - proposal backend evaluations: +61;
  - rejected line-search steps: +158.
- PdO bias-separated minus safe-total:
  - charged biased-proposal evaluations: +342;
  - proposal backend evaluations: +401;
  - rejected line-search steps: +332.

Decision: do not advance analytic bias separation to multi-seed G2. It changes
secant decisions but increases the primary proposal cost on both systems.
