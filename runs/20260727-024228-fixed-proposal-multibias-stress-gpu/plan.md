# FIRE-conditioned multi-bias stress replay

The primary one-bias tasks established optimizer-neutral behavior. This
secondary experiment tests cumulative-bias objectives without mixing them into
the primary estimator.

- Systems: C60 and PdO
- Reference path: production ASE FIRE
- Seeds: 42, 43, 44, 45
- Captured cumulative bias counts: 2 and 4
- Replay backends: FIRE, FIRE2, safe total-gradient L-BFGS
- Shared observation cap: 400
- Common force certificate: 0.05 eV/A

Because tasks after the first relaxation are conditioned on FIRE's trajectory,
the experiment is a mechanism stress test only. It is not an optimizer-neutral
sample of later SSW tasks.
