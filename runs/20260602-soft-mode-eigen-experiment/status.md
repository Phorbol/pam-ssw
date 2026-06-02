# Status

- Phase: completed
- Completed systems: 3/3
- Current step: summarized
- Rows: 274
- Verification: CUDA run exited 0; `results.csv` has 275 lines including header
- Current read: a=500 reference dimer is direction-locked on C60/PdO and often not softer than pool candidates; lower dimer bias improves softness but does not converge within 30 rotations. Lanczos finds much softer local modes, but those directions have low overlap with scored_pool and are not automatically better global-optimization directions.
