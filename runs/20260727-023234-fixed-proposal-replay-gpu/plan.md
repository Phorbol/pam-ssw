# Real-GPU fixed proposal replay

- Frozen code: `a64816b`
- Systems: C60 and PdO, using the same inputs/model/fixed masks as G1b
- Primary tasks: eight independent action seeds (`42..49`), first proposal
  relaxation only, all starting from one certified bootstrap minimum
- Backends: ASE FIRE, ASE FIRE2, safe total-gradient L-BFGS
- Shared task certificate: production `proposal_fmax`, `proposal_relax_steps`,
  and trust radius
- Primary metric: exact evaluator calls per paired frozen task
- Secondary metrics: certificate coverage, wall time after model warm-up, final
  biased energy, and endpoint displacement relative to FIRE
- No parameter tuning and no pooled score

The task distribution is optimizer-neutral but not claimed to be a canonical
PES distribution.
