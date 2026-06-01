# Status

- Phase: completed
- Unit verification:
  - targeted fixed-N0 test: passed
  - reference-dimer tests: passed
  - full unit suite: `412 passed`
- CUDA matrix:
  - system: `c60`
  - variants: `paw_ucb_reference_dimer_bias_relax`, `paw_ucb_reference_dimer_direct_qp_adaptive50`
  - seeds: `0,1,2`
  - trials: `40`
  - completed cases: `6/6`
- Verdict:
  - fixed-N0 improves reference-dimer bias-relax coverage and mean best versus the previous broken integration.
  - fixed-N0 does not solve C60 dimer rotation lock or Direct-QP collapse.
