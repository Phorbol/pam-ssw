# Local-only direction source ablation

This branch is a non-default, free nonperiodic fixed-cell source ablation from
baseline `6c70ef4b1a568866f084a9670a01eeae496d79fc`. The sole core change is in
`LocalDirectionState.update`: after the existing displacement/mask preparation,
the normalized displacement seed is replaced with zeros. Initial direction,
local coefficients, c9 calculation, generator, RNG consumption, marker
transition, and release behavior remain on the baseline path. No public config,
checkpoint format, or default behavior was added. No PES or search was run.

The verifier compares the working implementation to the baseline module loaded
from `git show`, then checks the explicit zero-seed update formula and the
zero-direction/release case using a saved free-cluster fixture:

```bash
/home/gengjianrui/.conda/envs/mace_env/bin/python \
  research/ga_ssw/evidence/direction-local-only-20261012/validate_local_only.py
```

Result: all four verifier contracts pass (initial output/state parity; update
RNG/marker/route/release parity; local-only vector formula; forbidden-pair
release behavior). This is implementation evidence only and gives no evidence
about search effectiveness.

The existing focused suite was also run:

```bash
/home/gengjianrui/.conda/envs/mace_env/bin/python -m pytest -q \
  tests/standalone/test_native_direction_control.py
```

It reports 13 passed and one expected failure:
`test_stage_state_uses_saved_center_not_previous_direction` asserts the
baseline displacement-seeded update vector, which this ablation deliberately
removes. The test was left unchanged; all other tests in that file pass. ASE
emits seven NumPy shape deprecation warnings from existing fixtures.

The next scientific comparison remains pending: matched full-search pairs on
the preselected hard cases, with the frozen protocol and complete E/F ledger.
This branch and verifier do not establish transfer or justify a default.
