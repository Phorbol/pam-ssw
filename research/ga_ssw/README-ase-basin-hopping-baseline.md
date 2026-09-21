# ASE BasinHopping baseline

This prepared research runner is a mature global-search baseline, not a new
PAM component. It reuses the exact frozen `cu13.extxyz`, `cu31_fixed.extxyz`
and `bicyclobutane.extxyz` inputs from `two-stage-ritz-comparison-20260912`,
with EMT for the metal cases and GFN2-xTB for bicyclobutane. It records ASE
3.26, global `numpy.random.seed(seed)`, `dr=0.5 Å` (ASE documentation example),
150 K, local `fmax=.01`, 400 local steps, 6000 E/F requests and 90 seconds.

The adapter calls the existing PAM Safe-total quench through ASE's optimizer
slot, preserving the unmodified BasinHopping move and Metropolis controller.
Each local `QuenchResult` (including a failed one) is written immediately to
`local-results.json`; a failed local quench aborts that arm and is never
reported as a minimum. Converged local landings are fresh checked. The ASE
framework's initial prequench (`ro`) behavior is retained and recorded in the
plan. ASE's `basin.py` is copied into the output as source evidence.

The runner only creates artifacts by default; `--execute` is required for PES
calls. This baseline is for matched developmental comparison and carries no
claim of efficiency or scientific superiority.
