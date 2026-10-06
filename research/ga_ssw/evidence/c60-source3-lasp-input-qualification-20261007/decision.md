# Source #3 external-input qualification: passed, search not tested

GPU1664642, source58d4579, completed in12seconds. The original archived
non-Ih source#3 endpoint was not relaxed or replaced. Same MH-1/omol,
float64 CUDA as the independent C60 panel. All raw inputs, helper/client
snapshots, native output and costs are under `run-1664642/`.

Two direct E/F calls plus one native callback cost3paid requests and3actual
MACE calculate calls; no errors/denials or process survivors. The isolated
and50Angstrom periodic evaluations differ by0eV and2.01e-15eV/Angstrom in
maximum force component, within the predefined1e-4 tolerances. Nearest
inter-image atom distance42.971Angstrom exceeds the6Angstrom model cutoff.
The native callback has the expected cell and unchanged coordinates; its
energy difference is0eV and force-component difference2.78e-15. These tiny
differences are reported as observations, not a demand for numerical parity.

Root independently verified cost closure, all archived input/source hashes,
the imported helper and local client paths, native process completion and
raw callback. Native `SSWsteps=1` returned only the initial minimum and
`SSW all done`; no escape or target discovery was performed. The native
force printed in its log is a component measure, not the Python maximum
per-atom vector norm.

Decision: this geometry can be used in a separately frozen native reference
protocol. Qualification does not establish search equivalence, general
vacuum equivalence, LS benefit, C60 global success or optimizer superiority.
Independent Python remains nonperiodic. The ongoing four-arm local LS and
random direction panels are unchanged. Do not automatically submit a native
search or count this singleton callback as an independent trajectory.

Engineering boundary: the archived external helper limits successful
responses, so the caller separately enforced3paid slots including failed
attempts. Synthetic repeated-failure and complete-main-path preflights
passed with zero physical PES; those are interface evidence only. A local
fix of the helper's attempt limit is separate from this frozen qualification
and does not require redoing these model evaluations.
