# Fixed-cell periodic GA candidate protocol (zero-PES preparation)

This document records an input and identity-matcher preflight only. It does
not authorize or contain an executed GA run. No structures are perturbed and
no calculator is called here.

## Candidate parents

The Cu31 parents are the three fresh-qualified observations from
`research/ga_ssw/evidence/stage-control-e2e-20260912/cu31_fixed-pam_height_width-baseline-seed11/`:

| raw index | EMT energy (eV) | fresh fmax (eV/A) |
|---:|---:|---:|
| 0 | 1.0295675667349808 | 0.004407446629500687 |
| 1 | 3.887121026606497 | 0.00934207524520511 |
| 2 | 3.959263228027533 | 0.009013081773596837 |

All three have 31 Cu atoms, PBC in all directions, and cell exactly
`diag(7.2, 7.2, 7.2) Angstrom`. The first is the saved initial observation;
the latter two are raw minima. These are three qualified observations, not a
claim that they are three independent basins.

The Al31 parents are the three fresh-qualified observations from
`research/ga_ssw/evidence/native-ls-periodic-multicase-20260912/Al31-native-mic/result.json`
and its `raw_result.minima` array. They have energies
`0.9180442732723102`, `0.9179928508443562`, and
`0.9180140596073172 eV`, fresh fmax below `0.0058 eV/A`, 31 Al atoms, PBC in
all directions, and cell exactly `diag(8.1, 8.1, 8.1) Angstrom`.
The energy spread is only about `5e-5 eV`; these must therefore be reported as
near-repeat candidates and must not be presented as three distinct basins.
No artificial displacement or noise is permitted to manufacture separation.

## Identity matcher decision

The public periodic GA entry point already requires an explicit `matcher` and
provides `pymatgen_identity(ltol=..., stol=..., angle_tol=...)`, backed by
`pymatgen.analysis.structure_matcher.StructureMatcher` with
`scale=False`, `primitive_cell=True`, and `attempt_supercell=True`. This is the
appropriate existing periodic matcher when its dependency is available. Its
tolerances must be recorded before any PES call and are caller settings, not a
new default.

The approved `mace_env` preflight can import `pymatgen`; the installed
`StructureMatcher` defaults are `ltol=0.2`, `stol=0.3`, and `angle_tol=5.0`.
The project helper requires these arguments explicitly. These library defaults
are retained for this bounded candidate check and must be recorded before any
PES call. The generic
`pamssw.archive.MinimaArchive` is unsuitable as a drop-in periodic identity
matcher: its documented descriptor/RMSD path does not resolve atom
permutations or equivalent lattice bases. The existing periodic GA wrapper
also keeps routing descriptors separate from caller identity, which should be
preserved.

The matcher must be called on copies and its result recorded per observation.
A matcher exception
retains the raw observation and records an identity failure; it never deletes
the landing. If no qualified matcher is available, the run should be retained
as raw observations with no basin claim, rather than forcing a heuristic.

## Frozen bounded execution parameters

Seed7; GA quick_steps=0, generations=1, generation_steps=0, fine_steps=1,
regions=3, fine_regions=1, min_ga=1, max_batches=1, max_cut_attempts=30,
max_pair_attempts=50, partition_max_draws=100, slots_per_parent=1,
cuts_per_slot=1. These are the bounded existing TYPE1 integration-test
operation counts. Initial quick and offspring stages perform true quenching
only; fine performs one SSW attempt. This is a lifecycle probe, not a long
GA-SSW search or performance comparison.

SSW: dimer, rotation_bias100, rotation_hvp100, rotation_tol.02, width.1A,
max_gaussians14, temperature150K, fd_step1e-4A, global directions,
translation_only, Safe-total memory10, relax_steps200, bias_fmax.1 and
true fmax.01eV/A. fixed_cell=True, no stage adapter, no LS. Neighbor descriptor
lengths Cu2.6/Al2.9A, range multiplier1.1, projection weights
[.3,.2,.2,.1,.1,.1]; same-element collision bound .5A. These are explicit
development settings, not universal chemical-distance recommendations.
Three supplied parents are also the frozen descriptor references, with no
extra structures injected into the population.

## Bounded test shape

Use one seed and the existing fixed-cell core SSW parameters. For each of Cu31
and Al31, use the existing `PeriodicGAConfig` three phases (quick, generation
offspring, and fine), with one fixed-cell arm, at most 4000 paid requests and
60 seconds. Preserve the source snapshot, complete config and matcher settings
before the first PES request. Use the saved cell and composition as immutable
input checks; do not permit cell changes or composition changes.

Each result must retain every raw observation, paid/denied ledger entry,
parent/offspring lineage, and matcher outcome. Fresh EMT checks must be
independent of the search calculator and must report energy, forces, exact cell,
composition, and PBC. The compact report should separately state raw
observations, matcher representatives, and rejected/failed identity calls.

Acceptance is limited to the fixed-cell public contract and observed TYPE1
offspring/fine wiring. It must not be interpreted as an efficiency result or
as evidence of independent basins. In particular, the Al31 near-repeat set is
an explicit stress test of matcher limits. If matching or fresh qualification
fails, preserve the paid evidence and report no proposal rather than altering
the inputs.

## Full-walk functional follow-up (frozen before execution)

After the initial wiring probe, run the same inputs and numerical/search
parameters with quick_steps=1, generation_steps=1, fine_steps=1 for seeds7/19.
There are four arms, each still capped at4000 paid E/F requests and60 seconds;
total search allowance16000. This addresses the missing nonzero SSW attempts
in quick/offspring, not a parameter optimization or an efficiency experiment.
Use `--full-walk --seed 7` or `--full-walk --seed 19` during preparation in
new output directories. Preserve all prior results. Verify actual record
counts and request costs per stage; a phase name alone is insufficient.
No retries, additional seeds, input perturbations or matcher changes.
