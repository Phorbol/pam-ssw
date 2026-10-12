# LJ38 full-search paired direction transfer

Status: harness/preflight only; no PES search has run. This is a finite-sample
development comparison, not a reproduction of the published 1000-start success
statistics and not a universal success-rate estimate.

The hard-case choice follows Wales and Doye, *J. Phys. Chem. A* **101**,
5111–5116 (1997), DOI [10.1021/jp970984n](https://doi.org/10.1021/jp970984n):
their discussion on pp. 1–2 identifies LJ38's competing fcc/icosahedral
funnels and explicitly contrasts it with the easier LJ55 landscape. The
Cambridge cluster database supplies the tabulated LJ38 reference
([LJ table, N=38](https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html));
the coordinates and matching potential contract used here are recorded below.
These paper claims motivate the benchmark choice only; the paired result will
be interpreted against the explicitly matched `FullPairLJ` potential below.

## Question and decision

Does the Oct. 7 full-direction configuration with its existing `per_atom`
radius policy improve discovery of the authoritative LJ38 global-minimum
structure over rotation-only search when both receive a full historical LJ38
search budget on the same difficult starting structures? The competing
explanations are that (A) the direction transfer improves target discovery,
(B) the short-panel energy signal does not transfer to target discovery, or
(C) both remain limited by shared search controls. A paired difference can
retain or reject this direction as a candidate for further testing on these
starts; it cannot establish general efficacy.

The sole treatment difference is `recovered_rotation` versus
`recovered_direction` with `c1_radius_policy=per_atom`. Use the frozen Oct. 7
`run_panel.py` settings for both arms. Do not tune settings or add components.

## Potential, target and initial structures

Use `FullPairLJ(epsilon=1 eV, sigma=2.7 Å)`, all atom pairs, no cutoff and
`pbc=False`; Ar is only an element label. The target is the Cambridge LJ38
reference at `research/ga_ssw/evidence/cluster-paper-reproduction-20260925/references/lj38.points`,
whose coordinates are in units of sigma. Its qualified energy is
−173.928427 eV. A candidate requires true quench convergence, energy no greater
than target +0.001 eV, maximum atom force no greater than 0.05 eV/Å, one
connected component under the existing 1.3 sigma graph, and a proper-rotation
plus atom-permutation match to the reference under the existing geometry
checker. A search hit is accepted only after independent fresh E/F re-evaluation
and the same checks. The 0.01 eV/Å force value is reported only as an
informational stricter subset; it does not replace the frozen 0.05 threshold.
The one-call CPU reference qualification repeats the pre-existing check in
`research/ga_ssw/evidence/cluster-paper-reproduction-20260925/lj-reference-qualification.json`:
energy within 1e-5 eV of the tabulated target and maximum force component below
0.04 epsilon/sigma. Its source protocol is `cluster-paper-reproduction-20260925/lj-pilot-plan.md`.

Use the archived raw, uniform-volume LJ38 structures from the full-budget
benchmark, seeds 25092501 and 25092502:

* `/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/cluster-paper-reproduction-20260925/compact-global-runs/lj38-seed25092501/initial.extxyz`
* `/home/gengjianrui/bin/pam-ssw-worktrees/c60-local-defect-qualification/research/ga_ssw/evidence/cluster-paper-reproduction-20260925/compact-global-runs/lj38-seed25092502/initial.extxyz`

The corresponding archived global and paper-direction input files are byte
identical per seed. The runner records path and SHA256 and verifies the paired
input digest before search. Each arm begins from that raw structure and runs
its own normal initialization quench; all those paid requests count against
the arm cap. No target, relaxed reference, archive minimum or prior checkpoint
is used as a search input.

## Budget, outcomes and costs

Run four trajectories (two starts × two arms), with at most 800,000 paid
search E/F requests including initialization, 2,000 outer steps, or 1,800
seconds per arm; whichever boundary occurs first is preserved. No automatic
retry, resume, or cap extension. At most two array tasks run concurrently.
Use the CPU-MISC/rush-cpu allocation and one CPU thread as in the source LJ
campaign. Fresh target-candidate validation is performed once if a target
candidate occurs. Otherwise independently recheck the initial and best minima,
up to two fresh requests. Failed and denied requests remain recorded.

The measured LJ38 historical runs each used 800,000 search requests in
797–866 seconds, supporting a 1,800-second per-arm wall cap. This experiment
therefore has a maximum of 3.2 million search requests. Four trajectories
allow eight independent per-arm E/F checks, plus one separate reference
qualification. The smoke check adds at most 4,000 paid search requests and four
fresh checks. With two concurrent slots the per-arm caps allow up to 60 minutes
wall time for the full matrix, excluding queueing (two waves of 30 minutes);
the aggregate full-search allocation is at most two CPU-hours. The smoke and
reference checks are separately bounded. The wall cap is an execution bound,
not a predicted completion time.

Report target hits only when all energy, force, connectivity, structure-match,
and fresh checks pass. Compare first-hit request cost and best qualified energy
at shared paid-request prefixes. Preserve non-hit, initialization failure,
budget cutoff, wall cutoff, callback error and geometry mismatch separately.
Do not treat outer-step counts, acceptance, energy-only candidates, or equal
endpoint energies as target discovery. These reused starts provide paired
development evidence only.

## Reproduction

Commit this campaign and keep its imported scientific source files clean at the
recorded commit before numerical qualification or search. From the repository
root, run
`/home/gengjianrui/.conda/envs/mace_env/bin/python research/ga_ssw/evidence/lj38-direction-target-20261012/run.py --preflight`
for zero-PES source, input, reference-geometry, and paired-digest checks. Then
submit `sbatch research/ga_ssw/evidence/lj38-direction-target-20261012/reference.sbatch`.
It independently evaluates the Cambridge coordinates once and writes
`runs/reference-JOBID.json`; this named artifact is never overwritten. Submit
the short integration check with
`sbatch research/ga_ssw/evidence/lj38-direction-target-20261012/smoke.sbatch research/ga_ssw/evidence/lj38-direction-target-20261012/runs/reference-JOBID.json`.
It runs both arms on seed 25092501, capped at one outer step, 2,000 paid
requests including startup and 120 seconds per arm. It checks the real kernel,
startup quench, target callback, accounting and artifact path without serving
as scientific outcome evidence. Review both smoke records before submitting
the full array:
`sbatch research/ga_ssw/evidence/lj38-direction-target-20261012/panel.sbatch research/ga_ssw/evidence/lj38-direction-target-20261012/runs/reference-JOBID.json`.
The full array refuses to start without a passed reference qualification that
matches its committed core and imported-source hashes. Array IDs map to
seed/arm in `panel.sbatch`; each run snapshots this runner and protocol and
writes provenance, effective settings, input digest, E/F ledger, outer-step
structures, checkpoint, final structure, and fresh qualification.
