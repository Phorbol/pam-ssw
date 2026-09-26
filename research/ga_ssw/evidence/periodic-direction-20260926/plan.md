# Periodic local direction approved implementation

Base ba5bca6 plus staged gap-driven plan. User approved independent periodic geometry.
No VC, no new radius/score/optimizer, old nonperiodic default preserved.
Two disjoint Luna implementation tasks (geometry, controller state), root entrypoints and
end-to-end evidence. New tests precede code; first entry RED1498804 (2 failed,1 passed).
That log remains in constrained-direction-20260926, not a physical result.

Cases: cases.json resolves existing paper-derived structures. Formula and fake-surface
checks are implementation only. Actual OMAT end-to-end protocol will freeze after
interfaces pass; no high-budget production task or spontaneous extension. No claim of
phase prediction from fixed-cell input or equality to original DFT/G-NN paper energies.

Frozen physical qualification protocol (before execution):3 cases x2 arms,3 outer
steps each,6000 live search requests/arm,360s/arm; <=36000 live requests,
<=18000 same-stream replay requests,<=24 independent endpoint checks; oneV100
45min job. Local arm uses same recorded real oracle stream for split1+2 state
verification, not an independent trajectory. Global arm uses identical recovered
rotation settings; only direction generation/memory policy differs. Actual
parameters in qualify.py/protocol.json. These are development cases, not holdout.
Failures/censoring stop each arm and remain in denominators, no budget increase.
Qualification requires force-certified endpoints and actual local direction use;
report multi-Gaussian memory coverage explicitly if no attempt exercises it.
No efficacy claim from3steps; use costs/events to design later fixed-protocol tests.

GPU submitted1498877 on8V100V0/rush-1o2gpu/accountsjtu-caoxiaoming.
Source frozen as source.patch relativeba5bca6 plus periodic_direction.py.
Final geometry9tests1498873 passed; integrated149tests1498859 passed before
last fixed-reference geometry regression. Root review identified draw_count
telemetry when only a fixed second reference is eligible: RED1498901 (1failed).
Only diagnostic count is wrong; defer one-line fix until frozen GPUjob ends.
No RNG/geometry changes required; preserve physical-source snapshot.

Surface followup candidate located: uploaded TYPE4-TiO2@Au24O4,514atoms,
original input has physical fixed1..297, direction exclusion1..351; segmentation
support1..486 is NOT the physical fixed mask. Existing source protocol/results
in adjacentga-ssw-behavior-parity/evidence/type4-source-direction-mace-v100
(correct full path underresearch/ga_ssw) show feasibility/cost; source must not
be cropped or cell-PBC changed merely to make new tests cheap. Not submitted.

1498877 search completed all6arms,3outersteps/75Gaussians each,6943live requests.
Runner then accessedresult.best.energy althoughbest is Atoms, causing reporting
failure beforefresh/replay. Preserve originalruns/summary and script. This is
runner error, not algorithm failure; no search retry. verify_saved.py loads
trusted savedcheckpoints and exact original source inputs, fresh-checks24minima
and replays3028requests, <=10minV100, job1498955. No newsearch.
Fixed-reference draw-count-only correction afterjobterminal: RED1498901 and
GREEN1498953(19passed). Physical oldsource remains source.patch+helpercopy;
all-active bulk drawcounts unchanged, permitting state replay.
