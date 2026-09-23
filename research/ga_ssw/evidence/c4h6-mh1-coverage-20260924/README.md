# Frozen C4H6 / MH-1 coverage comparison

Completed execution: all six tasks of array1469796 exited0 after400 attempts,
without budget censoring. CPU1470001 completed the independent offline readout
(errors=[]). Search:1549156 E/F requests,1363926 actual calculator invocations;
fresh:2405/2405 qualified. One SSW seed67 attempt stopped at nonpositive_height
(cost204 requests), preserving its failed-attempt position. It is not a failed
true-PES quench; no landing was produced. Five arms returned401 structures and
this arm400, including each initial minimum. No outcome revised the protocol.

At the fixed200000-request prefix, connected graph classes (including the
initial graph) are SSW5/5, paper-LS6/8, native-inspired-LS10/8 across seeds61/67.
Fragmented classes are separately2/4,7/7,3/6. This supports further qualification
of connected coverage; graph counts alone do not certify stable isomers or
chemical accuracy. The three-frame pilot curvature screen already found one
force-qualified but negative-curvature butadiene geometry. A separate, bounded
representative check is being prepared, without new search or retuning.

Native LS made130 normal strength updates in each400-step trajectory, matching
the recovered cadence. Its last response was0.4704/0.4698 eV/atom, versus
paper LS0.7000/0.6999. Thus the pilot's weak native response was a short-run
observation, not evidence that its long-run softening remains negligible.
These are controller-mechanism observations, not evidence that either response
level is optimal or that one LS implementation universally wins.

[Full readout](report.md); detailed per-frame analysis remains `analysis.json`
on shared storage. No claim of paper-PBE or full LASP trajectory reproduction.

- [Fixed protocol](plan.md) and [machine-readable plan](plan.json).
- `gpu.sbatch`: task0/1 SSW61/67, task2/3 paper-LS61/67, task4/5 native-LS61/67.
- `run.py`: one arm; independent fresh checks and completed/budget-censored/
  failed status are separate. Core tree remains frozen to the pilot.
- `analyze.py`, `analyze.sbatch`: common200000-request prefix, full endpoint,
  global graph identity, connected/dissociated classes and LS responses.
- `check_analysis.py`, `check-analysis.sbatch`: CPU1469991 passed using existing
  pilot traces and an explicitly synthetic no-landing record for parser checks;
  zeroPES. [Results](analysis-preflight.json). This is not a new scientific run.

Preparation verification: AST, shell syntax, standard-library provenance
preflight, independent source review, and real archived-data parser checks.
One report label was subsequently corrected from generic LS updates to native
update records: paper LS does not emit that native event. Its recorded energy
responses are counted separately, with no assumed event count from empty fields.

The separate paper-LS checkpoint-counter defect was repaired on an isolated
branch, verified with21 core/restart and2 pool tests, and merged only after all
six frozen runs finished (merge f142a26). The archived experiment core copies drop the response object's diagnostic steps and
last_response, but keep the frozen penalty amplitudes; the update formula does
not use either dropped field. These archived runs were not resumed and are not altered.
Response observations in each outer record remain available; do not reconstruct
controller state from the defective counter in these archived checkpoints.

Large raw files remain in each arm directory on shared storage. Protocol,
runner, compact summaries and derived readout are selectively archived inGit;
no copying of the entire repository or model. Do not overwrite or auto-retry a
failed run. Exact execution HEAD and model/input provenance are in arm summaries.
