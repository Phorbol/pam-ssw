# Frozen C4H6 / MH-1 coverage comparison

Current execution: array1469796, six tasks, at most2 concurrent V100s; status
and results are not inferred from submission. CPU1470001 is an afterany
dependency on the entire array and runs the offline readout even if an arm fails.
No outcome has been used to revise this protocol.

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

The separate paper-LS checkpoint-counter defect is under repair in an isolated
worktree. Current core copies drop the response object's diagnostic steps and
last_response, but keep the frozen penalty amplitudes; the update formula does
not use either dropped field. Current runs are not resumed and are not altered.
Response observations in each outer record remain available; do not reconstruct
controller state from the defective counter in these archived checkpoints.

Large raw files remain in each arm directory on shared storage. Protocol,
runner, compact summaries and derived readout are selectively archived inGit;
no copying of the entire repository or model. Do not overwrite or auto-retry a
failed run. Exact execution HEAD and model/input provenance are in arm summaries.
