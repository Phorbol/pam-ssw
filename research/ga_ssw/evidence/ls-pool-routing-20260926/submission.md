# Execution record — 2026-09-26

GPU array1500677 submitted from main9b63ba3 using frozen plan.json/run.sbatch.
Worker6703f35; source SHA in protocol.md matches the CPU-qualified source.
Analyzer59c85f7; CPU1500671 passed the current source and actual EMT artifacts.
All three branches were pushed. Core source is unchanged while jobs execute.

Six tasks0–5: C4H6 MC/uniform/PAM, then C60-isomer2 MC/uniform/PAM.
Raw outputs run-1500677-INDEX; view-1500677 contains only relative symlinks
to expose the analyzer's case/mode input layout. No raw data copied or rewritten.
CPU1500680 depends on afterany:1500677 and invokes analyze.sbatch: even missing
or failed GPU arms remain in the analysis. Analysis may exit nonzero for censor
or qualification/accounting errors; retain its output rather than retry search.

Account sjtu-caoxiaoming; GPU 8V100V0/rush-1o2gpu, one GPU per task,30minutes,
array concurrency2; CPU-MISC/rush-cpu one task,20minutes maximum for offline
graph analysis. Bounds remain480000 search+606 independent fresh calls,
3GPU-hours. No automatic rerun, model change, weight tuning, or second seed.

At submission: scheduler acceptance only. Numerical certificates and scientific
readout pending. Inspect GPU status at low frequency, approximately30minutes.
