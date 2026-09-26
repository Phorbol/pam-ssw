# Execution blocked before scientific results

2026-09-26, first scheduled status review at 21:56 Asia/Shanghai.

GPU array 1501316 (six arms) and its dependent CPU analysis 1501323 were
all `CANCELLED by 0` after 1–2 seconds. GPU starts occurred at 21:29:09–12
on 8v100v0n01/02; CPU analysis started at 21:30:16 on dpn01. No expected
worker directories, scientific summaries or Slurm stdout files were present.
There is no algorithm result or measured E/F/stress cost to compare. These
are missing scientific arms, not six numerical optimizer failures.

The saved `execution-1501316.tsv` includes successful qualification 1501259
and CPU preparation 1501315 for comparison. Account, GPU partition/QOS and
requested/allocated resources match the earlier successful GPU job (one GPU,
six allocated CPUs). CPU preparation and failed analysis both used CPU-MISC,
rush-cpu, the same group account and two allocated CPUs on dpn01. None of
these facts identifies the cancellation cause.

Readable site TaskProlog has GPU CPU-ratio guards, but accounting alone does
not reveal its actual runtime environment. Its GPU checks also do not explain
the CPU cancellation. Node-side Slurmd logs are not readable by this user;
the completed job records are no longer available through `scontrol show job`.
No resource overrides, account changes, algorithm changes or retries were made.

To resolve: obtain Slurmd/TaskProlog and controller cancellation evidence for
these job IDs and times, including who/what invoked cancellation. If relevant,
obtain the TaskProlog values of SLURM_JOB_PARTITION, SLURM_GPUS_ON_NODE,
SLURM_CPUS_ON_NODE and SLURM_CPUS_PER_TASK. Do not request full environment
dumps containing unrelated credentials. Then make one evidence-led correction
or retry under the existing bounded protocol. The completed 9-atom qualification
remains valid; the held-out 72-atom comparison has no scientific outcome.
