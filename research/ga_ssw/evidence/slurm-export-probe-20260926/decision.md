# Explicit sbatch export isolated as a startup failure trigger

The user identified `--export`/`--mem` as possible causes. Actual submission
commands and scripts contain explicit `--export=ALL,...`, but no explicit
memory setting. Automatic requested/allocated memory matches successful jobs.

One bounded pair, CPU-MISC/rush-cpu/sjtu-caoxiaoming, one-minute limit,
identical `probe.sbatch`, no calculator calls:

| Job | Submission difference | Outcome on dpn01 |
|---|---|---|
| 1502773 | no explicit export option | completed in 1 s, body start/end logged |
| 1502774 | `--export=ALL,SSW_EXPORT_PROBE=explicit` | cancelled by UID 0 in 1 s, no stdout |

This controlled observation supports explicit export as a trigger in this
environment, not a general Slurm restriction or an identified internal site
policy implementation. It does not establish that explicit memory caused any
of these jobs: memory was never explicitly requested. One paired observation
cannot exclude every transient platform effect; a successful corrected actual
task will qualify the recovery.

Correction: pass plan path and parent job ID as positional script arguments,
using normal sbatch environment inheritance. Keep account, QOS, resources,
input, calculator, algorithm parameters and scientific budget unchanged.
Validate shell syntax, then rerun the first scientific arm only before the
remaining five. Preserve all earlier cancellations and this probe. No need to
require inaccessible administrator logs before testing this supported correction.
