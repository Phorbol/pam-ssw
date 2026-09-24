# Account versus submission environment, 2026-09-24

User challenged the prior attribution of repeated UID0 cancellations to an external administrator block. That attribution was premature: UID0 identifies cancellation authority, not a reason.

Visible association: user gengjianrui / account sjtu-caoxiaoming permits rush-1o2gpu; QOS MaxSubmitPU10, MaxTRESPU GPU16. Ancestor account limits and funds are not fully visible (PrivateData includes accounts/usage). `sai status --account sjtu-caoxiaoming --format table` returned credentials_unavailable, requiring an SAI Identity device session. No authentication, account change or funds transfer was attempted.

Readable `/opt/slurm/scripts/taskprolog.bash` can cancel jobs for mismatched GPU/CPU allocation. Its existence is a hypothesis source, not evidence that its guard fired in these jobs. Exact runtime guard inputs from the failed jobs remain unavailable.

All following probes use sjtu-caoxiaoming/rush-1o2gpu/4V100 and only print allocation environment, with no model/PES work:

|Job|Node|Limit|GPU option|Explicit submission export|Result|
|---|---|---|---|---|---|
|1480694|4v100n05|30s|gpus-per-node=1|none|COMPLETED0,1s; GPU1 CPU8|
|1480698|4v100n05|30s|gres=gpu:1|none|COMPLETED0,1s; GPU1 CPU8|
|1480712|4v100n05|19min|gpus-per-node=1|ALL,PYTHONFAULTHANDLER=1|CANCELLED by0,2s; no program log|
|1480722|4v100n05|19min|gpus-per-node=1|none|COMPLETED0,1s; GPU1 CPU8|

The last pair differs only in the submission export argument (apart from output filenames). This supports that argument as a reproducible failure trigger, rather than a persistent account or wall-limit denial; temporal effects and exact site internals are not fully excluded. Do not generalize this to all Slurm installations. Original1477853 signal11 on an unresponsive node is a separate unresolved failure.

The experiment script now follows the official gpus-per-node spelling and sets PYTHONFAULTHANDLER inside the shell script. Job1480730, same account and19min limit without sbatch export override, completed all12 intended cases in44s:202 search+24 fresh=226 requests. All24 independent forces qualified. Scientific geometry readout remains separate from this execution result.
