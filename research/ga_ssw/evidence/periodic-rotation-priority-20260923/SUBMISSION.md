# Bounded experiment submission

2026-09-23. User authorized appropriate bounded research computations; no new model or public algorithm change. Group account sjtu-caoxiaoming.

- CPU1467850: configuration-only preflight passed (zeroPES).
- CPU1467864: rejected unused rotation_bias0 placeholder, corrected to positive100 required by SSWConfig. Original exception retained.
- CPU1467872: reached deliberate oracle sentinel, but preflight asserted wrong exception wrapper. Corrected test to exact RuntimeError sentinel; core/runner unchanged.
- CPU1467888: both physical inputs × both configurations pass public driver preflight to oracle boundary without model calls; preflight-v4.json.
- Independent static review: existing recovered-CBD path supports fullPBC translation projection. Fixed exception checkpoint-restore method names; harness hash updated. No runtime search affected by preparation corrections.
- GPU1467906: sbatch --wait gpu.sbatch; one4V100/rush-1o2gpu,1GPU,50min hard limit,4arms12000request and700s caps. Total48000search+8fresh. Sourcee41cc25 under source/pamssw.

Do not reinterpret submitted/running as completed or physically qualified. Final result and costs pending. Source/harness/input hashes checked by runner before model creation. No automatic retries or extensions.

Offline analysis job1468136 submitted with `--dependency=afterany:1467906`, CPU-MISC/rush-cpu, 5min, zeroPES; scheduler dependency prevents reading active GPU outputs. Partial output is saved after each arm. This replaces manual completion polling, not an additional search.
