"""Zero-PES shared-filesystem accounting benchmark; no algorithm tuning."""
import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
rows=[]
for label,path in [('per_request',HERE.parent/'c60-long-budget-20260924/preflight-r2-runner.py'),
                   ('reserved64',REPO/'research/ga_ssw/c60_long_budget.py')]:
    spec=importlib.util.spec_from_file_location(label,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    folder=HERE/('ledger-'+label);folder.mkdir()
    plan=dict(search_cap=1024,fresh_cap=3,wall_seconds=600)
    (folder/'plan.json').write_text(json.dumps(plan)+'\n')
    ledger=m.Budget(folder,plan);ledger.begin(120)
    started=time.monotonic()
    for _ in range(128):ledger.charge('search')
    ledger.finish('paused')
    elapsed=time.monotonic()-started
    rows.append(dict(mode=label,synthetic_accounting_operations=128,actual_PES_requests=0,
                     elapsed_seconds=elapsed,state=ledger.state))
(HERE/'ledger-benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
