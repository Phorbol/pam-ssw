"""Zero-PES terminal accounting; preserve all four denominators and failures."""
import json
import subprocess
from pathlib import Path
from production_runner import atomic_json
root=Path(__file__).resolve().parent
rows=[]
for name in ('ssw-17101','native-ls-17101','ssw-17102','native-ls-17102'):
    folder=root/name
    if not (folder/'budget.json').exists():
        rows.append(dict(arm=name,status='not_started',search=0,fresh=0));continue
    state=json.loads((folder/'budget.json').read_text())
    summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    fresh=state['fresh_checks']
    charged=max(state['search'],state.get('search_reserved',state['search'])) if state['status']=='running' else state['search']
    uncertain=state.get('unconfirmed_search_reservations',0)+charged-state['search']
    rows.append(dict(arm=name,status=state['status'],search=charged,fresh=state['fresh'],
        unconfirmed_search_reservations=uncertain,
        reserved_gpu_seconds=state['reserved_seconds'],
        joint_fresh_candidates=[k for k,v in fresh.items() if v.get('joint_target',False)],
        physical_cage_review='pending for any graph-qualified candidate',
        best=fresh.get('best'),first_stored_joint_index=summary.get('first_stored_joint_index'),
        scientific_scope='MH1 model-specific two-input result; not general success rate'))
errors=[]
if sum(r['search'] for r in rows)>10000000:errors.append('total search budget exceeded')
if sum(r['fresh'] for r in rows)>12:errors.append('total fresh budget exceeded')
if sum(r.get('reserved_gpu_seconds',0) for r in rows)>176*3600:errors.append('GPU reservation budget exceeded')
atomic_json(root/'readout.json',dict(rows=rows,errors=errors,
    incomplete=[r['arm'] for r in rows if r['status'] in ('not_started','running','paused')],
    note='Fresh graph/energy candidates require final geometric review; missing summaries retained.'))
print(json.dumps(rows,indent=2))

ledger=json.loads((root/'submission-ledger.json').read_text())
job_ids=[v['job_id'] for v in ledger['submissions'] if 'job_id' in v]
if job_ids:
    accounting=subprocess.run(['sacct','-j',','.join(job_ids),'-n','-P',
        '--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode'],text=True,capture_output=True)
    (root/'slurm-accounting.txt').write_text(accounting.stdout)
    (root/'slurm-accounting.err').write_text(accounting.stderr)
    if accounting.returncode:print('Scheduler accounting unavailable; retained reservation/runner costs.')
