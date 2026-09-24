"""Shared-boundary MH1 resume qualification. Each call is a new process."""
import json, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
start=time.monotonic()
for arm in ('ssw','native-ls'):
    base=ROOT/'preflight'/arm
    def run(name,n):
        subprocess.run([sys.executable,str(ROOT/'runner.py'),'--run-dir',str(base/name),
                        '--segment-seconds','600','--max-attempts',str(n),'--execute'],check=True)
        state=json.loads((base/name/'budget.json').read_text())
        if state['status']!='paused':raise RuntimeError((arm,name,state['status']))
    run('start',1)
    for name in ('continuous','split'):
        shutil.copytree(base/'start',base/name)
    run('continuous',2)
    run('split',1)
    run('split',1)
# These numerical imports execute on the allocated GPU node.
import numpy as np
from pamssw.standalone import load_ssw_checkpoint
rows=[]
for arm in ('ssw','native-ls'):
    base=ROOT/'preflight'/arm
    a,b=[load_ssw_checkpoint(base/name/'checkpoint.pkl') for name in ('continuous','split')]
    states=[json.loads((base/name/'budget.json').read_text()) for name in ('start','continuous','split')]
    actual_paid=states[1]['search']+states[2]['search']-states[0]['search']
    row=dict(arm=arm,actual_paid_requests=actual_paid,indices=[a.next_index,b.next_index],
             requests=[a.evaluation_requests,b.evaluation_requests],rng_equal=a.rng_state==b.rng_state,
             max_coordinate_difference_A=float(np.abs(a.current.positions-b.current.positions).max()),
             boundary_differences_A=[float(np.abs(x.last_atoms.positions-y.last_atoms.positions).max()) for x,y in zip(a.records,b.records)],
             status_acceptance_equal=[(x.status,x.accepted) for x in a.records]==[(x.status,x.accepted) for x in b.records],
             checkpoint_bytes=(base/'continuous/checkpoint.pkl').stat().st_size)
    row['exact_match']=bool(row['requests'][0]==row['requests'][1] and row['rng_equal'] and row['max_coordinate_difference_A']==0 and row['status_acceptance_equal'])
    rows.append(row)
(ROOT/'preflight/results.json').write_text(json.dumps(dict(rows=rows,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
print(json.dumps(rows,indent=2))
