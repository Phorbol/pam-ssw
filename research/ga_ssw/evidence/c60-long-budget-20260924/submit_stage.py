"""Submit each authorized array once, with CPU dependency for the next stage.

This schedules the four predetermined segments, never retries a failed segment.
No more than four GPU tasks are queued at once (rush MaxSubmitPU is 10).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from production_runner import atomic_json
parser=argparse.ArgumentParser()
parser.add_argument('--stage',type=int,choices=(1,2,3,4),required=True)
a=parser.parse_args()
with (root/'.submission.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=root/'submission-ledger.json'
    ledger=json.loads(ledger_path.read_text())
    if ledger.get('submission_disabled', False):
        raise RuntimeError('experiment stopped: production submission disabled')
    qualification=json.loads((root/'preflight-qualification.json').read_text())
    if qualification['status']!='passed':raise RuntimeError('preflight not passed')
    if any(v['stage']==a.stage for v in ledger['submissions']):
        raise RuntimeError('stage already attempted; manual reconciliation required, never double-submit')
    if a.stage>1 and not any(v['stage']==a.stage-1 and 'job_id' in v for v in ledger['submissions']):
        raise RuntimeError('prior stage not recorded')
    entry=dict(stage=a.stage,status='submission_intent',unix=time.time(),
               array='0-3%2',allocation_hours_per_task=(12,12,12,8)[a.stage-1])
    ledger['submissions'].append(entry);atomic_json(ledger_path,ledger)
    submit=['sbatch','--parsable']
    if a.stage>1:
        prior=next(v for v in ledger['submissions'] if v['stage']==a.stage-1)
        submit.append('--dependency=afterany:'+prior['job_id'])
    submit.append(str(root/f'production-s{a.stage}.sbatch'))
    result=subprocess.run(submit,
                          text=True,capture_output=True,check=True)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('unexpected sbatch response: '+result.stdout)
    entry.update(status='submitted',job_id=job,sbatch_stderr=result.stderr)
    ledger['status']='production_submitted';atomic_json(ledger_path,ledger)
    if a.stage<4:
        command=shlex.join([sys.executable,str(__file__),'--stage',str(a.stage+1)])
    else:
        command=shlex.join([sys.executable,str(root/'readout.py')])
    next_job=subprocess.run(['sbatch','--parsable','--account=sjtu-caoxiaoming',
        '--partition=CPU-MISC','--qos=rush-cpu','--nodes=1','--ntasks=1','--time=00:05:00',
        '--dependency=afterany:'+job,'--job-name=c60-chain-'+str(a.stage),
        '--output='+str(root/'chain-%j.out'),'--error='+str(root/'chain-%j.err'),
        '--chdir='+str(root),'--wrap='+command],text=True,capture_output=True,check=True)
    successor=next_job.stdout.strip().split(';')[0]
    if not successor.isdigit():raise RuntimeError('unexpected successor sbatch response')
    entry['successor_cpu_job_id']=successor;atomic_json(ledger_path,ledger)
    print(json.dumps(entry,indent=2))
