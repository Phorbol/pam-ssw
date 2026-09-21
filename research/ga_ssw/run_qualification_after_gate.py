"""Reviewed, one-shot qualification trigger; wait adds no PES or budget retries."""
from pathlib import Path
import json,os,subprocess,time

ROOT=Path(__file__).resolve().parents[2]
GATE=ROOT/'research/ga_ssw/prospective/complex-vc-feasibility'
STATE=GATE/'qualification-trigger.json'
PLAN=ROOT/'research/ga_ssw/material-gate-qualification-plan.json'

def main():
    # Exclusive state creation prevents repeated agents/turns launching duplicates.
    with STATE.open('x') as f:
        json.dump(dict(status='waiting_for_eight_terminal',pid=os.getpid(),started=time.time(),
            authorization='root reviewed fixed 6000 EFS / 3600 s CPU plan; user authorized routine resources'),f,indent=2)
    def save(**extra):
        state=json.loads(STATE.read_text());state.update(extra);STATE.write_text(json.dumps(state,indent=2))
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        state=json.loads((GATE/'execution.json').read_text())
        if state.get('status')!='running':
            terminal=len(state.get('runs',[]))==8 and all('returncode' in r for r in state['runs'])
            if not terminal:
                save(status='blocked_incomplete_gate',finished=time.time());return
            break
        time.sleep(10)
    else:
        save(status='wait_timeout',finished=time.time());return
    plan=json.loads(PLAN.read_text())
    env=os.environ.copy();env.update(plan['environment']);env['PYTHONPATH']=str(ROOT)
    env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    save(status='qualification_running',launched=time.time(),argv=plan['command'])
    with (GATE/'qualification.log').open('x') as log:
        process=subprocess.Popen(plan['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        save(qualification_pid=process.pid)
        code=process.wait()
    save(status='process_completed' if code==0 else 'process_failed',returncode=code,finished=time.time())

if __name__=='__main__':main()
