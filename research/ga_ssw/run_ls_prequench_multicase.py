"""Frozen four-arm fixed-cell LS threshold API check; 180s wall cap, no fitting."""
from pathlib import Path
import json, os, shutil, subprocess, sys, time, hashlib

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research/ga_ssw/evidence/constrained-ls-prequench-decoupled-20260912'
BASE=ROOT/'research/ga_ssw/evidence/constrained-gaussian-reference-20260912'

def main():
    OUT.mkdir(exist_ok=False)
    source=OUT/'source';shutil.copytree(ROOT/'pamssw',source/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    original=(ROOT/'research/ga_ssw/run_hookean_multicase.py').read_text()
    runner=original.replace('from dataclasses import asdict, is_dataclass','from dataclasses import asdict, is_dataclass, replace')
    runner=runner.replace('    args=parser.parse_args()',"    parser.add_argument('--case',required=True);parser.add_argument('--variant',required=True)\n    args=parser.parse_args()")
    runner=runner.replace("    plan=dict(purpose=", "    from pamssw.standalone import LSPrequenchSettings\n    ls_settings={k:tuple(replace(v,prequench=LSPrequenchSettings(.1,300)) for v in values) for k,values in ls_settings.items()}\n    cases={args.case:cases[args.case]}\n    plan=dict(purpose=")
    runner=runner.replace("variants=['ssw','ls_paper','ls_native']", "variants=[args.variant]")
    runner=runner.replace("ls=None if i==0 else ls_settings[name][i-1]", "ls=ls_settings[name][0 if variant=='ls_paper' else 1]")
    runner=runner.replace("    shutil.copytree('pamssw',out/'source/pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))", "    # Source is frozen once by the supervising runner.")
    (OUT/'arm_runner.py').write_text(runner);shutil.copy2(__file__,OUT/'supervisor.py')
    arms=[(c,v) for c in ('water_dimer','cu111') for v in ('ls_paper','ls_native')]
    plan=dict(arms=arms,denominator=4,only_parameter_change='softened prequench fmax .03 -> .1; steps300 retained',reference=str(BASE),seed=11,steps=1,per_arm_wall_limit=90,total_wall_limit=180,per_arm_search_cap=3998,per_arm_fresh_cap=2,source='source/pamssw',runner_sha256=hashlib.sha256(runner.encode()).hexdigest())
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    env=os.environ.copy();env.update(PYTHONPATH=str(source)+':/tmp/pam-ssw-tblite-20260909',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    begin=time.monotonic();rows=[]
    for case,variant in arms:
        remain=180-(time.monotonic()-begin);folder=OUT/(case+'-'+variant)
        if remain<=0:
            rows.append(dict(case=case,variant=variant,status='not_run_total_wall_cap'));continue
        with (OUT/(case+'-'+variant+'.log')).open('w') as log:
            cmd=[sys.executable,str(OUT/'arm_runner.py'),'--execute','--output',str(folder),'--case',case,'--variant',variant]
            try:
                result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(90,remain))
                status='process_completed' if result.returncode==0 else 'process_failed'
            except subprocess.TimeoutExpired:status='wall_censored'
        summary=folder/'summary.json'
        row=json.loads(summary.read_text())[0] if summary.exists() else dict(case=case,variant=variant,status=status)
        old=json.loads((BASE/(case+'-'+variant)/'summary.json').read_text())
        row['reference_summary']=old;rows.append(row)
        (OUT/'comparison.json').write_text(json.dumps(dict(denominator=4,rows=rows,elapsed_seconds=time.monotonic()-begin),indent=2)+'\n')
        print(case,variant,row['status'],row.get('search_requests'),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(dict(denominator=4,rows=rows,elapsed_seconds=time.monotonic()-begin),indent=2)+'\n')
if __name__=='__main__':main()
