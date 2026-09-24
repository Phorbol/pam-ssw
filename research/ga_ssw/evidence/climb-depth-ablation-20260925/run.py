"""Saved-path truncation diagnostic. No SSW rerun or algorithm changes."""
import argparse, hashlib, importlib.util, json, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BASE=HERE.parent
MODEL=Path('/home/gengjianrui/.cache/mace/mace-mh-1.model')
MODEL_SHA='a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47'

def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()

def prepare():
    if (HERE/'inputs.json').exists():raise FileExistsError('inputs already frozen')
    rows=[];sources=[]
    for system,folder,seeds,arms,ng,steps in (
        ('C4H6','c4h6-mh1-coverage-20260924',(61,67),('paper_ls','native_ls'),25,400),
        ('C60','c60-recovered-rotation-ls-prospective-20260922',(17093,17094),('native_ls',),12,1000)):
        for arm in arms:
            for seed in seeds:
                name=f'{arm}-seed{seed}' if system=='C4H6' else f'c60_{seed}-seed{seed}'
                p=BASE/folder/name/'result.json';raw=p.read_bytes();d=json.loads(raw)
                candidates=[r['index'] for r in d['records'] if r['status']=='gaussian_limit' and len(r['climb'])==ng and r.get('landing') and r['landing']['converged']]
                selected=[0,199,399] if system=='C4H6' else [candidates[0],candidates[-1]]
                sources.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),selected=selected))
                current=d['initial']['atoms']
                for r in d['records']:
                    assert not r.get('starter_selection')
                    if r['index'] in selected:
                        assert r['index'] in candidates
                        stages=r['climb'];full=r['landing']
                        overhead=r['evaluation_requests']-full['evaluation_requests']-sum(e['requests'] for e in stages)
                        assert overhead>=0
                        for depth in (1,ng//2):
                            # Stage k+1 starts at the result of biased quench k.
                            assert stages[depth]['index']==depth
                            point=dict(current);point['positions']=stages[depth]['center']
                            rows.append(dict(case_id=f'{system}-{arm}-{seed}-r{r["index"]}-k{depth}',
                                system=system,arm=arm,seed=seed,record_index=r['index'],depth=depth,full_depth=ng,
                                start=current,point=point,full=full,
                                prefix_requests=overhead+sum(e['requests'] for e in stages[:depth]),
                                nonstage_overhead=overhead,original_full_requests=r['evaluation_requests'],
                                steps=steps,fmax=.03,memory=500))
                    if r['accepted']:current=r['landing']['atoms']
    assert len(rows)==32
    save(HERE/'inputs.json',dict(rows=rows,sources=sources,core_tree=git('rev-parse','HEAD:pamssw'),
         git_head=git('rev-parse','HEAD'),model_sha256=MODEL_SHA,search_cap_per_case=1000,fresh_cap_per_case=2,total_cap=32064))
    print('prepared32 cases, no PES')

def execute():
    started=time.monotonic();data=json.loads((HERE/'inputs.json').read_text())
    assert git('rev-parse','HEAD:pamssw')==data['core_tree'] and not git('diff','HEAD','--','pamssw')
    assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==MODEL_SHA
    if (HERE/'runs').exists():raise FileExistsError('no automatic retry/overwrite')
    save(HERE/'execution.json',dict(git_head=git('rev-parse','HEAD'),core_tree=data['core_tree'],runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=hashlib.sha256((HERE/'inputs.json').read_bytes()).hexdigest(),python=sys.version))
    import numpy as np, torch
    sys.path[:0]=[str(ROOT),str(ROOT/'research/ga_ssw')]
    from mace.calculators import MACECalculator
    from pamssw.standalone.surface import quench
    from analyze_c4h6_ls_reaction_coverage import atoms_from_dict
    spec=importlib.util.spec_from_file_location('ledger',BASE/'periodic-rotation-priority-20260923/ledger.py')
    ledger=importlib.util.module_from_spec(spec);spec.loader.exec_module(ledger)
    torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    kw=dict(model_paths=str(MODEL),head='omol',device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    calc=MACECalculator(**kw);fresh_calc=MACECalculator(**kw)
    deadline=started+1740 # 30min total allocation, 1min for reporting
    rows=[]
    for item in data['rows']:
        if time.monotonic()>=deadline:break
        folder=HERE/'runs'/item['case_id'];folder.mkdir(parents=True)
        search=ledger.CountedSurface(calc,folder/'requests.jsonl',1000,max(0,deadline-time.monotonic()))
        fresh=ledger.CountedSurface(fresh_calc,folder/'fresh.jsonl',2,max(0,deadline-time.monotonic()))
        row=dict(case_id=item['case_id'],status='failed',fresh=[]);q=None
        try:
            calc.reset();q=quench(atoms_from_dict(item['point']),search,fmax=item['fmax'],steps=item['steps'],optimizer='safe-lbfgs-total',lbfgs_memory=item['memory'])
            row.update(status='completed',quench=q)
        except Exception as exc:row['error']=repr(exc)
        for label,atoms in [('truncated',None if q is None else q.atoms),('full',atoms_from_dict(item['full']['atoms']))]:
            check=dict(label=label,status='missing')
            if atoms is not None:
                try:
                    fresh_calc.reset();e,f=fresh.evaluate(atoms);fm=float(np.linalg.norm(f,axis=1).max())
                    check.update(status='completed',energy=e,fmax=fm,force_qualified=fm<=item['fmax'])
                except Exception as exc:check.update(status='failed',error=repr(exc))
            row['fresh'].append(check)
        row.update(search_requests=search.requests,fresh_requests=fresh.requests,boundary=search.boundary)
        ledger.dump(folder/'result.json',row);rows.append(row)
        ledger.dump(HERE/'runs/summary.json',dict(status='running',rows=rows))
    ledger.dump(HERE/'runs/summary.json',dict(status='complete' if len(rows)==32 else 'wall_censored',rows=rows,
        elapsed_seconds=time.monotonic()-started,total_requests=sum(x['search_requests']+x['fresh_requests'] for x in rows)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.prepare and a.execute:p.error('choose one stage')
    if a.prepare:prepare()
    elif a.execute:execute()
