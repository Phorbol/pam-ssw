"""Bounded state-vs-calculator discrimination on saved C60 preflight states."""
import argparse, importlib.util, json, os, pickle, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRE=HERE.parent/'c60-long-budget-20260924'
sys.path.insert(0,str(PRE/'source'))
import numpy as np
from ase import Atoms
from ase.io import read
from pamssw.standalone import (run_ssw,load_ssw_checkpoint,save_ssw_checkpoint,SSWConfig,
    NativeMCSettings,RecoveredRotationSettings)
from pamssw.standalone.surface import ASESurface

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
runner=module('runner',PRE/'preflight-r2-runner.py')

def canonical(x):
    if isinstance(x,Atoms):
        return dict(numbers=x.numbers.tolist(),positions=x.positions.tolist(),cell=x.cell.array.tolist(),
                    pbc=x.pbc.tolist(),masses=x.get_masses().tolist(),info=canonical(x.info))
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):canonical(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
    if hasattr(x,'__dict__'):return {'type':type(x).__name__,'state':canonical(vars(x))}
    return x

def dump(path,obj):path.write_text(json.dumps(canonical(obj),indent=2,allow_nan=False)+'\n')

def one(arm,mode):
    plan=json.loads((PRE/'preflight-r2'/arm/'start/plan.json').read_text())
    cp=load_ssw_checkpoint(PRE/'preflight-r2'/arm/'start/checkpoint.pkl' if mode=='warm' else HERE/arm/'cp2.pkl')
    folder=HERE/arm;folder.mkdir(exist_ok=True)
    calc=runner.calculator(plan);raw=ASESurface(calc);trace=[]
    cap=2500
    class Surface:
        requests=0
        def evaluate(self,atoms):
            if self.requests>=cap:raise RuntimeError('diagnostic request cap')
            self.requests+=1
            energy,force=raw.evaluate(atoms)
            trace.append((atoms.positions.copy(),energy,force.copy()))
            return energy,force
    surface=Surface();state_check={};second_start=None
    def boundary(snapshot):
        nonlocal second_start
        if mode=='warm' and snapshot.next_index==2:
            # Compare search-relevant live variables with the detached and disk snapshot.
            live=sys._getframe(1).f_locals
            fields=('initial','current','current_energy','best','minima','records','frozen','response','ls','native_mc_state')
            for name in fields:
                state_check[name]=canonical(live[name])==canonical(getattr(snapshot,name))
            state_check['rng']=canonical(live['rng'].bit_generator.state)==canonical(snapshot.rng_state)
            save_ssw_checkpoint(folder/'cp2.pkl',snapshot)
            loaded=load_ssw_checkpoint(folder/'cp2.pkl')
            state_check['disk_roundtrip']=canonical(snapshot)==canonical(loaded)
            second_start=len(trace)
        return False
    start=time.monotonic()
    result=run_ssw(read(PRE/'preflight-r2'/arm/'start/input.traj'),surface,
        steps=2 if mode=='warm' else 1,config=SSWConfig(**plan['ssw_config']),
        rng=np.random.default_rng(plan['seed']),checkpoint=cp,checkpoint_callback=boundary,
        mc=NativeMCSettings(plan['native_mc']['energy_tol_eV'],plan['native_mc']['maxtrap']),
        recovered_rotation=RecoveredRotationSettings(**plan['recovered_rotation']),ls=runner.settings(plan))
    save_ssw_checkpoint(folder/(mode+'.pkl'),result.checkpoint)
    if mode=='warm':trace=trace[second_start:] if second_start is not None else []
    np.savez_compressed(folder/(mode+'-trace.npz'),positions=np.array([x[0] for x in trace]),
                        energy=np.array([x[1] for x in trace]),forces=np.array([x[2] for x in trace]))
    dump(folder/(mode+'.json'),dict(status=result.status,requests=surface.requests,trace_requests=len(trace),
         elapsed_seconds=time.monotonic()-start,state_checks=state_check))
    if result.status!='completed':raise RuntimeError((arm,mode,result.status))


def analyze():
    rows=[]
    for arm in ('ssw','native-ls'):
        folder=HERE/arm
        a,b=[np.load(folder/(mode+'-trace.npz')) for mode in ('warm','cold')]
        warm=json.loads((folder/'warm.json').read_text())
        cold=json.loads((folder/'cold.json').read_text())
        row=dict(arm=arm,live_state_checks=warm['state_checks'],counts=[len(a['energy']),len(b['energy'])],
                 actual_requests=warm['requests']+cold['requests'])
        n=min(len(a['energy']),len(b['energy']))
        dp=np.max(np.abs(a['positions'][:n]-b['positions'][:n]),axis=(1,2))
        df=np.max(np.abs(a['forces'][:n]-b['forces'][:n]),axis=(1,2))
        de=np.abs(a['energy'][:n]-b['energy'][:n])
        equal=np.flatnonzero(dp==0)
        row['equal_geometry_count']=len(equal)
        row['max_force_difference_at_equal_geometry']=float(df[equal].max()) if len(equal) else None
        row['first_differences']=[dict(request=int(i),position_A=float(dp[i]),force_eV_A=float(df[i]),energy_eV=float(de[i])) for i in range(min(n,40)) if dp[i] or df[i] or de[i]]
        for threshold in (0.,1e-12,1e-8,1e-4):
            indices=np.flatnonzero(dp>threshold)
            row['first_geometry_gt_'+str(threshold)]=None if not len(indices) else int(indices[0])
        row['all_live_states_exact']=all(warm['state_checks'].values())
        rows.append(row)
    dump(HERE/'analysis.json',rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm');parser.add_argument('--mode');a=parser.parse_args()
    if a.arm:one(a.arm,a.mode)
    else:
        for arm in ('ssw','native-ls'):
            for mode in ('warm','cold'):
                subprocess.run([sys.executable,__file__,'--arm',arm,'--mode',mode],check=True)
        analyze()
