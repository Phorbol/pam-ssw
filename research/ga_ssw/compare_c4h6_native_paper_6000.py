"""Bounded preregistered C4H6/GFN2: ordinary, paper-feedback and native-derived LS."""
import json,time,signal,shutil
from pathlib import Path
import numpy as np
import networkx as nx
from ase.collections import g2
from ase.io import write
from pamssw.standalone import paper_reference as paper,ls_cycle
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.ls_native_reference import NativeLSSettings
from pamssw.standalone.native_ls import HC_BOND_ENERGIES,HC_BOND_LENGTHS
from research.ga_ssw.compare_vc_arms import serial
from research.ga_ssw.compare_c4h6_safe_ls import graph

OUT=Path('research/ga_ssw/evidence/c4h6-native-paper-two-step-6000')
class Budget(RuntimeError):pass

def dump(path,data):path.write_text(json.dumps(serial(data),indent=2,allow_nan=False))

def main():
    out=OUT;out.mkdir(parents=True,exist_ok=False)
    atoms=g2['butadiene'].copy();write(out/'input.extxyz',atoms)
    config=paper.SSWConfig(width=.1,rotation_bias=100.,max_gaussians=25,temperature_K=150.,
        fmax=.01,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,
        rotation_solver='dimer',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total')
    cutoffs={k:v+.1 for k,v in HC_BOND_LENGTHS.items()}
    paper_ls=paper.LSSettings(HC_BOND_ENERGIES,cutoffs,target_per_atom=.7)
    native_ls=NativeLSSettings(HC_BOND_ENERGIES,HC_BOND_LENGTHS,target_mev_per_atom=700.)
    dump(out/'plan.json',dict(seeds=[3,17],arms=['ssw','paper','native'],steps=2,
        per_arm_total_EF_cap=6000,search_cap=5997,fresh_reserve=3,total_cap=36000,total_wall_seconds=600,threads=1,
        config=config,paper_ls=paper_ls,native_ls=native_ls,
        backend='tblite 0.7 GFN2-xTB accuracy .001; CPU',
        source='ASE G2 trans-butadiene; parameters match compare_c4h6_safe_ls',
        comparison='same raw geometry/config/target (.7 eV or 700 meV/atom); paper-feedback uses recovered RAW tables as earlier run; native uses normalized B/A and cycle',
        qualification='independent calculator per minimum within same per-arm cap; connectivity cutoffs raw lengths+.1; no Hessian/isomer identity proof',
        failure='no retry/retune; preserve incomplete logs; shared wall guard includes independent checks'))
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'script.py')
    from tblite.ase import TBLite
    def calc():return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
    started=time.monotonic();deadline=started+600
    def alarm(*_):raise Budget('total 600-second wall deadline')
    signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,600)
    original_quench=paper.quench;original_ls_quench=ls_cycle.quench;original_prepare=paper.prepare_ls_step
    baseline,_=graph(atoms,cutoffs);runs=[]
    try:
        for seed in (3,17):
            for arm in ('ssw','paper','native'):
                runout=out/f'{arm}-seed{seed}';runout.mkdir();row=dict(seed=seed,arm=arm,status='running',search_requests=0,fresh_requests=0,checks=[])
                used=[0];log=(runout/'evaluations.jsonl').open('w');quenches=[];preparations=[]
                class Counted(ASESurface):
                    def __init__(self,fresh=False):super().__init__(calc());self.fresh=fresh
                    def evaluate(self,a):
                        if used[0]>=(6000 if self.fresh else 5997) or time.monotonic()>=deadline:
                            row['budget_reached']=True
                            raise Budget('per-arm request/shared wall cap')
                        used[0]+=1
                        row['fresh_requests' if self.fresh else 'search_requests']+=1
                        try:
                            e,f=super().evaluate(a)
                            log.write(json.dumps(serial(dict(call=used[0],fresh=self.fresh,atoms=a,energy=e,forces=f)))+'\n');log.flush();return e,f
                        except Exception as exc:
                            log.write(json.dumps(dict(call=used[0],fresh=self.fresh,error=repr(exc)))+'\n');log.flush();raise
                surface=Counted();t0=time.monotonic()
                def observed_quench(*args,**kwargs):
                    q=original_quench(*args,**kwargs);quenches.append(dict(call=used[0],result=q));dump(runout/'quenches.json',quenches);return q
                def observed_prepare(*args,**kwargs):
                    soft=kwargs['softening'];p=original_prepare(*args,**kwargs)
                    preparations.append(dict(call=used[0],pairs=soft.pairs,strengths=soft.strengths,
                        response=p.energy_response,requests=p.evaluation_requests))
                    dump(runout/'preparations.json',preparations);return p
                paper.quench=observed_quench;ls_cycle.quench=observed_quench;paper.prepare_ls_step=observed_prepare
                try:
                    ls=None if arm=='ssw' else paper_ls if arm=='paper' else native_ls
                    r=paper.run_ssw(atoms,surface,steps=2,config=config,rng=np.random.default_rng(seed),ls=ls)
                    dump(runout/'result.json',r)
                    row.update(status=r.status,step_statuses=[x.status for x in r.records],accepted=[x.accepted for x in r.records],
                        energy_responses=[x.energy_response for x in r.records],minima=len(r.minima))
                    # Every recorded valid landing, including rejects; fresh uses a new calculator.
                    for i,m in enumerate(r.minima):
                        fresh=Counted(True);e,f=fresh.evaluate(m.atoms);g,distances=graph(m.atoms,cutoffs)
                        row['checks'].append(dict(index=i,energy=e,energy_error=e-m.energy,
                            fmax=float(np.linalg.norm(f,axis=1).max()),force_pass=bool(np.linalg.norm(f,axis=1).max()<=.01),
                            components=nx.number_connected_components(g),connectivity_same=nx.is_isomorphic(g,baseline,node_match=lambda x,y:x['Z']==y['Z']),
                            dihedral=m.atoms.get_dihedral(0,1,2,3),edges=list(g.edges),atoms=m.atoms,distances=distances))
                        dump(runout/'fresh-checks.json',row['checks'])
                except Exception as exc:
                    row.update(status='censored' if isinstance(exc,Budget) else 'error',error=repr(exc))
                finally:
                    paper.quench=original_quench;ls_cycle.quench=original_ls_quench;paper.prepare_ls_step=original_prepare
                    row.update(total_requests=used[0],wall_seconds=time.monotonic()-t0)
                    # Internal soft-prequench error wrapping can retain the budget message.
                    if time.monotonic()>=deadline or row.get('budget_reached',False):
                        row['budget_reached']=True;row['status']='censored'
                    runs.append(row);dump(runout/'summary.json',row);dump(out/'summary.json',dict(runs=runs,total_requests=sum(x['total_requests'] for x in runs),wall_seconds=time.monotonic()-started))
                    log.close();print(json.dumps({k:serial(v) for k,v in row.items() if k!='checks'}),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        paper.quench=original_quench;ls_cycle.quench=original_ls_quench;paper.prepare_ls_step=original_prepare
        dump(out/'summary.json',dict(runs=runs,total_requests=sum(x['total_requests'] for x in runs),wall_seconds=time.monotonic()-started,
            limits='two seeds/two attempts; no Hessian or independent basin identity; GFN2 model not DFT; unequal completed work must not imply efficiency gain'))

if __name__=='__main__':main()
