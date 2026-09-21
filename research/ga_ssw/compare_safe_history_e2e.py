"""Prepare by default; --run launches only the reviewed eight-run history ablation."""
import argparse,json,time,signal,shutil,hashlib,importlib.metadata,datetime
from pathlib import Path
import numpy as np
import networkx as nx
from ase import Atoms
from ase.collections import g2
from ase.io import read,write
from ase.calculators.emt import EMT
from pamssw.standalone import paper_reference as paper
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.native_ls import HC_BOND_ENERGIES,HC_BOND_LENGTHS
import pamssw.relax as relax
from research.ga_ssw.compare_vc_arms import serial
def graph(atoms,cutoffs):
 g=nx.Graph();g.add_nodes_from((i,{'Z':int(z)}) for i,z in enumerate(atoms.numbers))
 for i in range(len(atoms)):
  for j in range(i+1,len(atoms)):
   pair=tuple(sorted((int(atoms.numbers[i]),int(atoms.numbers[j]))))
   if np.linalg.norm(atoms.positions[i]-atoms.positions[j])<cutoffs[pair]:g.add_edge(i,j)
 return g,None
DEFAULT=Path('research/ga_ssw/evidence/safe-history-newseed-e2e-prepared')
class Budget(RuntimeError):pass
def dump(p,x):p.write_text(json.dumps(serial(x),indent=2,allow_nan=False))
def configurations():
 common=dict(rotation_bias=100.,fmax=.01,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,rotation_solver='dimer',cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total')
 return dict(C4H6=paper.SSWConfig(**common,width=.1,max_gaussians=25,temperature_K=150.,relax_steps=400),Cu13=paper.SSWConfig(**common,width=.2,max_gaussians=14,temperature_K=300.,relax_steps=200))
def prepare(out):
 out.mkdir(parents=True,exist_ok=False)
 inputs=dict(C4H6=g2['butadiene'].copy(),Cu13=read('research/ga_ssw/evidence/independent-cu13-surface/quenched.extxyz'))
 for name,a in inputs.items():write(out/f'{name}-input.extxyz',a)
 configs=configurations();schedule=[dict(system=system,seed=seed,memory=m) for seed in [29,71] for system in ['Cu13','C4H6'] for m in [10,400]]
 dump(out/'plan.json',dict(status='prepared_not_launched',denominator=8,schedule=schedule,steps=2,configs=configs,per_run_total_EF=6000,search_cap=5997,fresh_reserve=3,total_EF_cap=48000,total_wall_seconds=600,threads=1,primary='all valid force-qualified landings, including MC rejects; chemical integrity and near recurrence plus full costs',hypothesis='Does only memory10->400 improve full-search useful outcomes on known systems/new seeds?',limits='not unseen-system generalization; no Hessian/GM certificate; no native height; no default change',parameter_sources=dict(C4H6='compare_c4h6_native_paper_6000.py: raw ASE G2 trans-butadiene, width .1 NG25 T150 paperLS target .7eV/atom; original Safe400/dimer numerical config',Cu13='compare_cu13_safe_total.py: original quenched.extxyz source, width .2 NG14 T300 Safe200; dimer branch fixed prospectively',memory='production10 vs original-native-derived400, no other changes',schedule='new seeds29/71; cheap Cu precedes molecular pair eachseed; memory10 then400; shared wall censoring remains explicit'),paper_LS=dict(target_per_atom=.7,initial_fraction=.03,xi=.2,learning_rate=1.8,bond_energies=HC_BOND_ENERGIES,cutoffs={k:v+.1 for k,v in HC_BOND_LENGTHS.items()}),backend=dict(C4H6='tblite0.7 GFN2-xTB accuracy .001',Cu13='ASE EMT'),failure='failed physical requests count; preserve all8 rows including not_started_wall; no retries/extensions',source_inputs={k:serial(v) for k,v in inputs.items()}))
 shutil.copy2(__file__,out/'runner.py');shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 dump(out/'source-sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('pamssw').rglob('*.py')})
 return out

def geometry(a,reference,system):
 d=a.get_all_distances();dr=reference.get_all_distances();mask=np.triu_indices(len(a),1)
 x=reference.positions-reference.positions.mean(0);y=a.positions-a.positions.mean(0);u,s,v=np.linalg.svd(y.T@x);rot=u@np.diag([1,1,np.linalg.det(u@v)])@v
 info=dict(min_distance=float(d[mask].min()),radius_gyration=float(np.sqrt(np.mean(np.sum(y*y,axis=1)))),aligned_same_atom_rmsd=float(np.sqrt(np.mean(np.sum((y@rot-x)**2,axis=1)))),sorted_pairdistance_rms=float(np.sqrt(np.mean((np.sort(d[mask])-np.sort(dr[mask]))**2))),identity_limit='diagnostics only, not exact symmetry/permutation basin certificate')
 if system=='C4H6':
  cut={k:v+.1 for k,v in HC_BOND_LENGTHS.items()};g,_=graph(a,cut);gr,_=graph(reference,cut)
  info.update(components=nx.number_connected_components(g),edges=list(g.edges),degrees=list(dict(g.degree()).values()),same_graph=nx.is_isomorphic(g,gr,node_match=lambda p,q:p['Z']==q['Z']))
 return info

def run(out):
 if (out/'runs').exists():raise ValueError('existing run directory; no implicit retry')
 hashes=json.loads((out/'source-sha256.json').read_text())
 if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v for p,v in hashes.items()):raise ValueError('prepared source changed; prepare a new reviewable artifact')
 plan=json.loads((out/'plan.json').read_text());configs={k:paper.SSWConfig(**v) for k,v in plan['configs'].items()};inputs={k:Atoms(**v) for k,v in plan['source_inputs'].items()};rows=[dict(x,status='not_started',search_requests=0,fresh_requests=0,checks=[]) for x in plan['schedule']]
 dump(out/'execution.json',dict(status='running',started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions={k:importlib.metadata.version(k) for k in ['ase','numpy','tblite']}));shutil.copy2(__file__,out/'executed-runner.py')
 (out/'runs').mkdir();start=time.monotonic();deadline=start+600;total=0;oldmemory=relax._SAFE_LBFGS_MEMORY;assert oldmemory==10
 def alarm(*_):raise Budget('shared600s wall')
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,600)
 try:
  for row in rows:
   if time.monotonic()>=deadline:row['status']='not_started_wall';continue
   name=row['system'];a=inputs[name];rundir=out/'runs'/f'{name}-seed{row["seed"]}-m{row["memory"]}';rundir.mkdir();used=0;t0=time.monotonic();last={}
   def calc():
    if name=='Cu13':return EMT()
    from tblite.ase import TBLite
    return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
   with (rundir/'evaluations.jsonl').open('w') as log:
    class Counted(ASESurface):
     def __init__(self,fresh=False):super().__init__(calc());self.fresh=fresh
     def evaluate(self,atoms):
      nonlocal used,total
      if used >= (6000 if self.fresh else 5997) or total>=48000 or time.monotonic()>=deadline:
       row['budget_censored']=True;raise Budget('run/shared budget')
      used+=1;total+=1;row['fresh_requests' if self.fresh else 'search_requests']+=1
      try:
       e,f=super().evaluate(atoms);entry=dict(request=used,fresh=self.fresh,atoms=atoms,energy=e,forces=f);last.clear();last.update(entry);log.write(json.dumps(serial(entry))+'\n');log.flush();return e,f
      except Exception as exc:log.write(json.dumps(serial(dict(request=used,atoms=atoms,error=repr(exc))))+'\n');log.flush();raise
    relax._SAFE_LBFGS_MEMORY=row['memory']
    try:
     ls=None if name=='Cu13' else paper.LSSettings(HC_BOND_ENERGIES,{k:v+.1 for k,v in HC_BOND_LENGTHS.items()},target_per_atom=.7)
     result=paper.run_ssw(a,Counted(),steps=2,config=configs[name],rng=np.random.default_rng(row['seed']),ls=ls);dump(rundir/'result.json',result)
     row.update(status=result.status,step_statuses=[x.status for x in result.records],accepted=[x.accepted for x in result.records],valid_minima=len(result.minima),responses=[x.energy_response for x in result.records])
     for i,m in enumerate(result.minima):
      e,f=Counted(True).evaluate(m.atoms);row['checks'].append(dict(index=i,atoms=m.atoms,energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),force_pass=bool(np.linalg.norm(f,axis=1).max()<=.01),energy_error=e-m.energy,geometry=geometry(m.atoms,result.initial.atoms,name)));dump(rundir/'fresh.json',row['checks'])
    except Exception as exc:row.update(status='censored' if isinstance(exc,Budget) or time.monotonic()>=deadline or used>=5997 else 'error',error=repr(exc));dump(rundir/'last-successful-evaluation.json',last)
    finally:
     relax._SAFE_LBFGS_MEMORY=oldmemory
     if row.get('budget_censored') or time.monotonic()>=deadline:row['status']='budget_censored'
     row['seconds']=time.monotonic()-t0;row['pending_fresh']=max(0,row.get('valid_minima',0)-len(row['checks']));dump(rundir/'summary.json',row);dump(out/'summary.json',dict(runs=rows,total_requests=total,seconds=time.monotonic()-start));print(json.dumps({k:serial(v) for k,v in row.items() if k!='checks'}),flush=True)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);relax._SAFE_LBFGS_MEMORY=oldmemory
  execution=json.loads((out/'execution.json').read_text());execution.update(status='finished',ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),total_requests=total,seconds=time.monotonic()-start);dump(out/'execution.json',execution)
  dump(out/'summary.json',dict(runs=rows,denominator=8,total_requests=total,seconds=time.monotonic()-start))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--run',action='store_true');args=parser.parse_args()
 if args.run:run(args.output)
 else:prepare(args.output)
