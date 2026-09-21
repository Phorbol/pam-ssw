"""Uploaded TYPE3 water15, independent GFN2 fixed-cell SSW/GA lifecycle."""
from pathlib import Path
import json,time,shutil,hashlib
import numpy as np
from ase import Atoms
from tblite.ase import TBLite
from pamssw.standalone import ASESurface,SSWConfig,PaperGAConfig,run_ssw,run_ga_ssw
from research.ga_ssw.compare_vc_arms import serial

def dump(p,x):p.write_text(json.dumps(serial(x),indent=2,allow_nan=False)+'\n')
def main():
 out=Path('research/ga_ssw/evidence/uploaded-water15-fixed-20260912');out.mkdir(exist_ok=False)
 fixture=Path('tests/fixtures/ga_ssw/water.json');data=json.loads(fixture.read_text());frame=data['frames'][0]
 atoms=Atoms(numbers=frame['numbers'],positions=frame['positions'],pbc=False)
 groups=tuple(tuple(range(i,i+3)) for i in range(0,45,3));bonds={(i,j):v for i,j,v in data['bond_lengths']}
 assert all(sorted(atoms.numbers[list(g)])==[1,1,8] for g in groups)
 config=SSWConfig(width=.1,rotation_bias=100,max_gaussians=3,temperature_K=50,
   fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,
   rotation_solver='dimer',cluster_frame='direction_only',direction_sampling='global',quench_optimizer='safe-lbfgs-total')
 ga=PaperGAConfig(quick_steps=1,generations=1,generation_steps=1,fine_steps=1,ga_candidates=1,
   regions=1,fine_regions=1,quench_fmax=.01,quench_steps=400,proposal_max_batches=1,
   proposal_max_cut_attempts=100,proposal_max_pair_attempts=100,partition_max_draws=100,
   projection_tolerance=.0001,energy_window=1.5,proposal_type=3)
 dump(out/'plan.json',dict(input=atoms,source=data['source'],fixture=str(fixture),seeds=[11,29],arms=['ssw','ga'],
   config=config,ga_config=ga,groups=groups,backend='tblite0.7 GFN2-xTB accuracy .001, CPU',
   per_arm_search_cap=3000,fresh_reserve=10,wall_seconds_per_arm=120,
   identity='legacy projection only; archive count is not basin diversity',
   scope='Uploaded 45atom TYPE3 actual structure and segmentation. ARC stored PBC is overridden by source IfPer=0. Native NN energies are not used. Three Gaussian stages and one GA generation are bounded lifecycle tests, not paper production budgets.',
   descriptor_source='Recovered water.json native reference descriptors/radii; neighbor2.0 and NNAWei from configure.non',
   collision_limits={(1,1):1.,(1,8):.5,(8,8):2.}))
 shutil.copy2(__file__,out/'runner.py');shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 dump(out/'source-manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')})
 summaries=[]
 for arm in ['ssw','ga']:
  for seed in [11,29]:
   d=out/f'{arm}-{seed}';d.mkdir();start=time.monotonic();row=dict(arm=arm,seed=seed)
   class Counted(ASESurface):
    denied=0
    def evaluate(self,a):
     if self.requests>=3000 or time.monotonic()-start>120:
      self.denied+=1;raise RuntimeError('water15 bounded request/wall cap')
     e,f=super().evaluate(a)
     with (d/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(serial(dict(request=self.requests,energy=e,forces=f,atoms=a)))+'\n')
     return e,f
   surface=Counted(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0));checks=[]
   try:
    if arm=='ssw':
     r=run_ssw(atoms,surface,steps=2,config=config,rng=np.random.default_rng(seed));landings=[(i,m.atoms,m.energy) for i,m in enumerate(r.minima)]
     row.update(step_statuses=[x.status for x in r.records],accepted=[x.accepted for x in r.records])
    else:
     r=run_ga_ssw([atoms],surface,groups=groups,references=data['reference_descriptors'],descriptor_bonds=bonds,
       descriptor_weights=(.3,.2,.2,.1,.1,.1),neighbor_range=2.,proposal_bond_limits={(1,1):1.,(1,8):.5,(8,8):2.},
       config=ga,ssw_config=config,rng=np.random.default_rng(seed))
     landings=[(x['id'],x['atoms'],x['energy']) for x in r.archive]
     row.update(stages=[dict(phase=x.phase,status=x.status,requests=x.evaluation_requests)for x in r.stages],failures=[x.reason for x in r.failures],identity_mode=r.identity_mode)
    dump(d/'result.json',r);row.update(status=r.status,ledger_consistent=r.evaluation_requests==surface.requests,landings=len(landings))
    fresh=ASESurface(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0))
    for i,a,e0 in landings[:10]:
     e,f=fresh.evaluate(a);checks.append(dict(id=i,energy=e,energy_error=e-e0,fmax=float(np.linalg.norm(f,axis=1).max()),certified=bool(np.linalg.norm(f,axis=1).max()<=.01)))
    row.update(fresh_requests=fresh.requests,unverified_landings=max(0,len(landings)-10))
   except Exception as exc:row.update(status='exception',error=repr(exc))
   row.update(search_requests=surface.requests,denied=surface.denied,checks=checks,wall_seconds=time.monotonic()-start)
   dump(d/'summary.json',row);summaries.append(row);dump(out/'summary.json',summaries);print(arm,seed,row,flush=True)
if __name__=='__main__':main()
