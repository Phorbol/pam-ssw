"""Two-material independent TYPE0 GA lifecycle audit with explicit identity."""
from pathlib import Path
import argparse,json,time,shutil,hashlib
import numpy as np
from ase.cluster import Icosahedron,Octahedron
from ase.geometry import distance
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface,SSWConfig,PaperGAConfig,run_ga_ssw
from pamssw.standalone.legacy_descriptor import cluster_descriptor
from research.ga_ssw.compare_vc_arms import serial

def dump(p,d):p.write_text(json.dumps(serial(d),indent=2,allow_nan=False)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cycles',type=int,default=1);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False)
    config=PaperGAConfig(quick_steps=1,generations=1,generation_steps=1,fine_steps=1,
        ga_candidates=1,regions=1,fine_regions=1,quench_fmax=.01,quench_steps=400,
        proposal_max_batches=1,proposal_max_cut_attempts=100,proposal_max_pair_attempts=100,
        partition_max_draws=100,projection_tolerance=1e-5,energy_window=100.,proposal_type=0,
        proposal_max_insertion_attempts=100,cycles=args.cycles)
    ssw=SSWConfig(width=.1,rotation_bias=100.,max_gaussians=3,temperature_K=150.,
        fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,
        rotation_solver='dimer',cluster_frame='direction_only',direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    dump(out/'plan.json',dict(cases=['Cu13','Al13'],seeds=[11,29],config=config,ssw=ssw,
        per_arm_search_cap=8000,fresh_reserve=10,per_arm_seconds=90,
        scope='Real TYPE0 GA lifecycle, not GA vs SSW efficiency or physical cluster global minimum proof',
        identity='ASE geometry.distance(permute=True)/sqrt(N)<=.1 A. Approximate inertia alignment/greedy assignment; symmetry degeneracy limitations retained. Not exact permutation minimization.',
        parameters='Numerical lifecycle budget only; 3 Gaussians/walk and one GA generation. No tuning from outcomes. NNA cutoff and collision .7*nearest-neighbor length are explicit diagnostic choices, not universal GA defaults.'))
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'));shutil.copy2(__file__,out/'runner.py')
    dump(out/'source-manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')})
    summaries=[]
    for element in ['Cu','Al']:
      initial=[Icosahedron(element,2),Octahedron(element,3,cutoff=1)];z=int(initial[0].numbers[0]);nearest={'Cu':3.61,'Al':4.05}[element]/np.sqrt(2);bonds={(z,z):nearest}
      references=[cluster_descriptor(a.numbers,a.positions,bonds,1.2) for a in initial]
      expanded=initial[0].copy();expanded.positions*=1.1
      references.append(cluster_descriptor(expanded.numbers,expanded.positions,bonds,1.2))
      for seed in [11,29]:
        d=out/f'{element}13-{seed}';d.mkdir();start=time.monotonic();row=dict(case=element+'13',seed=seed)
        class Bounded(ASESurface):
            denied=0
            def evaluate(self,a):
                if self.requests>=8000 or time.monotonic()-start>90:
                    self.denied+=1;raise RuntimeError('GA audit total request/wall cap')
                e,f=super().evaluate(a)
                with (d/'evaluations.jsonl').open('a') as s:s.write(json.dumps(serial(dict(request=self.requests,energy=e,forces=f,atoms=a)))+'\n')
                return e,f
        surface=Bounded(EMT());matchcalls=[0]
        def matcher(a,b):
            matchcalls[0]+=1
            return bool(distance(a,b,permute=True)/np.sqrt(len(a))<=.1)
        try:
            r=run_ga_ssw(initial,surface,groups=None,references=references,descriptor_bonds=bonds,
                descriptor_weights=(1.,)*6,neighbor_range=1.2,proposal_bond_limits={(z,z):.7*nearest},
                config=config,ssw_config=ssw,rng=np.random.default_rng(seed),structure_matcher=matcher,max_evaluations=8000)
            dump(d/'result.json',r)
            row.update(status=r.status,identity_mode=r.identity_mode,archive=len(r.archive),
                observations=len(r.observations),failures=[x.reason for x in r.failures],
                stages=[dict(phase=x.phase,cycle=x.cycle,status=x.status,requests=x.evaluation_requests) for x in r.stages],
                accounted=r.evaluation_requests==surface.requests,
                best_energy=None if not r.archive else min(x['energy'] for x in r.archive))
            fresh=ASESurface(EMT());checks=[]
            for item in sorted(r.archive,key=lambda x:x['energy'])[:10]:
                e,f=fresh.evaluate(item['atoms']);checks.append(dict(id=item['id'],energy_error=e-item['energy'],fmax=float(np.linalg.norm(f,axis=1).max()),certified=bool(np.linalg.norm(f,axis=1).max()<=.01)))
            row.update(fresh_requests=fresh.requests,checks=checks,unverified_archive=max(0,len(r.archive)-len(checks)))
        except Exception as exc:row.update(status='exception',error=repr(exc))
        row.update(search_requests=surface.requests,denied=surface.denied,matcher_calls=matchcalls[0],wall_seconds=time.monotonic()-start)
        dump(d/'summary.json',row);summaries.append(row);dump(out/'summary.json',summaries);print(element,seed,row,flush=True)
if __name__=='__main__':main()
