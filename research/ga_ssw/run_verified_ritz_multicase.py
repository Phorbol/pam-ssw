"""Frozen paired end-to-end test of direct-residual-certified Ritz stopping."""
import argparse,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
from dataclasses import asdict,replace
from collections import Counter
import numpy as np
from ase.cluster import Icosahedron
from ase.build import bulk
from ase.collections import g2
from ase.calculators.emt import EMT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(exist_ok=False)
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'runner.py');shutil.copy2('research/ga_ssw/verified_ritz_research.py',out/'verified_ritz_research.py')
    sys.path.insert(0,str(out/'source'))
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out/'source')
    from pamssw.standalone import ASESurface,SSWConfig,run_ssw
    import pamssw.standalone.direction as direction
    from research.ga_ssw.compare_vc_arms import serial
    spec=importlib.util.spec_from_file_location('paired_verified_ritz',out/'verified_ritz_research.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    original=direction.reference_soft_mode
    cu31=bulk('Cu','fcc',a=3.6,cubic=True).repeat((2,2,2));del cu31[0]
    cases=dict(cu13=Icosahedron('Cu',2),cu31_fixed=cu31,bicyclobutane=g2['bicyclobutane'].copy())
    base=SSWConfig(width=.1,rotation_bias=None,pre_rotation_hvp=5,max_gaussians=25,
        temperature_K=150.,fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,
        rotation_hvp=100,rotation_tol=.02,direction_sampling='global',rotation_solver='ritz',
        cluster_frame='direction_only',quench_optimizer='safe-lbfgs-total')
    def dump(path,value):path.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')
    plan=dict(cases=list(cases),seeds=[11,29],arms=['original','verified'],config=asdict(base),
        steps=100,search_cap=6000,wall_seconds=120,backend='EMT metals; GFN2-xTB accuracy .001 molecule; single CPU thread',
        hypothesis='Only replace surrogate-stop followed by final rejection with direct certification and continuation of the same Krylov basis inside unchanged HVP budget; no changed tolerance, finite separation, Gaussian law or optimizer',
        controls='same source, input, seed, calculator precision and cost bounds within each pair; original arm rerun, no reuse of rounded extxyz coordinates',
        scientific_scope='developmental end-to-end robustness and search comparison; no general efficiency/default promotion without result review')
    dump(out/'plan.json',plan);dump(out/'inputs.json',cases)
    dump(out/'source-manifest.json',dict(sha256={str(p.relative_to(out/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')},research_solver_sha256=hashlib.sha256((out/'verified_ritz_research.py').read_bytes()).hexdigest()))
    if not args.execute:return
    from tblite.ase import TBLite
    rows=[]
    for name,initial in cases.items():
      cfg=replace(base,cluster_frame='translation_only') if initial.pbc.all() else base
      def calculator():return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0) if name=='bicyclobutane' else EMT()
      for seed in (11,29):
       for arm in ('original','verified'):
        folder=out/f'{name}-{arm}-seed{seed}';folder.mkdir();ledger=folder/'evaluations.jsonl';start=time.monotonic()
        direction.reference_soft_mode=original if arm=='original' else module.reference_soft_mode
        class Counted(ASESurface):
            boundary=None;denied=0
            def evaluate(self,atoms):
                if self.requests>=6000 or time.monotonic()-start>=120:
                    self.boundary='request_cap' if self.requests>=6000 else 'wall_cap';self.denied+=1
                    with ledger.open('a') as f:f.write(json.dumps(serial(dict(kind='search_denial',request=self.requests,error=self.boundary,atoms=atoms)))+'\n')
                    raise RuntimeError(self.boundary)
                try:
                    e,forces=super().evaluate(atoms);item=dict(kind='search',request=self.requests,energy=e,forces=forces,atoms=atoms)
                except Exception as error:
                    with ledger.open('a') as f:f.write(json.dumps(serial(dict(kind='search_failure',request=self.requests,error=repr(error),atoms=atoms)))+'\n')
                    raise
                with ledger.open('a') as f:f.write(json.dumps(serial(item))+'\n')
                return e,forces
        surface=Counted(calculator());fresh=None;checks=[];row=dict(case=name,seed=seed,arm=arm)
        try:
            result=run_ssw(initial.copy(),surface,steps=100,config=cfg,rng=np.random.default_rng(seed))
            dump(folder/'result.json',result)
            assert result.evaluation_requests==surface.requests==result.initial.evaluation_requests+sum(r.evaluation_requests for r in result.records)
            assert all(e.get('force_requests',0)<=101 for r in result.records for e in r.climb)
            row.update(status=result.status,search_requests=surface.requests,minima=len(result.minima),record_statuses=dict(Counter(r.status for r in result.records)),best_delta=min(q.energy for q in result.minima)-result.initial.energy)
            fresh=ASESurface(calculator())
            for i,q in enumerate(result.minima):
                try:
                    fresh.calculator=calculator();e,f=fresh.evaluate(q.atoms)
                    checks.append(dict(index=i,energy=e,energy_error=e-q.energy,fmax=float(np.linalg.norm(f,axis=1).max()),force_qualified=bool(np.linalg.norm(f,axis=1).max()<=.01),cell_unchanged=bool(np.array_equal(q.atoms.cell.array,initial.cell.array))))
                except Exception as error:checks.append(dict(index=i,error=repr(error)))
                dump(folder/'fresh-checks.json',checks)
        except Exception as error:row.update(status='exception',error=repr(error),search_requests=surface.requests)
        finally:direction.reference_soft_mode=original
        row.update(fresh_requests=0 if fresh is None else fresh.requests,fresh_checks=checks,boundary=surface.boundary,denied=surface.denied,wall_seconds=time.monotonic()-start)
        dump(folder/'summary.json',row);rows.append(row);dump(out/'summary.json',rows)
        print(name,seed,arm,row['status'],row['search_requests'],flush=True)

if __name__=='__main__':main()
