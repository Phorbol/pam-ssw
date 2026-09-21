"""Public GA integration after verified-Ritz core correction; bounded, no retuning."""
import argparse, hashlib, json, shutil, subprocess, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ase.collections import g2
from ase.io import write
from ase.geometry import distance

def dump(path, value, serial):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--execute',action='store_true'); args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc')); shutil.copy2(__file__,out/'runner.py')
    sys.path.insert(0,str(out/'source'))
    from pamssw.standalone import ASESurface,PaperGAConfig,SSWConfig,run_ga_ssw
    import pamssw
    assert Path(pamssw.__file__).resolve().is_relative_to(out/'source')
    from pamssw.standalone.legacy_descriptor import cluster_descriptor
    from pamssw.standalone.native_ls import HC_BOND_LENGTHS
    from research.ga_ssw.compare_vc_arms import serial
    initial=[g2['bicyclobutane'].copy()]
    references=[g2[n].copy() for n in ('butadiene','cyclobutene','2-butyne')]
    bonds=dict(HC_BOND_LENGTHS); limits={k:.7*v for k,v in bonds.items()}
    cfg=PaperGAConfig(quick_steps=10,generations=1,generation_steps=1,fine_steps=1,offspring_steps=1,ga_candidates=4,regions=1,fine_regions=1,quench_fmax=.01,quench_steps=400,proposal_max_batches=4,proposal_max_cut_attempts=100,proposal_max_pair_attempts=100,partition_max_draws=100,projection_tolerance=1e-5,energy_window=100.,proposal_type=0,proposal_max_insertion_attempts=100)
    ssw=SSWConfig(width=.1,rotation_bias=None,pre_rotation_hvp=5,max_gaussians=25,temperature_K=150.,fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,rotation_tol=.02,rotation_solver='ritz',cluster_frame='direction_only',direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    dump(out/'plan.json',dict(system='bicyclobutane',initial=['bicyclobutane'],seeds=[11,29],quick_steps=10,generations=1,generation_steps=1,fine_steps=1,offspring_steps=1,config=asdict(cfg),ssw_config=asdict(ssw),max_search_evaluations=12000,per_arm_wall_seconds=120,fresh_maxima=10,descriptor='HC_BOND_LENGTHS',references='Three actual G2 descriptors: butadiene, cyclobutene, 2-butyne; references are not archive seeds',population_question='Can quick SSW from one bicyclobutane basin establish enough qualified archive parents for TYPE0 GA?',quota_source='TYPE0 mixed-element G//4 crossover: G=4 is first nonzero quota; quick archive must supply the parent pool.',collision='proposal cutoff fixed at 0.7*bond length; inherited unchanged',backend='GFN2-xTB accuracy .001; CPU single thread',scope='Single-basin parent-establishment lifecycle only; no SSW-vs-GA efficiency or superiority claim'),serial)
    write(out/'bicyclobutane.extxyz',initial[0])
    if not args.execute: return
    dump(out/'source-manifest.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sha256={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')}),serial)
    refs=[cluster_descriptor(a.numbers,a.positions,bonds,1.2) for a in references]; results=[]
    dump(out/'references.json',dict(names=['butadiene','cyclobutene','2-butyne'],atoms=references,descriptors=refs,role='fixed projection references only, never supplied as initial/parents'),serial)
    from tblite.ase import TBLite
    from research.ga_ssw.run_fixed_ga_staged_integration import molecular_components
    for seed in (11,29):
        runout=out/f'bicyclobutane-seed{seed}-plain'; runout.mkdir(); started=time.monotonic(); ledger=runout/'evaluations.jsonl'
        class Bounded(ASESurface):
            denied=0; boundary=None
            def evaluate(self,a):
                if self.requests>=12000 or time.monotonic()-started>=120:
                    self.denied+=1;self.boundary='request_cap' if self.requests>=12000 else 'wall_cap'
                    with ledger.open('a') as h:h.write(json.dumps(serial(dict(kind='search_denial',request=self.requests,error=self.boundary,atoms=a)))+'\n')
                    raise RuntimeError(self.boundary)
                try: e,f=super().evaluate(a)
                except Exception as error:
                    with ledger.open('a') as h:h.write(json.dumps(serial(dict(kind='search_failure',request=self.requests,error=repr(error),atoms=a)))+'\n')
                    raise
                with ledger.open('a') as h:h.write(json.dumps(serial(dict(kind='search',request=self.requests,energy=e,forces=f,atoms=a)))+'\n')
                return e,f
        surface=Bounded(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)); row=dict(system='bicyclobutane',seed=seed,arm='plain_single_basin')
        fresh=None;checks=[]
        try:
            result=run_ga_ssw([initial[0].copy()],surface,groups=None,references=refs,descriptor_bonds=bonds,descriptor_weights=(1.,)*6,neighbor_range=1.2,proposal_bond_limits=limits,config=cfg,ssw_config=ssw,rng=np.random.default_rng(seed),ls=None,max_evaluations=12000,structure_matcher=lambda a,b:bool(distance(a,b,permute=True)/np.sqrt(len(a))<=.1))
            dump(runout/'result.json',result,serial); stages=[dict(phase=x.phase,generation=x.generation,cycle=x.cycle,status=x.status,requests=x.evaluation_requests,observations=x.observations,details=x.details) for x in result.stages]
            row.update(status=result.status,search_requests=surface.requests,archive=len(result.archive),observations=len(result.observations),accounted=result.evaluation_requests==surface.requests,failures=[x.reason for x in result.failures],stages=stages,parent_ids=[list(x.parent_ids) for x in result.observations],phase_counts={p:sum(x.phase==p for x in result.stages) for p in sorted({x.phase for x in result.stages})})
            fresh=ASESurface(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)); checks=[]
            for item in sorted(result.archive,key=lambda x:x['energy'])[:10]:
                try:
                    fresh.calculator=TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0); e,f=fresh.evaluate(item['atoms']); check=dict(id=item['id'],energy=e,energy_error=e-item['energy'],fmax=float(np.linalg.norm(f,axis=1).max()),force_qualified=bool(np.linalg.norm(f,axis=1).max()<=.01),molecular_components=molecular_components(item['atoms'],bonds))
                except Exception as exc: check=dict(id=item['id'],error=repr(exc))
                checks.append(check)
                row.update(fresh_requests=fresh.requests,fresh_checks=checks,fresh_remaining=max(0,len(result.archive)-len(checks)))
                dump(runout/'fresh-checks.json',checks,serial)
            row.update(fresh_requests=fresh.requests,fresh_checks=checks,fresh_remaining=max(0,len(result.archive)-len(checks)))
        except Exception as exc: row.update(status='exception',error=repr(exc),search_requests=surface.requests,fresh_requests=0 if fresh is None else fresh.requests,fresh_checks=checks)
        ledger_rows=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
        paid=[x for x in ledger_rows if x['kind'] in ('search','search_failure')]
        assert [x['request'] for x in paid]==list(range(1,surface.requests+1))
        row.update(boundary=surface.boundary,ledger_paid=len(paid),wall_seconds=time.monotonic()-started,denied=surface.denied,ledger_count=sum(1 for _ in ledger.open()) if ledger.exists() else 0,accounted=row.get('accounted',False)); row['accounted']=row.get('accounted',False) and row['ledger_paid']==surface.requests; dump(runout/'summary.json',row,serial); results.append(row); dump(out/'summary.json',results,serial)
if __name__=='__main__': main()
