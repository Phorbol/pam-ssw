"""Bounded cross-system lifecycle audit; not a tuned performance benchmark."""
from pathlib import Path
import argparse, json, time, hashlib, shutil, subprocess
from dataclasses import asdict
import numpy as np
from ase.cluster import Icosahedron
from ase.build import bulk
from ase.collections import g2
from ase.io import write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, LSSettings, run_ssw, run_ls_ssw
from pamssw.standalone import NativeLSSettings, run_native_ls_ssw
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
from research.ga_ssw.compare_vc_arms import serial


def dump(path, data):
    path.write_text(json.dumps(serial(data), indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    cases={'cu13':Icosahedron('Cu',2),'al13':Icosahedron('Al',2),
           'butadiene':g2['butadiene'].copy(),'cyclobutene':g2['cyclobutene'].copy()}
    periodic=bulk('Cu','fcc',a=3.6,cubic=True).repeat((2,2,2));del periodic[0];cases['cu31_fixed']=periodic
    plan=dict(purpose='Complete public fixed-cell lifecycle across physical domains; development audit, not universal effectiveness',
              seeds=[11,29],steps=2,per_arm_total_EF=3000,fresh_reserve=3,per_arm_seconds=90,
              variants={'cu13':['ssw'],'al13':['ssw'],'cu31_fixed':['ssw'],
                        'butadiene':['ssw','ls_paper','ls_native'],'cyclobutene':['ssw','ls_paper','ls_native']},
              parameter_source='C4H6 existing compare_c4h6_native_paper.py width .1 rotation100 NG25 relax400; bias_fmax .1 within user range, outer .01. Same values used across these cases, no tuning.',
              limits='Molecular GFN2 and metal EMT are model evidence. Initial stationarity is not Hessian stability. Failed cases retained; no retries or cap extensions.',
              backends='ASE EMT for metals; isolated tblite0.7 GFN2-xTB accuracy .001 for C4H6, CPU single thread',
              sources='ASE Icosahedron second shell (13 atoms); ASE G2 molecules; ASE Cu FCC 2x2x2 conventional cells minus first site, fixed cell')
    dump(out/'plan.json',plan)
    for name,a in cases.items():write(out/f'{name}.extxyz',a)
    if not args.execute:return
    from tblite.ase import TBLite
    config=SSWConfig(width=.1,rotation_bias=100.,max_gaussians=25,temperature_K=150.,
        fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,
        rotation_tol=.02,rotation_solver='dimer',cluster_frame='direction_only',
        direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    paper_ls=LSSettings(HC_BOND_ENERGIES,{k:v+.1 for k,v in HC_BOND_LENGTHS.items()},target_per_atom=.7)
    native_ls=NativeLSSettings(HC_BOND_ENERGIES,HC_BOND_LENGTHS,target_mev_per_atom=700.)
    dump(out/'parameters.json',dict(config=asdict(config),paper_ls=asdict(paper_ls),native_ls=asdict(native_ls)))
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'runner.py')
    dump(out/'source-manifest.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_snapshot=True,
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')}))
    results=[]
    for name,a in cases.items():
      molecular=name in ('butadiene','cyclobutene')
      def calculator():return TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0) if molecular else EMT()
      for variant in plan['variants'][name]:
       for seed in plan['seeds']:
        from dataclasses import replace
        cfg=replace(config,cluster_frame='translation_only') if a.pbc.all() else config
        directory=out/f'{name}-{variant}-{seed}';directory.mkdir();start=time.monotonic()
        row=dict(case=name,variant=variant,seed=seed,checks=[],status='running')
        class Counted(ASESurface):
            limit=2997;phase='search';denied=0
            def evaluate(self,atoms):
                if self.requests>=self.limit or time.monotonic()-start>90:
                    self.denied+=1;raise RuntimeError('bounded lifecycle audit request/wall limit')
                try:
                    e,f=super().evaluate(atoms)
                except Exception as exc:
                    with (directory/'evaluations.jsonl').open('a') as stream:stream.write(json.dumps(dict(request=self.requests,phase=self.phase,error=str(exc)))+'\n')
                    raise
                with (directory/'evaluations.jsonl').open('a') as stream:
                    stream.write(json.dumps(serial(dict(request=self.requests,phase=self.phase,energy=e,forces=f,atoms=atoms)))+'\n')
                return e,f
        surface=Counted(calculator());result=None
        try:
            kw=dict(steps=2,config=cfg,rng=np.random.default_rng(seed))
            if variant=='ssw':result=run_ssw(a,surface,**kw)
            elif variant=='ls_paper':result=run_ls_ssw(a,surface,ls=paper_ls,**kw)
            else:result=run_native_ls_ssw(a,surface,ls=native_ls,**kw)
            dump(directory/'result.json',result)
            row.update(status=result.status,step_statuses=[r.status for r in result.records],
                       accepted=[r.accepted for r in result.records],minima=len(result.minima),
                       search_requests=surface.requests,best_delta=min(m.energy for m in result.minima)-result.initial.energy,
                       ledger_consistent=result.evaluation_requests==surface.requests==result.initial.evaluation_requests+sum(r.evaluation_requests for r in result.records))
        except Exception as exc:row.update(status='exception',error=repr(exc),search_requests=surface.requests)
        surface.limit=3000;surface.phase='fresh';surface.calculator=calculator()
        if result is not None:
            for i,m in enumerate(result.minima):
                try:
                    e,f=surface.evaluate(m.atoms)
                    row['checks'].append(dict(index=i,energy=e,fmax=float(np.linalg.norm(f,axis=1).max()),
                        certified=bool(np.linalg.norm(f,axis=1).max()<=cfg.fmax),
                        energy_error=e-m.energy,cell_unchanged=bool(np.array_equal(a.cell.array,m.atoms.cell.array))))
                except Exception as exc:row['checks'].append(dict(index=i,error=repr(exc)))
        row.update(total_requests=surface.requests,denied=surface.denied,wall_seconds=time.monotonic()-start)
        dump(directory/'summary.json',row);results.append(row);dump(out/'summary.json',results)
        print(name,variant,seed,row['status'],row['total_requests'],flush=True)

if __name__=='__main__':main()
