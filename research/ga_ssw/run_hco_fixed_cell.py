"""H/C/O LS lifecycle audit using recovered raw pair tables; not a performance benchmark."""
from pathlib import Path
import argparse, json, time, hashlib, shutil, subprocess
from dataclasses import asdict
import numpy as np
from ase.cluster import Icosahedron
from ase.build import bulk
from ase.collections import g2
from ase.data.s22 import create_s22_system
from ase.io import write
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, LSSettings, run_ssw, run_ls_ssw
from pamssw.standalone import NativeLSSettings, run_native_ls_ssw
from pamssw.standalone.native_ls import HCO_BOND_ENERGIES, HCO_BOND_LENGTHS
from research.ga_ssw.compare_vc_arms import serial


def dump(path, data):
    path.write_text(json.dumps(serial(data), indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    cases={'methanol':g2['CH3OH'].copy(),
           'water_dimer':create_s22_system('Water_dimer'),
           'formic_acid_dimer':create_s22_system('Formic_acid_dimer')}
    plan=dict(purpose='Real HCO SSW/LS lifecycle after native lookup and initialization-prefix recovery; not generic effectiveness',
              seeds=[11,29],steps=2,per_arm_total_EF=3000,fresh_reserve=3,per_arm_seconds=90,
              variants={name:['ssw','ls_paper','ls_native'] for name in cases},
              parameter_source='Existing fixed-cell diagnostic width .1 rotation100 relax400; 3 Gaussians/attempt; bias_fmax .1, outer .01. Explicit recovered HCO lookup tables; paper target .02 eV/atom and native target20 meV/atom are diagnostic defaults, not GFN2-fitted parameters.',
              limits='GFN2 semiempirical PES only, no Hessian or molecular stability claim. Hydrogen-bonded dimers have two covalent components at input; component count alone cannot establish binding or dissociation. All failures retained, no retries.',
              backends='isolated tblite0.7 GFN2-xTB accuracy .001, CPU single thread',
              sources='ASE G2 methanol; ASE S22 Water_dimer and Formic_acid_dimer; raw lookup/native H2O and CH3OH initialization evidence dated20260912')
    dump(out/'plan.json',plan)
    for name,a in cases.items():write(out/f'{name}.extxyz',a)
    if not args.execute:return
    from tblite.ase import TBLite
    config=SSWConfig(width=.1,rotation_bias=100.,max_gaussians=3,temperature_K=150.,
        fmax=.01,bias_fmax=.1,relax_steps=400,fd_step=1e-4,rotation_hvp=100,
        rotation_tol=.02,rotation_solver='dimer',cluster_frame='direction_only',
        direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    paper_ls=LSSettings(HCO_BOND_ENERGIES,{k:v+.1 for k,v in HCO_BOND_LENGTHS.items()},target_per_atom=.02)
    native_ls=NativeLSSettings(HCO_BOND_ENERGIES,HCO_BOND_LENGTHS,target_mev_per_atom=20.)
    dump(out/'parameters.json',dict(config=asdict(config),paper_ls=asdict(paper_ls),native_ls=asdict(native_ls)))
    shutil.copytree('pamssw',out/'source'/'pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,out/'runner.py')
    dump(out/'source-manifest.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_snapshot=True,
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'source').rglob('*.py')}))
    results=[]
    for name,a in cases.items():
      molecular=True
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
