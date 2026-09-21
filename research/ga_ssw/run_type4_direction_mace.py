"""Plan-bounded two-mask TYPE4 comparison on one V100."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
from dataclasses import is_dataclass
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms
from mace.calculators import MACECalculator
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig,run_constrained_ssw


def serial(x):
    if isinstance(x,Atoms):return dict(numbers=x.numbers.tolist(),positions=x.positions.tolist(),cell=x.cell.array.tolist(),pbc=x.pbc.tolist())
    if is_dataclass(x):return {k:serial(v) for k,v in vars(x).items()}
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v) for v in x]
    return x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);plan=json.loads((root/'plan.json').read_text())
    model=Path(plan['model'])
    assert hashlib.sha256(model.read_bytes()).hexdigest()==plan['model_sha256']
    if not torch.cuda.is_available():raise RuntimeError('declared CUDA backend unavailable; no CPU fallback')
    torch.set_num_threads(1)
    meta=dict(job_id=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(0),
              packages={name:importlib.metadata.version(name) for name in ('numpy','ase','torch','mace-torch')},
              environment={name:os.environ.get(name) for name in ('PYTHONPATH','PYTHONNOUSERSITE','CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS')})
    (root/'runtime.json').write_text(json.dumps(meta,indent=2)+'\n')
    atoms=(Atoms(**json.loads((root/'input.json').read_text())) if (root/'input.json').exists() else read(root/'input.arc',index=0));fixed=np.arange(297);active=np.arange(297,len(atoms))
    assert len(atoms)==514
    atoms.set_constraint(FixAtoms(indices=fixed))
    config=ConstrainedSSWConfig(**plan['config'])
    steps=plan.get('steps',1)
    if isinstance(steps,bool) or not isinstance(steps,int) or steps<1:raise ValueError('positive integer steps required')
    fresh_reserve=steps+1
    if plan['arm_total_ef']<=fresh_reserve:raise ValueError('budget must exceed fresh endpoint reserve')
    def calculator():return MACECalculator(model_paths=str(model),device='cuda',default_dtype='float64',enable_cueq=False)
    summary={}
    for arm in ('physical_mask_only','source_direction_mask'):
        directory=root/arm;directory.mkdir(exist_ok=False)
        start=time.monotonic();phase='search';calls=directory/'calls.jsonl'
        class Counted(ASESurface):
            exhausted=False
            def evaluate(self,a):
                cap=plan['arm_total_ef']-fresh_reserve if phase=='search' else plan['arm_total_ef']
                if self.requests>=cap or (phase=='search' and time.monotonic()-start>plan['arm_search_seconds']):
                    self.exhausted=True;raise RuntimeError('declared arm budget exhausted')
                item=dict(request=self.requests+1,phase=phase,atoms=serial(a))
                try:
                    e,f=super().evaluate(a);item.update(energy=e,forces=f.tolist());return e,f
                except Exception as exc:item['error']=repr(exc);raise
                finally:
                    item['charged_requests']=self.requests
                    with calls.open('a') as handle:handle.write(json.dumps(item)+'\n')
                    if self.requests%50==0:
                        (directory/'progress.json').write_text(json.dumps(dict(requests=self.requests,phase=phase,seconds=time.monotonic()-start))+'\n')
        surface=Counted(calculator());report={}
        try:
            kwargs={} if arm=='physical_mask_only' else dict(direction_fixed_indices=np.arange(351))
            result=run_constrained_ssw(atoms,surface,steps=steps,config=config,rng=np.random.default_rng(plan['seed']),**kwargs)
            report.update(result=serial(result),budget_exhausted=surface.exhausted,search_requests=surface.requests,search_seconds=time.monotonic()-start)
            # Persist all search outcomes before independent qualification.
            (directory/'search-result.json').write_text(json.dumps(report,indent=2)+'\n')
            phase='fresh';report['fresh']=[]
            surface.calculator=None;gc.collect();torch.cuda.empty_cache()
            surface.calculator=calculator()
            for minimum in result.minima:
                a=minimum.atoms.copy();a.set_constraint();e,f=surface.evaluate(a)
                report['fresh'].append(dict(energy=e,energy_error=e-minimum.energy,
                    active_fmax=float(np.linalg.norm(f[active],axis=1).max()),
                    raw_fmax=float(np.linalg.norm(f,axis=1).max()),
                    fixed_exact=bool(np.array_equal(a.positions[fixed],atoms.positions[fixed])),
                    cell_exact=bool(np.array_equal(a.cell.array,atoms.cell.array))))
        except Exception as exc:report.update(error=repr(exc),traceback=traceback.format_exc())
        finally:
            report.update(total_requests=surface.requests,seconds=time.monotonic()-start)
            (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            summary[arm]=dict(total_requests=surface.requests,seconds=report['seconds'],status=report.get('result',{}).get('status'),error=report.get('error'))
            surface.calculator=None;del surface;gc.collect();torch.cuda.empty_cache()
            (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(arm,summary[arm],flush=True)


if __name__=='__main__':main()
