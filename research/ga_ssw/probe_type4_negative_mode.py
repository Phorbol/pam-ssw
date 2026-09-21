"""Unbiased quench controls at a negative-curvature TYPE4 initial endpoint."""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from mace.calculators import MACECalculator
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.constrained_reference import constrained_quench


def encode(a):return dict(numbers=a.numbers.tolist(),positions=a.positions.tolist(),cell=a.cell.array.tolist(),pbc=a.pbc.tolist())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);plan=json.loads((root/'plan.json').read_text())
    model=Path(plan['model']);assert hashlib.sha256(model.read_bytes()).hexdigest()==plan['model_sha256']
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(1);initial=Atoms(**json.loads((root/'initial.json').read_text()))
    mode=np.load(root/'mode.npy');fixed=np.arange(297);active=np.arange(297,514)
    assert mode.shape==(217,3) and np.isclose(np.linalg.norm(mode),1)
    def calc():return MACECalculator(model_paths=str(model),device='cuda',default_dtype='float64',enable_cueq=False)
    result=dict(job_id=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(0),checks=[],arms={})
    start=time.monotonic();probe=ASESurface(calc())
    for scale in (0.,-.025,.025,-.05,.05):
        a=initial.copy();a.positions[active]+=scale*mode
        e,f=probe.evaluate(a)
        result['checks'].append(dict(scale=scale,energy=e,mode_gradient=float(-np.sum(f[active]*mode))))
    result['probe_requests']=probe.requests;probe.calculator=None;del probe;gc.collect();torch.cuda.empty_cache()
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,scale in (('unperturbed',0.),('negative',-.05),('positive',.05)):
        a=initial.copy();a.positions[active]+=scale*mode
        phase='quench';arm_start=time.monotonic();row={}
        class Counted(ASESurface):
            def evaluate(self,x):
                if self.requests>=plan['arm_cap'] or time.monotonic()-arm_start>plan['arm_seconds']:
                    raise RuntimeError('declared unbiased-control budget exhausted')
                e,f=super().evaluate(x)
                with (root/(name+'-calls.jsonl')).open('a') as handle:
                    handle.write(json.dumps(dict(request=self.requests,phase=phase,atoms=encode(x),energy=e,forces=f.tolist()))+'\n')
                return e,f
        surface=Counted(calc())
        try:
            minimum=constrained_quench(a,surface,fixed_indices=fixed,fmax=plan['fmax'],max_step=.2,maxiter=800,lbfgs_memory=500)
            row.update(converged=minimum.converged,energy=minimum.energy,active_fmax=minimum.active_fmax,quench_requests=surface.requests,certificate=minimum.certificate,atoms=encode(minimum.atoms))
            surface.calculator=None;gc.collect();torch.cuda.empty_cache();surface.calculator=calc();phase='fresh'
            b=minimum.atoms.copy();b.set_constraint();e,f=surface.evaluate(b)
            row['fresh']=dict(energy=e,energy_error=e-minimum.energy if minimum.energy is not None else None,active_fmax=float(np.linalg.norm(f[active],axis=1).max()),fixed_exact=bool(np.array_equal(b.positions[fixed],initial.positions[fixed])),cell_exact=bool(np.array_equal(b.cell.array,initial.cell.array)))
        except Exception as exc:row['error']=repr(exc)
        finally:
            row.update(requests=surface.requests,seconds=time.monotonic()-arm_start)
            result['arms'][name]=row;result['total_requests']=result['probe_requests']+sum(v['requests'] for v in result['arms'].values());result['seconds']=time.monotonic()-start
            (root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(name,row.get('energy'),row.get('converged'),row['requests'],flush=True)
            surface.calculator=None;del surface;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
