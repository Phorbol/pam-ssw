"""Independent strict quench and two-step constrained Hessians for TYPE4.

Three frozen endpoints; no continuation of the search and no parameter tuning.
"""
import argparse
import gc
import hashlib
import importlib.metadata
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);plan=json.loads((root/'plan.json').read_text());total=0;start=time.monotonic()
    model=Path(plan['model']);assert hashlib.sha256(model.read_bytes()).hexdigest()==plan['model_sha256']
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, no fallback')
    torch.set_num_threads(1)
    (root/'runtime.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),device=torch.cuda.get_device_name(0),packages={k:importlib.metadata.version(k) for k in ('numpy','ase','torch','mace-torch')},pythonpath=os.environ.get('PYTHONPATH')),indent=2)+'\n')
    summary={}
    for name in plan['endpoints']:
        directory=root/name;directory.mkdir(exist_ok=False)
        raw=json.loads((root/(name+'.json')).read_text());atoms=Atoms(**raw)
        fixed=np.arange(297);active=np.arange(297,len(atoms));phase='quench';case_start=time.monotonic()
        record={};surface=None
        def calc():return MACECalculator(model_paths=str(model),device='cuda',default_dtype='float64',enable_cueq=False)
        class Counted(ASESurface):
            def evaluate(self,a):
                nonlocal total
                if total>=plan['total_cap'] or time.monotonic()-start>plan['seconds']:
                    raise RuntimeError('qualification total budget exhausted')
                if phase=='quench' and self.requests>=plan['quench_cap']:
                    raise RuntimeError('endpoint quench budget exhausted')
                before=self.requests
                try:return super().evaluate(a)
                finally:total+=self.requests-before
        try:
            surface=Counted(calc())
            minimum=constrained_quench(atoms,surface,fixed_indices=fixed,fmax=plan['fmax'],max_step=.2,maxiter=500,lbfgs_memory=500)
            record.update(quench_converged=minimum.converged,quench_requests=surface.requests,energy=minimum.energy,active_fmax=minimum.active_fmax,full_raw_fmax=minimum.full_raw_fmax,certificate=minimum.certificate)
            a=minimum.atoms.copy();a.set_constraint()
            endpoint=dict(numbers=a.numbers.tolist(),positions=a.positions.tolist(),cell=a.cell.array.tolist(),pbc=a.pbc.tolist())
            (directory/'strict-atoms.json').write_text(json.dumps(endpoint,indent=2)+'\n')
            if not minimum.converged:raise RuntimeError('strict quench not converged; Hessian not attempted')
            phase='fresh';surface.calculator=None;gc.collect();torch.cuda.empty_cache();surface.calculator=calc()
            e,f=surface.evaluate(a)
            record['fresh']=dict(energy=e,energy_error=e-minimum.energy,active_fmax=float(np.linalg.norm(f[active],axis=1).max()),fixed_exact=bool(np.array_equal(a.positions[fixed],atoms.positions[fixed])),cell_exact=bool(np.array_equal(a.cell.array,atoms.cell.array)))
            assert record['fresh']['active_fmax']<=plan['fmax'] and record['fresh']['fixed_exact'] and record['fresh']['cell_exact']
            phase='hessian';dimension=3*len(active);matrices=[];record['hessians']=[]
            for h in plan['hessian_steps']:
                matrix=np.lib.format.open_memmap(directory/f'hessian-{h}.npy',mode='w+',dtype=float,shape=(dimension,dimension));matrix[:]=np.nan
                for column in range(dimension):
                    atom=active[column//3];axis=column%3
                    plus=a.copy();minus=a.copy();plus.positions[atom,axis]+=h;minus.positions[atom,axis]-=h
                    _,fp=surface.evaluate(plus);_,fm=surface.evaluate(minus)
                    matrix[:,column]=-(fp[active]-fm[active]).ravel()/(2*h)
                    if column%25==0:
                        matrix.flush();(directory/'progress.json').write_text(json.dumps(dict(phase=phase,step=h,completed_columns=column+1,requests=surface.requests,total_requests=total,seconds=time.monotonic()-start))+'\n')
                matrix.flush();symmetric=(matrix+matrix.T)/2;eig=np.linalg.eigvalsh(symmetric)
                np.save(directory/f'eigenvalues-{h}.npy',eig)
                record['hessians'].append(dict(step=h,dimension=dimension,min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1]),negative_count=int((eig<0).sum()),skew_spectral_norm=float(np.linalg.norm(matrix-matrix.T,2)),all_finite=bool(np.isfinite(matrix).all())))
                matrices.append(matrix)
            record['hessian_step_difference_spectral_norm']=float(np.linalg.norm(matrices[0]-matrices[1],2))
        except Exception as exc:record['error']=repr(exc)
        finally:
            record.update(requests=0 if surface is None else surface.requests,seconds=time.monotonic()-case_start)
            (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n')
            summary[name]=record
            summary['_total']=dict(requests=total,seconds=time.monotonic()-start)
            (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(name,record,flush=True)
            if surface is not None:surface.calculator=None;del surface
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
