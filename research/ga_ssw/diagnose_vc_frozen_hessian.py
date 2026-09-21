"""Full joint curvature at four archived Safe-total biased-quench failures."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from research.ga_ssw.compare_fe7c3_ls_frozen_quenches import (
    CountedSurface,_atoms,_softening,_combined,_frozen_bias)
from research.ga_ssw.compare_vc_arms import serial


def save(path,value):path.write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')

def run(path,out,factory,deadline):
    d=json.loads(path.read_text());q=np.asarray(d['safe_lbfgs']['q'])
    chart=SymmetricLogStrainChart(_atoms(d['chart_reference']),strain_length=d['strain_length'])
    soft=_softening(d['frozen_softening']);terms=d['frozen_gaussians']
    surface=CountedSurface(factory(),cap=973,deadline=deadline,path=out/'evaluations.jsonl')
    def evaluate(x):
        ev=chart.evaluate(x,lambda a:_combined(surface,soft,a),pressure=d['pressure'])
        return _frozen_bias(ev.objective,chart.project(ev.gradient),x,terms)
    result=dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status='started',requests=0,steps={})
    try:
        surface.stage='reconstruction'
        energy,g=evaluate(q)
        old=d['final_fresh']
        error=max(abs(energy-old['objective']),float(np.max(np.abs(g-old['projected_gradient']))))
        result['reconstruction_error']=error
        if error>1e-8:raise ValueError('archived complete objective mismatch')
        translations=np.zeros((q.size,3))
        for i in range(3):translations[i:-6:3,i]=1/np.sqrt(chart.natoms)
        basis=null_space(translations.T)
        assert basis.shape==(246,243)
        save(out/'basis.json',basis)
        matrices=[]
        for h in (1e-4,5e-5):
            key=f'{h:.0e}';surface.stage=f'hessian-{key}'
            raw=np.lib.format.open_memmap(out/f'hessian-{key}-raw.npy',mode='w+',dtype=float,shape=(243,243))
            raw[:]=np.nan
            for i in range(243):
                gp=evaluate(q+h*basis[:,i])[1];gm=evaluate(q-h*basis[:,i])[1]
                raw[:,i]=basis.T@(gp-gm)/(2*h)
                if (i+1)%20==0:
                    raw.flush();save(out/'progress.json',dict(step=key,columns=i+1,requests=surface.requests))
            raw.flush()
            symmetric=(np.asarray(raw)+np.asarray(raw).T)/2
            w,u=np.linalg.eigh(symmetric)
            np.save(out/f'hessian-{key}-sym.npy',symmetric)
            np.save(out/f'eigenvectors-{key}.npy',u)
            np.save(out/f'eigenvalues-{key}.npy',w)
            result['steps'][key]=dict(min=float(w[0]),max=float(w[-1]),
                skew_norm=float(np.linalg.norm(np.asarray(raw)-np.asarray(raw).T,2)),
                eigenvalues=w,cell_participation=np.sum((basis[-6:,:]@u)**2,axis=0))
            matrices.append(symmetric)
        result['step_difference_spectral']=float(np.linalg.norm(matrices[0]-matrices[1],2))
        for step in result['steps'].values():
            step['negative_beyond_step_difference']=int(np.sum(step['eigenvalues'] < -result['step_difference_spectral']))
        result['status']='completed'
    except Exception as error:
        result.update(status='failed_or_incomplete',error=f'{type(error).__name__}: {error}')
    result['requests']=surface.requests
    save(out/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('source','status','requests','error','step_difference_spectral')}),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False,parents=True)
    import torch
    from mace.calculators import MACECalculator
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    model_hash=hashlib.sha256(a.model.read_bytes()).hexdigest()
    assert model_hash=='0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5'
    def factory():return MACECalculator(model_paths=str(a.model),device='cuda',default_dtype='float64',enable_cueq=False)
    deadline=time.monotonic()+900.;results=[]
    for arm in ('ls_all','ls_filter'):
        for seed in (7,101):
            out=a.output/f'{arm}-seed{seed}';out.mkdir()
            path=a.inputs/f'{arm}-seed{seed}.json'
            if time.monotonic()>=deadline:
                r=dict(source=str(path),status='not_started_deadline',requests=0);save(out/'result.json',r)
            else:r=run(path,out,factory,deadline)
            results.append(r)
    save(a.output/'summary.json',dict(model_sha256=model_hash,requested_cases=4,total_requests=sum(r['requests'] for r in results),results=results))
