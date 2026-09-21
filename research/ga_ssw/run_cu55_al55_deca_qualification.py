"""Bounded EMT qualification of Cu55/Al55 decahedral cluster motifs."""
import argparse, hashlib, json, shutil, subprocess, time, dataclasses
from pathlib import Path
import numpy as np
from ase.cluster import Decahedron
from ase.calculators.emt import EMT
from ase.io import write
from pamssw.standalone.surface import ASESurface, quench


def serial(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if dataclasses.is_dataclass(v): return serial(dataclasses.asdict(v))
    if hasattr(v, 'get_positions') and hasattr(v, 'get_atomic_numbers'):
        return {'numbers': v.get_atomic_numbers().tolist(), 'positions': v.get_positions().tolist(), 'cell': v.cell.array.tolist(), 'pbc': v.pbc.tolist()}
    if isinstance(v, dict): return {k:serial(x) for k,x in v.items()}
    if isinstance(v, (list, tuple)): return [serial(x) for x in v]
    return v

def pair_fingerprint(atoms):
    p=atoms.get_positions(); d=np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2)
    return np.sort(d[np.triu_indices(len(atoms),1)])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(parents=False,exist_ok=False)
    git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    cases={}
    for element in ('Cu','Al'):
        cases[f'{element}55_deca']=Decahedron(element,3,3,0)
    plan={'status':'bounded_motif_qualification_not_search','cases':list(cases),'atoms_per_case':55,
          'calculator':'ASE EMT', 'optimizer':'safe-lbfgs-total','fmax':.01,'steps':400,
          'search_cap':1000,'fresh_checks':1,'hessian_requests':330,'hessian_step_A':1e-3,'cell':'nonperiodic','distortion':'none',
          'constructors':{'deca':'Decahedron(symbol,3,3,0)'},
          'source_commit':git,
          'scope':'Qualify realistic larger Cu55/Al55 motif starts. A lower relaxed motif is an attained reference; no global-minimum claim. Higher motif may be used later as metastable start only if distinct after this check.'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    manifest=[]
    for p in [Path(__file__)]:
        manifest.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (out/'source-manifest.json').write_text(json.dumps({'git':git,'runner':manifest},indent=2)+'\n')
    for name,a in cases.items(): write(out/f'{name}-input.extxyz',a)
    summary=[]
    for name, atoms in cases.items():
        started=time.monotonic(); folder=out/name; folder.mkdir();
        requests=[]; cap_state=None
        class Capped(ASESurface):
            def evaluate(self,candidate):
                nonlocal cap_state
                if self.requests>=1000:
                    cap_state='request_cap'; requests.append({'request':self.requests+1,'error':'request_cap'}); raise RuntimeError('request_cap')
                try: e,f=super().evaluate(candidate)
                except Exception as exc:
                    requests.append({'request':self.requests,'error':repr(exc)}); raise
                requests.append({'request':self.requests,'energy':float(e),'fmax':float(np.linalg.norm(f,axis=1).max())}); return e,f
        surface=Capped(EMT())
        row={'case':name,'search_cap':1000}
        try:
            result=quench(atoms,surface,fmax=.01,steps=400,optimizer='safe-lbfgs-total')
            with (folder/'result.json').open('w') as fh: json.dump(serial(result),fh,indent=2,allow_nan=False)
            write(folder/'relaxed.extxyz',result.atoms)
            fresh=ASESurface(EMT()); e,f=fresh.evaluate(result.atoms)
            check={'energy':float(e),'energy_error':float(e-result.energy),'fmax':float(np.linalg.norm(f,axis=1).max()),'force_qualified':bool(np.linalg.norm(f,axis=1).max()<=.01)}
            (folder/'fresh-check.json').write_text(json.dumps(check,indent=2)+'\n')
            # Independent central-force Hessian: 2*3N requests, excluding 6 rigid modes.
            from pamssw.standalone.cluster_frame import ClusterFrame
            frame=ClusterFrame(result.atoms); qfull=np.linalg.qr(frame.basis, mode='complete')[0]; qint=qfull[:,6:]
            h=1e-3; ncoord=3*len(result.atoms); H=np.empty((ncoord,ncoord)); hs=ASESurface(EMT()); hfailed=[]
            x=result.atoms.positions.copy()
            for j in range(ncoord):
                xp=x.copy(); xm=x.copy(); xp.ravel()[j]+=h; xm.ravel()[j]-=h
                try:
                    ap=result.atoms.copy(); am=result.atoms.copy(); ap.positions=xp; am.positions=xm
                    _,fp=hs.evaluate(ap); _,fm=hs.evaluate(am); H[:,j]=-(fp-fm).ravel()/(2*h)
                except Exception as error:
                    hfailed.append({'coordinate':j,'error':repr(error)}); break
            if hfailed: raise RuntimeError(f'hessian failure: {hfailed[0]}')
            Hint=qint.T@H@qint; antisym=float(np.linalg.norm(Hint-Hint.T)); eig=np.linalg.eigvalsh((Hint+Hint.T)/2)
            hessian={'step_A':h,'requests':hs.requests,'internal_dimension':int(qint.shape[1]),'antisymmetry_frobenius':antisym,'min_internal_eigenvalue':float(eig[0]),'negative_eigenvalues':int(np.count_nonzero(eig < 0)),'eigenvalues':eig.tolist()}
            (folder/'hessian.json').write_text(json.dumps(hessian,indent=2)+'\n')
            row.update(status='completed',energy=float(result.energy),max_force=float(result.max_force),converged=bool(result.converged),search_requests=surface.requests,fresh_requests=fresh.requests,hessian_requests=hs.requests,failed_requests=[x for x in requests if 'error' in x],fresh=check,hessian=hessian)
        except Exception as exc:
            row.update(status='exception',error=repr(exc),search_requests=surface.requests,fresh_requests=0,failed_requests=[x for x in requests if 'error' in x])
        (folder/'evaluations.jsonl').write_text(''.join(json.dumps(serial(x), allow_nan=False)+'\n' for x in requests))
        row['seconds']=time.monotonic()-started; summary.append(row); (out/'summary.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False)+'\n')
    (out/'motif-comparison.json').write_text(json.dumps([],indent=2)+'\n')

if __name__=='__main__': main()
