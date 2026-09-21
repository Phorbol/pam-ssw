"""Post-hoc GFN2 curvature checks for every observed connectivity class."""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
from ase import Atoms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = Path('research/ga_ssw/evidence/c4h6-ls-reaction-coverage-20260912').resolve()
    out = args.output.resolve(); out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/'runner.py')
    shutil.copytree(campaign/'source', out/'source', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(out/'source'))
    from pamssw.standalone.surface import ASESurface
    from pamssw.standalone.cluster_frame import ClusterFrame
    from tblite.ase import TBLite
    report=json.loads((campaign/'root-coverage-audit.json').read_text())
    assert report['complete']
    selected={}
    for arm in report['arms']:
        for item in arm['graph_categories']:
            if item['fresh_force_qualified'] is not True: continue
            key=item['class_id']; candidate=(item['energy'],arm['arm'],item['index'])
            if key not in selected or candidate < selected[key]: selected[key]=candidate
    manifest=dict(status='prepared',selection='lowest stored energy among fresh-qualified landings of each global graph class, all six arms, deterministic tie break arm/index',
        cases=selected,finite_difference_A=.001,planned_requests=len(selected)*61,
        backend='GFN2-xTB accuracy .001; fresh cold calculator for every E/F request',
        scope='post-hoc local-curvature diagnosis; no relaxation or search tuning; finite gradient and one displacement size limit stationary-point certification; only six global rigid modes removed even for fragments',
        parent=str(campaign),parent_audit_sha256=hashlib.sha256((campaign/'root-coverage-audit.json').read_bytes()).hexdigest())
    def dump(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    dump(out/'manifest.json',manifest)
    if not args.execute: return
    results=[];start=time.monotonic()
    for class_id, (_,arm,index) in sorted(selected.items()):
        source=campaign/arm/'result.json';minimum=json.loads(source.read_text())['minima'][index]
        atoms=Atoms(**minimum['atoms']);assert len(atoms)==10 and not atoms.pbc.any()
        folder=out/f'class-{class_id}';folder.mkdir()
        row=dict(class_id=class_id,arm=arm,index=index,input=minimum,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        surface=ASESurface(TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0))
        ledger=[]
        def evaluate(a,coordinate=None,sign=0):
            surface.calculator=TBLite(method='GFN2-xTB',accuracy=.001,verbosity=0)
            try:
                e,f=surface.evaluate(a)
                ledger.append(dict(request=surface.requests,coordinate=coordinate,sign=sign,energy=e,forces=f.tolist()))
                return e,f
            except Exception as error:
                ledger.append(dict(request=surface.requests,coordinate=coordinate,sign=sign,error=repr(error)))
                raise
        try:
            energy,forces=evaluate(atoms);h=.001;H=np.empty((30,30))
            for k in range(30):
                plus=atoms.copy();minus=atoms.copy();plus.positions.flat[k]+=h;minus.positions.flat[k]-=h
                ep,fp=evaluate(plus,k,1);em,fm=evaluate(minus,k,-1)
                H[:,k]=-(fp-fm).ravel()/(2*h)
            rigid=ClusterFrame(atoms).basis;assert rigid.shape==(30,6)
            internal=np.linalg.qr(rigid,mode='complete')[0][:,6:]
            sym=(H+H.T)/2;eigen=np.linalg.eigvalsh(internal.T@sym@internal)
            row.update(status='completed',energy=energy,energy_error=energy-minimum['energy'],
                fmax=float(np.linalg.norm(forces,axis=1).max()),eigenvalues_eV_A2=eigen.tolist(),
                negative_eigenvalues=int(np.sum(eigen<0)),hessian_skew_spectral=float(np.linalg.norm((H-H.T)/2,ord=2)),
                rigid_residual_spectral=float(np.linalg.norm(H@rigid,ord=2)))
            np.savez(folder/'hessian.npz',hessian=H,rigid_basis=rigid,internal_basis=internal)
        except Exception as error: row.update(status='failed',error=repr(error))
        row['requests']=surface.requests;dump(folder/'evaluations.json',ledger);dump(folder/'result.json',row)
        results.append(row);dump(out/'results.json',results)
    manifest.update(status='completed' if all(r['status']=='completed' for r in results) else 'completed_with_failures',requests=sum(r['requests'] for r in results),seconds=time.monotonic()-start)
    dump(out/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],requests=manifest['requests'],classes=len(results))))

if __name__=='__main__':main()
