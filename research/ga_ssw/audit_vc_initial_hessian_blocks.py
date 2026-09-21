"""Analyze archived physical joint Hessians; no new Hessian or PES evaluations."""
import json,argparse
from pathlib import Path
import numpy as np
from scipy.linalg import null_space,block_diag


def spectrum(a):
    w=np.linalg.eigvalsh(a)
    return dict(min=float(w[0]),max=float(w[-1]),condition=float(w[-1]/w[0]))

def inverse_root(a):
    w,u=np.linalg.eigh(a)
    if w[0]<=0: raise ValueError('physical Hessian block not positive')
    return (u/np.sqrt(w))@u.T

def audit(folder):
    q=np.asarray(json.loads((folder/'hessian-basis.json').read_text()))
    n=(q.shape[0]-6)//3
    t=np.zeros((3*n,3))
    for i in range(3):t[i::3,i]=1/np.sqrt(n)
    atoms=null_space(t.T)
    transform=q.T@block_diag(atoms,np.eye(6))
    assert np.allclose(transform.T@transform,np.eye(3*n+3),atol=1e-12)
    rows=[]
    for step in ('1e-04','5e-05'):
        h=transform.T@np.load(folder/f'hessian-{step}-sym.npy')@transform
        aa=h[:-6,:-6];cc=h[-6:,-6:];ac=h[:-6,-6:]
        schur=cc-ac.T@np.linalg.solve(aa,ac)
        rows.append(dict(step=step,joint=spectrum(h),atomic=spectrum(aa),cell=spectrum(cc),
            relaxed_cell_schur=spectrum(schur),
            whitened_coupling_norm=float(np.linalg.norm(inverse_root(aa)@ac@inverse_root(cc),2))))
    return dict(source=str(folder),natoms=n,rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('folders',nargs='+',type=Path);a=p.parse_args()
    rows=[audit(folder) for folder in a.folders]
    a.output.write_text(json.dumps(dict(scope='qualified initial physical PES only; not later LS/Gaussian objectives; zero new PES',systems=rows),indent=2)+'\n')
    for r in rows:print(r['source'],r['rows'][0])
