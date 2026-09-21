"""Read-only atomic/cell diagnostics of recorded Safe-total trajectories."""
import json,argparse
from pathlib import Path
import numpy as np
from pamssw.relax import _accept_lbfgs_curvature


def audit(path):
    data=json.loads(path.read_text()); opt=data['safe_lbfgs']; trace=opt['trace']
    frames=[]; pairs=[]
    for row in trace:
        g=np.asarray(row['gradient']);q=np.asarray(row['q'])
        frames.append(dict(step=row['step'],energy=row['energy'],
            atom_max=float(np.linalg.norm(g[:-6].reshape(-1,3),axis=1).max()),
            cell_l2=float(np.linalg.norm(g[-6:])),
            cell_fraction_gradient_l2_squared=float((g[-6:]@g[-6:])/(g@g)) if g@g else 0.))
    for a,b in zip(trace,trace[1:]):
        s=np.array(b['q'])-a['q']; y=np.array(b['gradient'])-a['gradient']
        pairs.append(dict(step=b['step'],sy_atom=float(s[:-6]@y[:-6]),
            sy_cell=float(s[-6:]@y[-6:]),sy_total=float(s@y),
            cell_fraction_y_l2_squared=float((y[-6:]@y[-6:])/(y@y)) if y@y else 0.,
            gamma=float((s@y)/(y@y)) if y@y else None,
            kept_by_safe_total=bool(_accept_lbfgs_curvature(s,y)),
            atom_step=float(np.linalg.norm(s[:-6].reshape(-1,3),axis=1).max()),
            cell_step=float(np.linalg.norm(s[-6:]))))
    assert sum(p['kept_by_safe_total'] for p in pairs)==opt['accepted_secants']
    return dict(source=str(path),status=opt['status'],requests=data['requests'],
        initial=frames[0],final=frames[-1],frames=frames,pairs=pairs,
        cell_dominant_frames=sum(f['cell_l2']>f['atom_max'] for f in frames),
        total_frames=len(frames),nonpositive_or_small_secants=sum(not p['kept_by_safe_total'] for p in pairs),
        negative_cell_secants=sum(p['sy_cell']<0 for p in pairs),
        negative_atomic_secants=sum(p['sy_atom']<0 for p in pairs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths=sorted(a.root.glob('seed*/ls_*/history*/result.json'));assert len(paths)==8
    rows=[audit(path) for path in paths]
    a.output.write_text(json.dumps(dict(scope='same total objective; block diagnostics only, no change to solver or PES calls',rows=rows),indent=2)+'\n')
    for r in rows:
        f=r['final'];print(r['source'],r['status'],f['atom_max'],f['cell_l2'],r['cell_dominant_frames'],r['total_frames'],r['nonpositive_or_small_secants'])
