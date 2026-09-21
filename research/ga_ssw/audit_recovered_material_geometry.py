"""Offline structural screening of returned minima; no physical certification.

Nearest-O assignments are diagnostic geometry, not bond-order calculations.
Distances use free-cluster Cartesian coordinates; periodic inputs are rejected.
"""
import argparse,json
from pathlib import Path
import numpy as np


def screen(atoms):
    if any(atoms['pbc']): raise ValueError('free clusters only')
    x=np.array(atoms['positions']);z=np.array(atoms['numbers'])
    distances=np.linalg.norm(x[:,None]-x[None,:],axis=-1)
    row=dict(diameter=float(distances.max()),minimum_pair_distance=float(distances[np.triu_indices(len(x),1)].min()))
    if set(z)=={1,8}:
        o=x[z==8];h=x[z==1];oh=np.linalg.norm(h[:,None]-o[None,:],axis=-1)
        row.update(nearest_oh_range=[float(oh.min(axis=1).min()),float(oh.min(axis=1).max())],
                   nearest_h_per_o=np.bincount(oh.argmin(axis=1),minlength=len(o)).tolist())
    return row


def analyze(root):
    rows=[]
    for path in sorted(root.glob('*/result.json')):
        data=json.loads(path.read_text());minima=data['minima'];best=min(range(len(minima)),key=lambda i:minima[i]['energy'])
        rows.append(dict(run=path.parent.name,source=str(path),
            initial_energy=minima[0]['energy'],best_energy=minima[best]['energy'],
            best_energy_change=minima[best]['energy']-minima[0]['energy'],best_index=best,
            frame_count=len(minima),frames=[dict(index=i,**screen(m['atoms'])) for i,m in enumerate(minima)]))
    return dict(rows=rows,scope='Geometric screening only. Does not certify MLIP domain, basin identity, positive Hessian or DFT energies.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.write_text(json.dumps(analyze(args.root),indent=2)+'\n')
