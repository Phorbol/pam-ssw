"""Audit completed Hessians and the existing Safe-total direction; zero PES."""
import json,argparse
from pathlib import Path
import numpy as np
from pamssw.relax import _accept_lbfgs_curvature,_lbfgs_inverse_product


def audit(root):
    rows=[]
    for arm in ('ls_all','ls_filter'):
        for seed in (7,101):
            folder=root/'diagnostic'/f'{arm}-seed{seed}';d=json.loads((folder/'result.json').read_text())
            assert d['status']=='completed' and d['requests']==973
            assert d['reconstruction_error']<1e-8
            ledger=[json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]
            assert sum(bool(r['charged']) for r in ledger)==973
            source=json.loads(Path(d['source']).read_text());trace=source['safe_lbfgs']['trace']
            history=[]
            for a,b in zip(trace,trace[1:]):
                s=np.array(b['q'])-a['q'];y=np.array(b['gradient'])-a['gradient']
                if _accept_lbfgs_curvature(s,y):history.append((s,y,1/float(s@y)));history=history[-10:]
            g=np.array(trace[-1]['gradient']);direction=-_lbfgs_inverse_product(g,history)
            basis=np.array(json.loads((folder/'basis.json').read_text()))
            h=np.load(folder/'hessian-5e-05-sym.npy');w,u=np.linalg.eigh(h)
            error=d['step_difference_spectral'];negative=w < -error
            gf=u.T@basis.T@g;df=u.T@basis.T@direction
            slope=float(g@direction);assert slope<0
            full_h=basis@h@basis.T
            atomic_w=np.linalg.eigvalsh(full_h[:-6,:-6])
            cell_w=np.linalg.eigvalsh(full_h[-6:,-6:])
            rows.append(dict(case=f'{arm}-seed{seed}',requests=973,min=float(w[0]),max=float(w[-1]),
                step_difference=error,negative_modes=int(negative.sum()),
                lowest_mode_cell_fraction=float(np.linalg.norm((basis@u[:,0])[-6:])**2),
                gradient_negative_subspace_fraction=float(np.sum(gf[negative]**2)/np.sum(gf**2)),
                direction_negative_subspace_fraction=float(np.sum(df[negative]**2)/np.sum(df**2)),
                safe_direction_slope=slope,safe_direction_rayleigh=float(np.sum(w*df**2)/np.sum(df**2)),
                atomic_min=float(atomic_w[0]),cell_min=float(cell_w[0]),cell_max=float(cell_w[-1]),
                reconstruction_error=d['reconstruction_error']))
    return dict(cases=rows,total_requests=sum(r['requests'] for r in rows),new_audit_requests=0,
        boundary='nonstationary complete biased objectives in declared logstrain chart; not material stability or optimizer efficacy proof')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=audit(a.root);(a.root/'diagnostic/audit-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for r in result['cases']:print(r)
