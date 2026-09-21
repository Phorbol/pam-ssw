"""Post-search strict Cu13/EMT structure and internal-Hessian diagnosis.

Sorted distances are permutation/rigid invariant but not injective: categories
are fingerprint-distinguishable candidates, not complete basin enumeration.
"""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface
from pamssw.standalone.surface import quench
from pamssw.standalone.cluster_frame import ClusterFrame


def fingerprint(a):
    return np.sort(a.get_all_distances()[np.triu_indices(len(a),1)])


def main(base='research/ga_ssw/evidence/dimer-ritz-cu13-eckart-final'):
    base=Path(base)
    out=base/'strict-validation';out.mkdir(exist_ok=False)
    (out/'script.py').write_text(Path(__file__).read_text())
    plan=dict(fmax=1e-5,quench_steps=300,fingerprint_max_abs_tolerance_A=1e-4,
        hessian_steps_A=[1e-4,5e-5],hessian='central force difference; symmetrized and rigid tangent space removed at each representative',
        note='post hoc validation; cost separate from search; not population retuning')
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    representatives=[];fps=[];rows=[];validation_requests=0
    for source in sorted(base.glob('[0-9]*-*.json')):
        data=json.loads(source.read_text());local=[]
        for i,m in enumerate(data['result']['minima']):
            a=Atoms(**m['atoms']);surface=ASESurface(EMT())
            q=quench(a,surface,fmax=plan['fmax'],steps=plan['quench_steps'])
            e,f=surface.evaluate(q.atoms);validation_requests+=surface.requests
            good=bool(np.linalg.norm(f,axis=1).max()<=plan['fmax']);identity=None
            if good:
                fp=fingerprint(q.atoms)
                for j,old in enumerate(fps):
                    if np.max(abs(fp-old))<=plan['fingerprint_max_abs_tolerance_A']:
                        identity=j;break
                if identity is None:
                    identity=len(fps);fps.append(fp);representatives.append(q.atoms.copy())
            row=dict(index=i,energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),qualified=good,
                fingerprint_group=identity,requests=surface.requests,positions=q.atoms.positions.tolist())
            local.append(row)
        (out/source.name).write_text(json.dumps(local,indent=2)+'\n')
        rows.append(dict(source=source.name,records=len(local),qualified=sum(r['qualified'] for r in local),fingerprint_groups=sorted(set(r['fingerprint_group'] for r in local if r['qualified'])),requests=sum(r['requests'] for r in local)))
    hessians=[]
    for i,a in enumerate(representatives):
        surface=ASESurface(EMT());e,f=surface.evaluate(a);frame=ClusterFrame(a)
        u,_,_=np.linalg.svd(frame.basis,full_matrices=True);q=u[:,6:];spectra=[]
        for h in plan['hessian_steps_A']:
            hessian=np.empty((a.positions.size,a.positions.size))
            for j in range(a.positions.size):
                d=np.zeros_like(a.positions);d.ravel()[j]=h
                plus=a.copy();plus.positions+=d;minus=a.copy();minus.positions-=d
                _,fp=surface.evaluate(plus);_,fm=surface.evaluate(minus)
                hessian[:,j]=-(fp-fm).ravel()/(2*h)
            reduced=q.T@((hessian+hessian.T)/2)@q
            spectra.append(dict(h=h,eigenvalues=np.linalg.eigvalsh(reduced).tolist(),antisymmetric_norm=float(np.linalg.norm(hessian-hessian.T))))
        hessians.append(dict(group=i,energy=e,min_pair_distance=float(fps[i][0]),diameter=float(fps[i][-1]),spectra=spectra,requests=surface.requests))
        validation_requests+=surface.requests
    result=dict(runs=rows,fingerprint_groups=len(representatives),representatives=hessians,total_validation_requests=validation_requests,
        scope='Cu13 ASE EMT, one initial geometry, two seeds per solver; positive finite-difference internal spectra are numerical local-stability evidence only')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(runs=rows,groups=len(representatives),total_validation_requests=validation_requests,
        representatives=[dict(group=r['group'],energy=r['energy'],lowest=[s['eigenvalues'][0] for s in r['spectra']]) for r in hessians]),indent=2))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='research/ga_ssw/evidence/dimer-ritz-cu13-eckart-final')
    main(parser.parse_args().input)
