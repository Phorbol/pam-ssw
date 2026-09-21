"""Real Cu/EMT joint-coordinate adapter check, not a global-search benchmark."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import ASEStressSurface, SymmetricLogStrainChart
from pamssw.standalone.generalized_numerics import safe_lbfgs
from research.ga_ssw.lbfgs_baselines import scipy_lbfgsb, ase_lbfgs_linesearch


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def main():
    out=Path('research/ga_ssw/evidence/lbfgs-baselines-cu-cell')
    out.mkdir(exist_ok=False)
    atoms=bulk('Cu','fcc',a=3.65,cubic=True)
    atoms.positions[0]+=[.03,-.02,.01]
    chart=SymmetricLogStrainChart(atoms,strain_length=3.6)
    q0=chart.pack(atoms)
    def norm(g):return max(float(np.linalg.norm(g[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(g[-6:])))
    rows=[]
    for pressure in [0.,.005]:
        for name,solver in [('safe_total',safe_lbfgs),('scipy_lbfgsb',scipy_lbfgsb),('ase_lbfgs_linesearch',ase_lbfgs_linesearch)]:
            surface=ASEStressSurface(EMT())
            def evaluate(q):
                ev=chart.evaluate(q,surface.evaluate,pressure=pressure)
                return ev.objective,chart.project(ev.gradient)
            kwargs=dict(gradient_norm=norm,step_norm=norm,gtol=.005,maxiter=150,max_requests=300)
            if name=='safe_total':kwargs.update(max_step=.2,lbfgs_memory=10)
            elif name=='scipy_lbfgsb':kwargs.update(maxcor=10)
            else:kwargs.update(memory=10)
            r=solver(q0,evaluate,**kwargs)
            assert r.requests==surface.requests
            verify=ASEStressSurface(EMT())
            ev=chart.evaluate(r.q,verify.evaluate,pressure=pressure)
            gradient=chart.project(ev.gradient)
            error=None if r.gradient is None else float(np.max(np.abs(gradient-r.gradient)))
            if error is not None:assert error<1e-10
            row=dict(solver=name,pressure=pressure,requests=r.requests,fresh_requests=verify.requests,
                steps=r.steps,status=r.status,common_gradient=norm(gradient),common_pass=norm(gradient)<=.005,
                physical_fmax=float(np.linalg.norm(ev.forces,axis=1).max()),
                physical_stress_max=float(np.abs(ev.stress+pressure*np.eye(3)).max()),
                objective=ev.objective,gradient_error=error,raw=asdict(r))
            rows.append(row)
            (out/f'{name}-p{pressure}.json').write_text(json.dumps(row,default=serial,indent=2)+'\n')
    summary=dict(scope='Cu4 EMT true joint quench adapter check, not SSW efficacy or optimizer ranking',
                 search_requests=sum(r['requests'] for r in rows),fresh_requests=sum(r['fresh_requests'] for r in rows),
                 rows=[{k:v for k,v in r.items() if k!='raw'} for r in rows])
    (out/'summary.json').write_text(json.dumps(summary,default=serial,indent=2)+'\n')
    print(json.dumps(summary,default=serial,indent=2))


if __name__=='__main__':main()
