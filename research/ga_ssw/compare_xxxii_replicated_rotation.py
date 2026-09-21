"""Frozen corrected-backend rotation: separate finite difference and subspace effects."""
import json, time, shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface
from pamssw.standalone.generalized_numerics import generalized_dimer
from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator
from research.ga_ssw.generalized_central_dimer_probe import solve as central_dimer
from research.ga_ssw.generalized_ritz_probe import solve as ritz
from research.ga_ssw.compare_vc_arms import serial

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-replicated-forward-control'
OUT=SRC/'frozen-rotation-comparison';OUT.mkdir(exist_ok=False)
record=json.loads((SRC/'result.json').read_text())['result']['records'][1]
a=Atoms(**record['chart_reference'])
top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a))
chart=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
q=np.array(record['climb'][3]['scaled_coordinates'])
anchor=np.random.default_rng(3).normal(size=chart.dimension);anchor/=np.linalg.norm(anchor)
for name in ('xxxii_replicated_calculator.py','generalized_central_dimer_probe.py','generalized_ritz_probe.py'):
    shutil.copy2(ROOT/'research/ga_ssw'/name,OUT/name)
shutil.copy2(__file__,OUT/'runner-executed.py')
(OUT/'plan.json').write_text(json.dumps(dict(stage=4,source=str(SRC),schemes=['forward-dimer','central-dimer','central-ritz'],q=q.tolist(),anchor=anchor.tolist(),max_API_each=100,max_seconds_each=30,fd_step=1e-4,tol=.02,beta=100.,repetitions=[1,1,2],purpose='same physical geometry and anchor; isolate central difference from larger Ritz subspace'),indent=2)+'\n')
rows=[]
for scheme in ('forward-dimer','central-dimer','central-ritz'):
    c=XXXIIReplicatedCalculator(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a,repetitions=(1,1,2))
    surface=ASEStressSurface(c);start=time.monotonic();out=dict(scheme=scheme)
    def evaluate(x):
        if surface.requests>=100 or time.monotonic()-start>=30:raise RuntimeError('declared100API/30secondcap')
        ev=chart.evaluate(x,surface.evaluate,pressure=0.)
        with (OUT/f'{scheme}-calls.jsonl').open('a') as fp:
            fp.write(json.dumps(dict(q=x.tolist(),energy=ev.energy,gradient=ev.gradient.tolist()))+'\n')
        return ev.objective,ev.gradient
    try:
        if scheme=='forward-dimer':mode=generalized_dimer(q,anchor,rotation_bias=100.,fd_step=1e-4,max_hvp=100,tol=.02,evaluate=evaluate)
        elif scheme=='central-dimer':mode=central_dimer(q,anchor,rotation_bias=100.,fd_step=1e-4,max_force_calls=100,tol=.02,evaluate=evaluate)
        else:mode=ritz(q,anchor,evaluate,100.,1e-4,.02,100,'central')
        out.update(status='completed',mode=serial(mode))
    except Exception as exc:out.update(status='failed',error=repr(exc))
    finally:
        c.close();out.update(API=surface.requests,engine_calls=c.engine_calls,atoms_evaluated=c.atoms_evaluated,wall_seconds=time.monotonic()-start);rows.append(out)
        (OUT/f'{scheme}-result.json').write_text(json.dumps(out,indent=2)+'\n')
        print({k:v for k,v in out.items() if k!='mode'}, {k:v for k,v in out.get('mode',{}).items() if k!='direction'})
(OUT/'result.json').write_text(json.dumps(dict(arms=rows,total_API=sum(x['API'] for x in rows),engine_calls=sum(x['engine_calls'] for x in rows),atoms_evaluated=sum(x['atoms_evaluated'] for x in rows)),indent=2)+'\n')
