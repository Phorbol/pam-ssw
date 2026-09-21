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
IN=SRC/'frozen-rotation-comparison'
OUT=SRC/'rotation-common-certificate';OUT.mkdir(exist_ok=False)
record=json.loads((SRC/'result.json').read_text())['result']['records'][1]
a=Atoms(**record['chart_reference'])
top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a))
chart=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
q=np.array(record['climb'][3]['scaled_coordinates'])
anchor=np.random.default_rng(3).normal(size=chart.dimension);anchor/=np.linalg.norm(anchor)
shutil.copy2(__file__,OUT/'runner-executed.py')
(OUT/'plan.json').write_text(json.dumps(dict(max_API=12,fd_steps=[1e-4,2.5e-5],source=str(IN),purpose='same central HVP certificate for all3 returned directions, two spacings; no rotation or search'),indent=2)+'\n')
rows=[]
with XXXIIReplicatedCalculator(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a,repetitions=(1,1,2)) as c:
 surface=ASEStressSurface(c)
 for scheme in ('forward-dimer','central-dimer','central-ritz'):
  mode=json.loads((IN/f'{scheme}-result.json').read_text())['mode'];n=np.array(mode['direction'])
  for h in (1e-4,2.5e-5):
   ep=chart.evaluate(q+h*n,surface.evaluate,pressure=0.);em=chart.evaluate(q-h*n,surface.evaluate,pressure=0.)
   hn=(ep.gradient-em.gradient)/(2*h)-100.*np.dot(anchor,n)*anchor;curvature=float(n@hn)
   rows.append(dict(scheme=scheme,h=h,curvature=curvature,residual=float(np.linalg.norm(hn-curvature*n)),plus_gradient=ep.gradient.tolist(),minus_gradient=em.gradient.tolist()))
 report=dict(status='completed',total_API=surface.requests,engine_calls=c.engine_calls,atoms_evaluated=c.atoms_evaluated,rows=rows)
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print([{k:v for k,v in x.items() if 'gradient' not in k} for x in rows])
