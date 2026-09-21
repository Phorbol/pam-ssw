"""Fresh EMT diagnosis and one unbiased continuation of archived failed landing."""
import json,time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read
from pamssw.standalone.vc_geometry import ASEStressSurface,SymmetricLogStrainChart
from pamssw.standalone.generalized_numerics import safe_lbfgs
base=Path('research/ga_ssw/evidence/joint-vc-cu4-l36');out=Path('research/ga_ssw/evidence/joint-vc-cu4-failed-landing-diagnosis');out.mkdir(exist_ok=False)
r=json.loads((base/'result.json').read_text());saved=r['result']['records'][-1]['landing']
a=Atoms(saved['symbols'],positions=saved['positions'],cell=saved['cell'],pbc=True)
surface=ASEStressSurface(EMT());e,f,s=surface.evaluate(a)
original=read(base/'input.extxyz');oldchart=SymmetricLogStrainChart(original,strain_length=3.6)
q=oldchart.pack(a);ev=oldchart.evaluate(q,surface.evaluate)
measure=lambda g:max(float(np.linalg.norm(g[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(g[-6:])))
report=dict(source=str(base),scope='one saved failed true-quench landing only; no walk rerun',initial=dict(energy=e,energy_error=e-saved['energy'],fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(abs(s).max()),volume=a.get_volume(),cell_condition=float(np.linalg.cond(a.cell.array)),cell_singular_values=np.linalg.svd(a.cell.array,compute_uv=False).tolist(),old_chart_gradient_norm=measure(ev.gradient),old_chart_atomic_norm=float(np.linalg.norm(ev.gradient[:-6].reshape(-1,3),axis=1).max()),old_chart_cell_norm=float(np.linalg.norm(ev.gradient[-6:]))),plan=dict(optimizer='Safe-total',chart='new reference at failed physical landing, no biases/history',strain_length=3.6,gtol=.001,max_step=.2,maxiter=300,max_requests=301))
(out/'plan.json').write_text(json.dumps(report,indent=2));(out/'script.py').write_text(Path(__file__).read_text());start=time.monotonic()
chart=SymmetricLogStrainChart(a,strain_length=3.6)
def evaluate(q):
 ev=chart.evaluate(q,surface.evaluate);return ev.objective,chart.project(ev.gradient)
continued=safe_lbfgs(chart.pack(a),evaluate,gradient_norm=measure,step_norm=measure,gtol=.001,max_step=.2,maxiter=300,max_requests=301)
final=chart.evaluate(continued.q,surface.evaluate)
report['continuation']=dict(status=continued.status,steps=continued.steps,requests=continued.requests,energy=final.energy,fmax=float(np.linalg.norm(final.forces,axis=1).max()),stress_max=float(abs(final.stress).max()),volume=final.volume,cell_condition=float(np.linalg.cond(final.atoms.cell.array)),positions=final.atoms.positions.tolist(),cell=final.atoms.cell.array.tolist())
report['total_requests']=surface.requests;report['seconds']=time.monotonic()-start
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
