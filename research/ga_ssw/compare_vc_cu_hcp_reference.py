"""One explicitly requested standard ASE hcp Cu reference, no global search."""
import json
from pathlib import Path
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.neighborlist import neighbor_list
from pamssw.standalone.vc_geometry import ASEStressSurface,SymmetricLogStrainChart
from pamssw.standalone.generalized_numerics import safe_lbfgs
out=Path('research/ga_ssw/evidence/vc-cu4-hcp-reference');out.mkdir(exist_ok=False)
a=bulk('Cu','hcp',a=2.55,c=4.16).repeat((2,1,1));chart=SymmetricLogStrainChart(a,strain_length=3.6);surface=ASEStressSurface(EMT());cache={}
plan=dict(source="ase.build.bulk('Cu','hcp',a=2.55,c=4.16).repeat((2,1,1))",initial_positions=a.positions.tolist(),initial_cell=a.cell.array.tolist(),fmax=1e-4,stress_tol=1e-5,strain_length=3.6,max_requests=149,maxiter=148,final_certificate_requests=1)
(out/'plan.json').write_text(json.dumps(plan,indent=2));(out/'script.py').write_text(Path(__file__).read_text())
def evaluate(q):
 ev=chart.evaluate(q,surface.evaluate);cache[q.tobytes()]=max(np.linalg.norm(ev.forces,axis=1).max()/1e-4,abs(ev.stress).max()/1e-5);return ev.objective,chart.project(ev.gradient)
def norm(g):return max(np.linalg.norm(g[:-6].reshape(-1,3),axis=1).max(),np.linalg.norm(g[-6:]))
r=safe_lbfgs(chart.pack(a),evaluate,gradient_norm=norm,step_norm=norm,gtol=1,max_step=.2,maxiter=148,max_requests=149,convergence_norm=lambda q,g:cache[q.tobytes()]);ev=chart.evaluate(r.q,surface.evaluate)
i,j,d=neighbor_list('ijd',ev.atoms,5.,self_interaction=False);dist=sorted(d.tolist());previous=json.loads(Path('research/ga_ssw/evidence/vc-cu4-minima-qualification/result.json').read_text());comparisons=[]
for k,row in enumerate(previous['rows']):
 old=row['fingerprint']['species_distances']['29-29'];err=float(max(abs(np.array(dist)-old))) if len(dist)==len(old) else None
 comparisons.append(dict(index=k,source=row['source'],landing_index=row['index'],max_shell_distance_difference=err,volume_per_atom_difference=abs(ev.volume-row['volume'])/4.,fingerprint_matches=bool(err is not None and err<=.001 and abs(ev.volume-row['volume'])<=.01)))
report=dict(plan=plan,status=r.status,requests=surface.requests,steps=r.steps,energy=ev.energy,volume=ev.volume,volume_per_atom=ev.volume/4,fmax=float(np.linalg.norm(ev.forces,axis=1).max()),stress_max=float(abs(ev.stress).max()),positions=ev.atoms.positions.tolist(),cell=ev.atoms.cell.array.tolist(),species_distances={'29-29':dist},comparisons=comparisons)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['species_distances','plan','positions','cell']},indent=2))
