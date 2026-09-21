"""Bounded real-EMT certificates for eight archived Cu4 joint-VC landings."""
import json,time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.neighborlist import neighbor_list
from pamssw.standalone.vc_geometry import ASEStressSurface,SymmetricLogStrainChart
from pamssw.standalone.generalized_numerics import safe_lbfgs

OUT=Path('research/ga_ssw/evidence/vc-cu4-minima-qualification');OUT.mkdir(exist_ok=False)
PLAN=dict(sources=['joint-vc-cu4-physical-stop','joint-vc-cu4-l36-seed17'],denominator=8,strain_length=3.6,fmax=1e-4,stress_tol=1e-5,quench_max_requests=101,quench_max_steps=100,hessian_steps=[1e-4,5e-5],hessian_dimension=15,total_request_cap=1500,neighbor_cutoff_A=5.,fingerprint_distance_tolerance_A=1e-3,volume_tolerance_A3=.01)
(OUT/'plan.json').write_text(json.dumps(PLAN,indent=2));(OUT/'script.py').write_text(Path(__file__).read_text())
class Surface(ASEStressSurface):
 total=0
 def evaluate(self,atoms):
  if Surface.total>=1500:raise RuntimeError('global 1500 request budget')
  Surface.total+=1;return super().evaluate(atoms)

def fingerprint(atoms):
 i,j,d=neighbor_list('ijd',atoms,5.,self_interaction=False)
 pairs={}
 for zi,zj,dist in zip(atoms.numbers[i],atoms.numbers[j],d):pairs.setdefault(f'{zi}-{zj}',[]).append(float(dist))
 for key in pairs:pairs[key].sort()
 return dict(volume=atoms.get_volume(),species_distances=pairs,neighbors_per_atom=[sorted(d[i==k].tolist()) for k in range(len(atoms))])

def maxnorm(v):return max(float(np.linalg.norm(v[:-6].reshape(-1,3),axis=1).max()),float(np.linalg.norm(v[-6:])))
rows=[];start=time.monotonic()
for name in PLAN['sources']:
 data=json.loads((Path('research/ga_ssw/evidence')/name/'result.json').read_text())
 for k,saved in enumerate(data['result']['minima']):
  a=Atoms(saved['symbols'],positions=saved['positions'],cell=saved['cell'],pbc=True);chart=SymmetricLogStrainChart(a,strain_length=3.6);surface=Surface(EMT());cache={}
  def evaluate(q):
   ev=chart.evaluate(q,surface.evaluate);cache[q.tobytes()]=max(np.linalg.norm(ev.forces,axis=1).max()/1e-4,abs(ev.stress).max()/1e-5)
   return ev.objective,chart.project(ev.gradient)
  relaxed=safe_lbfgs(chart.pack(a),evaluate,gradient_norm=maxnorm,step_norm=maxnorm,gtol=1.,max_step=.2,maxiter=100,max_requests=101,convergence_norm=lambda q,g:cache[q.tobytes()])
  ev=chart.evaluate(relaxed.q,surface.evaluate);b=ev.atoms
  # Fresh reference removes incidental distorted-chart scaling from curvature.
  local=SymmetricLogStrainChart(b,strain_length=3.6);q=local.pack(b);ndof=len(q)
  translations=np.zeros((ndof,3))
  for axis in range(3):translations[axis:ndof-6:3,axis]=1/np.sqrt(len(b))
  u,_,_=np.linalg.svd(translations,full_matrices=True);basis=u[:,3:]
  spectra=[]
  for h in PLAN['hessian_steps']:
   columns=[]
   for direction in basis.T:
    gp=local.evaluate(q+h*direction,surface.evaluate).gradient
    gm=local.evaluate(q-h*direction,surface.evaluate).gradient
    columns.append(basis.T@((gp-gm)/(2*h)))
   matrix=np.array(columns).T;spectra.append(dict(h=h,eigenvalues=np.linalg.eigvalsh((matrix+matrix.T)/2).tolist(),antisymmetry_norm=float(np.linalg.norm(matrix-matrix.T))))
  row=dict(source=name,index=k,quench_status=relaxed.status,quench_requests=relaxed.requests,requests=surface.requests,energy=ev.energy,fmax=float(np.linalg.norm(ev.forces,axis=1).max()),stress_max=float(abs(ev.stress).max()),volume=ev.volume,cell_condition=float(np.linalg.cond(b.cell.array)),positions=b.positions.tolist(),cell=b.cell.array.tolist(),symbols=b.get_chemical_symbols(),before_fingerprint=fingerprint(a),fingerprint=fingerprint(b),spectra=spectra)
  rows.append(row);(OUT/f'{name}-{k}.json').write_text(json.dumps(row,indent=2)+'\n');print(name,k,relaxed.status,ev.energy,row['fmax'],[s['eigenvalues'][0] for s in spectra],surface.requests,flush=True)
comparisons=[]
for i,a in enumerate(rows):
 for j,b in enumerate(rows[:i]):
  pa=a['fingerprint']['species_distances'];pb=b['fingerprint']['species_distances'];samekeys=pa.keys()==pb.keys()
  maxdiff=max((max(abs(np.array(pa[key])-np.array(pb[key])),default=0.) if len(pa[key])==len(pb[key]) else float('inf') for key in pa),default=float('inf')) if samekeys else float('inf')
  dv=abs(a['volume']-b['volume']);comparisons.append(dict(i=i,j=j,max_distance_difference=None if not np.isfinite(maxdiff) else float(maxdiff),volume_difference=dv,fingerprint_indistinguishable=bool(maxdiff<=.001 and dv<=.01)))
summary=dict(plan=PLAN,rows=rows,pair_comparisons=comparisons,total_requests=Surface.total,seconds=time.monotonic()-start)
(OUT/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print('TOTAL',Surface.total,time.monotonic()-start,flush=True)
