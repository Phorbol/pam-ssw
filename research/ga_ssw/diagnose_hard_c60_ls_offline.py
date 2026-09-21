"""Replay optimizer against recorded E/F only: no calculator imports or PES calls."""
import json,numpy as np
from pathlib import Path
from ase import Atoms
from pamssw.standalone.surface import quench
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.softening import FrozenBondSoftening
import pamssw.relax as relax
P=Path('research/ga_ssw/evidence/hard-c60-gfn2-one-step-2000/paper-seed3')
r=json.loads((P/'result.json').read_text());initial=Atoms(**r['initial']['atoms']);events=r['records'][0]['climb'];prep=json.loads((P/'preparations.json').read_text())[0]
soft=FrozenBondSoftening(tuple(initial.numbers),tuple(map(tuple,initial.cell.array)),tuple(map(bool,initial.pbc)),tuple(map(tuple,prep['pairs'])),tuple(initial.get_distance(i,j) for i,j in prep['pairs']),tuple(prep['strengths']),.2)
terms=[soft]+[ProjectedGaussian(np.array(c['center']),np.array(c['direction']),c['width'],c['weight']) for c in events]
last=events[-1];start=initial.copy();start.positions=np.array(last['center'])+last['width']*np.array(last['direction'])
allrows=[json.loads(s) for s in (P/'evaluations.jsonl').read_text().splitlines()]
trace=[x for x in allrows if 999<=x['call']<=1421]
from research.ga_ssw.compare_vc_arms import serial
out=P/'offline-stage8';out.mkdir(exist_ok=True)
(out/'frozen-objective.json').write_text(json.dumps(serial(dict(start=start,soft=soft,gaussians=events,first_request=999,last_request=1421,no_PES=True)),indent=2))
class Recorded:
 requests=0
 def evaluate(self,a):
  row=trace[self.requests];err=float(np.max(np.abs(a.positions-np.array(row['atoms']['positions']))));errors.append(err)
  if err>1e-9:raise ValueError(f'trajectory mismatch request {row["call"]}: {err}')
  self.requests+=1
  return row['energy'],np.array(row['forces'])
errors=[];pairs=[];original=relax._accept_lbfgs_curvature

def capture(s,y):
 ok=original(s,y);pairs.append(dict(accepted_request=998+surface.requests,s_dot_y=float(s@y),step_norm=float(np.linalg.norm(s)),max_atom_step=float(np.linalg.norm(s.reshape(-1,3),axis=1).max()),y_norm=float(np.linalg.norm(y)),retained=bool(ok)));return ok
relax._accept_lbfgs_curvature=capture
try:
 surface=Recorded();q=quench(start,surface,fmax=.01,steps=400,terms=terms,optimizer='safe-lbfgs-total')
 summary=dict(requests=surface.requests,matched_max_coordinate_error=max(errors),converged=q.converged,steps=q.optimizer_steps,max_force=q.max_force,energy=q.energy,curvature_pairs=len(pairs),negative_pairs=sum(x['s_dot_y']<0 for x in pairs),rejected_curvature=sum(not x['retained'] for x in pairs),extra_PES=0)
except Exception as exc:summary=dict(error=repr(exc),requests=surface.requests,extra_PES=0)
finally:relax._accept_lbfgs_curvature=original
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'curvature-pairs.json').write_text(json.dumps(pairs,indent=2));print(summary)

components=[]
accepted_calls={999}|{v['accepted_request'] for v in pairs}
for row in trace:
 a=Atoms(**row['atoms']);e0=row['energy'];f0=np.array(row['forces']);es,fs=soft.evaluate(a);eg=0.;fg=np.zeros_like(f0)
 for term in terms[1:]:
  e,f=term.evaluate(a);eg+=e;fg+=f
 force=f0+fs+fg;pos=a.positions-a.positions.mean(0);basis=[]
 for ax in np.eye(3):basis.append(np.tile(ax,(len(a),1)).ravel())
 for ax in np.eye(3):basis.append(np.cross(np.tile(ax,(len(a),1)),pos).ravel())
 u,sv,v=np.linalg.svd(np.array(basis).T,full_matrices=False);u=u[:,sv>1e-10];fr=u@(u.T@force.ravel());norm=np.linalg.norm(force)
 components.append(dict(request=row['call'],accepted=row['call'] in accepted_calls,raw_energy=e0,soft_energy=es,gaussian_energy=eg,total_energy=e0+es+eg,raw_fmax=float(np.linalg.norm(f0,axis=1).max()),soft_fmax=float(np.linalg.norm(fs,axis=1).max()),gaussian_fmax=float(np.linalg.norm(fg,axis=1).max()),total_fmax=float(np.linalg.norm(force,axis=1).max()),rigid_force_norm_fraction=float(np.linalg.norm(fr)/norm),max_atom_force_norm_fraction=float(np.linalg.norm(force,axis=1).max()/norm)))
(out/'evaluation-components.json').write_text(json.dumps(components,indent=2))
print('component endpoints',components[0],components[-1])
