"""Zero-PES replay of saved real C60 E/F through the explicit memory API."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.surface import quench
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.gaussian import ProjectedGaussian
import pamssw.relax as relax
P=Path('research/ga_ssw/evidence');out=P/'explicit-memory-api-replay';out.mkdir(exist_ok=False)
frozen=json.loads((P/'hard-c60-native-stage8/frozen-objective.json').read_text());a=Atoms(**frozen['start']);terms=[FrozenBondSoftening(**frozen['soft'])]+[ProjectedGaussian(np.array(t['center']),np.array(t['direction']),t['width'],t['weight']) for t in frozen['gaussians']]
old=[json.loads(l) for l in (P/'hard-c60-gfn2-one-step-2000/paper-seed3/evaluations.jsonl').read_text().splitlines() if 999<=json.loads(l)['call']<=1421]
new=[json.loads(l) for l in (P/'hard-c60-safe-memory400-stage8/evaluations.jsonl').read_text().splitlines() if not json.loads(l)['fresh']]
rows=[]
for memory,trace in [(None,old),(10,old),(400,new)]:
 errors=[]
 class Recorded:
  requests=0
  def evaluate(self,atoms):
   v=trace[self.requests];pos=v['atoms']['positions'] if memory!=400 else v['positions'];error=float(np.max(np.abs(atoms.positions-np.array(pos))));errors.append(error);assert error==0.,(memory,self.requests,error);self.requests+=1
   return (v['energy'],np.array(v['forces'])) if memory!=400 else (v['raw_energy'],np.array(v['raw_forces']))
 surface=Recorded();q=quench(a,surface,fmax=.01,steps=400,terms=terms,optimizer='safe-lbfgs-total',lbfgs_memory=memory)
 assert surface.requests==len(trace);assert relax._SAFE_LBFGS_MEMORY==10
 rows.append(dict(memory=memory,requests=surface.requests,steps=q.optimizer_steps,converged=q.converged,energy=q.energy,fmax=q.max_force,maximum_coordinate_error=max(errors)))
(out/'result.json').write_text(json.dumps(dict(rows=rows,new_physical_requests=0,global_memory=relax._SAFE_LBFGS_MEMORY),indent=2));(out/'script.py').write_text(Path(__file__).read_text());print(rows)
