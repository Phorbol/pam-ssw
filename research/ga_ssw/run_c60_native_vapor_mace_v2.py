import json,time,hashlib
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.surface import ASESurface,quench
OUT=Path('research/ga_ssw/evidence/c60-native-vapor-mace-counterfactual-v2');MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
def main():
 s=json.loads((OUT/'input-original.json').read_text());d=json.loads((OUT/'native-mode1.json').read_text());a=Atoms(numbers=s['numbers'],positions=d['positions'],cell=s['cell'],pbc=s['pbc'])
 from mace.calculators import MACECalculator
 calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 class C(ASESurface):
  def __init__(self):super().__init__(calc);self.t=time.monotonic()
  def evaluate(self,x):
   if self.requests>=399 or time.monotonic()-self.t>=180:raise RuntimeError('cap')
   return super().evaluate(x)
 surf=C();t=time.monotonic();r=quench(a,surf,fmax=.01,steps=400,optimizer='safe-lbfgs-total',lbfgs_memory=400)
 # fresh separate calculator, 1 reserved EF
 freshcalc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False);f=ASESurface(freshcalc);ef,ff=f.evaluate(r.atoms);fmax=float(np.linalg.norm(ff,axis=1).max())
 out={'status':'completed','quench_requests':surf.requests,'fresh_requests':f.requests,'total_requests':surf.requests+f.requests,'wall_seconds':time.monotonic()-t,'energy':r.energy,'max_force':r.max_force,'converged':r.converged,'optimizer_steps':r.optimizer_steps,'fresh_energy':ef,'fresh_max_force':fmax,'fresh_cutoff_A':1.6399999618530273,'atoms':{'numbers':r.atoms.numbers.tolist(),'positions':r.atoms.positions.tolist(),'cell':r.atoms.cell.tolist(),'pbc':r.atoms.pbc.tolist()}}
 init=float(json.loads((OUT/'input-original.json').read_text()).get('energy',float('nan')));out['original_ls_initial_energy']=init
 (OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','quench_requests','fresh_requests','total_requests','wall_seconds','energy','max_force','converged','fresh_energy','fresh_max_force')}))
if __name__=='__main__':main()
