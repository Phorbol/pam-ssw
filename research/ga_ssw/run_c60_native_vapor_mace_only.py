"""MACE-only stage consuming frozen native mode1 JSON; no Unicorn/PES preprocessing."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.surface import ASESurface,quench
OUT=Path('research/ga_ssw/evidence/c60-native-vapor-mace-counterfactual')
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
def main():
 d=json.loads((OUT/'native-mode1.json').read_text()); src=json.loads((OUT/'input-original.json').read_text())
 a=Atoms(numbers=src['numbers'],positions=d['positions'],cell=src['cell'],pbc=src['pbc'])
 from mace.calculators import MACECalculator
 calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 class Counted(ASESurface):
  def __init__(self):super().__init__(calc);self.begin=time.monotonic()
  def evaluate(self,x):
   if self.requests>=400 or time.monotonic()-self.begin>=180:raise RuntimeError('counterfactual budget exhausted')
   return super().evaluate(x)
 surf=Counted();t=time.monotonic();r=quench(a,surf,fmax=.01,steps=400,optimizer='safe-lbfgs-total',lbfgs_memory=400)
 out={'status':'completed','requests':surf.requests,'wall_seconds':time.monotonic()-t,'energy':r.energy,'max_force':r.max_force,'converged':r.converged,'optimizer_steps':r.optimizer_steps,'atoms':{'numbers':r.atoms.numbers.tolist(),'positions':r.atoms.positions.tolist(),'cell':r.atoms.cell.tolist(),'pbc':r.atoms.pbc.tolist()}}
 (OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','requests','wall_seconds','energy','max_force','converged','optimizer_steps')}))
if __name__=='__main__':main()
