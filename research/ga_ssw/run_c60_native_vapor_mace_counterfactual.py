"""Counterfactual: native mode=1 geometry preprocessing then one MACE Safe400 quench."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.surface import ASESurface,quench
from research.ga_ssw.probe_native_vapor_oracle import load_elf,run as vapor_run

BASE=Path('research/ga_ssw/evidence/hard-c60-gfn2-paper-ls-memory400-single-step/results/paper-seed3')
OUT=Path('research/ga_ssw/evidence/c60-native-vapor-mace-counterfactual')
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')

def main():
 OUT.mkdir(parents=False,exist_ok=True); q=json.loads((BASE/'quenches.json').read_text())[-2]['result']['atoms']; source=Atoms(**q)
 blob,segs=load_elf('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_program/lasp'); digest=hashlib.sha256(blob).hexdigest()
 transformed=vapor_run(segs,source.positions.tolist(),1,1.7)
 if transformed['status']!='ok': raise RuntimeError(transformed)
 pre=source.copy();pre.positions=np.asarray(transformed['positions'])
 (OUT/'input-original.json').write_text(json.dumps({'numbers':source.numbers.tolist(),'positions':source.positions.tolist(),'cell':source.cell.tolist(),'pbc':source.pbc.tolist()},indent=2)+'\n')
 (OUT/'native-mode1.json').write_text(json.dumps(transformed,indent=2)+'\n')
 manifest={'source':str(BASE/'quenches.json'),'source_quench_index':len(q),'source_call':q.get('call'),'elf_sha256':digest,'model':str(MODEL),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),'vapor_cri':1.7,'native_mode':1,'optimizer':'safe-lbfgs-total','lbfgs_memory':400,'fmax':.01,'maxiter':400,'max_requests':400,'wall_seconds':180,'scope':'counterfactual one-step state diagnostic; native mode1 preprocessing then MACE true quench; no trajectory parity claim'}
 (OUT/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
 from mace.calculators import MACECalculator
 calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 class Counted(ASESurface):
  def __init__(self):super().__init__(calc);self.begin=time.monotonic()
  def evaluate(self,a):
   if self.requests>=400 or time.monotonic()-self.begin>=180: raise RuntimeError('counterfactual budget exhausted')
   return super().evaluate(a)
 surface=Counted();t=time.monotonic();r=quench(pre,surface,fmax=.01,steps=400,optimizer='safe-lbfgs-total',lbfgs_memory=400)
 out={'status':'completed','requests':surface.requests,'wall_seconds':time.monotonic()-t,'energy':r.energy,'max_force':r.max_force,'converged':r.converged,'optimizer_steps':r.optimizer_steps,'atoms':{'numbers':r.atoms.numbers.tolist(),'positions':r.atoms.positions.tolist(),'cell':r.atoms.cell.tolist(),'pbc':r.atoms.pbc.tolist()}}
 (OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ('status','requests','wall_seconds','energy','max_force','converged','optimizer_steps')}))
if __name__=='__main__':main()
