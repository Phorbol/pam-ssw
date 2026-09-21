"""Two frozen real XXXII RC-VC rotations, same100API budget per Ritz method."""
import json,time,shutil
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
from research.ga_ssw.compare_vc_arms import serial
from research.ga_ssw.generalized_ritz_probe import solve
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-globalbudget-memory400';OUT=SRC/'rotation-ritz-comparison';OUT.mkdir(exist_ok=False)
d=json.loads((SRC/'result.json').read_text());record=d['result']['records'][1];cfg=json.loads((SRC/'plan.json').read_text())['config'];a=Atoms(**record['chart_reference']);top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=len(a));chart=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
anchor=np.random.default_rng(3).normal(size=chart.dimension);anchor/=np.linalg.norm(anchor)
shutil.copy2(__file__,OUT/'runner-executed.py');shutil.copy2(ROOT/'research/ga_ssw/generalized_ritz_probe.py',OUT/'generalized_ritz_probe.py')
(OUT/'plan.json').write_text(json.dumps(dict(source=str(SRC),stages=[0,4],schemes=['forward','central'],max_API_each=100,max_seconds_each=30,fd_step=1e-4,tol=.02,beta=100.,reference='docs/research/xxxii-rotation-subspace-diagnostic.md',scope='existingCartesianRitz adaptation in55D; no nativeCBDparity or productionchange'),indent=2)+'\n')
rows=[]
for index in (0,4):
 q=np.zeros(chart.dimension) if index==0 else np.array(record['climb'][index-1]['scaled_coordinates'])
 for scheme in ('forward','central'):
  target=OUT/f'stage{index}-{scheme}';target.mkdir()
  class FixedG(XXXIILammpsCalculator):
   def _new_engine(self):
    from lammps import lammps
    return lammps(cmdargs=['-log',str(target/'engine.log'),'-screen','none'])
   def _initialize(self):
    super()._initialize()
    for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069'):self._lmp.command(command)
  c=FixedG(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a);surface=ASEStressSurface(c);start=time.monotonic();out=dict(stage=index,scheme=scheme)
  def evaluate(x):
   if surface.requests>=100 or time.monotonic()-start>=30:raise RuntimeError('declared100API/30secondcap')
   e=chart.evaluate(x,surface.evaluate,pressure=0.)
   with (target/'calls.jsonl').open('a') as fp:fp.write(json.dumps(dict(q=x.tolist(),atoms=serial(e.atoms),energy=e.energy,forces=e.forces.tolist(),stress=e.stress.tolist(),gradient=e.gradient.tolist()))+'\n')
   return e.objective,e.gradient
  try:
   result=solve(q,anchor,evaluate,100.,1e-4,.02,100,scheme);out.update(status='completed',mode=serial(result))
  except Exception as exc:out.update(status='failed',error=repr(exc))
  finally:
   out.update(API=surface.requests,engine_calls=c.requests,seconds=time.monotonic()-start);c.close()
   (target/'result.json').write_text(json.dumps(out,indent=2)+'\n');rows.append(out);print({**{k:v for k,v in out.items() if k!='mode'},'mode':{k:v for k,v in out.get('mode',{}).items() if k!='direction'}})
(OUT/'result.json').write_text(json.dumps(dict(arms=rows,total_API=sum(x['API'] for x in rows)),indent=2)+'\n')
