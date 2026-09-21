"""Bounded experimental XXXII converted-GAFF RC-VC one-step runner; default prepare-only."""
import argparse,hashlib,json,os,signal,time,traceback,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity'); OUT=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-globalbudget-memory400'; SRC=ROOT/'research/ga_ssw/evidence/xxxii-lammps-qualification-table0-ewald12'; TOPO=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc'); FIX=ROOT/'tests/standalone/fixtures/type2_xxxii.extxyz'; MODEL=ROOT/'research/ga_ssw/evidence/xxxii-stock-charmm-converted'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 from ase.io import read
 from pamssw.standalone.rc_topology import read_rigid_topology
 from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
 from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
 a=read(FIX);top=read_rigid_topology(TOPO/'rigidbody',TOPO/'blist',natoms=len(a));lift=unwrap_rigid_molecules(a,top.bonds);chart=PrincipalRigidForestCellChart(lift.atoms,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
 cfg={'rotation_length':1.,'torsion_length':1.,'strain_length':5.,'width':.6,'rotation_bias':100.,'max_gaussians':12,'temperature_K':300.,'forward_force':.1,'gradient_tol':.005,'fmax':.01,'stress_tol':.001,'max_step':.2,'relax_steps':1498,'fd_step':1e-4,'rotation_hvp':100,'rotation_tol':.02,'pressure':0.,'lbfgs_memory':400}
 return {'status':'prepared','pes_calls':0,'input':str(FIX),'input_sha256':sha(FIX),'topology':{'rigidbody':str(TOPO/'rigidbody'),'blist':str(TOPO/'blist'),'components':len(top.components)},'model_files':{n:sha(MODEL/n) for n in ('lmp.data','in.simple','manifest.json')},'config':cfg,'fixed_G':.47570069,'pair_table':0,'ewald_accuracy':1e-12,'seed':3,'steps':1,'budget':{'max_EFS':1500,'search_cap':1498,'seconds':120,'fresh_reserved':2},'chart_dimension':chart.dimension,'scope':'experimental converted GAFF backend; not native whole-engine parity or production benchmark'}

def main(execute=False):
 from ase.io import read, write
 from pamssw.standalone.rc_topology import read_rigid_topology
 from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
 from pamssw.standalone.rc_vc_reference import RCVCSSWConfig, run_rc_vc_ssw
 from pamssw.standalone.vc_geometry import ASEStressSurface
 from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
 from research.ga_ssw.compare_vc_arms import serial
 if (OUT/'result.json').exists() or (OUT/'ledger.jsonl').exists():
  raise RuntimeError('refuse overwrite previously executed experiment')
 plan=prepare(); OUT.mkdir(exist_ok=True)
 plan['budget_control']='per-quench iteration ceiling equals shared1498searchAPI ceiling; total budget unchanged1500API incl2fresh. Comparememory10/400; no new convergence threshold.'
 plan['derivative_review']='fixedG-three-step-v2:52EFS; known erfc inconsistency and residual RC1.7e-7 retained; experimental at unchanged tolerances'
 (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 if not execute:
  print(json.dumps({'status':'prepared','pes_calls':0,'chart_dimension':plan['chart_dimension']}));return
 # Freeze the actual imported implementation before evaluation.
 shutil.copytree(ROOT/'pamssw',OUT/'source/pamssw',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 for name in ('xxxii_lammps_calculator.py','run_existing_lammps.py','compare_vc_arms.py'):
  shutil.copy2(ROOT/'research/ga_ssw'/name,OUT/name)
 shutil.copy2(__file__,OUT/'runner-executed.py')
 for path in (FIX,TOPO/'rigidbody',TOPO/'blist',MODEL/'lmp.data',MODEL/'in.simple',MODEL/'manifest.json'):
  shutil.copy2(path,OUT/path.name)
 import lammps,sys,ase
 (OUT/'environment.json').write_text(json.dumps(dict(python=sys.executable,lammps=lammps.__file__,lammps_version=lammps.__version__,ase=ase.__version__,numpy=np.__version__,variables={k:os.environ.get(k) for k in ('PYTHONPATH','LD_LIBRARY_PATH','PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')}),indent=2)+'\n')
 a=read(OUT/FIX.name);top=read_rigid_topology(OUT/'rigidbody',OUT/'blist',natoms=len(a));a=unwrap_rigid_molecules(a,top.bonds).atoms
 engines=[]
 class FixedG(XXXIILammpsCalculator):
  def _new_engine(self):
   from lammps import lammps
   return lammps(cmdargs=['-log',str(OUT/f'engine-{len(list(OUT.glob("engine-*.log")))}.log'),'-screen','none'])
  def _initialize(self):
   super()._initialize()
   for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069'):self._lmp.command(command)
 def calculator():
  c=FixedG(data_path=OUT/'lmp.data',input_path=OUT/'in.simple',model_manifest=OUT/'manifest.json',reference_atoms=a);engines.append(c);return c
 cfg=RCVCSSWConfig(**plan['config']); start=time.monotonic(); counts={'search':0,'fresh':0};attempts=[]
 class Counted(ASEStressSurface):
  def __init__(self,c,role):super().__init__(c);self.role=role
  def evaluate(self,atoms):
   row=dict(index=len(attempts),role=self.role,positions=atoms.positions.tolist(),cell=atoms.cell.array.tolist(),numbers=atoms.numbers.tolist(),pbc=atoms.pbc.tolist(),api_calls=0)
   attempts.append(row); engine_before=self.calculator.requests
   try:
    if sum(counts.values())>=1500 or counts[self.role]>=(1498 if self.role=='search' else 2):raise RuntimeError('declared request budget exhausted')
    if time.monotonic()-start>=120:raise RuntimeError('declared wall budget exhausted')
    counts[self.role]+=1;row['api_calls']=1
    e,f,s=super().evaluate(atoms);row.update(status='completed',energy=e,forces=f.tolist(),stress=s.tolist());return e,f,s
   except BaseException as exc:row.update(status='failed',error=repr(exc));raise
   finally:
    row['elapsed_seconds']=time.monotonic()-start
    row['engine_calls']=self.calculator.requests-engine_before
    with (OUT/'ledger.jsonl').open('a') as fp:fp.write(json.dumps(row)+'\n')
 def timeout(*args):raise RuntimeError('declared 120second wall limit')
 signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,120)
 out={'status':'running','result':None,'fresh':[]}
 try:
  surface=Counted(calculator(),'search')
  r=run_rc_vc_ssw(a,surface,trees=top.components,anchor=0,steps=1,config=cfg,rng=np.random.default_rng(3))
  out.update(status=r.status,result=serial(r))
  for i,m in enumerate(r.minima):
   if i>=2:break
   write(OUT/f'minimum-{i}.extxyz',m.atoms)
   fresh=Counted(calculator(),'fresh');e,f,s=fresh.evaluate(m.atoms)
   out['fresh'].append(dict(index=i,energy=e,energy_error=e-m.energy,forces=f.tolist(),stress=s.tolist(),fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(abs(s+cfg.pressure*np.eye(3)).max()),volume=m.atoms.get_volume()))
 except BaseException as exc:
  out.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
  for c in engines:c.close()
  out.update(api_requests=counts,total_EFS=sum(counts.values()),completed_EFS=sum(row['status']=='completed' for row in attempts),engine_calls=sum(c.requests for c in engines),wall_seconds=time.monotonic()-start)
  (OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n')
  print(json.dumps({k:v for k,v in out.items() if k not in ('result','fresh','traceback')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');main(p.parse_args().execute)
