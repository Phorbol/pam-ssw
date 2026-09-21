"""Approved 36-evaluation actual XXXII backend qualification; no relax/search."""
import json,time,signal,traceback,shutil,os,importlib.metadata
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from ase.io import read
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
OUT=Path('research/ga_ssw/evidence/xxxii-lammps-qualification-table0-ewald10');OUT.mkdir(exist_ok=False)
MODEL=Path('research/ga_ssw/evidence/xxxii-stock-charmm-converted');FIX=Path('tests/standalone/fixtures/type2_xxxii.extxyz')
TOPO=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc')
for p in (FIX,MODEL/'lmp.data',MODEL/'in.simple',MODEL/'manifest.json',TOPO/'rigidbody',TOPO/'blist'):shutil.copy2(p,OUT/p.name)
for p in ('research/ga_ssw/xxxii_lammps_calculator.py','pamssw/standalone/vc_geometry.py','pamssw/standalone/rc_vc_geometry.py','pamssw/standalone/rc_forest.py','pamssw/standalone/rc_geometry.py','pamssw/standalone/rc_optimization_domain.py'):shutil.copy2(p,OUT/Path(p).name)
shutil.copy2(__file__,OUT/'script.py')
a=read(FIX);top=read_rigid_topology(TOPO/'rigidbody',TOPO/'blist',natoms=len(a));a=unwrap_rigid_molecules(a,top.bonds).atoms
vc=SymmetricLogStrainChart(a,strain_length=5.);q=vc.pack(a)
rc=PrincipalRigidForestCellChart(a,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
rng=np.random.default_rng(17);atomdir=rng.normal(size=a.positions.shape);atomdir-=atomdir.mean(axis=0);atomdir/=np.linalg.norm(atomdir)
rcdir=np.random.default_rng(17).normal(size=rc.dimension);rcdir/=np.linalg.norm(rcdir)
rotation=Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
plan=dict(expected_EFS=36,max_EFS=40,max_seconds=60,seed=17,h=[1e-4,5e-5],atomic_direction=atomdir.tolist(),RC_direction=rcdir.tolist(),strain_length=5.,rotation_length=1.,torsion_length=1.,ewald_accuracy=1e-10,rotation=rotation.tolist(),translation='first complete43atom molecule + cell row0',model='algebraically converted original custom GAFF; wholeengine native comparison still pending',versions={p:importlib.metadata.version(p) for p in ('ase','numpy','lammps')},threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},scope='No quench/search; report measured errors, do not infer stable minimum')
plan.update(numerical_control='table0 retained; sole changed factor Ewald accuracy1e-10; matched36points, original72EFS failures retained',prior_development_EFS=72,authorization='root approved two precision levels <=80EFS/120s total; each capped40EFS/60s; no physical parameter changes')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
class Table0Diagnostic(XXXIILammpsCalculator):
 def _new_engine(self):
  from lammps import lammps
  return lammps(cmdargs=['-log',str(OUT/f'engine-{len(list(OUT.glob("engine-*.log")))}.log'),'-screen','none'])
 def _initialize(self):
  super()._initialize()
  self._lmp.command('pair_modify table 0')
  self._lmp.command('kspace_style ewald 1e-10')
def new():return Table0Diagnostic(data_path=MODEL/'lmp.data',input_path=MODEL/'in.simple',model_manifest=MODEL/'manifest.json',reference_atoms=a)
calc=new();calls=[];report={};start=time.monotonic()
def timeout(*args):raise RuntimeError('approved120secondlimit')
signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
def evaluate(b,label,calculator=None):
 if len(calls)>=40 or time.monotonic()-start>=60:raise RuntimeError('approvedqualificationbudget')
 c=calc if calculator is None else calculator;row=dict(index=len(calls),label=label,status='running',start_seconds=time.monotonic()-start);calls.append(row)
 (OUT/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
 try:
  c.calculate(b);e=c.results['energy'];f=c.results['forces'];from ase.stress import voigt_6_to_full_3x3_stress
  s=voigt_6_to_full_3x3_stress(c.results['stress']);row.update(status='completed',energy=e,forces=f.tolist(),stress=s.tolist(),fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(abs(s).max()),id_type_charge_verified=True);return e,f,s
 except BaseException as exc:row.update(status='failed',error=repr(exc));raise
 finally:
  row['elapsed_seconds']=time.monotonic()-start-row['start_seconds'];(OUT/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
try:
 base=evaluate(a,'base');vcg=vc.evaluate(q,lambda b:base).gradient;rcg=rc.evaluate(np.zeros(rc.dimension),lambda b:base).gradient;checks=[]
 directions=[('atomic',-float(np.sum(base[1]*atomdir)),lambda t: (lambda b:(b.set_positions(a.positions+t*atomdir),b)[1])(a.copy()))]
 for k in range(6):
  d=np.zeros_like(q);d[-6+k]=1
  directions.append((f'cell{k}',float(vcg[-6+k]),lambda t,d=d:vc.unpack(q+t*d)))
 directions.append(('RC',float(rcg@rcdir),lambda t:rc.unpack(t*rcdir)))
 for name,analytic,geometry in directions:
  for h in (1e-4,5e-5):
   ep=evaluate(geometry(h),f'{name}+{h}')[0];em=evaluate(geometry(-h),f'{name}-{h}')[0];fd=(ep-em)/(2*h)
   checks.append(dict(direction=name,h=h,analytic=analytic,finite_difference=fd,absolute_error=abs(fd-analytic),relative_error=abs(fd-analytic)/max(abs(fd),abs(analytic),np.finfo(float).tiny)))
 b=a.copy();b.positions[:43]+=a.cell.array[0];translated=evaluate(b,'molecule_translation')
 b=a.copy();b.positions=b.positions@rotation.T;b.cell=b.cell.array@rotation.T;rotated=evaluate(b,'whole_rotation')
 with new() as fresh:origin=evaluate(a,'fresh_engine_origin',fresh)
 report.update(status='completed',derivatives=checks,invariance=dict(translation_energy=translated[0]-base[0],translation_force_max=float(abs(translated[1]-base[1]).max()),translation_stress_max=float(abs(translated[2]-base[2]).max()),rotation_energy=rotated[0]-base[0],rotation_force_max=float(abs(rotated[1]-base[1]@rotation.T).max()),rotation_stress_max=float(abs(rotated[2]-rotation@base[2]@rotation.T).max()),fresh_energy=origin[0]-base[0],fresh_force_max=float(abs(origin[1]-base[1]).max()),fresh_stress_max=float(abs(origin[2]-base[2]).max())))
except BaseException as exc:report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
finally:
 signal.alarm(0);calc.close();report.update(total_EFS_attempts=len(calls),elapsed_seconds=time.monotonic()-start);(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
