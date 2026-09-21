"""Approved five-EFS/120 s single-thread real XXXII RC-VC derivative preflight."""
import json,os,time,signal,traceback,shutil
from pathlib import Path
import numpy as np
from ase.io import read,write
from tblite.ase import TBLite
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface
OUT=Path('research/ga_ssw/evidence/xxxii-rc-vc-gfn2-preflight');OUT.mkdir(exist_ok=True)
if (OUT/'result.json').exists() or (OUT/'calls.json').exists():raise RuntimeError('refuse repeat preflight')
source=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc')
fixture=Path('tests/standalone/fixtures/type2_xxxii.extxyz');shutil.copy2(fixture,OUT/'input.extxyz')
for name in ('rigidbody','blist'):shutil.copy2(source/name,OUT/name)
for name in ('rc_geometry','rc_forest','rc_vc_geometry','rc_optimization_domain','rc_topology','rc_periodic_input','vc_geometry'):
 shutil.copy2(Path('pamssw/standalone')/(name+'.py'),OUT/(name+'.py'))
shutil.copy2(__file__,OUT/'script.py')
a=read(fixture);top=read_rigid_topology(source/'rigidbody',source/'blist',natoms=len(a));lift=unwrap_rigid_molecules(a,top.bonds)
chart=PrincipalRigidForestCellChart(lift.atoms,top.components,anchor=0,rotation_length=1.,torsion_length=1.,strain_length=5.)
q=np.zeros(chart.dimension);direction=np.random.default_rng(17).normal(size=chart.dimension);direction/=np.linalg.norm(direction)
plan=dict(input=str(fixture),topology=str(source),natoms=len(a),formula=a.get_chemical_formula(),model='GFN2-xTB',model_note='independent alternative PES; not original GAFF equivalence',accuracy=.001,charge=0,multiplicity=1,seed=17,dimension=chart.dimension,direction=direction.tolist(),rotation_length=1.,torsion_length=1.,strain_length=5.,pressure=0.,h_values=[1e-4,5e-5],max_EFS=5,max_seconds=120,threads={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},order=['base','plus_h','minus_h','plus_half_h','minus_half_h'],image_shifts=lift.images.tolist(),note='No quench or search. Metrics are explicit numerical chart definitions, not optimal parameters.')
(OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');write(OUT/'lifted.extxyz',lift.atoms)
start=time.monotonic();calls=[];results=[];report=dict(status='running',physical_claim='none; fixed-point derivative preflight only')
class Limit(RuntimeError):pass
def alarm(*args):raise Limit('approved 120 second total elapsed limit')
signal.signal(signal.SIGALRM,alarm);signal.alarm(120)
surface=ASEStressSurface(TBLite(method='GFN2-xTB',accuracy=.001,charge=0,multiplicity=1,verbosity=0))
def oracle(atoms):
 if len(calls)>=5 or time.monotonic()-start>=120:raise Limit('approved preflight limit')
 row=dict(index=len(calls),start_seconds=time.monotonic()-start,status='running');calls.append(row)
 (OUT/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
 try:
  e,f,s=surface.evaluate(atoms)
  row.update(status='completed',energy=e,forces=f.tolist(),stress=s.tolist(),fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(abs(s).max()),volume=atoms.get_volume())
  return e,f,s
 except BaseException as exc:
  row.update(status='failed',error=repr(exc));raise
 finally:
  row['elapsed_seconds']=time.monotonic()-start-row['start_seconds'];(OUT/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
try:
 for step in (0.,1e-4,-1e-4,5e-5,-5e-5):
  ev=chart.evaluate(q+step*direction,oracle,pressure=0.);results.append(dict(step=step,energy=ev.energy,gradient=ev.gradient.tolist(),directional_gradient=float(ev.gradient@direction)))
 derivative=results[0]['directional_gradient'];pairs=[]
 for h,i in ((1e-4,1),(5e-5,3)):
  fd=(results[i]['energy']-results[i+1]['energy'])/(2*h);pairs.append(dict(h=h,finite_difference=fd,analytic=derivative,absolute_error=abs(fd-derivative),relative_error=abs(fd-derivative)/max(abs(fd),abs(derivative),np.finfo(float).tiny)))
 report.update(status='completed',derivative_checks=pairs)
except BaseException as exc:report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
finally:
 signal.alarm(0);report.update(results=results,total_EFS_attempts=len(calls),surface_requests=surface.requests,elapsed_seconds=time.monotonic()-start);(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2),flush=True)
