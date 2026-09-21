"""Bounded physical joint-Hessian qualification; no relaxation or SSW walk."""
import json, time, signal, traceback, hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import null_space
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
OUT=Path('research/ga_ssw/evidence/joint-vc-rutile12-l5-hessian');OUT.mkdir(exist_ok=False)
SOURCE=Path('research/ga_ssw/evidence/joint-vc-rutile12-l5/result.json')
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
report=dict(status='running',source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),model=str(MODEL),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),h=.001,strain_length=5.,pressure=0.,cap=180,timeout=60,requests=0,structures={})
(OUT/'plan.json').write_text(json.dumps(report,indent=2));(OUT/'script.py').write_text(Path(__file__).read_text())
started=time.monotonic()
def deadline(*args):raise TimeoutError('60 second qualification cap')
signal.signal(signal.SIGALRM,deadline);signal.alarm(60)
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from mace.calculators import MACECalculator
 calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 src=json.loads(SOURCE.read_text())['result']
 for label,r in [('initial',src['initial']),('landing',src['minima'][-1])]:
  atoms=Atoms(r['symbols'],positions=r['positions'],cell=r['cell'],pbc=True)
  chart=SymmetricLogStrainChart(atoms,strain_length=5.);q=chart.pack(atoms);n=len(atoms);t=np.zeros((len(q),3))
  for i in range(3):t[i:3*n:3,i]=1/np.sqrt(n)
  Q=null_space(t.T);assert Q.shape==(42,39)
  def evaluate(q,label_call):
   def oracle(a):
    if report['requests']>=180:raise RuntimeError('request cap')
    report['requests']+=1
    calc.calculate(a,properties=['energy','forces','stress'])
    e=float(calc.results['energy']);f=np.array(calc.results['forces']);s=voigt_6_to_full_3x3_stress(calc.results['stress'])
    row=dict(structure=label,label=label_call,request=report['requests'],energy=e,forces=f.tolist(),stress=s.tolist(),positions=a.positions.tolist(),cell=a.cell.tolist())
    with (OUT/'evaluations.jsonl').open('a') as fp:fp.write(json.dumps(row)+'\n')
    return e,f,s
   return chart.evaluate(q,oracle,pressure=0.)
  base=evaluate(q,'base');h=.001;H=np.empty((39,39))
  for i in range(39):
   plus=evaluate(q+h*Q[:,i],f'col{i}+');minus=evaluate(q-h*Q[:,i],f'col{i}-')
   H[:,i]=Q.T@(plus.gradient-minus.gradient)/(2*h)
  vals,vec=np.linalg.eigh((H+H.T)/2);v=Q@vec[:,0];checks=[]
  for dh in [h/2,h,2*h]:
   ep=evaluate(q+dh*v,f'mode+{dh}');em=evaluate(q-dh*v,f'mode-{dh}')
   checks.append(dict(h=dh,force_curvature=float(v@(ep.gradient-em.gradient)/(2*dh)),energy_curvature=float((ep.energy+em.energy-2*base.energy)/dh**2),energy_plus_delta=ep.energy-base.energy,energy_minus_delta=em.energy-base.energy))
  report['structures'][label]=dict(energy=base.energy,fmax=float(np.linalg.norm(base.forces,axis=1).max()),stress_frobenius=float(np.linalg.norm(base.stress)),projected_gradient_norm=float(np.linalg.norm(Q.T@base.gradient)),lowest_mode_gradient=float(v@base.gradient),eigenvalues=vals.tolist(),hessian_asymmetry=float(np.linalg.norm(H-H.T)),lowest_mode=v.tolist(),lowest_mode_atomic_norm=float(np.linalg.norm(v[:3*n])),lowest_mode_cell_norm=float(np.linalg.norm(v[3*n:])),checks=checks,hessian=H.tolist(),basis=Q.tolist(),reference_positions=atoms.positions.tolist(),reference_cell=atoms.cell.tolist())
  (OUT/'progress.json').write_text(json.dumps(report,indent=2));print(label,vals[:6],checks,flush=True)
 report['status']='completed'
except Exception as e:
 report['status']='failed';report['error']=repr(e);report['traceback']=traceback.format_exc()
finally:
 signal.alarm(0);report['seconds']=time.monotonic()-started;(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['requests'],report['seconds'],flush=True)
