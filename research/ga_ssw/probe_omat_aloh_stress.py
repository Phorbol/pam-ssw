"""13 CPU E/F/stress calls: uploaded AlOH first frame, fixed fractional strain FD."""
import hashlib,json,os,signal,time,traceback
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
from ase.io import read

OUT=Path('research/ga_ssw/evidence/omat-small-aloh-stress');OUT.mkdir(parents=True,exist_ok=False)
INPUT=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE1-AlOH/addition/add.arc')
MODEL=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
report=dict(status='running',input=str(INPUT),model=str(MODEL),model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),frame=0,device='cpu',dtype='float64',threads=1,h=1e-5,request_cap=13,timeout_seconds=180,stress_absolute_tolerance_eV_A3=1e-5,evaluations=[])
(OUT/'plan.json').write_text(json.dumps(report,indent=2));(OUT/'script.py').write_text(Path(__file__).read_text());start=time.monotonic()
def deadline(signum,frame):raise TimeoutError('180 second CPU preflight limit')
signal.signal(signal.SIGALRM,deadline);signal.alarm(180)
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from mace.calculators import MACECalculator
 report['versions']={n:metadata.version(n) for n in ['mace-torch','torch','ase','numpy']}
 a=read(INPUT,index=0,format='dmol-arc');assert len(a)==26 and a.pbc.all()
 report['structure']=dict(numbers=a.numbers.tolist(),positions=a.positions.tolist(),cell=a.cell.tolist(),pbc=a.pbc.tolist(),formula=a.get_chemical_formula())
 calc=MACECalculator(model_paths=str(MODEL),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 report['model_elements']=list(map(int,calc.z_table.zs));report['load_seconds']=time.monotonic()-start
 def evaluate(atoms,label):
  t=time.monotonic();calc.calculate(atoms,properties=['energy','forces','stress'])
  r=dict(label=label,energy=float(calc.results['energy']),forces=np.array(calc.results['forces']).tolist(),stress=np.array(calc.results['stress']).tolist(),cell=atoms.cell.tolist(),positions=atoms.positions.tolist(),seconds=time.monotonic()-t)
  assert np.isfinite(r['energy']) and np.isfinite(r['forces']).all() and np.isfinite(r['stress']).all()
  report['evaluations'].append(r);(OUT/'progress.json').write_text(json.dumps(report,indent=2));print(label,r['energy'],flush=True);return r
 initial=evaluate(a,'initial');report['initial_fmax']=float(np.linalg.norm(initial['forces'],axis=1).max());volume=a.get_volume();components=[]
 for k,(i,j) in enumerate([(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]):
  basis=np.zeros((3,3));basis[i,j]=1. if i==j else .5
  if i!=j:basis[j,i]=.5
  energies=[]
  for sign in [-1,1]:
   b=a.copy();b.set_cell(a.cell @ (np.eye(3)+sign*1e-5*basis),scale_atoms=True)
   energies.append(evaluate(b,f'{i}{j}:{sign:+}')['energy'])
  fd=(energies[1]-energies[0])/(2e-5*volume);analytic=initial['stress'][k]
  components.append(dict(voigt=k,ij=[i,j],finite_difference=fd,analytic=analytic,absolute_error=abs(fd-analytic)))
 report['stress_comparison']=components;report['max_absolute_error']=max(r['absolute_error'] for r in components)
 report['status']='passed' if report['max_absolute_error']<=1e-5 else 'stress_mismatch'
 report['convention']='ASE tensile stress = dE/dstrain / V; off-diagonal symmetric basis uses 1/2 entries; row-cell A_new=A@(I+strain), fixed fractions'
except Exception as exc:
 report['status']='failed';report['error']=repr(exc);report['traceback']=traceback.format_exc()
finally:
 signal.alarm(0);report['seconds']=time.monotonic()-start;report['requests']=len(report['evaluations']);(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['evaluations','structure','model_elements','traceback']},indent=2),flush=True)
