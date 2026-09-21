"""Bounded post-run structural and local Hessian diagnosis; no relaxation."""
import hashlib,json,os,signal,time,traceback
from pathlib import Path
import importlib.metadata as metadata
OUT=Path('research/ga_ssw/evidence/joint-vc-aloh26-diagnosis');OUT.mkdir(parents=True,exist_ok=False)
SOURCE=Path('research/ga_ssw/evidence/joint-vc-aloh26-l5-clean/result.json')
REPORT=dict(status='running',source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    request_cap=200,planned_requests=171,timeout_seconds=90,requests=0,h=1e-4,strain_length=5.,
    device='cpu',threads=1,geometry={},evaluations=[],scope='local approximate stability and geometry; no strict minimum or phase identification')
(OUT/'plan.json').write_text(json.dumps(REPORT,indent=2));(OUT/'script.py').write_text(Path(__file__).read_text())
(OUT/'vc_geometry.py').write_text(Path('pamssw/standalone/vc_geometry.py').read_text())
start=time.monotonic()
def deadline(s,f):raise TimeoutError('90 second AlOH diagnostic limit')
signal.signal(signal.SIGALRM,deadline);signal.alarm(90)
try:
 import numpy as np
 from scipy.linalg import null_space
 from ase import Atoms,units
 from ase.neighborlist import neighbor_list
 from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
 original=json.loads(SOURCE.read_text())['result']
 def atoms_of(r):return Atoms(symbols=r['symbols'],positions=r['positions'],cell=r['cell'],pbc=True)
 structures={k:atoms_of(r) for k,r in [('initial',original['initial']),('landing',original['minima'][-1])]}
 delta=original['minima'][-1]['objective']-original['initial']['objective']
 REPORT['mc']=dict(delta_enthalpy_eV=delta,temperature_K=300.,acceptance_probability=float(np.exp(-delta/(units.kB*300))),
    recorded_accepted=original['records'][-1]['accepted'],current_is_initial=original['current']==original['initial'],
    note='Positive delta does not deterministically forbid Metropolis acceptance; recorded rejection is consistent. RNG variate was not archived.')
 for name,a in structures.items():
  ii,jj,dd,shifts=neighbor_list('ijdS',a,6.,self_interaction=False)
  z=a.numbers;row=dict(formula=a.get_chemical_formula(),numbers=z.tolist(),volume_A3=a.get_volume(),
      density_g_cm3=float(a.get_masses().sum()/a.get_volume()*1.66053906660),
      minimum_distance_A=float(dd.min()),species_minimum_distances_A={},coordination={},nearest_oxygen_for_hydrogen=[],
      pair_data_below_3_3_A=[])
  for za in sorted(set(z)):
   for zb in sorted(set(z)):
    if za>zb:continue
    mask=(z[ii]==za)&(z[jj]==zb)
    row['species_minimum_distances_A'][f'{za}-{zb}']=float(dd[mask].min()) if mask.any() else None
  for center,neighbor,cutoffs in [(13,8,[2.1,2.3,2.5]),(8,1,[1.1,1.2,1.3,1.5]),(1,8,[1.1,1.2,1.3,1.5])]:
   ids=np.flatnonzero(z==center)
   row['coordination'][f'{center}-{neighbor}']={str(cutoff):[int(np.sum((ii==i)&(z[jj]==neighbor)&(dd<cutoff))) for i in ids] for cutoff in cutoffs}
   row['coordination'][f'{center}-{neighbor}']['center_indices']=ids.tolist()
  for i in np.flatnonzero(z==1):
   selected=np.flatnonzero((ii==i)&(z[jj]==8));k=selected[np.argmin(dd[selected])]
   row['nearest_oxygen_for_hydrogen'].append(dict(h_index=int(i),o_index=int(jj[k]),image=shifts[k].tolist(),distance_A=float(dd[k])))
  for i,j,d,shift in zip(ii,jj,dd,shifts):
   if d<3.3:row['pair_data_below_3_3_A'].append(dict(i=int(i),j=int(j),distance_A=float(d),image=shift.tolist()))
  REPORT['geometry'][name]=row
 REPORT['composition_preserved']=bool(np.array_equal(structures['initial'].numbers,structures['landing'].numbers))
 (OUT/'geometry.json').write_text(json.dumps({k:REPORT[k] for k in ['geometry','mc','composition_preserved']},indent=2))
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from mace.calculators import MACECalculator
 REPORT['versions']={n:metadata.version(n) for n in ('mace-torch','torch','ase','numpy','scipy')}
 model=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
 REPORT['model']=str(model);REPORT['model_sha256']=hashlib.sha256(model.read_bytes()).hexdigest()
 calc=MACECalculator(model_paths=str(model),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 landing=structures['landing'].copy();shift=landing.positions.mean(0);landing.positions-=shift
 REPORT['chart_origin_shift_A']=shift.tolist()
 chart=SymmetricLogStrainChart(landing,strain_length=5.);q=chart.pack(landing)
 translations=np.zeros((chart.ndof,3))
 translations[:-6]=np.tile(np.eye(3),(len(landing),1))/np.sqrt(len(landing))
 basis=null_space(translations.T);REPORT['internal_dimension']=basis.shape[1]
 def evaluate(a):
  if REPORT['requests']>=200:raise RuntimeError('EFS request cap')
  REPORT['requests']+=1;calc.calculate(a,properties=['energy','forces','stress'])
  from ase.stress import voigt_6_to_full_3x3_stress
  e=float(calc.results['energy']);f=np.asarray(calc.results['forces']);s=voigt_6_to_full_3x3_stress(np.asarray(calc.results['stress']))
  REPORT['evaluations'].append(dict(energy=e,forces=f.tolist(),stress=s.tolist(),positions=a.positions.tolist(),cell=a.cell.tolist()))
  return e,f,s
 base=chart.evaluate(q,evaluate);columns=[]
 for i in range(basis.shape[1]):
  direction=basis[:,i];plus=chart.evaluate(q+1e-4*direction,evaluate);minus=chart.evaluate(q-1e-4*direction,evaluate)
  columns.append(basis.T@(plus.gradient-minus.gradient)/(2e-4))
  if i%10==0:
   print('Hessian column',i,'requests',REPORT['requests'],flush=True)
   (OUT/'progress.json').write_text(json.dumps(dict(status='running',columns=i+1,requests=REPORT['requests'],seconds=time.monotonic()-start)))
 matrix=np.column_stack(columns);sym=.5*(matrix+matrix.T);values,vectors=np.linalg.eigh(sym)
 np.savez(OUT/'hessian.npz',q=q,basis=basis,hessian_raw=matrix,hessian_symmetric=sym,eigenvalues=values,eigenvectors=vectors)
 REPORT['hessian']=dict(eigenvalues_eV_A2=values.tolist(),antisymmetry_frobenius=float(np.linalg.norm(matrix-matrix.T)),
    symmetric_frobenius=float(np.linalg.norm(sym)),minimum_curvature=float(values[0]),
    base_gradient_norm=float(np.linalg.norm(base.gradient)),base_energy=base.energy,
    base_fmax=float(np.linalg.norm(base.forces,axis=1).max()),base_stress_max=float(abs(base.stress).max()),
    mode_sensitivity=[])
 for mode in range(2):
  direction=basis@vectors[:,mode]
  for h in [5e-5,2e-4]:
   plus=chart.evaluate(q+h*direction,evaluate);minus=chart.evaluate(q-h*direction,evaluate)
   REPORT['hessian']['mode_sensitivity'].append(dict(mode=mode,h=h,
      gradient_secant_curvature=float(direction@(plus.gradient-minus.gradient)/(2*h)),
      energy_second_difference=float((plus.objective+minus.objective-2*base.objective)/h**2)))
 REPORT['status']='completed'
except Exception as error:
 REPORT['status']='failed';REPORT['error']=repr(error);REPORT['traceback']=traceback.format_exc()
finally:
 signal.alarm(0);REPORT['seconds']=time.monotonic()-start
 (OUT/'result.json').write_text(json.dumps(REPORT,indent=2)+'\n')
 print(json.dumps({k:v for k,v in REPORT.items() if k not in ['evaluations','geometry','hessian','traceback']},indent=2),flush=True)
 if 'hessian' in REPORT:print(json.dumps(REPORT['hessian'],indent=2),flush=True)
