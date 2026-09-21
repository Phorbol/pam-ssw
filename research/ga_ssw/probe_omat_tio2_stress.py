"""39 calls maximum, 120 s CPU E/F/stress preflight of three published TiO2 cells."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import time
import traceback

OUT = Path('research/ga_ssw/evidence/omat-small-tio2-stress')
OUT.mkdir(parents=True, exist_ok=False)
START = time.monotonic()
REPORT = dict(status='running', request_cap=39, requests=0, timeout_seconds=120,
    device='cpu', threads=1, dtype='float64', strain_h=1e-5,
    stress_absolute_tolerance_eV_A3=1e-5, evaluations=[], structures=[],
    source_doi='10.1039/c7sc01459g', source_section='official SI section 7',
    source_pdf='literature/benchmark-sources/SC-008-C7SC01459G-s001.pdf',
    scope='unrelaxed E/F/stress consistency only; no phase stability ordering',
    environment={k:os.environ.get(k) for k in ('PYTHONNOUSERSITE','OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD')})
(OUT/'script.py').write_text(Path(__file__).read_text())
(OUT/'plan.json').write_text(json.dumps(REPORT, indent=2))
def deadline(signum, frame):
    raise TimeoutError('120 second TiO2 preflight limit')
signal.signal(signal.SIGALRM, deadline)
signal.alarm(120)
try:
    import numpy as np
    from ase.io import read
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from mace.calculators import MACECalculator
    REPORT['versions']={n:metadata.version(n) for n in ('mace-torch','torch','ase','numpy')}
    model=Path('/home/gengjianrui/.cache/mace/mace-omat-0-small.model')
    REPORT.update(model=str(model),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
        source_pdf_sha256=hashlib.sha256(Path(REPORT['source_pdf']).read_bytes()).hexdigest())
    manifest=json.loads(Path('literature/benchmark-sources/coordinate-manifest.json').read_text())
    calc=MACECalculator(model_paths=str(model),device='cpu',default_dtype='float64',enable_cueq=False,enable_oeq=False)
    REPORT['model_elements']=list(map(int,calc.z_table.zs))
    REPORT['load_seconds']=time.monotonic()-START
    def evaluate(atoms,label):
        if REPORT['requests'] >= REPORT['request_cap']:
            raise RuntimeError('request cap reached')
        REPORT['requests']+=1
        tick=time.monotonic()
        calc.calculate(atoms,properties=['energy','forces','stress'])
        result=dict(label=label,energy=float(calc.results['energy']),
            forces=np.asarray(calc.results['forces']).tolist(),stress=np.asarray(calc.results['stress']).tolist(),
            numbers=atoms.numbers.tolist(),positions=atoms.positions.tolist(),cell=atoms.cell.tolist(),
            pbc=atoms.pbc.tolist(),seconds=time.monotonic()-tick)
        if not all(np.isfinite(result[k]).all() for k in ('energy','forces','stress')):
            raise ValueError('nonfinite E/F/stress')
        REPORT['evaluations'].append(result)
        (OUT/'progress.json').write_text(json.dumps(REPORT,indent=2))
        print(label,result['energy'],flush=True)
        return result
    for name in ('rutile','anatase','tio2-b'):
        path=Path('literature/benchmark-sources/coordinates')/(name+'.extxyz')
        atoms=read(path)
        if len(atoms)!=12 or atoms.get_chemical_formula()!='O8Ti4' or not atoms.pbc.all():
            raise ValueError('expected published 12-atom fully periodic TiO2 cell')
        (OUT/path.name).write_bytes(path.read_bytes())
        record=dict(name=name,input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_manifest=next(r for r in manifest if r['extxyz']==str(path)),
            ase_input_info=atoms.info,volume=atoms.get_volume())
        REPORT['structures'].append(record)
        initial=evaluate(atoms,name+':initial')
        record['initial']=initial
        record['initial_fmax']=float(np.linalg.norm(initial['forces'],axis=1).max())
        record['stress_comparison']=[]
        for k,(i,j) in enumerate(((0,0),(1,1),(2,2),(1,2),(0,2),(0,1))):
            basis=np.zeros((3,3));basis[i,j]=1. if i==j else .5
            if i!=j:basis[j,i]=.5
            energies=[]
            for sign in (-1,1):
                trial=atoms.copy()
                trial.set_cell(atoms.cell @ (np.eye(3)+sign*1e-5*basis),scale_atoms=True)
                energies.append(evaluate(trial,f'{name}:{i}{j}:{sign:+}')['energy'])
            fd=(energies[1]-energies[0])/(2e-5*record['volume'])
            analytic=initial['stress'][k]
            record['stress_comparison'].append(dict(voigt=k,ij=[i,j],finite_difference=fd,
                analytic=analytic,absolute_error=abs(fd-analytic)))
        record['max_absolute_error']=max(r['absolute_error'] for r in record['stress_comparison'])
        record['status']='passed' if record['max_absolute_error']<=1e-5 else 'stress_mismatch'
    REPORT['status']='passed' if all(r['status']=='passed' for r in REPORT['structures']) else 'stress_mismatch'
    REPORT['convention']='ASE tensile-positive stress dE/dstrain/V; row Hnew=H(I+hB), fixed fractions; offdiagonal B entries=1/2'
except Exception as error:
    REPORT['status']='failed'
    REPORT['error']=repr(error)
    REPORT['traceback']=traceback.format_exc()
finally:
    signal.alarm(0)
    REPORT['seconds']=time.monotonic()-START
    (OUT/'result.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps(dict(status=REPORT['status'],requests=REPORT['requests'],seconds=REPORT['seconds'],
        error=REPORT.get('error'),structures=[{k:r.get(k) for k in ('name','initial_fmax','max_absolute_error','status')}
        for r in REPORT['structures']]),indent=2),flush=True)
