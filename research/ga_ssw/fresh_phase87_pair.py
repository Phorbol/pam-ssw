"""Two charged fresh E/F/stress checks on retained phase-87 starts."""
import json, time
from pathlib import Path
import numpy as np
from ase import Atoms
import torch
from mace.calculators import MACECalculator
BASE=Path('research/ga_ssw/evidence/tio2-phase87-vc-pqc-single-step')
torch.set_num_threads(1);torch.set_num_interop_threads(1)
calc=MACECalculator(model_paths='/home/gengjianrui/.cache/mace/mace-omat-0-small.model',device='cpu',default_dtype='float64')
rows=[];t=time.monotonic()
for arm in ('pqc','joint'):
 r=json.loads((BASE/arm/'result.json').read_text());e=r['landings'][0];a=Atoms(**e['atoms'])
 calc.reset();calc.calculate(a,properties=['energy','forces','stress'])
 rows.append(dict(arm=arm,requests=1,energy=float(calc.results['energy']),energy_error=float(calc.results['energy']-e['energy']),fmax=float(np.linalg.norm(calc.results['forces'],axis=1).max()),stress_max=float(np.max(np.abs(calc.results['stress']))),volume=float(a.get_volume()),forces=np.asarray(calc.results['forces']).tolist(),stress=np.asarray(calc.results['stress']).tolist()))
report=dict(rows=rows,requests=2,wall_seconds=time.monotonic()-t,scope='retained initial structures only; no completed proposals')
(BASE/'fresh.json').write_text(json.dumps(report,indent=2)+'\n');(BASE/'fresh-executed.py').write_bytes(Path(__file__).read_bytes());print([(r['arm'],r['fmax'],r['stress_max']) for r in rows])
