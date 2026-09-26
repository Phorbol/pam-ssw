#!/bin/bash
#SBATCH --partition=CPU-MISC
#SBATCH --qos=rush-cpu
#SBATCH --account=sjtu-caoxiaoming
#SBATCH --time=00:05:00
#SBATCH --job-name=c60-neb-interface
#SBATCH --output=research/ga_ssw/evidence/c60-path-diagnostic-20260927/smoke-%j.out
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/gengjianrui/.conda/envs/mace_env/bin/python - <<'PY'
from pathlib import Path
import json, os, subprocess, sys
import numpy as np
from ase import Atoms
from ase.io import write, read
from ase.calculators.emt import EMT
p=Path('research/ga_ssw/evidence/c60-path-diagnostic-20260927')
out=p/f'smoke-{os.environ["SLURM_JOB_ID"]}';out.mkdir()
a=Atoms('Cu3',positions=[[0,0,0],[2.5,0,0],[1.25,2.16,0]])
b=a.copy();b.positions[2]+=[.2,0,.2]
write(out/'initial.extxyz',a);write(out/'final.extxyz',b)
for label,cap in [('normal',3000),('cap',5)]:
 command=[sys.executable,str(p/'run_neb.py'),'--initial',str(out/'initial.extxyz'),'--final',str(out/'final.extxyz'),'--out',str(out/label),'--calculator','emt','--smoke','--call-cap',str(cap)]
 proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (out/f'{label}.log').write_text(proc.stdout)
 s=json.loads((out/label/'summary.json').read_text())
 if label=='normal':
  assert proc.returncode==0,(proc.returncode,proc.stdout)
  assert s['status']=='completed' and len(s['images'])==7
  frames=read(out/label/'neb.traj',index=':');assert len(frames)>=14 and len(frames)%7==0
  path=read(out/label/'path.extxyz',index=':');assert len(path)==7
  for image,row in zip(path,s['images']):
   image.calc=EMT()
   assert np.isclose(image.get_potential_energy(),row['energy_eV'],rtol=0,atol=1e-10)
   assert np.allclose(image.get_forces(),row['forces_eV_A'],rtol=0,atol=1e-10)
  assert not Path('idpp.traj').exists() and not Path('idpp.log').exists()
 else:
  assert proc.returncode!=0 and s['error_type']=='DiagnosticLimit',s
  assert s['total_calculator_calls']==5,s
 print(label, s['status'],s['total_calculator_calls'], 'PASS')
print('EMT interface and capped-failure checks passed; no scientific path conclusion')
PY
