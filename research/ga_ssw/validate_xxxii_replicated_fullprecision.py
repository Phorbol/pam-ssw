"""Four fresh EFS for both endpoints in two larger equivalent representations."""
import json, shutil, time
from pathlib import Path
import numpy as np
from ase.io import read
from ase import Atoms
from ase.stress import voigt_6_to_full_3x3_stress
from research.ga_ssw.xxxii_replicated_calculator import XXXIIReplicatedCalculator

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-replicated-completion'
OUT=SRC/'cross-representation-fullprecision'
OUT.mkdir(exist_ok=False)
shutil.copy2(__file__,OUT/'runner-executed.py')
shutil.copy2(Path(__file__).with_name('xxxii_replicated_calculator.py'),OUT)
original=json.loads((SRC/'result.json').read_text())
fresh_rows=[json.loads(line) for line in (SRC/'ledger.jsonl').read_text().splitlines() if json.loads(line)['role']=='fresh']
def full_atoms(index):
    row=fresh_rows[index]
    return Atoms(numbers=row['numbers'], positions=row['positions'], cell=row['cell'], pbc=row['pbc'])
initial=full_atoms(0)
rows=[];start=time.monotonic()
(OUT/'plan.json').write_text(json.dumps(dict(EFS=4,representations=[[1,1,3],[2,2,2]],points=['minimum-0','minimum-1'], coordinate_source='fullprecision ledger fresh rows; earlier extxyz validation retained and costs charged',scope='equivalent representation and independent physical tolerance check, not Hessian stability or global search efficiency'),indent=2)+'\n')
for index in (0,1):
    a=full_atoms(index)
    ref=original['fresh'][index]
    for rep in ((1,1,3),(2,2,2)):
        with XXXIIReplicatedCalculator(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=initial,repetitions=rep) as c:
            a.calc=c
            e=a.get_potential_energy();f=a.get_forces();s=a.get_stress(voigt=False)
            rows.append(dict(index=index,repetitions=rep,energy=e,forces=f.tolist(),stress=s.tolist(),energy_error=e-ref['energy'],force_max_error=float(abs(f-ref['forces']).max()),stress_max_error=float(abs(s-ref['stress']).max()),fmax=float(np.linalg.norm(f,axis=1).max()),stress_max=float(abs(s).max()),engine_calls=c.engine_calls,api_calls=c.api_calls,atoms_evaluated=c.atoms_evaluated,force_image_spread=c.last_force_image_max_difference))
        (OUT/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
report=dict(status='completed',EFS=sum(x['api_calls'] for x in rows),engine_calls=sum(x['engine_calls'] for x in rows),atoms_evaluated=sum(x['atoms_evaluated'] for x in rows),energy_max_error=max(abs(x['energy_error']) for x in rows),force_max_error=max(x['force_max_error'] for x in rows),stress_max_error=max(x['stress_max_error'] for x in rows),all_original_tolerances_pass=all(x['fmax']<=.01 and x['stress_max']<=.001 for x in rows),wall_seconds=time.monotonic()-start)
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
