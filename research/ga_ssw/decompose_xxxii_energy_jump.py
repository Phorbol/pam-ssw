"""Two real frozen evaluations to locate observed .2346eV discontinuity."""
import json,shutil,time
from pathlib import Path
import numpy as np
from ase import Atoms,units
from research.ga_ssw.xxxii_lammps_calculator import XXXIILammpsCalculator
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-central-ritz-completion';OUT=SRC/'energy-jump-components';OUT.mkdir(exist_ok=False)
points=[json.loads(x) for x in (SRC/'line-search-gradient/calls.jsonl').open()];chosen=[x for x in points if x['label'] in ('center','lbfgs+1e-06')];assert len(chosen)==2
shutil.copy2(__file__,OUT/'runner-executed.py');(OUT/'plan.json').write_text(json.dumps(dict(EFS=2,source=str(SRC/'line-search-gradient/calls.jsonl'),labels=[x['label'] for x in chosen],scope='energy decomposition only; unchangedbackend/geometry'),indent=2)+'\n')
class FixedG(XXXIILammpsCalculator):
 def _new_engine(self):
  from lammps import lammps
  return lammps(cmdargs=['-log',str(OUT/'engine.log'),'-screen','none'])
 def _initialize(self):
  super()._initialize()
  for command in ('pair_modify table 0','kspace_style ewald 1e-12','kspace_modify gewald 0.47570069','thermo_style custom step pe ebond eangle edihed eimp evdwl ecoul elong etail'):self._lmp.command(command)
a=Atoms(**chosen[0]['atoms']);c=FixedG(data_path=SRC/'lmp.data',input_path=SRC/'in.simple',model_manifest=SRC/'manifest.json',reference_atoms=a);rows=[]
try:
 for x in chosen:
  a=Atoms(**x['atoms']);c.calculate(a);parts={k:float(c._lmp.get_thermo(k))*units.kcal/units.mol for k in ('ebond','eangle','edihed','eimp','evdwl','ecoul','elong','etail')}
  rows.append(dict(label=x['label'],energy=c.results['energy'],energy_error=c.results['energy']-x['energy'],parts=parts,forces=c.results['forces'].tolist(),stress=c.results['stress'].tolist()))
 delta={k:rows[1]['parts'][k]-rows[0]['parts'][k] for k in rows[0]['parts']}
 report=dict(EFS=c.requests,rows=rows,delta=delta,total_delta=rows[1]['energy']-rows[0]['energy'])
 (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(delta=delta,total=report['total_delta']),indent=2))
finally:c.close()
