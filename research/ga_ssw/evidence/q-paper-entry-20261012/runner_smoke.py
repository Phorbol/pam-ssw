"""No MLIP evaluation: exercise runner wiring with an ASE quadratic calculator."""
import importlib.util, sys, json
from pathlib import Path
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
root=Path(sys.argv[1]);data=Path(sys.argv[2]);out=Path(sys.argv[3])
spec=importlib.util.spec_from_file_location('q_runner',root/'research/ga_ssw/probe_q_paper_cases.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.configure_data_root(data)
refs={x['head']:read(x['input']).positions.copy() for x in runner.CASES.values()}
class Quadratic(Calculator):
 implemented_properties=['energy','forces']
 def __init__(self,head,**kwargs):
  super().__init__();self.reference=refs[head]
 def calculate(self,atoms=None,properties=None,system_changes=all_changes):
  super().calculate(atoms,properties,system_changes)
  dx=atoms.positions-self.reference
  self.results={'energy':float(.5*np.sum(dx*dx)),'forces':-dx}
import mace.calculators
mace.calculators.MACECalculator=Quadratic
runner.SEARCH_CAP=250
runner.ARM_WALL_SECONDS=30
runner.MAX_GAUSSIANS=1
result=runner.execute(out)
assert len(result['runs'])==6, result
assert not any(r.get('failure_class','').startswith('hard') for r in result['runs']),result
assert result['search_requests_total']==sum(r['search_requests'] for r in result['runs'])
assert result['actual_calculate_calls_total']==sum(r['search_calculate_calls']+r.get('fresh_calculate_calls',0) for r in result['runs'])
print(json.dumps({'status':'runner_wiring_pass','arms':len(result['runs']),'search':result['search_requests_total'],'fresh':result['fresh_requests_total'],'scope':'quadratic wiring only; not MACE or scientific validation'}))
