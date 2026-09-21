import argparse,json,time,sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
import numpy as np
def main(out):
 sys.path.insert(0,str(out/'source'));from ase.io import read;from pamssw.standalone.paper_reference import SSWConfig,run_ssw;from pamssw.standalone.surface import ASESurface;from pamssw.standalone.pam_gaussian import PAMCurvatureGaussian;from ledger_helpers import CountedSurface,dump;from mace.calculators import MACECalculator
 plan=json.loads((out/'plan.json').read_text());rows=[]
 def calc():return MACECalculator(model_paths=plan['backend']['model'],device='cuda',default_dtype='float64',enable_cueq=False,enable_oeq=False)
 for case in plan['cases']:
  atoms=read(out/'inputs'/f'{case}.traj');base=SSWConfig(**plan['configs'][case])
  for arm in plan['arms']:
   folder=out/f'{case}-{arm["name"]}';folder.mkdir();ledger=folder/'requests.jsonl';ledger.touch();started=time.monotonic();search=CountedSurface(calc(),ledger,cap=6000,wall=600);validation=ASESurface(calc());row={'case':case,'arm':arm['name'],'seed':plan['seeds'][case]};checks=[]
   try:
    policy=PAMCurvatureGaussian(**arm['policy']);result=run_ssw(atoms.copy(),search,steps=100,config=base,rng=np.random.default_rng(plan['seeds'][case]),gaussian_policy=policy);dump(folder/'result.json',result)
    for i,m in enumerate(result.minima[:101]):
     try:
      e,f=validation.evaluate(m.atoms);fm=float(np.linalg.norm(f,axis=1).max()); comp=bool(np.array_equal(m.atoms.numbers,atoms.numbers)); cell=bool(np.array_equal(m.atoms.cell.array,atoms.cell.array));checks.append({'index':i,'energy':e,'energy_error':e-m.energy,'fmax':fm,'composition_match':comp,'fixed_cell':cell,'qualified':bool(fm<=base.fmax and comp and cell)})
     except Exception as exc:checks.append({'index':i,'qualified':False,'error':repr(exc)})
    dump(folder/'qualification.json',checks);row.update(execution=result.status,minima=len(result.minima),outer_statuses=dict(Counter(r.status for r in result.records)))
   except Exception as exc:row.update(execution='exception',error=repr(exc))
   row.update(search_requests=search.requests,fresh_requests=validation.requests,denials=search.denials,boundary=search.boundary,checks=checks,elapsed=time.monotonic()-started);dump(folder/'summary.json',row);rows.append(row);dump(out/'summary.json',rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output.resolve())
