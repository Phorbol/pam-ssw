"""Fresh modified-objective certificates for Safe-total frozen-stage results."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface
from pamssw.standalone.gaussian import ProjectedGaussian

def main():
    root=Path('research/ga_ssw/evidence');out=root/'cu13-failed-quench-pam';rows=[]
    for p in sorted(out.glob('*-safe-lbfgs-total.json')):
        d=json.loads(p.read_text());old=json.loads((root/'cu13-direction-only'/d['source']).read_text())
        record=next(r for r in old['result']['records'] if r['index']==d['step'])
        a=Atoms(**record['last_atoms']);a.positions=d['accepted_trace'][-1]['positions']
        surface=ASESurface(EMT());e,f=surface.evaluate(a)
        for g in record['climb']:
            de,df=ProjectedGaussian(np.array(g['center']),np.array(g['direction']),g['width'],g['weight']).evaluate(a)
            e+=de;f+=df
        maxf=float(np.linalg.norm(f,axis=1).max())
        rows.append(dict(file=p.name,energy=e,max_force=maxf,passed=maxf<=.01,requests=surface.requests))
    result=dict(scope='modified objective only, not true minima or full SSW success',checks=rows,total_requests=sum(r['requests'] for r in rows))
    (out/'fresh-certificates.json').write_text(json.dumps(result,indent=2)+'\n')
    assert len(rows)==31 and all(r['passed'] for r in rows)
    print('31/31 fresh modified-force certificates passed; 31 extra E/F requests')
if __name__=='__main__':main()
