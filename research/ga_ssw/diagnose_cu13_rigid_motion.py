"""Observe rigid-motion leakage in saved climbing centers, without rerunning SSW."""
import json
from pathlib import Path
import numpy as np
from .validate_cu13_escape import aligned_rmsd

def main(base='research/ga_ssw/evidence/dimer-ritz-cu13-escape'):
    base=Path(base); rows=[]
    for source in sorted(base.glob('[0-9]*-*.json')):
        d=json.loads(source.read_text()); pairs=[]
        for record in d['result']['records']:
            for a,b in zip(record['climb'],record['climb'][1:]):
                x=np.array(a['center']);y=np.array(b['center'])
                raw=float(np.sqrt(np.mean(np.sum((x-y)**2,axis=1))))
                rms=aligned_rmsd(x,y)
                pairs.append(dict(step=record['index'],gaussian=a['index'],raw_rms=raw,aligned_rms=rms,rigid_fit_fraction=1-rms*rms/(raw*raw) if raw else None))
        rows.append(dict(source=source.name,pairs=pairs))
    result=dict(note='successive stored centers only; excludes terminal displacement and failed stages; rigid-fit fraction is geometric squared-displacement reduction, not force/work fraction',runs=rows)
    (base/'rigid-motion.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        p=row['pairs'];print(row['source'],len(p),'raw median',np.median([r['raw_rms'] for r in p]),'aligned median',np.median([r['aligned_rms'] for r in p]),'rigid fraction median',np.median([r['rigid_fit_fraction'] for r in p]))
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='research/ga_ssw/evidence/dimer-ritz-cu13-escape')
    main(parser.parse_args().input)
