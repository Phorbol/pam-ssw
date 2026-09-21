"""Post hoc strict stationarity and labelled proper-rotation comparison.
No search parameters are changed. RMSD does not permute atoms or allow reflection.
"""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface, quench

def aligned_rmsd(x, y):
    x=x-x.mean(axis=0); y=y-y.mean(axis=0)
    u, _, vt=np.linalg.svd(x.T@y)
    correction=np.eye(3); correction[-1,-1]=np.linalg.det(u@vt)
    return float(np.sqrt(np.mean(np.sum((x@u@correction@vt-y)**2,axis=1))))

def main():
    base=Path('research/ga_ssw/evidence/dimer-ritz-cu13-escape')
    out=base/'strict-validation';out.mkdir(exist_ok=False)
    (out/'script.py').write_text(Path(__file__).read_text())
    (out/'plan.json').write_text(json.dumps(dict(fmax=1e-5,steps=300,backend='ASE EMT',comparison='proper Kabsch with original atom labels; no permutation search',purpose='post hoc diagnostic, not a retuned search'),indent=2))
    rows=[]; reference=None; energies=[]
    for source in sorted(base.glob('[0-9]*-*.json')):
        d=json.loads(source.read_text()); records=[]
        for i,m in enumerate(d['result']['minima']):
            a=Atoms(**m['atoms']); surface=ASESurface(EMT())
            q=quench(a,surface,fmax=1e-5,steps=300)
            fresh=ASESurface(EMT());e,f=fresh.evaluate(q.atoms)
            if reference is None: reference=q.atoms.positions.copy()
            row=dict(index=i,converged=q.converged,energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),requests=surface.requests+fresh.requests,rmsd_to_reference=aligned_rmsd(q.atoms.positions,reference),raw_rmsd_to_reference=aligned_rmsd(a.positions,reference),positions=q.atoms.positions.tolist())
            records.append(row);energies.append(e)
        (out/source.name).write_text(json.dumps(records,indent=2)+'\n')
        rows.append(dict(source=source.name,count=len(records),all_converged=all(r['converged'] for r in records),requests=sum(r['requests'] for r in records),max_rmsd=max(r['rmsd_to_reference'] for r in records),max_raw_rmsd=max(r['raw_rmsd_to_reference'] for r in records)))
    result=dict(runs=rows,total_records=sum(r['count'] for r in rows),energy_span=max(energies)-min(energies),interpretation='numerical structural equivalence only; no Hessian stability proof')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
