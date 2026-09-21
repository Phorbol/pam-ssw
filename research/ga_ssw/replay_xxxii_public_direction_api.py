"""No new PES: replay the complete real XXXII ledger through the public solver API."""
import json, shutil, time
from pathlib import Path
import numpy as np
from ase import Atoms
from pamssw.standalone.rc_topology import read_rigid_topology
from pamssw.standalone.rc_vc_reference import RCVCSSWConfig, run_rc_vc_ssw
from pamssw.standalone.generalized_numerics import generalized_central_ritz
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'research/ga_ssw/evidence/xxxii-rc-vc-replicated-completion'
OUT=SRC/'public-api-replay';OUT.mkdir(exist_ok=False)
shutil.copy2(__file__,OUT/'runner-executed.py')
for name in ('generalized_ritz.py','generalized_numerics.py','rc_vc_reference.py'):
    shutil.copy2(ROOT/'pamssw/standalone'/name,OUT/name)
rows=[json.loads(x) for x in (SRC/'ledger.jsonl').read_text().splitlines()]
rows=[x for x in rows if x['role']=='search']
def atoms_from(row):return Atoms(numbers=row['numbers'],positions=row['positions'],cell=row['cell'],pbc=row['pbc'])
a=atoms_from(rows[0]);top=read_rigid_topology(SRC/'rigidbody',SRC/'blist',natoms=172)
cfg=RCVCSSWConfig(**json.loads((SRC/'plan.json').read_text())['config'])
class Replay:
    requests=0
    max_position_error=0.
    max_cell_error=0.
    def evaluate(self,atoms):
        if self.requests>=len(rows):raise RuntimeError('replay requested beyond real ledger')
        row=rows[self.requests];self.requests+=1
        ep=float(abs(atoms.positions-row['positions']).max());ec=float(abs(atoms.cell.array-row['cell']).max())
        self.max_position_error=max(ep,self.max_position_error);self.max_cell_error=max(ec,self.max_cell_error)
        if ep>1e-11 or ec>1e-11:raise RuntimeError(f'replay geometry mismatch at{self.requests}: {ep},{ec}')
        return row['energy'],np.array(row['forces']),np.array(row['stress'])
surface=Replay();start=time.monotonic()
r=run_rc_vc_ssw(a,surface,trees=top.components,anchor=0,steps=1,config=cfg,rng=np.random.default_rng(3),direction_solver=generalized_central_ritz,rotation_force_calls=100)
report=dict(status=r.status,replayed_requests=surface.requests,expected_requests=len(rows),new_PES_calls=0,max_position_error=surface.max_position_error,max_cell_error=surface.max_cell_error,minima=len(r.minima),last_event=r.records[-1]['status'],delta=r.records[-1].get('delta'),accepted=r.records[-1].get('accepted'),error=r.records[-1].get('error'),wall_seconds=time.monotonic()-start,scope='deterministic public API integration regression against entire real trajectory; not an independent physical experiment')
(OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert r.status=='completed' and surface.requests==len(rows)
