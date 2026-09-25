"""Check baseline first-hit and calculator wiring on a known target, not search."""
from pathlib import Path
import importlib.util,time,numpy as np
from ase import Atoms
from ase.io import write
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('bh',HERE/'run_lj55_bh.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.INPUTS=HERE/'bh-smoke-inputs'
p=m.INPUTS/'lj55-seed917';p.mkdir(parents=True,exist_ok=False)
write(p/'initial.extxyz',Atoms('Ar55',positions=2.7*np.loadtxt(HERE/'references/lj55.points')))
out=HERE/'bh-smoke';out.mkdir(exist_ok=False)
m.ARM_REQUESTS=20
row=m.run_track(917,out,m.serializer(),time.monotonic()+30,[0],[0])
assert row['status']=='first_energy_candidate',row
assert row['search_requests']<=20 and len(row['fresh_checks'])==2,row
print(row)
