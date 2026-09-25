"""Bounded runner check on reference/known SLM: NOT random-search evidence."""
import importlib.util
import time
from pathlib import Path
import numpy as np
from ase import Atoms
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pilot',HERE/'run_lj_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
out=HERE/'runner-smoke'
m.preflight(out)
out.mkdir()
original=m.BoundedSurface.__init__
def capped(self,calculator,**kwargs):
 kwargs['arm_cap']=1000
 original(self,calculator,**kwargs)
m.BoundedSurface.__init__=capped
original_init=m.uniform_volume_cluster
def start(n,seed):
 _,rng=original_init(n,seed)
 path=(HERE/'references/lj55.points' if n==55 else Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/lj38-source-20260912/optim-finish'))
 return Atoms(f'Ar{n}',positions=2.7*np.loadtxt(path)),rng
m.uniform_volume_cluster=start
config,rotation=m.make_settings()
ledger=m.load_ledger()
for n in (55,38):
 (out/f'lj{n}-seed917').mkdir()
 row=m.run_one(n,917,out,config,rotation,time.monotonic()+60,[0],ledger)
 print(row)
 assert row['search_requests']<=1000
 if n==55: assert row['status']=='first_hit' and row['fresh_checks']
 else: assert row.get('completed_outer_attempts',0)>0,row
