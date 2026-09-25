"""Zero-PES checks on known identical and known different reference structures."""
import importlib.util
import numpy as np
from ase import Atoms
from pathlib import Path
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('analysis',p/'analyze_lj_pilot.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for n in (38,55):
 a=Atoms(f'Ar{n}',positions=2.7*np.loadtxt(p/'references'/f'lj{n}.points'))
 b=a[np.random.default_rng(917).permutation(n)]
 b.rotate(37,'z');b.translate([2,3,4])
 row=m.geometry(b,a)
 assert row['geometry_match'] and row['proper_rms_A']<1e-10,row
 print(n,row)
a=Atoms('Ar38',positions=2.7*np.loadtxt('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/lj38-source-20260912/optim-finish'))
b=Atoms('Ar38',positions=2.7*np.loadtxt(p/'references/lj38.points'))
row=m.geometry(a,b)
assert not row['geometry_match'],row
print('known non-GM',row)
