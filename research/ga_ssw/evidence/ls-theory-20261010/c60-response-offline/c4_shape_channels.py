"""Evaluate fixed-W shape normalization on two qualified C4 channels; no PES."""
from pathlib import Path
import json
import numpy as np
from ase.io import read
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.native_ls import HC_BOND_ENERGIES, HC_BOND_LENGTHS
ROOT=Path(__file__).resolve().parents[5]
base=ROOT / "research/ga_ssw/evidence/ls-theory-20261010/finite-channel-v1/run-1728623"
m=read(base/"gauche-reference.extxyz")
r0=np.linalg.norm(m.positions-m.positions.mean(axis=0))
w=FrozenBondSoftening.from_atoms(m,bond_energies=HC_BOND_ENERGIES,
    bond_lengths={k:v+.1 for k,v in HC_BOND_LENGTHS.items()},initial_fraction=.03,xi=.2)
wm,fm=w.evaluate(m)
z=np.load(ROOT/"research/ga_ssw/evidence/ls-theory-20261010/channel-v1/run-1728173/plus-0.050-hessian.npz")
Q=z["Q"];H=z["H_h0005"]
archived=read(ROOT/"research/ga_ssw/evidence/ls-theory-20261010/channel-v1/run-1728173/plus-0.050.extxyz")
assert np.max(np.abs(archived.positions-m.positions))<1e-12
assert np.linalg.eigvalsh(H)[0]>0
x=(m.positions-m.positions.mean(axis=0)).ravel();g=-fm.ravel();e=x/np.linalg.norm(x)
gshape=g-e*(e@g)
chi=float((Q.T@g)@np.linalg.solve(H,Q.T@g))
chi_shape=float((Q.T@gshape)@np.linalg.solve(H,Q.T@gshape))
rows=[]
for name,file in [("easy","easy-ts-aligned.extxyz"),("ring","ring-ts-aligned.extxyz")]:
 s=read(base/file);x=s.positions-s.positions.mean(axis=0);r=np.linalg.norm(x)
 raw,_=w.evaluate(s)
 normalized=s.copy();normalized.positions=x*r0/r
 value,_=w.evaluate(normalized)
 rows.append({"channel":name,"source":str(base/file),"W_min_eV":wm,
  "radius_TS_over_min":r/r0,"original_barrier_derivative_eV":raw-wm,
  "shape_normalized_barrier_derivative_eV":value-wm,
  "original_minus_b_over_sqrt_chi_sqrt_eV":-(raw-wm)/np.sqrt(chi),
  "shape_minus_b_over_sqrt_chi_sqrt_eV":-(value-wm)/np.sqrt(chi_shape)})
out={"scope":"Analytic bias evaluations at already qualified physical stationary points; first derivative at zero loading only. No new PES calls, no finite search, no algorithm or default change.",
 "origin":str(base/"gauche-reference.extxyz"),"R0_A":r0,"chi_original_eV":chi,"chi_shape_eV":chi_shape,"rows":rows}
(Path(__file__).parent/"c4-shape-response.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
