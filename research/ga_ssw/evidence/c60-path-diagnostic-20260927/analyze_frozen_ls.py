"""Evaluate existing initial LS penalty on a fixed saved path, with no MLIP calls."""
import argparse
import ast
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from ase.io import read
from pamssw.standalone.native_ls import initialize_native_ls

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
r = json.loads((a.run_dir/'summary.json').read_text())
if r['status'] != 'completed' or not r.get('neb_fmax_qualified'):
    raise ValueError('Requires a completed numerically qualified saved path')
plan_path = ROOT/'research/ga_ssw/evidence/ls-pool-routing-20260926/plan.json'
plan = json.loads(plan_path.read_text())
case = next(c for c in plan['cases'] if c['name']=='C60-isomer2')
settings = case['native_ls']
kwargs = {k: settings[k] for k in ('scale','amp_c','length_tolerance','bond_geometry')}
kwargs['atom_filter'] = settings.get('atom_filter')
for k in ('bond_energies','bond_lengths','energy_filter','length_filter'):
    kwargs[k] = None if settings.get(k) is None else {ast.literal_eval(t):v for t,v in settings[k].items()}
initial = read(r['initial_path'])
frozen = initialize_native_ls(initial, **kwargs).potential
rows=[]
for im in r['images']:
    atoms=initial.copy(); atoms.positions[:]=im['positions_A']
    energy,force=frozen.evaluate(atoms)
    rows.append(dict(image=im['image'],physical_energy_eV=im['energy_eV'],penalty_eV=energy,penalty_fmax_eV_A=float(np.linalg.norm(force,axis=1).max())))
for row in rows:
    row['physical_relative_eV']=row['physical_energy_eV']-rows[0]['physical_energy_eV']
    row['penalty_relative_eV']=row['penalty_eV']-rows[0]['penalty_eV']
    row['modified_relative_eV']=row['physical_relative_eV']+row['penalty_relative_eV']
# Closed-form second derivative of a pair exponential in a normalized
# first-segment direction, in Cartesian coordinates at the original minimum.
h=np.load(ROOT/'research/ga_ssw/evidence/c60-local-defect-20260925/curvature/isomer-2.npz')
u=(np.asarray(r['images'][1]['positions_A'])-initial.positions).ravel()
u=h['internal_basis']@(h['internal_basis'].T@u);u=u.reshape(-1,3)/np.linalg.norm(u)
radial=tangential=0.
for (i,j),r0,strength in zip(frozen.pairs,frozen.reference_distances,frozen.strengths):
    delta=initial.positions[j]-initial.positions[i];distance=np.linalg.norm(delta)
    unit=delta/distance; du=u[j]-u[i]; longitudinal=float(du@unit)
    length=frozen.xi*r0; value=strength*np.exp(-(distance-r0)/length)
    radial+=value/length**2*longitudinal**2
    tangential-=value/(length*distance)*(float(du@du)-longitudinal**2)
result=dict(scope='Frozen initial LS on unchanged bare-PES NEB coordinates; NOT a relaxed LS path or LS activation barrier.',new_physical_calculator_calls=0,settings_source=str(plan_path),case=case['name'],bond_count=len(frozen.pairs),initialization_kwargs_repr=repr(kwargs),rows=rows,
 first_segment_penalty_curvature_eV_A2=dict(radial=radial,tangential=tangential,total=radial+tangential),
 omissions=['No soft prequench', 'No adaptive amplitude updates', 'No LS path reoptimization', 'No new SSW search'])
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
