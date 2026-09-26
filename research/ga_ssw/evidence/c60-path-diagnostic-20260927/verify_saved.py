"""Verify saved trajectory provenance without new calculator evaluations."""
import json
from pathlib import Path
import numpy as np
from ase.io import read
p=Path(__file__).resolve().parent
s=json.loads((p/'run-1504112/summary.json').read_text())
t=read(p/'run-1504112/neb.traj', index=':')
assert len(t)%7==0
for i,(a,r) in enumerate(zip(t[-7:],s['images'])):
    np.testing.assert_allclose(a.positions,r['positions_A'],rtol=0,atol=1e-12)
    if i in (0,6):
        continue  # ASE iterimages does not freeze shared-calculator endpoint results.
    np.testing.assert_allclose(a.get_potential_energy(),r['energy_eV'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(a.get_forces(),r['forces_eV_A'],rtol=0,atol=1e-9)
assert s['total_calculator_calls_started']==s['total_calculator_calls_completed']==1596
print('PASS: all seven full-precision geometries and five cached interior E/F match fresh summary; frames=',len(t))
