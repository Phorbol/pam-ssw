"""Evaluate held-out published references with the existing full-pair LJ oracle."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from research.ga_ssw.full_pair_lj import FullPairLJ

HERE = Path(__file__).resolve().parent
rows = []
for n, target in [(38, -173.928427), (55, -279.248470), (75, -397.492331)]:
    atoms = Atoms(f'Ar{n}', positions=np.loadtxt(HERE/'references'/f'lj{n}.points')*2.7)
    atoms.calc = FullPairLJ()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    rows.append(dict(n=n, energy=energy, reference_energy=target,
        delta=energy-target, fmax=float(np.linalg.norm(forces, axis=1).max()),
        max_component=float(np.abs(forces).max()),
        qualified=bool(abs(energy-target)<1e-5 and np.abs(forces).max()<.04/2.7)))
(HERE/'lj-reference-qualification.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
if not all(row['qualified'] for row in rows):
    raise SystemExit('Reference qualification failed')
