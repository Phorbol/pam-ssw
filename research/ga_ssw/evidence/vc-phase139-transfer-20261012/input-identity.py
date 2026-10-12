"""Zero-PES identity screen of archived, certified starting minima."""
import itertools
import json
from pathlib import Path
import sys
from ase import Atoms
from ase.io import read
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from pamssw.standalone.periodic_ga_reference import pymatgen_identity
BASE = ROOT.parent / 'ga-ssw-behavior-parity/research/ga_ssw/evidence'
paths = {
    'phase139': BASE / 'tio2-phase139-joint-memory-compare/memory400/result.json',
    'phase87': BASE / 'tio2-phase87-vc-pqc-single-step/joint/result.json',
    'anatase': ROOT.parent / 'c60-local-defect-qualification/research/ga_ssw/evidence/tio2-phase-qualification-20261007/run-1663408/anatase_fs/endpoint.extxyz',
}
structures = {name: (Atoms(**json.loads(path.read_text())['common_start'])
                    if path.suffix == '.json' else read(path)) for name, path in paths.items()}
settings = {'tight': {'ltol': .05, 'stol': .1, 'angle_tol': 2.},
            'broad': {'ltol': .2, 'stol': .3, 'angle_tol': 5.}}
result = {'scope': 'saved relaxed geometry identity; not new PES qualification or phase proof',
          'pes_calls': 0, 'sources': {key: str(path) for key, path in paths.items()},
          'matcher': {'scale': False, 'primitive_cell': True, 'attempt_supercell': True},
          'settings': settings, 'structures': {name: {'natoms': len(atoms),
              'volume_A3': atoms.get_volume(), 'formula': atoms.get_chemical_formula(),
              'pbc': atoms.pbc.tolist()} for name, atoms in structures.items()}, 'matches': {}}
for label, config in settings.items():
    matcher = pymatgen_identity(**config)
    result['matches'][label] = {a + '--' + b: bool(matcher(structures[a], structures[b]))
                               for a, b in itertools.combinations(structures, 2)}
    result['matches'][label]['self_checks'] = {name: bool(matcher(atoms, atoms.copy()))
                                             for name, atoms in structures.items()}
print(json.dumps(result, indent=2))
